/**
 * @file laser_match_cuda_impl.cu
 * @brief 激光线匹配CUDA算子 - CUDA实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 量化左右点集行索引 (kernelQuantizeRowIdx)
 *   Step 2: 构建左点哈希表 (kernelBuildHash)
 *   Step 3: 探测右点匹配 (kernelProbeMatch)
 *   Step 4: CUB DeviceSelect::Flagged 压缩输出
 */

#include "laser_match_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(07, LaserMatchCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr unsigned int HASH_EMPTY_KEY = 0xFFFFFFFFu;

// ============================================================================
// Device Helpers
// ============================================================================

__device__ __forceinline__ unsigned int hashFunc(unsigned int key, unsigned int capacity) {
    key = ((key >> 16) ^ key) * 0x45d9f3bu;
    key = ((key >> 16) ^ key) * 0x45d9f3bu;
    key = (key >> 16) ^ key;
    return key & (capacity - 1);
}

__device__ __forceinline__ unsigned int makeCompositeKey(int row_idx, int frame_id) {
    return (static_cast<unsigned int>(row_idx) << 16) | (static_cast<unsigned int>(frame_id) & 0xFFFF);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void __launch_bounds__(256, 4) kernelQuantizeRowIdx(
    const float2* __restrict__ d_points,
    int count,
    float step,
    int* __restrict__ d_rowidx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    d_rowidx[idx] = static_cast<int>(roundf(d_points[idx].y / step));
}

__global__ void __launch_bounds__(256, 4) kernelBuildHash(
    const int* __restrict__ d_rowidx,
    const int* __restrict__ d_fids,
    int count,
    unsigned int* __restrict__ d_hash_keys,
    int* __restrict__ d_hash_vals,
    unsigned int capacity,
    int max_probe)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    unsigned int key = makeCompositeKey(d_rowidx[idx], d_fids[idx]);
    unsigned int slot = hashFunc(key, capacity);

    for (int probe = 0; probe < max_probe; ++probe) {
        unsigned int prev = atomicCAS(&d_hash_keys[slot], HASH_EMPTY_KEY, key);
        if (prev == HASH_EMPTY_KEY || prev == key) {
            d_hash_vals[slot] = idx;
            return;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
}

__global__ void __launch_bounds__(256, 4) kernelProbeMatch(
    const int* __restrict__ d_right_rowidx,
    const int* __restrict__ d_right_fids,
    const float2* __restrict__ d_right_points,
    const float2* __restrict__ d_left_points,
    const unsigned int* __restrict__ d_hash_keys,
    const int* __restrict__ d_hash_vals,
    int right_count,
    unsigned int capacity,
    int max_probe,
    float min_disp,
    float max_disp,
    int* __restrict__ d_flags,
    float2* __restrict__ d_match_left,
    float2* __restrict__ d_match_right,
    int* __restrict__ d_match_fids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= right_count) {
        return;
    }

    unsigned int key = makeCompositeKey(d_right_rowidx[idx], d_right_fids[idx]);
    unsigned int slot = hashFunc(key, capacity);

    for (int probe = 0; probe < max_probe; ++probe) {
        unsigned int stored_key = d_hash_keys[slot];
        if (stored_key == HASH_EMPTY_KEY) {
            d_flags[idx] = 0;
            return;
        }
        if (stored_key == key) {
            int left_idx = d_hash_vals[slot];
            float2 lp = d_left_points[left_idx];
            float2 rp = d_right_points[idx];
            float disparity = lp.x - rp.x;
            if (disparity >= min_disp && disparity <= max_disp) {
                d_flags[idx] = 1;
                d_match_left[idx] = lp;
                d_match_right[idx] = rp;
                d_match_fids[idx] = d_right_fids[idx];
            } else {
                d_flags[idx] = 0;
            }
            return;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }

    d_flags[idx] = 0;
}

// ============================================================================
// ScopedFlag (Debug-only thread safety)
// ============================================================================

#ifndef NDEBUG
class ScopedFlag {
public:
    explicit ScopedFlag(std::atomic<bool>* flag) : flag_(flag) {
        flag_->store(true);
    }
    ~ScopedFlag() { flag_->store(false); }
    ScopedFlag(const ScopedFlag&) = delete;
    ScopedFlag& operator=(const ScopedFlag&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// CUDA Error Handling
// ============================================================================

template<typename T>
static inline void safeCudaFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            CALIB_LOG_ERROR("cudaFree failed: {}", cudaGetErrorString(err));
        }
        ptr = nullptr;
    }
}

// ============================================================================
// Impl Implementation
// ============================================================================

LaserMatchCuda::Impl::Impl(const LaserMatchParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[07-LaserMatchCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[07-LaserMatchCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

LaserMatchCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }

    safeCudaFree(d_cub_temp_);
    cub_temp_size_ = 0;
    last_max_count_ = 0;
}

/*static*/ int LaserMatchCuda::Impl::nextPowerOf2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

bool LaserMatchCuda::Impl::allocateBuffers(int leftCount, int rightCount) {
    const int maxCount = (leftCount > rightCount) ? leftCount : rightCount;

    if (maxCount <= 0) {
        return true;
    }

    if (maxCount <= last_max_count_ &&
        !d_hash_keys_.empty() &&
        !d_flags_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for left={}, right={}", leftCount, rightCount);

    hash_capacity_ = nextPowerOf2(2 * leftCount);
    if (hash_capacity_ < 16) {
        hash_capacity_ = 16;
    }

    d_hash_keys_.create(1, hash_capacity_, CV_32SC1);
    d_hash_vals_.create(1, hash_capacity_, CV_32SC1);

    d_left_rowidx_.create(1, leftCount, CV_32SC1);
    d_right_rowidx_.create(1, rightCount, CV_32SC1);

    d_flags_.create(1, rightCount, CV_32SC1);
    d_temp_left_.create(1, rightCount, CV_32FC2);
    d_temp_right_.create(1, rightCount, CV_32FC2);
    d_temp_fids_.create(1, rightCount, CV_32SC1);

    d_out_left_.create(1, maxCount, CV_32FC2);
    d_out_right_.create(1, maxCount, CV_32FC2);
    d_out_fids_.create(1, maxCount, CV_32SC1);
    d_out_count_.create(1, 1, CV_32SC1);

    last_max_count_ = maxCount;
    return true;
}

void LaserMatchCuda::Impl::Warmup(int leftCount, int rightCount) {
    if (leftCount <= 0 || rightCount <= 0) {
        CALIB_LOG_WARN("warmup(): invalid leftCount={}, rightCount={}, skipping", leftCount, rightCount);
        return;
    }

    if (warmed_up_ && warmup_left_ == leftCount && warmup_right_ == rightCount) {
        CALIB_LOG_DEBUG("warmup(): already warmed up for left={}, right={}", leftCount, rightCount);
        return;
    }

    allocateBuffers(leftCount, rightCount);
    warmed_up_ = true;
    warmup_left_ = leftCount;
    warmup_right_ = rightCount;

    CALIB_LOG_INFO("warmup(): allocated GPU buffers for left={}, right={}", leftCount, rightCount);
}

LaserMatchResult LaserMatchCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    LaserMatchResult result;

    try {
        const int leftCount = d_left_points.rows * d_left_points.cols;
        const int rightCount = d_right_points.rows * d_right_points.cols;

        if (leftCount == 0 || rightCount == 0) {
            result.success = true;
            result.message = "Empty input, no matching";
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.matchCount = 0;
            return result;
        }

        if (!allocateBuffers(leftCount, rightCount)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        cudaError_t memset_err = cudaMemset(
            d_hash_keys_.ptr<unsigned int>(), 0xFF,
            hash_capacity_ * sizeof(unsigned int));
        if (memset_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("cudaMemset hash failed: ") + cudaGetErrorString(memset_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        const float2* d_left_ptr = d_left_points.ptr<float2>();
        const int* d_left_fid_ptr = d_left_line_ids.ptr<int>();
        const float2* d_right_ptr = d_right_points.ptr<float2>();
        const int* d_right_fid_ptr = d_right_line_ids.ptr<int>();

        const int left_grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int right_grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int max_probe = (leftCount < 128) ? leftCount : 128;

        kernelQuantizeRowIdx<<<left_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_ptr, leftCount, params_.epipolar_row_step,
            d_left_rowidx_.ptr<int>());

        kernelQuantizeRowIdx<<<right_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_right_ptr, rightCount, params_.epipolar_row_step,
            d_right_rowidx_.ptr<int>());

        kernelBuildHash<<<left_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_rowidx_.ptr<int>(), d_left_fid_ptr, leftCount,
            d_hash_keys_.ptr<unsigned int>(), d_hash_vals_.ptr<int>(),
            static_cast<unsigned int>(hash_capacity_), max_probe);

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        kernelProbeMatch<<<right_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_right_rowidx_.ptr<int>(), d_right_fid_ptr, d_right_ptr,
            d_left_ptr,
            d_hash_keys_.ptr<unsigned int>(), d_hash_vals_.ptr<int>(),
            rightCount, static_cast<unsigned int>(hash_capacity_), max_probe,
            params_.min_disparity, params_.max_disparity,
            d_flags_.ptr<int>(),
            d_temp_left_.ptr<float2>(), d_temp_right_.ptr<float2>(),
            d_temp_fids_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        size_t temp_bytes_left = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_left,
            d_temp_left_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_left_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);

        size_t temp_bytes_right = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_right,
            d_temp_right_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_right_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);

        size_t temp_bytes_fids = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_fids,
            d_temp_fids_.ptr<int>(), d_flags_.ptr<int>(),
            d_out_fids_.ptr<int>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);

        size_t max_temp_bytes = temp_bytes_left;
        if (temp_bytes_right > max_temp_bytes) max_temp_bytes = temp_bytes_right;
        if (temp_bytes_fids > max_temp_bytes) max_temp_bytes = temp_bytes_fids;

        if (max_temp_bytes > cub_temp_size_ || d_cub_temp_ == nullptr) {
            safeCudaFree(d_cub_temp_);

            cudaError_t alloc_err = cudaMalloc(&d_cub_temp_, max_temp_bytes);
            if (alloc_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(alloc_err);
                CALIB_LOG_ERROR("process(): {}", result.message);
                return result;
            }
            cub_temp_size_ = max_temp_bytes;
        }

        cudaError_t cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_left_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_left_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (left) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_right_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_right_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (right) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_fids_.ptr<int>(), d_flags_.ptr<int>(),
            d_out_fids_.ptr<int>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (fids) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

#ifndef NDEBUG
        cudaStreamSynchronize(cuda_stream);
        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel execution failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }
#endif

        int h_count = 0;
        cudaMemcpyAsync(&h_count, d_out_count_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.success = true;
        result.message = "Success";
        result.matchCount = h_count;
        result.d_matched_left = std::make_shared<cv::cuda::GpuMat>(d_out_left_.colRange(0, h_count).clone());
        result.d_matched_right = std::make_shared<cv::cuda::GpuMat>(d_out_right_.colRange(0, h_count).clone());
        result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>(d_out_fids_.colRange(0, h_count).clone());

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("process(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("process(): {}", result.message);
    }

    return result;
}

void LaserMatchCuda::Impl::SetParams(const LaserMatchParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[07-LaserMatchCuda] setParams() called during process()");
    }
#endif

    params.validate();

    bool deviceChanged = (params.deviceId != params_.deviceId);

    params_ = params;

    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }

    warmed_up_ = false;
    warmup_left_ = 0;
    warmup_right_ = 0;

    CALIB_LOG_INFO("setParams(): params updated, warmup reset");
}
