/**
 * @file epipolar_interp_cuda_impl.cu
 * @brief 激光中心点极线插值CUDA算子 - CUDA实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 对每对相邻点 (p_i, p_{i+1})，检查是否同帧
 *   Step 2: 检查 X 差值 < max_x_diff，Y 跨度 < max_y_span
 *   Step 3: 计算最近极线 y_target = ceil(y_min / step) * step
 *   Step 4: 验证 y_target 严格在 (y_min, y_max) 之间且距离 < 1.0
 *   Step 5: 线性插值 x_interp = x1 + t * (x2 - x1)
 *   Step 6: CUB DeviceSelect::Flagged 压缩输出
 */

#include "epipolar_interp_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(06, EpipolarInterpCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr float EPSILON = 1e-6f;
static constexpr float DIST_THRESHOLD = 1.0f;

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
// CUDA Kernel
// ============================================================================

__global__ void __launch_bounds__(256, 4) kernelMarkAndCompute(
    const float2* __restrict__ d_input,
    const int* __restrict__ d_line_ids,
    int pair_count,
    float epipolar_step,
    float max_x_diff,
    float max_y_span,
    bool line_id_check,
    int* __restrict__ d_flags,
    float2* __restrict__ d_interp_pts,
    int* __restrict__ d_interp_fids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= pair_count) {
        return;
    }

    if (line_id_check && d_line_ids[idx] != d_line_ids[idx + 1]) {
        d_flags[idx] = 0;
        return;
    }

    const int fid = d_line_ids[idx];

    float2 p1 = d_input[idx];
    float2 p2 = d_input[idx + 1];

    if (p1.y > p2.y) {
        float2 temp = p1;
        p1 = p2;
        p2 = temp;
    }

    const float dx = fabsf(p1.x - p2.x);
    if (dx >= max_x_diff) {
        d_flags[idx] = 0;
        return;
    }

    const float y_min = p1.y;
    const float y_max = p2.y;

    if ((y_max - y_min) >= max_y_span) {
        d_flags[idx] = 0;
        return;
    }

    const float y_target = ceilf(y_min / epipolar_step) * epipolar_step;

    const float dist_to_min = y_target - y_min;
    const float dist_to_max = y_max - y_target;

    if (dist_to_min > 0.0f &&
        dist_to_max > 0.0f &&
        dist_to_min < DIST_THRESHOLD &&
        dist_to_max < DIST_THRESHOLD)
    {
        const float denom = y_max - y_min;

        if (fabsf(denom) < EPSILON) {
            d_flags[idx] = 0;
            return;
        }

        const float t = dist_to_min / denom;
        const float x_interp = p1.x + t * (p2.x - p1.x);

        d_interp_pts[idx] = make_float2(x_interp, y_target);
        d_interp_fids[idx] = fid;
        d_flags[idx] = 1;
    }
    else {
        d_flags[idx] = 0;
    }
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
// Impl Implementation
// ============================================================================

EpipolarInterpCuda::Impl::Impl(const EpipolarInterpParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[06-EpipolarInterpCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[06-EpipolarInterpCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

EpipolarInterpCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }

    safeCudaFree(d_cub_temp_storage_);
    cub_temp_size_ = 0;
    last_max_pairs_count_ = 0;
}

bool EpipolarInterpCuda::Impl::allocateBuffers(int pointCount) {
    const int num_pairs = pointCount - 1;

    if (num_pairs <= 0) {
        return true;
    }

    if (num_pairs <= last_max_pairs_count_ &&
        !d_flags_.empty() &&
        !d_temp_interp_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for {} pairs", num_pairs);

    d_flags_.create(1, num_pairs, CV_32SC1);
    d_temp_interp_.create(1, num_pairs, CV_32FC2);
    d_temp_fids_.create(1, num_pairs, CV_32SC1);
    d_output_.create(1, num_pairs, CV_32FC2);
    d_output_fids_.create(1, num_pairs, CV_32SC1);
    d_output_count_.create(1, 1, CV_32SC1);

    last_max_pairs_count_ = num_pairs;
    return true;
}

void EpipolarInterpCuda::Impl::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }

    if (warmed_up_ && warmup_count_ == pointCount) {
        CALIB_LOG_DEBUG("Warmup(): already warmed up for {} points", pointCount);
        return;
    }

    allocateBuffers(pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;

    CALIB_LOG_INFO("Warmup(): allocated GPU buffers for {} points", pointCount);
}

EpipolarInterpResult EpipolarInterpCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    EpipolarInterpResult result;

    try {
        int pointCount = d_points.rows * d_points.cols;

        if (pointCount < 2) {
            result.success = true;
            result.message = "Less than 2 points, no interpolation";
            result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
            result.interpCount = 0;
            return result;
        }

        if (!allocateBuffers(pointCount)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        const float2* d_input_ptr = d_points.ptr<float2>();
        const int* d_fid_ptr = d_line_ids.ptr<int>();
        int* d_flags_ptr = d_flags_.ptr<int>();
        float2* d_temp_ptr = d_temp_interp_.ptr<float2>();
        int* d_temp_fid_ptr = d_temp_fids_.ptr<int>();
        float2* d_output_ptr = d_output_.ptr<float2>();
        int* d_output_fid_ptr = d_output_fids_.ptr<int>();
        int* d_count_ptr = d_output_count_.ptr<int>();

        const int num_pairs = pointCount - 1;
        const int grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernelMarkAndCompute<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            d_input_ptr, d_fid_ptr, num_pairs,
            params_.epipolar_row_step,
            params_.max_x_diff,
            params_.max_y_span,
            params_.lineIdCheck,
            d_flags_ptr, d_temp_ptr, d_temp_fid_ptr);

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        size_t temp_storage_bytes = 0;
        size_t temp_storage_fids = 0;

        cudaError_t cub_err = cub::DeviceSelect::Flagged(
            nullptr, temp_storage_bytes,
            d_temp_ptr, d_flags_ptr,
            d_output_ptr, d_count_ptr,
            num_pairs, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged size query failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            nullptr, temp_storage_fids,
            d_temp_fid_ptr, d_flags_ptr,
            d_output_fid_ptr, d_count_ptr,
            num_pairs, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged size query (fids) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        size_t required_temp = (temp_storage_bytes > temp_storage_fids) ? temp_storage_bytes : temp_storage_fids;

        if (required_temp > cub_temp_size_ || d_cub_temp_storage_ == nullptr) {
            safeCudaFree(d_cub_temp_storage_);

            cub_err = cudaMalloc(&d_cub_temp_storage_, required_temp);
            if (cub_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(cub_err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }

            cub_temp_size_ = required_temp;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_storage_, cub_temp_size_,
            d_temp_ptr, d_flags_ptr,
            d_output_ptr, d_count_ptr,
            num_pairs, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_storage_, cub_temp_size_,
            d_temp_fid_ptr, d_flags_ptr,
            d_output_fid_ptr, d_count_ptr,
            num_pairs, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

#ifndef NDEBUG
        cudaStreamSynchronize(cuda_stream);
        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel execution failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }
#endif

        int h_count = 0;
        cudaMemcpyAsync(&h_count, d_count_ptr, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.success = true;
        result.message = "Success";
        result.interpCount = h_count;
        result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>(d_output_.colRange(0, h_count).clone());
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(d_output_fids_.colRange(0, h_count).clone());

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    }

    return result;
}

void EpipolarInterpCuda::Impl::SetParams(const EpipolarInterpParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[06-EpipolarInterpCuda] setParams() called during process()");
    }
#endif

    params.validate();

    bool deviceChanged = (params.deviceId != params_.deviceId);

    params_ = params;

    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }

    warmed_up_ = false;
    warmup_count_ = 0;

    CALIB_LOG_INFO("SetParams(): params updated, warmup reset");
}

// ============================================================================
// Destroy()
// ============================================================================

void EpipolarInterpCuda::Impl::Destroy() {
    if (d_cub_temp_storage_) {
        cudaFree(d_cub_temp_storage_);
        d_cub_temp_storage_ = nullptr;
    }
    cub_temp_size_ = 0;
    d_flags_.release();
    d_temp_interp_.release();
    d_temp_fids_.release();
    d_output_.release();
    d_output_fids_.release();
    d_output_count_.release();
    warmed_up_ = false;
    warmup_count_ = 0;
    last_max_pairs_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
