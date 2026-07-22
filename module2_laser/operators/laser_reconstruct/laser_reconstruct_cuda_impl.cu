/**
 * @file laser_reconstruct_cuda_impl.cu
 * @brief 激光线三维重建CUDA算子 - CUDA实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 计算视差 (kernelComputeDisparity)
 *   Step 2: 使用Q矩阵进行三维重建 (kernelReconstruct3D)
 *   Step 3: CUB DeviceSelect::Flagged 压缩输出有效点
 */

#include "laser_reconstruct_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(08, LaserReconstructCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Constant Memory
// ============================================================================

__constant__ float c_Qf[16];

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void __launch_bounds__(256, 4) kernelComputeDisparity(
    const float2* __restrict__ d_left,
    const float2* __restrict__ d_right,
    int count,
    float* __restrict__ d_disparity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    d_disparity[idx] = d_left[idx].x - d_right[idx].x;
}

__global__ void __launch_bounds__(256, 4) kernelReconstruct3D(
    const float2* __restrict__ d_left,
    const float* __restrict__ d_disparity,
    int count,
    float minDepth,
    float maxDepth,
    float3* __restrict__ d_points3d,
    int* __restrict__ d_valid,
    int* __restrict__ d_out_fids,
    const int* __restrict__ d_in_fids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float x = d_left[idx].x;
    float y = d_left[idx].y;
    float d = d_disparity[idx];

    float W = c_Qf[14] * d + c_Qf[15];

    if (W <= 0.0f) {
        d_valid[idx] = 0;
        return;
    }

    float X = (c_Qf[0] * x + c_Qf[3]) / W;
    float Y = (c_Qf[5] * y + c_Qf[7]) / W;
    float Z = c_Qf[11] / W;

    if (Z < minDepth || Z > maxDepth) {
        d_valid[idx] = 0;
        return;
    }

    d_points3d[idx] = make_float3(X, Y, Z);
    d_valid[idx] = 1;
    d_out_fids[idx] = d_in_fids[idx];
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

LaserReconstructCuda::Impl::Impl(const LaserReconstructParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[08-LaserReconstructCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[08-LaserReconstructCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

LaserReconstructCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }

    safeCudaFree(d_cub_temp_);
    cub_temp_size_ = 0;
    last_max_count_ = 0;
}

bool LaserReconstructCuda::Impl::allocateBuffers(int pointCount) {
    if (pointCount <= 0) {
        return true;
    }

    if (pointCount <= last_max_count_ &&
        !d_disparity_.empty() &&
        !d_valid_flags_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for pointCount={}", pointCount);

    d_disparity_.create(1, pointCount, CV_32FC1);
    d_points3d_raw_.create(1, pointCount, CV_32FC3);
    d_valid_flags_.create(1, pointCount, CV_32SC1);
    d_temp_fids_.create(1, pointCount, CV_32SC1);

    d_out_points3d_.create(1, pointCount, CV_32FC3);
    d_out_fids_.create(1, pointCount, CV_32SC1);
    d_out_count_.create(1, 1, CV_32SC1);

    last_max_count_ = pointCount;
    return true;
}

void LaserReconstructCuda::Impl::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }

    if (warmed_up_ && warmup_count_ == pointCount) {
        CALIB_LOG_DEBUG("warmup(): already warmed up for pointCount={}", pointCount);
        return;
    }

    allocateBuffers(pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;

    CALIB_LOG_INFO("warmup(): allocated GPU buffers for pointCount={}", pointCount);
}

LaserReconstructResult LaserReconstructCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_matched_left,
    const cv::cuda::GpuMat& d_matched_right,
    const cv::cuda::GpuMat& d_matched_line_ids,
    const cv::Mat& Q,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    LaserReconstructResult result;

    try {
        const int count = d_matched_left.rows * d_matched_left.cols;

        if (count == 0) {
            result.success = true;
            result.message = "Empty input, no reconstruction";
            result.d_points3d = std::make_shared<cv::cuda::GpuMat>();
            result.d_valid_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.validCount = 0;
            result.totalInput = 0;
            return result;
        }

        if (!allocateBuffers(count)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        float Qf[16];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Qf[i * 4 + j] = static_cast<float>(Q.at<double>(i, j));
            }
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        cudaError_t cpy_err = cudaMemcpyToSymbolAsync(c_Qf, Qf, sizeof(Qf), 0,
                                                       cudaMemcpyHostToDevice, cuda_stream);
        if (cpy_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("cudaMemcpyToSymbolAsync failed: ") + cudaGetErrorString(cpy_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        const int grid = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernelComputeDisparity<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_matched_left.ptr<float2>(),
            d_matched_right.ptr<float2>(),
            count,
            d_disparity_.ptr<float>());

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelComputeDisparity launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        kernelReconstruct3D<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_matched_left.ptr<float2>(),
            d_disparity_.ptr<float>(),
            count,
            params_.minDepth,
            params_.maxDepth,
            d_points3d_raw_.ptr<float3>(),
            d_valid_flags_.ptr<int>(),
            d_temp_fids_.ptr<int>(),
            d_matched_line_ids.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelReconstruct3D launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        size_t temp_bytes_pts = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_pts,
            d_points3d_raw_.ptr<float3>(), d_valid_flags_.ptr<int>(),
            d_out_points3d_.ptr<float3>(), d_out_count_.ptr<int>(),
            count, cuda_stream);

        size_t temp_bytes_fids = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_fids,
            d_temp_fids_.ptr<int>(), d_valid_flags_.ptr<int>(),
            d_out_fids_.ptr<int>(), d_out_count_.ptr<int>(),
            count, cuda_stream);

        size_t max_temp_bytes = temp_bytes_pts;
        if (temp_bytes_fids > max_temp_bytes) max_temp_bytes = temp_bytes_fids;

        if (max_temp_bytes > cub_temp_size_ || d_cub_temp_ == nullptr) {
            safeCudaFree(d_cub_temp_);

            cudaError_t alloc_err = cudaMalloc(&d_cub_temp_, max_temp_bytes);
            if (alloc_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(alloc_err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            cub_temp_size_ = max_temp_bytes;
        }

        cudaError_t cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_points3d_raw_.ptr<float3>(), d_valid_flags_.ptr<int>(),
            d_out_points3d_.ptr<float3>(), d_out_count_.ptr<int>(),
            count, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (points3d) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_fids_.ptr<int>(), d_valid_flags_.ptr<int>(),
            d_out_fids_.ptr<int>(), d_out_count_.ptr<int>(),
            count, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (fids) failed: ") + cudaGetErrorString(cub_err);
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
        cudaMemcpyAsync(&h_count, d_out_count_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.success = true;
        result.message = "Success";
        result.validCount = h_count;
        result.totalInput = count;
        result.d_points3d = std::make_shared<cv::cuda::GpuMat>(d_out_points3d_.colRange(0, h_count).clone());
        result.d_valid_line_ids = std::make_shared<cv::cuda::GpuMat>(d_out_fids_.colRange(0, h_count).clone());

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

void LaserReconstructCuda::Impl::SetParams(const LaserReconstructParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[08-LaserReconstructCuda] setParams() called during process()");
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

void LaserReconstructCuda::Impl::Destroy() {
    if (d_cub_temp_) {
        cudaFree(d_cub_temp_);
        d_cub_temp_ = nullptr;
    }
    cub_temp_size_ = 0;
    d_disparity_.release();
    d_points3d_raw_.release();
    d_valid_flags_.release();
    d_temp_fids_.release();
    d_out_points3d_.release();
    d_out_fids_.release();
    d_out_count_.release();
    warmed_up_ = false;
    warmup_count_ = 0;
    last_max_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
