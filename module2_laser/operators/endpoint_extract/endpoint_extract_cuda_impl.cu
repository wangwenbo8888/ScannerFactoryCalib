/**
 * @file endpoint_extract_cuda_impl.cu
 * @brief 激光线3D端点提取CUDA算子 - CUDA实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: CUB DeviceReduce::Max 找到最大 line_id
 *   Step 2: kernelComputeLineSums 计算每条线的点坐标之和与计数
 *   Step 3: kernelNormalizeCentroids 计算每条线的质心
 *   Step 4: kernelFindMaxDist 找到距质心最远点的距离平方
 *   Step 5: kernelFindMaxDistIdx 找到距质心最远点的索引（端点A）
 *   Step 6: kernelGatherRef 收集端点A的坐标
 *   Step 7: kernelFindMaxDist 找到距端点A最远点的距离平方
 *   Step 8: kernelFindMaxDistIdx 找到距端点A最远点的索引（端点B）
 *   Step 9: kernelCollectEndpoints 收集端点并编号
 */

#include "endpoint_extract_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(09, EndpointExtractCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void __launch_bounds__(256, 4) kernelComputeLineSums(
    const float3* __restrict__ d_points,
    const int* __restrict__ d_line_ids,
    int count,
    float3* __restrict__ d_sums,
    int* __restrict__ d_counts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int lid = d_line_ids[idx];
    float3 p = d_points[idx];
    atomicAdd(&d_sums[lid].x, p.x);
    atomicAdd(&d_sums[lid].y, p.y);
    atomicAdd(&d_sums[lid].z, p.z);
    atomicAdd(&d_counts[lid], 1);
}

__global__ void __launch_bounds__(256, 4) kernelNormalizeCentroids(
    const float3* __restrict__ d_sums,
    const int* __restrict__ d_counts,
    int numLines,
    float3* __restrict__ d_centroids)
{
    const int lid = blockIdx.x * blockDim.x + threadIdx.x;
    if (lid >= numLines) return;

    if (d_counts[lid] > 0) {
        float inv = 1.0f / static_cast<float>(d_counts[lid]);
        d_centroids[lid] = make_float3(
            d_sums[lid].x * inv,
            d_sums[lid].y * inv,
            d_sums[lid].z * inv);
    } else {
        d_centroids[lid] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void __launch_bounds__(256, 4) kernelFindMaxDist(
    const float3* __restrict__ d_points,
    const int* __restrict__ d_line_ids,
    int count,
    const float3* __restrict__ d_ref,
    unsigned int* __restrict__ d_max_dist_sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int lid = d_line_ids[idx];
    float dx = d_points[idx].x - d_ref[lid].x;
    float dy = d_points[idx].y - d_ref[lid].y;
    float dz = d_points[idx].z - d_ref[lid].z;
    float dist_sq = dx * dx + dy * dy + dz * dz;
    unsigned int dist_uint = __float_as_uint(dist_sq);
    atomicMax(&d_max_dist_sq[lid], dist_uint);
}

__global__ void __launch_bounds__(256, 4) kernelFindMaxDistIdx(
    const float3* __restrict__ d_points,
    const int* __restrict__ d_line_ids,
    int count,
    const float3* __restrict__ d_ref,
    const unsigned int* __restrict__ d_max_dist_sq,
    int* __restrict__ d_max_idx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int lid = d_line_ids[idx];
    float dx = d_points[idx].x - d_ref[lid].x;
    float dy = d_points[idx].y - d_ref[lid].y;
    float dz = d_points[idx].z - d_ref[lid].z;
    float dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq == __uint_as_float(d_max_dist_sq[lid])) {
        d_max_idx[lid] = idx;
    }
}

__global__ void __launch_bounds__(256, 4) kernelGatherRef(
    const float3* __restrict__ d_points,
    const int* __restrict__ d_indices,
    int numLines,
    float3* __restrict__ d_ref)
{
    const int lid = blockIdx.x * blockDim.x + threadIdx.x;
    if (lid >= numLines) return;
    d_ref[lid] = d_points[d_indices[lid]];
}

__global__ void __launch_bounds__(256, 4) kernelCollectEndpoints(
    const float3* __restrict__ d_points,
    const int* __restrict__ d_ep_a_idx,
    const int* __restrict__ d_ep_b_idx,
    const int* __restrict__ d_line_counts,
    int numPossibleLines,
    float3* __restrict__ d_endpoints,
    int* __restrict__ d_endpoint_ids,
    int* __restrict__ d_line_ids,
    int* __restrict__ d_num_lines)
{
    const int lid = blockIdx.x * blockDim.x + threadIdx.x;
    if (lid >= numPossibleLines) return;

    if (d_line_counts[lid] <= 0) return;

    int outLineIdx = atomicAdd(d_num_lines, 1);
    int epBase = outLineIdx * 2;

    d_endpoints[epBase]     = d_points[d_ep_a_idx[lid]];
    d_endpoints[epBase + 1] = d_points[d_ep_b_idx[lid]];

    d_endpoint_ids[epBase]     = epBase;
    d_endpoint_ids[epBase + 1] = epBase + 1;

    d_line_ids[outLineIdx] = lid;
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

EndpointExtractCuda::Impl::Impl(const EndpointExtractParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[09-EndpointExtractCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[09-EndpointExtractCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

EndpointExtractCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }

    safeCudaFree(d_cub_temp_);
    cub_temp_size_ = 0;
    last_max_fid_ = 0;
}

bool EndpointExtractCuda::Impl::allocateBuffers(int maxFid, int pointCount) {
    if (maxFid < 0 || pointCount <= 0) return true;

    if (maxFid <= last_max_fid_ &&
        !d_line_sums_.empty() &&
        !d_line_counts_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for maxFid={}, pointCount={}", maxFid, pointCount);

    int numLines = maxFid + 1;

    d_line_sums_.create(1, numLines, CV_32FC3);
    d_line_counts_.create(1, numLines, CV_32SC1);
    d_centroids_.create(1, numLines, CV_32FC3);
    d_max_dist_sq_.create(1, numLines, CV_32SC1);
    d_ep_a_idx_.create(1, numLines, CV_32SC1);
    d_ep_b_idx_.create(1, numLines, CV_32SC1);
    d_endpoint_a_.create(1, numLines, CV_32FC3);
    d_max_fid_.create(1, 1, CV_32SC1);

    d_out_endpoints_.create(1, numLines * 2, CV_32FC3);
    d_out_ep_ids_.create(1, numLines * 2, CV_32SC1);
    d_out_line_ids_.create(1, numLines, CV_32SC1);
    d_out_num_lines_.create(1, 1, CV_32SC1);

    last_max_fid_ = maxFid;
    return true;
}

void EndpointExtractCuda::Impl::Warmup(int pointCount, int maxFrameId) {
    if (pointCount <= 0 || maxFrameId < 0) {
        CALIB_LOG_WARN("warmup(): invalid pointCount={} or maxFrameId={}, skipping",
                       pointCount, maxFrameId);
        return;
    }

    if (warmed_up_ && warmup_count_ == pointCount && warmup_max_fid_ == maxFrameId) {
        CALIB_LOG_DEBUG("warmup(): already warmed up for pointCount={}, maxFrameId={}",
                        pointCount, maxFrameId);
        return;
    }

    allocateBuffers(maxFrameId, pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;
    warmup_max_fid_ = maxFrameId;

    CALIB_LOG_INFO("warmup(): allocated GPU buffers for pointCount={}, maxFrameId={}",
                   pointCount, maxFrameId);
}

EndpointExtractResult EndpointExtractCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points3d,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    EndpointExtractResult result;

    const bool doTiming = params_.enableTiming;

    cudaEvent_t evStart = nullptr, evEnd = nullptr;
    cudaEvent_t ev[12];
    if (doTiming) {
        for (int i = 0; i < 12; ++i) cudaEventCreate(&ev[i]);
        cudaEventCreate(&evStart);
        cudaEventCreate(&evEnd);
    }

    try {
        const int count = d_points3d.rows * d_points3d.cols;

        if (count == 0) {
            result.success = true;
            result.message = "Empty input, no endpoints to extract";
            result.d_endpoints = std::make_shared<cv::cuda::GpuMat>();
            result.d_endpoint_ids = std::make_shared<cv::cuda::GpuMat>();
            result.d_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.numEndpoints = 0;
            result.numLines = 0;
            result.totalInput = 0;
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        if (doTiming) cudaEventRecord(evStart, cuda_stream);

        // Step 0: Find max line_id using CUB
        if (d_max_fid_.empty()) {
            d_max_fid_.create(1, 1, CV_32SC1);
        }

        size_t temp_bytes = 0;
        cub::DeviceReduce::Max(nullptr, temp_bytes,
                               d_line_ids.ptr<int>(),
                               d_max_fid_.ptr<int>(),
                               count, cuda_stream);

        if (temp_bytes > cub_temp_size_ || d_cub_temp_ == nullptr) {
            safeCudaFree(d_cub_temp_);
            cudaError_t alloc_err = cudaMalloc(&d_cub_temp_, temp_bytes);
            if (alloc_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp alloc failed: ") + cudaGetErrorString(alloc_err);
                CALIB_LOG_ERROR("process(): {}", result.message);
                return result;
            }
            cub_temp_size_ = temp_bytes;
        }

        cub::DeviceReduce::Max(d_cub_temp_, cub_temp_size_,
                               d_line_ids.ptr<int>(),
                               d_max_fid_.ptr<int>(),
                               count, cuda_stream);

        cudaStreamSynchronize(cuda_stream);

        int h_max_fid = 0;
        cudaMemcpy(&h_max_fid, d_max_fid_.ptr<int>(), sizeof(int),
                   cudaMemcpyDeviceToHost);

        if (h_max_fid < 0) h_max_fid = 0;
        const int numPossibleLines = h_max_fid + 1;

        CALIB_LOG_DEBUG("Max line_id={}, numPossibleLines={}", h_max_fid, numPossibleLines);

        if (!allocateBuffers(h_max_fid, count)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[0], cuda_stream);

        // Step 1: Zero per-line buffers
        cudaMemsetAsync(d_line_sums_.data, 0,
                        numPossibleLines * sizeof(float3), cuda_stream);
        cudaMemsetAsync(d_line_counts_.data, 0,
                        numPossibleLines * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_max_dist_sq_.data, 0,
                        numPossibleLines * sizeof(unsigned int), cuda_stream);

        if (doTiming) cudaEventRecord(ev[1], cuda_stream);

        // Step 2: Compute line sums and counts
        int grid = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelComputeLineSums<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_line_ids.ptr<int>(),
            count,
            d_line_sums_.ptr<float3>(),
            d_line_counts_.ptr<int>());

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelComputeLineSums failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[2], cuda_stream);

        // Step 3: Normalize centroids
        grid = (numPossibleLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelNormalizeCentroids<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_line_sums_.ptr<float3>(),
            d_line_counts_.ptr<int>(),
            numPossibleLines,
            d_centroids_.ptr<float3>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelNormalizeCentroids failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[3], cuda_stream);

        // Step 4: Find max squared distance from centroid
        grid = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelFindMaxDist<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_line_ids.ptr<int>(),
            count,
            d_centroids_.ptr<float3>(),
            d_max_dist_sq_.ptr<unsigned int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelFindMaxDist(centroid) failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[4], cuda_stream);

        // Step 5: Find endpoint A index
        kernelFindMaxDistIdx<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_line_ids.ptr<int>(),
            count,
            d_centroids_.ptr<float3>(),
            d_max_dist_sq_.ptr<unsigned int>(),
            d_ep_a_idx_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelFindMaxDistIdx(A) failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[5], cuda_stream);

        // Step 6: Gather endpoint A coordinates
        grid = (numPossibleLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelGatherRef<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_ep_a_idx_.ptr<int>(),
            numPossibleLines,
            d_endpoint_a_.ptr<float3>());

        if (doTiming) cudaEventRecord(ev[6], cuda_stream);

        // Step 7: Find max squared distance from endpoint A
        cudaMemsetAsync(d_max_dist_sq_.data, 0,
                        numPossibleLines * sizeof(unsigned int), cuda_stream);

        grid = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelFindMaxDist<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_line_ids.ptr<int>(),
            count,
            d_endpoint_a_.ptr<float3>(),
            d_max_dist_sq_.ptr<unsigned int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelFindMaxDist(endpointA) failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[7], cuda_stream);

        // Step 8: Find endpoint B index
        kernelFindMaxDistIdx<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_line_ids.ptr<int>(),
            count,
            d_endpoint_a_.ptr<float3>(),
            d_max_dist_sq_.ptr<unsigned int>(),
            d_ep_b_idx_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelFindMaxDistIdx(B) failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[8], cuda_stream);

        // Step 9: Collect endpoints and assign IDs
        cudaMemsetAsync(d_out_num_lines_.data, 0, sizeof(int), cuda_stream);

        grid = (numPossibleLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelCollectEndpoints<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_points3d.ptr<float3>(),
            d_ep_a_idx_.ptr<int>(),
            d_ep_b_idx_.ptr<int>(),
            d_line_counts_.ptr<int>(),
            numPossibleLines,
            d_out_endpoints_.ptr<float3>(),
            d_out_ep_ids_.ptr<int>(),
            d_out_line_ids_.ptr<int>(),
            d_out_num_lines_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelCollectEndpoints failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        if (doTiming) cudaEventRecord(ev[9], cuda_stream);

        // Step 10: D2H copy
        int h_num_lines = 0;
        cudaMemcpyAsync(&h_num_lines, d_out_num_lines_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        if (doTiming) {
            cudaEventRecord(ev[10], cuda_stream);
            cudaEventRecord(evEnd, cuda_stream);
            cudaEventSynchronize(evEnd);
        }

        result.success = true;
        result.message = "Success";
        result.numLines = h_num_lines;
        result.numEndpoints = h_num_lines * 2;
        result.totalInput = count;
        result.d_endpoints = std::make_shared<cv::cuda::GpuMat>(
            d_out_endpoints_.colRange(0, result.numEndpoints).clone());
        result.d_endpoint_ids = std::make_shared<cv::cuda::GpuMat>(
            d_out_ep_ids_.colRange(0, result.numEndpoints).clone());
        result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
            d_out_line_ids_.colRange(0, result.numLines).clone());

        if (doTiming) {
            auto elapsed = [&](cudaEvent_t a, cudaEvent_t b) -> float {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, a, b);
                return ms * 1000.0f;
            };
            result.timing.us_findMaxFid        = elapsed(evStart, ev[0]);
            result.timing.us_memsetBuffers     = elapsed(ev[0], ev[1]);
            result.timing.us_computeLineSums   = elapsed(ev[1], ev[2]);
            result.timing.us_normalizeCentroids= elapsed(ev[2], ev[3]);
            result.timing.us_findMaxDistA      = elapsed(ev[3], ev[4]);
            result.timing.us_findMaxDistIdxA   = elapsed(ev[4], ev[5]);
            result.timing.us_gatherRefA        = elapsed(ev[5], ev[6]);
            result.timing.us_findMaxDistB      = elapsed(ev[6], ev[7]);
            result.timing.us_findMaxDistIdxB   = elapsed(ev[7], ev[8]);
            result.timing.us_collectEndpoints  = elapsed(ev[8], ev[9]);
            result.timing.us_d2hCopy           = elapsed(ev[9], ev[10]);
            result.timing.us_total             = elapsed(evStart, ev[10]);
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("process(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("process(): {}", result.message);
    }

    if (doTiming) {
        for (int i = 0; i < 12; ++i) cudaEventDestroy(ev[i]);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evEnd);
    }

    return result;
}

void EndpointExtractCuda::Impl::SetParams(const EndpointExtractParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[09-EndpointExtractCuda] setParams() called during process()");
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
    warmup_max_fid_ = 0;

    CALIB_LOG_INFO("setParams(): params updated, warmup reset");
}
