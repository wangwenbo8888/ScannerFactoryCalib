/**
 * @file epipolar_interp_opt_cuda_impl.cu
 * @brief 激光中心点极线插值CUDA算子·优化版实现
 *
 * 与原版（epipolar_interp_cuda_impl.cu, v2 极线驱动重采样）的差异仅在执行方式:
 *   Step 1: thrust::sort_by_key → CUB DeviceRadixSort::SortPairs（基数排序）
 *   Step 2: host 全量下载+扫描 → GPU 端 边界标记+前缀和+scatter 建线号表，
 *           y 范围原子归约；仅 1 次 12B pinned 回传，线表零上传
 *   Step 4: CUB temp 与全部缓冲 grow-only 缓存
 * 插值 kernel 与原版逐字一致（optEpipolarInterp）。
 */

#include "epipolar_interp_opt_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace calib;

CALIB_DEFINE_LOG_TAG(06, EpipolarInterpOptCuda);

static constexpr int BLOCK_SIZE = 256;
static constexpr float EPSILON = 1e-4f;      // 精确命中判定（与原版一致）
static constexpr float DEGEN_EPS = 1e-6f;    // 除零保护

// ============================================================================
// kernels（全部带 opt 前缀，避免与原版 .cu 符号冲突）
// ============================================================================

__global__ void optBuildSortKeys(
    const float2* __restrict__ d_pts,
    const int* __restrict__ d_fids,
    unsigned long long* __restrict__ d_keys,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float y = d_pts[idx].y;
    // 与原版一致的键打包（本场景 y>=0 简化处理）
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;
    unsigned long long key =
        ((unsigned long long)(unsigned int)d_fids[idx] << 32) | (unsigned int)yi;
    d_keys[idx] = key;
}

__global__ void optFillIndices(int* __restrict__ d_idx, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_idx[idx] = idx;
}

__global__ void optGatherSorted(
    const float2* __restrict__ d_pts_src,
    const int* __restrict__ d_fids_src,
    const int* __restrict__ d_indices,
    float2* __restrict__ d_pts_dst,
    int* __restrict__ d_fids_dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = d_indices[i];
    d_pts_dst[i] = d_pts_src[src];
    d_fids_dst[i] = d_fids_src[src];
}

// 线号边界标记: flags[i] = (i==0 || fids[i]!=fids[i-1])
__global__ void optBoundaryFlags(
    const int* __restrict__ d_fids,
    int* __restrict__ d_flags,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_flags[i] = (i == 0 || d_fids[i] != d_fids[i - 1]) ? 1 : 0;
}

__global__ void optInitMeta(unsigned int* __restrict__ d_meta)
{
    if (threadIdx.x == 0) {
        d_meta[0] = 0u;             // num_lines
        d_meta[1] = 0xffffffffu;    // umin 初值
        d_meta[2] = 0u;             // umax 初值
    }
}

// 由 flags + 前缀和秩 构建线号表（升序去重 map + 区间 begin；末线程补 end 项与 num_lines）
__global__ void optScatterLineTable(
    const int* __restrict__ d_fids,
    const int* __restrict__ d_flags,
    const int* __restrict__ d_ranks,
    int n,
    int* __restrict__ d_map,
    int* __restrict__ d_begin,
    unsigned int* __restrict__ d_meta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_flags[i]) {
        int r = d_ranks[i];
        d_map[r] = d_fids[i];
        d_begin[r] = i;
    }
    if (i == n - 1) {
        int num_runs = d_ranks[i] + d_flags[i];
        d_begin[num_runs] = n;
        d_meta[0] = (unsigned int)num_runs;
    }
}

// y 范围原子归约（y>=0 时 uint 位序单调，与原版排序键假设一致）
__global__ void optYRange(
    const float2* __restrict__ d_pts,
    int n,
    unsigned int* __restrict__ d_meta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int u = (unsigned int)__float_as_int(d_pts[i].y);
    atomicMin(&d_meta[1], u);
    atomicMax(&d_meta[2], u);
}

// 插值主 kernel: 与原版 kernelEpipolarInterp 逐字一致
__global__ void __launch_bounds__(256, 4) optEpipolarInterp(
    const float2* __restrict__ d_pts_sorted,
    const int* __restrict__ d_fids_sorted,
    const int* __restrict__ d_line_begin,
    const int* __restrict__ d_line_ids_map,
    int num_lines,
    int num_rows,
    int row_begin,
    float row_step,
    float window,
    float max_x_diff,
    int* __restrict__ d_flags,
    float2* __restrict__ d_out_pts,
    int* __restrict__ d_out_fids)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= num_lines * num_rows) return;

    int line_idx = linear / num_rows;
    int row_idx = linear - line_idx * num_rows;

    int fid = d_line_ids_map[line_idx];
    int begin = d_line_begin[line_idx];
    int end = d_line_begin[line_idx + 1];
    int cnt = end - begin;
    if (cnt <= 0) { d_flags[linear] = 0; return; }

    float y_k = (float)(row_begin + row_idx) * row_step;

    int lo = begin, hi = end;
    float ylw = y_k - window;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_pts_sorted[mid].y < ylw) lo = mid + 1;
        else hi = mid;
    }
    int hit_idx = -1;
    float hit_dy = 1e30f;
    int up_idx = -1;
    float up_dy = 1e30f;
    int down_idx = -1;
    float down_dy = 1e30f;

    for (int i = lo; i < end; ++i) {
        float y = d_pts_sorted[i].y;
        if (y > y_k + window) break;
        float dy = y - y_k;
        float ady = fabsf(dy);
        if (ady < EPSILON) {
            if (ady < hit_dy) { hit_dy = ady; hit_idx = i; }
        } else if (dy < 0.0f) {
            if (ady < up_dy) { up_dy = ady; up_idx = i; }
        } else {
            if (ady < down_dy) { down_dy = ady; down_idx = i; }
        }
    }

    float2 out;
    if (hit_idx >= 0) {
        out = d_pts_sorted[hit_idx];
        out.y = y_k;
    } else {
        if (up_idx < 0 || down_idx < 0) { d_flags[linear] = 0; return; }
        float2 pu = d_pts_sorted[up_idx];
        float2 pd = d_pts_sorted[down_idx];
        if (fabsf(pu.x - pd.x) > max_x_diff) { d_flags[linear] = 0; return; }
        float denom = pd.y - pu.y;
        if (fabsf(denom) < DEGEN_EPS) { d_flags[linear] = 0; return; }
        float t = (y_k - pu.y) / denom;
        out.x = pu.x + t * (pd.x - pu.x);
        out.y = y_k;
    }

    d_out_pts[linear] = out;
    d_out_fids[linear] = fid;
    d_flags[linear] = 1;
}

// ============================================================================
// ScopedFlag (Debug-only thread safety)
// ============================================================================

#ifndef NDEBUG
class ScopedFlag {
public:
    explicit ScopedFlag(std::atomic<bool>* flag_) : flag_(flag_) {
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

EpipolarInterpOptCuda::Impl::Impl(const EpipolarInterpParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[06-EpipolarInterpOptCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[06-EpipolarInterpOptCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }

    if (cudaMallocHost(&h_meta_pinned_, 4 * sizeof(int)) != cudaSuccess) {
        h_meta_pinned_ = nullptr;
        throw std::runtime_error("[06-EpipolarInterpOptCuda] pinned host alloc failed");
    }

    d_meta_.create(1, 4, CV_32SC1);
}

EpipolarInterpOptCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
    if (h_meta_pinned_) {
        cudaFreeHost(h_meta_pinned_);
        h_meta_pinned_ = nullptr;
    }
}

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

void EpipolarInterpOptCuda::Impl::ensureCapacity(int pointCount) {
    if (pointCount <= capacity_) return;

    CALIB_LOG_DEBUG("Ensuring GPU capacity for {} points", pointCount);

    d_keys_in_.create(1, pointCount, CV_64FC1);
    d_keys_out_.create(1, pointCount, CV_64FC1);
    d_idx_in_.create(1, pointCount, CV_32SC1);
    d_idx_out_.create(1, pointCount, CV_32SC1);
    d_sorted_pts_.create(1, pointCount, CV_32FC2);
    d_sorted_fids_.create(1, pointCount, CV_32SC1);
    d_rl_flags_.create(1, pointCount, CV_32SC1);
    d_rl_ranks_.create(1, pointCount, CV_32SC1);
    d_line_begin_.create(1, pointCount + 1, CV_32SC1);
    d_line_ids_map_.create(1, pointCount, CV_32SC1);
    capacity_ = pointCount;
}

void EpipolarInterpOptCuda::Impl::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }
    if (warmed_up_ && warmup_count_ == pointCount) {
        CALIB_LOG_DEBUG("Warmup(): already warmed up for {} points", pointCount);
        return;
    }
    ensureCapacity(pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;
    CALIB_LOG_INFO("Warmup(): ensured GPU capacity for {} points", pointCount);
}

EpipolarInterpResult EpipolarInterpOptCuda::Impl::Execute(
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

        ensureCapacity(pointCount);
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
        const int gridN = (pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // ================= Step 1: 按 (fid, y) 排序（CUB 基数排序） =================
        optBuildSortKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_points.ptr<float2>(), d_line_ids.ptr<int>(),
            (unsigned long long*)d_keys_in_.ptr<double>(), pointCount);
        optFillIndices<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_idx_in_.ptr<int>(), pointCount);

        if (radix_cap_ < pointCount || d_radix_temp_ == nullptr) {
            size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(
                nullptr, bytes,
                (const unsigned long long*)nullptr, (unsigned long long*)nullptr,
                (const int*)nullptr, (int*)nullptr,
                pointCount, 0, sizeof(unsigned long long) * 8, cs);
            safeCudaFree(d_radix_temp_);
            cudaError_t err = cudaMalloc(&d_radix_temp_, bytes);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("radix temp alloc failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            radix_temp_size_ = bytes;
            radix_cap_ = pointCount;
        }
        {
            cudaError_t err = cub::DeviceRadixSort::SortPairs(
                d_radix_temp_, radix_temp_size_,
                (const unsigned long long*)d_keys_in_.ptr<double>(),
                (unsigned long long*)d_keys_out_.ptr<double>(),
                (const int*)d_idx_in_.ptr<int>(),
                d_idx_out_.ptr<int>(),
                pointCount, 0, sizeof(unsigned long long) * 8, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("radix sort failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
        }

        optGatherSorted<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_points.ptr<float2>(), d_line_ids.ptr<int>(),
            (const int*)d_idx_out_.ptr<int>(),
            d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(), pointCount);

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Sort kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        // ================= Step 2: GPU 端线号表 + y 范围 =================
        optBoundaryFlags<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_fids_.ptr<int>(), d_rl_flags_.ptr<int>(), pointCount);

        if (scan_cap_ < pointCount || d_scan_temp_ == nullptr) {
            size_t bytes = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr, bytes,
                (const int*)nullptr, (int*)nullptr, pointCount, cs);
            safeCudaFree(d_scan_temp_);
            cudaError_t err = cudaMalloc(&d_scan_temp_, bytes);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("scan temp alloc failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            scan_temp_size_ = bytes;
            scan_cap_ = pointCount;
        }
        {
            cudaError_t err = cub::DeviceScan::ExclusiveSum(
                d_scan_temp_, scan_temp_size_,
                (const int*)d_rl_flags_.ptr<int>(),
                d_rl_ranks_.ptr<int>(), pointCount, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("exclusive sum failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
        }

        optInitMeta<<<1, 32, 0, cs>>>((unsigned int*)d_meta_.ptr<int>());
        optScatterLineTable<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_fids_.ptr<int>(), d_rl_flags_.ptr<int>(),
            (const int*)d_rl_ranks_.ptr<int>(), pointCount,
            d_line_ids_map_.ptr<int>(), d_line_begin_.ptr<int>(),
            (unsigned int*)d_meta_.ptr<int>());
        optYRange<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_pts_.ptr<float2>(), pointCount, (unsigned int*)d_meta_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Line table kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        // 唯一一次中途回传: 12B pinned {num_lines, umin, umax}
        cudaMemcpyAsync(h_meta_pinned_, d_meta_.ptr<int>(), 3 * sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);

        const int num_lines = h_meta_pinned_[0];
        float y_min = 0.f, y_max = 0.f;
        {
            // y>=0 时 uint 位序与 float 大小序一致（与排序键同一假设），位型直读即可
            unsigned int umin = (unsigned int)h_meta_pinned_[1];
            unsigned int umax = (unsigned int)h_meta_pinned_[2];
            std::memcpy(&y_min, &umin, sizeof(float));
            std::memcpy(&y_max, &umax, sizeof(float));
        }

        const float step = params_.epipolar_row_step;
        const int row_begin = (int)std::floor(y_min / step);
        const int row_end = (int)std::ceil(y_max / step);
        const int num_rows = row_end - row_begin + 1;
        const int total_lines = num_lines * num_rows;

        // ================= Step 3: 极线插值 kernel =================
        if (total_lines > lines_cap_) {
            d_flags_.create(1, total_lines, CV_32SC1);
            d_temp_interp_.create(1, total_lines, CV_32FC2);
            d_temp_fids_.create(1, total_lines, CV_32SC1);
            d_output_.create(1, total_lines, CV_32FC2);
            d_output_fids_.create(1, total_lines, CV_32SC1);
            d_output_count_.create(1, 1, CV_32SC1);
            lines_cap_ = total_lines;
        }

        {
            const int gridL = (total_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;
            optEpipolarInterp<<<gridL, BLOCK_SIZE, 0, cs>>>(
                d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(),
                d_line_begin_.ptr<int>(), d_line_ids_map_.ptr<int>(),
                num_lines, num_rows, row_begin,
                step, params_.effectiveWindow(), params_.max_x_diff,
                d_flags_.ptr<int>(), d_temp_interp_.ptr<float2>(),
                d_temp_fids_.ptr<int>());
        }

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        // ================= Step 4: CUB 压缩（temp 容量缓存） =================
        if (cub_cap_ < total_lines || d_cub_temp_ == nullptr) {
            size_t t1 = 0, t2 = 0;
            cub::DeviceSelect::Flagged(nullptr, t1,
                (const float2*)nullptr, (const int*)nullptr,
                (float2*)nullptr, (int*)nullptr, total_lines, cs);
            cub::DeviceSelect::Flagged(nullptr, t2,
                (const int*)nullptr, (const int*)nullptr,
                (int*)nullptr, (int*)nullptr, total_lines, cs);
            const size_t need = t1 > t2 ? t1 : t2;
            safeCudaFree(d_cub_temp_);
            cudaError_t err = cudaMalloc(&d_cub_temp_, need);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            cub_temp_size_ = need;
            cub_cap_ = total_lines;
        }

        {
            cudaError_t err = cub::DeviceSelect::Flagged(
                d_cub_temp_, cub_temp_size_,
                (const float2*)d_temp_interp_.ptr<float2>(), (const int*)d_flags_.ptr<int>(),
                d_output_.ptr<float2>(), d_output_count_.ptr<int>(), total_lines, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            err = cub::DeviceSelect::Flagged(
                d_cub_temp_, cub_temp_size_,
                (const int*)d_temp_fids_.ptr<int>(), (const int*)d_flags_.ptr<int>(),
                d_output_fids_.ptr<int>(), d_output_count_.ptr<int>(), total_lines, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
        }

        int h_count = 0;
        cudaMemcpyAsync(&h_count, d_output_count_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);

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

void EpipolarInterpOptCuda::Impl::SetParams(const EpipolarInterpParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("SetParams(): called while Execute() is running");
        throw std::runtime_error("[06-EpipolarInterpOptCuda] SetParams() called during Execute()");
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

void EpipolarInterpOptCuda::Impl::Destroy() {
    safeCudaFree(d_radix_temp_);
    safeCudaFree(d_scan_temp_);
    safeCudaFree(d_cub_temp_);
    radix_temp_size_ = scan_temp_size_ = cub_temp_size_ = 0;
    radix_cap_ = scan_cap_ = cub_cap_ = 0;
    if (h_meta_pinned_) {
        cudaFreeHost(h_meta_pinned_);
        h_meta_pinned_ = nullptr;
    }
    d_keys_in_.release();
    d_keys_out_.release();
    d_idx_in_.release();
    d_idx_out_.release();
    d_sorted_pts_.release();
    d_sorted_fids_.release();
    d_rl_flags_.release();
    d_rl_ranks_.release();
    d_line_begin_.release();
    d_line_ids_map_.release();
    d_meta_.release();
    d_flags_.release();
    d_temp_interp_.release();
    d_temp_fids_.release();
    d_output_.release();
    d_output_fids_.release();
    d_output_count_.release();
    capacity_ = 0;
    lines_cap_ = 0;
    warmed_up_ = false;
    warmup_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
