/**
 * @file epipolar_interp_cuda_impl.cu
 * @brief 激光中心点极线插值CUDA算子实现（极线驱动重采样版）
 *
 * 算法（v2, 2026-08-25 重构）:
 *   Step 1: 按 (line_id, y) 稳定排序（sortPointsKernel + thrust 分段排序思路的
 *           简化实现：拷贝索引到 host 用 thrust::sort_by_key 排序后回传）
 *   Step 2: 逐极线 kernel——每个线程处理一条 (line_id, 极线行) 组合:
 *     a) 精确命中: |y - y_k| < EPS 的点直接采用（多者取最近）
 *     b) 否则双窗搜索: 上窗/下窗各取距 y_k 最近的点（二分定位 + 邻域扫描）
 *     c) X 约束: |x_up - x_down| <= max_x_diff
 *     d) 插值: 上点->下点连线与极线 y_k 的交点
 *   Step 3: CUB DeviceSelect::Flagged 压缩有效结果
 *
 * 输出仅含插值/命中点，原始输入点不透传（v2 行为变更）。
 */

#include "epipolar_interp_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace calib;


CALIB_DEFINE_LOG_TAG(06, EpipolarInterpCuda);

// ============================================================================
// 常量
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr float EPSILON = 1e-4f;      // 精确命中判定（y 恰在极线上）
static constexpr float DEGEN_EPS = 1e-6f;    // 除零保护

// ============================================================================
// 排序 kernel: 按 (fid, y) 生成 64 位排序键
//    key = ((int64)fid << 32) | float_as_ordered_int(y)
//    y>=0: bit31=0, 单调映射; y<0: 翻转保证全序（本场景 y>=0，简化处理）
// ============================================================================

__global__ void kernelBuildSortKeys(
    const float2* __restrict__ d_pts,
    const int* __restrict__ d_fids,
    long long* __restrict__ d_keys,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float y = d_pts[idx].y;
    // y>=0 单调映射到非负 int（分辨率 1/256 px 足够窗口比较用, 精确比较仍用原值）
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;   // 负数翻转（保序）
    long long key = ((long long)d_fids[idx] << 32) | (unsigned int)yi;
    d_keys[idx] = key;
}

// gather: 按 sorted 原索引收集点到排序后数组
__global__ void gatherKernel(
    const float2* __restrict__ d_pts_src,
    const int* __restrict__ d_fids_src,
    const int* __restrict__ d_indices,   // sorted 原索引（当前存于 fids 缓冲）
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

// ============================================================================
// 极线插值主 kernel
//    每线程一条 (line_id, 极线行)。输入点已按 (fid, y) 升序。
//    d_line_begin/d_line_end: 每个线号在排序后数组中的 [begin, end) 区间
// ============================================================================

__global__ void __launch_bounds__(256, 4) kernelEpipolarInterp(
    const float2* __restrict__ d_pts_sorted,   // 按 (fid, y) 排序的点
    const int* __restrict__ d_fids_sorted,
    const int* __restrict__ d_line_begin,      // 线号 -> 起始索引（线号即下标, 0..num_lines-1 映射见 d_line_ids_map）
    const int* __restrict__ d_line_ids_map,    // 紧凑线号表（升序去重）
    int num_lines,
    int num_rows,                              // 本帧覆盖的极线行数 [row_begin, row_begin+num_rows)
    int row_begin,
    float row_step,
    float window,
    float max_x_diff,
    int* __restrict__ d_flags,                 // 输出: 1=有效
    float2* __restrict__ d_out_pts,
    int* __restrict__ d_out_fids)
{
    // 线性索引: line_idx * num_rows + row_idx
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= num_lines * num_rows) return;

    int line_idx = linear / num_rows;
    int row_idx = linear - line_idx * num_rows;

    int fid = d_line_ids_map[line_idx];
    int begin = d_line_begin[line_idx];
    int end = d_line_begin[line_idx + 1];   // 注意 host 端多分配一格
    int cnt = end - begin;
    if (cnt <= 0) { d_flags[linear] = 0; return; }

    float y_k = (float)(row_begin + row_idx) * row_step;

    // ---- 二分定位: 第一个 y >= y_k - window 的点 ----
    int lo = begin, hi = end;
    float ylw = y_k - window;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_pts_sorted[mid].y < ylw) lo = mid + 1;
        else hi = mid;
    }
    // 从 lo 向上扫描到 y <= y_k + window 为止, 找三类候选
    int hit_idx = -1;        // 精确命中
    float hit_dy = 1e30f;
    int up_idx = -1;         // 上窗最近
    float up_dy = 1e30f;
    int down_idx = -1;       // 下窗最近
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
        // 精确命中: 直接采用, 不插值
        out = d_pts_sorted[hit_idx];
        out.y = y_k;
    } else {
        // 双窗必须都有
        if (up_idx < 0 || down_idx < 0) { d_flags[linear] = 0; return; }
        float2 pu = d_pts_sorted[up_idx];
        float2 pd = d_pts_sorted[down_idx];
        // X 约束
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

bool EpipolarInterpCuda::Impl::allocateBuffers(int pointCount) {
    if (pointCount <= 0) {
        return true;
    }

    if (pointCount <= last_max_pairs_count_ &&
        !d_flags_.empty() &&
        !d_temp_interp_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for {} points", pointCount);

    d_sort_keys_.create(1, pointCount, CV_64FC1);
    d_sorted_pts_.create(1, pointCount, CV_32FC2);
    d_sorted_fids_.create(1, pointCount, CV_32SC1);
    d_indices_.create(1, pointCount, CV_32SC1);
    last_max_pairs_count_ = pointCount;
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

        // ================= Step 1: 按 (fid, y) 排序 =================
        {
            int grid = (pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelBuildSortKeys<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_points.ptr<float2>(), d_line_ids.ptr<int>(),
                (long long*)d_sort_keys_.ptr<double>(), pointCount);

            // thrust: sort_by_key(keys -> 原索引), 再 gather 到排序数组
            // 索引缓冲独立于输出缓冲（避免同缓冲边读边写竞态）
            thrust::device_ptr<long long> keys_begin((long long*)d_sort_keys_.ptr<double>());
            thrust::device_ptr<int> idx_begin((int*)d_indices_.ptr<int>());
            thrust::sequence(thrust::cuda::par.on(cuda_stream),
                             idx_begin, idx_begin + pointCount);
            thrust::sort_by_key(thrust::cuda::par.on(cuda_stream),
                                keys_begin, keys_begin + pointCount,
                                idx_begin);
            gatherKernel<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_points.ptr<float2>(), d_line_ids.ptr<int>(),
                (int*)d_indices_.ptr<int>(),
                d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(),
                pointCount);
        }
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Sort kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        // ================= Step 2: host 侧线号区间统计 =================
        // 下载 sorted fids, 构建 line_ids_map + line_begin
        std::vector<int> h_fids_sorted(pointCount);
        cudaMemcpyAsync(h_fids_sorted.data(), d_sorted_fids_.ptr<int>(),
                        pointCount * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        float window = params_.effectiveWindow();
        std::vector<int> h_line_ids_map;         // 去重升序线号
        std::vector<int> h_line_begin;           // 每线号起始（末尾多一格=end）
        {
            int i = 0;
            while (i < pointCount) {
                int fid = h_fids_sorted[i];
                int j = i;
                while (j < pointCount && h_fids_sorted[j] == fid) ++j;
                h_line_ids_map.push_back(fid);
                h_line_begin.push_back(i);
                i = j;
            }
            h_line_begin.push_back(pointCount);
        }
        int num_lines = (int)h_line_ids_map.size();

        // y 范围（全体点） -> 极线行范围
        std::vector<float> h_ys(pointCount);
        {
            // 下载排序后 y
            std::vector<float2> h_pts(pointCount);
            cudaMemcpyAsync(h_pts.data(), d_sorted_pts_.ptr<float2>(),
                            pointCount * sizeof(float2), cudaMemcpyDeviceToHost, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            for (int i = 0; i < pointCount; ++i) h_ys[i] = h_pts[i].y;
        }
        float y_min = *std::min_element(h_ys.begin(), h_ys.end());
        float y_max = *std::max_element(h_ys.begin(), h_ys.end());
        float step = params_.epipolar_row_step;
        int row_begin = (int)std::floor(y_min / step);
        int row_end = (int)std::ceil(y_max / step);
        int num_rows = row_end - row_begin + 1;
        int total_lines = num_lines * num_rows;

        // 上传线表
        cv::cuda::GpuMat d_line_begin(1, num_lines + 1, CV_32SC1);
        cv::cuda::GpuMat d_line_ids_map(1, num_lines, CV_32SC1);
        cudaMemcpyAsync(d_line_begin.ptr<int>(), h_line_begin.data(),
                        (num_lines + 1) * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_line_ids_map.ptr<int>(), h_line_ids_map.data(),
                        num_lines * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);

        // ================= Step 3: 极线插值 kernel =================
        d_flags_.create(1, total_lines, CV_32SC1);
        d_temp_interp_.create(1, total_lines, CV_32FC2);
        d_temp_fids_.create(1, total_lines, CV_32SC1);
        d_output_.create(1, total_lines, CV_32FC2);
        d_output_fids_.create(1, total_lines, CV_32SC1);
        d_output_count_.create(1, 1, CV_32SC1);

        int grid = (total_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernelEpipolarInterp<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(),
            d_line_begin.ptr<int>(), d_line_ids_map.ptr<int>(),
            num_lines, num_rows, row_begin,
            step, window, params_.max_x_diff,
            d_flags_.ptr<int>(), d_temp_interp_.ptr<float2>(),
            d_temp_fids_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        // ================= Step 4: CUB 压缩 =================
        size_t temp_storage_bytes = 0;
        size_t temp_storage_fids = 0;

        cudaError_t cub_err = cub::DeviceSelect::Flagged(
            nullptr, temp_storage_bytes,
            d_temp_interp_.ptr<float2>(), d_flags_.ptr<int>(),
            d_output_.ptr<float2>(), d_output_count_.ptr<int>(),
            total_lines, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged size query failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            nullptr, temp_storage_fids,
            d_temp_fids_.ptr<int>(), d_flags_.ptr<int>(),
            d_output_fids_.ptr<int>(), d_output_count_.ptr<int>(),
            total_lines, cuda_stream);

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
            d_temp_interp_.ptr<float2>(), d_flags_.ptr<int>(),
            d_output_.ptr<float2>(), d_output_count_.ptr<int>(),
            total_lines, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_storage_, cub_temp_size_,
            d_temp_fids_.ptr<int>(), d_flags_.ptr<int>(),
            d_output_fids_.ptr<int>(), d_output_count_.ptr<int>(),
            total_lines, cuda_stream);

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
        cudaMemcpyAsync(&h_count, d_output_count_.ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
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
    d_sort_keys_.release();
    d_sorted_pts_.release();
    d_sorted_fids_.release();
    d_indices_.release();
    warmed_up_ = false;
    warmup_count_ = 0;
    last_max_pairs_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
