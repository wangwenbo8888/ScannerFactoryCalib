/**
 * @file epipolar_interp_dual_cuda_impl.cu
 * @brief 激光中心点极线插值CUDA算子·双模式版实现
 *
 * Labeled 路径：与 epipolar_interp_opt 逐位一致（kernel 复刻, 仅符号加 dual 前缀）。
 * Scan 路径（新增, 无线号）：
 *   Step 1: 按 y 排序（32bit 有序位型键, CUB SortPairs）
 *   Step 2: 逐极线行 kernel（固定 grid=SCAN_MAX_ROWS）:
 *     a) 窗口 [y_k±window] 内点收集到 smem（≤SCAN_ROW_PTS_CAP）
 *     b) 按 x 插入排序 + 间隙聚类（间隙 > scan_x_gap 分簇, 每簇≈一条线穿过该行）
 *     c) 簇内就近原则: 精确命中取最近；否则最近上窗点+最近下窗点配对（|Δx|≤max_x_diff）插值；
 *        单侧缺失兜底: 仅上窗或仅下窗有点时, 取该侧最近点垂直投影 (x 不变, y=y_k)
 *     d) 每簇输出一点, fid=0
 *   Step 3: CUB DeviceSelect::Flagged 压缩
 */

#include "epipolar_interp_dual_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace calib;

CALIB_DEFINE_LOG_TAG(06, EpipolarInterpDualCuda);

static constexpr int BLOCK_SIZE = 256;
static constexpr float EPSILON = 1e-4f;      // 精确命中判定（与原版一致）
static constexpr float DEGEN_EPS = 1e-6f;    // 除零保护

// ============================================================================
// kernels（dual 前缀, 避免与原版/opt 的 .cu 符号冲突）
// ============================================================================
// ── Labeled 路径（与 opt 逐位一致）──

__global__ void dualBuildSortKeys(
    const float2* __restrict__ d_pts,
    const int* __restrict__ d_fids,
    unsigned long long* __restrict__ d_keys,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float y = d_pts[idx].y;
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;
    unsigned long long key =
        ((unsigned long long)(unsigned int)d_fids[idx] << 32) | (unsigned int)yi;
    d_keys[idx] = key;
}

__global__ void dualFillIndices(int* __restrict__ d_idx, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_idx[idx] = idx;
}

__global__ void dualGatherSorted(
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
    if (d_fids_dst) d_fids_dst[i] = d_fids_src[src];
}

__global__ void dualBoundaryFlags(
    const int* __restrict__ d_fids,
    int* __restrict__ d_flags,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_flags[i] = (i == 0 || d_fids[i] != d_fids[i - 1]) ? 1 : 0;
}

__global__ void dualInitMeta(unsigned int* __restrict__ d_meta)
{
    if (threadIdx.x == 0) {
        d_meta[0] = 0u;
        d_meta[1] = 0xffffffffu;
        d_meta[2] = 0u;
    }
}

__global__ void dualScatterLineTable(
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

__global__ void dualYRange(
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

__global__ void __launch_bounds__(256, 4) dualEpipolarInterp(
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

// ── 模式自动判别：line_ids 最小/最大值 ──

__global__ void dualInitIdRange(int* __restrict__ d_meta2)
{
    if (threadIdx.x == 0) {
        d_meta2[0] = INT_MAX;   // min
        d_meta2[1] = INT_MIN;   // max
    }
}

__global__ void dualIdRange(
    const int* __restrict__ d_ids,
    int n,
    int* __restrict__ d_meta2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicMin(&d_meta2[0], d_ids[i]);
    atomicMax(&d_meta2[1], d_ids[i]);
}

// ── Scan 路径（新增）──

// 32bit 有序 y 键 + 顺带 idx 序列
__global__ void dualBuildYKeys(
    const float2* __restrict__ d_pts,
    int n,
    unsigned int* __restrict__ d_keys,
    int* __restrict__ d_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float y = d_pts[i].y;
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;
    d_keys[i] = yi;
    d_idx[i] = i;
}

// 逐极线行扫描插值：收集→x 排序→聚类→簇内就近配对（thread0 串行, 块=32）
__global__ void dualScanInterp(
    const float2* __restrict__ d_pts_sorted,   // y 升序
    int n,
    float step,
    float window,
    float max_x_diff,
    float x_gap,
    int* __restrict__ d_flags,                 // [row*SCAN_SLOTS + slot]
    float2* __restrict__ d_out_pts,
    int* __restrict__ d_out_fids)
{
    const int row = blockIdx.x;
    if (threadIdx.x != 0) return;

    __shared__ float sx[256];
    __shared__ float sy[256];

    const float y_k = row * step;

    // 二分: 第一个 y >= y_k - window
    int lo = 0, hi = n;
    const float ylw = y_k - window;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_pts_sorted[mid].y < ylw) lo = mid + 1;
        else hi = mid;
    }

    int cnt = 0;
    for (int i = lo; i < n; ++i) {
        float y = d_pts_sorted[i].y;
        if (y > y_k + window) break;
        if (cnt >= 256) break;   // CAP
        sx[cnt] = d_pts_sorted[i].x;
        sy[cnt] = y;
        ++cnt;
    }
    if (cnt == 0) return;        // flags 已由 host memset 清零

    // 按 x 插入排序（窗口内点规模 ~10-60）
    for (int i = 1; i < cnt; ++i) {
        float x = sx[i], y = sy[i];
        int j = i - 1;
        while (j >= 0 && sx[j] > x) {
            sx[j + 1] = sx[j];
            sy[j + 1] = sy[j];
            --j;
        }
        sx[j + 1] = x;
        sy[j + 1] = y;
    }

    // 间隙聚类 + 簇内就近插值
    int slot = 0;
    int start = 0;
    for (int i = 1; i <= cnt; ++i) {
        const bool boundary = (i == cnt) || (sx[i] - sx[i - 1] > x_gap);
        if (!boundary) continue;

        int hit = -1;  float hit_dy = 1e30f;
        int up = -1;   float up_dy = 1e30f;
        int dn = -1;   float dn_dy = 1e30f;
        for (int k = start; k < i; ++k) {
            const float dy = sy[k] - y_k;
            const float ady = fabsf(dy);
            if (ady < EPSILON) {
                if (ady < hit_dy) { hit_dy = ady; hit = k; }
            } else if (dy < 0.0f) {
                if (ady < up_dy) { up_dy = ady; up = k; }
            } else {
                if (ady < dn_dy) { dn_dy = ady; dn = k; }
            }
        }

        float ox = 0.f;
        bool ok = false;
        if (hit >= 0) {
            ox = sx[hit];
            ok = true;
        } else if (up >= 0 && dn >= 0) {
            if (fabsf(sx[up] - sx[dn]) <= max_x_diff) {
                const float denom = sy[dn] - sy[up];
                if (fabsf(denom) >= DEGEN_EPS) {
                    const float t = (y_k - sy[up]) / denom;
                    ox = sx[up] + t * (sx[dn] - sx[up]);
                    ok = true;
                }
            }
        } else if (up >= 0) {   // 单侧兜底: 仅上窗有点 → 垂直投影
            ox = sx[up];
            ok = true;
        } else if (dn >= 0) {   // 单侧兜底: 仅下窗有点 → 垂直投影
            ox = sx[dn];
            ok = true;
        }

        if (ok && slot < 32) {
            const int at = row * 32 + slot;
            d_out_pts[at] = make_float2(ox, y_k);
            d_out_fids[at] = at;    // 扫描模式: 簇 ID（row*SCAN_SLOTS+slot 编码）
            d_flags[at] = 1;
            ++slot;
        }
        start = i;
    }
}

// ── Scan 段链接：行间簇合并（每簇向近 gap 行内 |Δx|≤thr 的最近簇建父指针）──
__global__ void dualScanLinkParent(
    const float2* __restrict__ d_cpts,   // 压缩后（行序升序，行内 x 序）
    int n,
    float step,
    float dxThr,
    int gapRows,
    int* __restrict__ d_parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float2 p = d_cpts[i];
    const int row = __float2int_rn(p.y / step);
    int best = -1;
    int bestRowDiff = gapRows + 1;
    float bestDx = dxThr + 1.0f;
    for (int j = i - 1; j >= 0; --j) {
        const float2 q = d_cpts[j];
        const int rj = __float2int_rn(q.y / step);
        const int rd = row - rj;
        if (rd <= 0) continue;          // 同行不链（行内点连续位于前方）
        if (rd > gapRows) break;        // 出窗（行序保证后续更远）
        const float dx = fabsf(p.x - q.x);
        if (dx > dxThr) continue;
        if (rd < bestRowDiff || (rd == bestRowDiff && dx < bestDx)) {
            bestRowDiff = rd;
            bestDx = dx;
            best = j;
        }
    }
    d_parent[i] = best >= 0 ? best : i;
}

// pointer doubling 一轮（森林求根）
__global__ void dualScanLinkJump(
    int* __restrict__ d_label,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_label[i] = d_label[d_label[i]];
}

// ============================================================================
// ScopedFlag (Debug-only thread safety)
// ============================================================================

#ifndef NDEBUG
class DualScopedFlag {
public:
    explicit DualScopedFlag(std::atomic<bool>* flag_) : flag_(flag_) {
        flag_->store(true);
    }
    ~DualScopedFlag() { flag_->store(false); }
    DualScopedFlag(const DualScopedFlag&) = delete;
    DualScopedFlag& operator=(const DualScopedFlag&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// Impl
// ============================================================================

EpipolarInterpDualCuda::Impl::Impl(const EpipolarInterpDualParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[06-EpipolarInterpDualCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[06-EpipolarInterpDualCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }

    if (cudaMallocHost(&h_meta_pinned_, 4 * sizeof(int)) != cudaSuccess) {
        h_meta_pinned_ = nullptr;
        throw std::runtime_error("[06-EpipolarInterpDualCuda] pinned host alloc failed");
    }

    d_meta_.create(1, 4, CV_32SC1);
}

EpipolarInterpDualCuda::Impl::~Impl() {
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

void EpipolarInterpDualCuda::Impl::ensureCapacity(int pointCount) {
    if (pointCount <= capacity_) return;

    d_keys_in_.create(1, pointCount, CV_64FC1);
    d_keys_out_.create(1, pointCount, CV_64FC1);
    d_ykeys_in_.create(1, pointCount, CV_32SC1);
    d_ykeys_out_.create(1, pointCount, CV_32SC1);
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

void EpipolarInterpDualCuda::Impl::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }
    if (warmed_up_ && warmup_count_ == pointCount) return;

    ensureCapacity(pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;
    CALIB_LOG_INFO("Warmup(): ensured GPU capacity for {} points", pointCount);
}

EpipolarInterpResult EpipolarInterpDualCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    DualScopedFlag guard(&inProcess_);
#endif

    EpipolarInterpResult result;

    try {
        int pointCount = d_points.rows * d_points.cols;

        if (pointCount < 2) {
            result.success = true;
            result.message = "Less than 2 points, no interpolation";
            result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
            result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.interpCount = 0;
            return result;
        }

        ensureCapacity(pointCount);
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
        const int gridN = (pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // ================= 模式判别 =================
        bool scanMode;
        if (params_.mode == InterpMode::Labeled) {
            scanMode = false;
        } else if (params_.mode == InterpMode::Scan) {
            scanMode = true;
        } else {
            // Auto: 线号全部相同（含全 0）→ 扫描；存在多线号 → 标定
            dualInitIdRange<<<1, 32, 0, cs>>>(d_meta_.ptr<int>());
            dualIdRange<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_line_ids.ptr<int>(), pointCount, d_meta_.ptr<int>());
            int h_range[2] = {0, 0};
            cudaMemcpyAsync(h_range, d_meta_.ptr<int>(), 2 * sizeof(int),
                            cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            scanMode = (h_range[0] == h_range[1]);
            CALIB_LOG_DEBUG("mode auto-detect: id[min={},max={}] -> {}",
                            h_range[0], h_range[1], scanMode ? "scan" : "labeled");
        }

        // ================= 排序 =================
        cudaError_t kernel_err;
        if (scanMode) {
            // ── Scan: 32bit y 键 ──
            dualBuildYKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), pointCount,
                reinterpret_cast<unsigned int*>(d_ykeys_in_.ptr<int>()),
                d_idx_in_.ptr<int>());

            if (radix_cap_ < pointCount || d_radix_temp_ == nullptr) {
                size_t bytes = 0;
                cub::DeviceRadixSort::SortPairs(
                    nullptr, bytes,
                    (const unsigned int*)nullptr, (unsigned int*)nullptr,
                    (const int*)nullptr, (int*)nullptr,
                    pointCount, 0, sizeof(unsigned int) * 8, cs);
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
                    reinterpret_cast<const unsigned int*>(d_ykeys_in_.ptr<int>()),
                    reinterpret_cast<unsigned int*>(d_ykeys_out_.ptr<int>()),
                    d_idx_in_.ptr<int>(), d_idx_out_.ptr<int>(),
                    pointCount, 0, sizeof(unsigned int) * 8, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("radix sort failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
            }
            dualGatherSorted<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), nullptr,
                d_idx_out_.ptr<int>(),
                d_sorted_pts_.ptr<float2>(), nullptr, pointCount);
        } else {
            // ── Labeled: 64bit (fid,y) 键（与 opt 一致）──
            dualBuildSortKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), d_line_ids.ptr<int>(),
                reinterpret_cast<unsigned long long*>(d_keys_in_.ptr<double>()), pointCount);
            dualFillIndices<<<gridN, BLOCK_SIZE, 0, cs>>>(d_idx_in_.ptr<int>(), pointCount);

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
                    reinterpret_cast<const unsigned long long*>(d_keys_in_.ptr<double>()),
                    reinterpret_cast<unsigned long long*>(d_keys_out_.ptr<double>()),
                    d_idx_in_.ptr<int>(), d_idx_out_.ptr<int>(),
                    pointCount, 0, sizeof(unsigned long long) * 8, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("radix sort failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
            }
            dualGatherSorted<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), d_line_ids.ptr<int>(),
                d_idx_out_.ptr<int>(),
                d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(), pointCount);
        }

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Sort kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        if (scanMode) {
            // ================= Scan: 逐行聚类插值 + 压缩 =================
            const int totalSlots = SCAN_MAX_ROWS * SCAN_SLOTS;
            d_scan_flags_.create(1, totalSlots, CV_32SC1);
            d_scan_out_pts_.create(1, totalSlots, CV_32FC2);
            d_scan_out_fids_.create(1, totalSlots, CV_32SC1);
            d_scan_cpts_.create(1, totalSlots, CV_32FC2);
            d_scan_cfids_.create(1, totalSlots, CV_32SC1);
            d_scan_ccount_.create(1, 1, CV_32SC1);

            cudaMemsetAsync(d_scan_flags_.ptr<int>(), 0, totalSlots * sizeof(int), cs);

            dualScanInterp<<<SCAN_MAX_ROWS, 32, 0, cs>>>(
                d_sorted_pts_.ptr<float2>(), pointCount,
                params_.epipolar_row_step, params_.effectiveWindow(),
                params_.max_x_diff, params_.scan_x_gap,
                d_scan_flags_.ptr<int>(),
                d_scan_out_pts_.ptr<float2>(), d_scan_out_fids_.ptr<int>());

            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("dualScanInterp failed: ") + cudaGetErrorString(kernel_err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }

            if (cub_cap_ < totalSlots || d_cub_temp_ == nullptr) {
                size_t t1 = 0, t2 = 0;
                cub::DeviceSelect::Flagged(nullptr, t1,
                    (const float2*)nullptr, (const int*)nullptr,
                    (float2*)nullptr, (int*)nullptr, totalSlots, cs);
                cub::DeviceSelect::Flagged(nullptr, t2,
                    (const int*)nullptr, (const int*)nullptr,
                    (int*)nullptr, (int*)nullptr, totalSlots, cs);
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
                cub_cap_ = totalSlots;
            }
            {
                cudaError_t err = cub::DeviceSelect::Flagged(
                    d_cub_temp_, cub_temp_size_,
                    (const float2*)d_scan_out_pts_.ptr<float2>(),
                    (const int*)d_scan_flags_.ptr<int>(),
                    d_scan_cpts_.ptr<float2>(), d_scan_ccount_.ptr<int>(), totalSlots, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                err = cub::DeviceSelect::Flagged(
                    d_cub_temp_, cub_temp_size_,
                    (const int*)d_scan_out_fids_.ptr<int>(),
                    (const int*)d_scan_flags_.ptr<int>(),
                    d_scan_cfids_.ptr<int>(), d_scan_ccount_.ptr<int>(), totalSlots, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
            }

            int h_count = 0;
            cudaMemcpyAsync(&h_count, d_scan_ccount_.ptr<int>(), sizeof(int),
                            cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);

            result.success = true;
            result.message = "Success (scan mode)";
            result.interpCount = h_count;
            result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>(d_scan_cpts_.colRange(0, h_count).clone());

            // 段链接：行间簇合并 → 段 ID 写入 d_interp_line_ids
            // （禁用时输出簇 ID —— 同段行间不重号，仅断口不合并）
            bool linked = false;
            if (params_.scan_link_dx > 0.0f && h_count > 1) {
                d_link_parent_.create(1, h_count, CV_32SC1);
                d_link_label_.create(1, h_count, CV_32SC1);
                const int gridL = (h_count + 255) / 256;
                // [诊断计时] 段链接（连通计算）相位 GPU 耗时 → result.scan_link_ms；
                // RAII 保证错误提前返回时事件不泄漏
                struct EvPair {
                    cudaEvent_t a, b;
                    EvPair() { cudaEventCreate(&a); cudaEventCreate(&b); }
                    ~EvPair() { cudaEventDestroy(a); cudaEventDestroy(b); }
                } ev;
                cudaEventRecord(ev.a, cs);
                dualScanLinkParent<<<gridL, 256, 0, cs>>>(
                    d_scan_cpts_.ptr<float2>(), h_count,
                    params_.epipolar_row_step, params_.scan_link_dx,
                    params_.scan_link_gap_rows, d_link_parent_.ptr<int>());
                cudaError_t link_err = cudaGetLastError();
                if (link_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("dualScanLinkParent failed: ") + cudaGetErrorString(link_err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                cudaMemcpyAsync(d_link_label_.ptr<int>(), d_link_parent_.ptr<int>(),
                                h_count * sizeof(int), cudaMemcpyDeviceToDevice, cs);
                int rounds = 1;
                while ((1 << rounds) < h_count) ++rounds;
                for (int r = 0; r < rounds; ++r)
                    dualScanLinkJump<<<gridL, 256, 0, cs>>>(d_link_label_.ptr<int>(), h_count);
                link_err = cudaGetLastError();
                if (link_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("dualScanLinkJump failed: ") + cudaGetErrorString(link_err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                cudaEventRecord(ev.b, cs);
                cudaEventSynchronize(ev.b);
                float link_ms = 0.f;
                cudaEventElapsedTime(&link_ms, ev.a, ev.b);
                result.scan_link_ms = static_cast<double>(link_ms);
                linked = true;
            }
            if (linked)
                result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(
                    d_link_label_.colRange(0, h_count).clone());
            else
                result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(
                    d_scan_cfids_.colRange(0, h_count).clone());
            return result;
        }

        // ================= Labeled: 线号表 + y 范围（与 opt 一致） =================
        dualBoundaryFlags<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_fids_.ptr<int>(), d_rl_flags_.ptr<int>(), pointCount);

        if (scan_cap_ < pointCount || d_scan_temp_ == nullptr) {
            size_t bytes = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr, bytes, (const int*)nullptr, (int*)nullptr, pointCount, cs);
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
                (const int*)d_rl_flags_.ptr<int>(), d_rl_ranks_.ptr<int>(), pointCount, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("exclusive sum failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
        }

        dualInitMeta<<<1, 32, 0, cs>>>(reinterpret_cast<unsigned int*>(d_meta_.ptr<int>()));
        dualScatterLineTable<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_fids_.ptr<int>(), d_rl_flags_.ptr<int>(),
            d_rl_ranks_.ptr<int>(), pointCount,
            d_line_ids_map_.ptr<int>(), d_line_begin_.ptr<int>(),
            reinterpret_cast<unsigned int*>(d_meta_.ptr<int>()));
        dualYRange<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_pts_.ptr<float2>(), pointCount,
            reinterpret_cast<unsigned int*>(d_meta_.ptr<int>()));

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Line table kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cudaMemcpyAsync(h_meta_pinned_, d_meta_.ptr<int>(), 3 * sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);

        const int num_lines = h_meta_pinned_[0];
        float y_min = 0.f, y_max = 0.f;
        {
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
            dualEpipolarInterp<<<gridL, BLOCK_SIZE, 0, cs>>>(
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
        result.message = "Success (labeled mode)";
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

void EpipolarInterpDualCuda::Impl::SetParams(const EpipolarInterpDualParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("SetParams(): called while Execute() is running");
        throw std::runtime_error("[06-EpipolarInterpDualCuda] SetParams() called during Execute()");
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

void EpipolarInterpDualCuda::Impl::Destroy() {
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
    d_ykeys_in_.release();
    d_ykeys_out_.release();
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
    d_scan_flags_.release();
    d_scan_out_pts_.release();
    d_scan_out_fids_.release();
    d_scan_cpts_.release();
    d_scan_cfids_.release();
    d_scan_ccount_.release();
    capacity_ = 0;
    lines_cap_ = 0;
    warmed_up_ = false;
    warmup_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
