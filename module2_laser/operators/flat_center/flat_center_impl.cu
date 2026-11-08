/**
 * @file flat_center_impl.cu
 * @brief 平顶激光中心提取算子（flat_center）CUDA 实现 — Task 2 桩版
 *
 * ⚠️ 桩版：Impl 全方法最小定义（解链接）；Task 3 替换 Execute/extractFlat 真体
 * （Flat 路径）＋Task 5 填 ByLabel。设计 §3/§4。
 */

#include "flat_center_pimpl.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>

using namespace calib;

FlatCenterExtractor::Impl::Impl(const FlatCenterParams& params)
    : params_(params)
{
    params_.validate();
    bench_ = (getenv("JMW_FLATTIME") != nullptr);
    if (bench_) createEvents();
}

FlatCenterExtractor::Impl::~Impl() {
    freeAll();
}

void FlatCenterExtractor::Impl::createEvents() {
    if (events_ready_) return;
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_scan_);
    cudaEventCreate(&ev_end_);
    events_ready_ = true;
}

void FlatCenterExtractor::Impl::freeAll() {
    if (d_coords_)  { cudaFree(d_coords_); d_coords_ = nullptr; }
    if (d_ids_)     { cudaFree(d_ids_); d_ids_ = nullptr; }
    if (d_stats_)   { cudaFree(d_stats_); d_stats_ = nullptr; }
    if (d_hist_)    { cudaFree(d_hist_); d_hist_ = nullptr; }
    if (d_base_)    { cudaFree(d_base_); d_base_ = nullptr; }
    if (d_cursor_)  { cudaFree(d_cursor_); d_cursor_ = nullptr; }
    if (d_gcoords_) { cudaFree(d_gcoords_); d_gcoords_ = nullptr; }
    if (d_gids_)    { cudaFree(d_gids_); d_gids_ = nullptr; }
    if (d_gsmooth_) { cudaFree(d_gsmooth_); d_gsmooth_ = nullptr; }   // 【flat 优化 C】
    if (flatGraphExec_) { cudaGraphExecDestroy(flatGraphExec_); flatGraphExec_ = nullptr; }   // 【flat 优化·宏打包】
    if (flatGraph_)     { cudaGraphDestroy(flatGraph_); flatGraph_ = nullptr; }
    if (h_stats_)   { cudaFreeHost(h_stats_); h_stats_ = nullptr; }
    if (h_coords_)  { cudaFreeHost(h_coords_); h_coords_ = nullptr; }
    if (h_ids_)     { cudaFreeHost(h_ids_); h_ids_ = nullptr; }
    // 【P2-2 参数块】随容量缓冲同毁（freeAll⟺毁图绑定不变）
    if (d_params_)  { cudaFree(d_params_); d_params_ = nullptr; }
    if (h_params_)  { cudaFreeHost(h_params_); h_params_ = nullptr; }
    if (events_ready_) {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_scan_);
        cudaEventDestroy(ev_end_);
        events_ready_ = false;
    }
    capacity_ = 0;
    ++paramsEpoch_;   // 【G′·C4′】freeAll——缓冲/参数块全释，指针失效
}

void FlatCenterExtractor::Impl::ensureBuffers(int rows, int cols) {
    const int cap = rows * cols;
    if (cap <= capacity_) return;
    freeAll();
    capacity_ = cap;
    ++paramsEpoch_;   // 【G′·C4′】容量扩容——输出/暂存指针全体重建（freeAll 已 +1，
                      // 此处再 +1 显式标记「新容量就位」代际；重复递增无害）
    cudaMalloc(&d_coords_, capacity_ * sizeof(float2));
    cudaMalloc(&d_ids_, capacity_ * sizeof(int));
    cudaMalloc(&d_stats_, 7 * sizeof(int));
    cudaMalloc(&d_hist_, kLabelBins * sizeof(int));
    cudaMalloc(&d_base_, kLabelBins * sizeof(int));
    cudaMalloc(&d_cursor_, kLabelBins * sizeof(int));
    cudaMalloc(&d_gcoords_, capacity_ * sizeof(float2));
    cudaMalloc(&d_gids_, capacity_ * sizeof(int));
    cudaMalloc(&d_gsmooth_, (size_t)capacity_ * sizeof(float));   // 【flat 优化 C】预平滑
    cudaMallocHost(&h_stats_, 7 * sizeof(int));
    cudaMallocHost(&h_coords_, capacity_ * sizeof(float2));
    cudaMallocHost(&h_ids_, capacity_ * sizeof(int));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        CALIB_LOG_ERROR("ensureBuffers failed: {}", cudaGetErrorString(err));
        throw std::runtime_error(std::string("[10-FlatCenterExtractor] buffer alloc failed: ")
                                 + cudaGetErrorString(err));
    }
}

// ============================================================================
// CUDA Kernel（设计 §4：单 kernel、1 线程/列、逐段两次行走＋括号重心第三小遍）
// ============================================================================

// —— 平滑灰度读（段端读段外原始灰度＋图像边界钳位；统计遍/找沿遍/括号重心同款）——
__device__ __forceinline__ float ReadGS(const unsigned char* g, size_t step,
                                        int rows, int x, int y, bool s3) {
    const unsigned char* col = g + x;
    if (!s3) return static_cast<float>(col[(size_t)y * step]);
    const int y0 = max(y - 1, 0), y2 = min(y + 1, rows - 1);
    return (col[(size_t)y0 * step] + col[(size_t)y * step] + col[(size_t)y2 * step])
           * (1.f / 3.f);
}

// 【flat 优化 C+D】全图预平滑（逐位同 ReadGS(smooth3)：三抽头＋边界钳位，
// (a+b+c)*(1.f/3.f) 同式逐位一致）＋(0,0,0) 线程顺带清 d_stats_ 7 槽（吸收
// 独立 memsetAsync——本内核无人碰 stats，K1 流序在后见零初值）
// 【P2-2】体提公共函数：值参内核（逐发路径）与参数块内核（图路径）同源同体。
__device__ __forceinline__ void flatPreSmoothBody(
    const unsigned char* __restrict__ g, size_t step, int rows, int cols,
    float* __restrict__ out, int* __restrict__ stats7)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 7; ++i) stats7[i] = 0;
    }
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    const unsigned char* col = g + x;
    const int y0 = max(y - 1, 0), y2 = min(y + 1, rows - 1);
    out[(size_t)y * cols + x] =
        (col[(size_t)y0 * step] + col[(size_t)y * step] + col[(size_t)y2 * step])
        * (1.f / 3.f);
}

__global__ void FlatPreSmoothKernel(const unsigned char* __restrict__ g,
                                    size_t step, int rows, int cols,
                                    float* __restrict__ out,
                                    int* __restrict__ stats7) {
    flatPreSmoothBody(g, step, rows, cols, out, stats7);
}

// 【P2-2 参数块壳】图路径用：全部实参经 d_params_ 现读（memcpy 节点刷新）
__global__ void FlatPreSmoothKernelBlock(const FlatParamsDev* __restrict__ p) {
    flatPreSmoothBody(p->g, p->gstep, p->rows, p->cols, p->gs, p->stats);
}

// 【P2-2】K1 上下文：值参壳（逐发路径，签名不变）与参数块壳（图路径）共用
// 同一 flatCenterK1Body——单源保逐位一致。
struct FlatK1Ctx {
    const unsigned char* d_gray; const void* d_mask;
    size_t gray_step, mask_step; int rows, cols;
    float low_thresh, half_ratio, plateau_ratio; int plateau_min_w;
    float min_contrast, outlier_gate, max_line_width; int min_run; bool smooth3;
    const float* d_gs;
    float2* d_coords; int* d_ids;
    int* d_stats; int max_points;
};

// ============================================================================
// 261107 行模式（scanAxis=1）——竖直激光带质心提取
// 1 线程/行；段沿 x（连续同 label 掩膜段）；段内去背景亮度加权质心＝带中心。
// 门：minRun / bright(vmax) / contrast / 段宽 max_line_width（FWHM 口径简化
// 为段宽，0=off）。输出点 (cx, y)——x 向亚像素质心、y 为整数行（行栅格化
// 由下游 interp 承担）。列模式（原路径）不经过本函数——逐位契约不动。
// ============================================================================
template <bool MASK8>
__device__ __forceinline__ void flatCenterK1BodyRow(const FlatK1Ctx& c, int merge_gap)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= c.rows) return;
    const int bands = gridDim.y;
    const int bw = (c.cols + bands - 1) / bands;
    const int left = blockIdx.y * bw;
    if (left >= c.cols) return;
    const int right = min(c.cols, left + bw);
    int* d_count = c.d_stats;
    int* d_drop  = c.d_stats + 1;   // [0]minRun [1]contrast [2]bright [3]edge [4]gate [5]width
    auto maskAt = [&](int xx) -> int {
        if (MASK8)
            return reinterpret_cast<const unsigned char*>(
                reinterpret_cast<const char*>(c.d_mask) + (size_t)y * c.mask_step)[xx] > 0 ? 1 : 0;
        return reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(c.d_mask) + (size_t)y * c.mask_step)[xx];
    };
    auto RD = [&](int xx) -> float {
        return reinterpret_cast<const unsigned char*>(
            reinterpret_cast<const char*>(c.d_gray) + (size_t)y * c.gray_step)[xx];
    };
    int x = left;
    while (x < right) {
        const int label = maskAt(x);
        if (label <= 0) { ++x; continue; }
        if (x == left && left > 0 && maskAt(left - 1) == label) {
            while (x < c.cols && maskAt(x) == label) ++x;
            continue;
        }
        const int x0 = x;
        // 261107 段合并：gap≤merge_gap 的纯背景断口桥接（带内暗纹在 mask 上
        // 为 0——桥接后段完整、质心=带中心）；异 label（邻近带）禁止桥接
        int lastLabelX = x;
        while (true) {
            while (x < c.cols && maskAt(x) == label) { lastLabelX = x; ++x; }
            if (merge_gap <= 0 || x >= c.cols) break;
            int g = 0;
            while (g < merge_gap && x + g < c.cols && maskAt(x + g) == 0) ++g;
            if (g < merge_gap && x + g < c.cols && maskAt(x + g) == label) {
                x += g;   // 背景断口 ≤gap 且末端回同 label——桥接继续
                continue;
            }
            break;
        }
        const int x1 = lastLabelX + 1, len = x1 - x0;
        if (len < c.min_run) { atomicAdd(&d_drop[0], 1); continue; }
        // 统计：vmax/vmin（质心去背景基准）
        float vmax = -1.f, vmin = 1e9f;
        for (int k = x0; k < x1; ++k) {
            const float v = RD(k);
            if (v > vmax) vmax = v;
            if (v < vmin) vmin = v;
        }
        if (vmax < c.low_thresh) { atomicAdd(&d_drop[2], 1); continue; }
        if (vmax - vmin < c.min_contrast) { atomicAdd(&d_drop[1], 1); continue; }
        if (c.max_line_width > 0.f && static_cast<float>(len) > c.max_line_width) {
            atomicAdd(&d_drop[5], 1); continue;
        }
        // 去背景亮度加权质心（vmin≈段端衬底＝天然背景扣除）
        float sw = 0.f, sx = 0.f;
        for (int k = x0; k < x1; ++k) {
            const float w = RD(k) - vmin;
            if (w > 0.f) { sw += w; sx += k * w; }
        }
        if (sw < 1e-6f) { atomicAdd(&d_drop[1], 1); continue; }
        const float cx = sx / sw;
        if (!(cx == cx) || cx < -1.f || cx > c.cols) {
            atomicAdd(&d_drop[4], 1); continue;
        }
        const int idx = atomicAdd(d_count, 1);
        if (idx < c.max_points) {
            c.d_coords[idx] = make_float2(cx, static_cast<float>(y));
            c.d_ids[idx] = label;
        }
    }
}

template <bool MASK8>
__device__ __forceinline__ void flatCenterK1Body(const FlatK1Ctx& c)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= c.cols) return;
    // 【速度·列带切分（设计 §4 备选，规则钉死）】1 线程/列在 1400 行图像上仅
    // ~106 warp 全核延迟饥饿（实测 7.8ms/眼）；gridDim.y＝带数（8）：
    //   ① 段归段首所在带；② owner 向下走到段尾（越带无上界）；
    //   ③ 带首行若为上方延续段（maskAt(top−1)==label）跳到段尾不处理
    const int bands = gridDim.y;
    const int bh = (c.rows + bands - 1) / bands;
    const int top = blockIdx.y * bh;
    if (top >= c.rows) return;
    const int bot = min(c.rows, top + bh);
    int* d_count = c.d_stats;
    int* d_drop  = c.d_stats + 1;   // [0]minRun [1]contrast [2]bright [3]edge [4]gate [5]width
    // 列内段扫描（Flat：连续 mask>0；ByLabel：连续同 label，label 变化即断段）
    auto maskAt = [&](int y) -> int {
        if (MASK8)
            return reinterpret_cast<const unsigned char*>(
                reinterpret_cast<const char*>(c.d_mask) + (size_t)y * c.mask_step)[x] > 0 ? 1 : 0;
        return reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(c.d_mask) + (size_t)y * c.mask_step)[x];
    };
    // 【flat 优化 C】统一读口：预平滑缓冲优先（逐位同 ReadGS）；空＝原路径现算
    auto RD = [&](int yy) -> float {
        if (c.d_gs) return c.d_gs[(size_t)yy * c.cols + x];
        return ReadGS(c.d_gray, c.gray_step, c.rows, x, yy, c.smooth3);
    };
    int y = top;
    while (y < bot) {
        const int label = maskAt(y);
        if (label <= 0) { ++y; continue; }
        if (y == top && top > 0 && maskAt(top - 1) == label) {
            // 规则③：上方延续段（段首在上面的带）——跳到段尾，由 owner 处理
            while (y < c.rows && maskAt(y) == label) ++y;
            continue;
        }
        const int y0 = y;
        while (y < c.rows && maskAt(y) == label) ++y;          // 段=[y0, y)，段尾可越带
        const int y1 = y, len = y1 - y0;
        // —— 门1：minRun ——
        if (len < c.min_run) { atomicAdd(&d_drop[0], 1); continue; }
        // —— 统计遍第一小趟：vmax/yPeak（自上而下严格 > 首个）/vmin ——
        float vmax = -1.f, vmin = 1e9f; int yPeak = y0;
        for (int k = y0; k < y1; ++k) {
            const float v = RD(k);
            if (v > vmax) { vmax = v; yPeak = k; }
            if (v < vmin) vmin = v;
        }
        // —— 门2/3：bright / contrast（后者兼保 Σw>0 防除零）——
        if (vmax < c.low_thresh) { atomicAdd(&d_drop[2], 1); continue; }
        if (vmax - vmin < c.min_contrast) { atomicAdd(&d_drop[1], 1); continue; }
        // —— 统计遍第二小趟：pw 对最终 vmax ——
        const float thr_p = c.plateau_ratio * vmax;
        int pw = 0;
        for (int k = y0; k < y1; ++k)
            if (RD(k) >= thr_p) ++pw;
        float cy;
        float yL = 0.f, yR = 0.f;
        if (pw >= c.plateau_min_w) {
            // —— 平顶：半高中点。沿搜索界＝图像界 ∩ [yPeak−32, yPeak+33)（不钳段界：
            //    掩膜细缝/段端处越段读原始灰度找沿；±32 窗保证不跨入相邻线）——
            const int lo = max(0, yPeak - 32), hi = min(c.rows, yPeak + 33);
            // 【链级对拍修正 2026-10-19】半高＝相对背景（vmin＋halfRatio×对比度）——
            // 真实帧背景 ≈110-120（非 0），绝对半高 0.5×vmax 只比背景高 ~12 灰度级，
            // 线稍暗/背景稍亮即找不到沿（edge 弃 30%）；背景基准后合成图（bg=0）
            // 数值不变，现有用例全部兼容。
            const float half = vmin + c.half_ratio * (vmax - vmin);
            int k = yPeak;
            while (k >= lo && RD(k) >= half) --k;
            if (k < lo) { atomicAdd(&d_drop[3], 1); continue; }                  // edge
            const float g1 = RD(k + 1);
            const float g0 = RD(k);
            yL = (k + 1) - (g1 - half) / (g1 - g0);  // crossing ∈ (k, k+1]
            int j = yPeak;
            while (j < hi && RD(j) >= half) ++j;
            if (j >= hi) { atomicAdd(&d_drop[3], 1); continue; }
            const float h1 = RD(j - 1);
            const float h0 = RD(j);
            yR = (j - 1) + (h1 - half) / (h1 - h0); // crossing ∈ [j-1, j)
            // —— 宽度门（v4.4）：半高全宽超限 → 弃点（0=off 不拦）——
            if (c.max_line_width > 0.f && (yR - yL) > c.max_line_width) {
                atomicAdd(&d_drop[5], 1); continue;
            }
            cy = 0.5f * (yL + yR);
        } else {
            // —— 尖峰：去背景重心（vmin≈段内背景＝天然背景扣除，勿"修"）——
            float sw = 0.f, sy = 0.f;
            for (int k = y0; k < y1; ++k) {
                const float w = RD(k) - vmin;
                if (w > 0.f) { sw += w; sy += k * w; }
            }
            if (sw < 1e-6f) { atomicAdd(&d_drop[1], 1); continue; }
            cy = sy / sw;
        }
        // —— 统一自检（v4.3）：平顶＝**括号区间**重心（越缝括号覆盖全线→重心=线心，
        //    细缝不误杀；段内全剖面重心因半剖面偏移会误杀 ⑦）；尖峰＝改比锚点
        //    |yPeak−cy|（cy=段内重心，与重心比恒 0；兼捕侧翼亮斑拉锚构型）——
        float ref;
        if (pw >= c.plateau_min_w) {
            float sw = 0.f, sy = 0.f;
            const int b0 = max(0, (int)yL), b1 = min(c.rows - 1, (int)yR + 1);
            for (int k = b0; k <= b1; ++k) {
                const float w = RD(k) - vmin;
                if (w > 0.f) { sw += w; sy += k * w; }
            }
            ref = (sw >= 1e-6f) ? sy / sw : cy;
        } else {
            ref = static_cast<float>(yPeak);
        }
        if (fabsf(cy - ref) > c.outlier_gate) { atomicAdd(&d_drop[4], 1); continue; } // gate
        // 防御（v4.3）：cy=两沿中点必在括号内，段界防御废除；留图像界＋NaN 防护
        if (!(cy == cy) || cy < -1.f || cy > c.rows) { atomicAdd(&d_drop[4], 1); continue; }
        // —— 出点 (x, cy)（整数格点口径，同 steger kernel px=x+t·nx）——
        const int idx = atomicAdd(d_count, 1);
        if (idx < c.max_points) {
            c.d_coords[idx] = make_float2(static_cast<float>(x), cy);
            c.d_ids[idx] = label;
        }
    }
}

template <bool MASK8>
__global__ void FlatCenterK1Row(
    const unsigned char* __restrict__ d_gray, const void* __restrict__ d_mask,
    size_t gray_step, size_t mask_step, int rows, int cols,
    int min_run, float low_thresh, float min_contrast, float max_line_width,
    int merge_gap,
    float2* __restrict__ d_coords, int* __restrict__ d_ids,
    int* __restrict__ d_stats, int max_points)
{
    FlatK1Ctx c{d_gray, d_mask, gray_step, mask_step, rows, cols,
                low_thresh, 0.f, 0.f, 0, min_contrast, 0.f, max_line_width,
                min_run, false, nullptr, d_coords, d_ids, d_stats, max_points};
    flatCenterK1BodyRow<MASK8>(c, merge_gap);
}

template <bool MASK8>
__global__ void FlatCenterK1(
    const unsigned char* __restrict__ d_gray, const void* __restrict__ d_mask,
    size_t gray_step, size_t mask_step, int rows, int cols,
    float low_thresh, float half_ratio, float plateau_ratio, int plateau_min_w,
    float min_contrast, float outlier_gate, float max_line_width, int min_run,
    bool smooth3,
    const float* __restrict__ d_gs,   /* 【flat 优化 C】预平滑缓冲（空＝现算） */
    float2* __restrict__ d_coords, int* __restrict__ d_ids,
    int* __restrict__ d_stats /*[0]=count [1..6]=六门 drop*/, int max_points)
{
    FlatK1Ctx c{d_gray, d_mask, gray_step, mask_step, rows, cols,
                low_thresh, half_ratio, plateau_ratio, plateau_min_w,
                min_contrast, outlier_gate, max_line_width, min_run, smooth3,
                d_gs, d_coords, d_ids, d_stats, max_points};
    flatCenterK1Body<MASK8>(c);
}

// 【P2-2 参数块壳】图路径用：全部实参经 d_params_ 现读（图内 memcpy 节点
// 每次 launch 从 pinned h_params_ 刷新——lane 轮转输入指针随之自愈）
template <bool MASK8>
__global__ void FlatCenterK1Block(const FlatParamsDev* __restrict__ p) {
    FlatK1Ctx c;
    c.d_gray = p->g;   c.d_mask = p->mask;
    c.gray_step = p->gstep; c.mask_step = p->mstep;
    c.rows = p->rows;  c.cols = p->cols;
    c.low_thresh = p->low_thresh; c.half_ratio = p->half_ratio;
    c.plateau_ratio = p->plateau_ratio; c.plateau_min_w = p->plateau_min_w;
    c.min_contrast = p->min_contrast; c.outlier_gate = p->outlier_gate;
    c.max_line_width = p->max_line_width; c.min_run = p->min_run;
    c.smooth3 = p->smooth3 != 0;
    c.d_gs = p->gs; c.d_coords = p->coords; c.d_ids = p->ids;
    c.d_stats = p->stats; c.max_points = p->maxpts;
    flatCenterK1Body<MASK8>(c);
}

// ============================================================================
// Flat 快路（设计 §4 extractFlat 主流程）
// ============================================================================

// 【合并唤醒】提交段：不同步（stats D2H 已入队）——extractFlat＝Submit＋
// 同步＋finishFlat；流水线双眼可两次 Submit 共用一次流同步
StegerResult FlatCenterExtractor::Impl::extractFlatSubmit(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    StegerResult result;
    const int rows = d_grayImage.rows, cols = d_grayImage.cols;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    ensureBuffers(rows, cols);
    if (bench_ && !events_ready_) createEvents();
    if (bench_ && events_ready_) cudaEventRecord(ev_start_, cs);

    // 【flat 优化 C+D】smooth3 开启时：全图预平滑一次（段内 3-4 趟重读免 3 抽头
    // 放大；逐位同式）＋顺带清 stats；JMW_FLATPRE=0 关闭走原路径（A/B）。
    // 【flat 优化·宏打包】usePre 且未禁用时 [PreSmooth+K1] 免改参数图一次发射
    //（全参数跨帧稳定；首次捕获前排空本流——WDDM 规则）
    const bool usePre = params_.smooth3
                        && params_.scanAxis != 1   // 261107 行模式：RD 直读原灰度（竖带质心不需要预平滑——列模式契约不动）
                        && !(getenv("JMW_FLATPRE") && getenv("JMW_FLATPRE")[0] == '0');
    // 【flat 优化·宏打包】【批D·c2 默认开】flat 图翻转放行——历史「多收尾 lane
    // 必崩 0xC0000005 ×3/3」在现架构（P2-4 link 入图/B5 事件合一/C RLE 图后）
    // 崩溃压力 20/20＋对照 10/10 零崩不复现（2026-10-28 批 D 收尾裁决）。
    // JMW_FLATGRAPH=0 显式关；JMW_NOGRAPH=1 总闸优先。
    static const bool graphOn_ = [] {
        const char* ng = getenv("JMW_NOGRAPH");
        if (ng != nullptr && ng[0] == '1') return false;
        const char* e = getenv("JMW_FLATGRAPH");
        return !(e != nullptr && e[0] == '0');
    }();
    // 【批D·c2】legacy 默认流（cs==0，单测/无显式流调用方）禁图——捕获在 NULL
    // 流上非法（"operation not permitted when stream is capturing" 家族），历史
    // 默认关掩护了此路径；fivechain/产线均真流不受影响
    const bool useGraph = usePre && graphOn_ && cs != nullptr;
    if (usePre) {
        const dim3 pb(128, 8);
        const dim3 pg(static_cast<unsigned>((cols + 127) / 128),
                      static_cast<unsigned>((rows + 7) / 8));
        int nbandsG = 96;   // 【BANDS 定界】nsys 30 帧实测：24→808/48→698/64→655/96→545/128→570µs
                            // （avg/launch，96 最优；上限 128）
        if (const char* e = getenv("JMW_FLATBANDS"))
            nbandsG = std::max(1, std::min(128, atoi(e)));
        const dim3 block(256),
            grid((cols + 255) / 256, static_cast<unsigned>(nbandsG));
        if (useGraph) {
        // 【P2-2 参数块】lazy 分配：仅图模式首用（默认关臂零开销——census 逐位
        // 不变）；失败→本实例图模式永久回退（paramsFailed_）
        if (!paramsFailed_ && !d_params_) {
            cudaMalloc(&d_params_, sizeof(FlatParamsDev));
            cudaMallocHost(&h_params_, sizeof(FlatParamsDev));
            if (cudaGetLastError() != cudaSuccess || !d_params_ || !h_params_) {
                CALIB_LOG_WARN("[flat] params block alloc failed; graph mode disabled");
                if (d_params_) { cudaFree(d_params_); d_params_ = nullptr; }
                if (h_params_) { cudaFreeHost(h_params_); h_params_ = nullptr; }
                paramsFailed_ = true;
            }
        }
        cudaError_t gerr = cudaSuccess;
        if (!paramsFailed_ && d_params_ && h_params_) {
        // 【P2-2 参数块】每帧 launch 前 host 写 pinned 镜像（图内 memcpy 节点每次
        // 执行现读 src——lane 轮转的灰度/掩膜指针随之自愈，烧死缺陷根治）。
        // 单缓冲安全前提：同实例相邻 graphLaunch 之间必有流排水（fivechain 唯一
        // 排水点 gS2 waitForCompletion 每帧一次；双眼为独立实例各持私有块）——
        // 在飞图不会再读 h_params_。合并图若需每帧动态参数再升级双缓冲。
        h_params_->g = d_grayImage.ptr<unsigned char>();
        h_params_->gstep = d_grayImage.step;
        h_params_->mask = d_binaryMask.ptr<unsigned char>();
        h_params_->mstep = d_binaryMask.step;
        h_params_->gs = d_gsmooth_;
        h_params_->coords = d_coords_;
        h_params_->ids = d_ids_;
        h_params_->stats = d_stats_;
        h_params_->rows = rows;
        h_params_->cols = cols;
        h_params_->maxpts = FlatCenterExtractor::kOutCap;
        h_params_->low_thresh = params_.lowThreshold;
        h_params_->half_ratio = params_.halfRatio;
        h_params_->plateau_ratio = params_.plateauRatio;
        h_params_->plateau_min_w = params_.plateauMinWidth;
        h_params_->min_contrast = params_.minContrast;
        h_params_->outlier_gate = params_.outlierGatePx;
        h_params_->max_line_width = params_.maxLineWidth;
        h_params_->min_run = params_.minRunLength;
        h_params_->smooth3 = params_.smooth3 ? 1 : 0;
        // 【P2-2 维度守卫】rows/cols/bands 烧进 gridDim（图结构），参数块救不了：
        // 任一变化→毁图重捕（此前完全无检测＝静默错数据洞）
        if (flatGraphExec_ && (graph_rows_ != rows || graph_cols_ != cols
                               || graph_bands_ != nbandsG)) {
            cudaGraphExecDestroy(flatGraphExec_);
            flatGraphExec_ = nullptr;
            if (flatGraph_) { cudaGraphDestroy(flatGraph_); flatGraph_ = nullptr; }
        }
        if (!flatGraphExec_) {
            stream.waitForCompletion();   // 捕获前必须排空本流（WDDM 驱动崩溃防线）
            gerr = cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
            if (gerr == cudaSuccess) {
                // 【P2-2】捕获体＝参数块 memcpy 节点＋两个 Block 壳内核（免改参数）
                cudaMemcpyAsync(d_params_, h_params_, sizeof(FlatParamsDev),
                                cudaMemcpyHostToDevice, cs);
                FlatPreSmoothKernelBlock<<<pg, pb, 0, cs>>>(d_params_);
                FlatCenterK1Block<true><<<grid, block, 0, cs>>>(d_params_);
                cudaGraph_t g = nullptr;
                gerr = cudaStreamEndCapture(cs, &g);
                if (gerr == cudaSuccess)
                    gerr = cudaGraphInstantiate(&flatGraphExec_, g, nullptr, nullptr, 0);
                if (gerr == cudaSuccess && g) {
                    flatGraph_ = g;
                    graph_rows_ = rows;
                    graph_cols_ = cols;
                    graph_bands_ = nbandsG;
                } else {
                    if (g) cudaGraphDestroy(g);
                    flatGraphExec_ = nullptr;   // 捕获/实例化失败→回退逐发路径
                }
            }
        }
        }
            if (flatGraphExec_) {
                gerr = cudaGraphLaunch(flatGraphExec_, cs);
                if (gerr != cudaSuccess) {
                    CALIB_LOG_WARN("[flat] graphLaunch 失败回退逐发: {}",
                                   cudaGetErrorString(gerr));
                    flatGraphExec_ = nullptr;
                }
            }
            if (!flatGraphExec_) {
                // 回退：逐发原路径
                FlatPreSmoothKernel<<<pg, pb, 0, cs>>>(
                    d_grayImage.ptr<unsigned char>(), d_grayImage.step, rows, cols,
                    d_gsmooth_, d_stats_);
                FlatCenterK1<true><<<grid, block, 0, cs>>>(
                    d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
                    d_grayImage.step, d_binaryMask.step, rows, cols,
                    params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
                    params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
                    params_.maxLineWidth, params_.minRunLength, params_.smooth3,
                    d_gsmooth_, d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】出点容量截断
            }
        } else {
            FlatPreSmoothKernel<<<pg, pb, 0, cs>>>(
                d_grayImage.ptr<unsigned char>(), d_grayImage.step, rows, cols,
                d_gsmooth_, d_stats_);
            FlatCenterK1<true><<<grid, block, 0, cs>>>(
                d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
                d_grayImage.step, d_binaryMask.step, rows, cols,
                params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
                params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
                params_.maxLineWidth, params_.minRunLength, params_.smooth3,
                d_gsmooth_, d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】出点容量截断
        }
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    } else {
        // 【flat 优化 D】无预平滑路径：清零独立发射（stats 清零）
        cudaMemsetAsync(d_stats_, 0, 7 * sizeof(int), cs);
        int nbands = 96;   // 【BANDS 定界】同 nbandsG（nsys 实测 96 最优）
        if (const char* e = getenv("JMW_FLATBANDS"))
            nbands = std::max(1, std::min(128, atoi(e)));
        if (params_.scanAxis == 1) {
            // 261107 行模式（scanAxis=1）：竖带质心——1 线程/行、段沿 x
            const dim3 block2(256);
            const dim3 grid2(static_cast<unsigned>((rows + 255) / 256), 16);
            FlatCenterK1Row<true><<<grid2, block2, 0, cs>>>(
                d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
                d_grayImage.step, d_binaryMask.step, rows, cols,
                params_.minRunLength, params_.lowThreshold, params_.minContrast,
                params_.maxLineWidth, params_.segMergeGap, d_coords_, d_ids_, d_stats_,
                FlatCenterExtractor::kOutCap);
        } else {
        const dim3 block(256),
            grid((cols + 255) / 256, static_cast<unsigned>(nbands));
        FlatCenterK1<true><<<grid, block, 0, cs>>>(
            d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
            d_grayImage.step, d_binaryMask.step, rows, cols,
            params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
            params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
            params_.maxLineWidth, params_.minRunLength, params_.smooth3,
            static_cast<const float*>(nullptr),
            d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】
        }
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    }
    if (bench_ && events_ready_) cudaEventRecord(ev_scan_, cs);

    // d_stats_ 7 槽一次 D2H（pinned）——【合并唤醒】提交段到此为止（不同步）
    cudaMemcpyAsync(h_stats_, d_stats_, 7 * sizeof(int), cudaMemcpyDeviceToHost, cs);
    // 容量宽临时 ROI（N 未定——FinishDeferred 重建 [0,N)）
    d_out_cp_ = cv::cuda::GpuMat(1, FlatCenterExtractor::kOutCap, CV_32FC2, d_coords_);   // 【E5】容量宽视图
    d_out_pl_ = cv::cuda::GpuMat(1, FlatCenterExtractor::kOutCap, CV_32SC1, d_ids_);
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(
        d_out_cp_, cv::Rect(0, 0, FlatCenterExtractor::kOutCap, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
        d_out_pl_, cv::Rect(0, 0, FlatCenterExtractor::kOutCap, 1));
    result.totalPointCount = -1;   // 占位——FinishDeferred 填真值
    result.lineCount = 1;
    result.mergedCount = 0;
    result.success = true;   // 【修复】提交级成功（harness submitLaserFront 818 检查
                              // 要求提交期 success=true；N/零点在 FinishDeferred 定）
    return result;
}

void FlatCenterExtractor::Impl::finishFlat(StegerResult& result) {
    result.success = true;   // 【修复】原尾段置位随拆分遗漏——成功路径两分叉通用
    // 【E5】出点容量截断口径：K1 槽位≥kOutCap 不写但计数照加——min(N,kOutCap)
    // 与写入侧一致（超出容量点静默弃——实测 N max≈34k，2×余量下不触发）
    const int N = std::min(h_stats_[0], FlatCenterExtractor::kOutCap);   // 钳位（照 steger:1635）
    int dropSum = 0;
    for (int i = 1; i <= 6; ++i) dropSum += h_stats_[i];

    // —— 零点分叉（设计 §2）：全零掩膜→Warning；全被门弃→Degraded ——
    if (N == 0) {
        result.d_centerPoints.reset();
        result.d_line_ids.reset();
        result.success = true;
        result.totalPointCount = 0;
        result.lineCount = 0;
        result.mergedCount = 0;
        if (dropSum == 0) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "No laser points (mask all zero?)";
        } else {
            result.qualityFlag = QualityFlag::Degraded;
            result.message = "flat_center: all runs dropped";
        }
        bench_ms_[0] = bench_ms_[1] = bench_ms_[2] = 0.f;   // 早退帧清零防残留
        return;
    }

    // —— Flat 直通快路：[0,N) ROI 视图（帧内消费契约同 steger d_out_cp_/d_out_pl_）——
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(
        d_out_cp_, cv::Rect(0, 0, N, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
        d_out_pl_, cv::Rect(0, 0, N, 1));
    result.totalPointCount = N;
    result.lineCount = 1;         // Flat 单标签
    result.mergedCount = 0;

    if (dropSum > 0) {
        result.qualityFlag = QualityFlag::Degraded;
        result.message = "flat_center: dropped " + std::to_string(dropSum)
            + " runs, reasons=minRun:" + std::to_string(h_stats_[1])
            + " contrast:" + std::to_string(h_stats_[2])
            + " bright:" + std::to_string(h_stats_[3])
            + " edge:" + std::to_string(h_stats_[4])
            + " gate:" + std::to_string(h_stats_[5])
            + " width:" + std::to_string(h_stats_[6]);
        CALIB_LOG_DEBUG("{}", result.message);
    } else {
        result.qualityFlag = QualityFlag::Normal;
        result.message = "Subpixel extraction successful";
    }
}

StegerResult FlatCenterExtractor::Impl::extractFlat(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    StegerResult result = extractFlatSubmit(d_grayImage, d_binaryMask, stream);
    if (bench_ && events_ready_) cudaEventRecord(ev_end_, cs);
    stream.waitForCompletion();
    finishFlat(result);
    if (result.totalPointCount == 0) {
        result.success = true;
        return result;   // 零点早退（原口径：跳过 host 输出与 bench）
    }

    const int N = result.totalPointCount;
    // —— 可选 host 输出（collectHostPoints_=true）——
    if (collectHostPoints_) {
        cudaMemcpyAsync(h_coords_, d_coords_, N * sizeof(float2), cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(h_ids_, d_ids_, N * sizeof(int), cudaMemcpyDeviceToHost, cs);
        stream.waitForCompletion();
        std::vector<cv::Point2f> v(N);
        for (int i = 0; i < N; ++i) v[i] = cv::Point2f(h_coords_[i].x, h_coords_[i].y);
        result.centerPoints.emplace(1, std::move(v));   // Flat 单标签
    }

    result.success = true;

    // —— bench 汇总（JMW_FLATTIME=1 时）——
    if (bench_ && events_ready_) {
        cudaEventSynchronize(ev_end_);
        float t = 0;
        cudaEventElapsedTime(&t, ev_start_, ev_end_);  bench_ms_[0] = t;
        cudaEventElapsedTime(&t, ev_start_, ev_scan_); bench_ms_[1] = t;
        cudaEventElapsedTime(&t, ev_scan_, ev_end_);   bench_ms_[2] = t;
        cudaGetLastError();   // 吞事件非致命错防粘性（照 steger:1972 惯例）
        CALIB_LOG_DEBUG("[BENCH_FLAT] N={}", N);
    }
    return result;
}

// —— ByLabel 分组 kernel（照抄 steger LabelHist/LabelScatter 模式，键改 ids 直读）——
__global__ void FlatLabelHistKernel(const int* __restrict__ ids, int count, int bins,
                                    int* __restrict__ hist)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = ids[i];
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    atomicAdd(&hist[lb], 1);
}
__global__ void FlatLabelScatterKernel(
    const float2* __restrict__ coords, const int* __restrict__ ids, int count, int bins,
    const int* __restrict__ base, int* __restrict__ cursor,
    float2* __restrict__ outCoords, int* __restrict__ outIds)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = ids[i];
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    const int slot = atomicAdd(&cursor[lb], 1);
    const int idx = base[lb] + slot;
    outCoords[idx] = coords[i];
    outIds[idx] = lb;
}

// ============================================================================
// ByLabel 主路径（设计 §4：K1<false> 源序出点 → hist/前缀和/scatter 分组）
// ============================================================================

StegerResult FlatCenterExtractor::Impl::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask,
    bool maskIs8U,
    cv::cuda::Stream& stream)
{
    (void)maskIs8U;   // 本实现恒走 32S 路径（桥接层已保证类型）
    StegerResult result;
    const int rows = d_grayImage.rows, cols = d_grayImage.cols;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    ensureBuffers(rows, cols);
    if (bench_ && !events_ready_) createEvents();
    cudaMemsetAsync(d_stats_, 0, 7 * sizeof(int), cs);
    if (bench_ && events_ready_) cudaEventRecord(ev_start_, cs);

    if (params_.scanAxis == 1) {
        // 261107 行模式（scanAxis=1）：竖带质心——1 线程/行、段沿 x、
        // 段内去背景亮度加权质心（平顶/半高链不适用宽带）；收尾（stats
        // D2H/分组/出壳）复用下方 32S 原路
        const dim3 block2(256);
        const dim3 grid2(static_cast<unsigned>((rows + 255) / 256), 16);
        FlatCenterK1Row<false><<<grid2, block2, 0, cs>>>(
            d_grayImage.ptr<unsigned char>(), d_labeledMask.ptr(),
            d_grayImage.step, d_labeledMask.step, rows, cols,
            params_.minRunLength, params_.lowThreshold, params_.minContrast,
            params_.maxLineWidth, params_.segMergeGap, d_coords_, d_ids_, d_stats_,
            FlatCenterExtractor::kOutCap);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] row-kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    } else {
    {
        const dim3 block(256), grid((cols + 255) / 256, 8);   // 列带×8（设计 §4 备选）
        FlatCenterK1<false><<<grid, block, 0, cs>>>(
            d_grayImage.ptr<unsigned char>(), d_labeledMask.ptr<int>(),
            d_grayImage.step, d_labeledMask.step, rows, cols,
            params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
            params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
            params_.maxLineWidth, params_.minRunLength, params_.smooth3,
            static_cast<const float*>(nullptr),   /* 【flat 优化 C】ByLabel 不预平滑 */
            d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    }
    }
    if (bench_ && events_ready_) cudaEventRecord(ev_scan_, cs);

    cudaMemcpyAsync(h_stats_, d_stats_, 7 * sizeof(int), cudaMemcpyDeviceToHost, cs);
    stream.waitForCompletion();

    const int N = std::min(h_stats_[0], capacity_);
    int dropSum = 0;
    for (int i = 1; i <= 6; ++i) dropSum += h_stats_[i];

    if (N == 0) {
        result.success = true;
        result.totalPointCount = 0;
        result.lineCount = 0;
        result.mergedCount = 0;
        if (dropSum == 0) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "No laser points (mask all zero?)";
        } else {
            result.qualityFlag = QualityFlag::Degraded;
            result.message = "flat_center: all runs dropped";
        }
        bench_ms_[0] = bench_ms_[1] = bench_ms_[2] = 0.f;
        return result;
    }

    // —— 分组：hist → CPU 前缀和 → scatter（照 steger ByLabel 组装段模式）——
    cudaMemsetAsync(d_hist_, 0, kLabelBins * sizeof(int), cs);
    {
        const int threads = 256, blocks = (N + threads - 1) / threads;
        FlatLabelHistKernel<<<blocks, threads, 0, cs>>>(d_ids_, N, kLabelBins, d_hist_);
    }
    cudaMemcpyAsync(h_hist_, d_hist_, kLabelBins * sizeof(int), cudaMemcpyDeviceToHost, cs);
    stream.waitForCompletion();

    int running = 0, lineCount = 0;
    for (int lb = 0; lb < kLabelBins; ++lb) {
        h_base_[lb] = running;
        running += h_hist_[lb];
        if (h_hist_[lb] > 0) ++lineCount;
    }

    cudaMemcpyAsync(d_base_, h_base_, kLabelBins * sizeof(int), cudaMemcpyHostToDevice, cs);
    cudaMemsetAsync(d_cursor_, 0, kLabelBins * sizeof(int), cs);
    {
        const int threads = 256, blocks = (N + threads - 1) / threads;
        FlatLabelScatterKernel<<<blocks, threads, 0, cs>>>(
            d_coords_, d_ids_, N, kLabelBins, d_base_, d_cursor_, d_gcoords_, d_gids_);
    }
    if (bench_ && events_ready_) cudaEventRecord(ev_end_, cs);

    d_out_gc_ = cv::cuda::GpuMat(1, capacity_, CV_32FC2, d_gcoords_);
    d_out_gl_ = cv::cuda::GpuMat(1, capacity_, CV_32SC1, d_gids_);
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(d_out_gc_, cv::Rect(0, 0, N, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(d_out_gl_, cv::Rect(0, 0, N, 1));
    result.totalPointCount = N;
    result.lineCount = lineCount;               // hist 非零桶数（与 collect 开关无关）
    result.mergedCount = 0;

    if (dropSum > 0) {
        result.qualityFlag = QualityFlag::Degraded;
        result.message = "flat_center: dropped " + std::to_string(dropSum)
            + " runs, reasons=minRun:" + std::to_string(h_stats_[1])
            + " contrast:" + std::to_string(h_stats_[2])
            + " bright:" + std::to_string(h_stats_[3])
            + " edge:" + std::to_string(h_stats_[4])
            + " gate:" + std::to_string(h_stats_[5])
            + " width:" + std::to_string(h_stats_[6]);
        CALIB_LOG_DEBUG("{}", result.message);
    } else {
        result.qualityFlag = QualityFlag::Normal;
        result.message = "Subpixel extraction successful";
    }

    if (collectHostPoints_) {
        cudaMemcpyAsync(h_coords_, d_gcoords_, N * sizeof(float2), cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(h_ids_, d_gids_, N * sizeof(int), cudaMemcpyDeviceToHost, cs);
        stream.waitForCompletion();
        for (int lb = 0; lb < kLabelBins; ++lb) {
            const int n = h_hist_[lb];
            if (n <= 0) continue;
            std::vector<cv::Point2f> v(n);
            for (int i = 0; i < n; ++i) {
                const float2& p = h_coords_[h_base_[lb] + i];
                v[i] = cv::Point2f(p.x, p.y);
            }
            result.centerPoints.emplace(lb, std::move(v));
        }
    }

    result.success = true;

    if (bench_ && events_ready_) {
        cudaEventSynchronize(ev_end_);
        float t = 0;
        cudaEventElapsedTime(&t, ev_start_, ev_end_);  bench_ms_[0] = t;
        cudaEventElapsedTime(&t, ev_start_, ev_scan_); bench_ms_[1] = t;
        cudaEventElapsedTime(&t, ev_scan_, ev_end_);   bench_ms_[2] = t;
        cudaGetLastError();
        CALIB_LOG_DEBUG("[BENCH_FLAT] {}x{} | Total={:.2f}ms | Scan={:.2f}ms | Group/Out={:.2f}ms | N={} lines={}",
                        cols, rows, bench_ms_[0], bench_ms_[1], bench_ms_[2], N, lineCount);
    }
    return result;
}

void FlatCenterExtractor::Impl::Destroy() {
    freeAll();
}

void FlatCenterExtractor::Impl::Warmup(int rows, int cols) {
    ensureBuffers(rows, cols);
    warmed_up_ = true;
    warmup_rows_ = rows;
    warmup_cols_ = cols;
#ifndef NDEBUG
    // Debug 下 OOM 校验惯例（照 steger_fast_impl.cu:1338-1351）
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        CALIB_LOG_ERROR("Warmup device sync failed: {}", cudaGetErrorString(err));
    }
#endif
}

void FlatCenterExtractor::Impl::SetParams(const FlatCenterParams& params) {
    params.validate();
    if (params.deviceId != params_.deviceId) {
        old_device_id_ = params_.deviceId;
        // 设备切换：缓冲随设备走，重置后下次 Execute 重建
        freeAll();
        warmed_up_ = false;
    }
    // 【flat 优化·宏打包】参数变更→捕获图失效（内核实参已变）——销毁待重捕
    if (flatGraphExec_) { cudaGraphExecDestroy(flatGraphExec_); flatGraphExec_ = nullptr; }
    if (flatGraph_)     { cudaGraphDestroy(flatGraph_); flatGraph_ = nullptr; }
    params_ = params;
    ++paramsEpoch_;   // 【G′·C4′】SetParams——阈值标量/维度上限变化（自图已毁＋大图签名失配）
}

// ============================================================================
// 【G′·三合一大图】裸发射口三件——harness FrontBigGraph 捕获窗专用
// （发射体＝flat 自图捕获体同源：参数块 memcpy＋PreSmooth/K1 Block 壳；差异＝
//  stats 7 槽 D2H 并入发射体（大图节点化，anchor −2 mcaD2H/帧））
// ============================================================================

bool FlatCenterExtractor::Impl::prepareBigGraph_(
    const cv::cuda::GpuMat& d_grayImage, const cv::cuda::GpuMat& d_binaryMask) {
    if (paramsFailed_) return false;
    if (!d_params_ || !h_params_) {
        cudaMalloc(&d_params_, sizeof(FlatParamsDev));
        cudaMallocHost(&h_params_, sizeof(FlatParamsDev));
        if (cudaGetLastError() != cudaSuccess || !d_params_ || !h_params_) {
            CALIB_LOG_WARN("[flat] params block alloc failed; big-graph disabled");
            if (d_params_) { cudaFree(d_params_); d_params_ = nullptr; }
            if (h_params_) { cudaFreeHost(h_params_); h_params_ = nullptr; }
            paramsFailed_ = true;
            return false;
        }
    }
    // 维度守卫置位（与 extractFlatSubmit 同源的 nbands 环境口径）
    int nbandsG = 96;
    if (const char* e = getenv("JMW_FLATBANDS"))
        nbandsG = std::max(1, std::min(128, atoi(e)));
    graph_rows_ = d_grayImage.rows;
    graph_cols_ = d_grayImage.cols;
    graph_bands_ = nbandsG;
    refreshGraphParams_(d_grayImage, d_binaryMask);
    return true;
}

void FlatCenterExtractor::Impl::refreshGraphParams_(
    const cv::cuda::GpuMat& d_grayImage, const cv::cuda::GpuMat& d_binaryMask) {
    if (!h_params_) return;
    h_params_->g = d_grayImage.ptr<unsigned char>();
    h_params_->gstep = d_grayImage.step;
    h_params_->mask = d_binaryMask.ptr<unsigned char>();
    h_params_->mstep = d_binaryMask.step;
    h_params_->gs = d_gsmooth_;
    h_params_->coords = d_coords_;
    h_params_->ids = d_ids_;
    h_params_->stats = d_stats_;
    h_params_->rows = d_grayImage.rows;
    h_params_->cols = d_grayImage.cols;
    h_params_->maxpts = FlatCenterExtractor::kOutCap;
    h_params_->low_thresh = params_.lowThreshold;
    h_params_->half_ratio = params_.halfRatio;
    h_params_->plateau_ratio = params_.plateauRatio;
    h_params_->plateau_min_w = params_.plateauMinWidth;
    h_params_->min_contrast = params_.minContrast;
    h_params_->outlier_gate = params_.outlierGatePx;
    h_params_->max_line_width = params_.maxLineWidth;
    h_params_->min_run = params_.minRunLength;
    h_params_->smooth3 = params_.smooth3 ? 1 : 0;
}

bool FlatCenterExtractor::Impl::emitGraphBody_(cudaStream_t cs) {
    // 前置：参数块就绪＋维度守卫值有效（PrepareBigGraph 置位）
    if (paramsFailed_ || !d_params_ || !h_params_ || graph_rows_ <= 0) return false;
    const int rows = graph_rows_, cols = graph_cols_, nbandsG = graph_bands_;
    const dim3 pb(128, 8);
    const dim3 pg(static_cast<unsigned>((cols + 127) / 128),
                  static_cast<unsigned>((rows + 7) / 8));
    const dim3 block(256),
        grid((cols + 255) / 256, static_cast<unsigned>(nbandsG));
    cudaError_t err = cudaMemcpyAsync(d_params_, h_params_, sizeof(FlatParamsDev),
                                      cudaMemcpyHostToDevice, cs);
    if (err != cudaSuccess) return false;
    FlatPreSmoothKernelBlock<<<pg, pb, 0, cs>>>(d_params_);
    FlatCenterK1Block<true><<<grid, block, 0, cs>>>(d_params_);
    // stats 7 槽 D2H 并入图体（大图专属——自图路径该笔在图外逐帧发射）
    err = cudaMemcpyAsync(h_stats_, d_stats_, 7 * sizeof(int),
                          cudaMemcpyDeviceToHost, cs);
    return err == cudaSuccess;
}

StegerResult FlatCenterExtractor::Impl::makeDeferredShell_() {
    StegerResult result;
    d_out_cp_ = cv::cuda::GpuMat(1, FlatCenterExtractor::kOutCap, CV_32FC2, d_coords_);
    d_out_pl_ = cv::cuda::GpuMat(1, FlatCenterExtractor::kOutCap, CV_32SC1, d_ids_);
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(
        d_out_cp_, cv::Rect(0, 0, FlatCenterExtractor::kOutCap, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
        d_out_pl_, cv::Rect(0, 0, FlatCenterExtractor::kOutCap, 1));
    result.totalPointCount = -1;
    result.lineCount = 1;
    result.mergedCount = 0;
    result.success = true;
    return result;
}
