/**
 * @file steger_fast_impl.cu
 * @brief Steger激光中心亚像素提取算子（优化版）CUDA 实现
 *
 * 步骤（对照原版 ../steger/steger_extract_cuda_impl.cu 的 6 步）：
 *   Step 1+2a: RowFan3Kernel — uchar 直读一次算出行级 g/gx/gxx 三个中间图
 *              （原版 5 次行卷积、其中 g 与 g' 各被重复计算一次；本版读 1 遍写 3 遍）
 *   Step 2b:   ColFan5Kernel — 一次算出 Ix/Iy/Ixx/Iyy/Ixy 五个导数（原版 5 次列卷积）
 *   Step 3+4:  HessianEigenAndTaylorKernel — 与原版逐位一致（Flat 模式模板化 8U 掩膜）
 *   Step 5:    LabelHist/LabelScatter — 计数直写分组替代 thrust stable_sort
 *   Step 6:    单次 D2H + D2D 组装 CPU/GPU 双输出，替代原版 D2H→cv::Mat→H2D 往返
 *
 * 数值等价性：导数累加保持与原版相同的 k 升序与 renormalize（w_sum 除法）语义，
 * 输入 uchar→float 转换无损，因此导数与点坐标与原版逐位一致；
 * 分组后 label 内点序与原版一样由原子时序决定（本就非确定），坐标值不变。
 */

#include "steger_fast_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

using namespace calib;

CALIB_DEFINE_LOG_TAG(10, StegerExtractorFast);

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void RowFan3Kernel(
    const unsigned char* __restrict__ d_in,
    float* __restrict__ d_out_g, float* __restrict__ d_out_gx, float* __restrict__ d_out_gxx,
    int rows, int cols, size_t in_step, size_t out_step,
    const float* __restrict__ kg, const float* __restrict__ kgx, const float* __restrict__ kgxx,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int half = kernel_size / 2;
    const unsigned char* in_row = d_in + y * in_step;

    float s_g = 0.0f, ws_g = 0.0f, s_gx = 0.0f, s_gxx = 0.0f;
    for (int k = -half; k <= half; ++k) {
        int xx = x + k;
        if (xx >= 0 && xx < cols) {
            float v = static_cast<float>(in_row[xx]);
            float wg = kg[k + half];
            s_g += wg * v;
            ws_g += wg;
            s_gx += kgx[k + half] * v;
            s_gxx += kgxx[k + half] * v;
        }
    }

    char* og = reinterpret_cast<char*>(d_out_g) + y * out_step;
    char* ogx = reinterpret_cast<char*>(d_out_gx) + y * out_step;
    char* ogxx = reinterpret_cast<char*>(d_out_gxx) + y * out_step;
    reinterpret_cast<float*>(og)[x] = s_g / ws_g;
    reinterpret_cast<float*>(ogx)[x] = s_gx;
    reinterpret_cast<float*>(ogxx)[x] = s_gxx;
}

__global__ void ColFan5Kernel(
    const float* __restrict__ d_tg, const float* __restrict__ d_tgx, const float* __restrict__ d_tgxx,
    float* __restrict__ d_ix, float* __restrict__ d_iy,
    float* __restrict__ d_ixx, float* __restrict__ d_iyy, float* __restrict__ d_ixy,
    int rows, int cols, size_t in_step, size_t out_step,
    const float* __restrict__ kg, const float* __restrict__ kgx, const float* __restrict__ kgxx,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int half = kernel_size / 2;
    float s_ix = 0.0f, ws_ix = 0.0f;
    float s_iy = 0.0f;
    float s_ixx = 0.0f, ws_ixx = 0.0f;
    float s_iyy = 0.0f;
    float s_ixy = 0.0f;

    for (int k = -half; k <= half; ++k) {
        int yy = y + k;
        if (yy >= 0 && yy < rows) {
            const float* rg = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_tg) + yy * in_step);
            const float* rgx = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_tgx) + yy * in_step);
            const float* rgxx = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_tgxx) + yy * in_step);
            float vg = rg[x], vgx = rgx[x], vgxx = rgxx[x];
            float wg = kg[k + half], wgx = kgx[k + half], wgxx = kgxx[k + half];
            s_ix += wg * vgx;  ws_ix += wg;
            s_iy += wgx * vg;
            s_ixx += wg * vgxx;  ws_ixx += wg;
            s_iyy += wgxx * vg;
            s_ixy += wgx * vgx;
        }
    }

    char* oix = reinterpret_cast<char*>(d_ix) + y * out_step;
    char* oiy = reinterpret_cast<char*>(d_iy) + y * out_step;
    char* oixx = reinterpret_cast<char*>(d_ixx) + y * out_step;
    char* oiyy = reinterpret_cast<char*>(d_iyy) + y * out_step;
    char* oixy = reinterpret_cast<char*>(d_ixy) + y * out_step;
    reinterpret_cast<float*>(oix)[x] = s_ix / ws_ix;
    reinterpret_cast<float*>(oiy)[x] = s_iy;
    reinterpret_cast<float*>(oixx)[x] = s_ixx / ws_ixx;
    reinterpret_cast<float*>(oiyy)[x] = s_iyy;
    reinterpret_cast<float*>(oixy)[x] = s_ixy;
}

template <bool MASK8>
__global__ void HessianEigenAndTaylorKernel(
    const float* d_ix,  const float* d_iy,
    const float* d_ixx, const float* d_iyy, const float* d_ixy,
    const void* d_mask, size_t mask_step,
    const float* d_tg, size_t tg_step,
    int rows, int cols,
    size_t ix_step, size_t ixx_step,
    float low_thresh, float high_thresh,
    float max_line_width,
    FastPoint* d_out_points, int* d_out_count, int* d_reject_count,
    int max_points)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int label;
    if (MASK8) {
        const unsigned char* mask_row =
            reinterpret_cast<const unsigned char*>(d_mask) + y * mask_step;
        label = mask_row[x] > 0 ? 1 : 0;
    } else {
        const int* mask_row = reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(d_mask) + y * mask_step);
        label = mask_row[x];
    }
    if (label <= 0) return;

    const float* ix_row  = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_ix) + y * ix_step);
    const float* iy_row  = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_iy) + y * ix_step);
    const float* ixx_row = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_ixx) + y * ixx_step);
    const float* iyy_row = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_iyy) + y * ixx_step);
    const float* ixy_row = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_ixy) + y * ixx_step);

    float ix_val  = ix_row[x];
    float iy_val  = iy_row[x];
    float ixx_val = ixx_row[x];
    float iyy_val = iyy_row[x];
    float ixy_val = ixy_row[x];

    float trace = ixx_val + iyy_val;
    float det = ixx_val * iyy_val - ixy_val * ixy_val;
    float disc = trace * trace - 4.0f * det;

    if (disc < 0.0f) return;

    float sqrt_disc = sqrtf(disc);
    float lambda1 = 0.5f * (trace + sqrt_disc);
    float lambda2 = 0.5f * (trace - sqrt_disc);

    float lambda;
    float nx, ny;
    if (fabsf(lambda1) >= fabsf(lambda2)) {
        lambda = lambda1;
        float denom = ixy_val;
        if (fabsf(denom) < 1e-10f) {
            nx = 1.0f;
            ny = 0.0f;
        } else {
            nx = lambda1 - iyy_val;
            ny = ixy_val;
        }
    } else {
        lambda = lambda2;
        float denom = ixy_val;
        if (fabsf(denom) < 1e-10f) {
            nx = 0.0f;
            ny = 1.0f;
        } else {
            nx = lambda2 - iyy_val;
            ny = ixy_val;
        }
    }

    if (fabsf(lambda) < low_thresh) return;
    if (high_thresh > 0.0f && fabsf(lambda) > high_thresh) return;

    // 逐点宽度门禁(maxLineWidth>0 启用; off 时分支恒 false → 与原版逐位一致):
    // 平滑后口径 w² = g/|λ|（g=行平滑幅度, 近似法向峰幅; λ 已过 low_thresh 下限无除零）。
    // 过曝局部变宽/平顶剖面 → w² 显著大于正常线 → 弃点（防亚像素中心偏移噪点）。
    // g<0 数值噪声 → w²<0 → 比较为 false → 放行（等同 off, NaN 语义同）。
    if (max_line_width > 0.0f) {
        const float* tg_row = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(d_tg) + y * tg_step);
        const float w2 = tg_row[x] / fabsf(lambda);
        if (w2 > max_line_width * max_line_width) {
            atomicAdd(d_reject_count, 1);
            return;
        }
    }

    float norm = sqrtf(nx * nx + ny * ny);
    if (norm < 1e-10f) return;
    nx /= norm;
    ny /= norm;

    float denom2 = ixx_val * nx * nx + 2.0f * ixy_val * nx * ny + iyy_val * ny * ny;
    if (fabsf(denom2) < 1e-10f) return;

    float t = -(ix_val * nx + iy_val * ny) / denom2;

    if (fabsf(t) >= 0.5f) return;

    float px = static_cast<float>(x) + t * nx;
    float py = static_cast<float>(y) + t * ny;

    if (px < 0.0f || px >= static_cast<float>(cols) ||
        py < 0.0f || py >= static_cast<float>(rows)) return;

    int idx = atomicAdd(d_out_count, 1);
    if (idx < max_points) {
        d_out_points[idx].label = label;
        d_out_points[idx].px = px;
        d_out_points[idx].py = py;
    }
}

__global__ void LabelHistKernel(
    const FastPoint* __restrict__ pts, int count, int bins,
    int* __restrict__ hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = pts[i].label;
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    atomicAdd(&hist[lb], 1);
}

__global__ void LabelScatterKernel(
    const FastPoint* __restrict__ pts, int count, int bins,
    const int* __restrict__ base, int* __restrict__ cursor,
    FastPoint* __restrict__ out,
    float2* __restrict__ coords, int* __restrict__ ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = pts[i].label;
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    int slot = atomicAdd(&cursor[lb], 1);
    int idx = base[lb] + slot;
    FastPoint p = pts[i];
    p.label = lb;
    out[idx] = p;
    coords[idx] = make_float2(p.px, p.py);
    ids[idx] = lb;
}

// ============================================================================
// dedup v2（dedupMode=1）：平顶双点位移式合并——网格排序 + 邻域 PCA 几何切向
// + 位移门配对（定长邻接行）+ pull/shortcut 标签传播 + (簇键,rrid) 排序段归约。
// 判据常量与 CPU 验证版（TEST/steger_dedup_replay/src/steger_dedup_merge.cpp）逐字同值；
// 确定性：排序键拼 rrid（round 恢复有半 ulp 撞号洞——键重复时稳定排序保输入序，
// 见 D1 注）；PCA top-10 按 (d²,rrid) 平局裁决＋定序累加；传播双缓冲纯 pull
// 无原子（rootkey 的 atomicMin 结果序无关合法）。
// ============================================================================
constexpr float kDedupPairDist   = 1.6f;   // 配对欧氏距上限
constexpr float kDedupAlongGate  = 0.5f;   // 同簇：沿线偏移 ≤0.5px
constexpr float kDedupNormalGate = 1.5f;   //        法向偏移 ≤1.5px
constexpr float kDedupTangentDot = 0.5f;   // 切向一致门 |t̂ᵢ·t̂ⱼ|≥0.5
constexpr float kDedupNbrRadius  = 2.5f;   // PCA 邻域半径
constexpr int   kDedupNbrMax     = 10;     // PCA 邻居上限
constexpr int   kDedupClusterMax = 8;      // 簇点数上限（超则整段直通）
constexpr float kDedupSpanMax    = 2.5f;   // 簇法向跨度上限（超则直通）
constexpr int   kDedupAdjStride  = 24;     // 邻接行宽：几何上界 8（半径 1.6 圆盘含自己
                                           // ≤9 格点、每像素≤1 点），3 倍余量；溢出→flag 直通
constexpr int   kDedupCellBits   = 12;     // cell 坐标位宽（4096 守卫保证域）
constexpr int   kDedupRidxBits   = 22;     // rrid 位宽（4M 守卫保证域）

// D1 打包：cell=(floor(py)<<12|floor(px))，rrid=round 像素恢复 y*cols+x。
// ⚠ round 恢复并非绝对不变量：|px-x|=|t·nx|<0.5 严格，但 t 处于距 0.5 半个 ulp
// 内时 round 可舍到相邻像素 → rrid 撞号（角例）。因此 rrid 只作排序/平局裁决键，
// 不承担寻址唯一性（切向等中间数组按 D1 排序位索引，见 D2/D3/D6）。
// 撞号残余（文档化接受）：D1/D5 键重复时 CUB 稳定排序保 atomicAdd 输入序；
// 撞号两点坐标差 ~ulp 量级，仅影响中位并列时的极微取舍；D3 对门方规则用 >=
// 使等 rrid 对双侧一致弃权（直通，无害且确定）。
// 排序载荷用 u32 下标（16B DedupPt 载荷经 gather 物化——u32 载荷排序更省）
__global__ void DedupCellPackKernel(const FastPoint* pts, int n, int cols,
                                    unsigned long long* keys, unsigned* idxs,
                                    DedupPt* vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = pts[i].px, py = pts[i].py;
    int cell = ((int)floorf(py) << kDedupCellBits) | (int)floorf(px);
    int rrid = __float2int_rn(py) * cols + __float2int_rn(px);
    keys[i] = ((unsigned long long)(unsigned)cell << kDedupRidxBits)
            | (unsigned long long)(unsigned)rrid;
    idxs[i] = (unsigned)i;
    DedupPt v;
    v.rrid = rrid; v.label = pts[i].label; v.px = px; v.py = py;
    vals[i] = v;
}

// D1 gather：按排序后下标物化 DedupPt 载荷
__global__ void DedupGatherKernel(const unsigned* __restrict__ idxs,
                                  const DedupPt* __restrict__ vals_in,
                                  DedupPt* __restrict__ vals_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vals_out[i] = vals_in[idxs[i]];
}

// 排序键数组上按 cell 下界二分：首个 ≥ (cell<<22) 的位置（~22 步）
__device__ __forceinline__ int DedupCellLowerBound(
    const unsigned long long* __restrict__ keys, int n, unsigned long long target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 行带扫描定位（D2/D3 共用）：对每个 gy 行带 [gx_lo,gx_hi] 一次下界二分＋线性扫到
// 带尾（cell=1px² 且每像素 ≤1 点 → 带内 ≤ 带宽个点）——替代逐 cell 49/25 次二分。
// 行带尾 cell=((gy<<12)|(gx_hi+1))：gx_hi=4095 时恰为下一行首，break 条件仍正确
__device__ __forceinline__ void DedupBandRange(const unsigned long long* __restrict__ ckeys,
                                               int n, int gy, int gx_lo, int gx_hi,
                                               int& k_beg, unsigned& band_end_cell) {
    const unsigned long long bandLo =
        ((unsigned long long)((((unsigned)gy) << kDedupCellBits) | (unsigned)gx_lo)
         << kDedupRidxBits);
    band_end_cell = (((unsigned)gy) << kDedupCellBits) | (unsigned)(gx_hi + 1);
    k_beg = DedupCellLowerBound(ckeys, n, bandLo);
}

// D2 邻域 PCA 切向：±3 格（7×7）窗收 d²≤2.5² 候选，在线维护按 (d²,rrid) 全序的
// top-10（插入排序——劣于当前第 10 的候选可安全丢弃，无需全量收集），
// 定序累加 2×2 协方差 → 解析特征 → 各向异性门（CPU 严格式）→ 符号规约切向。
// 切向/valid 按 D1 排序位索引（非 rrid——round 恢复有半 ulp 撞号洞，见 D1 注）；
// 自跳过按排序位 k==i（撞号对方不再被误丢，更忠实 CPU kNN 语义）。
// 候选几何上界 20（半径 2.5 圆盘含自己 ≤21 格点、每像素≤1 点），
// top-10 在线法天然免上界约束
__global__ void DedupTangentKernel(const unsigned long long* __restrict__ ckeys,
                                   const DedupPt* __restrict__ cvals, int n,
                                   float2* __restrict__ tan_, char* __restrict__ tan_valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const DedupPt me = cvals[i];
    const int cx = (int)floorf(me.px), cy = (int)floorf(me.py);
    const float r2 = kDedupNbrRadius * kDedupNbrRadius;

    float top_d2[kDedupNbrMax], top_dx[kDedupNbrMax], top_dy[kDedupNbrMax];
    int top_rid[kDedupNbrMax];
    int cnt = 0;
    const int gx_lo = max(cx - 3, 0), gx_hi = min(cx + 3, (1 << kDedupCellBits) - 1);
    const int gy_lo = max(cy - 3, 0), gy_hi = min(cy + 3, (1 << kDedupCellBits) - 1);
    for (int gy = gy_lo; gy <= gy_hi; ++gy) {
        int k0; unsigned bandEnd;
        DedupBandRange(ckeys, n, gy, gx_lo, gx_hi, k0, bandEnd);
        for (int k = k0; k < n; ++k) {
            if ((unsigned)(ckeys[k] >> kDedupRidxBits) >= bandEnd) break;
            int rid = (int)(ckeys[k] & ((1u << kDedupRidxBits) - 1));
            if (k == i) continue;   // 自跳过按排序位（rrid 撞号对方不再误丢）
            float dx = cvals[k].px - me.px, dy = cvals[k].py - me.py;
            float d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            if (cnt == kDedupNbrMax &&
                (d2 > top_d2[kDedupNbrMax - 1] ||
                 (d2 == top_d2[kDedupNbrMax - 1] && rid >= top_rid[kDedupNbrMax - 1])))
                continue;   // 劣于当前第 10 → 丢弃（top-10 只会更好，安全）
            int p = cnt < kDedupNbrMax ? cnt++ : kDedupNbrMax - 1;
            while (p > 0 && (top_d2[p - 1] > d2 ||
                             (top_d2[p - 1] == d2 && top_rid[p - 1] > rid))) {
                top_d2[p] = top_d2[p - 1]; top_rid[p] = top_rid[p - 1];
                top_dx[p] = top_dx[p - 1]; top_dy[p] = top_dy[p - 1];
                --p;
            }
            top_d2[p] = d2; top_rid[p] = rid; top_dx[p] = dx; top_dy[p] = dy;
        }
    }

    if (cnt < 2) { tan_valid[i] = 0; return; }
    const int take = cnt;
    float sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int k = 0; k < take; ++k) {   // 定序累加（top 数组已按 (d²,rrid) 全序）
        sx += top_dx[k]; sy += top_dy[k];
        sxx += top_dx[k] * top_dx[k];
        syy += top_dy[k] * top_dy[k];
        sxy += top_dx[k] * top_dy[k];
    }
    const float mx = sx / take, my = sy / take;
    const float vxx = sxx / take - mx * mx, vyy = syy / take - my * my;
    const float vxy = sxy / take - mx * my;
    const float tr = vxx + vyy, det = vxx * vyy - vxy * vxy;
    const float disc = tr * tr * 0.25f - det;
    const float sq = sqrtf(fmaxf(0.f, disc));
    const float l1 = tr * 0.5f + sq, l2 = tr * 0.5f - sq;
    if (l1 <= 1e-6f || l1 < 4.f * fmaxf(l2, 1e-9f)) { tan_valid[i] = 0; return; }
    const float ang = 0.5f * atan2f(2.f * vxy, vxx - vyy);
    float tx = cosf(ang), ty = sinf(ang);
    if (tx < 0.f || (tx == 0.f && ty < 0.f)) { tx = -tx; ty = -ty; }
    tan_[i] = make_float2(tx, ty);
    tan_valid[i] = 1;
}

// D3 位移门配对（对称边）：±2 格（5×5）窗内，仅 rrid 小者（规范化对门方——CPU
// j>i 单侧求值口径的确定性规范化，设计 §3 良性偏差）以自己的切向求值；过门 →
// 边对称写入双方邻接行（atomicAdd 槽位，行内序不定——传播取 min 序无关，确定性
// 保持）。行满 → flags[0] 溢出（几何上界 8：半径 1.6 圆盘含自己 ≤9 格点、
// 每像素≤1 点；取 3 倍余量=24，防御门）。
// 切向/valid 按 D1 排序位索引；自跳过 k==i；对门方规则用 >=——等 rrid 病态对
// 双侧一致弃权（键重复时该对直通，无害且确定）。
// ⚠ 不可"各写各的"单侧行：pull 传播只沿行持有方拉取，单向边会把本应同簇的点
// 留在不同分量（CPU 并查集 union 天然对称——GPU 必须显式双边写）
__global__ void DedupPairEdgeKernel(const unsigned long long* __restrict__ ckeys,
                                    const DedupPt* __restrict__ cvals,
                                    const float2* __restrict__ tan_,
                                    const char* __restrict__ tan_valid,
                                    int n, int* __restrict__ adj, int* __restrict__ rowlen,
                                    int* __restrict__ flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const DedupPt me = cvals[i];
    if (!tan_valid[i]) return;
    const int cx = (int)floorf(me.px), cy = (int)floorf(me.py);
    const float2 t = tan_[i];
    const float pd2 = kDedupPairDist * kDedupPairDist;
    const int gx_lo = max(cx - 2, 0), gx_hi = min(cx + 2, (1 << kDedupCellBits) - 1);
    const int gy_lo = max(cy - 2, 0), gy_hi = min(cy + 2, (1 << kDedupCellBits) - 1);
    for (int gy = gy_lo; gy <= gy_hi; ++gy) {
        int k0; unsigned bandEnd;
        DedupBandRange(ckeys, n, gy, gx_lo, gx_hi, k0, bandEnd);
        for (int k = k0; k < n; ++k) {
            if ((unsigned)(ckeys[k] >> kDedupRidxBits) >= bandEnd) break;
            int rid = (int)(ckeys[k] & ((1u << kDedupRidxBits) - 1));
            if (k == i || !tan_valid[k]) continue;   // 自跳过按排序位＋valid 按排序位
            if (me.rrid >= rid) continue;   // 对门方=rrid 小者；等 rrid 双侧弃权（确定）
            float dx = cvals[k].px - me.px, dy = cvals[k].py - me.py;
            if (dx * dx + dy * dy > pd2) continue;
            if (fabsf(dx * t.x + dy * t.y) > kDedupAlongGate) continue;
            if (fabsf(-dx * t.y + dy * t.x) > kDedupNormalGate) continue;
            const float2 tj = tan_[k];
            if (fabsf(t.x * tj.x + t.y * tj.y) < kDedupTangentDot) continue;
            int si = atomicAdd(&rowlen[i], 1);
            if (si < kDedupAdjStride) adj[(size_t)i * kDedupAdjStride + si] = k;
            else atomicOr(&flags[0], 1);
            int sj = atomicAdd(&rowlen[k], 1);
            if (sj < kDedupAdjStride) adj[(size_t)k * kDedupAdjStride + sj] = i;
            else atomicOr(&flags[0], 1);
        }
    }
}

// D4 初始化：lab[i]=i（下标域——cur[cur[i]] 索引跳变合法性来源）
__global__ void DedupLabelInitKernel(int* lab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    lab[i] = i;
}

// D4 传播（pull+shortcut，纯 pull 无原子——min 幂等、序无关；双缓冲 ping-pong）：
// next[i]=min(cur[i], cur[cur[i]], min_{j∈adj行(i)} cur[j])——每轮可达距离 ≥ 翻倍。
// 邻接行以 rowlen[i] 为界（D3 对称写、atomic 槽位，行内序不定但 min 序无关）
__global__ void DedupPropagateKernel(const int* __restrict__ cur, int* __restrict__ next,
                                     const int* __restrict__ adj, const int* __restrict__ rowlen,
                                     int n, int* __restrict__ flags /*非空=收敛检查轮*/) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int m = cur[i];
    const int c = cur[m];              // shortcut（值为下标，合法）
    if (c < m) m = c;
    const int len = min(rowlen[i], kDedupAdjStride);
    const int* row = adj + (size_t)i * kDedupAdjStride;
    for (int k = 0; k < len; ++k) {
        const int j = row[k];
        if (cur[j] < m) m = cur[j];
    }
    next[i] = m;
    if (flags != nullptr && m != cur[i]) atomicOr(&flags[1], 1);
}

// 收敛后：root=cur[i]；簇键=分量最小 rrid（atomicMin 结果序无关、唯一——跨运行规范化）
__global__ void DedupRootKeyKernel(const int* __restrict__ cur,
                                   const DedupPt* __restrict__ cvals,
                                   int n, int* __restrict__ rootkey) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicMin(&rootkey[cur[i]], cvals[i].rrid);
}

// D5 簇排序打包：key=(clusterKey<<22)|rrid（clusterKey=rootkey[cur[i]]，Relabel
// 语义内联）；载荷=u32 下标（D6 经 lidx 间接读 cval_b——省 16B 载荷排序）
__global__ void DedupClusterPackKernel(const DedupPt* __restrict__ cvals,
                                       const int* __restrict__ cur,
                                       const int* __restrict__ rootkey,
                                       int n, unsigned long long* keys, unsigned* idxs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = ((unsigned long long)(unsigned)rootkey[cur[i]] << kDedupRidxBits)
            | (unsigned long long)(unsigned)cvals[i].rrid;
    idxs[i] = (unsigned)i;
}

// D6 段边界标志：flag[i]=1 表示 i 是其所在段末位（i 与 i+1 分属不同簇）。
// ExclusiveSum 后 seg[i]=i 之前的边界数=段号（0 基）——⚠ 不可把标志打在"段首"：
// 排他求和不含自身，段首会拿到前一段的段号（v1 隐患：段错位在合成小图上恰好
// 观测不出，真实大 N 下段与簇完全错配）
__global__ void DedupSegBoundaryKernel(const unsigned long long* __restrict__ keys,
                                       int n, int* __restrict__ seg_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    seg_id[i] = (i == n - 1 || (keys[i] >> kDedupRidxBits) != (keys[i + 1] >> kDedupRidxBits))
                ? 1 : 0;
}

// 段首判据（kernel 内复用）：i==0 或段号变化
__device__ __forceinline__ bool DedupIsSegStart(const int* seg, int i) {
    return i == 0 || seg[i] != seg[i - 1];
}
// 段尾+1（段首线程向前走）
__device__ __forceinline__ int DedupSegEnd(const int* seg, int i, int n) {
    int e = i + 1;
    while (e < n && seg[e] == seg[i]) ++e;
    return e;
}

// 段成员读取（D6）：排序后位置 k 的点 = cvals[lidx[k]]（间接一层）
__device__ __forceinline__ DedupPt DedupMember(const DedupPt* __restrict__ cvals,
                                               const unsigned* __restrict__ lidx, int k) {
    return cvals[lidx[k]];
}

// 段合并资格：2≤size≤8 ∧ label 全一致 ∧ 全员切向 valid ∧ 法向跨度≤2.5
// （切向取段首=簇内最小 rrid 成员——CPU mem[0] 口径的确定性规范化；
//   切向按 D1 排序位索引——段首成员的排序位即 lidx[i]）
__device__ __forceinline__ bool DedupMergeEligible(const DedupPt* v,
                                                   const unsigned* __restrict__ lidx,
                                                   const float2* __restrict__ tan_,
                                                   const char* __restrict__ tan_valid,
                                                   int i, int end) {
    int size = end - i;
    if (size < 2 || size > kDedupClusterMax) return false;
    const DedupPt h = DedupMember(v, lidx, i);
    const int lb = h.label;
    if (!tan_valid[lidx[i]]) return false;
    const float2 t = tan_[lidx[i]];
    float nmin = 1e9f, nmax = -1e9f;
    for (int k = i; k < end; ++k) {
        const DedupPt m = DedupMember(v, lidx, k);
        if (m.label != lb || !tan_valid[lidx[k]]) return false;
        const float pr = -m.px * t.y + m.py * t.x;
        nmin = fminf(nmin, pr);
        nmax = fmaxf(nmax, pr);
    }
    return (nmax - nmin) <= kDedupSpanMax;
}

__global__ void DedupSegCountKernel(const DedupPt* v, const unsigned* __restrict__ lidx,
                                    const float2* __restrict__ tan_,
                                    const char* __restrict__ tan_valid,
                                    const int* seg, int n, int* seg_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !DedupIsSegStart(seg, i)) return;
    int end = DedupSegEnd(seg, i, n);
    seg_cnt[seg[i]] = DedupMergeEligible(v, lidx, tan_, tan_valid, i, end) ? 1 : (end - i);
}

__global__ void DedupSegEmitKernel(const DedupPt* v, const unsigned* __restrict__ lidx,
                                   const float2* __restrict__ tan_, const int* seg, int n,
                                   const int* seg_cnt, const int* seg_base,
                                   FastPoint* out, int* out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !DedupIsSegStart(seg, i)) return;
    const int sid = seg[i];
    const int end = DedupSegEnd(seg, i, n);
    const int size = end - i;
    const int base = seg_base[sid];
    if (seg_cnt[sid] == 1 && size >= 2) {      // 合并段：1 点
        const DedupPt h = DedupMember(v, lidx, i);
        const float2 t = tan_[lidx[i]];        // 段首切向（按 D1 排序位索引）
        float px, py;
        if (size == 2) {
            const DedupPt g = DedupMember(v, lidx, i + 1);
            px = 0.5f * (h.px + g.px);
            py = 0.5f * (h.py + g.py);
        } else {                                // ≥3：按法向投影插入排序取中位
            float ns[kDedupClusterMax], xs[kDedupClusterMax], ys[kDedupClusterMax];
            for (int k = 0; k < size; ++k) {
                const DedupPt m = DedupMember(v, lidx, i + k);
                ns[k] = -m.px * t.y + m.py * t.x;
                xs[k] = m.px; ys[k] = m.py;
            }
            for (int a = 1; a < size; ++a) {    // size≤8，寄存器插入排序
                float kn = ns[a], kx = xs[a], ky = ys[a]; int b = a - 1;
                while (b >= 0 && ns[b] > kn) {
                    ns[b + 1] = ns[b]; xs[b + 1] = xs[b]; ys[b + 1] = ys[b]; --b;
                }
                ns[b + 1] = kn; xs[b + 1] = kx; ys[b + 1] = ky;
            }
            const int h2 = size / 2;
            if (size & 1) { px = xs[h2]; py = ys[h2]; }
            else { px = 0.5f * (xs[h2 - 1] + xs[h2]); py = 0.5f * (ys[h2 - 1] + ys[h2]); }
        }
        FastPoint p; p.label = h.label; p.px = px; p.py = py;
        out[base] = p;
        atomicAdd(out_count, 1);
    } else {                                    // 直通段：原样逐点
        for (int k = 0; k < size; ++k) {
            const DedupPt m = DedupMember(v, lidx, i + k);
            FastPoint p; p.label = m.label; p.px = m.px; p.py = m.py;
            out[base + k] = p;
            atomicAdd(out_count, 1);
        }
    }
}

// ============================================================================
// Impl
// ============================================================================

StegerExtractorFast::Impl::Impl(const StegerParams& params)
    : params_(params), old_device_id_(params.deviceId)
{
    params_.validate();

    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }

    if (params_.deviceId >= device_count) {
        throw std::invalid_argument(
            "StegerParams::deviceId=" + std::to_string(params_.deviceId)
            + " exceeds available device count=" + std::to_string(device_count));
    }
    cv::cuda::setDevice(params_.deviceId);

    buildGaussianKernels();
    createEvents();
}

StegerExtractorFast::Impl::~Impl() {
    Destroy();
}

int StegerExtractorFast::Impl::computeKernelSize() const {
    if (params_.kernelSize != 0) return params_.kernelSize;
    int ks = static_cast<int>(std::ceil(params_.sigma * 3.0f)) * 2 + 1;
    return std::max(3, std::min(ks, 31));
}

void StegerExtractorFast::Impl::buildGaussianKernels() {
    actualKernelSize_ = computeKernelSize();
    int half = actualKernelSize_ / 2;
    float sigma2 = params_.sigma * params_.sigma;

    std::vector<float> h_g(actualKernelSize_);
    std::vector<float> h_gx(actualKernelSize_);
    std::vector<float> h_gxx(actualKernelSize_);

    float g_sum = 0.0f;
    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_g[i] = std::exp(-x * x / (2.0f * sigma2));
        g_sum += h_g[i];
    }
    for (int i = 0; i < actualKernelSize_; ++i) {
        h_g[i] /= g_sum;
    }
    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_gx[i] = x / sigma2 * h_g[i];
    }
    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_gxx[i] = (x * x / sigma2 - 1.0f) / sigma2 * h_g[i];
    }

    if (!d_kg_) {
        cudaMalloc(&d_kg_, kMaxKernelSize * sizeof(float));
        cudaMalloc(&d_kgx_, kMaxKernelSize * sizeof(float));
        cudaMalloc(&d_kgxx_, kMaxKernelSize * sizeof(float));
    }
    cudaMemcpy(d_kg_, h_g.data(), actualKernelSize_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kgx_, h_gx.data(), actualKernelSize_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kgxx_, h_gxx.data(), actualKernelSize_ * sizeof(float), cudaMemcpyHostToDevice);

    CALIB_LOG_INFO("Gaussian kernels built: sigma={}, kernelSize={}", params_.sigma, actualKernelSize_);
}

void StegerExtractorFast::Impl::createEvents() {
    if (events_ready_) return;
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_row_);
    cudaEventCreate(&ev_col_);
    cudaEventCreate(&ev_hess_);
    cudaEventCreate(&ev_dedup_);
    cudaEventCreate(&ev_grp_);
    cudaEventCreate(&ev_out_);
    cudaEventCreate(&ev_end_);
    events_ready_ = true;
}

void StegerExtractorFast::Impl::ensureBuffers(int rows, int cols) {
    if (d_tg_.empty() || d_tg_.cols != cols || d_tg_.rows != rows) {
        d_tg_.create(rows, cols, CV_32FC1);
        d_tgx_.create(rows, cols, CV_32FC1);
        d_tgxx_.create(rows, cols, CV_32FC1);
        d_ix_.create(rows, cols, CV_32FC1);
        d_iy_.create(rows, cols, CV_32FC1);
        d_ixx_.create(rows, cols, CV_32FC1);
        d_iyy_.create(rows, cols, CV_32FC1);
        d_ixy_.create(rows, cols, CV_32FC1);
    }

    int64_t cap = static_cast<int64_t>(rows) * cols;
    if (capacity_ < cap) {
        if (d_points_) cudaFree(d_points_);
        if (d_sorted_) cudaFree(d_sorted_);
        if (d_coords_) cudaFree(d_coords_);
        if (d_ids_) cudaFree(d_ids_);
        if (h_pts_) cudaFreeHost(h_pts_);
        if (d_ckey_a_) cudaFree(d_ckey_a_);
        if (d_ckey_b_) cudaFree(d_ckey_b_);
        if (d_cidx_a_) cudaFree(d_cidx_a_);
        if (d_cidx_b_) cudaFree(d_cidx_b_);
        if (d_cval_a_) cudaFree(d_cval_a_);
        if (d_cval_b_) cudaFree(d_cval_b_);
        if (d_tan_) cudaFree(d_tan_);
        if (d_tan_valid_) cudaFree(d_tan_valid_);
        if (d_adj_) cudaFree(d_adj_);
        if (d_rowlen_) cudaFree(d_rowlen_);
        if (d_lab_a_) cudaFree(d_lab_a_);
        if (d_lab_b_) cudaFree(d_lab_b_);
        if (d_rootkey_) cudaFree(d_rootkey_);
        if (d_lkey_a_) cudaFree(d_lkey_a_);
        if (d_lkey_b_) cudaFree(d_lkey_b_);
        if (d_lidx_a_) cudaFree(d_lidx_a_);
        if (d_lidx_b_) cudaFree(d_lidx_b_);
        if (d_seg_id_) cudaFree(d_seg_id_);
        if (d_seg_scan_) cudaFree(d_seg_scan_);
        if (d_seg_cnt_) cudaFree(d_seg_cnt_);
        if (d_seg_base_) cudaFree(d_seg_base_);
        if (d_dedup_points_) cudaFree(d_dedup_points_);
        if (d_cub_temp_) { cudaFree(d_cub_temp_); d_cub_temp_ = nullptr; cub_temp_bytes_ = 0; }

        cudaError_t err = cudaMalloc(&d_points_, cap * sizeof(FastPoint));
        if (err != cudaSuccess) {
            d_points_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_points_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_sorted_, cap * sizeof(FastPoint));
        if (err != cudaSuccess) {
            d_sorted_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_sorted_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_coords_, cap * sizeof(float2));
        if (err != cudaSuccess) {
            d_coords_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_coords_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_ids_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_ids_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_ids_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaHostAlloc(&h_pts_, cap * sizeof(FastPoint), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            h_pts_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaHostAlloc h_pts_ failed: ")
                                     + cudaGetErrorString(err));
        }
        // —— dedup v2 缓冲（按 capacity 一并分配）——
        err = cudaMalloc(&d_ckey_a_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_ckey_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_ckey_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_ckey_b_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_ckey_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_ckey_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cidx_a_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_cidx_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cidx_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cidx_b_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_cidx_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cidx_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cval_a_, cap * sizeof(DedupPt));
        if (err != cudaSuccess) {
            d_cval_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cval_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cval_b_, cap * sizeof(DedupPt));
        if (err != cudaSuccess) {
            d_cval_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cval_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_tan_, cap * sizeof(float2));
        if (err != cudaSuccess) {
            d_tan_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_tan_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_tan_valid_, cap * sizeof(char));
        if (err != cudaSuccess) {
            d_tan_valid_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_tan_valid_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_adj_, cap * static_cast<size_t>(kDedupAdjStride) * sizeof(int));
        if (err != cudaSuccess) {
            d_adj_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_adj_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_rowlen_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_rowlen_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_rowlen_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lab_a_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_lab_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lab_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lab_b_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_lab_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lab_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_rootkey_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_rootkey_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_rootkey_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lkey_a_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_lkey_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lkey_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lkey_b_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_lkey_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lkey_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lidx_a_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_lidx_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lidx_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lidx_b_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_lidx_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lidx_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_id_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_id_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_id_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_scan_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_scan_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_scan_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_cnt_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_cnt_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_cnt_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_base_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_base_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_base_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_dedup_points_, cap * sizeof(FastPoint));
        if (err != cudaSuccess) {
            d_dedup_points_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_dedup_points_ failed: ")
                                     + cudaGetErrorString(err));
        }
        // CUB 临时空间：全部调用（两次 SortPairs＋一次 ExclusiveSum）stream 串行
        // 复用同一块，按 capacity 各空查询一次取 max
        size_t sort1_tmp = 0, sort2_tmp = 0, scan_tmp = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort1_tmp,
            (unsigned long long*)nullptr, (unsigned long long*)nullptr,
            (unsigned*)nullptr, (unsigned*)nullptr,
            static_cast<int>(cap), 0, 64, 0);
        cub::DeviceRadixSort::SortPairs(nullptr, sort2_tmp,
            (unsigned long long*)nullptr, (unsigned long long*)nullptr,
            (unsigned*)nullptr, (unsigned*)nullptr,
            static_cast<int>(cap), 0, 64, 0);
        cub::DeviceScan::ExclusiveSum(nullptr, scan_tmp,
            (int*)nullptr, (int*)nullptr, static_cast<int>(cap), 0);
        cub_temp_bytes_ = std::max(std::max(sort1_tmp, sort2_tmp), scan_tmp);
        err = cudaMalloc(&d_cub_temp_, cub_temp_bytes_);
        if (err != cudaSuccess) {
            d_cub_temp_ = nullptr;
            cub_temp_bytes_ = 0;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cub_temp_ failed: ")
                                     + cudaGetErrorString(err));
        }
        capacity_ = static_cast<int>(cap);
    }

    if (!d_count_) {
        cudaMalloc(&d_count_, sizeof(int));
        cudaMalloc(&d_reject_, sizeof(int));
        cudaMalloc(&d_hist_, kLabelBins * sizeof(int));
        cudaMalloc(&d_base_, kLabelBins * sizeof(int));
        cudaMalloc(&d_cursor_, kLabelBins * sizeof(int));
        cudaMalloc(&d_dedup_count_, sizeof(int));
        cudaMalloc(&d_flags_, 2 * sizeof(int));
        cudaHostAlloc(&h_count_, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_hist_, kLabelBins * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_base_, kLabelBins * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_dedup_count_, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_flags_, 2 * sizeof(int), cudaHostAllocDefault);
    }
}

void StegerExtractorFast::Impl::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() pre-allocating: {}x{}", rows, cols);
    ensureBuffers(rows, cols);

    warmup_rows_ = rows;
    warmup_cols_ = cols;

#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[10-StegerExtractorFast] GPU OOM: Warmup validation failed: ")
            + cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[10-StegerExtractorFast] Warmup() CUDA error: ")
            + cudaGetErrorString(err));
    }
#endif

    warmed_up_ = true;
    CALIB_LOG_INFO("Warmup() completed successfully");
}

void StegerExtractorFast::Impl::SetParams(const StegerParams& params) {
#ifndef NDEBUG
    assert(!inProcess_.load() && "SetParams() called while Execute() is running - NOT thread-safe!");
#endif

    params_ = params;
    params_.validate();

    if (params_.deviceId != old_device_id_) {
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (params_.deviceId >= device_count) {
            throw std::invalid_argument(
                "StegerParams::deviceId=" + std::to_string(params_.deviceId)
                + " exceeds available device count=" + std::to_string(device_count));
        }
        cv::cuda::setDevice(params_.deviceId);
        CALIB_LOG_INFO("SetParams() device switched: {} -> {}", old_device_id_, params_.deviceId);
        old_device_id_ = params_.deviceId;
    }

    buildGaussianKernels();

    CALIB_LOG_INFO("SetParams() updated: sigma={}, kernelSize={}, lowThreshold={}, highThreshold={}, maxLabels={}, deviceId={}",
                   params_.sigma, params_.kernelSize, params_.lowThreshold, params_.highThreshold,
                   params_.maxLabels, params_.deviceId);
}

void StegerExtractorFast::Impl::Destroy() {
    if (d_kg_) { cudaFree(d_kg_); d_kg_ = nullptr; }
    if (d_kgx_) { cudaFree(d_kgx_); d_kgx_ = nullptr; }
    if (d_kgxx_) { cudaFree(d_kgxx_); d_kgxx_ = nullptr; }
    if (d_points_) { cudaFree(d_points_); d_points_ = nullptr; }
    if (d_sorted_) { cudaFree(d_sorted_); d_sorted_ = nullptr; }
    if (d_coords_) { cudaFree(d_coords_); d_coords_ = nullptr; }
    if (d_ids_) { cudaFree(d_ids_); d_ids_ = nullptr; }
    if (d_count_) { cudaFree(d_count_); d_count_ = nullptr; }
    if (d_reject_) { cudaFree(d_reject_); d_reject_ = nullptr; }
    if (d_hist_) { cudaFree(d_hist_); d_hist_ = nullptr; }
    if (d_base_) { cudaFree(d_base_); d_base_ = nullptr; }
    if (d_cursor_) { cudaFree(d_cursor_); d_cursor_ = nullptr; }
    if (h_count_) { cudaFreeHost(h_count_); h_count_ = nullptr; }
    if (h_hist_) { cudaFreeHost(h_hist_); h_hist_ = nullptr; }
    if (h_base_) { cudaFreeHost(h_base_); h_base_ = nullptr; }
    if (h_pts_) { cudaFreeHost(h_pts_); h_pts_ = nullptr; }
    if (d_ckey_a_) { cudaFree(d_ckey_a_); d_ckey_a_ = nullptr; }
    if (d_ckey_b_) { cudaFree(d_ckey_b_); d_ckey_b_ = nullptr; }
    if (d_cidx_a_) { cudaFree(d_cidx_a_); d_cidx_a_ = nullptr; }
    if (d_cidx_b_) { cudaFree(d_cidx_b_); d_cidx_b_ = nullptr; }
    if (d_cval_a_) { cudaFree(d_cval_a_); d_cval_a_ = nullptr; }
    if (d_cval_b_) { cudaFree(d_cval_b_); d_cval_b_ = nullptr; }
    if (d_tan_) { cudaFree(d_tan_); d_tan_ = nullptr; }
    if (d_tan_valid_) { cudaFree(d_tan_valid_); d_tan_valid_ = nullptr; }
    if (d_adj_) { cudaFree(d_adj_); d_adj_ = nullptr; }
    if (d_rowlen_) { cudaFree(d_rowlen_); d_rowlen_ = nullptr; }
    if (d_lab_a_) { cudaFree(d_lab_a_); d_lab_a_ = nullptr; }
    if (d_lab_b_) { cudaFree(d_lab_b_); d_lab_b_ = nullptr; }
    if (d_rootkey_) { cudaFree(d_rootkey_); d_rootkey_ = nullptr; }
    if (d_lkey_a_) { cudaFree(d_lkey_a_); d_lkey_a_ = nullptr; }
    if (d_lkey_b_) { cudaFree(d_lkey_b_); d_lkey_b_ = nullptr; }
    if (d_lidx_a_) { cudaFree(d_lidx_a_); d_lidx_a_ = nullptr; }
    if (d_lidx_b_) { cudaFree(d_lidx_b_); d_lidx_b_ = nullptr; }
    if (d_seg_id_) { cudaFree(d_seg_id_); d_seg_id_ = nullptr; }
    if (d_seg_scan_) { cudaFree(d_seg_scan_); d_seg_scan_ = nullptr; }
    if (d_seg_cnt_) { cudaFree(d_seg_cnt_); d_seg_cnt_ = nullptr; }
    if (d_seg_base_) { cudaFree(d_seg_base_); d_seg_base_ = nullptr; }
    if (d_dedup_points_) { cudaFree(d_dedup_points_); d_dedup_points_ = nullptr; }
    if (d_dedup_count_) { cudaFree(d_dedup_count_); d_dedup_count_ = nullptr; }
    if (h_dedup_count_) { cudaFreeHost(h_dedup_count_); h_dedup_count_ = nullptr; }
    if (d_flags_) { cudaFree(d_flags_); d_flags_ = nullptr; }
    if (h_flags_) { cudaFreeHost(h_flags_); h_flags_ = nullptr; }
    if (cub_temp_bytes_ > 0 && d_cub_temp_) { cudaFree(d_cub_temp_); d_cub_temp_ = nullptr; }
    cub_temp_bytes_ = 0;
    d_tg_.release();
    d_tgx_.release();
    d_tgxx_.release();
    d_ix_.release();
    d_iy_.release();
    d_ixx_.release();
    d_iyy_.release();
    d_ixy_.release();
    if (events_ready_) {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_row_);
        cudaEventDestroy(ev_col_);
        cudaEventDestroy(ev_hess_);
        cudaEventDestroy(ev_dedup_);
        cudaEventDestroy(ev_grp_);
        cudaEventDestroy(ev_out_);
        cudaEventDestroy(ev_end_);
        events_ready_ = false;
    }
    capacity_ = 0;
    warmed_up_ = false;
    warmup_rows_ = 0;
    warmup_cols_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}

StegerResult StegerExtractorFast::Impl::extractFlat(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    return Execute(d_grayImage, d_binaryMask, true, stream);
}

StegerResult StegerExtractorFast::Impl::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_mask,
    bool maskIs8U,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load() && "Concurrent Execute() calls detected - NOT thread-safe!");

    struct ScopedFlag {
        std::atomic<bool>* flag;
        ScopedFlag(std::atomic<bool>* f) : flag(f) {}
        ~ScopedFlag() { flag->store(false); }
    };

    ScopedFlag guard(&inProcess_);
    inProcess_.store(true);
#endif

    StegerResult result;

    try {
        int rows = d_grayImage.rows;
        int cols = d_grayImage.cols;
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

        ensureBuffers(rows, cols);

        cudaEventRecord(ev_start_, cs);

        dim3 conv_block(16, 16);
        dim3 conv_grid((cols + conv_block.x - 1) / conv_block.x,
                       (rows + conv_block.y - 1) / conv_block.y);

        // === Step 1+2a: uchar 直读 → 行级 3 输出 ===
        RowFan3Kernel<<<conv_grid, conv_block, 0, cs>>>(
            d_grayImage.ptr<uchar>(),
            d_tg_.ptr<float>(), d_tgx_.ptr<float>(), d_tgxx_.ptr<float>(),
            rows, cols, d_grayImage.step, d_tg_.step,
            d_kg_, d_kgx_, d_kgxx_, actualKernelSize_);

        cudaEventRecord(ev_row_, cs);

        // === Step 2b: 列级 5 输出导数 ===
        ColFan5Kernel<<<conv_grid, conv_block, 0, cs>>>(
            d_tg_.ptr<float>(), d_tgx_.ptr<float>(), d_tgxx_.ptr<float>(),
            d_ix_.ptr<float>(), d_iy_.ptr<float>(),
            d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
            rows, cols, d_tg_.step, d_ix_.step,
            d_kg_, d_kgx_, d_kgxx_, actualKernelSize_);

        cudaEventRecord(ev_col_, cs);

        cudaError_t op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUDA error in convolution: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 3+4: Hessian + Taylor ===
        cudaMemsetAsync(d_count_, 0, sizeof(int), cs);
        cudaMemsetAsync(d_reject_, 0, sizeof(int), cs);

        const void* mask_ptr = maskIs8U
            ? static_cast<const void*>(d_mask.ptr<uchar>())
            : static_cast<const void*>(d_mask.ptr<int>());

        if (maskIs8U) {
            HessianEigenAndTaylorKernel<true><<<conv_grid, conv_block, 0, cs>>>(
                d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                mask_ptr, d_mask.step,
                d_tg_.ptr<float>(), d_tg_.step,
                rows, cols,
                d_ix_.step, d_ixx_.step,
                params_.lowThreshold, params_.highThreshold,
                params_.maxLineWidth,
                d_points_, d_count_, d_reject_, capacity_);
        } else {
            HessianEigenAndTaylorKernel<false><<<conv_grid, conv_block, 0, cs>>>(
                d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                mask_ptr, d_mask.step,
                d_tg_.ptr<float>(), d_tg_.step,
                rows, cols,
                d_ix_.step, d_ixx_.step,
                params_.lowThreshold, params_.highThreshold,
                params_.maxLineWidth,
                d_points_, d_count_, d_reject_, capacity_);
        }

        cudaEventRecord(ev_hess_, cs);

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUDA error in Hessian/Taylor kernel: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 5: 计数回读 + 直方图 + 前缀和 + 计数直写分组 ===
        cudaMemcpyAsync(h_count_, d_count_, sizeof(int), cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(&h_reject_, d_reject_, sizeof(int), cudaMemcpyDeviceToHost, cs);
        stream.waitForCompletion();
        int point_count = std::min(*h_count_, capacity_);

        CALIB_LOG_INFO("Extracted {} subpixel points{}", point_count,
                       params_.maxLineWidth > 0.0f
                           ? std::string(" (widthGate rejected ")
                                 + std::to_string(h_reject_) + ")"
                           : std::string());

        if (point_count == 0) {
            result.success = true;
            result.message = "No subpixel points extracted";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        // === Step 4.5: 平顶去重（dedupMode=1；Off 完全旁路——逐位一致红线）===
        FastPoint* src_points = d_points_;
        int point_count_pre = point_count;
        bool dedup_ran = false;
        bool dedup_fallback = false;
        bool dedup_overflow = false, dedup_converged = false;
        if (params_.dedupMode != 0 && point_count > 1) {
            if (static_cast<int64_t>(rows) * cols >= (1 << 22)
                || rows > 4096 || cols > 4096) {
                result.success = false;
                result.message = "dedup: image exceeds key-packing limits "
                                 "(dims<=4096, pixels<4M)";
                return result;
            }
            cudaMemsetAsync(d_dedup_count_, 0, sizeof(int), cs);
            cudaMemsetAsync(d_flags_, 0, 2 * sizeof(int), cs);
            cudaMemsetAsync(d_seg_cnt_, 0,
                            static_cast<size_t>(point_count) * sizeof(int), cs);
            cudaMemsetAsync(d_rowlen_, 0,   // 邻接行长度计数（行内容以长度为界）
                            static_cast<size_t>(point_count) * sizeof(int), cs);
            cudaMemsetAsync(d_rootkey_, 0x7F,   // 大值哨兵（0x7F7F7F7F > 任意 rrid）
                            static_cast<size_t>(point_count) * sizeof(int), cs);
            {
                const int threads = 256;
                const int blocks = (point_count + threads - 1) / threads;

                // D1 网格排序：(cell|rrid) 唯一键 → CUB RadixSort（输入原子序被全序吸收；
                // 键 < 2^46（cell 24bit + rrid 22bit）→ 只排低 46bit；u32 下标载荷，
                // 排序后 gather 物化 16B DedupPt）
                DedupCellPackKernel<<<blocks, threads, 0, cs>>>(
                    d_points_, point_count, cols, d_ckey_a_, d_cidx_a_, d_cval_a_);
                size_t tmp = cub_temp_bytes_;
                op_err = cub::DeviceRadixSort::SortPairs(d_cub_temp_, tmp,
                    d_ckey_a_, d_ckey_b_, d_cidx_a_, d_cidx_b_,
                    point_count, 0, 46, cs);
                if (op_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUDA error in dedup chain: ") + cudaGetErrorString(op_err);
                    return result;
                }
                DedupGatherKernel<<<blocks, threads, 0, cs>>>(
                    d_cidx_b_, d_cval_a_, d_cval_b_, point_count);

                // D2 邻域 PCA 几何切向（按 rrid 写 tan/tan_valid）
                DedupTangentKernel<<<blocks, threads, 0, cs>>>(
                    d_ckey_b_, d_cval_b_, point_count, d_tan_, d_tan_valid_);

                // D3 位移门配对 → 对称邻接行（行满置 flags[0]）
                DedupPairEdgeKernel<<<blocks, threads, 0, cs>>>(
                    d_ckey_b_, d_cval_b_, d_tan_, d_tan_valid_,
                    point_count, d_adj_, d_rowlen_, d_flags_);

                // D4 标签传播（下标域 pull+shortcut，纯 pull 无原子；双缓冲 ping-pong）。
                // 自适应轮次：真实簇直径小（对/三元组，2-3 轮收敛）——前 6 轮＋检查轮
                // 后回读 flag 早退；未收敛续跑满 ⌈log₂N⌉+1 轮再终检（最坏 rounds+1 次
                // 传播——检查轮亦推进一轮，幂等无害；输出与轮次无关——收敛=不动点）。
                // 邻接溢出（flags[0] 由 D3 置位）在首次回读即短路退出，D5/D6 不再空跑
                DedupLabelInitKernel<<<blocks, threads, 0, cs>>>(d_lab_a_, point_count);
                int* lab_cur = d_lab_a_;
                int* lab_nxt = d_lab_b_;
                int rounds = 1;
                while ((1 << rounds) < point_count) ++rounds;
                ++rounds;   // +1 余量（N<4M ⇒ ≤23）
                const int early = rounds < 7 ? rounds - 1 : 6;
                for (int r = 0; r < early; ++r) {
                    DedupPropagateKernel<<<blocks, threads, 0, cs>>>(
                        lab_cur, lab_nxt, d_adj_, d_rowlen_, point_count, nullptr);
                    int* sw = lab_cur; lab_cur = lab_nxt; lab_nxt = sw;
                }
                DedupPropagateKernel<<<blocks, threads, 0, cs>>>(   // 检查轮（亦推进一轮）
                    lab_cur, lab_nxt, d_adj_, d_rowlen_, point_count, d_flags_);
                int* sw = lab_cur; lab_cur = lab_nxt; lab_nxt = sw;
                cudaMemcpyAsync(h_flags_, d_flags_, 2 * sizeof(int),
                                cudaMemcpyDeviceToHost, cs);
                stream.waitForCompletion();
                dedup_overflow = (h_flags_[0] != 0);
                dedup_converged = (h_flags_[1] == 0);
                if (!dedup_overflow && !dedup_converged) {
                    cudaMemsetAsync(d_flags_ + 1, 0, sizeof(int), cs);
                    for (int r = early + 1; r < rounds; ++r) {
                        DedupPropagateKernel<<<blocks, threads, 0, cs>>>(
                            lab_cur, lab_nxt, d_adj_, d_rowlen_, point_count, nullptr);
                        int* sw2 = lab_cur; lab_cur = lab_nxt; lab_nxt = sw2;
                    }
                    DedupPropagateKernel<<<blocks, threads, 0, cs>>>(   // 终检轮
                        lab_cur, lab_nxt, d_adj_, d_rowlen_, point_count, d_flags_);
                    int* sw3 = lab_cur; lab_cur = lab_nxt; lab_nxt = sw3;
                    cudaMemcpyAsync(h_flags_ + 1, &d_flags_[1], sizeof(int),
                                    cudaMemcpyDeviceToHost, cs);
                    stream.waitForCompletion();
                    dedup_converged = (h_flags_[1] == 0);
                }
                if (!dedup_overflow) {
                    // 簇键=分量最小 rrid（atomicMin 序无关、唯一）
                    DedupRootKeyKernel<<<blocks, threads, 0, cs>>>(
                        lab_cur, d_cval_b_, point_count, d_rootkey_);

                    // D5 簇排序：((簇键)<<22|rrid) 全序 → 第二次 RadixSort
                    // （簇键=min rrid < 2^22 → 键 < 2^44，只排低 44bit；u32 下标载荷）
                    DedupClusterPackKernel<<<blocks, threads, 0, cs>>>(
                        d_cval_b_, lab_cur, d_rootkey_, point_count, d_lkey_a_, d_lidx_a_);
                    size_t tmp2 = cub_temp_bytes_;
                    op_err = cub::DeviceRadixSort::SortPairs(d_cub_temp_, tmp2,
                        d_lkey_a_, d_lkey_b_, d_lidx_a_, d_lidx_b_,
                        point_count, 0, 44, cs);
                    if (op_err != cudaSuccess) {
                        result.success = false;
                        result.message = std::string("CUDA error in dedup chain: ") + cudaGetErrorString(op_err);
                        return result;
                    }

                // D6 段归约（段=clusterKey 连续段；2 均值/≥3 法向投影中位）。
                // ExclusiveSum 输入/输出不得别名——CUB 多 tile 扫描下别名损坏结果
                // （单 tile 小 N 恰好无害——σ4 合成用例曾因此给出假绿）
                DedupSegBoundaryKernel<<<blocks, threads, 0, cs>>>(
                    d_lkey_b_, point_count, d_seg_id_);
                size_t tmp3 = cub_temp_bytes_;
                op_err = cub::DeviceScan::ExclusiveSum(d_cub_temp_, tmp3,
                    d_seg_id_, d_seg_scan_, point_count, cs);
                if (op_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUDA error in dedup chain: ") + cudaGetErrorString(op_err);
                    return result;
                }
                DedupSegCountKernel<<<blocks, threads, 0, cs>>>(
                    d_cval_b_, d_lidx_b_, d_tan_, d_tan_valid_, d_seg_scan_,
                    point_count, d_seg_cnt_);
                size_t tmp4 = cub_temp_bytes_;
                op_err = cub::DeviceScan::ExclusiveSum(d_cub_temp_, tmp4,
                    d_seg_cnt_, d_seg_base_, point_count, cs);
                if (op_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUDA error in dedup chain: ") + cudaGetErrorString(op_err);
                    return result;
                }
                DedupSegEmitKernel<<<blocks, threads, 0, cs>>>(
                    d_cval_b_, d_lidx_b_, d_tan_, d_seg_scan_, point_count,
                    d_seg_cnt_, d_seg_base_, d_dedup_points_, d_dedup_count_);
                }
            }
            op_err = cudaGetLastError();
            if (op_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUDA error in dedup chain: ") + cudaGetErrorString(op_err);
                return result;
            }
            cudaEventRecord(ev_dedup_, cs);
            dedup_ran = true;
            if (dedup_overflow || !dedup_converged) {
                // 邻接溢出/传播未收敛 → 本帧保守直通（坐标零改动、mergedCount=0）
                dedup_fallback = true;
                src_points = d_points_;
                point_count = point_count_pre;
                CALIB_LOG_WARN("dedup fallback ({}): frame passed through unmerged",
                               dedup_overflow ? "adjacency overflow"
                                              : "propagation not converged");
            } else {
                // N' 回读（须先于 hist：hist 只数前 N' 个有效点，链后立即 D2H＋wait
                // ——v1 修正后的时序沿用）
                cudaMemcpyAsync(h_dedup_count_, d_dedup_count_, sizeof(int),
                                cudaMemcpyDeviceToHost, cs);
                stream.waitForCompletion();
                src_points = d_dedup_points_;
                point_count = *h_dedup_count_;
                result.mergedCount = point_count_pre - point_count;
                CALIB_LOG_INFO("Extracted {} subpixel points after dedup (merged {})",
                               point_count, result.mergedCount);
                if (point_count == 0) {
                    result.success = true;
                    result.message = "No subpixel points after dedup";
                    result.qualityFlag = QualityFlag::Warning;
                    return result;
                }
            }
        }

        cudaMemsetAsync(d_hist_, 0, kLabelBins * sizeof(int), cs);
        {
            int threads = 256;
            int blocks = (point_count + threads - 1) / threads;
            LabelHistKernel<<<blocks, threads, 0, cs>>>(src_points, point_count, kLabelBins, d_hist_);
        }
        cudaMemcpyAsync(h_hist_, d_hist_, kLabelBins * sizeof(int), cudaMemcpyDeviceToHost, cs);
        stream.waitForCompletion();

        int running = 0;
        for (int lb = 0; lb < kLabelBins; ++lb) {
            h_base_[lb] = running;
            running += h_hist_[lb];
        }

        cudaMemcpyAsync(d_base_, h_base_, kLabelBins * sizeof(int), cudaMemcpyHostToDevice, cs);
        cudaMemsetAsync(d_cursor_, 0, kLabelBins * sizeof(int), cs);
        {
            int threads = 256;
            int blocks = (point_count + threads - 1) / threads;
            LabelScatterKernel<<<blocks, threads, 0, cs>>>(
                src_points, point_count, kLabelBins, d_base_, d_cursor_,
                d_sorted_, d_coords_, d_ids_);
        }

        cudaEventRecord(ev_grp_, cs);

        // === Step 6: 单次 D2H + D2D 组装 CPU/GPU 双输出 ===
        auto d_cp = std::make_shared<cv::cuda::GpuMat>();
        d_cp->create(1, point_count, CV_32FC2);
        cudaMemcpyAsync(d_cp->ptr<float2>(), d_coords_, point_count * sizeof(float2),
                        cudaMemcpyDeviceToDevice, cs);

        auto d_pl = std::make_shared<cv::cuda::GpuMat>();
        d_pl->create(1, point_count, CV_32SC1);
        cudaMemcpyAsync(d_pl->ptr<int>(), d_ids_, point_count * sizeof(int),
                        cudaMemcpyDeviceToDevice, cs);

        cudaMemcpyAsync(h_pts_, d_sorted_, point_count * sizeof(FastPoint),
                        cudaMemcpyDeviceToHost, cs);

        cudaEventRecord(ev_out_, cs);
        stream.waitForCompletion();

        result.d_centerPoints = d_cp;
        result.d_line_ids = d_pl;

        std::map<int, std::vector<cv::Point2f>> centerPoints;
        for (int lb = 1; lb < kLabelBins; ++lb) {
            int n = h_hist_[lb];
            if (n <= 0) continue;
            const FastPoint* src = h_pts_ + h_base_[lb];
            std::vector<cv::Point2f> v(n);
            for (int i = 0; i < n; ++i) {
                v[i] = cv::Point2f(src[i].px, src[i].py);
            }
            centerPoints.emplace(lb, std::move(v));
        }

        cudaEventRecord(ev_end_, cs);
        cudaEventSynchronize(ev_end_);

        float ms_total = 0, ms_row = 0, ms_col = 0, ms_hess = 0, ms_grp = 0, ms_out = 0;
        float ms_dedup = 0;
        cudaEventElapsedTime(&ms_total, ev_start_, ev_end_);
        cudaEventElapsedTime(&ms_row, ev_start_, ev_row_);
        cudaEventElapsedTime(&ms_col, ev_row_, ev_col_);
        cudaEventElapsedTime(&ms_hess, ev_col_, ev_hess_);
        if (dedup_ran) cudaEventElapsedTime(&ms_dedup, ev_hess_, ev_dedup_);
        cudaEventElapsedTime(&ms_grp, ev_hess_, ev_grp_);
        cudaEventElapsedTime(&ms_out, ev_grp_, ev_end_);

        CALIB_LOG_INFO("[BENCH_FAST] {}x{} | Total={:.2f}ms | Row3={:.2f}ms | Col5={:.2f}ms | Hessian={:.2f}ms | Dedup={:.2f}ms | Group={:.2f}ms | Out={:.2f}ms | N={} | merged={}",
                       cols, rows, ms_total, ms_row, ms_col, ms_hess, ms_dedup,
                       ms_grp, ms_out, point_count, result.mergedCount);

        result.centerPoints = std::move(centerPoints);
        result.totalPointCount = point_count;
        result.lineCount = static_cast<int>(result.centerPoints.size());
        result.success = true;

        int64_t total_pixels = static_cast<int64_t>(rows) * cols;
        if (point_count > total_pixels / 2) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "Too many points extracted (>50% of pixels), possible noise";
        } else {
            bool has_degraded = false;
            for (const auto& [label, pts] : result.centerPoints) {
                if (static_cast<int>(pts.size()) < 10) {
                    has_degraded = true;
                    break;
                }
            }
            if (has_degraded) {
                result.qualityFlag = calib::QualityFlag::Degraded;
                result.message = "Some lines have fewer than 10 points";
            } else {
                result.qualityFlag = calib::QualityFlag::Normal;
                result.message = "Subpixel extraction successful";
            }
        }

        if (dedup_fallback) {
            result.qualityFlag = QualityFlag::Warning;
            result.message += std::string(" | dedup fallback: ")
                + (h_flags_[0] != 0 ? "adjacency overflow" : "propagation not converged")
                + " — frame passed through unmerged";
        }

        CALIB_LOG_DEBUG("Execute() completed: {} points, {} lines, qualityFlag={}",
                        result.totalPointCount, result.lineCount,
                        static_cast<int>(result.qualityFlag));

    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        CALIB_LOG_ERROR("Execute() OpenCV exception: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        CALIB_LOG_ERROR("Execute() exception: {}", e.what());
    }

    return result;
}
