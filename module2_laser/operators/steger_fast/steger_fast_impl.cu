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
    int rows, int cols,
    size_t ix_step, size_t ixx_step,
    float low_thresh, float high_thresh,
    FastPoint* d_out_points, int* d_out_count,
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
        capacity_ = static_cast<int>(cap);
    }

    if (!d_count_) {
        cudaMalloc(&d_count_, sizeof(int));
        cudaMalloc(&d_hist_, kLabelBins * sizeof(int));
        cudaMalloc(&d_base_, kLabelBins * sizeof(int));
        cudaMalloc(&d_cursor_, kLabelBins * sizeof(int));
        cudaHostAlloc(&h_count_, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_hist_, kLabelBins * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_base_, kLabelBins * sizeof(int), cudaHostAllocDefault);
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
    if (d_hist_) { cudaFree(d_hist_); d_hist_ = nullptr; }
    if (d_base_) { cudaFree(d_base_); d_base_ = nullptr; }
    if (d_cursor_) { cudaFree(d_cursor_); d_cursor_ = nullptr; }
    if (h_count_) { cudaFreeHost(h_count_); h_count_ = nullptr; }
    if (h_hist_) { cudaFreeHost(h_hist_); h_hist_ = nullptr; }
    if (h_base_) { cudaFreeHost(h_base_); h_base_ = nullptr; }
    if (h_pts_) { cudaFreeHost(h_pts_); h_pts_ = nullptr; }
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

        const void* mask_ptr = maskIs8U
            ? static_cast<const void*>(d_mask.ptr<uchar>())
            : static_cast<const void*>(d_mask.ptr<int>());

        if (maskIs8U) {
            HessianEigenAndTaylorKernel<true><<<conv_grid, conv_block, 0, cs>>>(
                d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                mask_ptr, d_mask.step,
                rows, cols,
                d_ix_.step, d_ixx_.step,
                params_.lowThreshold, params_.highThreshold,
                d_points_, d_count_, capacity_);
        } else {
            HessianEigenAndTaylorKernel<false><<<conv_grid, conv_block, 0, cs>>>(
                d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                mask_ptr, d_mask.step,
                rows, cols,
                d_ix_.step, d_ixx_.step,
                params_.lowThreshold, params_.highThreshold,
                d_points_, d_count_, capacity_);
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
        stream.waitForCompletion();
        int point_count = std::min(*h_count_, capacity_);

        CALIB_LOG_INFO("Extracted {} subpixel points", point_count);

        if (point_count == 0) {
            result.success = true;
            result.message = "No subpixel points extracted";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        cudaMemsetAsync(d_hist_, 0, kLabelBins * sizeof(int), cs);
        {
            int threads = 256;
            int blocks = (point_count + threads - 1) / threads;
            LabelHistKernel<<<blocks, threads, 0, cs>>>(d_points_, point_count, kLabelBins, d_hist_);
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
                d_points_, point_count, kLabelBins, d_base_, d_cursor_,
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
        cudaEventElapsedTime(&ms_total, ev_start_, ev_end_);
        cudaEventElapsedTime(&ms_row, ev_start_, ev_row_);
        cudaEventElapsedTime(&ms_col, ev_row_, ev_col_);
        cudaEventElapsedTime(&ms_hess, ev_col_, ev_hess_);
        cudaEventElapsedTime(&ms_grp, ev_hess_, ev_grp_);
        cudaEventElapsedTime(&ms_out, ev_grp_, ev_end_);

        CALIB_LOG_INFO("[BENCH_FAST] {}x{} | Total={:.2f}ms | Row3={:.2f}ms | Col5={:.2f}ms | Hessian={:.2f}ms | Group={:.2f}ms | Out={:.2f}ms | N={}",
                       cols, rows, ms_total, ms_row, ms_col, ms_hess, ms_grp, ms_out, point_count);

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
