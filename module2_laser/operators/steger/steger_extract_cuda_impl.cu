/**
 * @file steger_extract_cuda_impl.cu
 * @brief Steger激光中心亚像素提取算子 CUDA 实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 输入转换 CV_8UC1 → CV_32FC1
 *   Step 2: 行列分离高斯卷积计算 Ix, Iy, Ixx, Iyy, Ixy
 *   Step 3: 对标签非零像素计算 Hessian 特征方向 + 特征值阈值过滤
 *   Step 4: 泰勒展开亚像素修正
 *   Step 5: Thrust 排序归组
 *   Step 6: D2H 拷贝构建结果
 */

#include "steger_extract_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(10, StegerExtractorCUDA);

struct SubpixelPointLabelComp {
    __host__ __device__ bool operator()(const SubpixelPoint& a, const SubpixelPoint& b) const {
        return a.label < b.label;
    }
};

struct SubpixelPointToCoord {
    __host__ __device__ float2 operator()(const SubpixelPoint& p) const {
        return make_float2(p.px, p.py);
    }
};

struct SubpixelPointToLabel {
    __host__ __device__ int operator()(const SubpixelPoint& p) const {
        return p.label;
    }
};

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void GaussianConvRowKernel(
    const float* d_input, float* d_output,
    int rows, int cols, size_t in_step, size_t out_step,
    const float* d_kernel, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int half = kernel_size / 2;
    float sum = 0.0f;
    float w_sum = 0.0f;

    for (int k = -half; k <= half; ++k) {
        int xx = x + k;
        if (xx >= 0 && xx < cols) {
            float w = d_kernel[k + half];
            const float* in_row = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_input) + y * in_step);
            sum += w * in_row[xx];
            w_sum += w;
        }
    }

    float* out_row = reinterpret_cast<float*>(
        reinterpret_cast<char*>(d_output) + y * out_step);
    out_row[x] = sum / w_sum;
}

__global__ void GaussianConvColKernel(
    const float* d_input, float* d_output,
    int rows, int cols, size_t in_step, size_t out_step,
    const float* d_kernel, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int half = kernel_size / 2;
    float sum = 0.0f;
    float w_sum = 0.0f;

    for (int k = -half; k <= half; ++k) {
        int yy = y + k;
        if (yy >= 0 && yy < rows) {
            float w = d_kernel[k + half];
            const float* in_row = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_input) + yy * in_step);
            sum += w * in_row[x];
            w_sum += w;
        }
    }

    float* out_row = reinterpret_cast<float*>(
        reinterpret_cast<char*>(d_output) + y * out_step);
    out_row[x] = sum / w_sum;
}

__global__ void Convert8UTo32FKernel(
    const unsigned char* d_input, float* d_output,
    int rows, int cols, size_t in_step, size_t out_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const unsigned char* in_row = d_input + y * in_step;
    float* out_row = reinterpret_cast<float*>(
        reinterpret_cast<char*>(d_output) + y * out_step);
    out_row[x] = static_cast<float>(in_row[x]);
}

__global__ void binaryToIntLabelKernel(
    const unsigned char* __restrict__ d_bin,
    int* __restrict__ d_intlabel,
    int rows, int cols,
    size_t bin_step, size_t int_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    const unsigned char* bin_row = d_bin + y * bin_step;
    int* int_row = reinterpret_cast<int*>(
        reinterpret_cast<char*>(d_intlabel) + y * int_step);
    int_row[x] = (bin_row[x] > 0) ? 1 : 0;
}

__global__ void HessianEigenAndTaylorKernel(
    const float* d_ix,  const float* d_iy,
    const float* d_ixx, const float* d_iyy, const float* d_ixy,
    const int* d_labels,
    int rows, int cols,
    size_t ix_step, size_t ixx_step, size_t label_step,
    float low_thresh, float high_thresh,
    SubpixelPoint* d_out_points, int* d_out_count,
    int max_points)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const int* label_row = reinterpret_cast<const int*>(
        reinterpret_cast<const char*>(d_labels) + y * label_step);
    int label = label_row[x];
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

// ============================================================================
// Impl 构造函数
// ============================================================================

StegerExtractorCUDA::Impl::Impl(const StegerParams& params)
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

    d_point_count_.resize(1);
    thrust::fill(d_point_count_.begin(), d_point_count_.end(), 0);

    buildGaussianKernels();
}

// ============================================================================
// 高斯核构建
// ============================================================================

int StegerExtractorCUDA::Impl::computeKernelSize() const {
    if (params_.kernelSize != 0) return params_.kernelSize;
    int ks = static_cast<int>(std::ceil(params_.sigma * 3.0f)) * 2 + 1;
    return std::max(3, std::min(ks, 31));
}

void StegerExtractorCUDA::Impl::buildGaussianKernels() {
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
        h_gx[i] = -x / sigma2 * h_g[i];
    }

    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_gxx[i] = (x * x / sigma2 - 1.0f) / sigma2 * h_g[i];
    }

    d_g_kernel_.assign(h_g.begin(), h_g.end());
    d_gx_kernel_.assign(h_gx.begin(), h_gx.end());
    d_gxx_kernel_.assign(h_gxx.begin(), h_gxx.end());

    CALIB_LOG_INFO("Gaussian kernels built: sigma={}, kernelSize={}", params_.sigma, actualKernelSize_);
}

// ============================================================================
// warmup()
// ============================================================================

void StegerExtractorCUDA::Impl::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() pre-allocating: {}x{}", rows, cols);

    d_float_.create(rows, cols, CV_32FC1);
    d_ix_.create(rows, cols, CV_32FC1);
    d_iy_.create(rows, cols, CV_32FC1);
    d_ixx_.create(rows, cols, CV_32FC1);
    d_iyy_.create(rows, cols, CV_32FC1);
    d_ixy_.create(rows, cols, CV_32FC1);
    d_temp_.create(rows, cols, CV_32FC1);
    d_int_mask_.create(rows, cols, CV_32SC1);

    int max_points = rows * cols;
    d_points_.resize(max_points);
    d_point_count_.resize(1);

    warmup_rows_ = rows;
    warmup_cols_ = cols;

#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[10-StegerExtractorCUDA] GPU OOM: Warmup validation failed: ")
            + cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[10-StegerExtractorCUDA] Warmup() CUDA error: ")
            + cudaGetErrorString(err));
    }
#endif

    warmed_up_ = true;
    CALIB_LOG_INFO("Warmup() completed successfully");
}

// ============================================================================
// SetParams()
// ============================================================================

void StegerExtractorCUDA::Impl::SetParams(const StegerParams& params) {
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

// ============================================================================
// Destroy()
// ============================================================================

void StegerExtractorCUDA::Impl::Destroy() {
    d_g_kernel_.clear();
    d_gx_kernel_.clear();
    d_gxx_kernel_.clear();
    d_ix_.release();
    d_iy_.release();
    d_ixx_.release();
    d_iyy_.release();
    d_ixy_.release();
    d_temp_.release();
    d_float_.release();
    d_int_mask_.release();
    d_points_.clear();
    d_points_.shrink_to_fit();
    d_point_count_.clear();
    d_point_count_.shrink_to_fit();
    warmed_up_ = false;
    warmup_rows_ = 0;
    warmup_cols_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}

// ============================================================================
// extractFlat() - binary mask → int label (internal helper for Flat mode)
// ============================================================================

StegerResult StegerExtractorCUDA::Impl::extractFlat(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    StegerResult result;

    try {
        int rows = d_grayImage.rows;
        int cols = d_grayImage.cols;
        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        if (d_int_mask_.empty() || d_int_mask_.cols != cols || d_int_mask_.rows != rows) {
            d_int_mask_.create(rows, cols, CV_32SC1);
        }

        dim3 conv_block(16, 16);
        dim3 conv_grid((cols + conv_block.x - 1) / conv_block.x,
                       (rows + conv_block.y - 1) / conv_block.y);

        binaryToIntLabelKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_binaryMask.ptr<uchar>(), d_int_mask_.ptr<int>(),
            rows, cols, d_binaryMask.step, d_int_mask_.step);

        cudaError_t op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUDA error in binaryToIntLabelKernel: ") + cudaGetErrorString(op_err);
            return result;
        }

    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        CALIB_LOG_ERROR("extractFlat() OpenCV exception: {}", e.what());
        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        CALIB_LOG_ERROR("extractFlat() exception: {}", e.what());
        return result;
    }

    return Execute(d_grayImage, d_int_mask_, stream);
}

// ============================================================================
// Execute() - 核心方法
// ============================================================================

StegerResult StegerExtractorCUDA::Impl::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load() && "Concurrent extract() calls detected - NOT thread-safe!");

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
        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        cudaEvent_t ev_start, ev_s1, ev_s2, ev_s3, ev_s5, ev_s6, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_s1);
        cudaEventCreate(&ev_s2);
        cudaEventCreate(&ev_s3);
        cudaEventCreate(&ev_s5);
        cudaEventCreate(&ev_s6);
        cudaEventCreate(&ev_end);
        cudaEventRecord(ev_start, cuda_stream);

        // === Step 1: CV_8UC1 → CV_32FC1 ===
        if (d_float_.empty() || d_float_.cols != cols || d_float_.rows != rows) {
            d_float_.create(rows, cols, CV_32FC1);
            d_ix_.create(rows, cols, CV_32FC1);
            d_iy_.create(rows, cols, CV_32FC1);
            d_ixx_.create(rows, cols, CV_32FC1);
            d_iyy_.create(rows, cols, CV_32FC1);
            d_ixy_.create(rows, cols, CV_32FC1);
            d_temp_.create(rows, cols, CV_32FC1);
        }

        dim3 conv_block(16, 16);
        dim3 conv_grid((cols + conv_block.x - 1) / conv_block.x,
                       (rows + conv_block.y - 1) / conv_block.y);

        Convert8UTo32FKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_grayImage.ptr<uchar>(), d_float_.ptr<float>(),
            rows, cols, d_grayImage.step, d_float_.step);

        cudaEventRecord(ev_s1, cuda_stream);

        // === Step 2: Separable Gaussian Convolution ===
        // 2a: Ix = (g' ⊗_row g) * I
        GaussianConvRowKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_float_.ptr<float>(), d_temp_.ptr<float>(),
            rows, cols, d_float_.step, d_temp_.step,
            thrust::raw_pointer_cast(d_gx_kernel_.data()), actualKernelSize_);

        GaussianConvColKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_temp_.ptr<float>(), d_ix_.ptr<float>(),
            rows, cols, d_temp_.step, d_ix_.step,
            thrust::raw_pointer_cast(d_g_kernel_.data()), actualKernelSize_);

        // 2b: Iy = g * (g' ⊗_col I)
        GaussianConvRowKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_float_.ptr<float>(), d_temp_.ptr<float>(),
            rows, cols, d_float_.step, d_temp_.step,
            thrust::raw_pointer_cast(d_g_kernel_.data()), actualKernelSize_);

        GaussianConvColKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_temp_.ptr<float>(), d_iy_.ptr<float>(),
            rows, cols, d_temp_.step, d_iy_.step,
            thrust::raw_pointer_cast(d_gx_kernel_.data()), actualKernelSize_);

        // 2c: Ixx = (g'' ⊗_row g) * I
        GaussianConvRowKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_float_.ptr<float>(), d_temp_.ptr<float>(),
            rows, cols, d_float_.step, d_temp_.step,
            thrust::raw_pointer_cast(d_gxx_kernel_.data()), actualKernelSize_);

        GaussianConvColKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_temp_.ptr<float>(), d_ixx_.ptr<float>(),
            rows, cols, d_temp_.step, d_ixx_.step,
            thrust::raw_pointer_cast(d_g_kernel_.data()), actualKernelSize_);

        // 2d: Iyy = g * (g'' ⊗_col I)
        GaussianConvRowKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_float_.ptr<float>(), d_temp_.ptr<float>(),
            rows, cols, d_float_.step, d_temp_.step,
            thrust::raw_pointer_cast(d_g_kernel_.data()), actualKernelSize_);

        GaussianConvColKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_temp_.ptr<float>(), d_iyy_.ptr<float>(),
            rows, cols, d_temp_.step, d_iyy_.step,
            thrust::raw_pointer_cast(d_gxx_kernel_.data()), actualKernelSize_);

        // 2e: Ixy = (g' ⊗_row g') * I
        GaussianConvRowKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_float_.ptr<float>(), d_temp_.ptr<float>(),
            rows, cols, d_float_.step, d_temp_.step,
            thrust::raw_pointer_cast(d_gx_kernel_.data()), actualKernelSize_);

        GaussianConvColKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_temp_.ptr<float>(), d_ixy_.ptr<float>(),
            rows, cols, d_temp_.step, d_ixy_.step,
            thrust::raw_pointer_cast(d_gx_kernel_.data()), actualKernelSize_);

        cudaEventRecord(ev_s2, cuda_stream);

        cudaError_t op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUDA error in convolution: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 3+4: Hessian Eigen + Taylor Subpixel ===
        int max_points = rows * cols;
        if (static_cast<int>(d_points_.size()) < max_points) {
            d_points_.resize(max_points);
        }
        thrust::fill(thrust::cuda::par_nosync.on(cuda_stream),
                     d_point_count_.begin(), d_point_count_.end(), 0);

        HessianEigenAndTaylorKernel<<<conv_grid, conv_block, 0, cuda_stream>>>(
            d_ix_.ptr<float>(), d_iy_.ptr<float>(),
            d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
            d_labeledMask.ptr<int>(),
            rows, cols,
            d_ix_.step, d_ixx_.step, d_labeledMask.step,
            params_.lowThreshold, params_.highThreshold,
            thrust::raw_pointer_cast(d_points_.data()),
            thrust::raw_pointer_cast(d_point_count_.data()),
            max_points);

        cudaEventRecord(ev_s3, cuda_stream);

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUDA error in Hessian/Taylor kernel: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 5: Get point count + sort by label ===
        thrust::host_vector<int> h_count(1);
        thrust::copy(d_point_count_.begin(), d_point_count_.end(), h_count.begin());
        int point_count = std::min(h_count[0], max_points);

        CALIB_LOG_INFO("Extracted {} subpixel points", point_count);

        if (point_count == 0) {
            result.success = true;
            result.message = "No subpixel points extracted";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        thrust::device_vector<SubpixelPoint> d_valid_points(
            d_points_.begin(), d_points_.begin() + point_count);

        thrust::stable_sort(
            thrust::cuda::par_nosync.on(cuda_stream),
            d_valid_points.begin(), d_valid_points.end(),
            SubpixelPointLabelComp());

        cudaEventRecord(ev_s5, cuda_stream);

        // === Step 5.5: Build GPU output arrays (d_centerPoints + d_line_ids) ===
        {
            thrust::device_vector<float2> d_coords(point_count);
            thrust::device_vector<int> d_labels_buf(point_count);

            thrust::transform(
                thrust::cuda::par_nosync.on(cuda_stream),
                d_valid_points.begin(),
                d_valid_points.begin() + point_count,
                d_coords.begin(),
                SubpixelPointToCoord());

            thrust::transform(
                thrust::cuda::par_nosync.on(cuda_stream),
                d_valid_points.begin(),
                d_valid_points.begin() + point_count,
                d_labels_buf.begin(),
                SubpixelPointToLabel());

            thrust::host_vector<float2> h_coords(d_coords);
            thrust::host_vector<int> h_labels_buf(d_labels_buf);

            cv::Mat h_coords_mat(1, point_count, CV_32FC2);
            memcpy(h_coords_mat.ptr<float2>(), h_coords.data(), point_count * sizeof(float2));

            cv::Mat h_labels_mat(1, point_count, CV_32SC1);
            memcpy(h_labels_mat.ptr<int>(), h_labels_buf.data(), point_count * sizeof(int));

            auto d_cp = std::make_shared<cv::cuda::GpuMat>();
            d_cp->upload(h_coords_mat, stream);

            auto d_pl = std::make_shared<cv::cuda::GpuMat>();
            d_pl->upload(h_labels_mat, stream);

            result.d_centerPoints = d_cp;
            result.d_line_ids = d_pl;
        }

        // === Step 6: D2H copy and build CPU result ===
        thrust::host_vector<SubpixelPoint> h_points(d_valid_points);

        std::map<int, std::vector<cv::Point2f>> centerPoints;
        for (const auto& pt : h_points) {
            centerPoints[pt.label].emplace_back(pt.px, pt.py);
        }

        cudaEventRecord(ev_s6, cuda_stream);
        cudaEventRecord(ev_end, cuda_stream);
        cudaEventSynchronize(ev_end);

        float ms_total=0, ms_s1=0, ms_s2=0, ms_s3=0, ms_s5=0, ms_s6=0;
        cudaEventElapsedTime(&ms_total, ev_start, ev_end);
        cudaEventElapsedTime(&ms_s1, ev_start, ev_s1);
        cudaEventElapsedTime(&ms_s2, ev_s1, ev_s2);
        cudaEventElapsedTime(&ms_s3, ev_s2, ev_s3);
        cudaEventElapsedTime(&ms_s5, ev_s3, ev_s5);
        cudaEventElapsedTime(&ms_s6, ev_s5, ev_s6);

        CALIB_LOG_INFO("[BENCH] {}x{} | Total={:.2f}ms | S1(Conv8U->32F)={:.2f}ms | S2(GaussConv5deriv)={:.2f}ms | S3(Hessian+Taylor)={:.2f}ms | S5(Sort)={:.2f}ms | S6(D2H+Group)={:.2f}ms",
                       cols, rows, ms_total, ms_s1, ms_s2, ms_s3, ms_s5, ms_s6);

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_s1);
        cudaEventDestroy(ev_s2);
        cudaEventDestroy(ev_s3);
        cudaEventDestroy(ev_s5);
        cudaEventDestroy(ev_s6);
        cudaEventDestroy(ev_end);

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
        CALIB_LOG_ERROR("extract() OpenCV exception: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        CALIB_LOG_ERROR("Execute() exception: {}", e.what());
    }

    return result;
}
