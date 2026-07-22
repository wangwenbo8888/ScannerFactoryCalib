/**
 * @file mask_extract_cuda_impl.cu
 * @brief 激光掩膜提取算子 - CUDA 实现（struct Impl 方法 + OpenCV CUDA API 调用）
 *
 * 本文件包含实际的 CUDA/GPU 代码。
 */

#include "mask_extract_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <stdexcept>
#include <memory>

using namespace calib;


// 日志标签（用于 extract/warmup 内部分配警告）
CALIB_DEFINE_LOG_TAG(06, MaskExtractCUDA);

// ============================================================
// Impl 构造函数
// ============================================================
MaskExtractCUDA::Impl::Impl(const MaskExtractParams& params)
    : params_(params)
{
    // 参数校验
    params_.validate();

    // 检查 CUDA 可用性
    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }

    // 创建形态学核
    rebuildFilters();
}

// ============================================================
// rebuildFilters() - 重建形态学核和滤波器
// ============================================================
void MaskExtractCUDA::Impl::rebuildFilters() {
    // 创建腐蚀核（去噪）
    kernel_erode_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(params_.erodeSize, params_.erodeSize)
    );

    // 创建膨胀核（恢复激光形状）
    kernel_dilate_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(params_.laserDilateSize, params_.laserDilateSize)
    );

    // 创建 CUDA 滤波器
    filter_erode_ = cv::cuda::createMorphologyFilter(
        cv::MORPH_ERODE, CV_8UC1, kernel_erode_
    );

    filter_dilate_ = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8UC1, kernel_dilate_
    );
}

// ============================================================
// executePipeline() - 执行形态学流水线
// 从成员变量 d_inputBuffer 读取输入
// ============================================================
void MaskExtractCUDA::Impl::executePipeline(
    cv::cuda::GpuMat& dst,
    cv::cuda::Stream& stream)
{
    // Step 1: Host→Device 上传（已在 extract() 中完成）
    // d_inputBuffer 已包含上传的图像

    // Step 2: GPU 二值化
    cv::cuda::threshold(d_inputBuffer, d_thresholded,
                        params_.threshold, 255.0,
                        cv::THRESH_BINARY, stream);

    // Step 3: GPU 腐蚀（去噪）
    filter_erode_->apply(d_thresholded, d_eroded, stream);

    // Step 4: GPU 膨胀（恢复激光形状）
    filter_dilate_->apply(d_eroded, d_laserMask, stream);

    // Step 5: 面积过滤（可选，TODO）
    if (params_.minArea > 0 || params_.maxArea < 1000000) {
        // TODO: 实现基于 CUDA 的连通域面积过滤
        // 当前版本直接复制结果
        d_laserMask.copyTo(d_cleanedMask, stream);
    } else {
        d_laserMask.copyTo(d_cleanedMask, stream);
    }

    // 输出到目标
    dst = d_cleanedMask;
}

// ============================================================
// releaseBuffers() - 释放 GPU 缓冲区
// ============================================================
void MaskExtractCUDA::Impl::releaseBuffers() {
    d_inputBuffer.release();
    d_thresholded.release();
    d_eroded.release();
    d_laserMask.release();
    d_cleanedMask.release();
    warmed_up_ = false;
}

// ============================================================
// Warmup() - 预热 GPU 资源
// ============================================================
void MaskExtractCUDA::Impl::Warmup(int rows, int cols) {
    // 预分配所有缓冲区（使用 createContinuous 确保显存连续分配，§2.3）
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_inputBuffer);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_thresholded);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_eroded);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_laserMask);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_cleanedMask);

    warmup_rows_ = rows;
    warmup_cols_ = cols;

    // 创建测试图像（全黑）
    cv::Mat test_image = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::cuda::GpuMat d_test;
    d_test.upload(test_image);

    // 执行一次空跑（不计时）
    cv::cuda::Stream stream;
    cv::cuda::threshold(d_test, d_thresholded, params_.threshold, 255.0, cv::THRESH_BINARY, stream);
    filter_erode_->apply(d_thresholded, d_eroded, stream);
    filter_dilate_->apply(d_eroded, d_laserMask, stream);
    stream.waitForCompletion();

#ifndef NDEBUG
    // Debug 模式：验证 GPU 分配有效性（§2.2 warmup 规范）
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("warmup() GPU allocation/validation failed: ") + cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("warmup() CUDA error after dry-run: ") + cudaGetErrorString(err));
    }
#endif

    warmed_up_ = true;
}

// ============================================================
// SetParams() - 动态更新参数
// ============================================================
void MaskExtractCUDA::Impl::SetParams(const MaskExtractParams& params) {
#ifndef NDEBUG
    // Debug 断言：确保 setParams() 不与 extract() 并发调用（§2.2）
    assert(!inProcess_.load() && "setParams() called while extract() is running - NOT thread-safe!");
#endif

    params_ = params;
    params_.validate();

    // 重建形态学核和滤波器
    rebuildFilters();

    // 重置预热状态（需要重新预热）
    warmed_up_ = false;
}

// ============================================================
// Execute() - 核心提取函数
// ============================================================
MaskExtractResult MaskExtractCUDA::Impl::Execute(
    const cv::Mat& grayImage,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    // Debug 模式：线程安全断言
    assert(!inProcess_.load() && "Concurrent extract() calls detected - NOT thread-safe!");

    // RAII guard 自动清除 inProcess_ 标记
    struct ScopedFlag {
        std::atomic<bool>* flag;
        ScopedFlag(std::atomic<bool>* f) : flag(f) {}
        ~ScopedFlag() {
            flag->store(false);
        }
    };

    ScopedFlag guard(&inProcess_);
    inProcess_.store(true);  // 标记为执行中
#endif

    MaskExtractResult result;

    try {
        // 检查是否需要重新分配缓冲区
        int rows = grayImage.rows;
        int cols = grayImage.cols;

        if (!warmed_up_ || warmup_rows_ != rows || warmup_cols_ != cols) {
            // 注意：此处分配违反 §2.3 "process()内禁止 cudaMalloc" 规范
            // 建议在正式使用前调用 warmup() 预分配
            CALIB_LOG_WARN("GPU buffer allocation during extract() - call warmup() beforehand for production use");
            cv::cuda::createContinuous(rows, cols, CV_8UC1, d_inputBuffer);
            cv::cuda::createContinuous(rows, cols, CV_8UC1, d_thresholded);
            cv::cuda::createContinuous(rows, cols, CV_8UC1, d_eroded);
            cv::cuda::createContinuous(rows, cols, CV_8UC1, d_laserMask);
            cv::cuda::createContinuous(rows, cols, CV_8UC1, d_cleanedMask);

            warmup_rows_ = rows;
            warmup_cols_ = cols;
            warmed_up_ = true;
        }

        // Step 1: Host→Device 上传
        d_inputBuffer.upload(grayImage, stream);

        // Step 2-5: 执行形态学流水线
        executePipeline(d_cleanedMask, stream);

        // 填充结果（使用 shared_ptr 管理 GpuMat，clone 确保结果独立于内部缓冲区）
        result.success = true;
        result.message = "Extraction successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        result.d_grayImage = std::make_shared<cv::cuda::GpuMat>(d_inputBuffer.clone());
        result.d_laserMask = std::make_shared<cv::cuda::GpuMat>(d_laserMask.clone());
        result.d_cleanedMask = std::make_shared<cv::cuda::GpuMat>(d_cleanedMask.clone());

    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        result.qualityFlag = calib::QualityFlag::Degraded;
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        result.qualityFlag = calib::QualityFlag::Degraded;
    }

    return result;
}
