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
    // 首腐蚀: erodeSize<=1 数学恒等, 不建滤波器（executePipeline 直接旁路）
    if (params_.erodeSize > 1) {
        kernel_erode_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(params_.erodeSize, params_.erodeSize)
        );
        filter_erode_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, kernel_erode_
        );
    } else {
        kernel_erode_.release();
        filter_erode_.release();
    }

    if (params_.morphApprox == 1) {
        // 线核两趟近似: 横一趟 + 竖一趟（矩形和, 大核下远快于 2D 椭圆核）
        const int d = params_.laserDilateSize;
        filter_dilate_h_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(d, 1)));
        filter_dilate_v_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, d)));
        if (params_.postErodeSize > 0) {
            const int e = params_.postErodeSize;
            filter_post_h_ = cv::cuda::createMorphologyFilter(
                cv::MORPH_ERODE, CV_8UC1,
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(e, 1)));
            filter_post_v_ = cv::cuda::createMorphologyFilter(
                cv::MORPH_ERODE, CV_8UC1,
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, e)));
        } else {
            filter_post_h_.release();
            filter_post_v_.release();
        }
        // 2D 大核路径不再使用
        kernel_dilate_.release();
        filter_dilate_.release();
        kernel_post_erode_.release();
        filter_post_erode_.release();
        return;
    }

    // 精确 2D 椭圆核路径（默认）
    filter_dilate_h_.release();
    filter_dilate_v_.release();
    filter_post_h_.release();
    filter_post_v_.release();

    kernel_dilate_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(params_.laserDilateSize, params_.laserDilateSize)
    );
    filter_dilate_ = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8UC1, kernel_dilate_
    );

    if (params_.postErodeSize > 0) {
        kernel_post_erode_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(params_.postErodeSize, params_.postErodeSize)
        );
        filter_post_erode_ = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, kernel_post_erode_
        );
    } else {
        kernel_post_erode_.release();
        filter_post_erode_.release();
    }
}

// ============================================================
// allocateBuffers() - 统一缓冲分配（Warmup 与 Execute 重分配共用）
// ============================================================
void MaskExtractCUDA::Impl::allocateBuffers(int rows, int cols) {
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_inputBuffer);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_thresholded);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_eroded);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_laserMask);
    cv::cuda::createContinuous(rows, cols, CV_8UC1, d_cleanedMask);
    if (params_.postErodeSize > 0)
        cv::cuda::createContinuous(rows, cols, CV_8UC1, d_postMask);
    else
        d_postMask.release();
    if (params_.morphApprox == 1)
        cv::cuda::createContinuous(rows, cols, CV_8UC1, d_morphTmp);
    else
        d_morphTmp.release();
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < 3; ++k)
            cv::cuda::createContinuous(rows, cols, CV_8UC1, d_resPool_[p][k]);
    resIdx_ = 0;
    warmup_rows_ = rows;
    warmup_cols_ = cols;
    warmed_up_ = true;
}

// ============================================================
// executePipeline() - 执行形态学流水线
// 从成员变量 d_inputBuffer 读取输入
// ============================================================
void MaskExtractCUDA::Impl::executePipeline(
    cv::cuda::GpuMat& dst,
    cv::cuda::Stream& stream)
{
    // Step 2: GPU 二值化
    cv::cuda::threshold(d_inputBuffer, d_thresholded,
                        params_.threshold, 255.0,
                        cv::THRESH_BINARY, stream);

    // Step 3: GPU 腐蚀（去噪; erodeSize<=1 恒等, 旁路）
    const cv::cuda::GpuMat* afterErode = &d_thresholded;
    if (params_.erodeSize > 1 && filter_erode_) {
        filter_erode_->apply(d_thresholded, d_eroded, stream);
        afterErode = &d_eroded;
    }

    // Step 4: GPU 膨胀（恢复激光形状; 线核两趟或 2D 椭圆核）
    if (params_.morphApprox == 1) {
        filter_dilate_h_->apply(*afterErode, d_morphTmp, stream);
        filter_dilate_v_->apply(d_morphTmp, d_laserMask, stream);
    } else {
        filter_dilate_->apply(*afterErode, d_laserMask, stream);
    }

    // Step 4b: 膨胀后二次腐蚀（收边, 可选）
    cv::cuda::GpuMat* mask_after_morph = &d_laserMask;
    if (params_.postErodeSize > 0) {
        if (params_.morphApprox == 1) {
            filter_post_h_->apply(d_laserMask, d_morphTmp, stream);
            filter_post_v_->apply(d_morphTmp, d_postMask, stream);
        } else if (filter_post_erode_) {
            filter_post_erode_->apply(d_laserMask, d_postMask, stream);
        }
        mask_after_morph = &d_postMask;
    }

    // Step 5: 输出清理后掩膜（面积过滤职责在 4-2 region_analyze，本算子不做）
    mask_after_morph->copyTo(d_cleanedMask, stream);

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
    d_postMask.release();
    d_cleanedMask.release();
    d_morphTmp.release();
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < 3; ++k)
            d_resPool_[p][k].release();
    warmed_up_ = false;
}

// ============================================================
// Warmup() - 预热 GPU 资源
// ============================================================
void MaskExtractCUDA::Impl::Warmup(int rows, int cols) {
    // 预分配所有缓冲区（使用 createContinuous 确保显存连续分配，§2.3）
    allocateBuffers(rows, cols);

    // 创建测试图像（全黑）并空跑一次完整流水线（不计时）
    cv::Mat test_image = cv::Mat::zeros(rows, cols, CV_8UC1);
    d_inputBuffer.upload(test_image);
    cv::cuda::Stream stream;
    executePipeline(d_cleanedMask, stream);
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
            allocateBuffers(rows, cols);
        }

        // Step 1: Host→Device 上传
        d_inputBuffer.upload(grayImage, stream);

        // Step 2-5: 执行形态学流水线
        executePipeline(d_cleanedMask, stream);

        // 填充结果: ping-pong 池流式拷贝（预分配缓冲, 免 clone 的
        // 临时 cudaMalloc + 默认流屏障; 结果至再执行两次 Execute 前有效）
        cv::cuda::GpuMat* pool = d_resPool_[resIdx_];
        d_inputBuffer.copyTo(pool[0], stream);
        d_laserMask.copyTo(pool[1], stream);
        d_cleanedMask.copyTo(pool[2], stream);
        resIdx_ = (resIdx_ + 1) % 2;

        result.success = true;
        result.message = "Extraction successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        result.d_grayImage = std::make_shared<cv::cuda::GpuMat>(pool[0]);
        result.d_laserMask = std::make_shared<cv::cuda::GpuMat>(pool[1]);
        result.d_cleanedMask = std::make_shared<cv::cuda::GpuMat>(pool[2]);

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
