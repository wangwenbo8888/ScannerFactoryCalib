/**
 * @file mask_extract_cuda.cpp
 * @brief 激光掩膜提取算子 - 桥接实现（构造/析构/warmup/setParams）
 *
 * 本文件实现 pImpl 模式的桥接函数，不包含 CUDA 代码。
 */

#include "mask_extract_cuda.h"
#include "mask_extract_cuda_pimpl.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getMaskExtractCUDAInfo() {
    return OperatorInfo{"MaskExtractCUDA", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

// 定义日志标签
CALIB_DEFINE_LOG_TAG(06, MaskExtractCUDA);

// ============================================================
// 构造函数
// ============================================================
MaskExtractCUDA::MaskExtractCUDA(const MaskExtractParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("MaskExtractCUDA initialized with threshold={}, erodeSize={}, dilateSize={}",
                   params.threshold, params.erodeSize, params.laserDilateSize);
}

// ============================================================
// Destroy()
// ============================================================
void MaskExtractCUDA::Destroy() {
    if (pImpl_) {
        pImpl_->releaseBuffers();
    }
}

MaskExtractCUDA::~MaskExtractCUDA() = default;

// ============================================================
// Execute() - 核心接口
// ============================================================
MaskExtractResult MaskExtractCUDA::Execute(const cv::Mat& grayImage, cv::cuda::Stream& stream) {
    CALIB_LOG_DEBUG("Execute() called: image size={}x{}, type={}",
                    grayImage.cols, grayImage.rows, grayImage.type());

    if (grayImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: input image is empty");
        MaskExtractResult result;
        result.success = false;
        result.message = "Input image is empty";
        return result;
    }

    if (grayImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input must be CV_8UC1, got type {}", grayImage.type());
        MaskExtractResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 grayscale image";
        return result;
    }

    return pImpl_->Execute(grayImage, stream);
}

// ============================================================
// Execute() - 默认空流便利重载（向后兼容）
// ============================================================
// [算子规范 §1.6 豁免] 本无 stream 重载仅供测试/调试使用，流水线调用必须
// 显式传 Stream&（§1.2）。defaultStream 为 thread_local，每线程独立实例，
// 无跨实例/跨线程共享，不构成 §1.6 禁止的可变共享状态。
MaskExtractResult MaskExtractCUDA::Execute(const cv::Mat& grayImage) {
    static thread_local cv::cuda::Stream defaultStream;
    return Execute(grayImage, defaultStream);
}

// ============================================================
// Warmup(int, int)
// ============================================================
void MaskExtractCUDA::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
    CALIB_LOG_INFO("Warmup() completed");
}

// ============================================================
// Warmup(WarmupConfig)
// ============================================================
void MaskExtractCUDA::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

// ============================================================
// SetParams()
// ============================================================
void MaskExtractCUDA::SetParams(const MaskExtractParams& params) {
    CALIB_LOG_INFO("SetParams() called: threshold={}, erodeSize={}, dilateSize={}",
                   params.threshold, params.erodeSize, params.laserDilateSize);
    pImpl_->SetParams(params);
}

// ============================================================
// GetParams()
// ============================================================
const MaskExtractParams& MaskExtractCUDA::GetParams() const {
    return pImpl_->GetParams();
}

