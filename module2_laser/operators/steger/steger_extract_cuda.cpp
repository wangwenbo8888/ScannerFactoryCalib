/**
 * @file steger_extract_cuda.cpp
 * @brief Steger婵€鍏変腑蹇冧簹鍍忕礌鎻愬彇绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "steger_extract_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getStegerExtractInfo() {
    return OperatorInfo{"StegerExtract", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(10, StegerExtractorCUDA);

#if BUILD_CUDA

#include "steger_extract_cuda_pimpl.h"

StegerExtractorCUDA::StegerExtractorCUDA(const StegerParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("StegerExtractorCUDA initialized: sigma={}, kernelSize={}, lowThreshold={}, highThreshold={}, maxLabels={}, deviceId={}",
                   params.sigma, params.kernelSize, params.lowThreshold, params.highThreshold,
                   params.maxLabels, params.deviceId);
}

StegerExtractorCUDA::~StegerExtractorCUDA() = default;

StegerResult StegerExtractorCUDA::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: gray={}x{} type={}, label={}x{} type={}",
                    d_grayImage.cols, d_grayImage.rows, d_grayImage.type(),
                    d_labeledMask.cols, d_labeledMask.rows, d_labeledMask.type());

    if (d_grayImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: gray image is empty");
        StegerResult result;
        result.success = false;
        result.message = "Gray image is empty";
        return result;
    }

    if (d_labeledMask.empty()) {
        CALIB_LOG_ERROR("Execute() failed: labeled mask is empty");
        StegerResult result;
        result.success = false;
        result.message = "Labeled mask is empty";
        return result;
    }

    if (d_grayImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: gray image type={}, expected CV_8UC1", d_grayImage.type());
        StegerResult result;
        result.success = false;
        result.message = "Gray image must be CV_8UC1";
        return result;
    }

    if (d_labeledMask.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: label mask type={}, expected CV_32SC1", d_labeledMask.type());
        StegerResult result;
        result.success = false;
        result.message = "Labeled mask must be CV_32SC1";
        return result;
    }

    if (d_grayImage.cols != d_labeledMask.cols || d_grayImage.rows != d_labeledMask.rows) {
        CALIB_LOG_ERROR("Execute() failed: size mismatch gray={}x{}, label={}x{}",
                        d_grayImage.cols, d_grayImage.rows,
                        d_labeledMask.cols, d_labeledMask.rows);
        StegerResult result;
        result.success = false;
        result.message = "Gray image and label mask size mismatch";
        return result;
    }

    return pImpl_->Execute(d_grayImage, d_labeledMask, stream);
}

StegerResult StegerExtractorCUDA::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask)
{
    cv::cuda::Stream stream;
    return Execute(d_grayImage, d_labeledMask, stream);
}

StegerResult StegerExtractorCUDA::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_mask,
    cv::cuda::Stream& stream,
    GroupMode groupMode)
{
    CALIB_LOG_DEBUG("Execute(groupMode={}) called: gray={}x{} type={}, mask={}x{} type={}",
                    static_cast<int>(groupMode),
                    d_grayImage.cols, d_grayImage.rows, d_grayImage.type(),
                    d_mask.cols, d_mask.rows, d_mask.type());

    if (d_grayImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: gray image is empty");
        StegerResult result;
        result.success = false;
        result.message = "Gray image is empty";
        return result;
    }

    if (d_mask.empty()) {
        CALIB_LOG_ERROR("Execute() failed: mask is empty");
        StegerResult result;
        result.success = false;
        result.message = "Mask is empty";
        return result;
    }

    if (d_grayImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: gray image type={}, expected CV_8UC1", d_grayImage.type());
        StegerResult result;
        result.success = false;
        result.message = "Gray image must be CV_8UC1";
        return result;
    }

    if (d_grayImage.cols != d_mask.cols || d_grayImage.rows != d_mask.rows) {
        CALIB_LOG_ERROR("Execute() failed: size mismatch gray={}x{}, mask={}x{}",
                        d_grayImage.cols, d_grayImage.rows,
                        d_mask.cols, d_mask.rows);
        StegerResult result;
        result.success = false;
        result.message = "Gray image and mask size mismatch";
        return result;
    }

    if (groupMode == GroupMode::ByLabel) {
        if (d_mask.type() != CV_32SC1) {
            CALIB_LOG_ERROR("Execute(ByLabel) failed: mask type={}, expected CV_32SC1", d_mask.type());
            StegerResult result;
            result.success = false;
            result.message = "Labeled mask must be CV_32SC1 for ByLabel mode";
            return result;
        }
        return pImpl_->Execute(d_grayImage, d_mask, stream);
    }

    if (d_mask.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute(Flat) failed: mask type={}, expected CV_8UC1", d_mask.type());
        StegerResult result;
        result.success = false;
        result.message = "Binary mask must be CV_8UC1 for Flat mode";
        return result;
    }
    return pImpl_->extractFlat(d_grayImage, d_mask, stream);
}

StegerResult StegerExtractorCUDA::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_mask,
    GroupMode groupMode)
{
    cv::cuda::Stream stream;
    return Execute(d_grayImage, d_mask, stream, groupMode);
}

void StegerExtractorCUDA::Destroy() {
    CALIB_LOG_INFO("Destroy() called");
    pImpl_->Destroy();
}

void StegerExtractorCUDA::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
    CALIB_LOG_INFO("Warmup() completed");
}

void StegerExtractorCUDA::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

void StegerExtractorCUDA::SetParams(const StegerParams& params) {
    CALIB_LOG_INFO("SetParams(): sigma={}, kernelSize={}, lowThreshold={}, highThreshold={}, maxLabels={}, deviceId={}",
                   params.sigma, params.kernelSize, params.lowThreshold, params.highThreshold,
                   params.maxLabels, params.deviceId);
    pImpl_->SetParams(params);
}

const StegerParams& StegerExtractorCUDA::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct StegerExtractorCUDA::Impl {};

StegerExtractorCUDA::StegerExtractorCUDA(const StegerParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("StegerExtractorCUDA: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

StegerExtractorCUDA::~StegerExtractorCUDA() = default;

StegerResult StegerExtractorCUDA::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

StegerResult StegerExtractorCUDA::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

StegerResult StegerExtractorCUDA::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&, GroupMode) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

StegerResult StegerExtractorCUDA::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, GroupMode) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void StegerExtractorCUDA::Destroy() {
    // No-op: no CUDA resources to release
}

void StegerExtractorCUDA::Warmup(int, int) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void StegerExtractorCUDA::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void StegerExtractorCUDA::SetParams(const StegerParams&) {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

const StegerParams& StegerExtractorCUDA::GetParams() const {
    throw std::runtime_error("[10-StegerExtractorCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA