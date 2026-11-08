/**
 * @file flat_center.cpp
 * @brief 平顶激光中心提取算子（flat_center）- 桥接实现（构造/析构/校验/委托）
 *
 * 输入校验与错误消息口径照抄 steger_fast.cpp 同款（算子名替换）；
 * 设计文档：docs/plans/2026-10-18-flat-center-extractor-design.md §2
 */

#include "flat_center.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getFlatCenterInfo() {
    return OperatorInfo{"FlatCenterExtractor", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(10, FlatCenterExtractor);

#if BUILD_CUDA

#include "flat_center_pimpl.h"

FlatCenterExtractor::FlatCenterExtractor(const FlatCenterParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("FlatCenterExtractor initialized: lowThreshold={}, halfRatio={}, plateauRatio={}, "
                   "plateauMinWidth={}, minRunLength={}, minContrast={}, outlierGatePx={}, "
                   "maxLineWidth={}, smooth3={}, maxLabels={}, deviceId={}",
                   params.lowThreshold, params.halfRatio, params.plateauRatio,
                   params.plateauMinWidth, params.minRunLength, params.minContrast,
                   params.outlierGatePx, params.maxLineWidth, params.smooth3,
                   params.maxLabels, params.deviceId);
}

FlatCenterExtractor::~FlatCenterExtractor() = default;

StegerResult FlatCenterExtractor::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute(ByLabel) called: gray={}x{} type={}, label={}x{} type={}",
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
    return pImpl_->Execute(d_grayImage, d_labeledMask, false, stream);
}

StegerResult FlatCenterExtractor::Execute(
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
        return pImpl_->Execute(d_grayImage, d_mask, false, stream);
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

StegerResult FlatCenterExtractor::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask)
{
    cv::cuda::Stream stream;
    return Execute(d_grayImage, d_labeledMask, stream);
}

StegerResult FlatCenterExtractor::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_mask,
    GroupMode groupMode)
{
    cv::cuda::Stream stream;
    return Execute(d_grayImage, d_mask, stream, groupMode);
}

StegerResult FlatCenterExtractor::ExecuteDeferred(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    return pImpl_->extractFlatSubmit(d_grayImage, d_binaryMask, stream);
}

void FlatCenterExtractor::FinishDeferred(StegerResult& result) {
    pImpl_->finishFlat(result);
}

const int* FlatCenterExtractor::GetDeviceCountPtr() const {
    return pImpl_->d_stats_;
}

void FlatCenterExtractor::Destroy() {
    CALIB_LOG_INFO("Destroy() called");
    pImpl_->Destroy();
}

void FlatCenterExtractor::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
    CALIB_LOG_INFO("Warmup() completed");
}

void FlatCenterExtractor::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

void FlatCenterExtractor::SetParams(const FlatCenterParams& params) {
    CALIB_LOG_INFO("SetParams(): lowThreshold={}, halfRatio={}, plateauRatio={}, "
                   "plateauMinWidth={}, minRunLength={}, minContrast={}, outlierGatePx={}, "
                   "maxLineWidth={}, smooth3={}, maxLabels={}, deviceId={}",
                   params.lowThreshold, params.halfRatio, params.plateauRatio,
                   params.plateauMinWidth, params.minRunLength, params.minContrast,
                   params.outlierGatePx, params.maxLineWidth, params.smooth3,
                   params.maxLabels, params.deviceId);
    pImpl_->SetParams(params);
}

void FlatCenterExtractor::SetCollectHostPoints(bool collect) {
    pImpl_->collectHostPoints_ = collect;
}

void FlatCenterExtractor::LastBenchMs(float out[3]) const {
    for (int i = 0; i < 3; ++i) out[i] = pImpl_->bench_ms_[i];
}

const FlatCenterParams& FlatCenterExtractor::GetParams() const {
    return pImpl_->GetParams();
}

uint64_t FlatCenterExtractor::GetParamsEpoch() const {
    return pImpl_->paramsEpoch_;
}

bool FlatCenterExtractor::PrepareBigGraph(const cv::cuda::GpuMat& d_grayImage,
                                          const cv::cuda::GpuMat& d_binaryMask) {
    return pImpl_->prepareBigGraph_(d_grayImage, d_binaryMask);
}

bool FlatCenterExtractor::EmitGraphBody(cv::cuda::Stream& stream) {
    return pImpl_->emitGraphBody_(cv::cuda::StreamAccessor::getStream(stream));
}

StegerResult FlatCenterExtractor::MakeDeferredShell() {
    return pImpl_->makeDeferredShell_();
}

bool FlatCenterExtractor::GraphDims(int& rows, int& cols, int& bands) const {
    if (pImpl_->paramsFailed_ || !pImpl_->d_params_) return false;
    rows = pImpl_->graph_rows_;
    cols = pImpl_->graph_cols_;
    bands = pImpl_->graph_bands_;
    return pImpl_->graph_rows_ > 0;
}

#else

struct FlatCenterExtractor::Impl {};

FlatCenterExtractor::FlatCenterExtractor(const FlatCenterParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("FlatCenterExtractor: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

FlatCenterExtractor::~FlatCenterExtractor() = default;

StegerResult FlatCenterExtractor::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

StegerResult FlatCenterExtractor::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&, GroupMode) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

StegerResult FlatCenterExtractor::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

StegerResult FlatCenterExtractor::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, GroupMode) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

void FlatCenterExtractor::Destroy() {
}

void FlatCenterExtractor::Warmup(int, int) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

void FlatCenterExtractor::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

void FlatCenterExtractor::SetParams(const FlatCenterParams&) {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

const FlatCenterParams& FlatCenterExtractor::GetParams() const {
    throw std::runtime_error("[10-FlatCenterExtractor] CUDA not available (BUILD_CUDA=OFF)");
}

void FlatCenterExtractor::SetCollectHostPoints(bool) {
}

void FlatCenterExtractor::LastBenchMs(float out[3]) const {
    for (int i = 0; i < 3; ++i) out[i] = 0.f;
}

#endif // BUILD_CUDA
