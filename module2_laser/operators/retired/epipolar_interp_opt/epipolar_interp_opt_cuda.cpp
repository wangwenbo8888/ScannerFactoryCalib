/**
 * @file epipolar_interp_opt_cuda.cpp
 * @brief 激光中心点极线插值CUDA算子·优化版 - 桥接实现（构造/析构/warmup/setParams）
 *
 * 本文件实现 pImpl 模式的桥接函数，不包含 CUDA 代码。
 */

#include "epipolar_interp_opt_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getEpipolarInterpOptInfo() {
    return OperatorInfo{"EpipolarInterpOptCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(06, EpipolarInterpOptCuda);

#if BUILD_CUDA

#include "epipolar_interp_opt_cuda_pimpl.h"

EpipolarInterpOptCuda::EpipolarInterpOptCuda(const EpipolarInterpParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EpipolarInterpOptCuda initialized: deviceId={}, step={}",
                   params.deviceId, params.epipolar_row_step);
}

EpipolarInterpOptCuda::~EpipolarInterpOptCuda() = default;

EpipolarInterpResult EpipolarInterpOptCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: points={}x{} type={}, line_ids={}x{} type={}",
                    d_points.cols, d_points.rows, d_points.type(),
                    d_line_ids.cols, d_line_ids.rows, d_line_ids.type());

    if (d_points.empty() || d_line_ids.empty()) {
        CALIB_LOG_WARN("Execute(): empty input, returning empty result");
        EpipolarInterpResult result;
        result.success = true;
        result.message = "Empty input, no points to interpolate";
        result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
        result.interpCount = 0;
        return result;
    }

    if (d_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: points type={}, expected CV_32FC2", d_points.type());
        EpipolarInterpResult result;
        result.success = false;
        result.message = "Input points must be CV_32FC2";
        return result;
    }

    if (d_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: line_ids type={}, expected CV_32SC1", d_line_ids.type());
        EpipolarInterpResult result;
        result.success = false;
        result.message = "Input line_ids must be CV_32SC1";
        return result;
    }

    int pointCount = d_points.rows * d_points.cols;
    int frameCount = d_line_ids.rows * d_line_ids.cols;
    if (pointCount != frameCount) {
        CALIB_LOG_ERROR("Execute() failed: points count={} != line_ids count={}",
                        pointCount, frameCount);
        EpipolarInterpResult result;
        result.success = false;
        result.message = "Points and line_ids must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_points, d_line_ids, stream);
}

EpipolarInterpResult EpipolarInterpOptCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_points, d_line_ids, stream);
}

void EpipolarInterpOptCuda::Destroy() {
    CALIB_LOG_INFO("Destroy() called");
    pImpl_->Destroy();
}

void EpipolarInterpOptCuda::Warmup(int pointCount) {
    CALIB_LOG_INFO("Warmup() called: pointCount={}", pointCount);
    pImpl_->Warmup(pointCount);
    CALIB_LOG_INFO("Warmup() completed");
}

void EpipolarInterpOptCuda::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: pointCount={}", pointCount);
    Warmup(pointCount);
}

void EpipolarInterpOptCuda::SetParams(const EpipolarInterpParams& params) {
    CALIB_LOG_INFO("SetParams(): deviceId={}, step={}", params.deviceId, params.epipolar_row_step);
    pImpl_->SetParams(params);
}

const EpipolarInterpParams& EpipolarInterpOptCuda::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct EpipolarInterpOptCuda::Impl {};

EpipolarInterpOptCuda::EpipolarInterpOptCuda(const EpipolarInterpParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("EpipolarInterpOptCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

EpipolarInterpOptCuda::~EpipolarInterpOptCuda() = default;

EpipolarInterpResult EpipolarInterpOptCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[06-EpipolarInterpOptCuda] CUDA not available (BUILD_CUDA=OFF)");
}

EpipolarInterpResult EpipolarInterpOptCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[06-EpipolarInterpOptCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpOptCuda::Destroy() {
}

void EpipolarInterpOptCuda::Warmup(int) {
    throw std::runtime_error("[06-EpipolarInterpOptCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpOptCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[06-EpipolarInterpOptCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpOptCuda::SetParams(const EpipolarInterpParams&) {
    throw std::runtime_error("[06-EpipolarInterpOptCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const EpipolarInterpParams& EpipolarInterpOptCuda::GetParams() const {
    throw std::runtime_error("[06-EpipolarInterpOptCuda] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA
