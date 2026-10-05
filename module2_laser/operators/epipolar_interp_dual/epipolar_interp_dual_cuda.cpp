/**
 * @file epipolar_interp_dual_cuda.cpp
 * @brief 激光中心点极线插值CUDA算子·双模式版 - 桥接实现
 */

#include "epipolar_interp_dual_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getEpipolarInterpDualInfo() {
    return OperatorInfo{"EpipolarInterpDualCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(06, EpipolarInterpDualCuda);

#if BUILD_CUDA

#include "epipolar_interp_dual_cuda_pimpl.h"

EpipolarInterpDualCuda::EpipolarInterpDualCuda(const EpipolarInterpDualParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EpipolarInterpDualCuda initialized: mode={}, step={}",
                   static_cast<int>(params.mode), params.epipolar_row_step);
}

EpipolarInterpDualCuda::~EpipolarInterpDualCuda() = default;

EpipolarInterpResult EpipolarInterpDualCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: points={}x{} type={}, ids type={}",
                    d_points.cols, d_points.rows, d_points.type(), d_line_ids.type());

    if (d_points.empty() || d_line_ids.empty()) {
        CALIB_LOG_WARN("Execute(): empty input");
        EpipolarInterpResult result;
        result.success = true;
        result.message = "Empty input, no interpolation";
        result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>();
        result.interpCount = 0;
        return result;
    }

    if (d_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: points type={}, expected CV_32FC2", d_points.type());
        EpipolarInterpResult result;
        result.success = false;
        result.message = "Points must be CV_32FC2";
        return result;
    }

    if (d_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: line_ids type={}, expected CV_32SC1", d_line_ids.type());
        EpipolarInterpResult result;
        result.success = false;
        result.message = "Line ids must be CV_32SC1";
        return result;
    }

    int pointCount = d_points.rows * d_points.cols;
    int idCount = d_line_ids.rows * d_line_ids.cols;
    if (pointCount != idCount) {
        CALIB_LOG_ERROR("Execute() failed: points count={} != ids count={}", pointCount, idCount);
        EpipolarInterpResult result;
        result.success = false;
        result.message = "Points and line_ids must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_points, d_line_ids, stream);
}

EpipolarInterpResult EpipolarInterpDualCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_points, d_line_ids, stream);
}

void EpipolarInterpDualCuda::Destroy() {
    pImpl_.reset();
}

void EpipolarInterpDualCuda::Warmup(int pointCount) {
    CALIB_LOG_INFO("Warmup() called: pointCount={}", pointCount);
    pImpl_->Warmup(pointCount);
}

void EpipolarInterpDualCuda::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    Warmup(pointCount);
}

void EpipolarInterpDualCuda::SetParams(const EpipolarInterpDualParams& params) {
    CALIB_LOG_INFO("SetParams(): mode={}", static_cast<int>(params.mode));
    pImpl_->SetParams(params);
}

const EpipolarInterpDualParams& EpipolarInterpDualCuda::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct EpipolarInterpDualCuda::Impl {};

EpipolarInterpDualCuda::EpipolarInterpDualCuda(const EpipolarInterpDualParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("EpipolarInterpDualCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

EpipolarInterpDualCuda::~EpipolarInterpDualCuda() = default;

EpipolarInterpResult EpipolarInterpDualCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[06-EpipolarInterpDualCuda] CUDA not available (BUILD_CUDA=OFF)");
}

EpipolarInterpResult EpipolarInterpDualCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[06-EpipolarInterpDualCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpDualCuda::Destroy() {}

void EpipolarInterpDualCuda::Warmup(int) {
    throw std::runtime_error("[06-EpipolarInterpDualCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpDualCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[06-EpipolarInterpDualCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpDualCuda::SetParams(const EpipolarInterpDualParams&) {
    throw std::runtime_error("[06-EpipolarInterpDualCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const EpipolarInterpDualParams& EpipolarInterpDualCuda::GetParams() const {
    throw std::runtime_error("[06-EpipolarInterpDualCuda] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA
