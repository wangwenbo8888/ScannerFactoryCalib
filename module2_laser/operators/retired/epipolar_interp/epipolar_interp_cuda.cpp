/**
 * @file epipolar_interp_cuda.cpp
 * @brief 婵€鍏変腑蹇冪偣鏋佺嚎鎻掑€糃UDA绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "epipolar_interp_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getEpipolarInterpCudaInfo() {
    return OperatorInfo{"EpipolarInterpCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(06, EpipolarInterpCuda);

#if BUILD_CUDA

#include "epipolar_interp_cuda_pimpl.h"

EpipolarInterpCuda::EpipolarInterpCuda(const EpipolarInterpParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EpipolarInterpCuda initialized: deviceId={}, step={}",
                   params.deviceId, params.epipolar_row_step);
}

EpipolarInterpCuda::~EpipolarInterpCuda() = default;

EpipolarInterpResult EpipolarInterpCuda::Execute(
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

EpipolarInterpResult EpipolarInterpCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_points, d_line_ids, stream);
}

void EpipolarInterpCuda::Destroy() {
    CALIB_LOG_INFO("Destroy() called");
    pImpl_->Destroy();
}

void EpipolarInterpCuda::Warmup(int pointCount) {
    CALIB_LOG_INFO("Warmup() called: pointCount={}", pointCount);
    pImpl_->Warmup(pointCount);
    CALIB_LOG_INFO("Warmup() completed");
}

void EpipolarInterpCuda::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: pointCount={}", pointCount);
    Warmup(pointCount);
}

void EpipolarInterpCuda::SetParams(const EpipolarInterpParams& params) {
    CALIB_LOG_INFO("SetParams(): deviceId={}, step={}", params.deviceId, params.epipolar_row_step);
    pImpl_->SetParams(params);
}

const EpipolarInterpParams& EpipolarInterpCuda::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct EpipolarInterpCuda::Impl {};

EpipolarInterpCuda::EpipolarInterpCuda(const EpipolarInterpParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("EpipolarInterpCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

EpipolarInterpCuda::~EpipolarInterpCuda() = default;

EpipolarInterpResult EpipolarInterpCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[06-EpipolarInterpCuda] CUDA not available (BUILD_CUDA=OFF)");
}

EpipolarInterpResult EpipolarInterpCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[06-EpipolarInterpCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpCuda::Destroy() {
    // No-op: no CUDA resources to release
}

void EpipolarInterpCuda::Warmup(int) {
    throw std::runtime_error("[06-EpipolarInterpCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[06-EpipolarInterpCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EpipolarInterpCuda::SetParams(const EpipolarInterpParams&) {
    throw std::runtime_error("[06-EpipolarInterpCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const EpipolarInterpParams& EpipolarInterpCuda::GetParams() const {
    throw std::runtime_error("[06-EpipolarInterpCuda] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA