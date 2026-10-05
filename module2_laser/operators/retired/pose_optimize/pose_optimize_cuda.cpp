#include "pose_optimize_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getPoseOptimizeCudaInfo() {
    return OperatorInfo{"PoseOptimizeCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::Hybrid};
}

CALIB_DEFINE_LOG_TAG(11, PoseOptimizeCuda);

#if BUILD_CUDA

#include "pose_optimize_cuda_pimpl.h"

PoseOptimizeCuda::PoseOptimizeCuda(const PoseOptimizeParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("PoseOptimizeCuda initialized: deviceId={}, maxIterations={}",
                   params.deviceId, params.maxIterations);
}

PoseOptimizeCuda::~PoseOptimizeCuda() = default;

PoseOptimizeResult PoseOptimizeCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    const cv::Matx33d& virtualK,
    const cv::Matx33d& virtualR,
    const cv::Vec3d& initialT,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("process() called: points={}x{} type={}, line_ids={}x{} type={}",
                    d_points.cols, d_points.rows, d_points.type(),
                    d_line_ids.cols, d_line_ids.rows, d_line_ids.type());

    if (d_points.empty() || d_line_ids.empty()) {
        CALIB_LOG_WARN("process(): empty input, returning empty result");
        PoseOptimizeResult result;
        result.success = true;
        result.message = "Empty input";
        result.virtualK = virtualK;
        result.virtualR = virtualR;
        result.initialT = initialT;
        return result;
    }

    if (d_points.type() != CV_32FC3) {
        CALIB_LOG_ERROR("process() failed: points type={}, expected CV_32FC3", d_points.type());
        PoseOptimizeResult result;
        result.success = false;
        result.message = "Input d_points must be CV_32FC3";
        return result;
    }

    if (d_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("process() failed: line_ids type={}, expected CV_32SC1", d_line_ids.type());
        PoseOptimizeResult result;
        result.success = false;
        result.message = "Input d_line_ids must be CV_32SC1";
        return result;
    }

    int ptCount = d_points.rows * d_points.cols;
    int lidCount = d_line_ids.rows * d_line_ids.cols;

    if (ptCount != lidCount) {
        CALIB_LOG_ERROR("process() failed: element count mismatch points={}, line_ids={}",
                        ptCount, lidCount);
        PoseOptimizeResult result;
        result.success = false;
        result.message = "d_points and d_line_ids must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_points, d_line_ids, virtualK, virtualR, initialT, stream);
}

PoseOptimizeResult PoseOptimizeCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    const cv::Matx33d& virtualK,
    const cv::Matx33d& virtualR,
    const cv::Vec3d& initialT)
{
    cv::cuda::Stream stream;
    return Execute(d_points, d_line_ids, virtualK, virtualR, initialT, stream);
}

void PoseOptimizeCuda::Warmup(int numPoints, int maxLineId) {
    CALIB_LOG_INFO("warmup() called: numPoints={}, maxLineId={}", numPoints, maxLineId);
    pImpl_->Warmup(numPoints, maxLineId);
    CALIB_LOG_INFO("warmup() completed");
}

void PoseOptimizeCuda::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    int pc = config.maxPointCount > 0 ? config.maxPointCount : 10000;
    Warmup(pc, 256);
}

void PoseOptimizeCuda::SetParams(const PoseOptimizeParams& params) {
    CALIB_LOG_INFO("setParams(): deviceId={}, maxIterations={}",
                   params.deviceId, params.maxIterations);
    pImpl_->SetParams(params);
}

const PoseOptimizeParams& PoseOptimizeCuda::GetParams() const {
    return pImpl_->GetParams();
}

void PoseOptimizeCuda::Destroy() {
}

#else

struct PoseOptimizeCuda::Impl {};

PoseOptimizeCuda::PoseOptimizeCuda(const PoseOptimizeParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("PoseOptimizeCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

PoseOptimizeCuda::~PoseOptimizeCuda() = default;

PoseOptimizeResult PoseOptimizeCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::Matx33d&, const cv::Matx33d&, const cv::Vec3d&,
    cv::cuda::Stream&) {
    throw std::runtime_error("[11-PoseOptimizeCuda] CUDA not available (BUILD_CUDA=OFF)");
}

PoseOptimizeResult PoseOptimizeCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::Matx33d&, const cv::Matx33d&, const cv::Vec3d&) {
    throw std::runtime_error("[11-PoseOptimizeCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PoseOptimizeCuda::Warmup(int, int) {
    throw std::runtime_error("[11-PoseOptimizeCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PoseOptimizeCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[11-PoseOptimizeCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PoseOptimizeCuda::SetParams(const PoseOptimizeParams&) {
    throw std::runtime_error("[11-PoseOptimizeCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const PoseOptimizeParams& PoseOptimizeCuda::GetParams() const {
    throw std::runtime_error("[11-PoseOptimizeCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PoseOptimizeCuda::Destroy() {
}

#endif // BUILD_CUDA