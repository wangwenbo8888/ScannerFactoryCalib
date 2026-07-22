/**
 * @file laser_match_cuda.cpp
 * @brief 婵€鍏夌嚎鍖归厤CUDA绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "laser_match_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserMatchCudaInfo() {
    return OperatorInfo{"LaserMatchCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(07, LaserMatchCuda);

#if BUILD_CUDA

#include "laser_match_cuda_pimpl.h"

LaserMatchCuda::LaserMatchCuda(const LaserMatchParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserMatchCuda initialized: deviceId={}, step={}",
                   params.deviceId, params.epipolar_row_step);
}

LaserMatchCuda::~LaserMatchCuda() = default;

LaserMatchResult LaserMatchCuda::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("process() called: left_points={}x{} type={}, left_line_ids={}x{} type={}, "
                    "right_points={}x{} type={}, right_line_ids={}x{} type={}",
                    d_left_points.cols, d_left_points.rows, d_left_points.type(),
                    d_left_line_ids.cols, d_left_line_ids.rows, d_left_line_ids.type(),
                    d_right_points.cols, d_right_points.rows, d_right_points.type(),
                    d_right_line_ids.cols, d_right_line_ids.rows, d_right_line_ids.type());

    if (d_left_points.empty() || d_right_points.empty()) {
        CALIB_LOG_WARN("process(): empty input, returning empty result");
        LaserMatchResult result;
        result.success = true;
        result.message = "Empty input, no points to match";
        result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
        result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
        result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
        result.matchCount = 0;
        return result;
    }

    if (d_left_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("process() failed: left_points type={}, expected CV_32FC2", d_left_points.type());
        LaserMatchResult result;
        result.success = false;
        result.message = "Input left_points must be CV_32FC2";
        return result;
    }

    if (d_right_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("process() failed: right_points type={}, expected CV_32FC2", d_right_points.type());
        LaserMatchResult result;
        result.success = false;
        result.message = "Input right_points must be CV_32FC2";
        return result;
    }

    if (d_left_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("process() failed: left_line_ids type={}, expected CV_32SC1", d_left_line_ids.type());
        LaserMatchResult result;
        result.success = false;
        result.message = "Input left_line_ids must be CV_32SC1";
        return result;
    }

    if (d_right_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("process() failed: right_line_ids type={}, expected CV_32SC1", d_right_line_ids.type());
        LaserMatchResult result;
        result.success = false;
        result.message = "Input right_line_ids must be CV_32SC1";
        return result;
    }

    int leftPointCount = d_left_points.rows * d_left_points.cols;
    int leftFrameCount = d_left_line_ids.rows * d_left_line_ids.cols;
    if (leftPointCount != leftFrameCount) {
        CALIB_LOG_ERROR("process() failed: left points count={} != left line_ids count={}",
                        leftPointCount, leftFrameCount);
        LaserMatchResult result;
        result.success = false;
        result.message = "Left points and left line_ids must have same number of elements";
        return result;
    }

    int rightPointCount = d_right_points.rows * d_right_points.cols;
    int rightFrameCount = d_right_line_ids.rows * d_right_line_ids.cols;
    if (rightPointCount != rightFrameCount) {
        CALIB_LOG_ERROR("process() failed: right points count={} != right line_ids count={}",
                        rightPointCount, rightFrameCount);
        LaserMatchResult result;
        result.success = false;
        result.message = "Right points and right line_ids must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_left_points, d_left_line_ids, d_right_points, d_right_line_ids, stream);
}

LaserMatchResult LaserMatchCuda::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_left_points, d_left_line_ids, d_right_points, d_right_line_ids, stream);
}

void LaserMatchCuda::Warmup(int leftCount, int rightCount) {
    CALIB_LOG_INFO("warmup() called: leftCount={}, rightCount={}", leftCount, rightCount);
    pImpl_->Warmup(leftCount, rightCount);
    CALIB_LOG_INFO("warmup() completed");
}

void LaserMatchCuda::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    CALIB_LOG_INFO("warmup(WarmupConfig) called: leftCount={}, rightCount={}", pointCount, pointCount);
    Warmup(pointCount, pointCount);
}

void LaserMatchCuda::SetParams(const LaserMatchParams& params) {
    CALIB_LOG_INFO("setParams(): deviceId={}, step={}", params.deviceId, params.epipolar_row_step);
    pImpl_->SetParams(params);
}

const LaserMatchParams& LaserMatchCuda::GetParams() const {
    return pImpl_->GetParams();
}

void LaserMatchCuda::Destroy() {
    pImpl_.reset();
}

#else

struct LaserMatchCuda::Impl {};

LaserMatchCuda::LaserMatchCuda(const LaserMatchParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("LaserMatchCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

LaserMatchCuda::~LaserMatchCuda() = default;

LaserMatchResult LaserMatchCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    cv::cuda::Stream&) {
    throw std::runtime_error("[07-LaserMatchCuda] CUDA not available (BUILD_CUDA=OFF)");
}

LaserMatchResult LaserMatchCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[07-LaserMatchCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchCuda::Warmup(int, int) {
    throw std::runtime_error("[07-LaserMatchCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[07-LaserMatchCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchCuda::SetParams(const LaserMatchParams&) {
    throw std::runtime_error("[07-LaserMatchCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const LaserMatchParams& LaserMatchCuda::GetParams() const {
    throw std::runtime_error("[07-LaserMatchCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchCuda::Destroy() {
    // no-op: pImpl_ is empty struct when BUILD_CUDA=OFF
}

#endif // BUILD_CUDA