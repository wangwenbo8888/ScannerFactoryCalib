/**
 * @file undistort_points_cuda.cpp
 * @brief 婵€鍏変腑蹇冧簹鍍忕礌鐐归泦鍘荤暩鍙?绔嬩綋鐭绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "undistort_points_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getUndistortPointsCudaInfo() {
    return OperatorInfo{"UndistortPointsCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(11, UndistortPointsCuda);

#if BUILD_CUDA

#include "undistort_points_cuda_pimpl.h"

UndistortPointsCuda::UndistortPointsCuda(const UndistortPointsParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("UndistortPointsCuda initialized: deviceId={}", params.deviceId);
}

UndistortPointsCuda::~UndistortPointsCuda() = default;

UndistortPointsResult UndistortPointsCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: points={}x{} type={}, line_ids={}x{} type={}",
                    d_points.cols, d_points.rows, d_points.type(),
                    d_line_ids.cols, d_line_ids.rows, d_line_ids.type());

    if (d_points.empty()) {
        CALIB_LOG_WARN("Execute(): empty points input, returning empty result");
        UndistortPointsResult result;
        result.success = true;
        result.message = "Empty input, no points to undistort";
        result.d_rectifiedPoints = std::make_shared<cv::cuda::GpuMat>();
        result.d_line_ids = std::make_shared<cv::cuda::GpuMat>();
        return result;
    }

    if (d_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: type={}, expected CV_32FC2", d_points.type());
        UndistortPointsResult result;
        result.success = false;
        result.message = "Input must be CV_32FC2";
        return result;
    }

    int pointCount = d_points.rows * d_points.cols;

    if (!d_line_ids.empty()) {
        int fidCount = d_line_ids.rows * d_line_ids.cols;
        if (d_line_ids.type() != CV_32SC1) {
            CALIB_LOG_ERROR("Execute() failed: line_ids type={}, expected CV_32SC1", d_line_ids.type());
            UndistortPointsResult result;
            result.success = false;
            result.message = "Input line_ids must be CV_32SC1";
            return result;
        }
        if (pointCount != fidCount) {
            CALIB_LOG_ERROR("Execute() failed: points count={} != line_ids count={}",
                            pointCount, fidCount);
            UndistortPointsResult result;
            result.success = false;
            result.message = "Points and line_ids must have same number of elements";
            return result;
        }
    }

    if (pointCount == 0) {
        UndistortPointsResult result;
        result.success = true;
        result.message = "Zero points";
        result.d_rectifiedPoints = std::make_shared<cv::cuda::GpuMat>();
        result.d_line_ids = std::make_shared<cv::cuda::GpuMat>();
        return result;
    }

    return pImpl_->Execute(d_points, d_line_ids, stream);
}

UndistortPointsResult UndistortPointsCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_points, d_line_ids, stream);
}

UndistortPointsResult UndistortPointsCuda::Execute(
    const cv::cuda::GpuMat& d_points,
    cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat empty_line_ids;
    return Execute(d_points, empty_line_ids, stream);
}

UndistortPointsResult UndistortPointsCuda::Execute(
    const cv::cuda::GpuMat& d_points)
{
    cv::cuda::Stream stream;
    cv::cuda::GpuMat empty_line_ids;
    return Execute(d_points, empty_line_ids, stream);
}

void UndistortPointsCuda::Destroy() {
    CALIB_LOG_INFO("Destroy() called");
    pImpl_->Destroy();
}

void UndistortPointsCuda::Warmup(int maxPointCount) {
    CALIB_LOG_INFO("Warmup() called: maxPointCount={}", maxPointCount);
    pImpl_->Warmup(maxPointCount);
    CALIB_LOG_INFO("Warmup() completed");
}

void UndistortPointsCuda::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: maxPointCount={}", pointCount);
    Warmup(pointCount);
}

void UndistortPointsCuda::SetParams(const UndistortPointsParams& params) {
    CALIB_LOG_INFO("SetParams(): deviceId={}", params.deviceId);
    pImpl_->SetParams(params);
}

const UndistortPointsParams& UndistortPointsCuda::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct UndistortPointsCuda::Impl {};

UndistortPointsCuda::UndistortPointsCuda(const UndistortPointsParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("UndistortPointsCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

UndistortPointsCuda::~UndistortPointsCuda() = default;

UndistortPointsResult UndistortPointsCuda::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

UndistortPointsResult UndistortPointsCuda::Execute(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

UndistortPointsResult UndistortPointsCuda::Execute(const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

UndistortPointsResult UndistortPointsCuda::Execute(const cv::cuda::GpuMat&) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void UndistortPointsCuda::Destroy() {
    // No-op: no CUDA resources to release
}

void UndistortPointsCuda::Warmup(int) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void UndistortPointsCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void UndistortPointsCuda::SetParams(const UndistortPointsParams&) {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const UndistortPointsParams& UndistortPointsCuda::GetParams() const {
    throw std::runtime_error("[11-UndistortPointsCuda] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA