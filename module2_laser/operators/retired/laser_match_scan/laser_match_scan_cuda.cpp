/**
 * @file laser_match_scan_cuda.cpp
 * @brief 婵€鍏夌嚎鍖归厤鎵弿CUDA绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams/loadTempTable/process锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "laser_match_scan_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserMatchScanCudaInfo() {
    return OperatorInfo{"LaserMatchScanCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCuda);

#if BUILD_CUDA

#include "laser_match_scan_cuda_pimpl.h"

LaserMatchScanCuda::LaserMatchScanCuda(const LaserMatchScanParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserMatchScanCuda initialized: deviceId={}, threshold={}",
                   params.deviceId, params.match_threshold);
}

LaserMatchScanCuda::~LaserMatchScanCuda() = default;

bool LaserMatchScanCuda::LoadTempTable(const std::string& jsonPath) {
    return pImpl_->LoadTempTable(jsonPath);
}

bool LaserMatchScanCuda::SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table) {
    return pImpl_->SetTempTable(std::move(table));
}

void LaserMatchScanCuda::SetCurrentTemperature(double temperature) {
    pImpl_->SetCurrentTemperature(temperature);
}

LaserMatchScanResult LaserMatchScanCuda::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: left={}x{} type={}, right={}x{} type={}",
                    d_left_points.cols, d_left_points.rows, d_left_points.type(),
                    d_right_points.cols, d_right_points.rows, d_right_points.type());

    if (d_left_points.empty() || d_left_line_ids.empty() ||
        d_right_points.empty() || d_right_line_ids.empty()) {
        CALIB_LOG_WARN("Execute(): empty input, returning empty result");
        LaserMatchScanResult result;
        result.success = true;
        result.message = "Empty input, no points to match";
        result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
        result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
        result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
        result.d_left_status = std::make_shared<cv::cuda::GpuMat>();
        result.d_right_status = std::make_shared<cv::cuda::GpuMat>();
        return result;
    }

    if (d_left_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: left_points type={}, expected CV_32FC2", d_left_points.type());
        LaserMatchScanResult result;
        result.success = false;
        result.message = "Left points must be CV_32FC2";
        return result;
    }

    if (d_right_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: right_points type={}, expected CV_32FC2", d_right_points.type());
        LaserMatchScanResult result;
        result.success = false;
        result.message = "Right points must be CV_32FC2";
        return result;
    }

    if (d_left_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: left_line_ids type={}, expected CV_32SC1", d_left_line_ids.type());
        LaserMatchScanResult result;
        result.success = false;
        result.message = "Left line_ids must be CV_32SC1";
        return result;
    }

    if (d_right_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: right_line_ids type={}, expected CV_32SC1", d_right_line_ids.type());
        LaserMatchScanResult result;
        result.success = false;
        result.message = "Right line_ids must be CV_32SC1";
        return result;
    }

    int leftPointCount = d_left_points.rows * d_left_points.cols;
    int leftIdCount = d_left_line_ids.rows * d_left_line_ids.cols;
    if (leftPointCount != leftIdCount) {
        CALIB_LOG_ERROR("Execute() failed: left points count={} != line_ids count={}",
                        leftPointCount, leftIdCount);
        LaserMatchScanResult result;
        result.success = false;
        result.message = "Left points and line_ids must have same number of elements";
        return result;
    }

    int rightPointCount = d_right_points.rows * d_right_points.cols;
    int rightIdCount = d_right_line_ids.rows * d_right_line_ids.cols;
    if (rightPointCount != rightIdCount) {
        CALIB_LOG_ERROR("Execute() failed: right points count={} != line_ids count={}",
                        rightPointCount, rightIdCount);
        LaserMatchScanResult result;
        result.success = false;
        result.message = "Right points and line_ids must have same number of elements";
        return result;
    }

    return pImpl_->process(d_left_points, d_left_line_ids,
                           d_right_points, d_right_line_ids, stream);
}

LaserMatchScanResult LaserMatchScanCuda::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_left_points, d_left_line_ids,
                   d_right_points, d_right_line_ids, stream);
}

void LaserMatchScanCuda::Destroy() {
    pImpl_.reset();
}

void LaserMatchScanCuda::Warmup(int maxLeftPoints, int maxRightPoints) {
    CALIB_LOG_INFO("Warmup() called: maxLeft={}, maxRight={}", maxLeftPoints, maxRightPoints);
    pImpl_->warmup(maxLeftPoints, maxRightPoints);
    CALIB_LOG_INFO("Warmup() completed");
}

void LaserMatchScanCuda::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: pointCount={}", pointCount);
    Warmup(pointCount, pointCount);
}

void LaserMatchScanCuda::SetParams(const LaserMatchScanParams& params) {
    CALIB_LOG_INFO("SetParams(): deviceId={}, threshold={}", params.deviceId, params.match_threshold);
    pImpl_->setParams(params);
}

const LaserMatchScanParams& LaserMatchScanCuda::GetParams() const {
    return pImpl_->getParams();
}

#else

struct LaserMatchScanCuda::Impl {};

LaserMatchScanCuda::LaserMatchScanCuda(const LaserMatchScanParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("LaserMatchScanCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

LaserMatchScanCuda::~LaserMatchScanCuda() = default;

bool LaserMatchScanCuda::LoadTempTable(const std::string&) {
    CALIB_LOG_WARN("loadTempTable: CUDA not available (BUILD_CUDA=OFF)");
    return false;
}

bool LaserMatchScanCuda::SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable>) {
    CALIB_LOG_WARN("SetTempTable: CUDA not available (BUILD_CUDA=OFF)");
    return false;
}

void LaserMatchScanCuda::SetCurrentTemperature(double) {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

LaserMatchScanResult LaserMatchScanCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    cv::cuda::Stream&) {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

LaserMatchScanResult LaserMatchScanCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchScanCuda::Destroy() {}

void LaserMatchScanCuda::Warmup(int, int) {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchScanCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserMatchScanCuda::SetParams(const LaserMatchScanParams&) {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const LaserMatchScanParams& LaserMatchScanCuda::GetParams() const {
    throw std::runtime_error("[07-LaserMatchScanCuda] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA