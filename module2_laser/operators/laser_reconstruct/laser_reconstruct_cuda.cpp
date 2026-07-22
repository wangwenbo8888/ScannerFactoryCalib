/**
 * @file laser_reconstruct_cuda.cpp
 * @brief 婵€鍏夌嚎涓夌淮閲嶅缓CUDA绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "laser_reconstruct_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserReconstructCudaInfo() {
    return OperatorInfo{"LaserReconstructCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(08, LaserReconstructCuda);

#if BUILD_CUDA

#include "laser_reconstruct_cuda_pimpl.h"

LaserReconstructCuda::LaserReconstructCuda(const LaserReconstructParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserReconstructCuda initialized: deviceId={}, minDepth={}, maxDepth={}",
                   params.deviceId, params.minDepth, params.maxDepth);
}

LaserReconstructCuda::~LaserReconstructCuda() = default;

LaserReconstructResult LaserReconstructCuda::Execute(
    const cv::cuda::GpuMat& d_matched_left,
    const cv::cuda::GpuMat& d_matched_right,
    const cv::cuda::GpuMat& d_matched_line_ids,
    const cv::Mat& Q,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: left={}x{} type={}, right={}x{} type={}, "
                    "line_ids={}x{} type={}, Q={}x{} type={}",
                    d_matched_left.cols, d_matched_left.rows, d_matched_left.type(),
                    d_matched_right.cols, d_matched_right.rows, d_matched_right.type(),
                    d_matched_line_ids.cols, d_matched_line_ids.rows, d_matched_line_ids.type(),
                    Q.cols, Q.rows, Q.type());

    if (d_matched_left.empty() || d_matched_right.empty() || d_matched_line_ids.empty()) {
        CALIB_LOG_WARN("Execute(): empty input, returning empty result");
        LaserReconstructResult result;
        result.success = true;
        result.message = "Empty input, no points to reconstruct";
        result.d_points3d = std::make_shared<cv::cuda::GpuMat>();
        result.d_valid_line_ids = std::make_shared<cv::cuda::GpuMat>();
        result.validCount = 0;
        result.totalInput = 0;
        return result;
    }

    if (d_matched_left.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: left type={}, expected CV_32FC2", d_matched_left.type());
        LaserReconstructResult result;
        result.success = false;
        result.message = "Input d_matched_left must be CV_32FC2";
        return result;
    }

    if (d_matched_right.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: right type={}, expected CV_32FC2", d_matched_right.type());
        LaserReconstructResult result;
        result.success = false;
        result.message = "Input d_matched_right must be CV_32FC2";
        return result;
    }

    if (d_matched_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: line_ids type={}, expected CV_32SC1", d_matched_line_ids.type());
        LaserReconstructResult result;
        result.success = false;
        result.message = "Input d_matched_line_ids must be CV_32SC1";
        return result;
    }

    if (Q.rows != 4 || Q.cols != 4 || Q.type() != CV_64FC1) {
        CALIB_LOG_ERROR("Execute() failed: Q={}x{} type={}, expected 4x4 CV_64FC1",
                        Q.cols, Q.rows, Q.type());
        LaserReconstructResult result;
        result.success = false;
        result.message = "Input Q must be 4x4 CV_64FC1";
        return result;
    }

    int leftCount = d_matched_left.rows * d_matched_left.cols;
    int rightCount = d_matched_right.rows * d_matched_right.cols;
    int fidCount = d_matched_line_ids.rows * d_matched_line_ids.cols;

    if (leftCount != rightCount || leftCount != fidCount) {
        CALIB_LOG_ERROR("Execute() failed: element count mismatch left={}, right={}, fids={}",
                        leftCount, rightCount, fidCount);
        LaserReconstructResult result;
        result.success = false;
        result.message = "All input GpuMats must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_matched_left, d_matched_right, d_matched_line_ids, Q, stream);
}

LaserReconstructResult LaserReconstructCuda::Execute(
    const cv::cuda::GpuMat& d_matched_left,
    const cv::cuda::GpuMat& d_matched_right,
    const cv::cuda::GpuMat& d_matched_line_ids,
    const cv::Mat& Q)
{
    cv::cuda::Stream stream;
    return Execute(d_matched_left, d_matched_right, d_matched_line_ids, Q, stream);
}

void LaserReconstructCuda::Destroy() {
    CALIB_LOG_INFO("Destroy() called");
    pImpl_->Destroy();
}

void LaserReconstructCuda::Warmup(int pointCount) {
    CALIB_LOG_INFO("Warmup() called: pointCount={}", pointCount);
    pImpl_->Warmup(pointCount);
    CALIB_LOG_INFO("Warmup() completed");
}

void LaserReconstructCuda::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols);
}

void LaserReconstructCuda::SetParams(const LaserReconstructParams& params) {
    CALIB_LOG_INFO("SetParams(): deviceId={}, minDepth={}, maxDepth={}",
                   params.deviceId, params.minDepth, params.maxDepth);
    pImpl_->SetParams(params);
}

const LaserReconstructParams& LaserReconstructCuda::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct LaserReconstructCuda::Impl {};

LaserReconstructCuda::LaserReconstructCuda(const LaserReconstructParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("LaserReconstructCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

LaserReconstructCuda::~LaserReconstructCuda() = default;

LaserReconstructResult LaserReconstructCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::Mat&,
    cv::cuda::Stream&) {
    throw std::runtime_error("[08-LaserReconstructCuda] CUDA not available (BUILD_CUDA=OFF)");
}

LaserReconstructResult LaserReconstructCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::Mat&) {
    throw std::runtime_error("[08-LaserReconstructCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserReconstructCuda::Destroy() {
    // No-op: no CUDA resources to release
}

void LaserReconstructCuda::Warmup(int) {
    throw std::runtime_error("[08-LaserReconstructCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserReconstructCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[08-LaserReconstructCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserReconstructCuda::SetParams(const LaserReconstructParams&) {
    throw std::runtime_error("[08-LaserReconstructCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const LaserReconstructParams& LaserReconstructCuda::GetParams() const {
    throw std::runtime_error("[08-LaserReconstructCuda] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA