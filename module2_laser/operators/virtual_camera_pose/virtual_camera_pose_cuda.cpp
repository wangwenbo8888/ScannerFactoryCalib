/**
 * @file virtual_camera_pose_cuda.cpp
 * @brief 婵€鍏夊櫒铏氭嫙鐩告満鍏夊績鍜屽垵姝ュ鍙侰UDA绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "virtual_camera_pose_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getVirtualCameraPoseCudaInfo() {
    return OperatorInfo{"VirtualCameraPoseCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::Hybrid};
}

CALIB_DEFINE_LOG_TAG(10, VirtualCameraPoseCuda);

#if BUILD_CUDA

#include "virtual_camera_pose_cuda_pimpl.h"

VirtualCameraPoseCuda::VirtualCameraPoseCuda(const VirtualCameraPoseParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("VirtualCameraPoseCuda initialized: deviceId={}, ransacThreshold={}",
                   params.deviceId, params.ransacThreshold);
}

VirtualCameraPoseCuda::~VirtualCameraPoseCuda() = default;

VirtualCameraPoseResult VirtualCameraPoseCuda::Execute(
    const cv::cuda::GpuMat& d_endpoints,
    const cv::cuda::GpuMat& d_line_ids,
    const cv::Matx33d& stereoK,
    const cv::Matx33d& stereoR,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("process() called: endpoints={}x{} type={}, line_ids={}x{} type={}",
                    d_endpoints.cols, d_endpoints.rows, d_endpoints.type(),
                    d_line_ids.cols, d_line_ids.rows, d_line_ids.type());

    if (d_endpoints.empty() || d_line_ids.empty()) {
        CALIB_LOG_WARN("process(): empty input, returning empty result");
        VirtualCameraPoseResult result;
        result.success = true;
        result.message = "Empty input, no endpoints to process";
        result.virtualK = stereoK;
        result.virtualR = stereoR;
        return result;
    }

    if (d_endpoints.type() != CV_32FC3) {
        CALIB_LOG_ERROR("process() failed: endpoints type={}, expected CV_32FC3", d_endpoints.type());
        VirtualCameraPoseResult result;
        result.success = false;
        result.message = "Input d_endpoints must be CV_32FC3";
        return result;
    }

    if (d_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("process() failed: line_ids type={}, expected CV_32SC1", d_line_ids.type());
        VirtualCameraPoseResult result;
        result.success = false;
        result.message = "Input d_line_ids must be CV_32SC1";
        return result;
    }

    int ptCount = d_endpoints.rows * d_endpoints.cols;
    int lidCount = d_line_ids.rows * d_line_ids.cols;

    if (ptCount != lidCount) {
        CALIB_LOG_ERROR("process() failed: element count mismatch endpoints={}, line_ids={}",
                        ptCount, lidCount);
        VirtualCameraPoseResult result;
        result.success = false;
        result.message = "d_endpoints and d_line_ids must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_endpoints, d_line_ids, stereoK, stereoR, stream);
}

VirtualCameraPoseResult VirtualCameraPoseCuda::Execute(
    const cv::cuda::GpuMat& d_endpoints,
    const cv::cuda::GpuMat& d_line_ids,
    const cv::Matx33d& stereoK,
    const cv::Matx33d& stereoR)
{
    cv::cuda::Stream stream;
    return Execute(d_endpoints, d_line_ids, stereoK, stereoR, stream);
}

void VirtualCameraPoseCuda::Warmup(int numEndpoints, int maxLineId) {
    CALIB_LOG_INFO("warmup() called: numEndpoints={}, maxLineId={}", numEndpoints, maxLineId);
    pImpl_->Warmup(numEndpoints, maxLineId);
    CALIB_LOG_INFO("warmup() completed");
}

void VirtualCameraPoseCuda::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    int pc = config.maxPointCount > 0 ? config.maxPointCount : 10000;
    Warmup(pc, 256);
}

void VirtualCameraPoseCuda::SetParams(const VirtualCameraPoseParams& params) {
    CALIB_LOG_INFO("setParams(): deviceId={}, ransacThreshold={}",
                   params.deviceId, params.ransacThreshold);
    pImpl_->SetParams(params);
}

const VirtualCameraPoseParams& VirtualCameraPoseCuda::GetParams() const {
    return pImpl_->GetParams();
}

void VirtualCameraPoseCuda::Destroy() {
}

#else

struct VirtualCameraPoseCuda::Impl {};

VirtualCameraPoseCuda::VirtualCameraPoseCuda(const VirtualCameraPoseParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("VirtualCameraPoseCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

VirtualCameraPoseCuda::~VirtualCameraPoseCuda() = default;

VirtualCameraPoseResult VirtualCameraPoseCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::Matx33d&, const cv::Matx33d&,
    cv::cuda::Stream&) {
    throw std::runtime_error("[10-VirtualCameraPoseCuda] CUDA not available (BUILD_CUDA=OFF)");
}

VirtualCameraPoseResult VirtualCameraPoseCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::Matx33d&, const cv::Matx33d&) {
    throw std::runtime_error("[10-VirtualCameraPoseCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void VirtualCameraPoseCuda::Warmup(int, int) {
    throw std::runtime_error("[10-VirtualCameraPoseCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void VirtualCameraPoseCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[10-VirtualCameraPoseCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void VirtualCameraPoseCuda::SetParams(const VirtualCameraPoseParams&) {
    throw std::runtime_error("[10-VirtualCameraPoseCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const VirtualCameraPoseParams& VirtualCameraPoseCuda::GetParams() const {
    throw std::runtime_error("[10-VirtualCameraPoseCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void VirtualCameraPoseCuda::Destroy() {
}

#endif // BUILD_CUDA