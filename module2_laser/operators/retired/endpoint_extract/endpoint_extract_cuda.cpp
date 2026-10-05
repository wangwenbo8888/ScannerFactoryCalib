/**
 * @file endpoint_extract_cuda.cpp
 * @brief 婵€鍏夌嚎3D绔偣鎻愬彇CUDA绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "endpoint_extract_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getEndpointExtractCudaInfo() {
    return OperatorInfo{"EndpointExtractCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(09, EndpointExtractCuda);

#if BUILD_CUDA

#include "endpoint_extract_cuda_pimpl.h"

EndpointExtractCuda::EndpointExtractCuda(const EndpointExtractParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EndpointExtractCuda initialized: deviceId={}, maxExpectedLines={}",
                   params.deviceId, params.maxExpectedLines);
}

EndpointExtractCuda::~EndpointExtractCuda() = default;

EndpointExtractResult EndpointExtractCuda::Execute(
    const cv::cuda::GpuMat& d_points3d,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("process() called: points={}x{} type={}, fids={}x{} type={}",
                    d_points3d.cols, d_points3d.rows, d_points3d.type(),
                    d_line_ids.cols, d_line_ids.rows, d_line_ids.type());

    if (d_points3d.empty() || d_line_ids.empty()) {
        CALIB_LOG_WARN("process(): empty input, returning empty result");
        EndpointExtractResult result;
        result.success = true;
        result.message = "Empty input, no points to extract endpoints";
        result.d_endpoints = std::make_shared<cv::cuda::GpuMat>();
        result.d_endpoint_ids = std::make_shared<cv::cuda::GpuMat>();
        result.d_line_ids = std::make_shared<cv::cuda::GpuMat>();
        result.numEndpoints = 0;
        result.numLines = 0;
        result.totalInput = 0;
        return result;
    }

    if (d_points3d.type() != CV_32FC3) {
        CALIB_LOG_ERROR("process() failed: points3d type={}, expected CV_32FC3", d_points3d.type());
        EndpointExtractResult result;
        result.success = false;
        result.message = "Input d_points3d must be CV_32FC3";
        return result;
    }

    if (d_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("process() failed: line_ids type={}, expected CV_32SC1", d_line_ids.type());
        EndpointExtractResult result;
        result.success = false;
        result.message = "Input d_line_ids must be CV_32SC1";
        return result;
    }

    int ptCount = d_points3d.rows * d_points3d.cols;
    int fidCount = d_line_ids.rows * d_line_ids.cols;

    if (ptCount != fidCount) {
        CALIB_LOG_ERROR("process() failed: element count mismatch points={}, fids={}",
                        ptCount, fidCount);
        EndpointExtractResult result;
        result.success = false;
        result.message = "d_points3d and d_line_ids must have same number of elements";
        return result;
    }

    return pImpl_->Execute(d_points3d, d_line_ids, stream);
}

EndpointExtractResult EndpointExtractCuda::Execute(
    const cv::cuda::GpuMat& d_points3d,
    const cv::cuda::GpuMat& d_line_ids)
{
    cv::cuda::Stream stream;
    return Execute(d_points3d, d_line_ids, stream);
}

void EndpointExtractCuda::Warmup(int pointCount, int maxFrameId) {
    CALIB_LOG_INFO("warmup() called: pointCount={}, maxFrameId={}", pointCount, maxFrameId);
    pImpl_->Warmup(pointCount, maxFrameId);
    CALIB_LOG_INFO("warmup() completed");
}

void EndpointExtractCuda::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    int pc = config.maxPointCount > 0 ? config.maxPointCount : 10000;
    Warmup(pc, 256);
}

void EndpointExtractCuda::SetParams(const EndpointExtractParams& params) {
    CALIB_LOG_INFO("setParams(): deviceId={}, maxExpectedLines={}",
                   params.deviceId, params.maxExpectedLines);
    pImpl_->SetParams(params);
}

const EndpointExtractParams& EndpointExtractCuda::GetParams() const {
    return pImpl_->GetParams();
}

void EndpointExtractCuda::Destroy() {
    pImpl_.reset();
}

#else

struct EndpointExtractCuda::Impl {};

EndpointExtractCuda::EndpointExtractCuda(const EndpointExtractParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("EndpointExtractCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

EndpointExtractCuda::~EndpointExtractCuda() = default;

EndpointExtractResult EndpointExtractCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    cv::cuda::Stream&) {
    throw std::runtime_error("[09-EndpointExtractCuda] CUDA not available (BUILD_CUDA=OFF)");
}

EndpointExtractResult EndpointExtractCuda::Execute(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&) {
    throw std::runtime_error("[09-EndpointExtractCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EndpointExtractCuda::Warmup(int, int) {
    throw std::runtime_error("[09-EndpointExtractCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EndpointExtractCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[09-EndpointExtractCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EndpointExtractCuda::SetParams(const EndpointExtractParams&) {
    throw std::runtime_error("[09-EndpointExtractCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const EndpointExtractParams& EndpointExtractCuda::GetParams() const {
    throw std::runtime_error("[09-EndpointExtractCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void EndpointExtractCuda::Destroy() {
    // no-op: pImpl_ is empty struct when BUILD_CUDA=OFF
}

#endif // BUILD_CUDA