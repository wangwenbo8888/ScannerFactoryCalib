#include "plane_map_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getPlaneMapCudaInfo() {
    return OperatorInfo{"PlaneMapCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(12, PlaneMapCuda);

#if BUILD_CUDA

#include "plane_map_cuda_pimpl.h"

PlaneMapCuda::PlaneMapCuda(const PlaneMapParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("PlaneMapCuda initialized: deviceId={}, method={}",
                   params.deviceId, static_cast<int>(params.method));
}

PlaneMapCuda::~PlaneMapCuda() = default;

PlaneMapResult PlaneMapCuda::Execute(
    const cv::cuda::GpuMat& d_virtual_pixels,
    const cv::Matx33d& virtualK,
    const cv::Matx33d& virtualR,
    const cv::Vec3d& virtualT,
    const calib::StereoCalibration& calib,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("process() called: pixels={}x{} type={}",
                    d_virtual_pixels.cols, d_virtual_pixels.rows, d_virtual_pixels.type());

    if (d_virtual_pixels.empty()) {
        CALIB_LOG_WARN("process(): empty input, returning empty result");
        PlaneMapResult result;
        result.success = true;
        result.message = "Empty input";
        return result;
    }

    if (d_virtual_pixels.type() != CV_32FC3) {
        CALIB_LOG_ERROR("process() failed: pixels type={}, expected CV_32FC3", d_virtual_pixels.type());
        PlaneMapResult result;
        result.success = false;
        result.message = "Input d_virtual_pixels must be CV_32FC3";
        return result;
    }

    if (calib.imageSize.width <= 0 || calib.imageSize.height <= 0) {
        CALIB_LOG_ERROR("process() failed: imageSize={}x{} invalid", calib.imageSize.width, calib.imageSize.height);
        PlaneMapResult result;
        result.success = false;
        result.message = "imageSize must have positive width and height";
        return result;
    }

    return pImpl_->Execute(d_virtual_pixels, virtualK, virtualR, virtualT, calib, stream);
}

PlaneMapResult PlaneMapCuda::Execute(
    const cv::cuda::GpuMat& d_virtual_pixels,
    const cv::Matx33d& virtualK,
    const cv::Matx33d& virtualR,
    const cv::Vec3d& virtualT,
    const calib::StereoCalibration& calib)
{
    cv::cuda::Stream stream;
    return Execute(d_virtual_pixels, virtualK, virtualR, virtualT, calib, stream);
}

void PlaneMapCuda::Warmup(int numVirtualPixels, int maxLineId) {
    CALIB_LOG_INFO("warmup() called: numVirtualPixels={}, maxLineId={}", numVirtualPixels, maxLineId);
    pImpl_->Warmup(numVirtualPixels, maxLineId);
    CALIB_LOG_INFO("warmup() completed");
}

void PlaneMapCuda::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    int pc = config.maxPointCount > 0 ? config.maxPointCount : 10000;
    Warmup(pc, 256);
}

void PlaneMapCuda::SetParams(const PlaneMapParams& params) {
    CALIB_LOG_INFO("setParams(): deviceId={}, method={}",
                   params.deviceId, static_cast<int>(params.method));
    pImpl_->SetParams(params);
}

const PlaneMapParams& PlaneMapCuda::GetParams() const {
    return pImpl_->GetParams();
}

void PlaneMapCuda::Destroy() {
}

#else

struct PlaneMapCuda::Impl {};

PlaneMapCuda::PlaneMapCuda(const PlaneMapParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("PlaneMapCuda: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

PlaneMapCuda::~PlaneMapCuda() = default;

PlaneMapResult PlaneMapCuda::Execute(
    const cv::cuda::GpuMat&, const cv::Matx33d&, const cv::Matx33d&,
    const cv::Vec3d&, const calib::StereoCalibration&, cv::cuda::Stream&) {
    throw std::runtime_error("[12-PlaneMapCuda] CUDA not available (BUILD_CUDA=OFF)");
}

PlaneMapResult PlaneMapCuda::Execute(
    const cv::cuda::GpuMat&, const cv::Matx33d&, const cv::Matx33d&,
    const cv::Vec3d&, const calib::StereoCalibration&) {
    throw std::runtime_error("[12-PlaneMapCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PlaneMapCuda::Warmup(int, int) {
    throw std::runtime_error("[12-PlaneMapCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PlaneMapCuda::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[12-PlaneMapCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PlaneMapCuda::SetParams(const PlaneMapParams&) {
    throw std::runtime_error("[12-PlaneMapCuda] CUDA not available (BUILD_CUDA=OFF)");
}

const PlaneMapParams& PlaneMapCuda::GetParams() const {
    throw std::runtime_error("[12-PlaneMapCuda] CUDA not available (BUILD_CUDA=OFF)");
}

void PlaneMapCuda::Destroy() {
}

#endif // BUILD_CUDA