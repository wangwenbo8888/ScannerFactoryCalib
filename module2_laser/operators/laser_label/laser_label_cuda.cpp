/**
 * @file laser_label_cuda.cpp
 * @brief 婵€鍏夌嚎缂栧彿绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 */

#include "laser_label_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserLabelerCUDAInfo() {
    return OperatorInfo{"LaserLabelerCUDA", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(09, LaserLabelerCUDA);

#if BUILD_CUDA

#include "laser_label_cuda_pimpl.h"

LaserLabelerCUDA::LaserLabelerCUDA(const LaserLabelParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserLabelerCUDA initialized: maxLabels={}, centerColOffset={}, deviceId={}",
                   params.maxLabels, params.centerColOffset, params.deviceId);
}

LaserLabelerCUDA::~LaserLabelerCUDA() = default;

LaserLabelResult LaserLabelerCUDA::Execute(
    const cv::cuda::GpuMat& d_inputMask,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("label() called: size={}x{}, type={}",
                    d_inputMask.cols, d_inputMask.rows, d_inputMask.type());

    if (d_inputMask.empty()) {
        CALIB_LOG_ERROR("label() failed: input mask is empty");
        LaserLabelResult result;
        result.success = false;
        result.message = "Input mask is empty";
        return result;
    }

    int inputType = d_inputMask.type();
    if (inputType != CV_32SC1) {
        CALIB_LOG_ERROR("label() failed: unsupported input type={}, expected CV_32SC1 (from 02region_analyze_cuda)",
                        inputType);
        LaserLabelResult result;
        result.success = false;
        result.message = "Input must be CV_32SC1 (labeled mask from region_analyze_cuda)";
        return result;
    }

    return pImpl_->Execute(d_inputMask, stream);
}

void LaserLabelerCUDA::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
    CALIB_LOG_INFO("warmup() completed");
}

void LaserLabelerCUDA::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

void LaserLabelerCUDA::SetParams(const LaserLabelParams& params) {
    CALIB_LOG_INFO("setParams(): maxLabels={}, centerColOffset={}, deviceId={}",
                   params.maxLabels, params.centerColOffset, params.deviceId);
    pImpl_->SetParams(params);
}

const LaserLabelParams& LaserLabelerCUDA::GetParams() const {
    return pImpl_->GetParams();
}

LaserLabelResult LaserLabelerCUDA::Execute(
    const cv::cuda::GpuMat& d_inputMask)
{
    cv::cuda::Stream stream;
    return Execute(d_inputMask, stream);
}

void LaserLabelerCUDA::Destroy() {
    pImpl_.reset();
}

#else

struct LaserLabelerCUDA::Impl {};

LaserLabelerCUDA::LaserLabelerCUDA(const LaserLabelParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("LaserLabelerCUDA: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

LaserLabelerCUDA::~LaserLabelerCUDA() = default;

LaserLabelResult LaserLabelerCUDA::Execute(const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[09-LaserLabelerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserLabelerCUDA::Warmup(int, int) {
    throw std::runtime_error("[09-LaserLabelerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserLabelerCUDA::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[09-LaserLabelerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserLabelerCUDA::SetParams(const LaserLabelParams&) {
    throw std::runtime_error("[09-LaserLabelerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

const LaserLabelParams& LaserLabelerCUDA::GetParams() const {
    throw std::runtime_error("[09-LaserLabelerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

LaserLabelResult LaserLabelerCUDA::Execute(const cv::cuda::GpuMat&) {
    throw std::runtime_error("[09-LaserLabelerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void LaserLabelerCUDA::Destroy() {
    // no-op: pImpl_ is empty struct when BUILD_CUDA=OFF
}

#endif // BUILD_CUDA