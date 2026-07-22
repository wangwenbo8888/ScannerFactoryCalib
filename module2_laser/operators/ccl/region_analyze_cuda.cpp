/**
 * @file region_analyze_cuda.cpp
 * @brief 婵€鍏夎繛閫氬煙鍒嗘瀽绠楀瓙 - 妗ユ帴瀹炵幇锛堟瀯閫?鏋愭瀯/warmup/setParams锛?
 *
 * 鏈枃浠跺疄鐜?pImpl 妯″紡鐨勬ˉ鎺ュ嚱鏁帮紝涓嶅寘鍚?CUDA 浠ｇ爜銆?
 * 浠呰礋璐ｏ細鏋勯€?鏋愭瀯銆佽緭鍏ユ牎楠屻€佸弬鏁版洿鏂般€侀鐑浆鍙戙€?
 *
 * 鏉′欢缂栬瘧锛?
 * - BUILD_CUDA=ON锛堟甯告ā寮忥級锛歩nclude _pimpl.h锛屾甯稿疄渚嬪寲 CUDA pImpl
 * - BUILD_CUDA=OFF锛圕PU-only 妯″紡锛夛細绌?Impl stub锛屾墍鏈夋柟娉曟姏鍑?runtime_error
 */

#include "region_analyze_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <stdexcept>

using namespace calib;

OperatorInfo getRegionAnalyzerCUDAInfo() {
    return OperatorInfo{"RegionAnalyzerCUDA", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(07, RegionAnalyzerCUDA);

#if BUILD_CUDA

#include "region_analyze_cuda_pimpl.h"

// ============================================================
// 鏋勯€犲嚱鏁?
// ============================================================
RegionAnalyzerCUDA::RegionAnalyzerCUDA(const RegionAnalyzerParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("RegionAnalyzerCUDA initialized: minArea={}, maxArea={}, deviceId={}",
                   params.minArea, params.maxArea, params.deviceId);
}

// ============================================================
// Destroy()
// ============================================================
void RegionAnalyzerCUDA::Destroy() {
    if (pImpl_) {
        pImpl_->freeStatsBuffers();
    }
}

RegionAnalyzerCUDA::~RegionAnalyzerCUDA() = default;

// ============================================================
// Execute() - shared_ptr 閲嶈浇锛堟帹鑽愶級
// ============================================================
RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const std::shared_ptr<cv::cuda::GpuMat>& d_mask,
    cv::cuda::Stream& stream)
{
    if (!d_mask) {
        CALIB_LOG_ERROR("Execute() failed: input shared_ptr is null");
        RegionAnalysisResult result;
        result.success = false;
        result.message = "Input shared_ptr<GpuMat> is null";
        return result;
    }

    CALIB_LOG_DEBUG("Execute(shared_ptr) called: size={}x{}, type={}",
                    d_mask->cols, d_mask->rows, d_mask->type());

    if (d_mask->empty()) {
        CALIB_LOG_ERROR("Execute() failed: input mask is empty");
        RegionAnalysisResult result;
        result.success = false;
        result.message = "Input binary mask is empty";
        return result;
    }

    if (d_mask->type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input type mismatch, expected CV_8UC1, got type={}",
                        d_mask->type());
        RegionAnalysisResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 binary mask";
        return result;
    }

    return pImpl_->Execute(d_mask, stream);
}

// ============================================================
// Execute() - const GpuMat& 閲嶈浇锛堝悜鍚庡吋瀹癸級
// ============================================================
RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const cv::cuda::GpuMat& d_inputBinaryMask,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: size={}x{}, type={}",
                    d_inputBinaryMask.cols, d_inputBinaryMask.rows, d_inputBinaryMask.type());

    if (d_inputBinaryMask.empty()) {
        CALIB_LOG_ERROR("Execute() failed: input mask is empty");
        RegionAnalysisResult result;
        result.success = false;
        result.message = "Input binary mask is empty";
        return result;
    }

    if (d_inputBinaryMask.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input type mismatch, expected CV_8UC1, got type={}",
                        d_inputBinaryMask.type());
        RegionAnalysisResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 binary mask";
        return result;
    }

    return pImpl_->Execute(d_inputBinaryMask, stream);
}

// ============================================================
// Execute() - 鏃?stream 閲嶈浇锛堝悜鍚庡吋瀹癸紝娴嬭瘯/璋冭瘯鐢級
// ============================================================
// [算子规范 §1.6 豁免] 本无 stream 重载仅供测试/调试使用，流水线调用必须
// 显式传 Stream&（§1.2）。defaultStream 为 thread_local，每线程独立实例，
// 无跨实例/跨线程共享，不构成 §1.6 禁止的可变共享状态。
RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const std::shared_ptr<cv::cuda::GpuMat>& d_mask)
{
    static thread_local cv::cuda::Stream defaultStream;
    return Execute(d_mask, defaultStream);
}

// [算子规范 §1.6 豁免] 同上，无 stream 重载仅供测试/调试使用。
RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const cv::cuda::GpuMat& d_inputBinaryMask)
{
    static thread_local cv::cuda::Stream defaultStream;
    return Execute(d_inputBinaryMask, defaultStream);
}

// ============================================================
// Warmup(int, int)
// ============================================================
void RegionAnalyzerCUDA::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
    CALIB_LOG_INFO("Warmup() completed");
}

// ============================================================
// Warmup(WarmupConfig)
// ============================================================
void RegionAnalyzerCUDA::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

// ============================================================
// SetParams()
// ============================================================
void RegionAnalyzerCUDA::SetParams(const RegionAnalyzerParams& params) {
    CALIB_LOG_INFO("SetParams(): minArea={}, maxArea={}, deviceId={}",
                   params.minArea, params.maxArea, params.deviceId);
    pImpl_->SetParams(params);
}

// ============================================================
// GetParams()
// ============================================================
const RegionAnalyzerParams& RegionAnalyzerCUDA::GetParams() const {
    return pImpl_->GetParams();
}

#else

// 鈹€鈹€鈹€ CPU-only 妯″紡锛氱┖ Impl stub 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

struct RegionAnalyzerCUDA::Impl {};

RegionAnalyzerCUDA::RegionAnalyzerCUDA(const RegionAnalyzerParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("RegionAnalyzerCUDA: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

RegionAnalyzerCUDA::~RegionAnalyzerCUDA() = default;

void RegionAnalyzerCUDA::Destroy() {
    // no-op: no CUDA resources allocated
}

RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const std::shared_ptr<cv::cuda::GpuMat>&, cv::cuda::Stream&)
{
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const cv::cuda::GpuMat&, cv::cuda::Stream&)
{
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const std::shared_ptr<cv::cuda::GpuMat>&)
{
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

RegionAnalysisResult RegionAnalyzerCUDA::Execute(
    const cv::cuda::GpuMat&)
{
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void RegionAnalyzerCUDA::Warmup(int, int) {
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void RegionAnalyzerCUDA::Warmup(const calib::WarmupConfig&) {
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

void RegionAnalyzerCUDA::SetParams(const RegionAnalyzerParams&) {
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

const RegionAnalyzerParams& RegionAnalyzerCUDA::GetParams() const {
    throw std::runtime_error("[07-RegionAnalyzerCUDA] CUDA not available (BUILD_CUDA=OFF)");
}

#endif // BUILD_CUDA