#pragma once

#include <opencv2/core.hpp>
#include "common/quality_flag.h"
#include "common/json_utils.h"
#include <string>
#include <memory>
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct StereoRectifyCpuParams {
    cv::Mat cameraMatrixL;
    cv::Mat distCoeffsL;
    cv::Mat cameraMatrixR;
    cv::Mat distCoeffsR;
    cv::Size imageSize;
    cv::Mat R;
    cv::Mat T;
    double alpha = 0.0;
    int flags = 1;

    nlohmann::json toJson() const;
    static StereoRectifyCpuParams fromJson(const nlohmann::json& j);
    void validate() const;
};

struct StereoRectifyCpuResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoiLeft, validRoiRight;

    StereoRectifyCpuResult() = default;
    // move-only (spec §3.3)
    StereoRectifyCpuResult(const StereoRectifyCpuResult&) = delete;
    StereoRectifyCpuResult& operator=(const StereoRectifyCpuResult&) = delete;
    StereoRectifyCpuResult(StereoRectifyCpuResult&&) noexcept = default;
    StereoRectifyCpuResult& operator=(StereoRectifyCpuResult&&) noexcept = default;

    nlohmann::json toJson() const;
    static StereoRectifyCpuResult fromJson(const nlohmann::json& j);
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 一次性立体矫正求解，结果按调用返回；SetParams 缓存内外参等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API StereoRectifyCpu {
public:
    inline static const char* kLogTag = "[04-StereoRectifyCpu]";

    explicit StereoRectifyCpu(const StereoRectifyCpuParams& params);
    ~StereoRectifyCpu();

    StereoRectifyCpu(const StereoRectifyCpu&) = delete;
    StereoRectifyCpu& operator=(const StereoRectifyCpu&) = delete;

    void SetParams(const StereoRectifyCpuParams& params);
    const StereoRectifyCpuParams& GetParams() const;

    void Warmup() { }

    void Destroy();

    StereoRectifyCpuResult Execute();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getStereoRectifyCpuInfo();

} // namespace calib
