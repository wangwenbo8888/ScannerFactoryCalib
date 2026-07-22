#pragma once

#include <opencv2/core.hpp>
#include "common/quality_flag.h"
#include "common/json_utils.h"
#include <vector>
#include <string>
#include <memory>
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct ExtrinsicCalibCpuParams {
    std::vector<std::vector<cv::Point2f>> leftPointsPerView;
    std::vector<std::vector<cv::Point2f>> rightPointsPerView;
    std::vector<cv::Point3f> objectPoints;
    cv::Size imageSize;
    int flags = 0;
    bool calibrateMono = false;
    cv::Size patternSize;
    float squareSize = 0.0f;
    double maxReprojError = 1.0;
    int minViewCount = 8;
    bool rotateRightImage180 = false;
    double maxEpipolarError = 0.05;

    nlohmann::json toJson() const;
    static ExtrinsicCalibCpuParams fromJson(const nlohmann::json& j);
    void validate() const;
};

struct ExtrinsicCalibCpuResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Mat R, T, E, F;
    double stereoReprojError = 0.0;
    double epipolarErrorMean = 0.0;
    double epipolarErrorStd = 0.0;
    std::vector<double> perViewErrors;
    std::vector<double> perViewEpipolarErrors;
    cv::Mat cameraMatrixL, distCoeffsL;
    cv::Mat cameraMatrixR, distCoeffsR;

    ExtrinsicCalibCpuResult() = default;
    // move-only (spec §3.3)
    ExtrinsicCalibCpuResult(const ExtrinsicCalibCpuResult&) = delete;
    ExtrinsicCalibCpuResult& operator=(const ExtrinsicCalibCpuResult&) = delete;
    ExtrinsicCalibCpuResult(ExtrinsicCalibCpuResult&&) noexcept = default;
    ExtrinsicCalibCpuResult& operator=(ExtrinsicCalibCpuResult&&) noexcept = default;

    nlohmann::json toJson() const;
    static ExtrinsicCalibCpuResult fromJson(const nlohmann::json& j);
};

struct ExtrinsicCalibCpuMonoResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double reprojError = 0.0;
    std::vector<double> perViewErrors;

    ExtrinsicCalibCpuMonoResult() = default;
    // move-only (spec §3.3)
    ExtrinsicCalibCpuMonoResult(const ExtrinsicCalibCpuMonoResult&) = delete;
    ExtrinsicCalibCpuMonoResult& operator=(const ExtrinsicCalibCpuMonoResult&) = delete;
    ExtrinsicCalibCpuMonoResult(ExtrinsicCalibCpuMonoResult&&) noexcept = default;
    ExtrinsicCalibCpuMonoResult& operator=(ExtrinsicCalibCpuMonoResult&&) noexcept = default;
};

struct ExtrinsicCalibCpuFullResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    ExtrinsicCalibCpuMonoResult monoLeft;
    ExtrinsicCalibCpuMonoResult monoRight;
    ExtrinsicCalibCpuResult stereo;

    ExtrinsicCalibCpuFullResult() = default;
    // move-only (spec §3.3)
    ExtrinsicCalibCpuFullResult(const ExtrinsicCalibCpuFullResult&) = delete;
    ExtrinsicCalibCpuFullResult& operator=(const ExtrinsicCalibCpuFullResult&) = delete;
    ExtrinsicCalibCpuFullResult(ExtrinsicCalibCpuFullResult&&) noexcept = default;
    ExtrinsicCalibCpuFullResult& operator=(ExtrinsicCalibCpuFullResult&&) noexcept = default;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 一次性标定求解，结果按调用返回；SetParams 缓存观测点集/标定板参数等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API ExtrinsicCalibCpu {
public:
    inline static const char* kLogTag = "[03-ExtrinsicCalibCpu]";

    explicit ExtrinsicCalibCpu(const ExtrinsicCalibCpuParams& params);
    ~ExtrinsicCalibCpu();

    ExtrinsicCalibCpu(const ExtrinsicCalibCpu&) = delete;
    ExtrinsicCalibCpu& operator=(const ExtrinsicCalibCpu&) = delete;

    void SetParams(const ExtrinsicCalibCpuParams& params);
    const ExtrinsicCalibCpuParams& GetParams() const;

    void Warmup() { }

    void Destroy();

    ExtrinsicCalibCpuResult Execute(
        const cv::Mat& cameraMatrixL, const cv::Mat& distCoeffsL,
        const cv::Mat& cameraMatrixR, const cv::Mat& distCoeffsR
    );

    ExtrinsicCalibCpuFullResult Execute();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getExtrinsicCalibCpuInfo();

} // namespace calib
