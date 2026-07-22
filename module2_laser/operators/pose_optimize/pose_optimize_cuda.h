#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

struct PoseOptimizeParams {
    int deviceId = 0;
    int maxIterations = 50;
    double convergenceThreshold = 1e-6;
    int minLinesForOptimize = 3;
    int minPointsPerLine = 10;
    bool enableTiming = false;

    void validate() const {
        if (deviceId < 0)
            throw std::invalid_argument("PoseOptimizeParams::deviceId must be >= 0");
        if (convergenceThreshold <= 0.0)
            throw std::invalid_argument("PoseOptimizeParams::convergenceThreshold must be > 0");
        if (maxIterations <= 0)
            throw std::invalid_argument("PoseOptimizeParams::maxIterations must be > 0");
        if (minLinesForOptimize < 2)
            throw std::invalid_argument("PoseOptimizeParams::minLinesForOptimize must be >= 2");
        if (minPointsPerLine < 3)
            throw std::invalid_argument("PoseOptimizeParams::minPointsPerLine must be >= 3");
    }

    nlohmann::json toJson() const {
        return {
            {"deviceId", deviceId},
            {"maxIterations", maxIterations},
            {"convergenceThreshold", convergenceThreshold},
            {"minLinesForOptimize", minLinesForOptimize},
            {"minPointsPerLine", minPointsPerLine},
            {"enableTiming", enableTiming}
        };
    }

    static PoseOptimizeParams fromJson(const nlohmann::json& j) {
        PoseOptimizeParams p;
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("maxIterations"))
            p.maxIterations = j.at("maxIterations").get<int>();
        if (j.contains("convergenceThreshold"))
            p.convergenceThreshold = j.at("convergenceThreshold").get<double>();
        if (j.contains("minLinesForOptimize"))
            p.minLinesForOptimize = j.at("minLinesForOptimize").get<int>();
        if (j.contains("minPointsPerLine"))
            p.minPointsPerLine = j.at("minPointsPerLine").get<int>();
        if (j.contains("enableTiming"))
            p.enableTiming = j.at("enableTiming").get<bool>();
        p.validate();
        return p;
    }
};

struct LaserLineCurve {
    int lineId = 0;
    double coeffs[3] = {0, 0, 0};
    char mainAxis = 'u';
    double fittingError = 0.0;
    int pointCount = 0;
};

struct PoseOptimizeResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Matx33d virtualK;
    cv::Matx33d virtualR;
    cv::Vec3d   virtualT;
    cv::Vec3d   initialT;

    int numLines = 0;
    double totalReprojectionError = 0.0;
    double initialReprojectionError = 0.0;
    std::vector<LaserLineCurve> lineCurves;

    PoseOptimizeResult() = default;
    ~PoseOptimizeResult() = default;

    PoseOptimizeResult(PoseOptimizeResult&&) = default;
    PoseOptimizeResult& operator=(PoseOptimizeResult&&) = default;

    PoseOptimizeResult(const PoseOptimizeResult&) = delete;
    PoseOptimizeResult& operator=(const PoseOptimizeResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 点云/线号/虚拟相机初值按调用传入；Impl 仅持有每调用重置的暂存缓冲，SetParams 缓存迭代阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API PoseOptimizeCuda {
public:
    static constexpr const char* kLogTag = "11-PoseOptimizeCuda";

    explicit PoseOptimizeCuda(const PoseOptimizeParams& params = {});
    ~PoseOptimizeCuda();

    PoseOptimizeCuda(const PoseOptimizeCuda&) = delete;
    PoseOptimizeCuda& operator=(const PoseOptimizeCuda&) = delete;

    PoseOptimizeResult Execute(
        const cv::cuda::GpuMat& d_points,
        const cv::cuda::GpuMat& d_line_ids,
        const cv::Matx33d& virtualK,
        const cv::Matx33d& virtualR,
        const cv::Vec3d& initialT,
        cv::cuda::Stream& stream);

    PoseOptimizeResult Execute(
        const cv::cuda::GpuMat& d_points,
        const cv::cuda::GpuMat& d_line_ids,
        const cv::Matx33d& virtualK,
        const cv::Matx33d& virtualR,
        const cv::Vec3d& initialT);

    void Warmup(int numPoints, int maxLineId);
    void Warmup(const WarmupConfig& config);
    void SetParams(const PoseOptimizeParams& params);
    const PoseOptimizeParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getPoseOptimizeCudaInfo();

} // namespace calib