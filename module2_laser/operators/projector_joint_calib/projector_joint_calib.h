#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"
#include "common/calib_warmup_config.h"

namespace calib {


struct ProjectorJointCalibParams {
    int    maxIterations         = 100;
    double convergenceThreshold  = 1e-8;
    int    minPoses              = 5;
    int    minPointsPerPose      = 50;
    double planeFitInlierThresh  = 0.6;    // 平面降噪内点阈值（3σ for σ=0.2mm 真实噪声）
    bool   enableTiming          = false;

    double lambda0              = 1.0;
    double lambdaDecay          = 0.95;
    double huberToCauchyThresh  = 1.0;
    double cauchyToL2Thresh     = 0.3;
    double topologyEpsilon      = 1e-3;
    int    curveDegree          = 3;       // 曲线阶数：2=二阶多项式, 3=三阶/隐式6参数
    bool   useCeres             = false;  // 优化后端：false=手写LM, true=Ceres(需BUILD_CERES)

    double anomalyRmsThreshold  = 0.15;   // 异常拟合检测：finalSampsonRms 超此值则判 Warning（实测 σ=0.2 正常≈0.045，伪极小值陷阱≈0.48）

    void validate() const {
        if (convergenceThreshold <= 0.0)
            throw std::invalid_argument("ProjectorJointCalibParams::convergenceThreshold must be > 0");
        if (maxIterations <= 0)
            throw std::invalid_argument("ProjectorJointCalibParams::maxIterations must be > 0");
        if (minPoses < 3)
            throw std::invalid_argument("ProjectorJointCalibParams::minPoses must be >= 3");
        if (minPointsPerPose < 10)
            throw std::invalid_argument("ProjectorJointCalibParams::minPointsPerPose must be >= 10");
        if (planeFitInlierThresh <= 0.0)
            throw std::invalid_argument("ProjectorJointCalibParams::planeFitInlierThresh must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"maxIterations", maxIterations},
            {"convergenceThreshold", convergenceThreshold},
            {"minPoses", minPoses},
            {"minPointsPerPose", minPointsPerPose},
            {"planeFitInlierThresh", planeFitInlierThresh},
            {"enableTiming", enableTiming},
            {"lambda0", lambda0},
            {"lambdaDecay", lambdaDecay},
            {"huberToCauchyThresh", huberToCauchyThresh},
            {"cauchyToL2Thresh", cauchyToL2Thresh},
            {"topologyEpsilon", topologyEpsilon},
            {"anomalyRmsThreshold", anomalyRmsThreshold}
        };
    }

    static ProjectorJointCalibParams fromJson(const nlohmann::json& j) {
        ProjectorJointCalibParams p;
        if (j.contains("maxIterations")) p.maxIterations = j.at("maxIterations").get<int>();
        if (j.contains("convergenceThreshold")) p.convergenceThreshold = j.at("convergenceThreshold").get<double>();
        if (j.contains("minPoses")) p.minPoses = j.at("minPoses").get<int>();
        if (j.contains("minPointsPerPose")) p.minPointsPerPose = j.at("minPointsPerPose").get<int>();
        if (j.contains("planeFitInlierThresh")) p.planeFitInlierThresh = j.at("planeFitInlierThresh").get<double>();
        if (j.contains("enableTiming")) p.enableTiming = j.at("enableTiming").get<bool>();
        if (j.contains("lambda0")) p.lambda0 = j.at("lambda0").get<double>();
        if (j.contains("lambdaDecay")) p.lambdaDecay = j.at("lambdaDecay").get<double>();
        if (j.contains("huberToCauchyThresh")) p.huberToCauchyThresh = j.at("huberToCauchyThresh").get<double>();
        if (j.contains("cauchyToL2Thresh")) p.cauchyToL2Thresh = j.at("cauchyToL2Thresh").get<double>();
        if (j.contains("topologyEpsilon")) p.topologyEpsilon = j.at("topologyEpsilon").get<double>();
        if (j.contains("anomalyRmsThreshold")) p.anomalyRmsThreshold = j.at("anomalyRmsThreshold").get<double>();
        p.validate();
        return p;
    }
};


struct PosePointSet {
    std::vector<cv::Vec3f> points3d;
    std::vector<int>       lineIds;
};


struct ProjectorJointCalibInput {
    std::vector<PosePointSet> poses;
    double      f = 0.0;
    cv::Point2d principalPoint = cv::Point2d(0.0, 0.0);
    cv::Vec3d   initialT = cv::Vec3d(0, 0, 0);
};


struct ImplicitCurve {
    double coeffs[6] = {0, 0, 0, 0, 0, 0};
    double discriminant = 0.0;
    double sampsonRms = 0.0;
    int    pointCount = 0;
};


struct ProjectorJointCalibResult {
    bool               success = false;
    std::string        message;
    QualityFlag        qualityFlag = QualityFlag::Normal;

    cv::Vec3d      projectorT = cv::Vec3d(0, 0, 0);
    cv::Vec3d      initialT   = cv::Vec3d(0, 0, 0);
    ImplicitCurve  emissionCurve;

    double initialSampsonRms = 0.0;
    double finalSampsonRms   = 0.0;
    double improvementRatio  = 1.0;
    int    poseCount         = 0;
    int    totalPointCount   = 0;
    double jacobianConditionNumber = 0.0;  // JᵀJ 条件数（姿态退化检测，>1e10 警示 t_z 不可信）
    std::vector<cv::Vec3d> denoisedPoints;  // 诊断：Step 1.5 曲线降噪后的点（左相机系）

    ProjectorJointCalibResult() = default;
    ~ProjectorJointCalibResult() = default;

    ProjectorJointCalibResult(ProjectorJointCalibResult&&) = default;
    ProjectorJointCalibResult& operator=(ProjectorJointCalibResult&&) = default;

    ProjectorJointCalibResult(const ProjectorJointCalibResult&) = delete;
    ProjectorJointCalibResult& operator=(const ProjectorJointCalibResult&) = delete;
};


// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 多姿态点云/焦距/初值按调用传入；Execute 一次性求解，无跨调用累积。
//       二期退火/核函数为单次 Execute 内部的 epoch 调度，非跨调用状态。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API ProjectorJointCalib {
public:
    static constexpr const char* kLogTag = "14-ProjectorJointCalib";

    explicit ProjectorJointCalib(const ProjectorJointCalibParams& params = {});
    ~ProjectorJointCalib() = default;

    ProjectorJointCalib(const ProjectorJointCalib&) = delete;
    ProjectorJointCalib& operator=(const ProjectorJointCalib&) = delete;

    ProjectorJointCalibResult Execute(const ProjectorJointCalibInput& input);

    void Warmup(int /*maxPoses*/, int /*maxPointsPerPose*/) {}
    void Warmup(const WarmupConfig& /*config*/) {}
    void SetParams(const ProjectorJointCalibParams& params);
    const ProjectorJointCalibParams& GetParams() const;

    void Destroy() {}

private:
    ProjectorJointCalibParams params_;
};


OperatorInfo getProjectorJointCalibInfo();

} // namespace calib
