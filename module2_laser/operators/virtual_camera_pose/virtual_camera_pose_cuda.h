/**
 * @file virtual_camera_pose_cuda.h
 * @brief 激光器虚拟相机光心和初步外参CUDA算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?0步（endpoint_extract_cuda 之后�? * 平台：GPU（CUDA），核心计算�?CPU（Eigen�? *
 * 功能：根据多帧激光线端点三维数据，将激光器虚拟为无畸变相机�? *       求解虚拟相机光心位置（相对于左相机的外参平移向量�? *
 * 算法�? *   1. �?line_id 分组端点
 *   2. 逐线 RANSAC + PCA 拟合3D直线
 *   3. 解析求距离所有直线之和最小的交点
 *
 * 精度容差档次：档次③（亚像素/浮点类）
 */

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

// ============================================================================
// VirtualCameraPoseParams
// ============================================================================

struct VirtualCameraPoseParams {
    int deviceId = 0;
    double ransacThreshold = 1.0;
    double ransacConfidence = 0.99;
    int ransacMaxIterations = 1000;
    int minLinesForSolve = 3;
    int minPointsPerLine = 3;
    bool enableTiming = false;

    void validate() const {
        if (deviceId < 0)
            throw std::invalid_argument("VirtualCameraPoseParams::deviceId must be >= 0");
        if (ransacThreshold <= 0.0)
            throw std::invalid_argument("VirtualCameraPoseParams::ransacThreshold must be > 0");
        if (ransacConfidence <= 0.0 || ransacConfidence >= 1.0)
            throw std::invalid_argument("VirtualCameraPoseParams::ransacConfidence must be in (0, 1)");
        if (ransacMaxIterations <= 0)
            throw std::invalid_argument("VirtualCameraPoseParams::ransacMaxIterations must be > 0");
        if (minLinesForSolve < 2)
            throw std::invalid_argument("VirtualCameraPoseParams::minLinesForSolve must be >= 2");
        if (minPointsPerLine < 2)
            throw std::invalid_argument("VirtualCameraPoseParams::minPointsPerLine must be >= 2");
    }

    nlohmann::json toJson() const {
        return {
            {"deviceId", deviceId},
            {"ransacThreshold", ransacThreshold},
            {"ransacConfidence", ransacConfidence},
            {"ransacMaxIterations", ransacMaxIterations},
            {"minLinesForSolve", minLinesForSolve},
            {"minPointsPerLine", minPointsPerLine},
            {"enableTiming", enableTiming}
        };
    }

    static VirtualCameraPoseParams fromJson(const nlohmann::json& j) {
        VirtualCameraPoseParams p;
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("ransacThreshold"))
            p.ransacThreshold = j.at("ransacThreshold").get<double>();
        if (j.contains("ransacConfidence"))
            p.ransacConfidence = j.at("ransacConfidence").get<double>();
        if (j.contains("ransacMaxIterations"))
            p.ransacMaxIterations = j.at("ransacMaxIterations").get<int>();
        if (j.contains("minLinesForSolve"))
            p.minLinesForSolve = j.at("minLinesForSolve").get<int>();
        if (j.contains("minPointsPerLine"))
            p.minPointsPerLine = j.at("minPointsPerLine").get<int>();
        if (j.contains("enableTiming"))
            p.enableTiming = j.at("enableTiming").get<bool>();
        p.validate();
        return p;
    }
};

// ============================================================================
// VirtualCameraPoseResult
// ============================================================================

struct VirtualCameraPoseResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Matx33d virtualK;
    cv::Matx33d virtualR;
    cv::Vec3d   virtualT;

    int numLines = 0;
    int totalEndpoints = 0;
    double avgLineFittingError = 0.0;
    double avgDistToCenter = 0.0;
    std::vector<int> lineInlierCounts;

    VirtualCameraPoseResult() = default;
    ~VirtualCameraPoseResult() = default;

    VirtualCameraPoseResult(VirtualCameraPoseResult&&) = default;
    VirtualCameraPoseResult& operator=(VirtualCameraPoseResult&&) = default;

    VirtualCameraPoseResult(const VirtualCameraPoseResult&) = delete;
    VirtualCameraPoseResult& operator=(const VirtualCameraPoseResult&) = delete;
};

// ============================================================================
// VirtualCameraPoseCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 端点/立体内外参按调用传入；Impl 仅持有每调用重置的暂存缓冲，SetParams 缓存 RANSAC 阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API VirtualCameraPoseCuda {
public:
    static constexpr const char* kLogTag = "10-VirtualCameraPoseCuda";

    explicit VirtualCameraPoseCuda(const VirtualCameraPoseParams& params = {});
    ~VirtualCameraPoseCuda();

    VirtualCameraPoseCuda(const VirtualCameraPoseCuda&) = delete;
    VirtualCameraPoseCuda& operator=(const VirtualCameraPoseCuda&) = delete;

    VirtualCameraPoseResult Execute(
        const cv::cuda::GpuMat& d_endpoints,
        const cv::cuda::GpuMat& d_line_ids,
        const cv::Matx33d& stereoK,
        const cv::Matx33d& stereoR,
        cv::cuda::Stream& stream);

    VirtualCameraPoseResult Execute(
        const cv::cuda::GpuMat& d_endpoints,
        const cv::cuda::GpuMat& d_line_ids,
        const cv::Matx33d& stereoK,
        const cv::Matx33d& stereoR);

    void Warmup(int numEndpoints, int maxLineId);
    void Warmup(const WarmupConfig& config);
    void SetParams(const VirtualCameraPoseParams& params);
    const VirtualCameraPoseParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getVirtualCameraPoseCudaInfo();

} // namespace calib