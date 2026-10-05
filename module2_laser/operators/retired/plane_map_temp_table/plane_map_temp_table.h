#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct LineMapStats;

struct PlaneMapTempTableParams {
    cv::Mat cameraMatrixL;
    cv::Mat distCoeffsL;
    cv::Mat cameraMatrixR;
    cv::Mat distCoeffsR;
    cv::Size imageSize;

    cv::Mat R;
    cv::Mat T;

    cv::Matx33d virtualK;
    cv::Matx33d virtualR;
    cv::Vec3d virtualT;

    std::vector<int> lineIds;

    double referenceTemp = 25.0;
    double cte = 23.6e-6;
    double tempStep = 0.2;
    double tempRangeMin = -10.0;
    double tempRangeMax = 10.0;

    double alpha = 0.0;
    int flags = 1;

    int deviceId = 0;
    float gridStep = 0.5f;
    float depthMin = 100.0f;
    float depthMax = 5000.0f;
    int depthSamples = 200;
    float epipolarStep = 0.5f;

    void validate() const;

    nlohmann::json toJson() const;
    static PlaneMapTempTableParams fromJson(const nlohmann::json& j);
};

struct PlaneMapTempEntry {
    double temperature = 0.0;
    double deltaT = 0.0;

    cv::Mat compensatedCameraMatrixL;
    cv::Mat compensatedCameraMatrixR;
    cv::Mat compensatedT;
    cv::Mat R1, R2, P1, P2, Q;

    std::shared_ptr<cv::cuda::GpuMat> d_left_to_right;
    std::shared_ptr<cv::cuda::GpuMat> d_right_u;
    int totalPairs = 0;
    std::vector<LineMapStats> lineStats;

    nlohmann::json toJson() const;
};

struct PlaneMapTempTableResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double referenceTemp = 0.0;
    double cte = 0.0;
    int tableSize = 0;

    std::vector<PlaneMapTempEntry> table;

    PlaneMapTempTableResult() = default;
    ~PlaneMapTempTableResult() = default;

    PlaneMapTempTableResult(PlaneMapTempTableResult&&) = default;
    PlaneMapTempTableResult& operator=(PlaneMapTempTableResult&&) = default;

    PlaneMapTempTableResult(const PlaneMapTempTableResult&) = delete;
    PlaneMapTempTableResult& operator=(const PlaneMapTempTableResult&) = delete;

    nlohmann::json toJson() const;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 一次性按温度批量求解平面映射表并返回；SetParams 缓存内外参/虚拟相机/温度参数等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API PlaneMapTempTable {
public:
    static constexpr const char* kLogTag = "13-PlaneMapTempTable";

    explicit PlaneMapTempTable(const PlaneMapTempTableParams& params = {});
    ~PlaneMapTempTable();

    PlaneMapTempTable(const PlaneMapTempTable&) = delete;
    PlaneMapTempTable& operator=(const PlaneMapTempTable&) = delete;

    PlaneMapTempTableResult Execute();

    void SetParams(const PlaneMapTempTableParams& params);
    const PlaneMapTempTableParams& GetParams() const;

    void Warmup() { }

    void Destroy();

private:
    PlaneMapTempTableParams params_;
};

OperatorInfo getPlaneMapTempTableInfo();

} // namespace calib