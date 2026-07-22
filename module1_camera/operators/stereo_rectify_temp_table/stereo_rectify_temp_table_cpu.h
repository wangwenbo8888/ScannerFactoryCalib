/**
 * @file stereo_rectify_temp_table_cpu.h
 * @brief 温度补偿立体矫正参数�?CPU 算子
 *
 * 所属流程：标定 05
 * 平台：CPU
 *
 * 功能：根据温度补偿后的内参和外参，按温度步距批量计算
 *       每个温度点对应的立体矫正参数 (R1, R2, P1, P2, Q, validRoi)�? *       输出完整的温�?矫正参数对照表�? *
 * 数据流：
 *   wendu/01 内参补偿 + wendu/02 外参补偿 �?本算�?�?立体矫正参数�? *
 * 物理模型�? *   内参：scale = 1 + α × ΔT, fx(T)=fx₀×scale ...
 *   外参：T(T)=T₀×scale, R不变
 *   立体矫正：cv::stereoRectify(补偿后内参L, 补偿后内参R, R, T补偿, ...)
 */

#pragma once


#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct StereoRectifyTempTableParams {
    cv::Mat cameraMatrixL;
    cv::Mat distCoeffsL;
    cv::Mat cameraMatrixR;
    cv::Mat distCoeffsR;
    cv::Size imageSize;

    cv::Mat R;
    cv::Mat T;

    double referenceTemp = 25.0;
    double cte = 23.6e-6;
    double tempStep = 0.2;
    double tempRangeMin = -10.0;
    double tempRangeMax = 10.0;

    double alpha = 0.0;
    int flags = 1;

    void validate() const;

    nlohmann::json toJson() const;
    static StereoRectifyTempTableParams fromJson(const nlohmann::json& j);
};

struct StereoRectifyTempEntry {
    double temperature = 0.0;
    double deltaT = 0.0;

    cv::Mat compensatedCameraMatrixL;
    cv::Mat compensatedCameraMatrixR;
    cv::Mat compensatedT;

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoiLeft, validRoiRight;

    nlohmann::json toJson() const;
};

struct StereoRectifyTempTableResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double referenceTemp = 0.0;
    double cte = 0.0;
    int tableSize = 0;

    std::vector<StereoRectifyTempEntry> table;

    StereoRectifyTempTableResult() = default;
    ~StereoRectifyTempTableResult() = default;

    StereoRectifyTempTableResult(StereoRectifyTempTableResult&&) = default;
    StereoRectifyTempTableResult& operator=(StereoRectifyTempTableResult&&) = default;

    StereoRectifyTempTableResult(const StereoRectifyTempTableResult&) = delete;
    StereoRectifyTempTableResult& operator=(const StereoRectifyTempTableResult&) = delete;

    nlohmann::json toJson() const;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 一次性按温度批量求解矫正参数表并返回；SetParams 缓存内外参/温度参数等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API StereoRectifyTempTableCpu {
public:
    static constexpr const char* kLogTag = "05-StereoRectifyTempTableCpu";

    explicit StereoRectifyTempTableCpu(const StereoRectifyTempTableParams& params = {});
    ~StereoRectifyTempTableCpu();

    StereoRectifyTempTableCpu(const StereoRectifyTempTableCpu&) = delete;
    StereoRectifyTempTableCpu& operator=(const StereoRectifyTempTableCpu&) = delete;

    StereoRectifyTempTableResult Execute();

    void SetParams(const StereoRectifyTempTableParams& params);
    const StereoRectifyTempTableParams& GetParams() const;

    void Warmup() { }

    void Destroy();

private:
    StereoRectifyTempTableParams params_;
};

OperatorInfo getStereoRectifyTempTableCpuInfo();

} // namespace calib