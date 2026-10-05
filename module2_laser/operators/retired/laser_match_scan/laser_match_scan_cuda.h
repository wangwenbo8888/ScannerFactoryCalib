/**
 * @file laser_match_scan_cuda.h
 * @brief 激光线匹配扫描CUDA算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（epipolar_interp_cuda 之后�? * 平台：GPU（CUDA�? *
 * 功能：对左右相机的亚像素激光中心点集进行匹配�? *       根据当前温度从温度补偿映射表（第13步输出）查表�? *       预测每条激光线在右相机的位置，在阈值范围内寻找唯一匹配�? *
 * 匹配规则�? *   - 恰好 1 个右相机候�?�?匹配成功
 *   - 0�?�? �?�?标记排除（后续三维重建不参与�? *   - 已标记占用的右点不再参与后续匹配
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/calib_result_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================================
// LaserMatchScanParams
// ============================================================================

struct LaserMatchScanParams {
    float match_threshold = 1.0f;
    float epipolar_row_step = 0.5f;
    int max_right_per_row = 1024;
    float vL_tolerance = 0.01f;
    int deviceId = 0;

    void validate() const {
        if (match_threshold <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParams::match_threshold must be > 0");
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParams::epipolar_row_step must be > 0");
        if (max_right_per_row <= 0)
            throw std::invalid_argument("LaserMatchScanParams::max_right_per_row must be > 0");
        if (vL_tolerance < 0.0f)
            throw std::invalid_argument("LaserMatchScanParams::vL_tolerance must be >= 0");
        if (deviceId < 0)
            throw std::invalid_argument("LaserMatchScanParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"match_threshold", match_threshold},
            {"epipolar_row_step", epipolar_row_step},
            {"max_right_per_row", max_right_per_row},
            {"vL_tolerance", vL_tolerance},
            {"deviceId", deviceId}
        };
    }

    static LaserMatchScanParams fromJson(const nlohmann::json& j) {
        LaserMatchScanParams p;
        if (j.contains("match_threshold"))
            p.match_threshold = j.at("match_threshold").get<float>();
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("max_right_per_row"))
            p.max_right_per_row = j.at("max_right_per_row").get<int>();
        if (j.contains("vL_tolerance"))
            p.vL_tolerance = j.at("vL_tolerance").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserMatchScanResult
// ============================================================================

struct LaserMatchScanResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::shared_ptr<cv::cuda::GpuMat> d_matched_left;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_right;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_line_ids;
    int matchedCount = 0;

    std::shared_ptr<cv::cuda::GpuMat> d_left_status;
    std::shared_ptr<cv::cuda::GpuMat> d_right_status;

    int totalLeftPoints = 0;
    int totalRightPoints = 0;
    int excludedLeftCount = 0;
    int excludedRightCount = 0;

    LaserMatchScanResult() = default;
    ~LaserMatchScanResult() = default;

    LaserMatchScanResult(LaserMatchScanResult&&) = default;
    LaserMatchScanResult& operator=(LaserMatchScanResult&&) = default;

    LaserMatchScanResult(const LaserMatchScanResult&) = delete;
    LaserMatchScanResult& operator=(const LaserMatchScanResult&) = delete;
};

// ============================================================================
// LaserMatchScanCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 温度补偿映射表与当前温度由调用方经 SetTempTable（或 JSON 便捷重载 LoadTempTable）/SetCurrentTemperature 注入，影响匹配结果；实例不持有跨调用累积状态。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserMatchScanCuda {
public:
    static constexpr const char* kLogTag = "07-LaserMatchScanCuda";

    explicit LaserMatchScanCuda(const LaserMatchScanParams& params = {});
    ~LaserMatchScanCuda();

    LaserMatchScanCuda(const LaserMatchScanCuda&) = delete;
    LaserMatchScanCuda& operator=(const LaserMatchScanCuda&) = delete;

    bool LoadTempTable(const std::string& jsonPath);

    // 算子规范 §3.6：只读共享资源（温度补偿映射表）经注入传入，算子内部不写入。
    // 内部将 LaserPlaneMapTempTable 转为 GPU 查表所需的扁平表示。
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);

    void SetCurrentTemperature(double temperature);

    LaserMatchScanResult Execute(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream);

    LaserMatchScanResult Execute(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids);

    void Destroy();

    void Warmup(int maxLeftPoints, int maxRightPoints);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserMatchScanParams& params);
    const LaserMatchScanParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserMatchScanCudaInfo();

} // namespace calib