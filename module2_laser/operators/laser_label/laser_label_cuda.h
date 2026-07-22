/**
 * @file laser_label_cuda.h
 * @brief 激光线编号算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（region_analyze_cuda 之后�? * 平台：GPU（CUDA + Thrust�? *
 * 功能：对连通域掩膜按中心列最�?Y 值重编号�?-indexed，Y 小→编号小）
 *       仅接�?CV_32SC1（由 02region_analyze_cuda 输出的已标记掩膜�? *       不再支持 CV_8UC1 输入，CCL 由上�?02 号算子负�? *
 * 精度容差档次：档次②（整像素/几何类）
 */

#pragma once


#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct WarmupConfig;

// ============================================================================
// LaserLabelParams
// ============================================================================

struct LaserLabelParams {
    int maxLabels = 256;
    int centerColOffset = 0;
    int deviceId = 0;

    void validate() const {
        if (maxLabels < 1 || maxLabels > 4096)
            throw std::invalid_argument("LaserLabelParams::maxLabels must be [1, 4096]");
        if (centerColOffset < -500 || centerColOffset > 500)
            throw std::invalid_argument("LaserLabelParams::centerColOffset must be [-500, 500]");
        if (deviceId < 0)
            throw std::invalid_argument("LaserLabelParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"maxLabels", maxLabels},
            {"centerColOffset", centerColOffset},
            {"deviceId", deviceId}
        };
    }

    static LaserLabelParams fromJson(const nlohmann::json& j) {
        LaserLabelParams p;
        if (j.contains("maxLabels")) p.maxLabels = j.at("maxLabels").get<int>();
        if (j.contains("centerColOffset")) p.centerColOffset = j.at("centerColOffset").get<int>();
        if (j.contains("deviceId")) p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserLabelResult
// ============================================================================

struct LaserLabelResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::shared_ptr<cv::cuda::GpuMat> d_labeledMask;
    int componentCount = 0;

    LaserLabelResult() = default;
    ~LaserLabelResult() = default;

    LaserLabelResult(LaserLabelResult&&) = default;
    LaserLabelResult& operator=(LaserLabelResult&&) = default;

    LaserLabelResult(const LaserLabelResult&) = delete;
    LaserLabelResult& operator=(const LaserLabelResult&) = delete;
};

// ============================================================================
// LaserLabelerCUDA
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存 maxLabels 等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserLabelerCUDA {
public:
    static constexpr const char* kLogTag = "09-LaserLabelerCUDA";

    explicit LaserLabelerCUDA(const LaserLabelParams& params = {});
    ~LaserLabelerCUDA();

    LaserLabelerCUDA(const LaserLabelerCUDA&) = delete;
    LaserLabelerCUDA& operator=(const LaserLabelerCUDA&) = delete;

    LaserLabelResult Execute(const cv::cuda::GpuMat& d_inputMask,
                              cv::cuda::Stream& stream);

    LaserLabelResult Execute(const cv::cuda::GpuMat& d_inputMask);

    void Warmup(int rows, int cols);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserLabelParams& params);
    const LaserLabelParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserLabelerCUDAInfo();

} // namespace calib