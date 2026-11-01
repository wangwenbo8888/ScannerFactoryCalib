/**
 * @file steger_types.h
 * @brief Steger 算子族共享类型（GroupMode / StegerParams / StegerResult）唯一驻点
 *
 * 2026-09-04 自 steger/steger_extract_cuda.h 原文抽出（定义逐字未动）：
 * steger_fast 借用原版头取类型的传递依赖就此解除——steger_fast 仅依赖本头，
 * retired/steger 目录整体缺席亦可独立编译；原版算子头同样改 include 本头，
 * 两版类型仍是同一份定义（对拍测试无重定义风险）。
 *
 * ⚠ 09 侧同步债务：modules/09_operatorlib 的 steger/steger_fast 仍为原布局
 * （类型定义在 steger_extract_cuda.h），将来同步需对齐本次抽头。
 */

#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

// ============================================================================
// GroupMode
// ============================================================================

enum class GroupMode {
    ByLabel,
    Flat
};

// ============================================================================
// StegerParams
// ============================================================================

struct StegerParams {
    float sigma = 1.5f;
    int kernelSize = 0;
    float lowThreshold = 2.0f;
    float highThreshold = 0.0f;
    /// 逐点宽度门禁(px, 平滑后口径 w=sqrt(g/|λ|), 0=off):
    /// 过曝局部变宽/平顶剖面 → w 显著大于正常线 → 该点不提取中心
    /// (防亚像素中心偏移噪点)。off 时算子与不加门禁逐位一致。
    /// 注意口径: w 含平滑核 σ 贡献(w²≈σ线²+σ_filter²), 正常线(全宽4-5px)
    /// 在 σ=1.5 下 w≈2.4, 阈值须按平滑后尺度标定(非原始线宽)。
    float maxLineWidth = 0.0f;
    int maxLabels = 256;
    int deviceId = 0;
    /// 平顶双点去重（0=Off 默认——与历史行为逐位一致；1=Centroid——同簇
    /// 2 点均值/≥3 点中位合并；2=HalfMax 号位保留未实现）。判据常量编译期
    /// 固定（见 steger_fast_impl.cu）：along≤0.5 ∧ normal≤1.5、簇≤8、跨≤2.5px
    int dedupMode = 0;

    void validate() const {
        if (sigma < 0.5f || sigma > 10.0f)
            throw std::invalid_argument("StegerParams::sigma must be [0.5, 10.0]");
        if (kernelSize != 0 && kernelSize != 3 && kernelSize != 5
            && kernelSize != 7 && kernelSize != 9)
            throw std::invalid_argument("StegerParams::kernelSize must be 0 (auto), 3, 5, 7, or 9");
        if (lowThreshold < 0.0f)
            throw std::invalid_argument("StegerParams::lowThreshold must be >= 0");
        if (highThreshold < 0.0f)
            throw std::invalid_argument("StegerParams::highThreshold must be >= 0");
        if (maxLineWidth < 0.0f)
            throw std::invalid_argument("StegerParams::maxLineWidth must be >= 0 (0=off)");
        if (maxLabels < 1 || maxLabels > 4096)
            throw std::invalid_argument("StegerParams::maxLabels must be [1, 4096]");
        if (deviceId < 0)
            throw std::invalid_argument("StegerParams::deviceId must be >= 0");
        if (dedupMode != 0 && dedupMode != 1)
            throw std::invalid_argument("StegerParams::dedupMode must be 0 (off) or 1 (centroid)");
    }

    nlohmann::json toJson() const {
        return {
            {"sigma", sigma},
            {"kernelSize", kernelSize},
            {"lowThreshold", lowThreshold},
            {"highThreshold", highThreshold},
            {"maxLineWidth", maxLineWidth},
            {"maxLabels", maxLabels},
            {"deviceId", deviceId},
            {"dedupMode", dedupMode}
        };
    }

    static StegerParams fromJson(const nlohmann::json& j) {
        StegerParams p;
        if (j.contains("sigma")) p.sigma = j.at("sigma").get<float>();
        if (j.contains("kernelSize")) p.kernelSize = j.at("kernelSize").get<int>();
        if (j.contains("lowThreshold")) p.lowThreshold = j.at("lowThreshold").get<float>();
        if (j.contains("highThreshold")) p.highThreshold = j.at("highThreshold").get<float>();
        if (j.contains("maxLineWidth")) p.maxLineWidth = j.at("maxLineWidth").get<float>();
        if (j.contains("maxLabels")) p.maxLabels = j.at("maxLabels").get<int>();
        if (j.contains("deviceId")) p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("dedupMode")) p.dedupMode = j.at("dedupMode").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// StegerResult
// ============================================================================

struct StegerResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::map<int, std::vector<cv::Point2f>> centerPoints;
    int totalPointCount = 0;
    int lineCount = 0;
    int mergedCount = 0;    // dedup 合并掉的原有点数（dedupMode=0 恒 0）

    std::shared_ptr<cv::cuda::GpuMat> d_centerPoints;
    std::shared_ptr<cv::cuda::GpuMat> d_line_ids;

    StegerResult() = default;
    ~StegerResult() = default;

    StegerResult(StegerResult&&) = default;
    StegerResult& operator=(StegerResult&&) = default;

    StegerResult(const StegerResult&) = delete;
    StegerResult& operator=(const StegerResult&) = delete;
};

} // namespace calib
