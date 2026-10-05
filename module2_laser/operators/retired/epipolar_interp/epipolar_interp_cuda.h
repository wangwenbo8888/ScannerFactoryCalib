/**
 * @file epipolar_interp_cuda.h
 * @brief 激光中心点极线插值CUDA算子 - 公开头文件（纯 C++，不含 CUDA 类型）
 *
 * 所属流程：激光标定流程4-6 / 扫描流程（undistort_points_cuda 之后）
 * 平台：GPU（CUDA）
 * 功能：极线驱动重采样——对每条极线栅格行 y_k = k*epipolar_row_step，
 *       在上下搜索窗内各取最近同线点，连线与极线交点作为插值点；
 *       点恰在极线上时直接采用。输出仅含结果点（原始输入点不透传）。
 * 精度容差档次：档次③（亚像素/浮点类）
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================================
// EpipolarInterpParams
// ============================================================================

struct EpipolarInterpParams {
    /// 极线栅格行距（像素）。y_k = k * epipolar_row_step
    float epipolar_row_step = 0.7f;
    /// 上下搜索窗半径（像素）。上窗 [y_k - window_offset, y_k)，下窗 (y_k, y_k + window_offset]。
    /// 默认 -1 表示取 epipolar_row_step（上下各一格）。
    float window_offset = -1.0f;
    /// 上下两选点 X 差上限（像素），超限视为跨段/断裂，放弃该极线
    float max_x_diff = 1.0f;
    int deviceId = 0;
    /// true=标定（同 line_id 才互相插值）; false=扫描（只按几何，忽略线号）
    bool lineIdCheck = true;

    /// 实际生效的窗口半径（window_offset<=0 时取 epipolar_row_step）
    float effectiveWindow() const {
        return window_offset > 0.0f ? window_offset : epipolar_row_step;
    }

    void validate() const {
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("EpipolarInterpParams::epipolar_row_step must be > 0");
        if (window_offset <= 0.0f && window_offset != -1.0f)
            throw std::invalid_argument("EpipolarInterpParams::window_offset must be > 0 or -1 (auto)");
        if (max_x_diff <= 0.0f)
            throw std::invalid_argument("EpipolarInterpParams::max_x_diff must be > 0");
        if (deviceId < 0)
            throw std::invalid_argument("EpipolarInterpParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"epipolar_row_step", epipolar_row_step},
            {"window_offset", window_offset},
            {"max_x_diff", max_x_diff},
            {"deviceId", deviceId},
            {"lineIdCheck", lineIdCheck}
        };
    }

    static EpipolarInterpParams fromJson(const nlohmann::json& j) {
        EpipolarInterpParams p;
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("window_offset"))
            p.window_offset = j.at("window_offset").get<float>();
        if (j.contains("max_x_diff"))
            p.max_x_diff = j.at("max_x_diff").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("lineIdCheck"))
            p.lineIdCheck = j.at("lineIdCheck").get<bool>();
        p.validate();
        return p;
    }
};

// ============================================================================
// EpipolarInterpResult
// ============================================================================

struct EpipolarInterpResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_interpPoints;
    std::shared_ptr<cv::cuda::GpuMat> d_interp_line_ids;
    int interpCount = 0;
    // [dual·Scan 专用诊断] 段链接相位（行间簇连通计算：dualScanLinkParent + pointer
    // doubling）的 GPU 耗时（ms）。仅 EpipolarInterpDualCuda Scan 模式且 scan_link_dx>0
    // 时非零；Labeled 路径与其余使用者恒为 0（默认值，行为无变化）。
    double scan_link_ms = 0.0;

    EpipolarInterpResult() = default;
    ~EpipolarInterpResult() = default;

    EpipolarInterpResult(EpipolarInterpResult&&) = default;
    EpipolarInterpResult& operator=(EpipolarInterpResult&&) = default;

    EpipolarInterpResult(const EpipolarInterpResult&) = delete;
    EpipolarInterpResult& operator=(const EpipolarInterpResult&) = delete;
};

// ============================================================================
// EpipolarInterpCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存极线步距/窗口等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API EpipolarInterpCuda {
public:
    static constexpr const char* kLogTag = "06-EpipolarInterpCuda";

    explicit EpipolarInterpCuda(const EpipolarInterpParams& params = {});
    ~EpipolarInterpCuda();

    EpipolarInterpCuda(const EpipolarInterpCuda&) = delete;
    EpipolarInterpCuda& operator=(const EpipolarInterpCuda&) = delete;

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids,
                                 cv::cuda::Stream& stream);

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids);

    void Destroy();
    void Warmup(int pointCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const EpipolarInterpParams& params);
    const EpipolarInterpParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getEpipolarInterpInfo();

} // namespace calib
