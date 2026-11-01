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
    /// 扫描方向: 0=竖切中心列(默认, 按交点Y排序), 1=横切中心行(按交点X排序)
    /// 斜姿态激光线(左右延伸不跨中心列)需用 1 才能切到全部线
    int scanDirection = 0;
    /// 横切模式下的中心行偏移 (row = rows/2 + centerRowOffset)
    int centerRowOffset = 0;
    /// 拆分扫描行上的粘连游程(仅横切生效, 默认 true):
    /// 膨胀导致相邻平行线连通域粘连时, 同一标签在扫描行上出现多个不连续游程段,
    /// 每段各自成线编号(按段最小 x 排序), 恢复正确线数
    bool splitMergedRuns = true;
    /// 真激光线甄别容差(像素, 0=关闭):
    /// 对扫描线上排序后的交点序列, 若当前交点与左右交点的间距差
    /// |dL-dR| <= 本容差 → 当前交点所在域为真激光线(局部等距判据);
    /// 确认的真线其左右紧邻域也判定为真线(端点传播)。
    /// 域数 <3 时不甄别(无法构成左-中-右)。剔除域在输出掩膜中置 0。
    int realLineTolerance = 0;
    /// 期望激光线条数门禁(0=off, >0 启用):
    /// 编号后有效线数(componentCount) != 该值 → result.frameRejected=true。
    /// 防护: 编号按交点位置顺序分配(无绝对锚定), 中部缺线/粘连将导致
    /// 后续线整体错位——线数不符的帧必须整帧弃用(调用方按帧跳过 L+R)。
    int expectedLineCount = 0;
    /// 扫描线切割游程宽度门禁(像素, 0=off, >0 启用):
    /// 任一标签在扫描线上的展布(末游程末端-首游程起点, 含游程间隙)
    /// > 该值 → 判定多条激光线粘连(如高光过曝融合) → frameRejected=true。
    /// 测量在 splitMergedRuns 拆分之前(原始游程), 防粘连被拆分"凑回"
    /// 正确线数而漏检。正常线宽 ~12-22px(1688x1400 斜线形态实测),
    /// 双线粘连 >=38px, 建议阈值 30。
    int maxRunWidth = 0;
    int deviceId = 0;

    void validate() const {
        if (maxLabels < 1 || maxLabels > 4096)
            throw std::invalid_argument("LaserLabelParams::maxLabels must be [1, 4096]");
        if (centerColOffset < -500 || centerColOffset > 500)
            throw std::invalid_argument("LaserLabelParams::centerColOffset must be [-500, 500]");
        if (scanDirection != 0 && scanDirection != 1)
            throw std::invalid_argument("LaserLabelParams::scanDirection must be 0 or 1");
        if (centerRowOffset < -500 || centerRowOffset > 500)
            throw std::invalid_argument("LaserLabelParams::centerRowOffset must be [-500, 500]");
        if (realLineTolerance < 0)
            throw std::invalid_argument("LaserLabelParams::realLineTolerance must be >= 0 (0=off)");
        if (expectedLineCount < 0 || expectedLineCount > maxLabels)
            throw std::invalid_argument("LaserLabelParams::expectedLineCount must be [0, maxLabels] (0=off)");
        if (maxRunWidth < 0)
            throw std::invalid_argument("LaserLabelParams::maxRunWidth must be >= 0 (0=off)");
        if (deviceId < 0)
            throw std::invalid_argument("LaserLabelParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"maxLabels", maxLabels},
            {"centerColOffset", centerColOffset},
            {"scanDirection", scanDirection},
            {"centerRowOffset", centerRowOffset},
            {"splitMergedRuns", splitMergedRuns},
            {"realLineTolerance", realLineTolerance},
            {"expectedLineCount", expectedLineCount},
            {"maxRunWidth", maxRunWidth},
            {"deviceId", deviceId}
        };
    }

    static LaserLabelParams fromJson(const nlohmann::json& j) {
        LaserLabelParams p;
        if (j.contains("maxLabels")) p.maxLabels = j.at("maxLabels").get<int>();
        if (j.contains("centerColOffset")) p.centerColOffset = j.at("centerColOffset").get<int>();
        if (j.contains("scanDirection")) p.scanDirection = j.at("scanDirection").get<int>();
        if (j.contains("centerRowOffset")) p.centerRowOffset = j.at("centerRowOffset").get<int>();
        if (j.contains("splitMergedRuns")) p.splitMergedRuns = j.at("splitMergedRuns").get<bool>();
        if (j.contains("realLineTolerance")) p.realLineTolerance = j.at("realLineTolerance").get<int>();
        if (j.contains("expectedLineCount")) p.expectedLineCount = j.at("expectedLineCount").get<int>();
        if (j.contains("maxRunWidth")) p.maxRunWidth = j.at("maxRunWidth").get<int>();
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

    /// 门禁诊断: 编号后有效激光线条数(与 componentCount 同值, 语义化字段)
    int lineCount = 0;
    /// 数据质量门禁拒绝标志(非执行错误, success 仍为 true):
    /// expectedLineCount/maxRunWidth 门禁任一触发时置 true, 调用方应整帧弃用
    bool frameRejected = false;
    /// 拒绝原因(多原因以 "; " 连接, 如 "runWidth=132>30 label=5; lineCount=23!=25")
    std::string rejectReason;

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