/**
 * @file region_analyze_cuda.h
 * @brief 激光标记点区域连通域分析算子 - 公开头文件（纯 C++，不含 CUDA 类型）
 *
 * 所属流程：标定流程1❺ / 标定流程2❻（共用）
 * 平台：GPU（OpenCV CUDA API）
 *
 * 功能：对激光二值掩膜执行全 GPU 连通域标记（CCL, connectivity=8），
 *       统计计算、面积过滤、重编号全部在 GPU 上完成，
 *       仅下载极小的统计结果到 CPU。
 *
 * 精度容差档次：档次②（整像素/几何类）
 * - CPU vs CUDA 误差 < 0.1px（或整像素一致性）
 * - 跨 SM 架构间 CUDA 互比差异 < 0.1px
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

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================================
// RegionAnalyzerParams
// ============================================================================

/**
 * @brief 连通域分析参数
 *
 * 异常行为：
 * - fromJson() 字段缺失或类型不匹配 → 抛出 std::invalid_argument
 * - fromJson() 值超出合法范围 → 抛出 std::invalid_argument（与 validate() 一致）
 * - fromJson() 未知字段 → 忽略（不抛异常），保证前向兼容
 */
struct RegionAnalyzerParams {
    int minArea = 100;       ///< 最小面积阈值（像素）
    int maxArea = 100000;    ///< 最大面积阈值（像素）
    int deviceId = 0;        ///< GPU 设备 ID（多 GPU 场景），默认设备 0

    /**
     * @brief 参数合法性校验
     * @throws std::invalid_argument 参数不合法
     */
    void validate() const {
        if (minArea < 0)
            throw std::invalid_argument("RegionAnalyzerParams::minArea must be >= 0");
        if (maxArea <= minArea)
            throw std::invalid_argument("RegionAnalyzerParams::maxArea must be > minArea");
        if (deviceId < 0)
            throw std::invalid_argument("RegionAnalyzerParams::deviceId must be >= 0");
    }

    /**
     * @brief 序列化为 JSON
     */
    nlohmann::json toJson() const {
        return {
            {"minArea", minArea},
            {"maxArea", maxArea},
            {"deviceId", deviceId}
        };
    }

    /**
     * @brief 从 JSON 反序列化
     * @throws std::invalid_argument 参数不合法
     */
    static RegionAnalyzerParams fromJson(const nlohmann::json& j) {
        RegionAnalyzerParams p;
        if (j.contains("minArea")) p.minArea = j.at("minArea").get<int>();
        if (j.contains("maxArea")) p.maxArea = j.at("maxArea").get<int>();
        if (j.contains("deviceId")) p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// ComponentStats
// ============================================================================

/**
 * @brief 单个连通域的统计信息（仅包围盒，无面积/质心）
 */
struct ComponentStats {
    int label;               ///< 连通域标签 ID（1-indexed，对应 d_labeledMask 中的值）
    int boundingBoxX;        ///< 包围盒左上角 X
    int boundingBoxY;        ///< 包围盒左上角 Y
    int boundingBoxWidth;    ///< 包围盒宽
    int boundingBoxHeight;   ///< 包围盒高
};

// ============================================================================
// RegionAnalysisResult
// ============================================================================

/**
 * @brief 连通域分析结果
 *
 * - success: 处理是否成功（二值快速分支）
 * - message: 错误或辅助信息（success=false 时保证非空）
 * - qualityFlag: 质量标记（success=true 时有效，三值精细报告）
 *   - Normal:   连通域数量合理
 *   - Degraded: 连通域数量异常多（>200）
 *   - Warning:  连通域数量为 0（仍 success=true，下游可处理空结果）
 *
 * 设计决策：
 * - 使用 shared_ptr<GpuMat> 传递 GPU 数据所有权（pImpl 隔离需要）
 */
struct RegionAnalysisResult {
    bool success = false;                                         ///< 处理是否成功
    std::string message;                                          ///< 错误或辅助信息
    QualityFlag qualityFlag = QualityFlag::Normal;                ///< 质量标记

    std::shared_ptr<cv::cuda::GpuMat> d_labeledMask;  ///< GPU 连通域标记掩膜（CV_32SC1, 1-indexed）
    int componentCount = 0;                            ///< 过滤后连通域数量
    std::vector<ComponentStats> components;            ///< 各连通域统计信息

    RegionAnalysisResult() = default;
    ~RegionAnalysisResult() = default;

    std::vector<cv::Rect> toRectList() const {
        std::vector<cv::Rect> rects;
        rects.reserve(components.size());
        for (const auto& cs : components) {
            rects.emplace_back(cs.boundingBoxX, cs.boundingBoxY,
                               cs.boundingBoxWidth, cs.boundingBoxHeight);
        }
        return rects;
    }

    RegionAnalysisResult(RegionAnalysisResult&&) = default;
    RegionAnalysisResult& operator=(RegionAnalysisResult&&) = default;

    RegionAnalysisResult(const RegionAnalysisResult&) = delete;
    RegionAnalysisResult& operator=(const RegionAnalysisResult&) = delete;
};

// ============================================================================
// RegionAnalyzerCUDA
// ============================================================================

/**
 * @brief 激光连通域分析算子（CUDA 实现）
 *
 * 使用 pImpl 模式隔离 CUDA 类型，公开头文件不含任何 CUDA 依赖。
 *
 * 线程安全约束：
 * - 非线程安全：不同线程不得同时调用 Execute() 或 setParams()
 * - 单线程流水线中应在帧间隙调用 setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 * - Debug 模式下维护 inProcess_ 原子变量进行断言检查
 * - Release 模式下并发调用为未定义行为（数据竞争）
 */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存面积阈值等只读配置（可改为按调用传入），无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API RegionAnalyzerCUDA {
public:
    static constexpr const char* kLogTag = "07-RegionAnalyzerCUDA";

    /**
     * @brief 构造函数
     * @param params 连通域分析参数
     * @throws std::invalid_argument 参数校验失败
     */
    explicit RegionAnalyzerCUDA(const RegionAnalyzerParams& params = {});

    ~RegionAnalyzerCUDA();

    RegionAnalyzerCUDA(const RegionAnalyzerCUDA&) = delete;
    RegionAnalyzerCUDA& operator=(const RegionAnalyzerCUDA&) = delete;

    /**
     * @brief 连通域分析（核心方法 - shared_ptr 重载，推荐）
     *
     * 接受 shared_ptr<GpuMat>，通过引用计数自动延长上游 GPU 数据生命周期。
     * 避免上游 Result 析构导致悬空引用。
     *
     * @param d_mask GPU 二值掩膜的 shared_ptr（CV_8UC1）
     * @param stream CUDA 流引用（流水线调用必须显式传入）
     * @return RegionAnalysisResult 包含标记掩膜和统计信息
     */
    RegionAnalysisResult Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_mask,
                                 cv::cuda::Stream& stream);

    /**
     * @brief 连通域分析（const GpuMat& 重载，向后兼容）
     *
     * 输入 GPU 二值掩膜（CV_8UC1，由 06 输出），执行 GPU CCL + CPU 统计。
     * 输出 GPU 标记掩膜（CV_32SC1, 1-indexed）和 CPU 端连通域统计信息。
     *
     * @warning 调用方必须保证 d_inputBinaryMask 在 Execute() 返回前始终有效。
     *          推荐使用 shared_ptr 重载以自动管理生命周期。
     *
     * @param d_inputBinaryMask GPU 二值掩膜（CV_8UC1）
     * @param stream CUDA 流引用（流水线调用必须显式传入）
     * @return RegionAnalysisResult 包含标记掩膜和统计信息
     */
    RegionAnalysisResult Execute(const cv::cuda::GpuMat& d_inputBinaryMask,
                                 cv::cuda::Stream& stream);

    /**
     * @brief 连通域分析（无 stream 重载，向后兼容）
     *
     * 使用 thread_local 默认流，仅用于测试/调试。
     * 流水线调用必须使用带 Stream& 的重载。
     */
    RegionAnalysisResult Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_mask);
    RegionAnalysisResult Execute(const cv::cuda::GpuMat& d_inputBinaryMask);

    /**
     * @brief 销毁所有预分配资源（GPU 缓冲等）
     *
     * 必须 idempotent（多次调用安全、无副作用）。
     * 析构前必须调用 Destroy()。
     */
    void Destroy();

    /**
     * @brief 预热 GPU 资源
     * @param rows 图像行数
     * @param cols 图像列数
     *
     * 预分配 GPU 缓冲区，首次 Execute() 无冷启动延迟。
     * 返回 void 而非 Result：在初始化阶段调用，失败即不可恢复。
     * @throws std::runtime_error 显存分配失败
     */
    void Warmup(int rows, int cols);

    /**
     * @brief 使用统一配置预热
     * @param config 预热配置（来自 common/calib_warmup_config.h）
     */
    void Warmup(const WarmupConfig& config);

    /**
     * @brief 动态更新参数
     * @param params 新参数
     * @throws std::invalid_argument 参数校验失败
     *
     * 缓冲区调整策略：对比新旧参数对应的 rows/cols
     * - 若新尺寸 > 旧尺寸 → 释放重新分配
     * - 若新尺寸 ≤ 旧尺寸 → 复用现有缓冲区不缩容
     *
     * @warning 不得与 Execute() 并发调用
     */
    void SetParams(const RegionAnalyzerParams& params);

    /**
     * @brief 获取当前参数
     */
    const RegionAnalyzerParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getRegionAnalyzerCUDAInfo();

} // namespace calib