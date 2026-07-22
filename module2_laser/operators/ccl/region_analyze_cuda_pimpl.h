/**
 * @file region_analyze_cuda_pimpl.h
 * @brief 激光连通域分析算子 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 全 GPU 管线（精简版，仅包围盒统计）:
 *   CCL -> initStats(5数组) -> computeStats(5原子) -> buildRemap -> relabel -> compactStats
 *   无 minMaxLoc，无质心计算，pinned memory D2H
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <atomic>
#include <cassert>
#include <vector>
#include <climits>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "region_analyze_cuda.h"

namespace calib {

// ============================================================================
// GPU 端统计结果结构（与 ComponentStats 内存布局完全一致，可 memcpy）
// ============================================================================

struct GPUComponentResult {
    int label;
    int bbox_x;
    int bbox_y;
    int bbox_w;
    int bbox_h;
};

static_assert(sizeof(GPUComponentResult) == sizeof(ComponentStats),
              "GPUComponentResult must match ComponentStats layout");

// ============================================================================
// RegionAnalyzerCUDA::Impl 完整声明
// ============================================================================

struct RegionAnalyzerCUDA::Impl {
    RegionAnalyzerParams params_;

    // ── GPU 图像缓冲 ──
    cv::cuda::GpuMat d_labels_;        // CCL 输出标签图 (CV_32SC1)
    cv::cuda::GpuMat d_relabeled_;     // 重编号输出标签图 (CV_32SC1)

    // ── GPU 统计缓冲（raw CUDA memory，按 stats_capacity_ 分配）──
    int*                d_areas_     = nullptr;  // 面积（仅用于 buildRemap 过滤）
    int*                d_min_x_     = nullptr;
    int*                d_max_x_     = nullptr;
    int*                d_min_y_     = nullptr;
    int*                d_max_y_     = nullptr;
    int*                d_remap_     = nullptr;  // 稀疏->稠密标签映射表
    int*                d_num_valid_ = nullptr;  // 过滤后有效连通域计数
    GPUComponentResult* d_compact_   = nullptr;  // 压缩后的统计结果

    // ── Pinned host memory（降低 D2H 延迟）──
    int*                h_pinned_count_   = nullptr;  // pinned int
    GPUComponentResult* h_pinned_compact_ = nullptr;  // pinned array

    int stats_capacity_ = 0;           // 当前已分配的 max_label+1 容量
    static constexpr int COMPACT_MAX = 65536;  // d_compact_ 最大条目

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;
    int old_device_id = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const RegionAnalyzerParams& params);
    ~Impl() = default;

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    RegionAnalysisResult Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_mask,
                                 cv::cuda::Stream& stream);

    RegionAnalysisResult Execute(const cv::cuda::GpuMat& d_inputBinaryMask,
                                 cv::cuda::Stream& stream);

    void Warmup(int rows, int cols);

    void SetParams(const RegionAnalyzerParams& params);

    const RegionAnalyzerParams& GetParams() const { return params_; }

    void allocateStatsBuffers(int max_labels);
    void freeStatsBuffers();
};

} // namespace calib
