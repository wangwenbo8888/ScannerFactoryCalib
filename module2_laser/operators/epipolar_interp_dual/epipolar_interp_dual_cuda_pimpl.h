/**
 * @file epipolar_interp_dual_cuda_pimpl.h
 * @brief 激光中心点极线插值CUDA算子·双模式版 - 内部桥接头文件
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <atomic>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "epipolar_interp_dual_cuda.h"

namespace calib {

struct EpipolarInterpDualCuda::Impl {
    static constexpr int SCAN_MAX_ROWS = 8192;     // 扫描模式固定行网格（同 match 算子约定）
    static constexpr int SCAN_ROW_PTS_CAP = 256;   // 每行窗口内点收集上限（smem 2KB）
    static constexpr int SCAN_SLOTS = 32;          // 每行输出簇上限

    EpipolarInterpDualParams params_;

    // ── 排序（Labeled: 64bit (fid,y)；Scan: 32bit y）──
    cv::cuda::GpuMat d_keys_in_, d_keys_out_;      // CV_64FC1
    cv::cuda::GpuMat d_ykeys_in_, d_ykeys_out_;    // CV_32SC1
    cv::cuda::GpuMat d_idx_in_, d_idx_out_;
    cv::cuda::GpuMat d_sorted_pts_, d_sorted_fids_;
    void* d_radix_temp_ = nullptr; size_t radix_temp_size_ = 0; int radix_cap_ = 0;

    // ── Labeled: 线号表 ──
    cv::cuda::GpuMat d_rl_flags_, d_rl_ranks_;
    cv::cuda::GpuMat d_line_begin_, d_line_ids_map_;
    void* d_scan_temp_ = nullptr; size_t scan_temp_size_ = 0; int scan_cap_ = 0;

    // ── 元信息 {num_lines/umin/umax 或 id_min/id_max} ──
    cv::cuda::GpuMat d_meta_;
    int* h_meta_pinned_ = nullptr;

    // ── Labeled 插值输出（temp+compact）──
    cv::cuda::GpuMat d_flags_, d_temp_interp_, d_temp_fids_;
    cv::cuda::GpuMat d_output_, d_output_fids_, d_output_count_;
    int lines_cap_ = 0;

    // ── Scan 插值输出（[row*SCAN_SLOTS + slot] 布局 + compact）──
    cv::cuda::GpuMat d_scan_flags_, d_scan_out_pts_, d_scan_out_fids_;
    cv::cuda::GpuMat d_scan_cpts_, d_scan_cfids_, d_scan_ccount_;

    // ── Scan 段链接（行间簇合并 → 段 ID，pointer doubling）──
    cv::cuda::GpuMat d_link_parent_, d_link_label_;

    void* d_cub_temp_ = nullptr; size_t cub_temp_size_ = 0; int cub_cap_ = 0;

    int capacity_ = 0;
    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EpipolarInterpDualParams& params);
    ~Impl();

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids,
                                 cv::cuda::Stream& stream);

    void Warmup(int pointCount);
    void SetParams(const EpipolarInterpDualParams& params);
    const EpipolarInterpDualParams& GetParams() const { return params_; }

    void ensureCapacity(int pointCount);
    void Destroy();
};

} // namespace calib
