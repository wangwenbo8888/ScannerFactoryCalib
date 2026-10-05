/**
 * @file epipolar_interp_opt_cuda_pimpl.h
 * @brief 极线插值CUDA算子·优化版 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 * .cpp 和 .cu 均 include 此文件。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <memory>
#include <atomic>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "epipolar_interp_opt_cuda.h"

namespace calib {

struct EpipolarInterpOptCuda::Impl {
    EpipolarInterpParams params_;

    // Step1 排序（radix 双缓冲）
    cv::cuda::GpuMat d_keys_in_, d_keys_out_;    // CV_64FC1 (fid,y) 打包键
    cv::cuda::GpuMat d_idx_in_, d_idx_out_;      // CV_32SC1 索引
    cv::cuda::GpuMat d_sorted_pts_;              // CV_32FC2 排序后点
    cv::cuda::GpuMat d_sorted_fids_;             // CV_32SC1 排序后 fid

    // Step2 线号表（GPU 端构建，无 H2D）
    cv::cuda::GpuMat d_rl_flags_;                // CV_32SC1 边界标记
    cv::cuda::GpuMat d_rl_ranks_;                // CV_32SC1 前缀和秩
    cv::cuda::GpuMat d_line_begin_;              // CV_32SC1 [0..num_lines]
    cv::cuda::GpuMat d_line_ids_map_;            // CV_32SC1 去重升序线号
    cv::cuda::GpuMat d_meta_;                    // 1×4 CV_32SC1 {num_lines, umin, umax, -}

    // Step3/4
    cv::cuda::GpuMat d_flags_;
    cv::cuda::GpuMat d_temp_interp_;
    cv::cuda::GpuMat d_temp_fids_;
    cv::cuda::GpuMat d_output_;
    cv::cuda::GpuMat d_output_fids_;
    cv::cuda::GpuMat d_output_count_;

    // CUB temp（grow-only 缓存）
    void* d_radix_temp_ = nullptr;
    size_t radix_temp_size_ = 0;
    int radix_cap_ = 0;
    void* d_scan_temp_ = nullptr;
    size_t scan_temp_size_ = 0;
    int scan_cap_ = 0;
    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;
    int cub_cap_ = 0;

    int capacity_ = 0;        // 按 pointCount grow-only
    int lines_cap_ = 0;       // 按 total_lines grow-only

    int* h_meta_pinned_ = nullptr;   // 12B 中途回传 {num_lines, umin, umax}

    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EpipolarInterpParams& params);
    ~Impl();

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids,
                                 cv::cuda::Stream& stream);
    void Destroy();
    void Warmup(int pointCount);
    void SetParams(const EpipolarInterpParams& params);
    const EpipolarInterpParams& GetParams() const { return params_; }

    void ensureCapacity(int pointCount);
};

} // namespace calib
