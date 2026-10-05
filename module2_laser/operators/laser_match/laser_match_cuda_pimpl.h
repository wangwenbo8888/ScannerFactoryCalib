/**
 * @file laser_match_cuda_pimpl.h
 * @brief 激光线匹配CUDA算子 - 内部桥接头文件（声明 struct Impl 完整结构）
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

#include "laser_match_cuda.h"

namespace calib {

struct LaserMatchCuda::Impl {
    LaserMatchParams params_;

    // 确定性匹配结构（2026-09-04 重写, 见 impl 文件头）: 排序键 (复合键<<32)|左点下标
    // （CV_32SC2 复用为 8B/元素）; d_sort_alt_ 为 CUB 排序输出缓冲
    cv::cuda::GpuMat d_sort_keys_;
    cv::cuda::GpuMat d_sort_alt_;

    cv::cuda::GpuMat d_left_rowidx_;
    cv::cuda::GpuMat d_right_rowidx_;

    cv::cuda::GpuMat d_flags_;
    cv::cuda::GpuMat d_temp_left_;
    cv::cuda::GpuMat d_temp_right_;
    cv::cuda::GpuMat d_temp_fids_;

    cv::cuda::GpuMat d_out_left_;
    cv::cuda::GpuMat d_out_right_;
    cv::cuda::GpuMat d_out_fids_;
    cv::cuda::GpuMat d_out_count_;

    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;
    int last_max_count_ = 0;

    bool warmed_up_ = false;
    int warmup_left_ = 0;
    int warmup_right_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserMatchParams& params);
    ~Impl();

    LaserMatchResult Execute(const cv::cuda::GpuMat& d_left_points,
                             const cv::cuda::GpuMat& d_left_line_ids,
                             const cv::cuda::GpuMat& d_right_points,
                             const cv::cuda::GpuMat& d_right_line_ids,
                             cv::cuda::Stream stream);
    void Warmup(int leftCount, int rightCount);
    void SetParams(const LaserMatchParams& params);
    const LaserMatchParams& GetParams() const { return params_; }

    bool allocateBuffers(int leftCount, int rightCount);
};

} // namespace calib
