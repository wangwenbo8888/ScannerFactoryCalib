/**
 * @file laser_reconstruct_cuda_pimpl.h
 * @brief 激光线三维重建CUDA算子 - 内部桥接头文件（声明 struct Impl 完整结构）
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

#include "laser_reconstruct_cuda.h"

namespace calib {

struct LaserReconstructCuda::Impl {
    LaserReconstructParams params_;

    cv::cuda::GpuMat d_disparity_;
    cv::cuda::GpuMat d_points3d_raw_;
    cv::cuda::GpuMat d_valid_flags_;
    cv::cuda::GpuMat d_temp_fids_;

    cv::cuda::GpuMat d_out_points3d_;
    cv::cuda::GpuMat d_out_fids_;
    cv::cuda::GpuMat d_out_count_;

    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;
    int last_max_count_ = 0;

    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserReconstructParams& params);
    ~Impl();

    LaserReconstructResult Execute(const cv::cuda::GpuMat& d_matched_left,
                                   const cv::cuda::GpuMat& d_matched_right,
                                   const cv::cuda::GpuMat& d_matched_line_ids,
                                   const cv::Mat& Q,
                                   cv::cuda::Stream& stream);
    void Destroy();
    void Warmup(int pointCount);
    void SetParams(const LaserReconstructParams& params);
    const LaserReconstructParams& GetParams() const { return params_; }

    bool allocateBuffers(int pointCount);
};

} // namespace calib
