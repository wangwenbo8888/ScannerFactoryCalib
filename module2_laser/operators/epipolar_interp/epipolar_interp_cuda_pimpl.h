/**
 * @file epipolar_interp_cuda_pimpl.h
 * @brief 激光中心点极线插值CUDA算子 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 * .cpp 和 .cu 均 include 此文件。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <memory>
#include <atomic>
#include <vector>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "epipolar_interp_cuda.h"

namespace calib {

struct EpipolarInterpCuda::Impl {
    EpipolarInterpParams params_;

    cv::cuda::GpuMat d_flags_;
    cv::cuda::GpuMat d_temp_interp_;
    cv::cuda::GpuMat d_temp_fids_;
    cv::cuda::GpuMat d_output_;
    cv::cuda::GpuMat d_output_fids_;
    cv::cuda::GpuMat d_output_count_;
    void* d_cub_temp_storage_ = nullptr;
    size_t cub_temp_size_ = 0;
    int last_max_pairs_count_ = 0;

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

    bool allocateBuffers(int pointCount);
};

} // namespace calib
