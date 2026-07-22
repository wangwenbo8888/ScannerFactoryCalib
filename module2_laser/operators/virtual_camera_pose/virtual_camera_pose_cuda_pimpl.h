/**
 * @file virtual_camera_pose_cuda_pimpl.h
 * @brief 激光器虚拟相机光心和初步外参CUDA算子 - 内部桥接头文件（声明 struct Impl 完整结构）
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

#include "virtual_camera_pose_cuda.h"

namespace calib {

struct VirtualCameraPoseCuda::Impl {
    VirtualCameraPoseParams params_;

    cv::cuda::GpuMat d_endpoints_buf_;
    cv::cuda::GpuMat d_line_ids_buf_;

    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int warmup_max_lid_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const VirtualCameraPoseParams& params);
    ~Impl();

    VirtualCameraPoseResult Execute(
        const cv::cuda::GpuMat& d_endpoints,
        const cv::cuda::GpuMat& d_line_ids,
        const cv::Matx33d& stereoK,
        const cv::Matx33d& stereoR,
        cv::cuda::Stream stream);
    void Warmup(int numEndpoints, int maxLineId);
    void SetParams(const VirtualCameraPoseParams& params);
    const VirtualCameraPoseParams& GetParams() const { return params_; }
};

} // namespace calib
