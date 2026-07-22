/**
 * @file undistort_points_cuda_pimpl.h
 * @brief 激光中心亚像素点集去畸变+立体矫正算子 - 内部桥接头文件（声明 struct Impl 完整结构）
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

#include "undistort_points_cuda.h"

namespace calib {

struct UndistortPointsCuda::Impl {
    UndistortPointsParams params_;

    float fx_, fy_, cx_, cy_;
    float k1_, k2_, p1_, p2_, k3_, k4_, k5_, k6_;
    float R_[9];
    float fx_p_, fy_p_, cx_p_, cy_p_, tx_p_;

    cv::cuda::GpuMat d_output_;
    cv::cuda::GpuMat d_R_;

    bool warmed_up_ = false;
    int warmup_pointCount_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const UndistortPointsParams& params);
    ~Impl() = default;

    UndistortPointsResult Execute(const cv::cuda::GpuMat& d_points,
                                   const cv::cuda::GpuMat& d_line_ids,
                                   cv::cuda::Stream& stream);
    void Destroy();
    void Warmup(int maxPointCount);
    void SetParams(const UndistortPointsParams& params);
    const UndistortPointsParams& GetParams() const { return params_; }

    void extractCalibParams();
};

} // namespace calib
