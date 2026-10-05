#pragma once

#include <opencv2/core/cuda.hpp>
#include <memory>
#include <atomic>
#include <vector>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "pose_optimize_cuda.h"

namespace calib {

struct PoseOptimizeCuda::Impl {
    PoseOptimizeParams params_;

    cv::cuda::GpuMat d_points_buf_;
    cv::cuda::GpuMat d_line_ids_buf_;

    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int warmup_max_lid_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const PoseOptimizeParams& params);
    ~Impl();

    PoseOptimizeResult Execute(
        const cv::cuda::GpuMat& d_points,
        const cv::cuda::GpuMat& d_line_ids,
        const cv::Matx33d& virtualK,
        const cv::Matx33d& virtualR,
        const cv::Vec3d& initialT,
        cv::cuda::Stream stream);
    void Warmup(int numPoints, int maxLineId);
    void SetParams(const PoseOptimizeParams& params);
    const PoseOptimizeParams& GetParams() const { return params_; }
};

} // namespace calib
