#pragma once

#include <opencv2/core/cuda.hpp>
#include <memory>
#include <atomic>
#include <vector>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "plane_map_cuda.h"

namespace calib {

struct PlaneMapCuda::Impl {
    PlaneMapParams params_;

    cv::cuda::GpuMat d_pixels_raw_buf_;
    cv::cuda::GpuMat d_cand_buf_;
    cv::cuda::GpuMat d_compact_buf_;
    cv::cuda::GpuMat d_compact_count_buf_;
    cv::cuda::GpuMat d_Kv_inv_buf_;
    cv::cuda::GpuMat d_Rvt_buf_;
    cv::cuda::GpuMat d_Tv_buf_;
    cv::cuda::GpuMat d_R1_buf_;
    cv::cuda::GpuMat d_R2_buf_;
    cv::cuda::GpuMat d_P1_buf_;
    cv::cuda::GpuMat d_P2_buf_;
    cv::cuda::GpuMat d_FVL_buf_;
    cv::cuda::GpuMat d_FVR_buf_;
    cv::cuda::GpuMat d_valid_counts_buf_;

    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int warmup_max_lid_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const PlaneMapParams& params);
    ~Impl();

    PlaneMapResult Execute(
        const cv::cuda::GpuMat& d_virtual_pixels,
        const cv::Matx33d& virtualK,
        const cv::Matx33d& virtualR,
        const cv::Vec3d& virtualT,
        const StereoCalibration& calib,
        cv::cuda::Stream stream);
    void Warmup(int numVirtualPixels, int maxLineId);
    void SetParams(const PlaneMapParams& params);
    const PlaneMapParams& GetParams() const { return params_; }
};

} // namespace calib
