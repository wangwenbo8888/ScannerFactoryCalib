/**
 * @file endpoint_extract_cuda_pimpl.h
 * @brief 激光线3D端点提取CUDA算子 - 内部桥接头文件（声明 struct Impl 完整结构）
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

#include "endpoint_extract_cuda.h"

namespace calib {

struct EndpointExtractCuda::Impl {
    EndpointExtractParams params_;

    cv::cuda::GpuMat d_line_sums_;
    cv::cuda::GpuMat d_line_counts_;
    cv::cuda::GpuMat d_centroids_;
    cv::cuda::GpuMat d_max_dist_sq_;
    cv::cuda::GpuMat d_ep_a_idx_;
    cv::cuda::GpuMat d_endpoint_a_;
    cv::cuda::GpuMat d_ep_b_idx_;
    cv::cuda::GpuMat d_max_fid_;

    cv::cuda::GpuMat d_out_endpoints_;
    cv::cuda::GpuMat d_out_ep_ids_;
    cv::cuda::GpuMat d_out_line_ids_;
    cv::cuda::GpuMat d_out_num_lines_;

    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;
    int last_max_fid_ = 0;

    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int warmup_max_fid_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EndpointExtractParams& params);
    ~Impl();

    EndpointExtractResult Execute(const cv::cuda::GpuMat& d_points3d,
                                  const cv::cuda::GpuMat& d_line_ids,
                                  cv::cuda::Stream stream);
    void Warmup(int pointCount, int maxFrameId);
    void SetParams(const EndpointExtractParams& params);
    const EndpointExtractParams& GetParams() const { return params_; }

    bool allocateBuffers(int maxFid, int pointCount);
};

} // namespace calib
