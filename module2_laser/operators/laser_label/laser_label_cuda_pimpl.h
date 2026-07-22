/**
 * @file laser_label_cuda_pimpl.h
 * @brief 激光线编号算子 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 * .cpp 和 .cu 均 include 此文件。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/device_vector.h>
#include <memory>
#include <atomic>
#include <cassert>
#include <climits>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "laser_label_cuda.h"

namespace calib {

struct LaserLabelerCUDA::Impl {
    LaserLabelParams params_;

    cv::cuda::GpuMat d_labels_buf_;
    cv::cuda::GpuMat d_output_buf_;

    thrust::device_vector<int> d_min_y_coords_;
    thrust::device_vector<int> d_label_ids_;
    thrust::device_vector<int> d_map_table_;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;
    int old_device_id = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserLabelParams& params);
    ~Impl() = default;

    LaserLabelResult Execute(const cv::cuda::GpuMat& d_inputMask,
                           cv::cuda::Stream stream);
    void Warmup(int rows, int cols);
    void SetParams(const LaserLabelParams& params);
    const LaserLabelParams& GetParams() const { return params_; }
};

} // namespace calib
