/**
 * @file steger_fast_pimpl.h
 * @brief Steger激光中心亚像素提取算子（优化版）- 内部桥接头文件
 *
 * 声明 struct Impl 完整结构；仅供 steger_fast.cpp / steger_fast_impl.cu 使用。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <atomic>

#include "steger_fast.h"

namespace calib {

struct FastPoint {
    int label;
    float px;
    float py;
};

struct StegerExtractorFast::Impl {
    static constexpr int kLabelBins = 4096;
    static constexpr int kMaxKernelSize = 31;

    StegerParams params_;
    int actualKernelSize_ = 0;

    cv::cuda::GpuMat d_tg_, d_tgx_, d_tgxx_;
    cv::cuda::GpuMat d_ix_, d_iy_, d_ixx_, d_iyy_, d_ixy_;

    float* d_kg_ = nullptr;
    float* d_kgx_ = nullptr;
    float* d_kgxx_ = nullptr;

    FastPoint* d_points_ = nullptr;
    FastPoint* d_sorted_ = nullptr;
    float2* d_coords_ = nullptr;
    int* d_ids_ = nullptr;
    int capacity_ = 0;

    int* d_count_ = nullptr;
    int* d_hist_ = nullptr;
    int* d_base_ = nullptr;
    int* d_cursor_ = nullptr;

    int* h_count_ = nullptr;
    int* h_hist_ = nullptr;
    int* h_base_ = nullptr;
    FastPoint* h_pts_ = nullptr;

    cudaEvent_t ev_start_ = nullptr, ev_row_ = nullptr, ev_col_ = nullptr,
                ev_hess_ = nullptr, ev_grp_ = nullptr, ev_out_ = nullptr, ev_end_ = nullptr;
    bool events_ready_ = false;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const StegerParams& params);
    ~Impl();

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         bool maskIs8U,
                         cv::cuda::Stream& stream);
    StegerResult extractFlat(const cv::cuda::GpuMat& d_grayImage,
                             const cv::cuda::GpuMat& d_binaryMask,
                             cv::cuda::Stream& stream);
    void Destroy();
    void Warmup(int rows, int cols);
    void SetParams(const StegerParams& params);
    const StegerParams& GetParams() const { return params_; }

    void buildGaussianKernels();
    int computeKernelSize() const;
    void ensureBuffers(int rows, int cols);
    void createEvents();
};

} // namespace calib
