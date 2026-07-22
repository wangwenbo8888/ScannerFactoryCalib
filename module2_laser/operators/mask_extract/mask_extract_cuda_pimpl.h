/**
 * @file mask_extract_cuda_pimpl.h
 * @brief 激光掩膜提取算子 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <memory>
#include <atomic>
#include <cassert>
#include <string>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "mask_extract_cuda.h"

namespace calib {

/**
 * @brief MaskExtractCUDA 内部实现结构
 */
struct MaskExtractCUDA::Impl {
    // 参数
    MaskExtractParams params_;

    // GPU 缓冲区（d_ 前缀表示 device 端数据）
    cv::cuda::GpuMat d_inputBuffer;       ///< 上传目标（输入缓冲）
    cv::cuda::GpuMat d_thresholded;       ///< 二值化结果
    cv::cuda::GpuMat d_eroded;            ///< 腐蚀结果
    cv::cuda::GpuMat d_laserMask;         ///< 激光掩膜（膨胀后）
    cv::cuda::GpuMat d_cleanedMask;       ///< 面积过滤后的掩膜

    // 形态学核（CPU 端创建，GPU 端使用）
    cv::Mat kernel_erode_;
    cv::Mat kernel_dilate_;

    // CUDA Filter 对象缓存
    cv::Ptr<cv::cuda::Filter> filter_erode_;
    cv::Ptr<cv::cuda::Filter> filter_dilate_;

    // 预热状态
    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

#ifndef NDEBUG
    // 线程安全约束（仅 Debug 模式，§2.2）
    std::atomic<bool> inProcess_{false};  ///< Execute() 执行中标记
#endif

    explicit Impl(const MaskExtractParams& params);

    ~Impl() = default;

    void rebuildFilters();

    void executePipeline(cv::cuda::GpuMat& d_outputMask,
                        cv::cuda::Stream& stream);

    void filterByArea(cv::cuda::GpuMat& d_mask, int minArea, int maxArea);

    MaskExtractResult Execute(const cv::Mat& grayImage, cv::cuda::Stream& stream);

    void SetParams(const MaskExtractParams& params);

    void Warmup(int rows, int cols);

    const MaskExtractParams& GetParams() const { return params_; }

    void releaseBuffers();
};

} // namespace calib
