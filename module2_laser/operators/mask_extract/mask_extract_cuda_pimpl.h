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
    cv::cuda::GpuMat d_postMask;          ///< 膨胀后二次腐蚀结果（postErodeSize>0 时）
    cv::cuda::GpuMat d_cleanedMask;       ///< 面积过滤后的掩膜

    // 形态学核（CPU 端创建，GPU 端使用）
    cv::Mat kernel_erode_;
    cv::Mat kernel_dilate_;
    cv::Mat kernel_post_erode_;

    // CUDA Filter 对象缓存
    cv::Ptr<cv::cuda::Filter> filter_erode_;
    cv::Ptr<cv::cuda::Filter> filter_dilate_;
    cv::Ptr<cv::cuda::Filter> filter_post_erode_;
    // morphApprox=1 线核滤波器（横/竖两趟近似, 见 Params 注释）
    cv::Ptr<cv::cuda::Filter> filter_dilate_h_;
    cv::Ptr<cv::cuda::Filter> filter_dilate_v_;
    cv::Ptr<cv::cuda::Filter> filter_post_h_;
    cv::Ptr<cv::cuda::Filter> filter_post_v_;
    cv::cuda::GpuMat d_morphTmp;          ///< 线核两趟的中间缓冲

    // 结果输出池（ping-pong ×2, [gray, laserMask, cleanedMask]）:
    // 预分配+流式 copyTo, 替代每次 clone 的临时 cudaMalloc+默认流屏障。
    // 有效性语义: 返回结果至少到「再执行两次 Execute」前有效
    cv::cuda::GpuMat d_resPool_[2][3];
    int resIdx_ = 0;

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

    void allocateBuffers(int rows, int cols);

    void executePipeline(cv::cuda::GpuMat& d_outputMask,
                        cv::cuda::Stream& stream);

    MaskExtractResult Execute(const cv::Mat& grayImage, cv::cuda::Stream& stream);

    void SetParams(const MaskExtractParams& params);

    void Warmup(int rows, int cols);

    const MaskExtractParams& GetParams() const { return params_; }

    void releaseBuffers();
};

} // namespace calib
