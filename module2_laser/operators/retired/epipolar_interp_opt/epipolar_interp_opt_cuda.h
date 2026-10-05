/**
 * @file epipolar_interp_opt_cuda.h
 * @brief 激光中心点极线插值CUDA算子·优化版 - 公开头文件（纯 C++，不含 CUDA 类型）
 *
 * 由 epipolar_interp（v2 极线驱动重采样版）复制优化而来，算法语义/输出契约不变：
 *   - Step1 排序: thrust::sort_by_key → CUB DeviceRadixSort（基数排序，O(n)）
 *   - Step2 线号统计: 全量点下载+host 扫描 → GPU 端 flags+前缀和+scatter 建表，
 *                    仅 1 次 12B pinned 回传（num_lines/y_min/y_max），线表不再 H2D 上传
 *   - Step4 CUB temp: 每次 size 查询 → 容量增长时才查询
 *   - 缓冲策略: 排序/插值/CUB 缓冲 grow-only，消除每帧 realloc 抖动
 * 未变更: 插值 kernel 逐字一致；结果 clone/interpCount 契约一致（二期优化项另立项）
 */

#pragma once

#include "epipolar_interp_cuda.h"

#include <memory>
#include <stdexcept>

namespace calib {

class SCANNER_API EpipolarInterpOptCuda {
public:
    static constexpr const char* kLogTag = "06-EpipolarInterpOptCuda";

    explicit EpipolarInterpOptCuda(const EpipolarInterpParams& params = {});
    ~EpipolarInterpOptCuda();

    EpipolarInterpOptCuda(const EpipolarInterpOptCuda&) = delete;
    EpipolarInterpOptCuda& operator=(const EpipolarInterpOptCuda&) = delete;

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids,
                                 cv::cuda::Stream& stream);

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids);

    void Destroy();
    void Warmup(int pointCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const EpipolarInterpParams& params);
    const EpipolarInterpParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getEpipolarInterpOptInfo();

} // namespace calib
