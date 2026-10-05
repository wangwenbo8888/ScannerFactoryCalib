/**
 * @file steger_extract_cuda.h
 * @brief Steger激光中心亚像素提取算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（laser_label_cuda 之后�? * 平台：GPU（CUDA + Thrust�? *
 * 功能：基�?Hessian 矩阵 + 泰勒展开，从灰度图像中提取激光线亚像素中心点
 *       输入灰度图像（CV_8UC1�? 编号标签图（CV_32SC1），输出每条线的中心点集
 *
 * 精度容差档次：档次③（亚像素/浮点类）
 * - 亚像素精度：理论精度 0.01~0.1 像素级别
 * - CPU vs CUDA 互比差异 < 0.05px
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"
// GroupMode / StegerParams / StegerResult 定义已抽至 common/steger_types.h
// （2026-09-04；定义逐字未动——steger_fast 不再传递依赖本头）
#include "common/steger_types.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================================
// StegerExtractorCUDA
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存 sigma/阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API StegerExtractorCUDA {
public:
    static constexpr const char* kLogTag = "10-StegerExtractorCUDA";

    explicit StegerExtractorCUDA(const StegerParams& params = {});
    ~StegerExtractorCUDA();

    StegerExtractorCUDA(const StegerExtractorCUDA&) = delete;
    StegerExtractorCUDA& operator=(const StegerExtractorCUDA&) = delete;

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         cv::cuda::Stream& stream);

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask);

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_mask,
                         cv::cuda::Stream& stream,
                         GroupMode groupMode);

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_mask,
                         GroupMode groupMode);

    void Destroy();
    void Warmup(int rows, int cols);
    void Warmup(const WarmupConfig& config);
    void SetParams(const StegerParams& params);
    const StegerParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getStegerExtractInfo();

} // namespace calib