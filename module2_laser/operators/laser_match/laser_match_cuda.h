/**
 * @file laser_match_cuda.h
 * @brief 激光线匹配CUDA算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（epipolar_interp_cuda 之后�? * 平台：GPU（CUDA�? *
 * 功能：对左右相机激光中心亚像素点集进行立体匹配，使�?GPU Hash Join
 *       实现基于极线约束的快速匹配，输出匹配点对及帧 ID
 *
 * 精度容差档次：档次③（亚像素/浮点类）
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================================
// LaserMatchParams
// ============================================================================

struct LaserMatchParams {
    float epipolar_row_step = 0.5f;
    float min_disparity = 0.0f;
    float max_disparity = 500.0f;
    int deviceId = 0;

    void validate() const {
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("LaserMatchParams::epipolar_row_step must be > 0");
        if (min_disparity < 0.0f)
            throw std::invalid_argument("LaserMatchParams::min_disparity must be >= 0");
        if (max_disparity <= min_disparity)
            throw std::invalid_argument("LaserMatchParams::max_disparity must be > min_disparity");
        if (deviceId < 0)
            throw std::invalid_argument("LaserMatchParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"epipolar_row_step", epipolar_row_step},
            {"min_disparity", min_disparity},
            {"max_disparity", max_disparity},
            {"deviceId", deviceId}
        };
    }

    static LaserMatchParams fromJson(const nlohmann::json& j) {
        LaserMatchParams p;
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("min_disparity"))
            p.min_disparity = j.at("min_disparity").get<float>();
        if (j.contains("max_disparity"))
            p.max_disparity = j.at("max_disparity").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserMatchResult
// ============================================================================

struct LaserMatchResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_left;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_right;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_line_ids;
    int matchCount = 0;

    LaserMatchResult() = default;
    ~LaserMatchResult() = default;

    LaserMatchResult(LaserMatchResult&&) = default;
    LaserMatchResult& operator=(LaserMatchResult&&) = default;

    LaserMatchResult(const LaserMatchResult&) = delete;
    LaserMatchResult& operator=(const LaserMatchResult&) = delete;
};

// ============================================================================
// LaserMatchCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存极线步距/视差范围等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserMatchCuda {
public:
    static constexpr const char* kLogTag = "07-LaserMatchCuda";

    explicit LaserMatchCuda(const LaserMatchParams& params = {});
    ~LaserMatchCuda();

    LaserMatchCuda(const LaserMatchCuda&) = delete;
    LaserMatchCuda& operator=(const LaserMatchCuda&) = delete;

    LaserMatchResult Execute(const cv::cuda::GpuMat& d_left_points,
                             const cv::cuda::GpuMat& d_left_line_ids,
                             const cv::cuda::GpuMat& d_right_points,
                             const cv::cuda::GpuMat& d_right_line_ids,
                             cv::cuda::Stream& stream);

    LaserMatchResult Execute(const cv::cuda::GpuMat& d_left_points,
                             const cv::cuda::GpuMat& d_left_line_ids,
                             const cv::cuda::GpuMat& d_right_points,
                             const cv::cuda::GpuMat& d_right_line_ids);

    void Warmup(int leftCount, int rightCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserMatchParams& params);
    const LaserMatchParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserMatchCudaInfo();

} // namespace calib