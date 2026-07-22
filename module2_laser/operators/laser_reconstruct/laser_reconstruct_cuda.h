/**
 * @file laser_reconstruct_cuda.h
 * @brief 激光线三维重建CUDA算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（laser_match_cuda 之后�? * 平台：GPU（CUDA�? *
 * 功能：对左右相机匹配的激光中心亚像素点对，使用立体矫�?Q 矩阵
 *       进行三维重建，输�?D点云及激光线编号
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
// LaserReconstructParams
// ============================================================================

struct LaserReconstructParams {
    float minDepth = 0.0f;
    float maxDepth = 10000.0f;
    int deviceId = 0;

    void validate() const {
        if (minDepth < 0.0f)
            throw std::invalid_argument("LaserReconstructParams::minDepth must be >= 0");
        if (maxDepth <= minDepth)
            throw std::invalid_argument("LaserReconstructParams::maxDepth must be > minDepth");
        if (deviceId < 0)
            throw std::invalid_argument("LaserReconstructParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"minDepth", minDepth},
            {"maxDepth", maxDepth},
            {"deviceId", deviceId}
        };
    }

    static LaserReconstructParams fromJson(const nlohmann::json& j) {
        LaserReconstructParams p;
        if (j.contains("minDepth"))
            p.minDepth = j.at("minDepth").get<float>();
        if (j.contains("maxDepth"))
            p.maxDepth = j.at("maxDepth").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserReconstructResult
// ============================================================================

struct LaserReconstructResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_points3d;
    std::shared_ptr<cv::cuda::GpuMat> d_valid_line_ids;
    int validCount = 0;
    int totalInput = 0;

    LaserReconstructResult() = default;
    ~LaserReconstructResult() = default;

    LaserReconstructResult(LaserReconstructResult&&) = default;
    LaserReconstructResult& operator=(LaserReconstructResult&&) = default;

    LaserReconstructResult(const LaserReconstructResult&) = delete;
    LaserReconstructResult& operator=(const LaserReconstructResult&) = delete;
};

// ============================================================================
// LaserReconstructCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；Q 矩阵按调用传入，SetParams 仅缓存深度阈值，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserReconstructCuda {
public:
    static constexpr const char* kLogTag = "08-LaserReconstructCuda";

    explicit LaserReconstructCuda(const LaserReconstructParams& params = {});
    ~LaserReconstructCuda();

    LaserReconstructCuda(const LaserReconstructCuda&) = delete;
    LaserReconstructCuda& operator=(const LaserReconstructCuda&) = delete;

    LaserReconstructResult Execute(const cv::cuda::GpuMat& d_matched_left,
                                   const cv::cuda::GpuMat& d_matched_right,
                                   const cv::cuda::GpuMat& d_matched_line_ids,
                                   const cv::Mat& Q,
                                   cv::cuda::Stream& stream);

    LaserReconstructResult Execute(const cv::cuda::GpuMat& d_matched_left,
                                   const cv::cuda::GpuMat& d_matched_right,
                                   const cv::cuda::GpuMat& d_matched_line_ids,
                                   const cv::Mat& Q);

    void Destroy();
    void Warmup(int pointCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserReconstructParams& params);
    const LaserReconstructParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserReconstructCudaInfo();

} // namespace calib