/**
 * @file epipolar_interp_cuda.h
 * @brief 激光中心点极线插值CUDA算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（undistort_points_cuda 之后�? * 平台：GPU（CUDA�? *
 * 功能：对输入的激光中心亚像素点集 (CV_32FC2)，根据设定的极线行距�? *       在相邻点之间的最近极线上线性插值，输出插值点�? *
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
// EpipolarInterpParams
// ============================================================================

struct EpipolarInterpParams {
    float epipolar_row_step = 0.5f;
    float max_x_diff = 1.0f;
    float max_y_span = 2.0f;
    int deviceId = 0;
    bool lineIdCheck = true;  // true=calibration (same-line by line_id); false=scanning (geometry only)

    void validate() const {
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("EpipolarInterpParams::epipolar_row_step must be > 0");
        if (max_x_diff <= 0.0f)
            throw std::invalid_argument("EpipolarInterpParams::max_x_diff must be > 0");
        if (max_y_span <= 0.0f)
            throw std::invalid_argument("EpipolarInterpParams::max_y_span must be > 0");
        if (deviceId < 0)
            throw std::invalid_argument("EpipolarInterpParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"epipolar_row_step", epipolar_row_step},
            {"max_x_diff", max_x_diff},
            {"max_y_span", max_y_span},
            {"deviceId", deviceId},
            {"lineIdCheck", lineIdCheck}
        };
    }

    static EpipolarInterpParams fromJson(const nlohmann::json& j) {
        EpipolarInterpParams p;
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("max_x_diff"))
            p.max_x_diff = j.at("max_x_diff").get<float>();
        if (j.contains("max_y_span"))
            p.max_y_span = j.at("max_y_span").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("lineIdCheck"))
            p.lineIdCheck = j.at("lineIdCheck").get<bool>();
        p.validate();
        return p;
    }
};

// ============================================================================
// EpipolarInterpResult
// ============================================================================

struct EpipolarInterpResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_interpPoints;
    std::shared_ptr<cv::cuda::GpuMat> d_interp_line_ids;
    int interpCount = 0;

    EpipolarInterpResult() = default;
    ~EpipolarInterpResult() = default;

    EpipolarInterpResult(EpipolarInterpResult&&) = default;
    EpipolarInterpResult& operator=(EpipolarInterpResult&&) = default;

    EpipolarInterpResult(const EpipolarInterpResult&) = delete;
    EpipolarInterpResult& operator=(const EpipolarInterpResult&) = delete;
};

// ============================================================================
// EpipolarInterpCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存极线步距/阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API EpipolarInterpCuda {
public:
    static constexpr const char* kLogTag = "06-EpipolarInterpCuda";

    explicit EpipolarInterpCuda(const EpipolarInterpParams& params = {});
    ~EpipolarInterpCuda();

    EpipolarInterpCuda(const EpipolarInterpCuda&) = delete;
    EpipolarInterpCuda& operator=(const EpipolarInterpCuda&) = delete;

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

OperatorInfo getEpipolarInterpCudaInfo();

} // namespace calib