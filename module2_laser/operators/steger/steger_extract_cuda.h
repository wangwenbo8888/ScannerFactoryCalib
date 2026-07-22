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

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================================
// GroupMode
// ============================================================================

enum class GroupMode {
    ByLabel,
    Flat
};

// ============================================================================
// StegerParams
// ============================================================================

struct StegerParams {
    float sigma = 1.5f;
    int kernelSize = 0;
    float lowThreshold = 2.0f;
    float highThreshold = 0.0f;
    int maxLabels = 256;
    int deviceId = 0;

    void validate() const {
        if (sigma < 0.5f || sigma > 10.0f)
            throw std::invalid_argument("StegerParams::sigma must be [0.5, 10.0]");
        if (kernelSize != 0 && kernelSize != 3 && kernelSize != 5
            && kernelSize != 7 && kernelSize != 9)
            throw std::invalid_argument("StegerParams::kernelSize must be 0 (auto), 3, 5, 7, or 9");
        if (lowThreshold < 0.0f)
            throw std::invalid_argument("StegerParams::lowThreshold must be >= 0");
        if (highThreshold < 0.0f)
            throw std::invalid_argument("StegerParams::highThreshold must be >= 0");
        if (maxLabels < 1 || maxLabels > 4096)
            throw std::invalid_argument("StegerParams::maxLabels must be [1, 4096]");
        if (deviceId < 0)
            throw std::invalid_argument("StegerParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"sigma", sigma},
            {"kernelSize", kernelSize},
            {"lowThreshold", lowThreshold},
            {"highThreshold", highThreshold},
            {"maxLabels", maxLabels},
            {"deviceId", deviceId}
        };
    }

    static StegerParams fromJson(const nlohmann::json& j) {
        StegerParams p;
        if (j.contains("sigma")) p.sigma = j.at("sigma").get<float>();
        if (j.contains("kernelSize")) p.kernelSize = j.at("kernelSize").get<int>();
        if (j.contains("lowThreshold")) p.lowThreshold = j.at("lowThreshold").get<float>();
        if (j.contains("highThreshold")) p.highThreshold = j.at("highThreshold").get<float>();
        if (j.contains("maxLabels")) p.maxLabels = j.at("maxLabels").get<int>();
        if (j.contains("deviceId")) p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// StegerResult
// ============================================================================

struct StegerResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::map<int, std::vector<cv::Point2f>> centerPoints;
    int totalPointCount = 0;
    int lineCount = 0;

    std::shared_ptr<cv::cuda::GpuMat> d_centerPoints;
    std::shared_ptr<cv::cuda::GpuMat> d_line_ids;

    StegerResult() = default;
    ~StegerResult() = default;

    StegerResult(StegerResult&&) = default;
    StegerResult& operator=(StegerResult&&) = default;

    StegerResult(const StegerResult&) = delete;
    StegerResult& operator=(const StegerResult&) = delete;
};

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