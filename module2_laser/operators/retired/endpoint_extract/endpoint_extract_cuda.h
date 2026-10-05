/**
 * @file endpoint_extract_cuda.h
 * @brief 激光线3D端点提取CUDA算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（laser_reconstruct_cuda 之后�? * 平台：GPU（CUDA�? *
 * 功能：对激光三维重建输出的3D点云及激光线编号，提取每条激光线�? *       两个端点（距离最远的两个点），并编号输出
 *
 * 算法：对每条激光线，先求质心，取距质心最远点为端点A�? *       再取距端点A最远点为端点B（两�?O(n) 近似直径算法�? *
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
// EndpointExtractParams
// ============================================================================

struct EndpointExtractParams {
    int deviceId = 0;
    int maxExpectedLines = 4096;
    bool enableTiming = false;

    void validate() const {
        if (deviceId < 0)
            throw std::invalid_argument("EndpointExtractParams::deviceId must be >= 0");
        if (maxExpectedLines <= 0)
            throw std::invalid_argument("EndpointExtractParams::maxExpectedLines must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"deviceId", deviceId},
            {"maxExpectedLines", maxExpectedLines},
            {"enableTiming", enableTiming}
        };
    }

    static EndpointExtractParams fromJson(const nlohmann::json& j) {
        EndpointExtractParams p;
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("maxExpectedLines"))
            p.maxExpectedLines = j.at("maxExpectedLines").get<int>();
        if (j.contains("enableTiming"))
            p.enableTiming = j.at("enableTiming").get<bool>();
        p.validate();
        return p;
    }
};

// ============================================================================
// EndpointExtractResult
// ============================================================================

struct EndpointExtractTimings {
    float us_findMaxFid        = 0.0f;
    float us_memsetBuffers     = 0.0f;
    float us_computeLineSums   = 0.0f;
    float us_normalizeCentroids= 0.0f;
    float us_findMaxDistA      = 0.0f;
    float us_findMaxDistIdxA   = 0.0f;
    float us_gatherRefA        = 0.0f;
    float us_findMaxDistB      = 0.0f;
    float us_findMaxDistIdxB   = 0.0f;
    float us_collectEndpoints  = 0.0f;
    float us_d2hCopy           = 0.0f;
    float us_total             = 0.0f;

    nlohmann::json toJson() const {
        return {
            {"us_findMaxFid",        us_findMaxFid},
            {"us_memsetBuffers",     us_memsetBuffers},
            {"us_computeLineSums",   us_computeLineSums},
            {"us_normalizeCentroids",us_normalizeCentroids},
            {"us_findMaxDistA",      us_findMaxDistA},
            {"us_findMaxDistIdxA",   us_findMaxDistIdxA},
            {"us_gatherRefA",        us_gatherRefA},
            {"us_findMaxDistB",      us_findMaxDistB},
            {"us_findMaxDistIdxB",   us_findMaxDistIdxB},
            {"us_collectEndpoints",  us_collectEndpoints},
            {"us_d2hCopy",           us_d2hCopy},
            {"us_total",             us_total}
        };
    }
};

struct EndpointExtractResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_endpoints;
    std::shared_ptr<cv::cuda::GpuMat> d_endpoint_ids;
    std::shared_ptr<cv::cuda::GpuMat> d_line_ids;
    int numEndpoints = 0;
    int numLines = 0;
    int totalInput = 0;
    EndpointExtractTimings timing;

    EndpointExtractResult() = default;
    ~EndpointExtractResult() = default;

    EndpointExtractResult(EndpointExtractResult&&) = default;
    EndpointExtractResult& operator=(EndpointExtractResult&&) = default;

    EndpointExtractResult(const EndpointExtractResult&) = delete;
    EndpointExtractResult& operator=(const EndpointExtractResult&) = delete;
};

// ============================================================================
// EndpointExtractCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存 deviceId/线数上限等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API EndpointExtractCuda {
public:
    static constexpr const char* kLogTag = "09-EndpointExtractCuda";

    explicit EndpointExtractCuda(const EndpointExtractParams& params = {});
    ~EndpointExtractCuda();

    EndpointExtractCuda(const EndpointExtractCuda&) = delete;
    EndpointExtractCuda& operator=(const EndpointExtractCuda&) = delete;

    EndpointExtractResult Execute(const cv::cuda::GpuMat& d_points3d,
                                   const cv::cuda::GpuMat& d_line_ids,
                                   cv::cuda::Stream& stream);

    EndpointExtractResult Execute(const cv::cuda::GpuMat& d_points3d,
                                   const cv::cuda::GpuMat& d_line_ids);

    void Warmup(int pointCount, int maxFrameId);
    void Warmup(const WarmupConfig& config);
    void SetParams(const EndpointExtractParams& params);
    const EndpointExtractParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getEndpointExtractCudaInfo();

} // namespace calib