#pragma once


#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class Stream; } }

namespace calib {


struct WarmupConfig;

enum class PlaneMapMethod {
    Projective = 0,
    FundamentalMatrix = 1
};

struct PlaneMapParams {
    int deviceId = 0;
    PlaneMapMethod method = PlaneMapMethod::Projective;
    float gridStep = 0.5f;
    float depthMin = 100.0f;
    float depthMax = 5000.0f;
    int depthSamples = 200;
    float epipolarStep = 0.5f;
    bool enableTiming = false;

    void validate() const {
        if (deviceId < 0)
            throw std::invalid_argument("PlaneMapParams::deviceId must be >= 0");
        if (gridStep <= 0.0f)
            throw std::invalid_argument("PlaneMapParams::gridStep must be > 0");
        if (depthMin <= 0.0f)
            throw std::invalid_argument("PlaneMapParams::depthMin must be > 0");
        if (depthMax <= depthMin)
            throw std::invalid_argument("PlaneMapParams::depthMax must be > depthMin");
        if (depthSamples <= 0)
            throw std::invalid_argument("PlaneMapParams::depthSamples must be > 0");
        if (epipolarStep <= 0.0f)
            throw std::invalid_argument("PlaneMapParams::epipolarStep must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"deviceId", deviceId},
            {"method", static_cast<int>(method)},
            {"gridStep", gridStep},
            {"depthMin", depthMin},
            {"depthMax", depthMax},
            {"depthSamples", depthSamples},
            {"epipolarStep", epipolarStep},
            {"enableTiming", enableTiming}
        };
    }

    static PlaneMapParams fromJson(const nlohmann::json& j) {
        PlaneMapParams p;
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("method"))
            p.method = static_cast<PlaneMapMethod>(j.at("method").get<int>());
        if (j.contains("gridStep"))
            p.gridStep = j.at("gridStep").get<float>();
        if (j.contains("depthMin"))
            p.depthMin = j.at("depthMin").get<float>();
        if (j.contains("depthMax"))
            p.depthMax = j.at("depthMax").get<float>();
        if (j.contains("depthSamples"))
            p.depthSamples = j.at("depthSamples").get<int>();
        if (j.contains("epipolarStep"))
            p.epipolarStep = j.at("epipolarStep").get<float>();
        if (j.contains("enableTiming"))
            p.enableTiming = j.at("enableTiming").get<bool>();
        p.validate();
        return p;
    }
};

struct LineMapStats {
    int lineId = -1;
    int numPairs = 0;
    float uMin = 0, uMax = 0, vMin = 0, vMax = 0;
};

struct PlaneMapResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::shared_ptr<cv::cuda::GpuMat> d_left_to_right;
    std::shared_ptr<cv::cuda::GpuMat> d_right_u;

    std::vector<LineMapStats> lineStats;
    int totalPairs = 0;

    PlaneMapResult() = default;
    ~PlaneMapResult() = default;

    PlaneMapResult(PlaneMapResult&&) = default;
    PlaneMapResult& operator=(PlaneMapResult&&) = default;

    PlaneMapResult(const PlaneMapResult&) = delete;
    PlaneMapResult& operator=(const PlaneMapResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 虚拟像素与标定参数按调用传入；Impl 仅持有每调用重置的 GPU 暂存缓冲，SetParams 缓存网格/深度等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API PlaneMapCuda {
public:
    static constexpr const char* kLogTag = "12-PlaneMapCuda";

    explicit PlaneMapCuda(const PlaneMapParams& params = {});
    ~PlaneMapCuda();

    PlaneMapCuda(const PlaneMapCuda&) = delete;
    PlaneMapCuda& operator=(const PlaneMapCuda&) = delete;

    PlaneMapResult Execute(
        const cv::cuda::GpuMat& d_virtual_pixels,
        const cv::Matx33d& virtualK,
        const cv::Matx33d& virtualR,
        const cv::Vec3d& virtualT,
        const StereoCalibration& calib,
        cv::cuda::Stream& stream);

    PlaneMapResult Execute(
        const cv::cuda::GpuMat& d_virtual_pixels,
        const cv::Matx33d& virtualK,
        const cv::Matx33d& virtualR,
        const cv::Vec3d& virtualT,
        const StereoCalibration& calib);

    void Warmup(int numVirtualPixels, int maxLineId);
    void Warmup(const WarmupConfig& config);
    void SetParams(const PlaneMapParams& params);
    const PlaneMapParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getPlaneMapCudaInfo();

} // namespace calib