/**
 * @file undistort_points_cuda.h
 * @brief 激光中心亚像素点集去畸�?立体矫正算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：激光器虚拟相机标定 �?�?步（steger_extract_cuda 之后�? * 平台：GPU（CUDA�? *
 * 功能：对输入的激光中心亚像素点集 (CV_32FC2)，执行去畸变 + 立体矫正
 *       输出矫正后的像素坐标 (CV_32FC2)
 *
 * 精度容差档次：档次③（亚像素/浮点类）
 * - �?cv::undistortPoints 结果差异 < 0.01 像素
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
// UndistortPointsParams
// ============================================================================

struct UndistortPointsParams {
    cv::Mat cameraMatrix;  // 来自 stereo_rectify_temp_table 当前温度补偿内参
    cv::Mat distCoeffs;    // 同上
    cv::Mat R;             // 矫正旋转矩阵 R1(�?/R2(�?，来�?stereo_rectify_temp_table
    cv::Mat P;             // 投影矩阵 P1(�?/P2(�?，来�?stereo_rectify_temp_table
    int deviceId = 0;

    void validate() const {
        if (cameraMatrix.empty())
            throw std::invalid_argument("UndistortPointsParams::cameraMatrix must not be empty");
        if (cameraMatrix.size() != cv::Size(3, 3))
            throw std::invalid_argument("UndistortPointsParams::cameraMatrix must be 3x3");
        if (cameraMatrix.type() != CV_64F && cameraMatrix.type() != CV_32F)
            throw std::invalid_argument("UndistortPointsParams::cameraMatrix must be CV_32F or CV_64F");

        if (distCoeffs.empty())
            throw std::invalid_argument("UndistortPointsParams::distCoeffs must not be empty");
        if (distCoeffs.total() < 4 || distCoeffs.total() > 8)
            throw std::invalid_argument("UndistortPointsParams::distCoeffs must have 4~8 elements");
        if (distCoeffs.type() != CV_64F && distCoeffs.type() != CV_32F)
            throw std::invalid_argument("UndistortPointsParams::distCoeffs must be CV_32F or CV_64F");

        if (!R.empty()) {
            if (R.size() != cv::Size(3, 3))
                throw std::invalid_argument("UndistortPointsParams::R must be 3x3 if provided");
            if (R.type() != CV_64F && R.type() != CV_32F)
                throw std::invalid_argument("UndistortPointsParams::R must be CV_32F or CV_64F");
        }

        if (!P.empty()) {
            if (P.rows != 3 || P.cols != 4)
                throw std::invalid_argument("UndistortPointsParams::P must be 3x4 if provided");
            if (P.type() != CV_64F && P.type() != CV_32F)
                throw std::invalid_argument("UndistortPointsParams::P must be CV_32F or CV_64F");
        }

        if (deviceId < 0)
            throw std::invalid_argument("UndistortPointsParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        auto matToJson = [](const cv::Mat& m) -> nlohmann::json {
            if (m.empty()) return nullptr;
            cv::Mat tmp;
            m.convertTo(tmp, CV_64F);
            tmp = tmp.reshape(1, 1);
            std::vector<double> v(tmp.begin<double>(), tmp.end<double>());
            return v;
        };

        return {
            {"cameraMatrix", matToJson(cameraMatrix)},
            {"distCoeffs", matToJson(distCoeffs)},
            {"R", matToJson(R)},
            {"P", matToJson(P)},
            {"deviceId", deviceId}
        };
    }

    static UndistortPointsParams fromJson(const nlohmann::json& j) {
        UndistortPointsParams p;

        auto jsonToMat = [](const nlohmann::json& jv, int rows, int cols) -> cv::Mat {
            if (jv.is_null()) return cv::Mat();
            std::vector<double> v = jv.get<std::vector<double>>();
            cv::Mat m(v, true);
            return m.reshape(1, rows);
        };

        if (j.contains("cameraMatrix"))
            p.cameraMatrix = jsonToMat(j.at("cameraMatrix"), 3, 3);
        if (j.contains("distCoeffs")) {
            auto dv = j.at("distCoeffs").get<std::vector<double>>();
            p.distCoeffs = cv::Mat(dv, true).reshape(1, 1);
        }
        if (j.contains("R"))
            p.R = jsonToMat(j.at("R"), 3, 3);
        if (j.contains("P"))
            p.P = jsonToMat(j.at("P"), 3, 4);
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();

        p.validate();
        return p;
    }
};

// ============================================================================
// UndistortPointsResult
// ============================================================================

struct UndistortPointsResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_rectifiedPoints;
    std::shared_ptr<cv::cuda::GpuMat> d_line_ids;

    UndistortPointsResult() = default;
    ~UndistortPointsResult() = default;

    UndistortPointsResult(UndistortPointsResult&&) = default;
    UndistortPointsResult& operator=(UndistortPointsResult&&) = default;

    UndistortPointsResult(const UndistortPointsResult&) = delete;
    UndistortPointsResult& operator=(const UndistortPointsResult&) = delete;
};

// ============================================================================
// UndistortPointsCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存相机内参/畸变/矫正矩阵等只读配置（可改为按调用传入），无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API UndistortPointsCuda {
public:
    static constexpr const char* kLogTag = "11-UndistortPointsCuda";

    explicit UndistortPointsCuda(const UndistortPointsParams& params = {});
    ~UndistortPointsCuda();

    UndistortPointsCuda(const UndistortPointsCuda&) = delete;
    UndistortPointsCuda& operator=(const UndistortPointsCuda&) = delete;

    UndistortPointsResult Execute(const cv::cuda::GpuMat& d_points,
                                   const cv::cuda::GpuMat& d_line_ids,
                                   cv::cuda::Stream& stream);

    UndistortPointsResult Execute(const cv::cuda::GpuMat& d_points,
                                   const cv::cuda::GpuMat& d_line_ids);

    UndistortPointsResult Execute(const cv::cuda::GpuMat& d_points,
                                   cv::cuda::Stream& stream);

    UndistortPointsResult Execute(const cv::cuda::GpuMat& d_points);

    void Destroy();
    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const UndistortPointsParams& params);
    const UndistortPointsParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getUndistortPointsCudaInfo();

} // namespace calib