#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "common/json_utils.h"
#include "common/quality_flag.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct IntrinsicCalibParams {
    int chessboard_width = 11;
    int chessboard_height = 8;
    double square_size_mm = 15.0;
    int image_width = 2048;
    int image_height = 1536;
    bool use_calibrateCameraRO = true;
    int calib_flags = 0;
    double reproj_error_threshold = 0.012;
    double temperature_coeff = 5.0e-6;
    double plate_temp = 21.0;

    static IntrinsicCalibParams fromJson(const std::string& json_path);
    cv::Size imageSize() const noexcept;
    int totalCorners() const noexcept;
};

struct MonocularCalibResult {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double rms_error = 0.0;
    std::vector<double> per_view_errors;
    int valid_frame_count = 0;

    // 算子规范 §5.1 单一错误模型
    bool success = false;
    calib::QualityFlag qualityFlag = calib::QualityFlag::Normal;
    std::string message;

    MonocularCalibResult() = default;
    // move-only (spec §3.3)
    MonocularCalibResult(const MonocularCalibResult&) = delete;
    MonocularCalibResult& operator=(const MonocularCalibResult&) = delete;
    MonocularCalibResult(MonocularCalibResult&&) noexcept = default;
    MonocularCalibResult& operator=(MonocularCalibResult&&) noexcept = default;

    bool isValid() const noexcept;
    void clear();

    nlohmann::json toJson() const;
    static MonocularCalibResult fromJson(const nlohmann::json& j);
};

struct IntrinsicCalibResult {
    MonocularCalibResult left;
    MonocularCalibResult right;
    double reproj_error_mean = 0.0;
    double reproj_error_std = 0.0;
    int total_frames_input = 0;
    int valid_frames_count = 0;

    // 算子规范 §5.1 单一错误模型
    bool success = false;
    calib::QualityFlag qualityFlag = calib::QualityFlag::Normal;
    std::string message;

    IntrinsicCalibResult() = default;
    // move-only (spec §3.3)
    IntrinsicCalibResult(const IntrinsicCalibResult&) = delete;
    IntrinsicCalibResult& operator=(const IntrinsicCalibResult&) = delete;
    IntrinsicCalibResult(IntrinsicCalibResult&&) noexcept = default;
    IntrinsicCalibResult& operator=(IntrinsicCalibResult&&) noexcept = default;

    bool isValid() const noexcept;

    nlohmann::json toJson() const;
    static IntrinsicCalibResult fromJson(const nlohmann::json& j);
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 观测点集按调用传入，一次性标定求解并返回；SetParams 缓存棋盘格参数等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API IntrinsicCalibCPU {
public:
    explicit IntrinsicCalibCPU(const IntrinsicCalibParams& params);
    ~IntrinsicCalibCPU();

    IntrinsicCalibCPU(const IntrinsicCalibCPU&) = delete;
    IntrinsicCalibCPU& operator=(const IntrinsicCalibCPU&) = delete;
    IntrinsicCalibCPU(IntrinsicCalibCPU&&) noexcept;
    IntrinsicCalibCPU& operator=(IntrinsicCalibCPU&&) noexcept;

    bool Execute(
        const std::vector<std::vector<cv::Point2f>>& left_points,
        const std::vector<std::vector<cv::Point2f>>& right_points,
        IntrinsicCalibResult& result);

    bool Execute(
        const std::vector<std::vector<cv::Point2f>>& image_points,
        MonocularCalibResult& result);

    void Warmup() { }

    void Destroy();

    void SetParams(const IntrinsicCalibParams& params);
    const IntrinsicCalibParams& GetParams() const noexcept;

    static bool SaveResult(const std::string& filepath, const IntrinsicCalibResult& result);
    static bool LoadResult(const std::string& filepath, IntrinsicCalibResult& result);
    static bool SaveMonoResult(const std::string& filepath, const std::string& prefix, const MonocularCalibResult& result);
    static bool LoadMonoResult(const std::string& filepath, const std::string& prefix, MonocularCalibResult& result);

    static bool SaveResultJson(const std::string& filepath, const IntrinsicCalibResult& result);
    static bool LoadResultJson(const std::string& filepath, IntrinsicCalibResult& result);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    static void saveMonoResultInternal(cv::FileStorage& fs, const std::string& prefix, const MonocularCalibResult& result);
    static void loadMonoResultInternal(cv::FileStorage& fs, const std::string& prefix, MonocularCalibResult& result);
};

OperatorInfo getIntrinsicCalibCPUInfo();

} // namespace calib
