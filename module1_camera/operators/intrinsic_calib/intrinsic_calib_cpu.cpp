#include "intrinsic_calib_cpu.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <spdlog/spdlog.h>
#include <sstream>

namespace calib {

OperatorInfo getIntrinsicCalibCPUInfo() {
    return OperatorInfo{"IntrinsicCalibCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

cv::Size IntrinsicCalibParams::imageSize() const noexcept {
    return {image_width, image_height};
}

int IntrinsicCalibParams::totalCorners() const noexcept {
    return chessboard_width * chessboard_height;
}

IntrinsicCalibParams IntrinsicCalibParams::fromJson(const std::string& json_path) {
    IntrinsicCalibParams params;
    try {
        std::ifstream ifs(json_path);
        if (!ifs.is_open()) {
            spdlog::warn("Cannot open params file: " + json_path + ", using defaults");
            return params;
        }
        nlohmann::json j = nlohmann::json::parse(ifs);

        if (j.contains("chessboard_width"))      params.chessboard_width = j.at("chessboard_width").get<int>();
        if (j.contains("chessboard_height"))     params.chessboard_height = j.at("chessboard_height").get<int>();
        if (j.contains("square_size_mm"))         params.square_size_mm = j.at("square_size_mm").get<double>();
        if (j.contains("image_width"))            params.image_width = j.at("image_width").get<int>();
        if (j.contains("image_height"))           params.image_height = j.at("image_height").get<int>();
        if (j.contains("use_calibrateCameraRO"))  params.use_calibrateCameraRO = j.at("use_calibrateCameraRO").get<bool>();
        if (j.contains("calib_flags"))            params.calib_flags = j.at("calib_flags").get<int>();
        if (j.contains("reproj_error_threshold")) params.reproj_error_threshold = j.at("reproj_error_threshold").get<double>();
        if (j.contains("temperature_coeff"))      params.temperature_coeff = j.at("temperature_coeff").get<double>();
        if (j.contains("plate_temp"))             params.plate_temp = j.at("plate_temp").get<double>();

        spdlog::info("Loaded params from " + json_path);
    } catch (const std::exception& e) {
        spdlog::warn("Failed to parse params file " + json_path + ": " + e.what());
    }
    return params;
}

bool MonocularCalibResult::isValid() const noexcept {
    return !camera_matrix.empty() && !dist_coeffs.empty();
}

void MonocularCalibResult::clear() {
    camera_matrix.release();
    dist_coeffs.release();
    rvecs.clear();
    tvecs.clear();
    rms_error = 0.0;
    per_view_errors.clear();
    valid_frame_count = 0;
    success = false;
    qualityFlag = calib::QualityFlag::Normal;
    message.clear();
}

bool IntrinsicCalibResult::isValid() const noexcept {
    return left.isValid() && right.isValid();
}

class IntrinsicCalibCPU::Impl {
public:
    explicit Impl(IntrinsicCalibParams p) : params_(std::move(p)) {}

    bool Execute(
        const std::vector<std::vector<cv::Point2f>>& left_points,
        const std::vector<std::vector<cv::Point2f>>& right_points,
        IntrinsicCalibResult& result)
    {
        if (left_points.size() != right_points.size()) {
            spdlog::error("Left/right frame count mismatch");
            result.success = false;
            result.message = "Left/right frame count mismatch";
            return false;
        }
        if (left_points.empty()) {
            spdlog::error("No input point data provided");
            result.success = false;
            result.message = "No input point data provided";
            return false;
        }

        result.total_frames_input = static_cast<int>(left_points.size());
        result.left.clear();
        result.right.clear();
        result.valid_frames_count = 0;

        std::vector<std::vector<cv::Point2f>> valid_left, valid_right;
        std::vector<std::vector<cv::Point3f>> object_points;

        filterValidFrames(left_points, right_points,
                          valid_left, valid_right,
                          object_points, result.valid_frames_count);

        if (result.valid_frames_count == 0) {
            spdlog::error("No valid frames with correct point count");
            result.success = false;
            result.message = "No valid frames with correct point count";
            return false;
        }

        spdlog::info("Valid frames: " + std::to_string(result.valid_frames_count) + "/" + std::to_string(left_points.size()));

        calibrateMonocular(object_points, valid_left, result.left);
        calibrateMonocular(object_points, valid_right, result.right);

        computeQualityMetrics(result);

        spdlog::info("Calibration complete.");
        result.success = true;
        result.qualityFlag = calib::QualityFlag::Normal;
        result.message.clear();
        return true;
    }

    bool Execute(
        const std::vector<std::vector<cv::Point2f>>& image_points,
        MonocularCalibResult& result)
    {
        if (image_points.empty()) {
            spdlog::error("No input point data provided");
            result.success = false;
            result.message = "No input point data provided";
            return false;
        }

        result.clear();
        std::vector<std::vector<cv::Point2f>> valid_points;
        std::vector<std::vector<cv::Point3f>> object_points;
        int valid_count = 0;

        filterValidFramesSingle(image_points, valid_points, object_points, valid_count);

        if (valid_count == 0) {
            spdlog::error("No valid frames with correct point count");
            result.success = false;
            result.message = "No valid frames with correct point count";
            return false;
        }

        calibrateMonocular(object_points, valid_points, result);

        spdlog::info("Single camera calibration complete.");
        return true;
    }

    void SetParams(const IntrinsicCalibParams& p) { params_ = p; }
    const IntrinsicCalibParams& GetParams() const noexcept { return params_; }

private:
    IntrinsicCalibParams params_;

    std::vector<cv::Point3f> generateObjectPoints() const {
        double actual_size = params_.square_size_mm *
            (1.0 + params_.temperature_coeff * (params_.plate_temp - 20.0));
        std::vector<cv::Point3f> obj;
        obj.reserve(params_.totalCorners());
        for (int i = 0; i < params_.chessboard_height; ++i) {
            for (int j = 0; j < params_.chessboard_width; ++j) {
                obj.emplace_back(
                    static_cast<float>(j * actual_size),
                    static_cast<float>(i * actual_size),
                    0.0f);
            }
        }
        return obj;
    }

    void filterValidFrames(
        const std::vector<std::vector<cv::Point2f>>& left_points,
        const std::vector<std::vector<cv::Point2f>>& right_points,
        std::vector<std::vector<cv::Point2f>>& valid_left,
        std::vector<std::vector<cv::Point2f>>& valid_right,
        std::vector<std::vector<cv::Point3f>>& object_points,
        int& valid_count) const
    {
        valid_count = 0;
        auto obj = generateObjectPoints();
        int expected = params_.totalCorners();

        for (size_t i = 0; i < left_points.size(); ++i) {
            if (static_cast<int>(left_points[i].size()) != expected) {
                spdlog::warn("Frame: point count mismatch, skipping");
                continue;
            }
            if (static_cast<int>(right_points[i].size()) != expected) {
                spdlog::warn("Frame: point count mismatch, skipping");
                continue;
            }

            bool has_nan = false;
            for (const auto& pt : left_points[i]) {
                if (std::isnan(pt.x) || std::isnan(pt.y)) { has_nan = true; break; }
            }
            if (!has_nan) {
                for (const auto& pt : right_points[i]) {
                    if (std::isnan(pt.x) || std::isnan(pt.y)) { has_nan = true; break; }
                }
            }
            if (has_nan) {
                spdlog::warn("Frame: contains NaN points, skipping");
                continue;
            }
            valid_left.push_back(left_points[i]);
            valid_right.push_back(right_points[i]);
            object_points.push_back(obj);
            ++valid_count;
        }
    }

    void filterValidFramesSingle(
        const std::vector<std::vector<cv::Point2f>>& image_points,
        std::vector<std::vector<cv::Point2f>>& valid_points,
        std::vector<std::vector<cv::Point3f>>& object_points,
        int& valid_count) const
    {
        valid_count = 0;
        auto obj = generateObjectPoints();
        int expected = params_.totalCorners();

        for (size_t i = 0; i < image_points.size(); ++i) {
            if (static_cast<int>(image_points[i].size()) != expected) {
                spdlog::warn("Frame: point count mismatch, skipping");
                continue;
            }

            bool has_nan = false;
            for (const auto& pt : image_points[i]) {
                if (std::isnan(pt.x) || std::isnan(pt.y)) { has_nan = true; break; }
            }
            if (has_nan) {
                spdlog::warn("Frame: contains NaN points, skipping");
                continue;
            }
            valid_points.push_back(image_points[i]);
            object_points.push_back(obj);
            ++valid_count;
        }
    }

    void calibrateMonocular(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        MonocularCalibResult& result) const
    {
        result.clear();
        result.valid_frame_count = static_cast<int>(image_points.size());

        if (params_.use_calibrateCameraRO) {
            cv::Mat new_obj_points;
            result.rms_error = cv::calibrateCameraRO(
                object_points, image_points, params_.imageSize(),
                0,
                result.camera_matrix, result.dist_coeffs,
                result.rvecs, result.tvecs, new_obj_points,
                params_.calib_flags);
        } else {
            result.rms_error = cv::calibrateCamera(
                object_points, image_points, params_.imageSize(),
                result.camera_matrix, result.dist_coeffs,
                result.rvecs, result.tvecs,
                params_.calib_flags);
        }

        result.per_view_errors = computePerViewErrors(
            object_points, image_points,
            result.camera_matrix, result.dist_coeffs,
            result.rvecs, result.tvecs);

        result.success = true;
        result.qualityFlag = calib::QualityFlag::Normal;
        result.message.clear();

        spdlog::info("Monocular calibration RMS done");
        spdlog::info("Camera matrix computed");
        spdlog::info("Distortion coefficients computed");
    }

    static std::vector<double> computePerViewErrors(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs)
    {
        std::vector<double> errors;
        errors.reserve(object_points.size());
        for (size_t i = 0; i < object_points.size(); ++i) {
            if (object_points[i].empty()) continue;
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points[i], rvecs[i], tvecs[i],
                              camera_matrix, dist_coeffs, projected);
            double total_err = 0.0;
            for (size_t j = 0; j < image_points[i].size(); ++j) {
                double dx = image_points[i][j].x - projected[j].x;
                double dy = image_points[i][j].y - projected[j].y;
                total_err += std::sqrt(dx * dx + dy * dy);
            }
            errors.push_back(total_err / static_cast<double>(image_points[i].size()));
        }
        return errors;
    }

    void computeQualityMetrics(IntrinsicCalibResult& result) const {
        auto calcStats = [](const std::vector<double>& values) -> std::pair<double, double> {
            if (values.empty()) return {0.0, 0.0};
            double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
            double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
            double stddev = std::sqrt(sq_sum / static_cast<double>(values.size()) - mean * mean);
            return {mean, stddev};
        };

        std::vector<double> all_errors;
        all_errors.reserve(result.left.per_view_errors.size() + result.right.per_view_errors.size());
        all_errors.insert(all_errors.end(), result.left.per_view_errors.begin(), result.left.per_view_errors.end());
        all_errors.insert(all_errors.end(), result.right.per_view_errors.begin(), result.right.per_view_errors.end());

        auto stats = calcStats(all_errors);
        result.reproj_error_mean = stats.first;
        result.reproj_error_std = stats.second;

        spdlog::info("Quality metrics computed");

        if (result.reproj_error_mean <= params_.reproj_error_threshold) {
            spdlog::info("Reprojection error within threshold");
        } else {
            spdlog::warn("Reprojection error exceeds threshold");
        }
    }
};

IntrinsicCalibCPU::IntrinsicCalibCPU(const IntrinsicCalibParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    spdlog::info("IntrinsicCalibCPU initialized");
}

IntrinsicCalibCPU::~IntrinsicCalibCPU() = default;

IntrinsicCalibCPU::IntrinsicCalibCPU(IntrinsicCalibCPU&&) noexcept = default;
IntrinsicCalibCPU& IntrinsicCalibCPU::operator=(IntrinsicCalibCPU&&) noexcept = default;

bool IntrinsicCalibCPU::Execute(
    const std::vector<std::vector<cv::Point2f>>& left_points,
    const std::vector<std::vector<cv::Point2f>>& right_points,
    IntrinsicCalibResult& result)
{
    return pImpl_->Execute(left_points, right_points, result);
}

bool IntrinsicCalibCPU::Execute(
    const std::vector<std::vector<cv::Point2f>>& image_points,
    MonocularCalibResult& result)
{
    return pImpl_->Execute(image_points, result);
}

void IntrinsicCalibCPU::SetParams(const IntrinsicCalibParams& params) {
    pImpl_->SetParams(params);
}

const IntrinsicCalibParams& IntrinsicCalibCPU::GetParams() const noexcept {
    return pImpl_->GetParams();
}

void IntrinsicCalibCPU::Destroy() {
}

bool IntrinsicCalibCPU::SaveResult(const std::string& filepath, const IntrinsicCalibResult& result) {
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        spdlog::error("Cannot open file for writing: " + filepath);
        return false;
    }
    saveMonoResultInternal(fs, "left", result.left);
    saveMonoResultInternal(fs, "right", result.right);
    fs << "reproj_error_mean" << result.reproj_error_mean;
    fs << "reproj_error_std" << result.reproj_error_std;
    fs << "valid_frames_count" << result.valid_frames_count;
    fs << "total_frames_input" << result.total_frames_input;
    spdlog::info("Calibration result saved to " + filepath);
    return true;
}

bool IntrinsicCalibCPU::LoadResult(const std::string& filepath, IntrinsicCalibResult& result) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        spdlog::error("Cannot open file for reading: " + filepath);
        return false;
    }
    loadMonoResultInternal(fs, "left", result.left);
    loadMonoResultInternal(fs, "right", result.right);
    fs["reproj_error_mean"] >> result.reproj_error_mean;
    fs["reproj_error_std"] >> result.reproj_error_std;
    fs["valid_frames_count"] >> result.valid_frames_count;
    fs["total_frames_input"] >> result.total_frames_input;
    result.left.success = true;
    result.right.success = true;
    result.success = true;
    result.qualityFlag = calib::QualityFlag::Normal;
    result.message.clear();
    spdlog::info("Calibration result loaded from " + filepath);
    return true;
}

bool IntrinsicCalibCPU::SaveMonoResult(const std::string& filepath, const std::string& prefix,
                                        const MonocularCalibResult& result) {
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    saveMonoResultInternal(fs, prefix, result);
    return true;
}

bool IntrinsicCalibCPU::LoadMonoResult(const std::string& filepath, const std::string& prefix,
                                        MonocularCalibResult& result) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    loadMonoResultInternal(fs, prefix, result);
    result.success = true;
    result.qualityFlag = calib::QualityFlag::Normal;
    result.message.clear();
    return true;
}

void IntrinsicCalibCPU::saveMonoResultInternal(cv::FileStorage& fs, const std::string& prefix,
                                                const MonocularCalibResult& result) {
    fs << (prefix + "_camera_matrix") << result.camera_matrix;
    fs << (prefix + "_dist_coeffs") << result.dist_coeffs;
    fs << (prefix + "_rms_error") << result.rms_error;
    fs << (prefix + "_valid_frame_count") << result.valid_frame_count;

    fs << (prefix + "_per_view_errors") << "[";
    for (double err : result.per_view_errors) {
        fs << err;
    }
    fs << "]";

    fs << (prefix + "_rvecs") << "[";
    for (const auto& r : result.rvecs) fs << r;
    fs << "]";

    fs << (prefix + "_tvecs") << "[";
    for (const auto& t : result.tvecs) fs << t;
    fs << "]";
}

void IntrinsicCalibCPU::loadMonoResultInternal(cv::FileStorage& fs, const std::string& prefix,
                                                MonocularCalibResult& result) {
    fs[(prefix + "_camera_matrix")] >> result.camera_matrix;
    fs[(prefix + "_dist_coeffs")] >> result.dist_coeffs;
    fs[(prefix + "_rms_error")] >> result.rms_error;
    fs[(prefix + "_valid_frame_count")] >> result.valid_frame_count;

    result.per_view_errors.clear();
    cv::FileNode pve = fs[(prefix + "_per_view_errors")];
    if (pve.type() == cv::FileNode::SEQ) {
        for (auto it = pve.begin(); it != pve.end(); ++it) {
            result.per_view_errors.push_back((*it).real());
        }
    }

    result.rvecs.clear();
    cv::FileNode rv = fs[(prefix + "_rvecs")];
    if (rv.type() == cv::FileNode::SEQ) {
        for (auto it = rv.begin(); it != rv.end(); ++it) {
            cv::Mat m;
            *it >> m;
            result.rvecs.push_back(m);
        }
    }

    result.tvecs.clear();
    cv::FileNode tv = fs[(prefix + "_tvecs")];
    if (tv.type() == cv::FileNode::SEQ) {
        for (auto it = tv.begin(); it != tv.end(); ++it) {
            cv::Mat m;
            *it >> m;
            result.tvecs.push_back(m);
        }
    }
}

nlohmann::json MonocularCalibResult::toJson() const {
    nlohmann::json j;
    j["camera_matrix"] = calib::matToJson(camera_matrix);
    j["dist_coeffs"] = calib::matToJson(dist_coeffs);
    j["rms_error"] = rms_error;
    j["valid_frame_count"] = valid_frame_count;
    j["per_view_errors"] = per_view_errors;

    nlohmann::json rvecsArr = nlohmann::json::array();
    for (const auto& r : rvecs) {
        rvecsArr.push_back(calib::matToJson(r));
    }
    j["rvecs"] = rvecsArr;

    nlohmann::json tvecsArr = nlohmann::json::array();
    for (const auto& t : tvecs) {
        tvecsArr.push_back(calib::matToJson(t));
    }
    j["tvecs"] = tvecsArr;
    return j;
}

MonocularCalibResult MonocularCalibResult::fromJson(const nlohmann::json& j) {
    MonocularCalibResult r;
    r.camera_matrix = calib::jsonToMatAuto(j.at("camera_matrix"));
    r.dist_coeffs = calib::jsonToMatAuto(j.at("dist_coeffs"));
    r.rms_error = j.at("rms_error").get<double>();
    r.valid_frame_count = j.at("valid_frame_count").get<int>();
    r.per_view_errors = j.at("per_view_errors").get<std::vector<double>>();

    r.rvecs.clear();
    for (const auto& item : j.at("rvecs")) {
        r.rvecs.push_back(calib::jsonToMatAuto(item));
    }
    r.tvecs.clear();
    for (const auto& item : j.at("tvecs")) {
        r.tvecs.push_back(calib::jsonToMatAuto(item));
    }
    r.success = true;
    r.qualityFlag = calib::QualityFlag::Normal;
    r.message.clear();
    return r;
}

nlohmann::json IntrinsicCalibResult::toJson() const {
    nlohmann::json j;
    j["left"] = left.toJson();
    j["right"] = right.toJson();
    j["reproj_error_mean"] = reproj_error_mean;
    j["reproj_error_std"] = reproj_error_std;
    j["valid_frames_count"] = valid_frames_count;
    j["total_frames_input"] = total_frames_input;
    return j;
}

IntrinsicCalibResult IntrinsicCalibResult::fromJson(const nlohmann::json& j) {
    IntrinsicCalibResult r;
    r.left = MonocularCalibResult::fromJson(j.at("left"));
    r.right = MonocularCalibResult::fromJson(j.at("right"));
    r.reproj_error_mean = j.at("reproj_error_mean").get<double>();
    r.reproj_error_std = j.at("reproj_error_std").get<double>();
    r.valid_frames_count = j.at("valid_frames_count").get<int>();
    r.total_frames_input = j.at("total_frames_input").get<int>();
    r.success = r.left.success && r.right.success;
    r.qualityFlag = calib::QualityFlag::Normal;
    r.message.clear();
    return r;
}

bool IntrinsicCalibCPU::SaveResultJson(const std::string& filepath, const IntrinsicCalibResult& result) {
    try {
        nlohmann::json j = result.toJson();
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            spdlog::error("Cannot open file for writing: " + filepath);
            return false;
        }
        ofs << j.dump(2);
        spdlog::info("Calibration result saved as JSON to " + filepath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to save JSON result: " + std::string(e.what()));
        return false;
    }
}

bool IntrinsicCalibCPU::LoadResultJson(const std::string& filepath, IntrinsicCalibResult& result) {
    try {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            spdlog::error("Cannot open file for reading: " + filepath);
            return false;
        }
        nlohmann::json j = nlohmann::json::parse(ifs);
        result = IntrinsicCalibResult::fromJson(j);
        spdlog::info("Calibration result loaded from JSON " + filepath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to load JSON result: " + std::string(e.what()));
        return false;
    }
}

} // namespace calib
