#include "extrinsic_calib_cpu.h"
#include <opencv2/calib3d.hpp>
#include <numeric>
#include <sstream>
#include <spdlog/spdlog.h>

namespace calib {

OperatorInfo getExtrinsicCalibCpuInfo() {
    return OperatorInfo{"ExtrinsicCalibCpu", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

namespace {
constexpr double kFxMin = 100.0;
constexpr double kFxMax = 10000.0;
constexpr double kFyMin = 100.0;
constexpr double kFyMax = 10000.0;
constexpr double kPrincipalPointRatio = 0.3;
constexpr double kAspectRatioMin = 0.8;
constexpr double kAspectRatioMax = 1.2;

void point2fVectorToJson(const std::vector<std::vector<cv::Point2f>>& pts, nlohmann::json& arr) {
    arr = nlohmann::json::array();
    for (const auto& view : pts) {
        nlohmann::json viewArr = nlohmann::json::array();
        for (const auto& pt : view) {
            nlohmann::json p = nlohmann::json::array();
            p.push_back(pt.x);
            p.push_back(pt.y);
            viewArr.push_back(p);
        }
        arr.push_back(viewArr);
    }
}

void point3fVectorToJson(const std::vector<cv::Point3f>& pts, nlohmann::json& arr) {
    arr = nlohmann::json::array();
    for (const auto& pt : pts) {
        nlohmann::json p = nlohmann::json::array();
        p.push_back(pt.x);
        p.push_back(pt.y);
        p.push_back(pt.z);
        arr.push_back(p);
    }
}

std::vector<std::vector<cv::Point2f>> jsonToPoint2fVector(const nlohmann::json& arr) {
    std::vector<std::vector<cv::Point2f>> result;
    for (size_t i = 0; i < arr.size(); ++i) {
        std::vector<cv::Point2f> view;
        const auto& viewArr = arr[i];
        for (size_t j = 0; j < viewArr.size(); ++j) {
            float x = static_cast<float>(viewArr[j][0].get<double>());
            float y = static_cast<float>(viewArr[j][1].get<double>());
            view.emplace_back(x, y);
        }
        result.push_back(std::move(view));
    }
    return result;
}

std::vector<cv::Point3f> jsonToPoint3fVector(const nlohmann::json& arr) {
    std::vector<cv::Point3f> result;
    for (size_t i = 0; i < arr.size(); ++i) {
        float x = static_cast<float>(arr[i][0].get<double>());
        float y = static_cast<float>(arr[i][1].get<double>());
        float z = static_cast<float>(arr[i][2].get<double>());
        result.emplace_back(x, y, z);
    }
    return result;
}

void checkIntrinsicsImpl(const cv::Mat& K, const std::string& label,
                          const cv::Size& imageSize,
                          bool& hasWarning, std::string& warnMsg) {
    if (K.empty() || K.rows != 3 || K.cols != 3) return;

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    std::ostringstream oss;
    bool warn = false;

    if (fx < kFxMin || fx > kFxMax || fy < kFyMin || fy > kFyMax) {
        oss << label << ": focal length abnormal (fx=" << fx << ", fy=" << fy << "); ";
        warn = true;
    }
    double cxCenter = imageSize.width / 2.0;
    double cyCenter = imageSize.height / 2.0;
    if (std::abs(cx - cxCenter) > imageSize.width * kPrincipalPointRatio ||
        std::abs(cy - cyCenter) > imageSize.height * kPrincipalPointRatio) {
        oss << label << ": principal point offset abnormal (cx=" << cx << ", cy=" << cy << "); ";
        warn = true;
    }
    if (fy > 0 && (fx / fy < kAspectRatioMin || fx / fy > kAspectRatioMax)) {
        oss << label << ": aspect ratio abnormal (fx/fy=" << fx / fy << "); ";
        warn = true;
    }

    if (warn) {
        hasWarning = true;
        warnMsg += oss.str();
    }
}

ExtrinsicCalibCpuMonoResult calibrateMonoSideImpl(
    const ExtrinsicCalibCpuParams& params,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::string& sideName) {
    ExtrinsicCalibCpuMonoResult result;

    std::vector<std::vector<cv::Point3f>> objPointsPerView(
        imagePoints.size(), params.objectPoints);

    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat newObjPoints;

    try {
        double rms = cv::calibrateCameraRO(
            objPointsPerView, imagePoints,
            params.imageSize, 0,
            K, D, rvecs, tvecs, newObjPoints
        );

        result.cameraMatrix = K.clone();
        result.distCoeffs = D.clone();
        result.rvecs = rvecs;
        result.tvecs = tvecs;
        result.reprojError = rms;
        result.success = true;

        for (size_t i = 0; i < objPointsPerView.size(); ++i) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(objPointsPerView[i], rvecs[i], tvecs[i], K, D, projected);
            double totalErr = 0.0;
            for (size_t j = 0; j < imagePoints[i].size(); ++j) {
                double dx = imagePoints[i][j].x - projected[j].x;
                double dy = imagePoints[i][j].y - projected[j].y;
                totalErr += std::sqrt(dx * dx + dy * dy);
            }
            result.perViewErrors.push_back(totalErr / imagePoints[i].size());
        }

        bool warn = false;
        std::string warnMsg;
        checkIntrinsicsImpl(K, sideName, params.imageSize, warn, warnMsg);
        if (warn) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = warnMsg;
            spdlog::warn(std::string(ExtrinsicCalibCpu::kLogTag) + " WARN: " + warnMsg);
        }

        spdlog::info(std::string(ExtrinsicCalibCpu::kLogTag) + " INFO: " + sideName +
                     " mono calibration RMS: " + std::to_string(rms) + " pixels");
    }
    catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string(e.what());
        spdlog::error(std::string(ExtrinsicCalibCpu::kLogTag) + " ERROR: " + sideName +
                      " mono calibrateCameraRO failed: " + e.what());
    }
    catch (const std::exception& e) {
        result.success = false;
        result.message = std::string(e.what());
        spdlog::error(std::string(ExtrinsicCalibCpu::kLogTag) + " ERROR: " + sideName +
                      " mono calibration failed: " + e.what());
    }
    catch (...) {
        result.success = false;
        result.message = "memory allocation failed or unknown error";
        spdlog::error(std::string(ExtrinsicCalibCpu::kLogTag) + " ERROR: " + sideName +
                      " mono calibration unknown error");
    }

    return result;
}

std::vector<double> computePerViewErrorsStereoImpl(
    const std::vector<std::vector<cv::Point3f>>& objPointsPerView,
    const std::vector<std::vector<cv::Point2f>>& leftPoints,
    const std::vector<std::vector<cv::Point2f>>& rightPoints,
    const cv::Mat& KL, const cv::Mat& DL,
    const cv::Mat& KR, const cv::Mat& DR,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& R, const cv::Mat& T) {
    std::vector<double> errors;

    cv::Mat R_mat = R;
    if (R_mat.rows == 3 && R_mat.cols == 1) {
        cv::Rodrigues(R_mat, R_mat);
    }

    for (size_t i = 0; i < objPointsPerView.size(); ++i) {
        std::vector<cv::Point2f> projectedLeft;
        cv::projectPoints(objPointsPerView[i], rvecs[i], tvecs[i], KL, DL, projectedLeft);
        double leftErr = 0.0;
        for (size_t j = 0; j < leftPoints[i].size(); ++j) {
            double dx = leftPoints[i][j].x - projectedLeft[j].x;
            double dy = leftPoints[i][j].y - projectedLeft[j].y;
            leftErr += std::sqrt(dx * dx + dy * dy);
        }
        leftErr /= leftPoints[i].size();

        cv::Mat R_left;
        if (rvecs[i].rows == 3 && rvecs[i].cols == 1) {
            cv::Rodrigues(rvecs[i], R_left);
        } else {
            R_left = rvecs[i];
        }
        cv::Mat R_right = R_mat * R_left;
        cv::Mat t_right = R_mat * tvecs[i] + T;
        cv::Mat rvec_right;
        cv::Rodrigues(R_right, rvec_right);

        std::vector<cv::Point2f> projectedRight;
        cv::projectPoints(objPointsPerView[i], rvec_right, t_right, KR, DR, projectedRight);
        double rightErr = 0.0;
        for (size_t j = 0; j < rightPoints[i].size(); ++j) {
            double dx = rightPoints[i][j].x - projectedRight[j].x;
            double dy = rightPoints[i][j].y - projectedRight[j].y;
            rightErr += std::sqrt(dx * dx + dy * dy);
        }
        rightErr /= rightPoints[i].size();

        errors.push_back((leftErr + rightErr) / 2.0);
    }

    return errors;
}

std::vector<double> computeEpipolarErrorsImpl(
    const std::vector<std::vector<cv::Point2f>>& leftPoints,
    const std::vector<std::vector<cv::Point2f>>& rightPoints,
    const cv::Mat& F) {
    std::vector<double> errors;

    for (size_t i = 0; i < leftPoints.size(); ++i) {
        std::vector<cv::Vec3f> lines;
        cv::computeCorrespondEpilines(leftPoints[i], 1, F, lines);

        double totalError = 0.0;
        for (size_t j = 0; j < rightPoints[i].size(); ++j) {
            double a = lines[j].val[0];
            double b = lines[j].val[1];
            double c = lines[j].val[2];
            double dist = std::abs(a * rightPoints[i][j].x +
                                    b * rightPoints[i][j].y + c) /
                          std::sqrt(a * a + b * b);
            totalError += dist;
        }
        errors.push_back(totalError / leftPoints[i].size());
    }

    return errors;
}

void computeEpipolarStatsImpl(const std::vector<double>& perViewErrors,
                               double& mean, double& stddev) {
    if (perViewErrors.empty()) {
        mean = 0.0;
        stddev = 0.0;
        return;
    }
    double sum = std::accumulate(perViewErrors.begin(), perViewErrors.end(), 0.0);
    mean = sum / perViewErrors.size();
    double sqSum = std::inner_product(perViewErrors.begin(), perViewErrors.end(),
                                       perViewErrors.begin(), 0.0);
    double variance = sqSum / perViewErrors.size() - mean * mean;
    stddev = std::sqrt(std::max(0.0, variance));
}

} // anonymous namespace

// ═══════════════════════════════════════════════
// ExtrinsicCalibCpuResult JSON serialization
// ═══════════════════════════════════════════════

nlohmann::json ExtrinsicCalibCpuResult::toJson() const {
    nlohmann::json j;
    j["success"] = success;
    j["message"] = message;
    j["qualityFlag"] = static_cast<int>(qualityFlag);
    j["R"] = calib::matToJson(R);
    j["T"] = calib::matToJson(T);
    j["E"] = calib::matToJson(E);
    j["F"] = calib::matToJson(F);
    j["stereoReprojError"] = stereoReprojError;
    j["epipolarErrorMean"] = epipolarErrorMean;
    j["epipolarErrorStd"] = epipolarErrorStd;
    j["perViewErrors"] = perViewErrors;
    j["perViewEpipolarErrors"] = perViewEpipolarErrors;
    if (!cameraMatrixL.empty()) j["camera_matrix_l"] = calib::matToJson(cameraMatrixL);
    if (!distCoeffsL.empty()) j["dist_coeffs_l"] = calib::matToJson(distCoeffsL);
    if (!cameraMatrixR.empty()) j["camera_matrix_r"] = calib::matToJson(cameraMatrixR);
    if (!distCoeffsR.empty()) j["dist_coeffs_r"] = calib::matToJson(distCoeffsR);
    return j;
}

ExtrinsicCalibCpuResult ExtrinsicCalibCpuResult::fromJson(const nlohmann::json& j) {
    ExtrinsicCalibCpuResult r;
    r.success = j.at("success").get<bool>();
    r.message = j.at("message").get<std::string>();
    r.qualityFlag = static_cast<QualityFlag>(j.at("qualityFlag").get<int>());
    r.R = calib::jsonToMatAuto(j.at("R"));
    r.T = calib::jsonToMatAuto(j.at("T"));
    r.E = calib::jsonToMatAuto(j.at("E"));
    r.F = calib::jsonToMatAuto(j.at("F"));
    r.stereoReprojError = j.at("stereoReprojError").get<double>();
    r.epipolarErrorMean = j.at("epipolarErrorMean").get<double>();
    r.epipolarErrorStd = j.at("epipolarErrorStd").get<double>();
    r.perViewErrors = j.at("perViewErrors").get<std::vector<double>>();
    r.perViewEpipolarErrors = j.at("perViewEpipolarErrors").get<std::vector<double>>();
    if (j.contains("camera_matrix_l") && j.at("camera_matrix_l").is_array())
        r.cameraMatrixL = calib::jsonToMatAuto(j.at("camera_matrix_l"));
    if (j.contains("dist_coeffs_l") && j.at("dist_coeffs_l").is_array())
        r.distCoeffsL = calib::jsonToMatAuto(j.at("dist_coeffs_l"));
    if (j.contains("camera_matrix_r") && j.at("camera_matrix_r").is_array())
        r.cameraMatrixR = calib::jsonToMatAuto(j.at("camera_matrix_r"));
    if (j.contains("dist_coeffs_r") && j.at("dist_coeffs_r").is_array())
        r.distCoeffsR = calib::jsonToMatAuto(j.at("dist_coeffs_r"));
    return r;
}

// ═══════════════════════════════════════════════
// ExtrinsicCalibCpuParams implementation
// ═══════════════════════════════════════════════

nlohmann::json ExtrinsicCalibCpuParams::toJson() const {
    nlohmann::json j;

    nlohmann::json leftArr, rightArr, objArr;
    point2fVectorToJson(leftPointsPerView, leftArr);
    point2fVectorToJson(rightPointsPerView, rightArr);
    point3fVectorToJson(objectPoints, objArr);

    j["leftPointsPerView"] = leftArr;
    j["rightPointsPerView"] = rightArr;
    j["objectPoints"] = objArr;
    j["image_width"] = imageSize.width;
    j["image_height"] = imageSize.height;
    j["flags"] = flags;
    j["calibrate_mono"] = calibrateMono;
    j["pattern_width"] = patternSize.width;
    j["pattern_height"] = patternSize.height;
    j["square_size"] = squareSize;
    j["max_reproj_error"] = maxReprojError;
    j["min_view_count"] = minViewCount;
    j["rotate_right_image_180"] = rotateRightImage180;
    j["max_epipolar_error"] = maxEpipolarError;
    return j;
}

ExtrinsicCalibCpuParams ExtrinsicCalibCpuParams::fromJson(const nlohmann::json& j) {
    ExtrinsicCalibCpuParams p;
    const std::string ctx = "ExtrinsicCalibCpuParams";

    p.leftPointsPerView = jsonToPoint2fVector(getRequired<nlohmann::json>(j, "leftPointsPerView", ctx));
    p.rightPointsPerView = jsonToPoint2fVector(getRequired<nlohmann::json>(j, "rightPointsPerView", ctx));
    p.objectPoints = jsonToPoint3fVector(getRequired<nlohmann::json>(j, "objectPoints", ctx));
    p.imageSize = cv::Size(
        getRequired<int>(j, "image_width", ctx),
        getRequired<int>(j, "image_height", ctx)
    );
    p.flags = j.contains("flags") ? j.at("flags").get<int>() : 0;
    p.calibrateMono = j.contains("calibrate_mono") ? j.at("calibrate_mono").get<bool>() : false;
    if (j.contains("pattern_width") && j.contains("pattern_height")) {
        p.patternSize = cv::Size(j.at("pattern_width").get<int>(), j.at("pattern_height").get<int>());
    }
    p.squareSize = j.contains("square_size") ? static_cast<float>(j.at("square_size").get<double>()) : 0.0f;
    p.maxReprojError = j.contains("max_reproj_error") ? j.at("max_reproj_error").get<double>() : 1.0;
    p.minViewCount = j.contains("min_view_count") ? j.at("min_view_count").get<int>() : 8;
    p.rotateRightImage180 = j.contains("rotate_right_image_180") ? j.at("rotate_right_image_180").get<bool>() : false;
    p.maxEpipolarError = j.contains("max_epipolar_error") ? j.at("max_epipolar_error").get<double>() : 0.05;

    p.validate();
    return p;
}

void ExtrinsicCalibCpuParams::validate() const {
    if (leftPointsPerView.empty()) {
        throw std::invalid_argument("ExtrinsicCalibCpuParams: leftPointsPerView must not be empty");
    }
    if (static_cast<int>(leftPointsPerView.size()) < minViewCount) {
        throw std::invalid_argument(
            "ExtrinsicCalibCpuParams: leftPointsPerView has " +
            std::to_string(leftPointsPerView.size()) + " views, need at least " +
            std::to_string(minViewCount));
    }
    if (rightPointsPerView.empty()) {
        throw std::invalid_argument("ExtrinsicCalibCpuParams: rightPointsPerView must not be empty");
    }
    if (leftPointsPerView.size() != rightPointsPerView.size()) {
        throw std::invalid_argument(
            "ExtrinsicCalibCpuParams: left/right view count mismatch (" +
            std::to_string(leftPointsPerView.size()) + " vs " +
            std::to_string(rightPointsPerView.size()) + ")");
    }
    for (size_t i = 0; i < leftPointsPerView.size(); ++i) {
        if (leftPointsPerView[i].empty()) {
            throw std::invalid_argument("ExtrinsicCalibCpuParams: leftPointsPerView[" + std::to_string(i) + "] is empty");
        }
        if (leftPointsPerView[i].size() != objectPoints.size()) {
            throw std::invalid_argument(
                "ExtrinsicCalibCpuParams: leftPointsPerView[" + std::to_string(i) + "] has " +
                std::to_string(leftPointsPerView[i].size()) + " points, expected " +
                std::to_string(objectPoints.size()));
        }
    }
    for (size_t i = 0; i < rightPointsPerView.size(); ++i) {
        if (rightPointsPerView[i].empty()) {
            throw std::invalid_argument("ExtrinsicCalibCpuParams: rightPointsPerView[" + std::to_string(i) + "] is empty");
        }
        if (rightPointsPerView[i].size() != objectPoints.size()) {
            throw std::invalid_argument(
                "ExtrinsicCalibCpuParams: rightPointsPerView[" + std::to_string(i) + "] has " +
                std::to_string(rightPointsPerView[i].size()) + " points, expected " +
                std::to_string(objectPoints.size()));
        }
    }
    if (objectPoints.size() < 4) {
        throw std::invalid_argument("ExtrinsicCalibCpuParams: objectPoints must have at least 4 points");
    }
    if (imageSize.width <= 0 || imageSize.height <= 0) {
        throw std::invalid_argument("ExtrinsicCalibCpuParams: imageSize must have positive width and height");
    }
    if (maxReprojError <= 0) {
        throw std::invalid_argument("ExtrinsicCalibCpuParams: maxReprojError must be > 0");
    }
    if (minViewCount < 3) {
        throw std::invalid_argument("ExtrinsicCalibCpuParams: minViewCount must be >= 3");
    }
    if (calibrateMono) {
        if (patternSize.width <= 0 || patternSize.height <= 0) {
            throw std::invalid_argument("ExtrinsicCalibCpuParams: patternSize must have positive dimensions when calibrateMono=true");
        }
        if (squareSize <= 0.0f) {
            throw std::invalid_argument("ExtrinsicCalibCpuParams: squareSize must be > 0 when calibrateMono=true");
        }
        if (static_cast<int>(objectPoints.size()) != patternSize.width * patternSize.height) {
            throw std::invalid_argument(
                "ExtrinsicCalibCpuParams: objectPoints count (" + std::to_string(objectPoints.size()) +
                ") != patternSize.width * patternSize.height (" +
                std::to_string(patternSize.width * patternSize.height) + ")");
        }
    }
}

// ═══════════════════════════════════════════════
// ExtrinsicCalibCpu::Impl implementation
// ═══════════════════════════════════════════════

class ExtrinsicCalibCpu::Impl {
public:
    explicit Impl(ExtrinsicCalibCpuParams p) : params_(std::move(p)) { params_.validate(); }

    void SetParams(const ExtrinsicCalibCpuParams& params) { params.validate(); params_ = params; }
    const ExtrinsicCalibCpuParams& GetParams() const { return params_; }

    ExtrinsicCalibCpuResult Execute(
        const cv::Mat& cameraMatrixL, const cv::Mat& distCoeffsL,
        const cv::Mat& cameraMatrixR, const cv::Mat& distCoeffsR);

    ExtrinsicCalibCpuFullResult Execute();

private:
    ExtrinsicCalibCpuParams params_;

    void checkIntrinsics(const cv::Mat& K, const std::string& label,
                         bool& hasWarning, std::string& warnMsg);
    ExtrinsicCalibCpuMonoResult calibrateMonoSide(
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        const std::string& sideName);
    std::vector<double> computePerViewErrorsStereo(
        const std::vector<std::vector<cv::Point3f>>& objPointsPerView,
        const std::vector<std::vector<cv::Point2f>>& leftPoints,
        const std::vector<std::vector<cv::Point2f>>& rightPoints,
        const cv::Mat& KL, const cv::Mat& DL,
        const cv::Mat& KR, const cv::Mat& DR,
        const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
        const cv::Mat& R, const cv::Mat& T);
    std::vector<double> computeEpipolarErrors(
        const std::vector<std::vector<cv::Point2f>>& leftPoints,
        const std::vector<std::vector<cv::Point2f>>& rightPoints,
        const cv::Mat& F);
    void computeEpipolarStats(const std::vector<double>& perViewErrors,
                              double& mean, double& stddev);
};

void ExtrinsicCalibCpu::Impl::checkIntrinsics(const cv::Mat& K, const std::string& label,
                                                bool& hasWarning, std::string& warnMsg) {
    checkIntrinsicsImpl(K, label, params_.imageSize, hasWarning, warnMsg);
}

ExtrinsicCalibCpuMonoResult ExtrinsicCalibCpu::Impl::calibrateMonoSide(
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::string& sideName) {
    return calibrateMonoSideImpl(params_, imagePoints, sideName);
}

std::vector<double> ExtrinsicCalibCpu::Impl::computePerViewErrorsStereo(
    const std::vector<std::vector<cv::Point3f>>& objPointsPerView,
    const std::vector<std::vector<cv::Point2f>>& leftPoints,
    const std::vector<std::vector<cv::Point2f>>& rightPoints,
    const cv::Mat& KL, const cv::Mat& DL,
    const cv::Mat& KR, const cv::Mat& DR,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& R, const cv::Mat& T) {
    return computePerViewErrorsStereoImpl(objPointsPerView, leftPoints, rightPoints,
                                           KL, DL, KR, DR, rvecs, tvecs, R, T);
}

std::vector<double> ExtrinsicCalibCpu::Impl::computeEpipolarErrors(
    const std::vector<std::vector<cv::Point2f>>& leftPoints,
    const std::vector<std::vector<cv::Point2f>>& rightPoints,
    const cv::Mat& F) {
    return computeEpipolarErrorsImpl(leftPoints, rightPoints, F);
}

void ExtrinsicCalibCpu::Impl::computeEpipolarStats(const std::vector<double>& perViewErrors,
                                                     double& mean, double& stddev) {
    computeEpipolarStatsImpl(perViewErrors, mean, stddev);
}

ExtrinsicCalibCpuResult ExtrinsicCalibCpu::Impl::Execute(
    const cv::Mat& cameraMatrixL, const cv::Mat& distCoeffsL,
    const cv::Mat& cameraMatrixR, const cv::Mat& distCoeffsR) {
    ExtrinsicCalibCpuResult result;

    if (params_.rotateRightImage180) {
        spdlog::info(std::string(kLogTag) + " INFO: Right image 180 deg rotation noted (preprocessed by caller)");
    }

    bool warnL = false, warnR = false;
    std::string warnMsg;
    checkIntrinsics(cameraMatrixL, "Left input intrinsics", warnL, warnMsg);
    checkIntrinsics(cameraMatrixR, "Right input intrinsics", warnR, warnMsg);
    if (warnL || warnR) {
        result.qualityFlag = QualityFlag::Warning;
        result.message = warnMsg;
        spdlog::warn(std::string(kLogTag) + " WARN: " + warnMsg);
    }

    std::vector<std::vector<cv::Point3f>> objPointsPerView(
        params_.leftPointsPerView.size(), params_.objectPoints);

    try {
        cv::Mat R, T, E, F;
        std::vector<cv::Mat> rvecsStereo, tvecsStereo;

        double rms = cv::stereoCalibrate(
            objPointsPerView, params_.leftPointsPerView, params_.rightPointsPerView,
            cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,
            params_.imageSize,
            R, T, E, F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6)
        );

        result.R = R.clone();
        result.T = T.clone();
        result.E = E.clone();
        result.F = F.clone();
        result.cameraMatrixL = cameraMatrixL.clone();
        result.cameraMatrixR = cameraMatrixR.clone();
        result.distCoeffsL = distCoeffsL.clone();
        result.distCoeffsR = distCoeffsR.clone();
        result.stereoReprojError = rms;

        if (!rvecsStereo.empty() && !tvecsStereo.empty()) {
            result.perViewErrors = computePerViewErrorsStereo(
                objPointsPerView, params_.leftPointsPerView, params_.rightPointsPerView,
                cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,
                rvecsStereo, tvecsStereo, R, T);
        }

        result.perViewEpipolarErrors = computeEpipolarErrors(
            params_.leftPointsPerView, params_.rightPointsPerView, F);
        computeEpipolarStats(result.perViewEpipolarErrors,
                              result.epipolarErrorMean, result.epipolarErrorStd);

        result.success = true;
        if (rms > params_.maxReprojError) {
            result.qualityFlag = QualityFlag::Degraded;
            result.message = "Reprojection error " + std::to_string(rms) +
                             "px exceeds threshold " + std::to_string(params_.maxReprojError) + "px, result degraded";
            spdlog::warn(std::string(kLogTag) + " WARN: " + result.message);
        }

        if (result.epipolarErrorMean > params_.maxEpipolarError) {
            result.qualityFlag = maxQuality(result.qualityFlag, QualityFlag::Degraded);
            std::string epipolarMsg = "Epipolar error " + std::to_string(result.epipolarErrorMean) +
                                      "px exceeds threshold " + std::to_string(params_.maxEpipolarError) + "px";
            if (!result.message.empty()) result.message += "; ";
            result.message += epipolarMsg;
            spdlog::warn(std::string(kLogTag) + " WARN: " + epipolarMsg);
        }

        spdlog::info(std::string(kLogTag) + " INFO: Stereo calibration completed. RMS: " +
                     std::to_string(rms) + "px, Epipolar error mean: " +
                     std::to_string(result.epipolarErrorMean) + "px");
    }
    catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("stereoCalibrate failed: ") + e.what();
        spdlog::error(std::string(kLogTag) + " ERROR: " + result.message);
    }
    catch (const std::bad_alloc&) {
        result.success = false;
        result.message = "memory allocation failed";
        spdlog::error(std::string(kLogTag) + " ERROR: memory allocation failed");
    }
    catch (const std::exception& e) {
        result.success = false;
        result.message = std::string(e.what());
        spdlog::error(std::string(kLogTag) + " ERROR: " + result.message);
    }
    catch (...) {
        result.success = false;
        result.message = "unknown error";
        spdlog::error(std::string(kLogTag) + " ERROR: unknown error");
    }

    return result;
}

ExtrinsicCalibCpuFullResult ExtrinsicCalibCpu::Impl::Execute() {
    ExtrinsicCalibCpuFullResult result;

    ExtrinsicCalibCpuMonoResult monoLeft, monoRight;

    if (params_.calibrateMono) {
        monoLeft = calibrateMonoSide(params_.leftPointsPerView, "Left");
        monoRight = calibrateMonoSide(params_.rightPointsPerView, "Right");
    }

    cv::Mat KL, DL, KR, DR;

    if (params_.calibrateMono) {
        if (monoLeft.success && !monoLeft.cameraMatrix.empty()) {
            KL = monoLeft.cameraMatrix;
            DL = monoLeft.distCoeffs;
        } else {
            KL = cv::Mat::eye(3, 3, CV_64F);
            KL.at<double>(0, 0) = params_.imageSize.width;
            KL.at<double>(1, 1) = params_.imageSize.width;
            KL.at<double>(0, 2) = params_.imageSize.width / 2.0;
            KL.at<double>(1, 2) = params_.imageSize.height / 2.0;
            DL = cv::Mat::zeros(1, 5, CV_64F);
        }
        if (monoRight.success && !monoRight.cameraMatrix.empty()) {
            KR = monoRight.cameraMatrix;
            DR = monoRight.distCoeffs;
        } else {
            KR = cv::Mat::eye(3, 3, CV_64F);
            KR.at<double>(0, 0) = params_.imageSize.width;
            KR.at<double>(1, 1) = params_.imageSize.width;
            KR.at<double>(0, 2) = params_.imageSize.width / 2.0;
            KR.at<double>(1, 2) = params_.imageSize.height / 2.0;
            DR = cv::Mat::zeros(1, 5, CV_64F);
        }
    }

    ExtrinsicCalibCpuResult stereoResult;
    if (params_.calibrateMono) {
        stereoResult = Execute(KL, DL, KR, DR);
    } else {
        stereoResult = Execute(
            cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 5, CV_64F),
            cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 5, CV_64F));
    }

    result.stereo = std::move(stereoResult);

    if (params_.calibrateMono) {
        result.monoLeft = std::move(monoLeft);
        result.monoRight = std::move(monoRight);
    }

    result.success = result.stereo.success;
    if (params_.calibrateMono) {
        result.success = result.success && result.monoLeft.success && result.monoRight.success;
    }

    result.qualityFlag = result.stereo.qualityFlag;
    if (params_.calibrateMono) {
        result.qualityFlag = maxQuality(result.qualityFlag, result.monoLeft.qualityFlag);
        result.qualityFlag = maxQuality(result.qualityFlag, result.monoRight.qualityFlag);
    }

    std::ostringstream oss;
    if (params_.calibrateMono) {
        if (!result.monoLeft.success || !result.monoLeft.message.empty()) {
            oss << "mono left: " << (result.monoLeft.success ? result.monoLeft.message : "FAILED(" + result.monoLeft.message + ")") << "; ";
        }
        if (!result.monoRight.success || !result.monoRight.message.empty()) {
            oss << "mono right: " << (result.monoRight.success ? result.monoRight.message : "FAILED(" + result.monoRight.message + ")") << "; ";
        }
    }
    if (!result.stereo.success || !result.stereo.message.empty()) {
        oss << "stereo: " << (result.stereo.success ? result.stereo.message : "FAILED(" + result.stereo.message + ")");
    }
    result.message = oss.str();
    if (result.message.empty() && result.success) {
        std::ostringstream summary;
        summary << params_.leftPointsPerView.size() << " views calibrated, stereo RMS="
                << result.stereo.stereoReprojError << "px, epipolar mean="
                << result.stereo.epipolarErrorMean << "px";
        result.message = summary.str();
    }

    return result;
}

// ═══════════════════════════════════════════════
// ExtrinsicCalibCpu public methods (pImpl delegation)
// ═══════════════════════════════════════════════

ExtrinsicCalibCpu::ExtrinsicCalibCpu(const ExtrinsicCalibCpuParams& params)
    : pImpl_(std::make_unique<Impl>(params)) {}

ExtrinsicCalibCpu::~ExtrinsicCalibCpu() = default;

void ExtrinsicCalibCpu::SetParams(const ExtrinsicCalibCpuParams& params) {
    pImpl_->SetParams(params);
}

const ExtrinsicCalibCpuParams& ExtrinsicCalibCpu::GetParams() const {
    return pImpl_->GetParams();
}

ExtrinsicCalibCpuResult ExtrinsicCalibCpu::Execute(
    const cv::Mat& cameraMatrixL, const cv::Mat& distCoeffsL,
    const cv::Mat& cameraMatrixR, const cv::Mat& distCoeffsR) {
    return pImpl_->Execute(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR);
}

ExtrinsicCalibCpuFullResult ExtrinsicCalibCpu::Execute() {
    return pImpl_->Execute();
}

void ExtrinsicCalibCpu::Destroy() {
}

} // namespace calib
