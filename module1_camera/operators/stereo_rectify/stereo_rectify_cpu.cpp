#include "stereo_rectify_cpu.h"
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

namespace calib {

OperatorInfo getStereoRectifyCpuInfo() {
    return OperatorInfo{"StereoRectifyCpu", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

// ═══════════════════════════════════════════════
// StereoRectifyCpuResult JSON serialization
// ═══════════════════════════════════════════════

nlohmann::json StereoRectifyCpuResult::toJson() const {
    nlohmann::json j;
    j["success"] = success;
    j["message"] = message;
    j["qualityFlag"] = static_cast<int>(qualityFlag);
    j["R1"] = calib::matToJson(R1);
    j["R2"] = calib::matToJson(R2);
    j["P1"] = calib::matToJson(P1);
    j["P2"] = calib::matToJson(P2);
    j["Q"] = calib::matToJson(Q);
    j["validRoiLeft"] = {{"x", validRoiLeft.x}, {"y", validRoiLeft.y},
                          {"w", validRoiLeft.width}, {"h", validRoiLeft.height}};
    j["validRoiRight"] = {{"x", validRoiRight.x}, {"y", validRoiRight.y},
                           {"w", validRoiRight.width}, {"h", validRoiRight.height}};
    return j;
}

StereoRectifyCpuResult StereoRectifyCpuResult::fromJson(const nlohmann::json& j) {
    StereoRectifyCpuResult r;
    r.success = j.at("success").get<bool>();
    r.message = j.at("message").get<std::string>();
    r.qualityFlag = static_cast<QualityFlag>(j.at("qualityFlag").get<int>());
    r.R1 = calib::jsonToMatAuto(j.at("R1"));
    r.R2 = calib::jsonToMatAuto(j.at("R2"));
    r.P1 = calib::jsonToMatAuto(j.at("P1"));
    r.P2 = calib::jsonToMatAuto(j.at("P2"));
    r.Q = calib::jsonToMatAuto(j.at("Q"));
    auto vl = j.at("validRoiLeft");
    r.validRoiLeft = cv::Rect(vl.at("x").get<int>(), vl.at("y").get<int>(),
                               vl.at("w").get<int>(), vl.at("h").get<int>());
    auto vr = j.at("validRoiRight");
    r.validRoiRight = cv::Rect(vr.at("x").get<int>(), vr.at("y").get<int>(),
                                vr.at("w").get<int>(), vr.at("h").get<int>());
    return r;
}

// ═══════════════════════════════════════════════
// StereoRectifyCpuParams implementation
// ═══════════════════════════════════════════════

nlohmann::json StereoRectifyCpuParams::toJson() const {
    nlohmann::json j;

    j["camera_matrix_l"] = matToJson(cameraMatrixL);
    j["dist_coeffs_l"] = matToJson(distCoeffsL);
    j["camera_matrix_r"] = matToJson(cameraMatrixR);
    j["dist_coeffs_r"] = matToJson(distCoeffsR);
    j["R"] = matToJson(R);

    nlohmann::json tArr = nlohmann::json::array();
    for (int i = 0; i < T.rows; ++i) {
        tArr.push_back(T.at<double>(i, 0));
    }
    j["T"] = tArr;

    j["image_width"] = imageSize.width;
    j["image_height"] = imageSize.height;
    j["alpha"] = alpha;
    j["flags"] = flags;

    return j;
}

StereoRectifyCpuParams StereoRectifyCpuParams::fromJson(const nlohmann::json& j) {
    StereoRectifyCpuParams p;
    const std::string ctx = "StereoRectifyCpuParams";

    p.cameraMatrixL = jsonToMat(
        getRequired<nlohmann::json>(j, "camera_matrix_l", ctx), 3, 3);
    p.distCoeffsL = jsonToMat(
        getRequired<nlohmann::json>(j, "dist_coeffs_l", ctx), 1, 5);
    p.cameraMatrixR = jsonToMat(
        getRequired<nlohmann::json>(j, "camera_matrix_r", ctx), 3, 3);
    p.distCoeffsR = jsonToMat(
        getRequired<nlohmann::json>(j, "dist_coeffs_r", ctx), 1, 5);
    p.R = jsonToMat(
        getRequired<nlohmann::json>(j, "R", ctx), 3, 3);

    auto tArr = getRequired<nlohmann::json>(j, "T", ctx);
    p.T = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        p.T.at<double>(i, 0) = tArr[i].get<double>();
    }

    p.imageSize = cv::Size(
        getRequired<int>(j, "image_width", ctx),
        getRequired<int>(j, "image_height", ctx)
    );
    p.alpha = j.contains("alpha") ? j.at("alpha").get<double>() : 0.0;
    p.flags = j.contains("flags") ? j.at("flags").get<int>() : 1;

    p.validate();
    return p;
}

void StereoRectifyCpuParams::validate() const {
    if (cameraMatrixL.empty() || cameraMatrixL.rows != 3 || cameraMatrixL.cols != 3) {
        throw std::invalid_argument("StereoRectifyCpuParams: cameraMatrixL must be 3x3");
    }
    if (cameraMatrixR.empty() || cameraMatrixR.rows != 3 || cameraMatrixR.cols != 3) {
        throw std::invalid_argument("StereoRectifyCpuParams: cameraMatrixR must be 3x3");
    }
    if (R.empty() || R.rows != 3 || R.cols != 3) {
        throw std::invalid_argument("StereoRectifyCpuParams: R must be 3x3");
    }
    if (T.empty() || T.rows != 3 || T.cols != 1) {
        throw std::invalid_argument("StereoRectifyCpuParams: T must be 3x1");
    }
    if (imageSize.width <= 0 || imageSize.height <= 0) {
        throw std::invalid_argument("StereoRectifyCpuParams: imageSize must have positive dimensions");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("StereoRectifyCpuParams: alpha must be in [0.0, 1.0]");
    }
    if (flags != 0 && flags != cv::CALIB_ZERO_DISPARITY) {
        throw std::invalid_argument("StereoRectifyCpuParams: flags must be 0 or CALIB_ZERO_DISPARITY");
    }
}

// ═══════════════════════════════════════════════
// StereoRectifyCpu::Impl
// ═══════════════════════════════════════════════

class StereoRectifyCpu::Impl {
public:
    explicit Impl(StereoRectifyCpuParams p) : params_(std::move(p)) { params_.validate(); }

    void SetParams(const StereoRectifyCpuParams& params) { params.validate(); params_ = params; }
    const StereoRectifyCpuParams& GetParams() const { return params_; }

    StereoRectifyCpuResult Execute() {
        StereoRectifyCpuResult result;

        try {
            cv::Mat R1, R2, P1, P2, Q;
            cv::Rect validRoiLeft, validRoiRight;

            cv::stereoRectify(
                params_.cameraMatrixL, params_.distCoeffsL,
                params_.cameraMatrixR, params_.distCoeffsR,
                params_.imageSize, params_.R, params_.T,
                R1, R2, P1, P2, Q,
                params_.flags, params_.alpha,
                cv::Size(), &validRoiLeft, &validRoiRight
            );

            result.R1 = R1.clone();
            result.R2 = R2.clone();
            result.P1 = P1.clone();
            result.P2 = P2.clone();
            result.Q = Q.clone();
            result.validRoiLeft = validRoiLeft;
            result.validRoiRight = validRoiRight;

            result.success = true;

            if (validRoiLeft.area() == 0 || validRoiRight.area() == 0) {
                result.qualityFlag = QualityFlag::Warning;
                result.message = "validRoi has zero area";
                spdlog::warn(std::string(kLogTag) + " WARN: " + result.message);
            }

            spdlog::info(std::string(kLogTag) + " INFO: stereoRectify completed. "
                         "validRoiLeft=" + std::to_string(validRoiLeft.x) + "," +
                         std::to_string(validRoiLeft.y) + "," +
                         std::to_string(validRoiLeft.width) + "," +
                         std::to_string(validRoiLeft.height) + " "
                         "validRoiRight=" + std::to_string(validRoiRight.x) + "," +
                         std::to_string(validRoiRight.y) + "," +
                         std::to_string(validRoiRight.width) + "," +
                         std::to_string(validRoiRight.height));
        }
        catch (const cv::Exception& e) {
            result.success = false;
            result.message = std::string("stereoRectify failed: ") + e.what();
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

private:
    StereoRectifyCpuParams params_;
};

// ═══════════════════════════════════════════════
// StereoRectifyCpu public interface
// ═══════════════════════════════════════════════

StereoRectifyCpu::StereoRectifyCpu(const StereoRectifyCpuParams& params)
    : pImpl_(std::make_unique<Impl>(params)) {}

StereoRectifyCpu::~StereoRectifyCpu() = default;

void StereoRectifyCpu::SetParams(const StereoRectifyCpuParams& params) {
    pImpl_->SetParams(params);
}

const StereoRectifyCpuParams& StereoRectifyCpu::GetParams() const {
    return pImpl_->GetParams();
}

StereoRectifyCpuResult StereoRectifyCpu::Execute() {
    return pImpl_->Execute();
}

void StereoRectifyCpu::Destroy() {
}

} // namespace calib
