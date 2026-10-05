#include "stereo_rectify_temp_table_cpu.h"
#include "common/json_utils.h"
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <chrono>

using namespace calib;

OperatorInfo getStereoRectifyTempTableCpuInfo() {
    return OperatorInfo{"StereoRectifyTempTableCpu", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

static nlohmann::json matToJson(const cv::Mat& mat) {
    return calib::matToJson(mat);
}

static cv::Mat jsonToMat(const nlohmann::json& j, int rows, int cols) {
    return calib::jsonToMat(j, rows, cols);
}

static nlohmann::json rectToJson(const cv::Rect& r) {
    return {{"x", r.x}, {"y", r.y}, {"w", r.width}, {"h", r.height}};
}

static cv::Rect jsonToRect(const nlohmann::json& j) {
    return {j.at("x").get<int>(), j.at("y").get<int>(),
            j.at("w").get<int>(), j.at("h").get<int>()};
}

static cv::Mat compensateIntrinsics(const cv::Mat& K, double deltaT, double cte) {
    double scale = 1.0 + cte * deltaT;
    cv::Mat Kc = K.clone();
    Kc.at<double>(0, 0) = K.at<double>(0, 0) * scale;
    Kc.at<double>(1, 1) = K.at<double>(1, 1) * scale;
    Kc.at<double>(0, 2) = K.at<double>(0, 2) * scale;
    Kc.at<double>(1, 2) = K.at<double>(1, 2) * scale;
    return Kc;
}

static cv::Mat compensateT(const cv::Mat& T, double deltaT, double cte) {
    double scale = 1.0 + cte * deltaT;
    return T * scale;
}

void StereoRectifyTempTableParams::validate() const {
    const std::string ctx = "StereoRectifyTempTableParams";
    if (cameraMatrixL.empty() || cameraMatrixL.rows != 3 || cameraMatrixL.cols != 3)
        throw std::invalid_argument(ctx + ": cameraMatrixL must be 3x3");
    if (cameraMatrixR.empty() || cameraMatrixR.rows != 3 || cameraMatrixR.cols != 3)
        throw std::invalid_argument(ctx + ": cameraMatrixR must be 3x3");
    if (R.empty() || R.rows != 3 || R.cols != 3)
        throw std::invalid_argument(ctx + ": R must be 3x3");
    if (T.empty() || T.rows != 3 || T.cols != 1)
        throw std::invalid_argument(ctx + ": T must be 3x1");
    if (imageSize.width <= 0 || imageSize.height <= 0)
        throw std::invalid_argument(ctx + ": imageSize must have positive dimensions");
    if (cte <= 0.0)
        throw std::invalid_argument(ctx + ": cte must be > 0");
    if (tempStep <= 0.0)
        throw std::invalid_argument(ctx + ": tempStep must be > 0");
    if (tempRangeMin > tempRangeMax)
        throw std::invalid_argument(ctx + ": tempRangeMin must be <= tempRangeMax");
    if (alpha < 0.0 || alpha > 1.0)
        throw std::invalid_argument(ctx + ": alpha must be in [0.0, 1.0]");
    if (flags != 0 && flags != cv::CALIB_ZERO_DISPARITY)
        throw std::invalid_argument(ctx + ": flags must be 0 or CALIB_ZERO_DISPARITY");
}

nlohmann::json StereoRectifyTempTableParams::toJson() const {
    nlohmann::json j;
    j["cameraMatrixL"] = matToJson(cameraMatrixL);
    j["distCoeffsL"] = matToJson(distCoeffsL);
    j["cameraMatrixR"] = matToJson(cameraMatrixR);
    j["distCoeffsR"] = matToJson(distCoeffsR);
    j["R"] = matToJson(R);
    j["T"] = matToJson(T);
    j["imageWidth"] = imageSize.width;
    j["imageHeight"] = imageSize.height;
    j["referenceTemp"] = referenceTemp;
    j["cte"] = cte;
    j["tempStep"] = tempStep;
    j["tempRangeMin"] = tempRangeMin;
    j["tempRangeMax"] = tempRangeMax;
    j["alpha"] = alpha;
    j["flags"] = flags;
    return j;
}

StereoRectifyTempTableParams StereoRectifyTempTableParams::fromJson(const nlohmann::json& j) {
    StereoRectifyTempTableParams p;
    const std::string ctx = "StereoRectifyTempTableParams";

    p.cameraMatrixL = jsonToMat(calib::getRequired<nlohmann::json>(j, "cameraMatrixL", ctx), 3, 3);
    p.distCoeffsL = jsonToMat(calib::getRequired<nlohmann::json>(j, "distCoeffsL", ctx), 1, 5);
    p.cameraMatrixR = jsonToMat(calib::getRequired<nlohmann::json>(j, "cameraMatrixR", ctx), 3, 3);
    p.distCoeffsR = jsonToMat(calib::getRequired<nlohmann::json>(j, "distCoeffsR", ctx), 1, 5);
    p.R = jsonToMat(calib::getRequired<nlohmann::json>(j, "R", ctx), 3, 3);

    auto tArr = calib::getRequired<nlohmann::json>(j, "T", ctx);
    p.T = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i)
        p.T.at<double>(i, 0) = tArr[i].get<double>();

    p.imageSize = cv::Size(
        calib::getRequired<int>(j, "imageWidth", ctx),
        calib::getRequired<int>(j, "imageHeight", ctx));

    p.referenceTemp = j.contains("referenceTemp") ? j.at("referenceTemp").get<double>() : 25.0;
    p.cte = j.contains("cte") ? j.at("cte").get<double>() : 23.6e-6;
    p.tempStep = j.contains("tempStep") ? j.at("tempStep").get<double>() : 0.2;
    p.tempRangeMin = j.contains("tempRangeMin") ? j.at("tempRangeMin").get<double>() : -10.0;
    p.tempRangeMax = j.contains("tempRangeMax") ? j.at("tempRangeMax").get<double>() : 10.0;
    p.alpha = j.contains("alpha") ? j.at("alpha").get<double>() : 0.0;
    p.flags = j.contains("flags") ? j.at("flags").get<int>() : 1;

    p.validate();
    return p;
}

nlohmann::json StereoRectifyTempEntry::toJson() const {
    nlohmann::json j;
    j["temperature"] = temperature;
    j["deltaT"] = deltaT;
    j["compensatedCameraMatrixL"] = matToJson(compensatedCameraMatrixL);
    j["compensatedCameraMatrixR"] = matToJson(compensatedCameraMatrixR);
    j["compensatedT"] = matToJson(compensatedT);
    j["R1"] = matToJson(R1);
    j["R2"] = matToJson(R2);
    j["P1"] = matToJson(P1);
    j["P2"] = matToJson(P2);
    j["Q"] = matToJson(Q);
    j["validRoiLeft"] = rectToJson(validRoiLeft);
    j["validRoiRight"] = rectToJson(validRoiRight);
    return j;
}

nlohmann::json StereoRectifyTempTableResult::toJson() const {
    nlohmann::json j;
    j["success"] = success;
    j["message"] = message;
    j["qualityFlag"] = static_cast<int>(qualityFlag);
    j["referenceTemp"] = referenceTemp;
    j["cte"] = cte;
    j["tableSize"] = static_cast<int>(table.size());

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& entry : table)
        arr.push_back(entry.toJson());
    j["table"] = arr;
    return j;
}

StereoRectifyTempTableCpu::StereoRectifyTempTableCpu(const StereoRectifyTempTableParams& params)
    : params_(params) {
    params_.validate();
}

StereoRectifyTempTableCpu::~StereoRectifyTempTableCpu() = default;

void StereoRectifyTempTableCpu::SetParams(const StereoRectifyTempTableParams& params) {
    params.validate();
    params_ = params;
}

const StereoRectifyTempTableParams& StereoRectifyTempTableCpu::GetParams() const {
    return params_;
}

StereoRectifyTempTableResult StereoRectifyTempTableCpu::Execute() {
    using Clock = std::chrono::high_resolution_clock;

    auto t0_total = Clock::now();

    StereoRectifyTempTableResult result;
    result.referenceTemp = params_.referenceTemp;
    result.cte = params_.cte;

    try {
        int failCount = 0;
        int warnCount = 0;

        double tMin = params_.referenceTemp + params_.tempRangeMin;
        double tMax = params_.referenceTemp + params_.tempRangeMax;

        int stepIdx = 0;
        double total_compensate_us = 0.0;
        double total_rectify_us = 0.0;

        for (double temp = tMin; temp <= tMax + 1e-9; temp += params_.tempStep) {
            double deltaT = temp - params_.referenceTemp;

            auto t1 = Clock::now();
            cv::Mat KcL = compensateIntrinsics(params_.cameraMatrixL, deltaT, params_.cte);
            cv::Mat KcR = compensateIntrinsics(params_.cameraMatrixR, deltaT, params_.cte);
            cv::Mat Tc = compensateT(params_.T, deltaT, params_.cte);
            auto t2 = Clock::now();
            double dt_compensate = std::chrono::duration<double, std::micro>(t2 - t1).count();
            total_compensate_us += dt_compensate;

            StereoRectifyTempEntry entry;
            entry.temperature = temp;
            entry.deltaT = deltaT;
            entry.compensatedCameraMatrixL = KcL;
            entry.compensatedCameraMatrixR = KcR;
            entry.compensatedT = Tc;

            try {
                auto t3 = Clock::now();
                cv::stereoRectify(
                    KcL, params_.distCoeffsL,
                    KcR, params_.distCoeffsR,
                    params_.imageSize, params_.R, Tc,
                    entry.R1, entry.R2, entry.P1, entry.P2, entry.Q,
                    params_.flags, params_.alpha,
                    cv::Size(), &entry.validRoiLeft, &entry.validRoiRight);
                auto t4 = Clock::now();
                double dt_rectify = std::chrono::duration<double, std::micro>(t4 - t3).count();
                total_rectify_us += dt_rectify;

                spdlog::info("{} STEP[{:>3d}] temp={:>7.2f} deltaT={:>7.2f} "
                             "compensate={:>8.1f}us stereoRectify={:>8.1f}us",
                             kLogTag, stepIdx, temp, deltaT, dt_compensate, dt_rectify);

                if (entry.validRoiLeft.area() == 0 || entry.validRoiRight.area() == 0) {
                    warnCount++;
                    spdlog::warn("{} WARN: temp={:.1f} validRoi zero area",
                                 kLogTag, temp);
                }
            }
            catch (const cv::Exception& e) {
                failCount++;
                spdlog::warn("{} STEP[{:>3d}] temp={:>7.2f} stereoRectify FAILED: {}",
                             kLogTag, stepIdx, temp, e.what());
                stepIdx++;
                continue;
            }

            result.table.push_back(std::move(entry));
            stepIdx++;
        }

        auto t5_total = Clock::now();
        double dt_total_ms = std::chrono::duration<double, std::milli>(t5_total - t0_total).count();

        result.success = true;
        result.tableSize = static_cast<int>(result.table.size());

        if (failCount > 0) {
            result.qualityFlag = calib::QualityFlag::Degraded;
            result.message = std::to_string(failCount) + " temperature points failed stereoRectify";
        }
        if (warnCount > 0) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = (result.message.empty() ? "" : result.message + "; ")
                             + std::to_string(warnCount) + " points have zero validRoi";
        }

        spdlog::info("{} ====== TIMING SUMMARY ======", kLogTag);
        spdlog::info("{} Total entries: {}", kLogTag, result.table.size());
        spdlog::info("{} Total compensate:    {:>10.1f} us  (avg {:.1f} us/step)",
                     kLogTag, total_compensate_us,
                     total_compensate_us / std::max(1.0, (double)result.table.size()));
        spdlog::info("{} Total stereoRectify: {:>10.1f} us  (avg {:.1f} us/step)",
                     kLogTag, total_rectify_us,
                     total_rectify_us / std::max(1.0, (double)result.table.size()));
        spdlog::info("{} Total compute:       {:>10.2f} ms", kLogTag, dt_total_ms);
        spdlog::info("{} Failures: {}  Warnings: {}", kLogTag, failCount, warnCount);
    }
    catch (const std::exception& e) {
        result.success = false;
        result.message = e.what();
        spdlog::error("{} ERROR: {}", kLogTag, e.what());
    }

    return result;
}

void StereoRectifyTempTableCpu::Destroy() {
}
