#include "plane_map_temp_table.h"
#include "plane_map_cuda.h"
#include "virtual_pixel_gen.h"
#include "common/calib_logging.h"
#include "common/json_utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <chrono>

using namespace calib;

OperatorInfo getPlaneMapTempTableInfo() {
    return OperatorInfo{"PlaneMapTempTable", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(13, PlaneMapTempTable);

static nlohmann::json matToJson(const cv::Mat& mat) {
    return calib::matToJson(mat);
}

static cv::Mat jsonToMat(const nlohmann::json& j, int rows, int cols) {
    return calib::jsonToMat(j, rows, cols);
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

static cv::Vec3d compensateVirtualT(const cv::Vec3d& T, double deltaT, double cte) {
    double scale = 1.0 + cte * deltaT;
    return T * scale;
}

void PlaneMapTempTableParams::validate() const {
    const std::string ctx = "PlaneMapTempTableParams";
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
    if (lineIds.empty())
        throw std::invalid_argument(ctx + ": lineIds must not be empty");
}

nlohmann::json PlaneMapTempTableParams::toJson() const {
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
    j["deviceId"] = deviceId;
    j["gridStep"] = gridStep;
    j["depthMin"] = depthMin;
    j["depthMax"] = depthMax;
    j["depthSamples"] = depthSamples;
    j["epipolarStep"] = epipolarStep;
    nlohmann::json vkArr = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < 3; ++j)
            row.push_back(virtualK(i, j));
        vkArr.push_back(row);
    }
    j["virtualK"] = vkArr;
    nlohmann::json vrArr = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < 3; ++j)
            row.push_back(virtualR(i, j));
        vrArr.push_back(row);
    }
    j["virtualR"] = vrArr;
    auto tArr = nlohmann::json::array();
    tArr.push_back(virtualT(0));
    tArr.push_back(virtualT(1));
    tArr.push_back(virtualT(2));
    j["virtualT"] = tArr;
    auto lidArr = nlohmann::json::array();
    for (int id : lineIds) lidArr.push_back(id);
    j["lineIds"] = lidArr;
    return j;
}

PlaneMapTempTableParams PlaneMapTempTableParams::fromJson(const nlohmann::json& j) {
    PlaneMapTempTableParams p;
    const std::string ctx = "PlaneMapTempTableParams";

    p.cameraMatrixL = jsonToMat(calib::getRequired<nlohmann::json>(j, "cameraMatrixL", ctx), 3, 3);
    p.distCoeffsL = jsonToMat(calib::getRequired<nlohmann::json>(j, "distCoeffsL", ctx), 1, 5);
    p.cameraMatrixR = jsonToMat(calib::getRequired<nlohmann::json>(j, "cameraMatrixR", ctx), 3, 3);
    p.distCoeffsR = jsonToMat(calib::getRequired<nlohmann::json>(j, "distCoeffsR", ctx), 1, 5);
    p.R = jsonToMat(calib::getRequired<nlohmann::json>(j, "R", ctx), 3, 3);

    auto tArr = calib::getRequired<nlohmann::json>(j, "T", ctx);
    p.T = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        if (tArr[i].is_array())
            p.T.at<double>(i, 0) = tArr[i][0].get<double>();
        else
            p.T.at<double>(i, 0) = tArr[i].get<double>();
    }

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
    p.deviceId = j.contains("deviceId") ? j.at("deviceId").get<int>() : 0;
    p.gridStep = j.contains("gridStep") ? j.at("gridStep").get<float>() : 0.5f;
    p.depthMin = j.contains("depthMin") ? j.at("depthMin").get<float>() : 100.0f;
    p.depthMax = j.contains("depthMax") ? j.at("depthMax").get<float>() : 5000.0f;
    p.depthSamples = j.contains("depthSamples") ? j.at("depthSamples").get<int>() : 200;
    p.epipolarStep = j.contains("epipolarStep") ? j.at("epipolarStep").get<float>() : 0.5f;

    if (j.contains("virtualK"))
        p.virtualK = jsonToMat(j.at("virtualK"), 3, 3);
    if (j.contains("virtualR"))
        p.virtualR = jsonToMat(j.at("virtualR"), 3, 3);
    if (j.contains("virtualT") && j["virtualT"].is_array()) {
        for (int i = 0; i < 3 && i < static_cast<int>(j["virtualT"].size()); ++i)
            p.virtualT(i) = j["virtualT"][i].get<double>();
    }
    if (j.contains("lineIds") && j["lineIds"].is_array()) {
        for (const auto& id : j["lineIds"])
            p.lineIds.push_back(id.get<int>());
    }

    p.validate();
    return p;
}

nlohmann::json PlaneMapTempEntry::toJson() const {
    nlohmann::json j;
    j["temperature"] = temperature;
    j["deltaT"] = deltaT;
    j["totalPairs"] = totalPairs;
    j["compensatedCameraMatrixL"] = matToJson(compensatedCameraMatrixL);
    j["compensatedCameraMatrixR"] = matToJson(compensatedCameraMatrixR);
    j["compensatedT"] = matToJson(compensatedT);
    j["R1"] = matToJson(R1);
    j["R2"] = matToJson(R2);
    j["P1"] = matToJson(P1);
    j["P2"] = matToJson(P2);
    j["Q"] = matToJson(Q);
    nlohmann::json statsArr = nlohmann::json::array();
    for (const auto& st : lineStats) {
        statsArr.push_back({{"lineId", st.lineId}, {"numPairs", st.numPairs},
                            {"uMin", st.uMin}, {"uMax", st.uMax},
                            {"vMin", st.vMin}, {"vMax", st.vMax}});
    }
    j["lineStats"] = statsArr;
    return j;
}

nlohmann::json PlaneMapTempTableResult::toJson() const {
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

PlaneMapTempTable::PlaneMapTempTable(const PlaneMapTempTableParams& params)
    : params_(params) {
    params_.validate();
}

PlaneMapTempTable::~PlaneMapTempTable() = default;

void PlaneMapTempTable::SetParams(const PlaneMapTempTableParams& params) {
    params.validate();
    params_ = params;
}

const PlaneMapTempTableParams& PlaneMapTempTable::GetParams() const {
    return params_;
}

PlaneMapTempTableResult PlaneMapTempTable::Execute() {
#if !BUILD_CUDA
    PlaneMapTempTableResult result;
    result.success = false;
    result.message = "CUDA not available (BUILD_CUDA=OFF)";
    return result;
#else
    using Clock = std::chrono::high_resolution_clock;
    auto t0_total = Clock::now();

    PlaneMapTempTableResult result;
    result.referenceTemp = params_.referenceTemp;
    result.cte = params_.cte;

    try {
        PlaneMapParams pmParams;
        pmParams.deviceId = params_.deviceId;
        pmParams.gridStep = params_.gridStep;
        pmParams.depthMin = params_.depthMin;
        pmParams.depthMax = params_.depthMax;
        pmParams.depthSamples = params_.depthSamples;
        pmParams.epipolarStep = params_.epipolarStep;

        PlaneMapCuda planeMap(pmParams);

        VirtualPixelGenParams vpgParams;
        vpgParams.deviceId = params_.deviceId;
        vpgParams.gridStep = params_.gridStep;
        VirtualPixelGenerator pixelGen(vpgParams);

        auto vpgResult = pixelGen.Execute(
            params_.virtualK, params_.imageSize, params_.lineIds);

        if (!vpgResult.success) {
            result.success = false;
            result.message = "Virtual pixel generation failed: " + vpgResult.message;
            spdlog::error("{} Virtual pixel generation FAILED: {}", kLogTag, vpgResult.message);
            return result;
        }

        cv::cuda::GpuMat d_virtual_pixels = std::move(vpgResult.d_virtualPixels);

        spdlog::info("{} Virtual pixels generated: {}x{}", kLogTag,
                     d_virtual_pixels.cols, d_virtual_pixels.rows);

        double tMin = params_.referenceTemp + params_.tempRangeMin;
        double tMax = params_.referenceTemp + params_.tempRangeMax;

        int failCount = 0;
        int warnCount = 0;
        int stepIdx = 0;
        double total_compensate_us = 0.0;
        double total_rectify_us = 0.0;
        double total_plane_map_us = 0.0;

        for (double temp = tMin; temp <= tMax + 1e-9; temp += params_.tempStep) {
            double deltaT = temp - params_.referenceTemp;

            auto t1 = Clock::now();
            cv::Mat KcL = compensateIntrinsics(params_.cameraMatrixL, deltaT, params_.cte);
            cv::Mat KcR = compensateIntrinsics(params_.cameraMatrixR, deltaT, params_.cte);
            cv::Mat Tc = compensateT(params_.T, deltaT, params_.cte);
            cv::Vec3d virtualTc = compensateVirtualT(params_.virtualT, deltaT, params_.cte);
            auto t2 = Clock::now();
            total_compensate_us += std::chrono::duration<double, std::micro>(t2 - t1).count();

            PlaneMapTempEntry entry;
            entry.temperature = temp;
            entry.deltaT = deltaT;
            entry.compensatedCameraMatrixL = KcL;
            entry.compensatedCameraMatrixR = KcR;
            entry.compensatedT = Tc;

            auto t3 = Clock::now();
            try {
                cv::stereoRectify(
                    KcL, params_.distCoeffsL,
                    KcR, params_.distCoeffsR,
                    params_.imageSize, params_.R, Tc,
                    entry.R1, entry.R2, entry.P1, entry.P2, entry.Q,
                    params_.flags, params_.alpha,
                    cv::Size(), nullptr, nullptr);
            } catch (const cv::Exception& e) {
                failCount++;
                spdlog::warn("{} STEP[{:>3d}] temp={:>7.2f} stereoRectify FAILED: {}",
                             kLogTag, stepIdx, temp, e.what());
                stepIdx++;
                continue;
            }
            auto t4 = Clock::now();
            total_rectify_us += std::chrono::duration<double, std::micro>(t4 - t3).count();

            calib::StereoCalibration calib;
            calib.R1 = entry.R1;
            calib.R2 = entry.R2;
            calib.P1 = entry.P1;
            calib.P2 = entry.P2;
            calib.imageSize = params_.imageSize;

            auto t5 = Clock::now();
            PlaneMapResult pmResult = planeMap.Execute(
                d_virtual_pixels,
                params_.virtualK,
                params_.virtualR,
                virtualTc,
                calib);
            auto t6 = Clock::now();
            total_plane_map_us += std::chrono::duration<double, std::micro>(t6 - t5).count();

            if (!pmResult.success) {
                failCount++;
                spdlog::warn("{} STEP[{:>3d}] temp={:>7.2f} planeMap FAILED: {}",
                             kLogTag, stepIdx, temp, pmResult.message);
                stepIdx++;
                continue;
            }

            entry.d_left_to_right = pmResult.d_left_to_right;
            entry.d_right_u = pmResult.d_right_u;
            entry.totalPairs = pmResult.totalPairs;
            entry.lineStats = std::move(pmResult.lineStats);

            spdlog::info("{} STEP[{:>3d}] temp={:>7.2f} deltaT={:>7.2f} "
                         "compensate={:>8.1f}us stereoRectify={:>8.1f}us planeMap={:>8.1f}us pairs={}",
                         kLogTag, stepIdx, temp, deltaT,
                         std::chrono::duration<double, std::micro>(t2 - t1).count(),
                         std::chrono::duration<double, std::micro>(t4 - t3).count(),
                         std::chrono::duration<double, std::micro>(t6 - t5).count(),
                         entry.totalPairs);

            if (pmResult.qualityFlag != calib::QualityFlag::Normal)
                warnCount++;

            result.table.push_back(std::move(entry));
            stepIdx++;
        }

        auto t_end = Clock::now();
        double dt_total_ms = std::chrono::duration<double, std::milli>(t_end - t0_total).count();

        result.success = true;
        result.tableSize = static_cast<int>(result.table.size());

        if (failCount > 0) {
            result.qualityFlag = calib::QualityFlag::Degraded;
            result.message = std::to_string(failCount) + " temperature points failed";
        }
        if (warnCount > 0 && result.qualityFlag == calib::QualityFlag::Normal) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = std::to_string(warnCount) + " points with non-Normal quality";
        }

        int totalSteps = std::max(1, static_cast<int>(result.table.size()));
        spdlog::info("{} ====== TIMING SUMMARY ======", kLogTag);
        spdlog::info("{} Total entries: {}", kLogTag, result.table.size());
        spdlog::info("{} Total compensate:    {:>10.1f} us  (avg {:.1f} us/step)",
                     kLogTag, total_compensate_us, total_compensate_us / totalSteps);
        spdlog::info("{} Total stereoRectify: {:>10.1f} us  (avg {:.1f} us/step)",
                     kLogTag, total_rectify_us, total_rectify_us / totalSteps);
        spdlog::info("{} Total planeMap:      {:>10.1f} us  (avg {:.1f} us/step)",
                     kLogTag, total_plane_map_us, total_plane_map_us / totalSteps);
        spdlog::info("{} Total compute:       {:>10.2f} ms", kLogTag, dt_total_ms);
        spdlog::info("{} Failures: {}  Warnings: {}", kLogTag, failCount, warnCount);

    } catch (const std::exception& e) {
        result.success = false;
        result.message = e.what();
        spdlog::error("{} ERROR: {}", kLogTag, e.what());
    }

    return result;
#endif
}

void PlaneMapTempTable::Destroy() {
}
