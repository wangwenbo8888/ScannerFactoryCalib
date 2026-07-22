#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

namespace fc {

struct CameraCalibConfig {
    // 棋盘格
    int chessWidth = 11;
    int chessHeight = 8;
    double squareSizeMm = 2.0;
    // 图像
    int imageWidth = 2048;
    int imageHeight = 1536;
    // 内参
    int intrinsicFlags = 0;
    bool useCalibrateCameraRO = true;
    double reprojErrorThreshold = 0.012;
    // 温度
    double cte = 23.6e-6;
    double referenceTemp = 22.5;
    double tempRangeMin = -10.0;
    double tempRangeMax = 10.0;
    double tempStep = 0.2;
    // 矫正
    double rectifyAlpha = 0.0;
    int rectifyFlags = 1;
    // 温度系数（标定板热膨胀，喂 intrinsic_calib 的 temperature_coeff）
    double plateTempCoeff = 5.0e-6;
    double plateTemp = 21.0;

    static CameraCalibConfig fromJson(const std::string& path);
};

struct FramePair {
    cv::Mat leftGray;
    cv::Mat rightGray;
};

struct CameraInput {
    CameraCalibConfig config;
    std::vector<FramePair> frames;
};

// 读 data_in/camera/ 目录
std::optional<CameraInput> loadCameraInput(const std::string& dir);

// 把模块1 全部结果与配置汇总成单个 json（复用各算子的 toJson()）
nlohmann::json buildCameraCalibJson(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const calib::StereoRectifyCpuResult& rectify,
    const calib::IntrinsicCompensateCPUResult& intrinTableL,
    const calib::IntrinsicCompensateCPUResult& intrinTableR,
    const calib::ExtrinsicCompensateCPUResult& extrinTable,
    const calib::StereoRectifyTempTableResult& rectifyTable);

bool writeJson(const std::string& path, const nlohmann::json& j);

} // namespace fc
