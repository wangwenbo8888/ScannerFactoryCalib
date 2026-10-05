#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
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
    double tempRangeMin = -15.0;
    double tempRangeMax = 15.0;
    double tempStep = 0.5;
    // 矫正
    double rectifyAlpha = 0.0;
    int rectifyFlags = cv::CALIB_ZERO_DISPARITY;
    // 温度系数（标定板热膨胀，喂 intrinsic_calib 的 temperature_coeff）
    double plateTempCoeff = 5.0e-6;
    double plateTemp = 21.0;

    static CameraCalibConfig fromJson(const std::string& path);
};

// ============================================================================
// CameraOpParams —— 算子层可调参数（2026-10-04 主库化；编译默认＝CLI 原硬编码）
// 主库 camera_calib_params.json 与数据集 config.json 均可逐叶覆盖
// （同 schema 同解析器 applyCameraParamsJson）。
// ============================================================================
struct CameraOpParams {
    double extrinsicMaxReprojFactor = 100.0;  // maxReproj = reproj_error_threshold × 此系数
    int    extrinsicMinViewCount    = 4;
    int    framesMinValidFrames     = 4;      // 角点有效帧下限
};

// 统一解析器：对一份 JSON 文档做 present-key 覆盖（主库与数据集 config 共用）
void applyCameraParamsJson(const nlohmann::json& j,
                           CameraCalibConfig& c, CameraOpParams& ops);

struct FramePair {
    cv::Mat leftGray;
    cv::Mat rightGray;
};

struct CameraInput {
    CameraCalibConfig config;
    CameraOpParams ops;
    std::vector<FramePair> frames;
};

// 读 data_in/camera/ 目录；baseCfg/baseOps＝上层（主库）合并结果，数据集键覆盖之
std::optional<CameraInput> loadCameraInput(const std::string& dir,
                                           const CameraCalibConfig* baseCfg = nullptr,
                                           const CameraOpParams* baseOps = nullptr);

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
