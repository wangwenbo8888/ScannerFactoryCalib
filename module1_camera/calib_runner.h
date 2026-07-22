#pragma once

// =============================================================================
// calib_runner.h —— 相机标定分步 + 全流程 API
//
// 把 camera_calib_cli.cpp 的 6 步主体抽成可分别调用的函数，便于 GUI 按钮触发。
// 同时提供 runCameraCalib() 一键全流程（CLI 复用）。
//
// 线程：每个函数同步执行；GUI 应放进 QtConcurrent::run 工作线程。
// 进度：通过 CalibCallbacks 暴露给调用方。
// =============================================================================

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <functional>

#include "calib_io.h"
#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

namespace fc {

// 进度/日志/取消回调（GUI 注入；CLI 默认空）
struct CalibCallbacks {
    // step ∈ [0, totalSteps]；msg 简短描述当前在做什么
    std::function<void(int step, int totalSteps, const std::string& msg)> onProgress;
    // 一行已格式化日志（spdlog 风格；GUI 直接 appendPlainText）
    std::function<void(const std::string& line)> onLog;
    // GUI 取消按钮；返回 true 时业务尽早退出
    std::function<bool()> shouldCancel;
};

// ---- Step 1: 棋盘格角点提取 ----
struct CornerExtractionResult {
    std::vector<std::vector<cv::Point2f>> leftPoints;   // [view][corner]
    std::vector<std::vector<cv::Point2f>> rightPoints;
    int totalFrames = 0;
    int validFrames = 0;
    bool success = false;
    std::string message;
};
CornerExtractionResult extractCorners(const CameraInput& input,
                                       const CalibCallbacks& cb = {});

// ---- Step 2: 内参标定（L/R 各一）----
calib::IntrinsicCalibResult calibrateIntrinsic(
    const CameraCalibConfig& cfg,
    const CornerExtractionResult& corners,
    const CalibCallbacks& cb = {});

// ---- Step 3: 外参（立体 R/T + E/F）----
calib::ExtrinsicCalibCpuResult calibrateExtrinsic(
    const CameraCalibConfig& cfg,
    const CornerExtractionResult& corners,
    const calib::IntrinsicCalibResult& intrin,
    const CalibCallbacks& cb = {});

// ---- Step 4: 立体矫正（R1/R2/P1/P2/Q + validRoiL/R）----
calib::StereoRectifyCpuResult stereoRectify(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const CalibCallbacks& cb = {});

// ---- Step 5: 四张温度表 ----
struct TempTableSet {
    calib::IntrinsicCompensateCPUResult intrinL;
    calib::IntrinsicCompensateCPUResult intrinR;
    calib::ExtrinsicCompensateCPUResult extrin;
    calib::StereoRectifyTempTableResult rectify;
    bool success = false;
};
TempTableSet buildTempTables(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const CalibCallbacks& cb = {});

// ---- 全流程一键：等价于 camera_calib.exe ----
//   返回 true 表示 6 步全成功 + JSON 已写盘
bool runCameraCalib(const std::string& inputDir,
                    const std::string& outputPath,
                    const CalibCallbacks& cb = {});

}  // namespace fc
