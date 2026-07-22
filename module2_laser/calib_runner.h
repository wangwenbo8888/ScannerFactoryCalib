#pragma once

// =============================================================================
// calib_runner.h —— laser_calib 流水线的可调用入口
//
// 注意：CalibCallbacks 与 module1_camera/calib_runner.h 等价但独立定义
// （两个模块编译期解耦；GUI 调用时构造一次即可传给两边）。
// =============================================================================

#include <functional>
#include <string>

namespace fc {

struct CalibCallbacks {
    std::function<void(int step, int totalSteps, const std::string& msg)> onProgress;
    std::function<void(const std::string& line)> onLog;
    std::function<bool()> shouldCancel;
};

// 模块2：激光虚拟相机 + 平面映射 + 温度补偿
//   inputDir  : 含 config.json + camera_calib.json + pose_*/L_tube*.png + R_tube*.png
//   outputPath: 输出 JSON 路径
// 返回：true = exit code 0；false = 部分失败（详见 spdlog）
bool runLaserCalib(const std::string& inputDir,
                   const std::string& outputPath,
                   const CalibCallbacks& cb = {});

}  // namespace fc
