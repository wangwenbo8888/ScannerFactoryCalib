// =============================================================================
// calib_runner.cpp —— 模块2 激光标定的 runner 包装
//
// 设计：laser_calib_cli.cpp 把主体抽成 runLaserCalibRaw()，本文件薄壳包装：
//   - 暴露统一的 CalibCallbacks API（与 module1 一致）
//   - spdlog 转发：GUI 注册全局 sink 即可捕获（不在 runner 内做转换）
//   - 取消：当前不实现细粒度取消；GUI 可强杀 worker 线程（粗粒度）
// =============================================================================

#include "calib_runner.h"          // CalibCallbacks 前向声明
#include "laser_calib_runner_internal.h"

#include <spdlog/spdlog.h>

namespace fc {

bool runLaserCalib(const std::string& inputDir,
                   const std::string& outputPath,
                   const CalibCallbacks& cb) {
    if (cb.onLog) cb.onLog(std::string("--- 加载激光输入: ") + inputDir);
    if (cb.onProgress) cb.onProgress(0, 1, "starting");

    int rc = runLaserCalibRaw(inputDir, outputPath);

    bool ok = (rc == 0);
    if (cb.onProgress) cb.onProgress(1, 1, ok ? "done" : "failed");
    if (cb.onLog) {
        cb.onLog(std::string("--- laser_calib ") + (ok ? "OK" : "PARTIAL")
                 + " -> " + outputPath);
    }
    return ok;
}

}  // namespace fc
