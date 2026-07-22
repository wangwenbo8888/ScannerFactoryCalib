// =============================================================================
// laser_calib_cli.cpp 内部共用：抽出 main() 主体，供 CLI 与 runner 复用
// =============================================================================
#pragma once
#include <string>

namespace fc {
// 返回 0=完整成功；1=部分失败；2=参数错误（与原 CLI 退出码一致）
int runLaserCalibRaw(const std::string& inDir, const std::string& outPath);
}
