#pragma once

#include <QString>
#include <QStringList>

namespace fc::gui {

// 五种激光线类型（采集模式与标定类型共用同一张表；261105 固件五管语义：
// T=左斜 V=右斜 C=深孔 D=精细 E=备用中竖25线——下位机 N10 按管开关分线型）
struct LaserTypeSpec {
    const char* key;    // 子目录名 / 输出文件名后缀
    const char* name;   // 中文名（UI 显示）
};
inline constexpr LaserTypeSpec kLaserTypes[] = {
    {"left_skew",    "左斜"},
    {"right_skew",   "右斜"},
    {"fine",         "精细"},
    {"deep_hole",    "深孔"},
    {"mid_vertical", "中竖"},   // 261105 增：备用中竖25线（E 管）
};
inline constexpr int kLaserTypeCount = 5;
inline QString laserTypeDir(int i) {   // data_in/laser/<key>
    return QStringLiteral("data_in/laser/") + QString::fromLatin1(kLaserTypes[i].key);
}
inline QString laserTypeOutPath(int i) {  // data_out/laser_calib_<key>.json
    return QStringLiteral("data_out/laser_calib_") + QString::fromLatin1(kLaserTypes[i].key) + ".json";
}
inline QString laserTypeName(int i) { return QString::fromUtf8(kLaserTypes[i].name); }

// 编译期注入的 CLI 路径（来自顶层 CMakeLists 的 target_compile_definitions）
#ifdef FC_CAMERA_CALIB_EXE
inline constexpr const char* kCameraCalibExe = FC_CAMERA_CALIB_EXE;
#else
inline constexpr const char* kCameraCalibExe = "camera_calib.exe";
#endif

#ifdef FC_LASER_CALIB_EXE
inline constexpr const char* kLaserCalibExe = FC_LASER_CALIB_EXE;
#else
inline constexpr const char* kLaserCalibExe = "laser_calib.exe";
#endif

// 把 stdout/stderr 的字节流按 UTF-8 解码并按行切分（兼容 CRLF/LF）
QStringList bytesToLines(const QByteArray& bytes);

// 在 Windows 下为 Qt5 重定向 PATH（让子进程找到 opencv_world4130.dll 等）。
// 简单做法：把 exe 同目录加入子进程 PATH。
QString buildChildPathEnv(const QString& exePath);

}  // namespace fc::gui
