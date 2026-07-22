#pragma once

#include <QString>
#include <QStringList>

namespace fc::gui {

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
