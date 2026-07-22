#pragma once

// =============================================================================
// HikvisionCameraSource —— 海康威视 MVS SDK 包装
//
// 仅当 CMake 找到 MvCameraControl.lib 时实现文件才会被编译进二进制。
// 找不到 SDK 时，工厂返回 nullptr（GUI 自动回退到 simulated）。
// =============================================================================

#include "ICameraSource.h"
#include <string>

namespace fc::gui {

class HikvisionCameraSource : public ICameraSource {
public:
    HikvisionCameraSource();
    ~HikvisionCameraSource() override;

    bool open(int deviceId = 0) override;
    void close() override;
    bool isOpen() const override;

    bool startAcquisition() override;
    void stopAcquisition() override;
    bool isAcquiring() const override;

    bool grabFrame(CameraFrame& out, int timeoutMs = 1000) override;

    void setExposureUs(double us) override;
    void setGainDb(double db) override;
    void setTriggerMode(bool on, int32_t src = 0) override;

    std::string displayName() const override;
    int width() const override;
    int height() const override;
    CameraCapability capability() const override;

    // 枚举所有可见设备（GigE + USB），返回 display 字符串
    static std::vector<std::string> enumerateDevices();

private:
    void* handle_ = nullptr;       // MV_CC_HANDLE
    int width_ = 0;
    int height_ = 0;
    bool opened_ = false;
    bool acquiring_ = false;
    std::string displayName_;
    int64_t frameIndex_ = 0;
};

}  // namespace fc::gui
