#pragma once

#include "ICameraSource.h"
#include <string>
#include <vector>

namespace fc::gui {

// 从一个文件夹按文件名顺序循环读取图像（png/bmp/jpg）。
// 用途：
//   1. 无相机硬件时验证 GUI 流程
//   2. 回放历史采集做参数调试
class SimulatedCameraSource : public ICameraSource {
public:
    explicit SimulatedCameraSource(const std::string& folder = std::string());
    ~SimulatedCameraSource() override;

    bool open(int deviceId = 0) override;
    void close() override;
    bool isOpen() const override;

    bool startAcquisition() override;
    void stopAcquisition() override;
    bool isAcquiring() const override;

    bool grabFrame(CameraFrame& out, int timeoutMs = 1000) override;

    std::string displayName() const override;
    int width() const override;
    int height() const override;
    CameraCapability capability() const override { return CameraCapability{}; }

    // 设置源文件夹并刷新文件列表
    void setFolder(const std::string& dir);
    const std::string& folder() const { return folder_; }

private:
    std::string folder_;
    std::vector<std::string> files_;
    size_t cursor_ = 0;
    bool opened_ = false;
    bool acquiring_ = false;
    int width_ = 0;
    int height_ = 0;
    int64_t frameIndex_ = 0;
};

}  // namespace fc::gui
