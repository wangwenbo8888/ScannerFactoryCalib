#pragma once

#include "ICameraSource.h"
#include <memory>
#include <string>

namespace fc::gui {

// 双相机 rig：包装两个 ICameraSource（左/右），暴露统一接口。
// 用法：
//   rig.configure("hikvision", leftId=0, rightId=1);
//   rig.open();
//   rig.startAcquisition();
//   while (...) { if (rig.grabStereo(L, R, 1000)) ... }
//
// 同步策略：连续采集模式下不严格同步（先后抓）；触发模式下应在外部触发
// 同时点亮两台相机，理论上同步。本类仅做轮询包装，不做时间戳匹配。
class StereoCameraRig {
public:
    StereoCameraRig() = default;
    ~StereoCameraRig();

    // 指定左右源类型 + 设备号；folder 仅 simulated 用
    bool configure(const std::string& sourceType,
                   int leftDeviceId, int rightDeviceId,
                   const std::string& leftFolder = std::string(),
                   const std::string& rightFolder = std::string());

    bool open();
    void close();
    bool isOpen() const;

    bool startAcquisition();
    void stopAcquisition();
    bool isAcquiring() const;

    // 同时抓 L + R；任一失败返回 false（已抓到的仍写入 out）
    bool grabStereo(CameraFrame& leftOut, CameraFrame& rightOut, int timeoutMs = 1000);

    // 仿 LeadScanK2/CameraControl.cpp::GetScannerImages 的"开采→触发→抓帧→停采"流程。
    // 用于 galaxy 等支持触发的源；默认软触发 (Software)。
    // 步骤：
    //   1. setExposureUs(exposureUs)         应用曝光
    //   2. setTriggerMode(true, Software)    切软触发
    //   3. startAcquisition                   开采
    //   4. triggerOnce() L+R                  同时发软触发，各出一帧
    //   5. grabStereo(L, R, timeoutMs)        等帧回读
    //   6. stopAcquisition                    停采
    // 右相机 180° 旋转由 GalaxyCameraSource::rightRotate180_ 默认 true 处理。
    // exposureUs<=0 → 不改曝光，沿用打开时的值。
    bool captureScannerStereo(CameraFrame& leftOut, CameraFrame& rightOut,
                              double exposureUs = -1.0, int timeoutMs = 3000);

    ICameraSource* left()  { return left_.get(); }
    ICameraSource* right() { return right_.get(); }

    // 注册左右帧到达回调（事件驱动，透传给底层源）。
    // 应在 open() 之后、startAcquisition() 之前调用。
    void setLeftFrameCallback(FrameCallback cb)  { if (left_)  left_->setFrameCallback(std::move(cb)); }
    void setRightFrameCallback(FrameCallback cb) { if (right_) right_->setFrameCallback(std::move(cb)); }

private:
    std::unique_ptr<ICameraSource> left_;
    std::unique_ptr<ICameraSource> right_;
    std::string sourceType_;
    int leftId_ = 0;
    int rightId_ = 1;
    std::string leftFolder_;
    std::string rightFolder_;
};

}  // namespace fc::gui
