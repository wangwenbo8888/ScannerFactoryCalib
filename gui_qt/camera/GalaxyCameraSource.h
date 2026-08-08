#pragma once

// =============================================================================
// GalaxyCameraSource —— 大恒 Galaxy SDK 包装
//
// 参考 E:\workfold\20260509intergrate\LeadScanK2\series\CameraControl.cpp
// 用 IGXFactory + OpenDeviceBySN + OpenStream + GetImage 阻塞取帧。
//
// 默认开启 rightRotate180_：右相机物理倒装，采集后旋转 180°（与参考一致）。
// =============================================================================

#include "ICameraSource.h"
#include <atomic>
#include <mutex>
#include <string>

namespace fc::gui {

class GalaxyCameraSource : public ICameraSource {
public:
    GalaxyCameraSource();
    ~GalaxyCameraSource() override;

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
    void triggerOnce() override;
    void setFrameCallback(FrameCallback cb) override;
    void executeAcquisitionStart() override;

    std::string displayName() const override;
    int width() const override;
    int height() const override;
    CameraCapability capability() const override;

    // 180° 旋转（per-instance，左右相机可独立配置）
    // 参考代码默认旋转右图；本工程镜像安装，左右都可能需要旋转
    void setRotate180(bool on) { rotate180_ = on; }
    bool rotate180() const { return rotate180_; }

    static std::vector<std::string> enumerateDevices();

    // 供内部 SDK 采集回调线程调用：把已组装好的 CameraFrame 推给已注册的 frameCb_。
    // 不暴露 Galaxy 类型，仅用 cv 类型。
    void dispatchFrame(const CameraFrame& f);

private:
    // Galaxy SDK 智能指针（实现文件里使用具体类型）
    void* devicePtr_      = nullptr;  // CGXDevicePointer*
    void* streamPtr_      = nullptr;  // CGXStreamPointer*
    void* featureCtrlPtr_ = nullptr;  // CGXFeatureControlPointer*
    void* captureHandler_ = nullptr;  // GalaxyCaptureHandler* (ICaptureEventHandler)

    int width_  = 0;
    int height_ = 0;
    bool opened_ = false;
    bool acquiring_ = false;
    bool rotate180_ = false;   // per-instance; 由 AcquisitionTab 配置
    std::string displayName_;
    int64_t frameIndex_ = 0;
    double exposureUs_ = 0.0;  // setExposureUs 缓存; startAcquisition 时按参考顺序应用

    FrameCallback frameCb_;
    std::mutex cbMutex_;
    // SDK 采集线程上正在执行的 dispatchFrame 计数；stopAcquisition 用它等待
    // in-flight 回调退出，避免析构期间回调访问已释放的 GUI 对象。
    std::atomic<int> cbInFlight_{0};
};

}  // namespace fc::gui
