// =============================================================================
// GalaxyCameraSource.cpp —— 大恒 Galaxy SDK 实现
// 仅当 FC_HAVE_GALAXY 定义时编译（CMake 找到 GxIAPICPPEx.lib 才定义）
// =============================================================================

#ifdef FC_HAVE_GALAXY

// Galaxy SDK 会通过 windows.h 引入 min/max 宏污染 std::numeric_limits，先挡掉
#ifndef NOMINMAX
#define NOMINMAX 1
#endif

#include "GalaxyCameraSource.h"

#include <GalaxyIncludes.h>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>

namespace fc::gui {

namespace {
// 全局工厂 Init/Uninit 引用计数（多个 GalaxyCameraSource 实例共享）
std::mutex g_galaxyMutex;
int g_galaxyRefcount = 0;

void galaxyInit() {
    std::lock_guard<std::mutex> lk(g_galaxyMutex);
    if (g_galaxyRefcount == 0) {
        try { IGXFactory::GetInstance().Init(); }
        catch (const std::exception& e) {
            spdlog::warn("[Galaxy] Init 失败: {}", e.what());
        }
    }
    ++g_galaxyRefcount;
}
void galaxyUninit() {
    std::lock_guard<std::mutex> lk(g_galaxyMutex);
    --g_galaxyRefcount;
    if (g_galaxyRefcount == 0) {
        try { IGXFactory::GetInstance().Uninit(); }
        catch (...) {}
    }
}

// 采集回调 handler：在 Galaxy SDK 采集线程上被调用。
// 仿 LeadScanK2/series/CameraControl.cpp::CSampleCaptureEventHandler。
// 这里只做 SDK buffer → cv::Mat 的转换，再交 owner->dispatchFrame 推给上层。
class GalaxyCaptureHandler : public ICaptureEventHandler {
public:
    explicit GalaxyCaptureHandler(GalaxyCameraSource* owner) : owner_(owner) {}

    void DoOnImageCaptured(CImageDataPointer& img, void* /*pUserParam*/) override {
        static std::atomic<int64_t> s_count{0};
        int64_t n = ++s_count;
        if (n % 30 == 0) {
            std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app);
            d << "DoOnImg #" << n << " ok=" << (img->GetStatus() == GX_FRAME_STATUS_SUCCESS) << "\n";
        }
        if (img->GetStatus() != GX_FRAME_STATUS_SUCCESS) {
            return;
        }
        const int h = static_cast<int>(img->GetHeight());
        const int w = static_cast<int>(img->GetWidth());
        cv::Mat src(h, w, CV_8UC1);
        std::memcpy(src.data, img->GetBuffer(),
                    static_cast<size_t>(h) * static_cast<size_t>(w));

        CameraFrame f;
        f.frameIndex = static_cast<int64_t>(img->GetFrameID());  // 照搬 LeadScanK2：真实 frame id 供配对
        f.timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (owner_->rotate180()) {
            cv::Mat rotated;
            cv::rotate(src, rotated, cv::ROTATE_180);
            f.image = rotated;
        } else {
            f.image = src;
        }
        owner_->dispatchFrame(f);
    }

private:
    GalaxyCameraSource* owner_;
};
}  // namespace

GalaxyCameraSource::GalaxyCameraSource() = default;
GalaxyCameraSource::~GalaxyCameraSource() { close(); }

bool GalaxyCameraSource::open(int deviceId) {
    close();
    galaxyInit();

    try {
        GxIAPICPP::gxdeviceinfo_vector devList;
        IGXFactory::GetInstance().UpdateDeviceList(1000, devList);
        if (devList.empty()) {
            spdlog::warn("[Galaxy] 未找到设备");
            close();
            return false;
        }
        if (deviceId < 0 || static_cast<size_t>(deviceId) >= devList.size()) {
            spdlog::warn("[Galaxy] deviceId={} 超出范围 (共 {} 台)", deviceId, devList.size());
            close();
            return false;
        }

        // 仿参考 CameraControl::open_ScannerCamera: OpenDeviceBySN
        // （只 OpenDevice；GetRemoteFeatureControl + OpenStream 留给 startAcquisition，
        //   每次 start 重新获取、stop 释放——这是 LeadScanK2 能多次稳定采集的关键）
        GxIAPICPP::gxstring sn = devList[deviceId].GetSN();
        auto* device = new CGXDevicePointer(
            IGXFactory::GetInstance().OpenDeviceBySN(sn, GX_ACCESS_EXCLUSIVE));
        devicePtr_ = device;

        // 临时探测 width/height（不保持 feature control；正式的在 startAcquisition 获取）
        try {
            CGXFeatureControlPointer feat = (*device)->GetRemoteFeatureControl();
            width_  = static_cast<int>(feat->GetIntFeature("Width")->GetValue());
            height_ = static_cast<int>(feat->GetIntFeature("Height")->GetValue());
        } catch (...) {}

        displayName_ = std::string("[") + std::to_string(deviceId) + "] " +
                       devList[deviceId].GetDisplayName().c_str();
        opened_ = true;
        spdlog::info("[Galaxy] open OK: {} ({}x{})", displayName_, width_, height_);
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] open 异常: {}", e.what());
        close();
        return false;
    }
}

void GalaxyCameraSource::close() {
    if (streamPtr_) {
        try {
            auto* sp = static_cast<CGXStreamPointer*>(streamPtr_);
            if (acquiring_) {
                try { (*sp)->UnregisterCaptureCallback(); } catch (...) {}
                try { (*sp)->StopGrab(); } catch (...) {}
            }
            try { (*sp)->Close(); } catch (...) {}
            delete sp;
        } catch (...) {}
        streamPtr_ = nullptr;
    }
    delete static_cast<GalaxyCaptureHandler*>(captureHandler_);
    captureHandler_ = nullptr;
    if (featureCtrlPtr_) {
        delete static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        featureCtrlPtr_ = nullptr;
    }
    if (devicePtr_) {
        try {
            auto* dp = static_cast<CGXDevicePointer*>(devicePtr_);
            (*dp)->Close();
            delete dp;
        } catch (...) {}
        devicePtr_ = nullptr;
    }
    if (opened_) galaxyUninit();
    opened_ = false;
    acquiring_ = false;
    width_ = 0; height_ = 0;
}

bool GalaxyCameraSource::isOpen() const { return opened_; }

bool GalaxyCameraSource::startAcquisition() {
    { std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "startAcq begin open=" << opened_ << " acq=" << acquiring_ << "\n"; }
    if (!opened_) return false;
    try {
        auto* device = static_cast<CGXDevicePointer*>(devicePtr_);

        // 照搬 LeadScanK2：feature control 用赋值（*=），不 delete+new。第一次 new 对象，后续赋值。
        if (!featureCtrlPtr_) featureCtrlPtr_ = new CGXFeatureControlPointer;
        *static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_) = (*device)->GetRemoteFeatureControl();
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);

        if (streamPtr_) {
            try {
                auto* old = static_cast<CGXStreamPointer*>(streamPtr_);
                (*old)->Close();
                delete old;
            } catch (...) {}
            streamPtr_ = nullptr;
        }
        streamPtr_ = new CGXStreamPointer((*device)->OpenStream(0));
        auto* sp = static_cast<CGXStreamPointer*>(streamPtr_);

        // 照搬 LeadScanK2：每次 new handler（不 if check），避免复用旧 handler 到新 stream
        delete static_cast<GalaxyCaptureHandler*>(captureHandler_);
        captureHandler_ = new GalaxyCaptureHandler(this);
        (*sp)->RegisterCaptureCallback(
            static_cast<GalaxyCaptureHandler*>(captureHandler_), nullptr);

        (*sp)->StartGrab();

        try {
            (*fp)->GetEnumFeature("ExposureAuto")->SetValue("Off");
            if (exposureUs_ > 0) {
                (*fp)->GetFloatFeature("ExposureTime")->SetValue(exposureUs_);
            }
            (*fp)->GetEnumFeature("TriggerSelector")->SetValue("FrameStart");
            (*fp)->GetEnumFeature("TriggerMode")->SetValue("On");
            (*fp)->GetEnumFeature("TriggerSource")->SetValue("Line2");
        } catch (const std::exception& e) {
            spdlog::warn("[Galaxy] startAcquisition 应用曝光/触发参数异常: {}", e.what());
        }

        // AcquisitionStart 延迟到 executeAcquisitionStart()：仿 LeadScanK2，
        // 两相机都 config 完触发后再分别 AcquisitionStart，避免触发 race。

        if (width_ == 0 || height_ == 0) {
            try {
                width_  = static_cast<int>((*fp)->GetIntFeature("Width")->GetValue());
                height_ = static_cast<int>((*fp)->GetIntFeature("Height")->GetValue());
            } catch (...) {}
        }

        acquiring_ = true;
        { std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "startAcq OK\n"; }
        spdlog::info("[Galaxy] startAcquisition OK: {}, exposure={}us, Line2 trigger",
                     displayName_, exposureUs_);
        return true;
    } catch (const std::exception& e) {
        { std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "startAcq EXC: " << e.what() << "\n"; }
        spdlog::warn("[Galaxy] startAcquisition 异常: {}", e.what());
        return false;
    }
}

void GalaxyCameraSource::stopAcquisition() {
    { std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "stopAcq begin acq=" << acquiring_ << "\n"; }
    if (!opened_ || !acquiring_) return;
    // 仿参考 close_ScannerAcquisition: AcquisitionStop → StopGrab → Unregister → Close stream。
    // stream 每次关掉，下次 startAcquisition 重新 OpenStream（否则第二次不出帧）。
    try {
        auto* sp = static_cast<CGXStreamPointer*>(streamPtr_);
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        try { (*fp)->GetCommandFeature("AcquisitionStop")->Execute(); } catch (...) {}
        try { (*sp)->StopGrab(); } catch (...) {}
        try { (*sp)->UnregisterCaptureCallback(); } catch (...) {}
        delete static_cast<GalaxyCaptureHandler*>(captureHandler_);
        captureHandler_ = nullptr;
        try { (*sp)->Close(); } catch (...) {}
        delete sp;
        streamPtr_ = nullptr;
    } catch (...) {}
    acquiring_ = false;
}

bool GalaxyCameraSource::isAcquiring() const { return acquiring_; }

void GalaxyCameraSource::executeAcquisitionStart() {
    if (!opened_ || !featureCtrlPtr_) {
        std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "execAcqStart SKIP\n";
        return;
    }
    try {
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        (*fp)->GetCommandFeature("AcquisitionStart")->Execute();
        std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "execAcqStart done\n";
    } catch (const std::exception& e) {
        std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "execAcqStart EXC: " << e.what() << "\n";
        spdlog::warn("[Galaxy] executeAcquisitionStart 异常: {}", e.what());
    }
}

bool GalaxyCameraSource::grabFrame(CameraFrame& out, int timeoutMs) {
    if (!opened_ || !acquiring_) return false;
    try {
        auto* sp = static_cast<CGXStreamPointer*>(streamPtr_);
        CImageDataPointer img = (*sp)->GetImage(timeoutMs);
        if (img->GetStatus() != GX_FRAME_STATUS_SUCCESS) return false;

        cv::Mat src(img->GetHeight(), img->GetWidth(), CV_8UC1);
        std::memcpy(src.data, img->GetBuffer(),
                    static_cast<size_t>(img->GetHeight()) * img->GetWidth());

        if (rotate180_) {
            cv::Mat rotated;
            cv::rotate(src, rotated, cv::ROTATE_180);
            out.image = rotated;
        } else {
            out.image = src;
        }
        out.frameIndex = frameIndex_++;
        out.timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] grabFrame 异常: {}", e.what());
        return false;
    }
}

void GalaxyCameraSource::setExposureUs(double us) {
    exposureUs_ = us;  // 始终缓存, startAcquisition 按参考顺序应用
    if (!opened_ || !featureCtrlPtr_) return;  // featureCtrl 仅在 startAcquisition 后可用
    try {
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        (*fp)->GetEnumFeature("ExposureAuto")->SetValue("Off");
        (*fp)->GetFloatFeature("ExposureTime")->SetValue(us);
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] setExposureUs 异常: {}", e.what());
    }
}

void GalaxyCameraSource::setGainDb(double db) {
    if (!opened_ || !featureCtrlPtr_) return;
    try {
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        (*fp)->GetEnumFeature("GainAuto")->SetValue("Off");
        (*fp)->GetFloatFeature("Gain")->SetValue(db);
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] setGainDb 异常: {}", e.what());
    }
}

void GalaxyCameraSource::setTriggerMode(bool on, int32_t src) {
    if (!opened_ || !featureCtrlPtr_) return;
    try {
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        (*fp)->GetEnumFeature("TriggerSelector")->SetValue("FrameStart");
        (*fp)->GetEnumFeature("TriggerMode")->SetValue(on ? "On" : "Off");
        if (on) {
            // src=0 → Line1, 1 → Line2, 2 → Software 等
            const char* srcs[] = { "Line1", "Line2", "Software", "Counter" };
            const char* s = (src >= 0 && src < 4) ? srcs[src] : "Line2";
            (*fp)->GetEnumFeature("TriggerSource")->SetValue(s);
        }
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] setTriggerMode 异常: {}", e.what());
    }
}

void GalaxyCameraSource::triggerOnce() {
    if (!opened_) return;
    try {
        auto* fp = static_cast<CGXFeatureControlPointer*>(featureCtrlPtr_);
        (*fp)->GetCommandFeature("TriggerSoftware")->Execute();
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] triggerOnce 异常: {}", e.what());
    }
}

void GalaxyCameraSource::setFrameCallback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(cbMutex_);
    frameCb_ = std::move(cb);
}

void GalaxyCameraSource::dispatchFrame(const CameraFrame& f) {
    // 在 SDK 采集线程上被调用：先在锁内拷贝出回调对象，再无锁调用（避免持锁回调死锁）
    std::function<void(const CameraFrame&)> cb;
    {
        std::lock_guard<std::mutex> lk(cbMutex_);
        cb = frameCb_;
    }
    static std::atomic<int64_t> s_n{0};
    int64_t n = ++s_n;
    if (n % 20 == 0) spdlog::info("[Galaxy] dispatchFrame#{} cb_set={}", n, cb ? 1 : 0);
    if (cb) cb(f);
}

std::string GalaxyCameraSource::displayName() const { return displayName_; }
int GalaxyCameraSource::width() const { return width_; }
int GalaxyCameraSource::height() const { return height_; }

CameraCapability GalaxyCameraSource::capability() const {
    CameraCapability c;
    c.exposureUs = true;
    c.gainDb = true;
    c.triggerMode = true;
    return c;
}

std::vector<std::string> GalaxyCameraSource::enumerateDevices() {
    std::vector<std::string> names;
    galaxyInit();
    try {
        GxIAPICPP::gxdeviceinfo_vector devList;
        IGXFactory::GetInstance().UpdateDeviceList(1000, devList);
        for (size_t i = 0; i < devList.size(); ++i) {
            names.push_back("[" + std::to_string(i) + "] " +
                            std::string(devList[i].GetDisplayName().c_str()));
        }
    } catch (const std::exception& e) {
        spdlog::warn("[Galaxy] enumerate 异常: {}", e.what());
    }
    galaxyUninit();
    return names;
}

}  // namespace fc::gui

#endif  // FC_HAVE_GALAXY
