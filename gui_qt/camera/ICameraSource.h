#pragma once

// =============================================================================
// ICameraSource —— 单相机采集抽象
//
// 设计：
//   - 同步 API（grabFrame 阻塞），GUI 用 QTimer 或 QThread 轮询
//   - 子类：SimulatedCameraSource（读文件夹）、HikvisionCameraSource（MVS SDK）
//   - 工厂 createCameraSource(type) 按 type 实例化；可用类型走 enumerateSourceTypes()
// =============================================================================

#include <opencv2/core.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fc::gui {

struct CameraFrame {
    cv::Mat image;          // CV_8UC1（灰度）或 CV_8UC3（BGR）
    int64_t timestampNs = 0; // 设备时钟（模拟时 = system clock）
    int64_t frameIndex = 0;
};

// 帧到达回调（事件驱动采集）。
// 注意：实现在 SDK 采集线程上调用，调用方必须自行跨线程投递（如 Qt queued signal）。
using FrameCallback = std::function<void(const CameraFrame&)>;

struct CameraCapability {
    bool exposureUs = false;
    bool gainDb = false;
    bool triggerMode = false;
};

class ICameraSource {
public:
    virtual ~ICameraSource() = default;

    // —— 生命周期 ——
    virtual bool open(int deviceId = 0) = 0;
    virtual void close() = 0;
    virtual bool isOpen() const = 0;

    // —— 采集控制 ——
    virtual bool startAcquisition() = 0;
    virtual void stopAcquisition() = 0;
    virtual bool isAcquiring() const = 0;

    // 阻塞取一帧；timeoutMs<=0 → 非阻塞（无帧返回空 Mat）
    // 返回 true 表示拿到帧；false 表示超时/已停止
    virtual bool grabFrame(CameraFrame& out, int timeoutMs = 1000) = 0;

    // —— 参数（仅当 CameraCapability 对应位为 true 时有效）——
    virtual void setExposureUs(double us) { (void)us; }
    virtual void setGainDb(double db) { (void)db; }
    virtual void setTriggerMode(bool on, int32_t src = 0) { (void)on; (void)src; }

    // 软触发：发一帧指令（仅 setTriggerMode(true, Software) 后有意义）。
    // 默认 no-op；不支持的源（simulated 等）静默忽略。
    virtual void triggerOnce() {}

    // 注册帧到达回调（事件驱动）。
    // 仅支持回调推送的源（如 Galaxy）会 override；默认 no-op。
    // 应在 open() 之后、startAcquisition() 之前调用。
    virtual void setFrameCallback(FrameCallback cb) { (void)cb; }

    // 单独执行 AcquisitionStart。仿 LeadScanK2：startAcquisition 只 config（开流+触发），
    // 由上层在所有相机 config 完后统一调用此方法，避免触发 race。
    virtual void executeAcquisitionStart() {}

    // —— 信息 ——
    virtual std::string displayName() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual CameraCapability capability() const = 0;

    // 枚举所有可用设备（不打开）；返回 display 列表，索引即 deviceId
    static std::vector<std::string> enumerateDevices(const std::string& sourceType);
};

// 工厂：sourceType ∈ {"simulated", "hikvision", "galaxy"}
//   "simulated" 始终可用；"hikvision" 仅当 CMake 找到 MVS 时编译进实现
//   folder 仅 simulated 用：作为图像序列源
std::unique_ptr<ICameraSource> createCameraSource(
    const std::string& sourceType,
    const std::string& folder = std::string());

// 当前编译进二进制的 source 类型
std::vector<std::string> availableSourceTypes();

}  // namespace fc::gui
