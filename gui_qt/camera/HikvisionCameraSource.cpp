// =============================================================================
// HikvisionCameraSource.cpp —— MVS SDK 实现
// 仅当 FC_HAVE_MVS 定义时编译（CMake 找到 MvCameraControl.lib 才定义）
// =============================================================================

#ifdef FC_HAVE_MVS

#include "HikvisionCameraSource.h"

#include <MvCameraControl.h>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <chrono>
#include <string>

namespace fc::gui {

namespace {
std::string mvErrorCodeToStr(int code) {
    switch (code) {
    case MV_E_HANDLE:       return "invalid handle";
    case MV_E_SUPPORT:      return "not supported";
    case MV_E_BUFOVER:      return "buffer overflow";
    case MV_E_CALLORDER:    return "call order error";
    case MV_E_PARAMETER:    return "parameter error";
    case MV_E_RESOURCE:     return "insufficient resource";
    case MV_E_NODATA:       return "no data";
    case MV_E_PRECONDITION: return "precondition error";
    case MV_E_VERSION:      return "version mismatch";
    case MV_E_NOENOUGH_BUF: return "insufficient memory";
    case MV_E_ABNORMAL_IMAGE: return "abnormal image";
    case MV_E_UNKNOW:       return "unknown error";
    default:                return "code=" + std::to_string(code);
    }
}

void checkOk(int code, const char* tag) {
    if (code != MV_OK) {
        spdlog::warn("[Hikvision] {} failed: {} ({})", tag, mvErrorCodeToStr(code), code);
    }
}
}  // namespace

HikvisionCameraSource::HikvisionCameraSource() = default;
HikvisionCameraSource::~HikvisionCameraSource() { close(); }

bool HikvisionCameraSource::open(int deviceId) {
    close();

    MV_CC_DEVICE_INFO_LIST list{};
    int code = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
    if (code != MV_OK || list.nDeviceNum == 0) {
        spdlog::warn("[Hikvision] enum failed: code={} devices={}",
                     code, list.nDeviceNum);
        return false;
    }
    if (deviceId < 0 || static_cast<unsigned>(deviceId) >= list.nDeviceNum) {
        spdlog::warn("[Hikvision] deviceId={} out of range (nDeviceNum={})",
                     deviceId, list.nDeviceNum);
        return false;
    }

    MV_CC_DEVICE_INFO* info = list.pDeviceInfo[deviceId];
    code = MV_CC_CreateHandle(&handle_, info);
    if (code != MV_OK) {
        spdlog::warn("[Hikvision] create handle failed: {}", mvErrorCodeToStr(code));
        return false;
    }
    code = MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0);
    if (code != MV_OK) {
        spdlog::warn("[Hikvision] open failed: {}", mvErrorCodeToStr(code));
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return false;
    }

    // 关闭触发模式（连续采集）；SetEnumValue 第三参数按值传
    MV_CC_SetEnumValue(handle_, "TriggerMode", 0u);

    // 探测 width/height
    MVCC_INTVALUE_EX w{}, h{};
    MV_CC_GetIntValueEx(handle_, "Width",  &w);
    MV_CC_GetIntValueEx(handle_, "Height", &h);
    width_  = static_cast<int>(w.nCurValue);
    height_ = static_cast<int>(h.nCurValue);

    // 构造 display name
    if (info->nTLayerType == MV_GIGE_DEVICE) {
        unsigned int ip = info->SpecialInfo.stGigEInfo.nCurrentIp;
        displayName_ = "GigE[" + std::to_string((ip >> 24) & 0xFF) + "." +
                                  std::to_string((ip >> 16) & 0xFF) + "." +
                                  std::to_string((ip >> 8)  & 0xFF) + "." +
                                  std::to_string(ip & 0xFF) + "]";
    } else {
        // USB 序列号是 unsigned char[]，需以 \0 终止
        char buf[65] = {0};
        std::memcpy(buf, info->SpecialInfo.stUsb3VInfo.chSerialNumber, 64);
        displayName_ = std::string("USB[") + buf + "]";
    }

    opened_ = true;
    spdlog::info("[Hikvision] open OK: {} ({}x{})", displayName_, width_, height_);
    return true;
}

void HikvisionCameraSource::close() {
    stopAcquisition();
    if (handle_) {
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }
    opened_ = false;
}

bool HikvisionCameraSource::isOpen() const { return opened_; }

bool HikvisionCameraSource::startAcquisition() {
    if (!opened_) return false;
    int code = MV_CC_StartGrabbing(handle_);
    if (code != MV_OK) {
        spdlog::warn("[Hikvision] start grabbing failed: {}", mvErrorCodeToStr(code));
        return false;
    }
    acquiring_ = true;
    return true;
}

void HikvisionCameraSource::stopAcquisition() {
    if (handle_ && acquiring_) {
        MV_CC_StopGrabbing(handle_);
    }
    acquiring_ = false;
}

bool HikvisionCameraSource::isAcquiring() const { return acquiring_; }

bool HikvisionCameraSource::grabFrame(CameraFrame& out, int timeoutMs) {
    if (!opened_ || !acquiring_) return false;

    MV_FRAME_OUT frame{};
    int code = MV_CC_GetImageBuffer(handle_, &frame, static_cast<unsigned int>(timeoutMs));
    if (code != MV_OK || !frame.pBufAddr) {
        return false;
    }
    const MV_FRAME_OUT_INFO_EX& info = frame.stFrameInfo;

    cv::Mat src;
    if (info.enPixelType == PixelType_Gvsp_Mono8) {
        src = cv::Mat(info.nHeight, info.nWidth, CV_8UC1, frame.pBufAddr).clone();
    } else if (info.enPixelType == PixelType_Gvsp_BayerGR8 ||
               info.enPixelType == PixelType_Gvsp_BayerRG8 ||
               info.enPixelType == PixelType_Gvsp_BayerGB8 ||
               info.enPixelType == PixelType_Gvsp_BayerBG8) {
        cv::Mat bayer(info.nHeight, info.nWidth, CV_8UC1, frame.pBufAddr);
        cv::cvtColor(bayer, src, cv::COLOR_BayerGR2BGR);
    } else if (info.enPixelType == PixelType_Gvsp_RGB8_Packed) {
        cv::Mat rgb(info.nHeight, info.nWidth, CV_8UC3, frame.pBufAddr);
        cv::cvtColor(rgb, src, cv::COLOR_RGB2BGR);
    } else {
        spdlog::warn("[Hikvision] unsupported pixel type: 0x{:x}",
                     static_cast<unsigned>(info.enPixelType));
        MV_CC_FreeImageBuffer(handle_, &frame);
        return false;
    }
    MV_CC_FreeImageBuffer(handle_, &frame);

    out.image = src;
    out.frameIndex = frameIndex_++;
    out.timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return true;
}

void HikvisionCameraSource::setExposureUs(double us) {
    if (!opened_) return;
    checkOk(MV_CC_SetFloatValue(handle_, "ExposureTime", static_cast<float>(us)),
            "setExposure");
}

void HikvisionCameraSource::setGainDb(double db) {
    if (!opened_) return;
    checkOk(MV_CC_SetFloatValue(handle_, "Gain", static_cast<float>(db)),
            "setGain");
}

void HikvisionCameraSource::setTriggerMode(bool on, int32_t src) {
    if (!opened_) return;
    MV_CC_SetEnumValue(handle_, "TriggerMode", on ? 1u : 0u);
    if (on) {
        MV_CC_SetEnumValue(handle_, "TriggerSource", static_cast<unsigned int>(src));
    }
}

std::string HikvisionCameraSource::displayName() const { return displayName_; }
int HikvisionCameraSource::width() const { return width_; }
int HikvisionCameraSource::height() const { return height_; }

CameraCapability HikvisionCameraSource::capability() const {
    CameraCapability c;
    c.exposureUs = true;
    c.gainDb = true;
    c.triggerMode = true;
    return c;
}

std::vector<std::string> HikvisionCameraSource::enumerateDevices() {
    MV_CC_DEVICE_INFO_LIST list{};
    std::vector<std::string> names;
    int code = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
    if (code != MV_OK) return names;
    for (unsigned i = 0; i < list.nDeviceNum; ++i) {
        MV_CC_DEVICE_INFO* info = list.pDeviceInfo[i];
        if (!info) continue;
        std::string s = "[" + std::to_string(i) + "] ";
        if (info->nTLayerType == MV_GIGE_DEVICE) {
            unsigned int ip = info->SpecialInfo.stGigEInfo.nCurrentIp;
            s += "GigE " + std::to_string((ip >> 24) & 0xFF) + "." +
                 std::to_string((ip >> 16) & 0xFF) + "." +
                 std::to_string((ip >> 8)  & 0xFF) + "." +
                 std::to_string(ip & 0xFF);
        } else {
            char buf[65] = {0};
            std::memcpy(buf, info->SpecialInfo.stUsb3VInfo.chSerialNumber, 64);
            s += std::string("USB ") + buf;
        }
        names.push_back(s);
    }
    return names;
}

}  // namespace fc::gui

#endif  // FC_HAVE_MVS
