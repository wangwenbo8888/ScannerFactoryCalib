#include "StereoCameraRig.h"

#include <spdlog/spdlog.h>

namespace fc::gui {

StereoCameraRig::~StereoCameraRig() { close(); }

bool StereoCameraRig::configure(const std::string& sourceType,
                                int leftDeviceId, int rightDeviceId,
                                const std::string& leftFolder,
                                const std::string& rightFolder) {
    sourceType_ = sourceType;
    leftId_ = leftDeviceId;
    rightId_ = rightDeviceId;
    leftFolder_ = leftFolder;
    rightFolder_ = rightFolder;
    left_  = createCameraSource(sourceType, leftFolder);
    right_ = createCameraSource(sourceType, rightFolder);
    return left_ && right_;
}

bool StereoCameraRig::open() {
    if (!left_ || !right_) return false;
    if (!left_->open(leftId_)) {
        spdlog::warn("[StereoCameraRig] 左源 open 失败");
        return false;
    }
    if (!right_->open(rightId_)) {
        spdlog::warn("[StereoCameraRig] 右源 open 失败");
        left_->close();
        return false;
    }
    return true;
}

void StereoCameraRig::close() {
    if (left_)  left_->close();
    if (right_) right_->close();
}

bool StereoCameraRig::isOpen() const {
    return left_ && right_ && left_->isOpen() && right_->isOpen();
}

bool StereoCameraRig::startAcquisition() {
    if (!isOpen()) return false;
    // 仿 LeadScanK2：先两相机都 config（开流+触发），再分别 AcquisitionStart，避免触发 race
    if (!left_->startAcquisition())  return false;
    if (!right_->startAcquisition()) {
        left_->stopAcquisition();
        return false;
    }
    left_->executeAcquisitionStart();
    right_->executeAcquisitionStart();
    return true;
}

void StereoCameraRig::stopAcquisition() {
    if (left_)  left_->stopAcquisition();
    if (right_) right_->stopAcquisition();
}

bool StereoCameraRig::isAcquiring() const {
    return left_ && right_ && left_->isAcquiring() && right_->isAcquiring();
}

bool StereoCameraRig::grabStereo(CameraFrame& leftOut, CameraFrame& rightOut,
                                  int timeoutMs) {
    if (!isAcquiring()) return false;
    bool okL = left_->grabFrame(leftOut, timeoutMs);
    bool okR = right_->grabFrame(rightOut, timeoutMs);
    return okL && okR;
}

bool StereoCameraRig::captureScannerStereo(CameraFrame& leftOut, CameraFrame& rightOut,
                                            double exposureUs, int timeoutMs) {
    if (!isOpen()) {
        spdlog::warn("[StereoCameraRig] captureScannerStereo: 设备未打开");
        return false;
    }

    // 1. 应用曝光（GalaxyCameraSource 内部缓存, startAcquisition 按参考顺序应用）
    if (exposureUs > 0) {
        if (left_->capability().exposureUs)  left_->setExposureUs(exposureUs);
        if (right_->capability().exposureUs) right_->setExposureUs(exposureUs);
    }

    // 2. 开采（GalaxyCameraSource::startAcquisition 内部按参考顺序：
    //       StartGrab → ExposureTime → Line2 触发 → AcquisitionStart）
    if (!startAcquisition()) {
        spdlog::warn("[StereoCameraRig] captureScannerStereo: startAcquisition 失败");
        return false;
    }

    // 3. 阻塞等扫描仪硬件触发帧到达（Line2 脉冲来 → 相机出一帧 → GetImage 返回）
    bool ok = grabStereo(leftOut, rightOut, timeoutMs);

    // 4. 停采（参考 close_ScannerAcquisition: AcquisitionStop + StopGrab）
    stopAcquisition();

    if (!ok) {
        spdlog::warn("[StereoCameraRig] captureScannerStereo: 抓帧超时/失败 "
                     "(检查扫描仪 Line2 是否在发触发脉冲)");
    }
    return ok;
}

}  // namespace fc::gui
