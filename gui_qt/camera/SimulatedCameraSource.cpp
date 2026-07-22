#include "SimulatedCameraSource.h"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

namespace fc::gui {

namespace {
bool isImageFile(const std::string& name) {
    auto pos = name.find_last_of('.');
    if (pos == std::string::npos) return false;
    std::string ext = name.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "png" || ext == "bmp" || ext == "jpg" || ext == "jpeg"
        || ext == "tif" || ext == "tiff";
}
}  // namespace

SimulatedCameraSource::SimulatedCameraSource(const std::string& folder)
    : folder_(folder) {}

SimulatedCameraSource::~SimulatedCameraSource() { close(); }

void SimulatedCameraSource::setFolder(const std::string& dir) {
    folder_ = dir;
    files_.clear();
    cursor_ = 0;
}

bool SimulatedCameraSource::open(int deviceId) {
    (void)deviceId;
    close();
    if (folder_.empty()) {
        spdlog::warn("[SimulatedCameraSource] folder 未设置");
        return false;
    }
    cv::glob(folder_, files_);
    std::vector<std::string> filtered;
    for (const auto& f : files_) if (isImageFile(f)) filtered.push_back(f);
    files_ = std::move(filtered);
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) {
        spdlog::warn("[SimulatedCameraSource] folder={} 无图像", folder_);
        return false;
    }
    cv::Mat probe = cv::imread(files_[0], cv::IMREAD_GRAYSCALE);
    if (probe.empty()) {
        spdlog::warn("[SimulatedCameraSource] 读首帧失败: {}", files_[0]);
        return false;
    }
    width_ = probe.cols;
    height_ = probe.rows;
    opened_ = true;
    frameIndex_ = 0;
    spdlog::info("[SimulatedCameraSource] open: {} ({} files, {}x{})",
                 folder_, files_.size(), width_, height_);
    return true;
}

void SimulatedCameraSource::close() {
    stopAcquisition();
    opened_ = false;
    width_ = 0; height_ = 0;
}

bool SimulatedCameraSource::isOpen() const { return opened_; }

bool SimulatedCameraSource::startAcquisition() {
    if (!opened_) return false;
    acquiring_ = true;
    return true;
}

void SimulatedCameraSource::stopAcquisition() { acquiring_ = false; }

bool SimulatedCameraSource::isAcquiring() const { return acquiring_; }

bool SimulatedCameraSource::grabFrame(CameraFrame& out, int timeoutMs) {
    (void)timeoutMs;
    if (!opened_ || !acquiring_ || files_.empty()) return false;
    const std::string& path = files_[cursor_];
    cursor_ = (cursor_ + 1) % files_.size();  // 循环
    out.image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (out.image.empty()) return false;
    out.frameIndex = frameIndex_++;
    out.timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return true;
}

std::string SimulatedCameraSource::displayName() const {
    return std::string("Simulated[") + folder_ + "]";
}
int SimulatedCameraSource::width() const { return width_; }
int SimulatedCameraSource::height() const { return height_; }

}  // namespace fc::gui
