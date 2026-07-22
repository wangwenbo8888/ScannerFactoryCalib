#include "calib_io.h"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fc {

namespace fs = std::filesystem;
using json = nlohmann::json;

CameraCalibConfig CameraCalibConfig::fromJson(const std::string& path) {
    CameraCalibConfig c;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::warn("config not found: {}, using defaults", path);
        return c;
    }
    json j = json::parse(ifs, nullptr, true);
    if (j.contains("chessboard")) {
        const auto& cb = j["chessboard"];
        if (cb.contains("width"))        c.chessWidth = cb["width"];
        if (cb.contains("height"))       c.chessHeight = cb["height"];
        if (cb.contains("square_size_mm")) c.squareSizeMm = cb["square_size_mm"];
    }
    if (j.contains("image_size")) {
        c.imageWidth  = j["image_size"][0];
        c.imageHeight = j["image_size"][1];
    }
    if (j.contains("intrinsic_flags"))      c.intrinsicFlags = j["intrinsic_flags"];
    if (j.contains("use_calibrateCameraRO")) c.useCalibrateCameraRO = j["use_calibrateCameraRO"];
    if (j.contains("reproj_error_threshold")) c.reprojErrorThreshold = j["reproj_error_threshold"];
    if (j.contains("temperature")) {
        const auto& t = j["temperature"];
        if (t.contains("cte"))           c.cte = t["cte"];
        if (t.contains("referenceTemp")) c.referenceTemp = t["referenceTemp"];
        if (t.contains("tempRangeMin"))  c.tempRangeMin = t["tempRangeMin"];
        if (t.contains("tempRangeMax"))  c.tempRangeMax = t["tempRangeMax"];
        if (t.contains("tempStep"))      c.tempStep = t["tempStep"];
    }
    if (j.contains("rectify")) {
        const auto& r = j["rectify"];
        if (r.contains("alpha")) c.rectifyAlpha = r["alpha"];
        if (r.contains("flags")) c.rectifyFlags = r["flags"];
    }
    return c;
}

std::optional<CameraInput> loadCameraInput(const std::string& dir) {
    CameraInput in;
    in.config = CameraCalibConfig::fromJson(dir + "/config.json");

    // 参考温度：优先 temps.txt 的 ref_temp 行，否则用 config.referenceTemp
    std::ifstream tf(dir + "/temps.txt");
    if (tf) {
        std::string key; double v;
        while (tf >> key >> v) {
            if (key == "ref_temp") { in.config.referenceTemp = v; break; }
        }
    }

    fs::path ldir = fs::path(dir) / "left";
    fs::path rdir = fs::path(dir) / "right";
    if (!fs::exists(ldir) || !fs::exists(rdir)) {
        spdlog::error("left/ or right/ missing in {}", dir);
        return std::nullopt;
    }
    std::vector<fs::path> lfiles;
    for (auto& e : fs::directory_iterator(ldir))
        if (e.path().extension() == ".png" || e.path().extension() == ".jpg")
            lfiles.push_back(e.path());
    std::sort(lfiles.begin(), lfiles.end());
    for (const auto& lf : lfiles) {
        fs::path rf = rdir / lf.filename();
        if (!fs::exists(rf)) {
            spdlog::warn("skip {}: no right pair", lf.filename().string());
            continue;
        }
        cv::Mat l = cv::imread(lf.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat r = cv::imread(rf.string(), cv::IMREAD_GRAYSCALE);
        if (l.empty() || r.empty()) {
            spdlog::warn("skip {}: read failed", lf.filename().string());
            continue;
        }
        in.frames.push_back({l, r});
    }
    spdlog::info("loaded {} frame pairs from {}", in.frames.size(), dir);
    if (in.frames.empty()) return std::nullopt;
    return in;
}

nlohmann::json fc::buildCameraCalibJson(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const calib::StereoRectifyCpuResult& rectify,
    const calib::IntrinsicCompensateCPUResult& intrinTableL,
    const calib::IntrinsicCompensateCPUResult& intrinTableR,
    const calib::ExtrinsicCompensateCPUResult& extrinTable,
    const calib::StereoRectifyTempTableResult& rectifyTable)
{
    nlohmann::json j;
    j["schema"] = "factory_calib.camera_calib.v1";
    j["imageSize"] = {cfg.imageWidth, cfg.imageHeight};
    j["referenceTemp"] = cfg.referenceTemp;
    j["cte"] = cfg.cte;
    j["tempRangeMin"] = cfg.tempRangeMin;
    j["tempRangeMax"] = cfg.tempRangeMax;
    j["tempStep"] = cfg.tempStep;
    j["intrinsic"] = intrin.toJson();
    j["extrinsic"] = extrin.toJson();
    j["rectify"] = rectify.toJson();
    j["intrinsicTempTableL"] = intrinTableL.toJson();
    j["intrinsicTempTableR"] = intrinTableR.toJson();
    j["extrinsicTempTable"] = extrinTable.toJson();
    j["stereoRectifyTempTable"] = rectifyTable.toJson();
    return j;
}

bool fc::writeJson(const std::string& path, const nlohmann::json& j) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        spdlog::error("cannot write {}", path);
        return false;
    }
    ofs << j.dump(2);
    return true;
}

} // namespace fc
