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

namespace {

// 完整组装（含过程数据）——两个公开 build*Json 的单一数据源
nlohmann::json assembleFull(
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

// intrinsic.left/right 里属于过程数据的键
constexpr const char* kIntrinsicProcessKeys[] = {"rvecs", "tvecs", "per_view_errors"};
// extrinsic 里属于过程数据的键（K/D 副本与 intrinsic 节重复）
constexpr const char* kExtrinsicProcessKeys[] = {
    "perViewErrors", "perViewEpipolarErrors", "message",
    "camera_matrix_l", "dist_coeffs_l", "camera_matrix_r", "dist_coeffs_r"};

} // namespace

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
    nlohmann::json j = assembleFull(cfg, intrin, extrin, rectify,
                                    intrinTableL, intrinTableR, extrinTable, rectifyTable);
    // 结果文件只留下游需要的矩阵 + 汇总指标，剔除过程/诊断数据
    if (j.contains("intrinsic") && j["intrinsic"].is_object()) {
        for (const char* side : {"left", "right"}) {
            if (j["intrinsic"].contains(side) && j["intrinsic"][side].is_object()) {
                for (const char* k : kIntrinsicProcessKeys)
                    j["intrinsic"][side].erase(k);
            }
        }
    }
    if (j.contains("extrinsic") && j["extrinsic"].is_object()) {
        for (const char* k : kExtrinsicProcessKeys)
            j["extrinsic"].erase(k);
    }
    return j;
}

nlohmann::json fc::buildCameraCalibProcessJson(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin)
{
    nlohmann::json p;
    p["schema"] = "factory_calib.camera_calib_process.v1";

    // 运行配置复述（问题追溯用）
    p["config"] = {
        {"chessboard", {{"width", cfg.chessWidth},
                        {"height", cfg.chessHeight},
                        {"square_size_mm", cfg.squareSizeMm}}},
        {"image_size", {cfg.imageWidth, cfg.imageHeight}},
        {"intrinsic", {{"flags", cfg.intrinsicFlags},
                       {"use_calibrateCameraRO", cfg.useCalibrateCameraRO},
                       {"reproj_error_threshold", cfg.reprojErrorThreshold}}},
        {"plate", {{"tempCoeff", cfg.plateTempCoeff}, {"temp", cfg.plateTemp}}},
        {"temperature", {{"referenceTemp", cfg.referenceTemp},
                         {"cte", cfg.cte},
                         {"tempRangeMin", cfg.tempRangeMin},
                         {"tempRangeMax", cfg.tempRangeMax},
                         {"tempStep", cfg.tempStep}}},
        {"rectify", {{"alpha", cfg.rectifyAlpha},
                     {"flags", cfg.rectifyFlags}}},
    };

    // 逐视角过程数据
    nlohmann::json intr = intrin.toJson();
    for (const char* side : {"left", "right"}) {
        if (intr.contains(side) && intr[side].is_object()) {
            nlohmann::json ps = nlohmann::json::object();
            for (const char* k : kIntrinsicProcessKeys) {
                if (intr[side].contains(k))
                    ps[k] = std::move(intr[side][k]);
            }
            if (!ps.empty()) p["intrinsic"][side] = std::move(ps);
        }
    }
    nlohmann::json ext = extrin.toJson();
    nlohmann::json pe = nlohmann::json::object();
    for (const char* k : kExtrinsicProcessKeys) {
        if (ext.contains(k))
            pe[k] = std::move(ext[k]);
    }
    if (!pe.empty()) p["extrinsic"] = std::move(pe);
    return p;
}

std::string fc::deriveProcessPath(const std::string& outputPath) {
    fs::path p(outputPath);
    return (p.parent_path() / (p.stem().string() + "_process.json")).string();
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
