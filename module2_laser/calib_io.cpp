#include "calib_io.h"

#include "common/json_utils.h"   // calib::jsonToMatAuto

#include <opencv2/imgcodecs.hpp> // cv::imread (Release OpenCV 必带 imgcodecs)
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fc {

namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================================
// LaserCalibConfig::fromJson
// 兼容两种写法：嵌套 {"plane_map": {...}} 或顶层平铺。
// 温度缺省不覆盖（继承 handoff）。
// ============================================================================
LaserCalibConfig LaserCalibConfig::fromJson(const std::string& path) {
    LaserCalibConfig c;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::warn("laser config not found: {}, using defaults", path);
        return c;
    }
    json j = json::parse(ifs, nullptr, true);

    // 嵌套 plane_map（优先）
    if (j.contains("plane_map") && j["plane_map"].is_object()) {
        const auto& pm = j["plane_map"];
        if (pm.contains("gridStep"))     c.gridStep     = pm["gridStep"].get<float>();
        if (pm.contains("depthMin"))     c.depthMin     = pm["depthMin"].get<float>();
        if (pm.contains("depthMax"))     c.depthMax     = pm["depthMax"].get<float>();
        if (pm.contains("depthSamples")) c.depthSamples = pm["depthSamples"].get<int>();
        if (pm.contains("epipolarStep")) c.epipolarStep = pm["epipolarStep"].get<float>();
    }
    // 顶层平铺（fallback）
    if (j.contains("gridStep"))     c.gridStep     = j["gridStep"].get<float>();
    if (j.contains("depthMin"))     c.depthMin     = j["depthMin"].get<float>();
    if (j.contains("depthMax"))     c.depthMax     = j["depthMax"].get<float>();
    if (j.contains("depthSamples")) c.depthSamples = j["depthSamples"].get<int>();
    if (j.contains("epipolarStep")) c.epipolarStep = j["epipolarStep"].get<float>();

    if (j.contains("deviceId")) c.deviceId = j["deviceId"].get<int>();

    if (j.contains("lineIds") && j["lineIds"].is_array()) {
        c.lineIds = j["lineIds"].get<std::vector<int>>();
    }

    if (j.contains("temperature") && j["temperature"].is_object()) {
        const auto& t = j["temperature"];
        if (t.contains("cte"))           c.cte           = t["cte"].get<double>();
        if (t.contains("referenceTemp")) c.referenceTemp = t["referenceTemp"].get<double>();
        if (t.contains("tempRangeMin"))  c.tempRangeMin  = t["tempRangeMin"].get<double>();
        if (t.contains("tempRangeMax"))  c.tempRangeMax  = t["tempRangeMax"].get<double>();
        if (t.contains("tempStep"))      c.tempStep      = t["tempStep"].get<double>();
    }

    if (j.contains("rectify") && j["rectify"].is_object()) {
        const auto& r = j["rectify"];
        if (r.contains("alpha")) c.rectifyAlpha = r["alpha"].get<double>();
        if (r.contains("flags")) c.rectifyFlags = r["flags"].get<int>();
    }

    return c;
}

// ============================================================================
// loadCameraCalibHandoff
// 解析模块1 输出的 camera_calib.json（schema 见 calib_io.h 注释）
// ============================================================================
std::optional<CameraCalibHandoff> loadCameraCalibHandoff(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::error("cannot open handoff: {}", path);
        return std::nullopt;
    }

    json j;
    try {
        j = json::parse(ifs, nullptr, true);
    } catch (const std::exception& e) {
        spdlog::error("handoff JSON parse failed: {}", e.what());
        return std::nullopt;
    }

    CameraCalibHandoff h;

    try {
        if (j.contains("schema")) h.schema = j["schema"].get<std::string>();

        if (j.contains("imageSize") && j["imageSize"].is_array()
            && j["imageSize"].size() >= 2) {
            h.imageSize = cv::Size(j["imageSize"][0].get<int>(),
                                   j["imageSize"][1].get<int>());
        }

        if (j.contains("referenceTemp")) h.referenceTemp = j["referenceTemp"].get<double>();
        if (j.contains("cte"))           h.cte           = j["cte"].get<double>();
        if (j.contains("tempRangeMin"))  h.tempRangeMin  = j["tempRangeMin"].get<double>();
        if (j.contains("tempRangeMax"))  h.tempRangeMax  = j["tempRangeMax"].get<double>();
        if (j.contains("tempStep"))      h.tempStep      = j["tempStep"].get<double>();

        if (j.contains("intrinsic") && j["intrinsic"].is_object()) {
            const auto& intr = j["intrinsic"];
            if (intr.contains("left") && intr["left"].is_object()) {
                const auto& L = intr["left"];
                if (L.contains("camera_matrix"))
                    h.cameraMatrixL = calib::jsonToMatAuto(L["camera_matrix"]);
                if (L.contains("dist_coeffs"))
                    h.distCoeffsL = calib::jsonToMatAuto(L["dist_coeffs"]);
            }
            if (intr.contains("right") && intr["right"].is_object()) {
                const auto& Rr = intr["right"];
                if (Rr.contains("camera_matrix"))
                    h.cameraMatrixR = calib::jsonToMatAuto(Rr["camera_matrix"]);
                if (Rr.contains("dist_coeffs"))
                    h.distCoeffsR = calib::jsonToMatAuto(Rr["dist_coeffs"]);
            }
        }

        if (j.contains("extrinsic") && j["extrinsic"].is_object()) {
            const auto& ext = j["extrinsic"];
            if (ext.contains("R")) h.R = calib::jsonToMatAuto(ext["R"]);
            if (ext.contains("T")) h.T = calib::jsonToMatAuto(ext["T"]);
            // 可选 fallback：若 intrinsic 节点缺，从 extrinsic.camera_matrix_l/r 取
            if (h.cameraMatrixL.empty() && ext.contains("camera_matrix_l"))
                h.cameraMatrixL = calib::jsonToMatAuto(ext["camera_matrix_l"]);
            if (h.distCoeffsL.empty() && ext.contains("dist_coeffs_l"))
                h.distCoeffsL = calib::jsonToMatAuto(ext["dist_coeffs_l"]);
            if (h.cameraMatrixR.empty() && ext.contains("camera_matrix_r"))
                h.cameraMatrixR = calib::jsonToMatAuto(ext["camera_matrix_r"]);
            if (h.distCoeffsR.empty() && ext.contains("dist_coeffs_r"))
                h.distCoeffsR = calib::jsonToMatAuto(ext["dist_coeffs_r"]);
        }

        if (j.contains("rectify") && j["rectify"].is_object()) {
            const auto& rec = j["rectify"];
            if (rec.contains("R1")) h.R1 = calib::jsonToMatAuto(rec["R1"]);
            if (rec.contains("R2")) h.R2 = calib::jsonToMatAuto(rec["R2"]);
            if (rec.contains("P1")) h.P1 = calib::jsonToMatAuto(rec["P1"]);
            if (rec.contains("P2")) h.P2 = calib::jsonToMatAuto(rec["P2"]);
            if (rec.contains("Q"))  h.Q  = calib::jsonToMatAuto(rec["Q"]);
        }
    } catch (const std::exception& e) {
        spdlog::error("handoff field extract failed: {}", e.what());
        return std::nullopt;
    }

    // 必需矩阵校验（K_L/R, R, T, Q 是 4-1~4-13 流程不可缺的）
    if (h.cameraMatrixL.empty() || h.cameraMatrixR.empty()
        || h.R.empty() || h.T.empty() || h.Q.empty()
        || h.R1.empty() || h.R2.empty() || h.P1.empty() || h.P2.empty()) {
        spdlog::error("handoff missing required matrices "
                      "(K_L/R, dist_L/R, R, T, R1, R2, P1, P2, Q)");
        return std::nullopt;
    }

    // review I3: imageSize 缺失会让 4-13 (PlaneMapTempTable) 构造抛异常 → 崩溃
    // (plane_map_temp_table.cpp:58-59 校验 imageSize 必须正数). 改 warn→reject.
    if (h.imageSize.width <= 0 || h.imageSize.height <= 0) {
        spdlog::error("handoff missing/invalid imageSize ({}x{}); "
                      "required by 4-13 plane_map_temp_table",
                      h.imageSize.width, h.imageSize.height);
        return std::nullopt;
    }

    return h;
}

// ============================================================================
// validateHandoffConsistency
// ============================================================================
bool validateHandoffConsistency(const LaserCalibConfig& cfg,
                                const CameraCalibHandoff& h,
                                std::string& why) {
    auto near = [](double a, double b, double eps = 1e-9) {
        return std::fabs(a - b) <= eps;
    };

    if (!near(cfg.cte, h.cte)) {
        why = "cte mismatch: config=" + std::to_string(cfg.cte)
              + " handoff=" + std::to_string(h.cte);
        return false;
    }
    if (!near(cfg.tempRangeMin, h.tempRangeMin)) {
        why = "tempRangeMin mismatch: config=" + std::to_string(cfg.tempRangeMin)
              + " handoff=" + std::to_string(h.tempRangeMin);
        return false;
    }
    if (!near(cfg.tempRangeMax, h.tempRangeMax)) {
        why = "tempRangeMax mismatch: config=" + std::to_string(cfg.tempRangeMax)
              + " handoff=" + std::to_string(h.tempRangeMax);
        return false;
    }
    if (!near(cfg.tempStep, h.tempStep)) {
        why = "tempStep mismatch: config=" + std::to_string(cfg.tempStep)
              + " handoff=" + std::to_string(h.tempStep);
        return false;
    }
    // review I2: referenceTemp 锁定所有温度补偿表, 必须一致 (默认值偏差 2.5°C 会引入
    // ~5.9e-5 系统误差). 默认 LaserCalibConfig.referenceTemp=25.0 vs handoff=22.5 常见.
    if (!near(cfg.referenceTemp, h.referenceTemp)) {
        why = "referenceTemp mismatch: config=" + std::to_string(cfg.referenceTemp)
              + " handoff=" + std::to_string(h.referenceTemp)
              + " (锁定所有温度补偿表, 必须一致)";
        return false;
    }
    return true;
}

// ============================================================================
// loadLaserInput
// 扫描 <dir>/pose_*/  下 L_tube*.png + R_tube*.png 配对
// ============================================================================
std::optional<LaserInput> loadLaserInput(const std::string& dir) {
    if (!fs::exists(dir)) {
        spdlog::error("laser input dir not found: {}", dir);
        return std::nullopt;
    }

    LaserInput in;
    in.config = LaserCalibConfig::fromJson(dir + "/config.json");

    auto h = loadCameraCalibHandoff(dir + "/camera_calib.json");
    if (!h) {
        spdlog::error("load handoff failed from {}/camera_calib.json", dir);
        return std::nullopt;
    }
    in.handoff = std::move(*h);

    std::string why;
    if (!validateHandoffConsistency(in.config, in.handoff, why)) {
        spdlog::error("handoff inconsistent: {}", why);
        return std::nullopt;
    }

    // 扫 pose_* 子目录（按名排序保证可复现）
    std::vector<fs::path> poseDirs;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_directory()) continue;
        std::string name = e.path().filename().string();
        if (name.rfind("pose", 0) == 0) poseDirs.push_back(e.path());
    }
    std::sort(poseDirs.begin(), poseDirs.end());

    if (poseDirs.empty()) {
        spdlog::error("no pose_* subdirectories in {}", dir);
        return std::nullopt;
    }

    for (auto& pd : poseDirs) {
        std::vector<PoseFrame> tubeFrames;

        std::vector<fs::path> lFiles;
        for (auto& e : fs::directory_iterator(pd)) {
            std::string name = e.path().filename().string();
            // 接受 L_tube0.png / L_tube00.png / l_tube0.png
            if (name.size() > 6 &&
                (name[0] == 'L' || name[0] == 'l') &&
                name.rfind("_tube", 1) == 1 &&
                (e.path().extension() == ".png" || e.path().extension() == ".jpg")) {
                lFiles.push_back(e.path());
            }
        }
        std::sort(lFiles.begin(), lFiles.end());

        for (auto& lf : lFiles) {
            // L_tubeN.png -> R_tubeN.png
            std::string rname = "R" + lf.filename().string().substr(1);
            fs::path rf = pd / rname;
            if (!fs::exists(rf)) {
                spdlog::warn("skip {}: no right pair {}",
                             lf.filename().string(), rname);
                continue;
            }
            cv::Mat l = cv::imread(lf.string(), cv::IMREAD_GRAYSCALE);
            cv::Mat r = cv::imread(rf.string(), cv::IMREAD_GRAYSCALE);
            if (l.empty() || r.empty()) {
                spdlog::warn("skip {}: read failed", lf.filename().string());
                continue;
            }
            tubeFrames.push_back({l, r});
        }

        if (tubeFrames.empty()) {
            spdlog::warn("pose {} has no valid tube pairs, skip",
                         pd.filename().string());
            continue;
        }

        in.poseDirs.push_back(pd.filename().string());
        in.poseFrames.push_back(std::move(tubeFrames));
    }

    if (in.poseFrames.empty()) {
        spdlog::error("no valid pose frames loaded from {}", dir);
        return std::nullopt;
    }

    size_t totalTubes = 0;
    for (const auto& pv : in.poseFrames) totalTubes += pv.size();
    spdlog::info("loaded {} poses ({} tubes total) from {}",
                 in.poseFrames.size(), totalTubes, dir);

    return in;
}

// ============================================================================
// writeLaserJson（模块2 自有，与 module1::writeJson 隔离）
// ============================================================================
bool writeLaserJson(const std::string& path, const nlohmann::json& j) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        spdlog::error("cannot write {}", path);
        return false;
    }
    ofs << j.dump(2);
    return true;
}

} // namespace fc
