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
// applyLaserParamsJson —— 统一参数解析器（主库与数据集 config 共用）
// 兼容两种写法：嵌套 {"plane_map": {...}} 或顶层平铺。
// present-key 覆盖：键出现才覆盖，未出现保持入参现值（层叠合并的基础）。
// 温度缺省不覆盖（继承 handoff）。
// ============================================================================
void applyLaserParamsJson(const json& j, LaserCalibConfig& c, LaserOpParams& ops) {
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

    // F7 漂移解线程数: 缺省 0=自动; <=0 视为 1（串行回退）
    if (j.contains("f7Threads")) {
        const int v = j["f7Threads"].get<int>();
        c.f7Threads = v > 0 ? v : 1;
    }

    // CMTT 档级并行线程数: 原样透传（0=自动由生成器解析; <=-1 生成器视为 1）
    if (j.contains("cmttThreads")) {
        c.cmttThreads = j["cmttThreads"].get<int>();
    }

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

    if (j.contains("laser_label") && j["laser_label"].is_object()) {
        const auto& ll = j["laser_label"];
        if (ll.contains("scanDirection")) c.labelScanDirection = ll["scanDirection"].get<int>();
        if (ll.contains("centerRowOffset")) c.labelCenterRowOffset = ll["centerRowOffset"].get<int>();
        if (ll.contains("realLineTolerance"))
            ops.labelRealLineTolerance = ll["realLineTolerance"].get<int>();
    }

    // ---- 算子层可调组（2026-10-04 主库化；编译默认＝产线定稿值）----

    if (j.contains("mask") && j["mask"].is_object()) {
        const auto& m = j["mask"];
        if (m.contains("threshold"))       ops.maskThreshold       = m["threshold"].get<int>();
        if (m.contains("erodeSize"))       ops.maskErodeSize       = m["erodeSize"].get<int>();
        if (m.contains("laserDilateSize")) ops.maskLaserDilateSize = m["laserDilateSize"].get<int>();
        if (m.contains("postErodeSize"))   ops.maskPostErodeSize   = m["postErodeSize"].get<int>();
    }
    if (j.contains("ccl") && j["ccl"].is_object()) {
        const auto& m = j["ccl"];
        if (m.contains("minArea"))   ops.cclMinArea   = m["minArea"].get<int>();
        if (m.contains("topXCount")) ops.cclTopXCount = m["topXCount"].get<int>();
    }
    if (j.contains("steger") && j["steger"].is_object()) {
        const auto& s = j["steger"];
        if (s.contains("sigma"))         ops.stegerSigma         = s["sigma"].get<float>();
        if (s.contains("kernelSize"))    ops.stegerKernelSize    = s["kernelSize"].get<int>();
        if (s.contains("lowThreshold"))  ops.stegerLowThreshold  = s["lowThreshold"].get<float>();
        if (s.contains("highThreshold")) ops.stegerHighThreshold = s["highThreshold"].get<float>();
        if (s.contains("maxLabels"))     ops.stegerMaxLabels     = s["maxLabels"].get<int>();
    }
    if (j.contains("match") && j["match"].is_object()) {
        const auto& m = j["match"];
        if (m.contains("disparityMarginFactor"))
            ops.matchDisparityMarginFactor = m["disparityMarginFactor"].get<double>();
    }
    if (j.contains("pjc") && j["pjc"].is_object()) {
        const auto& p = j["pjc"];
        if (p.contains("maxIterations"))        ops.pjcMaxIterations        = p["maxIterations"].get<int>();
        if (p.contains("convergenceThreshold")) ops.pjcConvergenceThreshold = p["convergenceThreshold"].get<double>();
        if (p.contains("minPoses"))             ops.pjcMinPoses             = p["minPoses"].get<int>();
        if (p.contains("minPointsPerPose"))     ops.pjcMinPointsPerPose     = p["minPointsPerPose"].get<int>();
        if (p.contains("planeFitInlierThresh")) ops.pjcPlaneFitInlierThresh = p["planeFitInlierThresh"].get<double>();
        if (p.contains("enableTiming"))         ops.pjcEnableTiming         = p["enableTiming"].get<bool>();
        if (p.contains("lambda0"))              ops.pjcLambda0              = p["lambda0"].get<double>();
        if (p.contains("lambdaDecay"))          ops.pjcLambdaDecay          = p["lambdaDecay"].get<double>();
        if (p.contains("huberToCauchyThresh"))  ops.pjcHuberToCauchyThresh  = p["huberToCauchyThresh"].get<double>();
        if (p.contains("cauchyToL2Thresh"))     ops.pjcCauchyToL2Thresh     = p["cauchyToL2Thresh"].get<double>();
        if (p.contains("topologyEpsilon"))      ops.pjcTopologyEpsilon      = p["topologyEpsilon"].get<double>();
        if (p.contains("curveDegree"))          ops.pjcCurveDegree          = p["curveDegree"].get<int>();
        if (p.contains("useCeres"))             ops.pjcUseCeres             = p["useCeres"].get<bool>();
        if (p.contains("useBlockJtJ"))          ops.pjcUseBlockJtJ          = p["useBlockJtJ"].get<bool>();
        if (p.contains("anomalyRmsThreshold"))  ops.pjcAnomalyRmsThreshold  = p["anomalyRmsThreshold"].get<double>();
        if (p.contains("robustEnabled"))        ops.pjcRobustEnabled        = p["robustEnabled"].get<bool>();
        if (p.contains("huberDelta0"))          ops.pjcHuberDelta0          = p["huberDelta0"].get<double>();
        if (p.contains("irlsMaxRounds"))        ops.pjcIrlsMaxRounds        = p["irlsMaxRounds"].get<int>();
        if (p.contains("poseWeightEnabled"))    ops.pjcPoseWeightEnabled    = p["poseWeightEnabled"].get<bool>();
        if (p.contains("acceptance") && p["acceptance"].is_object()) {
            const auto& a = p["acceptance"];
            if (a.contains("a1MaxLineRms"))      ops.pjcAccA1MaxLineRms      = a["a1MaxLineRms"].get<double>();
            if (a.contains("a2MaxP95"))          ops.pjcAccA2MaxP95          = a["a2MaxP95"].get<double>();
            if (a.contains("a3MaxOutlierRatio")) ops.pjcAccA3MaxOutlierRatio = a["a3MaxOutlierRatio"].get<double>();
            if (a.contains("a5MaxLineDrift"))    ops.pjcAccA5MaxLineDrift    = a["a5MaxLineDrift"].get<double>();
            if (a.contains("a6MaxPoseDrift"))    ops.pjcAccA6MaxPoseDrift    = a["a6MaxPoseDrift"].get<double>();
            if (a.contains("a7MaxBasinSpread"))  ops.pjcAccA7MaxBasinSpread  = a["a7MaxBasinSpread"].get<double>();
        }
    }
    if (j.contains("cmtt") && j["cmtt"].is_object()) {
        const auto& m = j["cmtt"];
        if (m.contains("tempHalfRange"))
            ops.cmttTempHalfRange = m["tempHalfRange"].get<double>();
        if (m.contains("clampedWarnRatio"))
            ops.cmttClampedWarnRatio = m["clampedWarnRatio"].get<double>();
    }
}

// ============================================================================
// LaserCalibConfig::fromJson —— 薄壳（兼容旧调用；新组解析后丢弃）
// ============================================================================
LaserCalibConfig LaserCalibConfig::fromJson(const std::string& path) {
    LaserCalibConfig c;
    LaserOpParams ops;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::warn("laser config not found: {}, using defaults", path);
        return c;
    }
    json j = json::parse(ifs, nullptr, true);
    applyLaserParamsJson(j, c, ops);
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

    // ========================================================================
    // stereoRectifyTempTable（模块1 温度矫正表整节, CMTT 4-14 源①）
    // 容错解析：档内键缺失/形状不符 → 弃该档继续；全弃或节缺失 →
    // haveRectifyTempTable=false 仅 warn 不报错（容忍模块1 老输出无此节）
    // ========================================================================
    try {
        if (!j.contains("stereoRectifyTempTable")
            || !j["stereoRectifyTempTable"].is_object()) {
            spdlog::warn("handoff has no stereoRectifyTempTable node "
                         "(legacy module1 output?); rectify temp table unavailable");
        } else {
            const auto& tt = j["stereoRectifyTempTable"];
            calib::StereoRectifyTempTableResult res;
            if (tt.contains("referenceTemp"))
                res.referenceTemp = tt["referenceTemp"].get<double>();
            if (tt.contains("cte"))
                res.cte = tt["cte"].get<double>();
            if (tt.contains("message"))
                res.message = tt["message"].get<std::string>();
            if (tt.contains("qualityFlag"))
                res.qualityFlag = static_cast<calib::QualityFlag>(
                    tt["qualityFlag"].get<int>());

            int dropped = 0;
            if (tt.contains("table") && tt["table"].is_array()) {
                for (const auto& e : tt["table"]) {
                    try {
                        if (!e.is_object())
                            throw std::runtime_error("entry is not an object");

                        calib::StereoRectifyTempEntry entry;
                        entry.temperature = e.at("temperature").get<double>();
                        if (!std::isfinite(entry.temperature))
                            throw std::runtime_error("temperature not finite");

                        entry.R1 = calib::jsonToMatAuto(e.at("R1"));
                        entry.R2 = calib::jsonToMatAuto(e.at("R2"));
                        entry.P1 = calib::jsonToMatAuto(e.at("P1"));
                        entry.P2 = calib::jsonToMatAuto(e.at("P2"));
                        entry.Q  = calib::jsonToMatAuto(e.at("Q"));
                        // 形状契约（cv::stereoRectify 输出）: R1/R2 3×3, P1/P2 3×4, Q 4×4
                        if ((entry.R1.rows != 3 || entry.R1.cols != 3)
                            || (entry.R2.rows != 3 || entry.R2.cols != 3)
                            || (entry.P1.rows != 3 || entry.P1.cols != 4)
                            || (entry.P2.rows != 3 || entry.P2.cols != 4)
                            || (entry.Q.rows != 4 || entry.Q.cols != 4))
                            throw std::runtime_error("matrix shape mismatch");

                        if (e.contains("deltaT"))
                            entry.deltaT = e["deltaT"].get<double>();
                        if (e.contains("compensatedCameraMatrixL"))
                            entry.compensatedCameraMatrixL =
                                calib::jsonToMatAuto(e["compensatedCameraMatrixL"]);
                        if (e.contains("compensatedCameraMatrixR"))
                            entry.compensatedCameraMatrixR =
                                calib::jsonToMatAuto(e["compensatedCameraMatrixR"]);
                        if (e.contains("compensatedT"))
                            entry.compensatedT = calib::jsonToMatAuto(e["compensatedT"]);
                        if (e.contains("validRoiLeft") && e["validRoiLeft"].is_object())
                            entry.validRoiLeft = {e["validRoiLeft"].at("x").get<int>(),
                                                  e["validRoiLeft"].at("y").get<int>(),
                                                  e["validRoiLeft"].at("w").get<int>(),
                                                  e["validRoiLeft"].at("h").get<int>()};
                        if (e.contains("validRoiRight") && e["validRoiRight"].is_object())
                            entry.validRoiRight = {e["validRoiRight"].at("x").get<int>(),
                                                   e["validRoiRight"].at("y").get<int>(),
                                                   e["validRoiRight"].at("w").get<int>(),
                                                   e["validRoiRight"].at("h").get<int>()};

                        res.table.push_back(std::move(entry));
                    } catch (const std::exception& e2) {
                        dropped++;
                        spdlog::warn("stereoRectifyTempTable: drop malformed entry #{}: {}",
                                     dropped, e2.what());
                    }
                }
            } else {
                spdlog::warn("stereoRectifyTempTable has no table array; ignored");
            }

            if (res.table.empty()) {
                spdlog::warn("stereoRectifyTempTable.table has no valid entries "
                             "({} dropped); ignored", dropped);
            } else {
                // temperature 升序（对齐主工程惯例）
                std::stable_sort(res.table.begin(), res.table.end(),
                                 [](const calib::StereoRectifyTempEntry& a,
                                     const calib::StereoRectifyTempEntry& b) {
                                      return a.temperature < b.temperature;
                                  });

                // 表示层校正: 0.2 步距网格的浮点累计误差会让参考温节点精确匹配失败
                // (12.5+50×0.2=22.500000000000004)。该档语义上就是参考温节点
                // (09 TempParamInterpolator 按表节点 referenceTemp 逐档 == 匹配),
                // 就地校正 temperature 表示, 不触碰该档任何补偿参数。
                // 阈值 1e-6 << 步距 0.2, 循环至多命中一档, 不影响升序。
                for (auto& e : res.table) {
                    if (std::abs(e.temperature - res.referenceTemp) < 1e-6
                        && e.temperature != res.referenceTemp) {
                        spdlog::info("rectifyTempTable: tier {:.17g} corrected to referenceTemp "
                                     "{:.17g} (float grid representation fix)",
                                     e.temperature, res.referenceTemp);
                        e.temperature = res.referenceTemp;
                    }
                }
                res.success = true;
                res.tableSize = static_cast<int>(res.table.size());
                if (dropped > 0)
                    res.message += (res.message.empty() ? "" : "; ")
                                 + std::to_string(dropped) + " malformed entries dropped";
                h.rectifyTempTable = std::move(res);
                h.haveRectifyTempTable = true;
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("stereoRectifyTempTable parse failed ({}); ignored", e.what());
        h.haveRectifyTempTable = false;
    }

    // 必需矩阵校验（K_L/R, R, T, Q 是 4-1~4-14 流程不可缺的）
    if (h.cameraMatrixL.empty() || h.cameraMatrixR.empty()
        || h.R.empty() || h.T.empty() || h.Q.empty()
        || h.R1.empty() || h.R2.empty() || h.P1.empty() || h.P2.empty()) {
        spdlog::error("handoff missing required matrices "
                      "(K_L/R, dist_L/R, R, T, R1, R2, P1, P2, Q)");
        return std::nullopt;
    }

    // review I3: imageSize 缺失会让 4-14 (curve_map_temp_table) 生成抛异常 → 崩溃
    // (curve_map_temp_table.cpp:110-130 参数校验抛 invalid_argument). 改 warn→reject.
    if (h.imageSize.width <= 0 || h.imageSize.height <= 0) {
        spdlog::error("handoff missing/invalid imageSize ({}x{}); "
                      "required by curve_map_temp_table (4-14)",
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
    // review I2: referenceTemp 锚定所有温度补偿表, 必须一致 (默认值偏差 2.5°C 会引入
    // ~5.9e-5 系统误差). 默认 LaserCalibConfig.referenceTemp=25.0 vs handoff=22.5 常见.
    if (!near(cfg.referenceTemp, h.referenceTemp)) {
        why = "referenceTemp mismatch: config=" + std::to_string(cfg.referenceTemp)
              + " handoff=" + std::to_string(h.referenceTemp)
              + " (锚定所有温度补偿表, 必须一致)";
        return false;
    }
    // CMTT 4-14 源① 三方一致性: 温表锚点 referenceTemp 与 config 必须一致
    if (h.haveRectifyTempTable
        && !near(cfg.referenceTemp, h.rectifyTempTable.referenceTemp)) {
        why = "rectifyTempTable.referenceTemp mismatch: config="
              + std::to_string(cfg.referenceTemp)
              + " table=" + std::to_string(h.rectifyTempTable.referenceTemp)
              + " (CMTT 4-14 源① 三方一致, 必须一致)";
        return false;
    }
    // 温窗覆盖: 表首末档（升序）应覆盖 [referenceTemp-15, referenceTemp+15];
    // 不足不 fail, 运行期 clamped 兜底（对齐主工程偏差 B 语义）——仅提示
    if (h.haveRectifyTempTable && !h.rectifyTempTable.table.empty()) {
        const auto& tbl = h.rectifyTempTable.table;
        double lo = cfg.referenceTemp - 15.0;
        double hi = cfg.referenceTemp + 15.0;
        if (tbl.front().temperature > lo + 1e-9
            || tbl.back().temperature < hi - 1e-9) {
            spdlog::warn("rectifyTempTable coverage [{:.2f},{:.2f}] does not cover "
                         "[{:.2f},{:.2f}]; clamped fallback applies "
                         "(main-project deviation-B semantics)",
                         tbl.front().temperature, tbl.back().temperature, lo, hi);
        }
    }
    return true;
}

// ============================================================================
// loadLaserInput
// 扫描 <dir>/pose_*/  下 L_tube*.png + R_tube*.png 配对
// ============================================================================
std::optional<LaserInput> loadLaserInput(const std::string& dir,
                                         const LaserCalibConfig* baseCfg,
                                         const LaserOpParams* baseOps) {
    if (!fs::exists(dir)) {
        spdlog::error("laser input dir not found: {}", dir);
        return std::nullopt;
    }

    LaserInput in;
    // 三层合并的最后一层：数据集 config.json 逐叶覆盖上层（主库/内置）值
    in.config = baseCfg ? *baseCfg : LaserCalibConfig{};
    in.ops    = baseOps ? *baseOps : LaserOpParams{};
    {
        const std::string cfgPath = dir + "/config.json";
        std::ifstream ifs(cfgPath);
        if (ifs.is_open()) {
            applyLaserParamsJson(json::parse(ifs, nullptr, true), in.config, in.ops);
        } else {
            spdlog::warn("laser config not found: {}, keep base/master values", cfgPath);
        }
    }

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
// writeLaserJson（命名避让模块1 的 writeJson，防 GUI 双库链接同符号）
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
