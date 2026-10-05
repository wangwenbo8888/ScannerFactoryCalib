// L6 真实数据全链分析测试（module2 L6 诊断/可视化测试同款定位）
//
// 输入: factory_calib/data_in/CAMERA261003（52 对 L{n}.bmp/R{n}.bmp，2048x1536 8bpp 灰度，
//       平铺布局——CLI 的 loadCameraInput 不适用，本测试自装载）
// 参数: factory_calib/CONFIG/camera261003.json（全量五类：数据集/温度/算子入参/矫正/门禁；
//       缺文件或单键缺失 -> 内置默认值回退并打日志，默认值与 JSON 基线一致）
// 链路: 角点提取 -> 内参 -> 外参 -> 立体矫正 -> 温度补偿四表
//       （复刻 camera_calib_cli.cpp:77-157 调用序列）
// 断言: 门禁见 gates 组；产物落 factory_calib/data_out/camera261003/
// 缺数据 -> GTEST_SKIP（无数据机器回归保持绿）

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "chessboard_corner.h"
#include "calib_io.h"  // buildCameraCalibJson/writeJson：handoff schema 组装（与 CLI 同源）
#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

using namespace fc;
using namespace calib;
namespace fs = std::filesystem;

namespace {

// ---- 全量输入参数（默认值 = CONFIG/camera261003.json 基线）----
struct ChainConfig {
    // 数据集（chessboard / image_size / dataset 组）
    int    chessW = 11;             // 内角点数
    int    chessH = 8;
    double squareMm = 15.0;         // 方格物理尺寸 mm
    int    imgW = 2048;
    int    imgH = 1536;
    int    maxFrameNum = 52;        // L1..L52 / R1..R52
    bool   swapLR = false;          // true=L{n}/R{n} 文件与物理左右对调（装载时互换），修正镜像 handoff
    // 温度（temperature 组）
    double cte = 23.6e-6;           // 铝合金线膨胀系数
    double refTemp = 22.5;          // 参考温度（数据无 temps.txt）
    double tempMin = -15.0;
    double tempMax = 15.0;
    double tempStep = 0.5;
    int    expectedTiers = 61;      // ±15/0.5
    // 算子入参（operator 组；非门禁）
    bool   useCalibCameraRO = true;
    int    intrinsicFlags = 0;
    double intrinsicLogReproj = 1.0;   // intrinsic 算子日志口径（算子内部不门禁）
    double plateTempCoeff = 5.0e-6;    // 板热膨胀
    double plateTemp = 21.0;
    double extrinsicMaxReproj = 100.0; // 外参最大重投影（同 CLI threshold×100 公式）
    int    extrinsicMinViewCount = 4;  // 同 CLI
    // 矫正（rectify 组）
    double rectifyAlpha = 0.0;
    int    rectifyFlags = cv::CALIB_ZERO_DISPARITY;  // =1024
    // 门禁（gates 组）
    int    gateMinDetected = 20;      // 检出对数下限
    double gateReprojMean = 1.0;      // px
    double gateFxDiff = 0.05;         // L/R fx 相对差
    double gateROrtho = 1e-6;         // ‖R·Rᵀ−I‖∞
    double gateBaselineMin = 10.0;    // mm
    double gateBaselineMax = 500.0;   // mm
    double gateRoiAreaRatio = 0.5;    // validRoi 面积/画幅
    double gateEpiMedian = 0.5;       // px（极线自检中位）
    double gateEpiP95 = 1.0;          // px（极线自检 P95）

    nlohmann::json toJson() const {
        return {
            {"chessboard", {{"width", chessW}, {"height", chessH}, {"square_size_mm", squareMm}}},
            {"image_size", {imgW, imgH}},
            {"dataset", {{"max_frame_num", maxFrameNum}, {"swap_lr", swapLR}}},
            {"temperature", {{"cte", cte}, {"referenceTemp", refTemp},
                             {"tempRangeMin", tempMin}, {"tempRangeMax", tempMax},
                             {"tempStep", tempStep}}},
            {"operator", {{"use_calibrateCameraRO", useCalibCameraRO},
                          {"intrinsic_flags", intrinsicFlags},
                          {"intrinsic_log_reproj", intrinsicLogReproj},
                          {"plate_temp_coeff", plateTempCoeff}, {"plate_temp", plateTemp},
                          {"extrinsic_max_reproj", extrinsicMaxReproj},
                          {"extrinsic_min_view_count", extrinsicMinViewCount}}},
            {"rectify", {{"alpha", rectifyAlpha}, {"flags", rectifyFlags}}},
            {"gates", {{"minDetectedPairs", gateMinDetected},
                       {"reprojMeanMax", gateReprojMean},
                       {"fxRelDiffMax", gateFxDiff},
                       {"rotOrthoMax", gateROrtho},
                       {"baselineMinMm", gateBaselineMin}, {"baselineMaxMm", gateBaselineMax},
                       {"roiAreaRatioMin", gateRoiAreaRatio},
                       {"epiMedianMax", gateEpiMedian}, {"epiP95Max", gateEpiP95},
                       {"expectedTempTiers", expectedTiers}}},
        };
    }
};

// 逐键回退式解析：文件缺失/损坏 -> 保持默认值；单键类型异常 -> 整体回退默认并报错
bool loadChainConfig(const fs::path& p, ChainConfig& c) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    try {
        std::ifstream f(p.string());
        nlohmann::json j = nlohmann::json::parse(f, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded()) return false;
        if (j.contains("chessboard")) {
            const auto& s = j.at("chessboard");
            c.chessW = s.value("width", c.chessW);
            c.chessH = s.value("height", c.chessH);
            c.squareMm = s.value("square_size_mm", c.squareMm);
        }
        if (j.contains("image_size") && j.at("image_size").is_array()
            && j.at("image_size").size() >= 2) {
            c.imgW = j.at("image_size")[0].get<int>();
            c.imgH = j.at("image_size")[1].get<int>();
        }
        if (j.contains("dataset")) {
            const auto& s = j.at("dataset");
            c.maxFrameNum = s.value("max_frame_num", c.maxFrameNum);
            c.swapLR = s.value("swap_lr", c.swapLR);
        }
        if (j.contains("temperature")) {
            const auto& s = j.at("temperature");
            c.cte = s.value("cte", c.cte);
            c.refTemp = s.value("referenceTemp", c.refTemp);
            c.tempMin = s.value("tempRangeMin", c.tempMin);
            c.tempMax = s.value("tempRangeMax", c.tempMax);
            c.tempStep = s.value("tempStep", c.tempStep);
        }
        if (j.contains("operator")) {
            const auto& s = j.at("operator");
            c.useCalibCameraRO = s.value("use_calibrateCameraRO", c.useCalibCameraRO);
            c.intrinsicFlags = s.value("intrinsic_flags", c.intrinsicFlags);
            c.intrinsicLogReproj = s.value("intrinsic_log_reproj", c.intrinsicLogReproj);
            c.plateTempCoeff = s.value("plate_temp_coeff", c.plateTempCoeff);
            c.plateTemp = s.value("plate_temp", c.plateTemp);
            c.extrinsicMaxReproj = s.value("extrinsic_max_reproj", c.extrinsicMaxReproj);
            c.extrinsicMinViewCount = s.value("extrinsic_min_view_count", c.extrinsicMinViewCount);
        }
        if (j.contains("rectify")) {
            c.rectifyAlpha = j.at("rectify").value("alpha", c.rectifyAlpha);
            c.rectifyFlags = j.at("rectify").value("flags", c.rectifyFlags);
        }
        if (j.contains("gates")) {
            const auto& s = j.at("gates");
            c.gateMinDetected = s.value("minDetectedPairs", c.gateMinDetected);
            c.gateReprojMean = s.value("reprojMeanMax", c.gateReprojMean);
            c.gateFxDiff = s.value("fxRelDiffMax", c.gateFxDiff);
            c.gateROrtho = s.value("rotOrthoMax", c.gateROrtho);
            c.gateBaselineMin = s.value("baselineMinMm", c.gateBaselineMin);
            c.gateBaselineMax = s.value("baselineMaxMm", c.gateBaselineMax);
            c.gateRoiAreaRatio = s.value("roiAreaRatioMin", c.gateRoiAreaRatio);
            c.gateEpiMedian = s.value("epiMedianMax", c.gateEpiMedian);
            c.gateEpiP95 = s.value("epiP95Max", c.gateEpiP95);
            c.expectedTiers = s.value("expectedTempTiers", c.expectedTiers);
        }
        return true;
    } catch (const std::exception&) {
        return false;  // 类型不匹配等 -> 调用方按默认值处理
    }
}

struct FrameRecord {
    int idx = 0;              // 帧号 1..52
    bool loaded = false;      // imread 成功且尺寸一致
    bool lOk = false;         // 左图角点检出
    bool rOk = false;         // 右图角点检出
    bool normalized = false;  // L/R 顺序归一成功
    int grayIdx = -1;         // 装载后在 grayL/grayR 中的下标
    std::string reason;       // 跳过原因
};

fs::path findDataDir() {
    const std::vector<fs::path> cands = {
        // 2026-10-05 起统一资产位：仓库根 test_data/camera261003/images（独立仓库布局）
        fs::path("test_data/camera261003/images"),                       // repo 根直跑
        fs::path("../../test_data/camera261003/images"),                 // ctest cwd=<repo>/build_*/module1_camera
        fs::path("../../../test_data/camera261003/images"),              // 更深构建目录
        // 兼容旧布局（factory_calib 为仓库子目录的工作区/作者机）
        fs::path("factory_calib/data_in/CAMERA261003"),                  // 工作区根直跑
        fs::path("../../factory_calib/data_in/CAMERA261003"),            // ctest cwd=build_fc1_rel/module1_camera
        fs::path("../../../factory_calib/data_in/CAMERA261003"),
        fs::path("E:/JEAMMWARE2601001/factory_calib/data_in/CAMERA261003"),  // 绝对兜底
    };
    for (const auto& c : cands) {
        std::error_code ec;
        if (fs::exists(c / "L1.bmp", ec) && fs::exists(c / "R1.bmp", ec)) return c;
    }
    return {};
}

// config 探测：新布局（config.json 与 images 同级）优先，回退旧 CONFIG/ 布局
fs::path findChainConfig(const fs::path& dataDir) {
    const std::vector<fs::path> cands = {
        dataDir.parent_path() / "config.json",                            // test_data/camera261003/config.json
        dataDir.parent_path().parent_path() / "CONFIG" / "camera261003.json",  // 旧布局
    };
    for (const auto& c : cands) {
        std::error_code ec;
        if (fs::exists(c, ec)) return c;
    }
    return {};
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (p <= 0.0) return v.front();  // nearest-rank 口径：p<=0 取最小值
    size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(v.size()))) - 1;
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

double matMaxAbs(const cv::Mat& m) {
    cv::Mat d;
    m.convertTo(d, CV_64F);
    double s = 0.0;
    for (int r = 0; r < d.rows; ++r)
        for (int c = 0; c < d.cols; ++c)
            s = std::max(s, std::abs(d.at<double>(r, c)));
    return s;
}

} // namespace

TEST(CameraChainAnalysis, FullChainOnCAMERA261003) {
    // ============ 0. 数据探测与装载 ============
    fs::path dataDir = findDataDir();
    if (dataDir.empty()) {
        GTEST_SKIP() << "CAMERA261003 data not found; skip full chain analysis";
    }
    // 输出目录：推导仓库根（新布局 test_data/camera261003/images 上溯 3 级；旧布局 data_in/X 上溯 2 级）
    fs::path repoRoot = (dataDir.parent_path().filename() == "camera261003")
        ? dataDir.parent_path().parent_path().parent_path()   // 新统一资产布局
        : dataDir.parent_path().parent_path();                // 旧 data_in 布局
    fs::path outDir = repoRoot / "data_out" / "camera261003";
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        ADD_FAILURE() << "create_directories failed: " << ec.message();
        return;
    }
    // 参数：test_data/camera261003/config.json（新布局）或旧 CONFIG/camera261003.json 优先，缺失回退内置默认
    ChainConfig cfg;
    const fs::path cfgPath = findChainConfig(dataDir);
    const bool cfgLoaded = !cfgPath.empty() && loadChainConfig(cfgPath, cfg);
    std::cout << "[chain] data=" << dataDir.string() << "\n[chain] out=" << outDir.string() << "\n";
    std::cout << "[chain] config=" << (cfgLoaded ? cfgPath.string() : "BUILT-IN DEFAULTS")
              << (cfgLoaded ? "" : " (file missing/corrupt)") << "\n";

    std::vector<cv::Mat> grayL, grayR;      // 成功装载的帧
    std::vector<FrameRecord> records;
    records.reserve(cfg.maxFrameNum);
    int loadFail = 0, sizeMismatch = 0;
    for (int n = 1; n <= cfg.maxFrameNum; ++n) {
        FrameRecord rec;
        rec.idx = n;
        const std::string lname = cfg.swapLR ? "R" : "L";   // swapLR：文件名与物理左右对调，装载时互换
        const std::string rname = cfg.swapLR ? "L" : "R";
        fs::path pl = dataDir / (lname + std::to_string(n) + ".bmp");
        fs::path pr = dataDir / (rname + std::to_string(n) + ".bmp");
        if (!fs::exists(pl) || !fs::exists(pr)) { rec.reason = "file missing"; records.push_back(rec); continue; }
        cv::Mat l = cv::imread(pl.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat r = cv::imread(pr.string(), cv::IMREAD_GRAYSCALE);
        if (l.empty() || r.empty()) { rec.reason = "imread failed"; ++loadFail; records.push_back(rec); continue; }
        if (l.size() != cv::Size(cfg.imgW, cfg.imgH) || r.size() != cv::Size(cfg.imgW, cfg.imgH)) {
            rec.reason = "size mismatch"; ++sizeMismatch; records.push_back(rec); continue;
        }
        rec.loaded = true;
        rec.grayIdx = static_cast<int>(grayL.size());
        grayL.push_back(std::move(l));
        grayR.push_back(std::move(r));
        records.push_back(rec);
    }
    ASSERT_GE(static_cast<int>(grayL.size()), cfg.gateMinDetected)
        << "loadable pairs=" << grayL.size();
    EXPECT_EQ(sizeMismatch, 0) << "all loaded frames must be configured image size";
    std::cout << "[chain] loaded " << grayL.size() << "/" << cfg.maxFrameNum
              << " pairs (loadFail=" << loadFail << ")\n";

    // ============ 1. 角点提取 + L/R 顺序归一 ============
    ChessboardCornerParams cp;
    cp.patternSize = cv::Size(cfg.chessW, cfg.chessH);
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    int firstDetFrame = -1, firstDetGrayIdx = -1;  // 首个检出帧（可视化用）
    for (auto& rec : records) {
        if (!rec.loaded) continue;
        const int i = rec.grayIdx;
        ChessboardCornerResult rl, rr;
        if (!extractChessboardCorners(grayL[i], cp, rl) || !rl.found) {
            rec.reason = "L corner not found"; continue;
        }
        rec.lOk = true;
        if (!extractChessboardCorners(grayR[i], cp, rr) || !rr.found) {
            rec.reason = "R corner not found"; continue;
        }
        rec.rOk = true;
        if (!normalizeLRCornerOrder(rl, rr)) {
            rec.reason = "normalizeLRCornerOrder failed"; continue;
        }
        rec.normalized = true;
        if (firstDetFrame < 0) {
            firstDetFrame = rec.idx;
            firstDetGrayIdx = i;
        }
        lpts.push_back(std::move(rl.corners));
        rpts.push_back(std::move(rr.corners));
    }
    const int nDetected = static_cast<int>(lpts.size());
    std::cout << "[chain] corner detected pairs=" << nDetected << "/" << grayL.size() << "\n";
    ASSERT_GE(nDetected, cfg.gateMinDetected) << "detected pairs=" << nDetected;

    // ============ 2. 内参 ============
    IntrinsicCalibParams ip;
    ip.chessboard_width = cfg.chessW;
    ip.chessboard_height = cfg.chessH;
    ip.square_size_mm = cfg.squareMm;
    ip.image_width = cfg.imgW;
    ip.image_height = cfg.imgH;
    ip.use_calibrateCameraRO = cfg.useCalibCameraRO;
    ip.calib_flags = cfg.intrinsicFlags;
    ip.reproj_error_threshold = cfg.intrinsicLogReproj;  // 仅日志口径（算子内部不门禁）
    ip.temperature_coeff = cfg.plateTempCoeff;
    ip.plate_temp = cfg.plateTemp;
    IntrinsicCalibCPU intrin(ip);
    IntrinsicCalibResult intrinRes;
    ASSERT_TRUE(intrin.Execute(lpts, rpts, intrinRes)) << "intrinsic Execute returned false";
    ASSERT_TRUE(intrinRes.success) << "intrinsic failed: " << intrinRes.message;

    const cv::Mat& KL = intrinRes.left.camera_matrix;
    const cv::Mat& KR = intrinRes.right.camera_matrix;
    const double fxL = KL.at<double>(0, 0), fyL = KL.at<double>(1, 1);
    const double fxR = KR.at<double>(0, 0), fyR = KR.at<double>(1, 1);
    const double cxL = KL.at<double>(0, 2), cyL = KL.at<double>(1, 2);
    const double cxR = KR.at<double>(0, 2), cyR = KR.at<double>(1, 2);
    std::cout << "[chain] intrinsic rms L=" << intrinRes.left.rms_error
              << " R=" << intrinRes.right.rms_error
              << " reproj_mean=" << intrinRes.reproj_error_mean << " px\n";
    std::cout << "[chain] fxL=" << fxL << " fyL=" << fyL << " cxL=" << cxL << " cyL=" << cyL << "\n";
    std::cout << "[chain] fxR=" << fxR << " fyR=" << fyR << " cxR=" << cxR << " cyR=" << cyR << "\n";

    // ============ 3. 外参 ============
    ExtrinsicCalibCpuParams ep;
    ep.leftPointsPerView = lpts;
    ep.rightPointsPerView = rpts;
    ep.imageSize = cv::Size(cfg.imgW, cfg.imgH);
    ep.patternSize = cv::Size(cfg.chessW, cfg.chessH);
    ep.squareSize = static_cast<float>(cfg.squareMm);
    ep.maxReprojError = cfg.extrinsicMaxReproj;
    ep.minViewCount = cfg.extrinsicMinViewCount;
    const double actualSize = cfg.squareMm * (1.0 + cfg.plateTempCoeff * (cfg.plateTemp - 20.0));  // 板热膨胀
    for (int i = 0; i < cfg.chessH; ++i)
        for (int j = 0; j < cfg.chessW; ++j)
            ep.objectPoints.emplace_back(
                static_cast<float>(j * actualSize),
                static_cast<float>(i * actualSize), 0.0f);
    ExtrinsicCalibCpu extrin(ep);
    ExtrinsicCalibCpuResult extrinRes = extrin.Execute(KL, intrinRes.left.dist_coeffs,
                                                        KR, intrinRes.right.dist_coeffs);
    ASSERT_TRUE(extrinRes.success) << "extrinsic failed: " << extrinRes.message;

    cv::Mat R64; extrinRes.R.convertTo(R64, CV_64F);
    cv::Mat T64; extrinRes.T.convertTo(T64, CV_64F);
    const double orthoErr = matMaxAbs(R64 * R64.t() - cv::Mat::eye(3, 3, CV_64F));
    const double baselineMm = cv::norm(T64);
    std::cout << "[chain] extrinsic stereoReproj=" << extrinRes.stereoReprojError
              << " epipolarMean=" << extrinRes.epipolarErrorMean
              << " baseline=" << baselineMm << " mm ortho=" << orthoErr << "\n";

    // ============ 4. 立体矫正 ============
    StereoRectifyCpuParams rp;
    rp.cameraMatrixL = KL;
    rp.distCoeffsL = intrinRes.left.dist_coeffs;
    rp.cameraMatrixR = KR;
    rp.distCoeffsR = intrinRes.right.dist_coeffs;
    rp.imageSize = cv::Size(cfg.imgW, cfg.imgH);
    rp.R = extrinRes.R;
    rp.T = extrinRes.T;
    rp.alpha = cfg.rectifyAlpha;
    rp.flags = cfg.rectifyFlags;
    StereoRectifyCpu rectify(rp);
    StereoRectifyCpuResult rectifyRes = rectify.Execute();
    ASSERT_TRUE(rectifyRes.success) << "rectify failed: " << rectifyRes.message;

    const double roiLRatio = rectifyRes.validRoiLeft.area() / static_cast<double>(cfg.imgW * cfg.imgH);
    const double roiRRatio = rectifyRes.validRoiRight.area() / static_cast<double>(cfg.imgW * cfg.imgH);
    std::cout << "[chain] rectify validRoiL=" << rectifyRes.validRoiLeft
              << " (" << roiLRatio << ") validRoiR=" << rectifyRes.validRoiRight
              << " (" << roiRRatio << ")\n";

    // 极线自检：角点对映射到矫正面，逐对 |yL - yR|
    std::vector<double> epi;
    for (size_t f = 0; f < lpts.size(); ++f) {
        std::vector<cv::Point2f> lu, ru;
        cv::undistortPoints(lpts[f], lu, KL, intrinRes.left.dist_coeffs,
                            rectifyRes.R1, rectifyRes.P1);
        cv::undistortPoints(rpts[f], ru, KR, intrinRes.right.dist_coeffs,
                            rectifyRes.R2, rectifyRes.P2);
        for (size_t i = 0; i < lu.size(); ++i)
            epi.push_back(std::abs(lu[i].y - ru[i].y));
    }
    const double epiMedian = percentile(epi, 0.5);
    const double epiP95 = percentile(epi, 0.95);
    const double epiMax = percentile(epi, 1.0);
    std::cout << "[chain] epipolar |yL-yR| samples=" << epi.size()
              << " median=" << epiMedian << " p95=" << epiP95 << " max=" << epiMax << " px\n";

    // ============ 5. 温度补偿四表 ============
    CameraIntrinsics cL{fxL, fyL, cxL, cyL, cfg.refTemp};
    CameraIntrinsics cR{fxR, fyR, cxR, cyR, cfg.refTemp};
    IntrinsicCompensateCPUParams icp;
    icp.cte = cfg.cte; icp.tempStep = cfg.tempStep;
    icp.tempRangeMin = cfg.tempMin; icp.tempRangeMax = cfg.tempMax;
    IntrinsicCompensateCPU icomp(icp);
    IntrinsicCompensateCPUResult tableL = icomp.Execute(cL);
    IntrinsicCompensateCPUResult tableR = icomp.Execute(cR);
    ASSERT_TRUE(tableL.success) << tableL.message;
    ASSERT_TRUE(tableR.success) << tableR.message;

    CameraExtrinsics ce;
    ce.referenceTemp = cfg.refTemp;
    for (int i = 0; i < 3; ++i) ce.T[i] = T64.at<double>(i);
    for (int i = 0; i < 9; ++i) ce.R[i] = R64.at<double>(i / 3, i % 3);
    ExtrinsicCompensateCPUParams ecp;
    ecp.cte = cfg.cte; ecp.tempStep = cfg.tempStep;
    ecp.tempRangeMin = cfg.tempMin; ecp.tempRangeMax = cfg.tempMax;
    ExtrinsicCompensateCPU ecomp(ecp);
    ExtrinsicCompensateCPUResult tableE = ecomp.Execute(ce);
    ASSERT_TRUE(tableE.success) << tableE.message;

    StereoRectifyTempTableParams strp;
    strp.cameraMatrixL = rp.cameraMatrixL; strp.distCoeffsL = rp.distCoeffsL;
    strp.cameraMatrixR = rp.cameraMatrixR; strp.distCoeffsR = rp.distCoeffsR;
    strp.imageSize = rp.imageSize; strp.R = rp.R; strp.T = rp.T;
    strp.referenceTemp = cfg.refTemp; strp.cte = cfg.cte;
    strp.tempStep = cfg.tempStep; strp.tempRangeMin = cfg.tempMin; strp.tempRangeMax = cfg.tempMax;
    strp.alpha = cfg.rectifyAlpha; strp.flags = cfg.rectifyFlags;
    StereoRectifyTempTableCpu strtab(strp);
    StereoRectifyTempTableResult tableR2 = strtab.Execute();
    ASSERT_TRUE(tableR2.success) << tableR2.message;

    // 温表数值检查（左表为代表：锚点/单调；矫正表：档数/Q 漂移）
    const CompensatedEntry* anchor = nullptr;
    bool fxMonotonic = true;
    for (size_t i = 0; i < tableL.table.size(); ++i) {
        const auto& e = tableL.table[i];
        if (std::abs(e.deltaT) < 1e-12) anchor = &e;
        if (i > 0 && e.fx < tableL.table[i - 1].fx - 1e-6) fxMonotonic = false;
    }
    ASSERT_NE(anchor, nullptr) << "no reference-temp anchor in intrinsic temp table";
    ASSERT_FALSE(tableR2.table.empty()) << "rectify temp table empty";
    const double qFirst = tableR2.table.front().Q.at<double>(3, 2);
    const double qLast = tableR2.table.back().Q.at<double>(3, 2);
    std::cout << "[chain] temp tables tiers L=" << tableL.table.size()
              << " R=" << tableR.table.size() << " E=" << tableE.table.size()
              << " rect=" << tableR2.table.size()
              << " anchorFx=" << anchor->fx << " fxMonotonic=" << fxMonotonic
              << " Q32 drift=" << (qLast - qFirst) << "\n";

    // ============ 6. 门禁断言 ============
    // 内参
    EXPECT_LT(intrinRes.reproj_error_mean, cfg.gateReprojMean);
    EXPECT_TRUE(intrinRes.left.isValid() && intrinRes.right.isValid());
    EXPECT_NEAR(fxL / fxR, 1.0, cfg.gateFxDiff) << "L/R fx relative diff";
    EXPECT_NEAR(fyL / fyR, 1.0, cfg.gateFxDiff) << "L/R fy relative diff";
    EXPECT_GT(cxL, 0.0);
    EXPECT_LT(cxL, cfg.imgW);
    EXPECT_GT(cyL, 0.0);
    EXPECT_LT(cyL, cfg.imgH);
    EXPECT_GT(cxR, 0.0);
    EXPECT_LT(cxR, cfg.imgW);
    EXPECT_GT(cyR, 0.0);
    EXPECT_LT(cyR, cfg.imgH);
    // 外参
    EXPECT_LT(orthoErr, cfg.gateROrtho);
    EXPECT_GE(baselineMm, cfg.gateBaselineMin);
    EXPECT_LE(baselineMm, cfg.gateBaselineMax);
    // 矫正
    EXPECT_GT(roiLRatio, cfg.gateRoiAreaRatio) << "validRoiLeft too small";
    EXPECT_GT(roiRRatio, cfg.gateRoiAreaRatio) << "validRoiRight too small";
    // 极线自检
    EXPECT_LT(epiMedian, cfg.gateEpiMedian);
    EXPECT_LT(epiP95, cfg.gateEpiP95);
    // 温度表
    EXPECT_EQ(tableL.table.size(), static_cast<size_t>(cfg.expectedTiers));
    EXPECT_EQ(tableR.table.size(), static_cast<size_t>(cfg.expectedTiers));
    EXPECT_EQ(tableE.table.size(), static_cast<size_t>(cfg.expectedTiers));
    EXPECT_EQ(tableR2.table.size(), static_cast<size_t>(cfg.expectedTiers));
    EXPECT_NEAR(anchor->fx / tableL.referenceIntrinsics.fx, 1.0, 1e-9) << "anchor fx restore";
    EXPECT_TRUE(fxMonotonic) << "fx must be monotonic in temperature";
    EXPECT_GT(std::abs(qLast - qFirst), 0.0) << "Q[3][2] must drift with temperature";

    // ============ 7. 分析产物 ============
    // 7.1 analysis.json
    nlohmann::json jFrames = nlohmann::json::array();
    for (const auto& rec : records) {
        jFrames.push_back({
            {"frame", rec.idx}, {"loaded", rec.loaded},
            {"cornerL", rec.lOk}, {"cornerR", rec.rOk},
            {"normalized", rec.normalized}, {"reason", rec.reason},
        });
    }
    nlohmann::json j;
    j["config"] = {{"source", cfgLoaded ? cfgPath.string() : "built-in defaults"},
                   {"fileLoaded", cfgLoaded},
                   {"params", cfg.toJson()}};
    j["dataset"] = {{"dir", dataDir.string()},
                    {"pattern", std::to_string(cfg.chessW) + "x" + std::to_string(cfg.chessH)},
                    {"squareMm", cfg.squareMm}, {"referenceTemp", cfg.refTemp},
                    {"pairsOnDisk", cfg.maxFrameNum},
                    {"pairsLoaded", grayL.size()},
                    {"pairsDetected", nDetected}};
    j["frames"] = jFrames;
    j["intrinsic"] = intrinRes.toJson();
    j["extrinsic"] = extrinRes.toJson();
    j["rectify"] = rectifyRes.toJson();
    j["epipolarCheck"] = {{"samples", epi.size()}, {"median", epiMedian},
                          {"p95", epiP95}, {"max", epiMax}};
    j["tempTableIntrinsicL"] = tableL.toJson();
    j["tempTableIntrinsicR"] = tableR.toJson();
    j["tempTableExtrinsic"] = tableE.toJson();
    j["tempTableRectify"] = tableR2.toJson();
    j["gates"] = {
        {"detectedPairs", {{"value", nDetected}, {"min", cfg.gateMinDetected}}},
        {"reprojMean", {{"value", intrinRes.reproj_error_mean}, {"max", cfg.gateReprojMean}}},
        {"fxRatioLR", {{"value", fxL / fxR}, {"tol", cfg.gateFxDiff}}},
        {"rotOrtho", {{"value", orthoErr}, {"max", cfg.gateROrtho}}},
        {"baselineMm", {{"value", baselineMm},
                        {"range", {cfg.gateBaselineMin, cfg.gateBaselineMax}}}},
        {"roiAreaRatioL", roiLRatio}, {"roiAreaRatioR", roiRRatio},
        {"epiMedian", {{"value", epiMedian}, {"max", cfg.gateEpiMedian}}},
        {"epiP95", {{"value", epiP95}, {"max", cfg.gateEpiP95}}},
        {"tempTiers", {{"value", tableL.table.size()}, {"expect", cfg.expectedTiers}}},
        {"fxMonotonic", fxMonotonic},
        {"q32Drift", qLast - qFirst},
    };
    {
        std::ofstream f((outDir / "analysis.json").string());
        f << j.dump(2);
        EXPECT_TRUE(f.good()) << "analysis.json write failed";
    }

    // 7.2 handoff 输出：与 camera_calib.exe 同 schema（复用 fc1_io 组装，模块2 laser_calib 可直接消费）
    {
        CameraCalibConfig cc;
        cc.chessWidth = cfg.chessW;
        cc.chessHeight = cfg.chessH;
        cc.squareSizeMm = cfg.squareMm;
        cc.imageWidth = cfg.imgW;
        cc.imageHeight = cfg.imgH;
        cc.intrinsicFlags = cfg.intrinsicFlags;
        cc.useCalibrateCameraRO = cfg.useCalibCameraRO;
        cc.reprojErrorThreshold = cfg.intrinsicLogReproj;
        cc.cte = cfg.cte;
        cc.referenceTemp = cfg.refTemp;
        cc.tempRangeMin = cfg.tempMin;
        cc.tempRangeMax = cfg.tempMax;
        cc.tempStep = cfg.tempStep;
        cc.rectifyAlpha = cfg.rectifyAlpha;
        cc.rectifyFlags = cfg.rectifyFlags;
        cc.plateTempCoeff = cfg.plateTempCoeff;
        cc.plateTemp = cfg.plateTemp;
        nlohmann::json hj = buildCameraCalibJson(cc, intrinRes, extrinRes, rectifyRes,
                                                 tableL, tableR, tableE, tableR2);
        const fs::path handoffPath = outDir / "camera_calib.json";
        EXPECT_TRUE(writeJson(handoffPath.string(), hj)) << "handoff write failed";
        std::cout << "[chain] handoff written to " << handoffPath.string() << "\n";
    }

    // 7.3 report.txt（人读汇总）
    {
        std::ofstream f((outDir / "report.txt").string());
        f << std::fixed << std::setprecision(6);
        f << "CAMERA261003 full chain analysis\n";
        f << "================================\n";
        f << "data            : " << dataDir.string() << "\n";
        f << "config          : " << (cfgLoaded ? cfgPath.string() : "BUILT-IN DEFAULTS") << "\n";
        f << "pattern         : " << cfg.chessW << "x" << cfg.chessH << ", square " << cfg.squareMm << " mm\n";
        f << "pairs loaded    : " << grayL.size() << " / " << cfg.maxFrameNum << "\n";
        f << "pairs detected  : " << nDetected << " (gate >= " << cfg.gateMinDetected << ")\n";
        for (const auto& rec : records)
            if (!rec.reason.empty())
                f << "  skipped frame " << rec.idx << ": " << rec.reason << "\n";
        f << "intrinsic reproj_mean : " << intrinRes.reproj_error_mean << " px (gate < " << cfg.gateReprojMean << ")\n";
        f << "  rms L/R             : " << intrinRes.left.rms_error << " / " << intrinRes.right.rms_error << "\n";
        f << "  fx  L/R             : " << fxL << " / " << fxR << "\n";
        f << "  c   L               : (" << cxL << ", " << cyL << ")\n";
        f << "  c   R               : (" << cxR << ", " << cyR << ")\n";
        f << "extrinsic stereoReproj: " << extrinRes.stereoReprojError << "\n";
        f << "  epipolar mean/std   : " << extrinRes.epipolarErrorMean << " / " << extrinRes.epipolarErrorStd << "\n";
        f << "  baseline            : " << baselineMm << " mm\n";
        f << "rectify roi L/R ratio : " << roiLRatio << " / " << roiRRatio << "\n";
        f << "epipolar |yL-yR| med/p95/max : " << epiMedian << " / " << epiP95 << " / " << epiMax << " px\n";
        f << "temp tables tiers     : " << tableL.table.size() << " (expect " << cfg.expectedTiers << ")\n";
        f << "  anchor fx (deltaT=0): " << anchor->fx << " (ref " << tableL.referenceIntrinsics.fx << ")\n";
        f << "  fx monotonic        : " << (fxMonotonic ? "yes" : "NO") << "\n";
        f << "  Q[3][2] drift       : " << (qLast - qFirst) << "\n";
        f << "artifacts             : analysis.json, camera_calib.json (handoff), corners_frameXX.png, rectified_frameXX.png\n";
        EXPECT_TRUE(f.good()) << "report.txt write failed";
    }

    // 7.4 可视化 PNG（首个检出帧）
    ASSERT_GE(firstDetGrayIdx, 0);
    const int visFrame = firstDetFrame;                 // 帧号
    {
        cv::Mat color;
        cv::cvtColor(grayL[firstDetGrayIdx], color, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(color, cv::Size(cfg.chessW, cfg.chessH), lpts[0], true);
        EXPECT_TRUE(cv::imwrite((outDir / ("corners_frame" + std::to_string(visFrame) + ".png")).string(), color))
            << "corners png write failed";
    }
    {
        cv::Mat m1, m2, recL, recR, pair;
        cv::initUndistortRectifyMap(KL, intrinRes.left.dist_coeffs, rectifyRes.R1,
                                    rectifyRes.P1, cv::Size(cfg.imgW, cfg.imgH), CV_32FC1, m1, m2);
        cv::remap(grayL[firstDetGrayIdx], recL, m1, m2, cv::INTER_LINEAR);
        cv::initUndistortRectifyMap(KR, intrinRes.right.dist_coeffs, rectifyRes.R2,
                                    rectifyRes.P2, cv::Size(cfg.imgW, cfg.imgH), CV_32FC1, m1, m2);
        cv::remap(grayR[firstDetGrayIdx], recR, m1, m2, cv::INTER_LINEAR);
        cv::vconcat(recL, recR, pair);
        cv::cvtColor(pair, pair, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < pair.rows; y += 128)
            cv::line(pair, cv::Point(0, y), cv::Point(pair.cols - 1, y),
                     cv::Scalar(0, 0, 255), 1);
        EXPECT_TRUE(cv::imwrite((outDir / ("rectified_frame" + std::to_string(visFrame) + ".png")).string(), pair))
            << "rectified png write failed";
    }
    std::cout << "[chain] artifacts written to " << outDir.string() << "\n";
}
