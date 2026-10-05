// test_golden_ref_tier.cpp — Task 5: 参考档金标准逐字节对拍（数据缺失 SKIP）
//
// 对象: CurveMapTempTableGenerator 参考档路径（硬规则 #4 零插值＋原始曲线不重标）:
//       单条目源表（参考温各一条）→ Generate → CmttReader 载入 sidecarBytes →
//       tierBytes(referenceIndex()) 取参考档 blob，与 fcstepdump --curve-map
//       落盘金标准（factory_calib/data_out/curve_map_table.bin）逐字节比对。
// 金标准口径（fcstepdump.cpp runCurveMap 实读字段，主工作区只读核对）:
//   curves   ← laser_calib.json pjc.emissionCurves[].coeffs[0..5]（诊断字段 0）
//   f/pp     ← pjc.f/pjc.principalPoint——其本身由 PJC 以 P1(0,0)/P1(0,2)/P1(1,2)
//              派生（LaserChain.cpp:443-444），故本测试走插值器 P1 派生路径与
//              金标准同源同位（hasPjcF 时附口径对照诊断断言）
//   B        ← |P2(0,3)|/f（P2 同在 rectify 节）
//   imageSize← camera JSON 顶层 [w,h] 数组（loadCameraCalibHandoff 同款）
//   roi      ← rectify.validRoiLeft;   virtualT ← pjc.projectorT
//   rowStep  ← pjc.epipolarRowStep（缺省 0.7）
//   depth    ← [100,700]（CurveMapParams 默认；金标准头 52B 实测一致）
//   参考温    ← pjc.referenceTemp → camera referenceTemp → 25.0（fcstepdump 未用，
//              不影响档 blob 字节，仅源表/参数簿记一致）
// 数据探测: 根目录候选 = {JMW_CMTT_GOLDEN_ROOT 环境变量, CWD 相对上溯若干级,
//       E:/JEAMMWARE260705}，其下 factory_calib/data_out/ 三件
//       （curve_map_table.bin / laser_calib.json / camera_calib.json）
//       任一缺失 → SKIP（模式参照 factory_calib test_pose06_lid_vote）。
// 期望: 本机无数据 → RefTierByteIdentical SKIP ＋ AssemblySanitySynthetic PASS；
//       产线/真实数据机器上（三件齐备）期望 RefTierByteIdentical PASS。

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../curve_map_temp_table.h"

using namespace calib;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// ============================================================================
// 数据探测（缺失即 SKIP，pose06 风格候选目录扫描）
// ============================================================================

struct GoldenPaths {
    fs::path bin;      // 金标准 curve_map_table.bin（fcstepdump --curve-map 落盘）
    fs::path laser;    // laser_calib.json
    fs::path camera;   // camera_calib.json
};

bool findGolden(GoldenPaths& out) {
    std::vector<fs::path> roots;
    if (const char* env = std::getenv("JMW_CMTT_GOLDEN_ROOT"))
        if (*env) roots.emplace_back(env);
    for (const char* rel : {"../../..", "../../../..", "../.."})
        roots.push_back(fs::absolute(rel).lexically_normal());
    roots.emplace_back("E:/JEAMMWARE260705");
    for (const auto& root : roots) {
        const fs::path d = root / "factory_calib" / "data_out";
        const GoldenPaths p{d / "curve_map_table.bin", d / "laser_calib.json",
                            d / "camera_calib.json"};
        if (fs::is_regular_file(p.bin) && fs::is_regular_file(p.laser) &&
            fs::is_regular_file(p.camera)) {
            out = p;
            return true;
        }
    }
    return false;
}

// ============================================================================
// 输入组装（金标准口径逐字段复刻，见文件头；不做任何数值变换）
// ============================================================================

struct GoldenSources {
    std::vector<ImplicitCurve> curves;   // pjc.emissionCurves（保持 JSON 序）
    cv::Mat P1;                          // rectify.P1（3x4 CV_64F）
    cv::Mat P2;                          // rectify.P2（3x4 CV_64F）
    cv::Rect roi{0, 0, 0, 0};            // rectify.validRoiLeft
    cv::Size imageSize{0, 0};            // camera JSON 顶层 [w, h]
    cv::Vec3d virtualT{0, 0, 0};         // pjc.projectorT
    float rowStep = 0.7f;                // pjc.epipolarRowStep（缺省 0.7）
    double referenceTemp = 25.0;         // pjc.referenceTemp → camera → 25.0
    bool hasPjcF = false;                // 口径对照诊断（fcstepdump 直读 pjc.f/pp）
    double pjcF = 0.0;
    cv::Point2d pjcPP{0.0, 0.0};
};

[[noreturn]] void failField(const std::string& what, const fs::path& p) {
    throw std::runtime_error(what + " (in " + p.string() + ")");
}

json readJson(const fs::path& p) {
    std::ifstream f(p);
    if (!f) failField("cannot open", p);
    try {
        return json::parse(f);
    } catch (const std::exception& e) {
        failField(std::string("parse failed: ") + e.what(), p);
    }
}

cv::Mat parseMat64F(const json& j, const char* what, const fs::path& p) {
    if (!j.is_array() || j.empty() || !j[0].is_array())
        failField(std::string(what) + ": expected 2D array", p);
    const int rows = static_cast<int>(j.size());
    const int cols = static_cast<int>(j[0].size());
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r) {
        if (!j[r].is_array() || static_cast<int>(j[r].size()) != cols)
            failField(std::string(what) + ": ragged rows", p);
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = j[r][c].get<double>();
    }
    return m;
}

GoldenSources assembleSources(const fs::path& cameraJson, const fs::path& laserJson) {
    const json cam = readJson(cameraJson);
    const json las = readJson(laserJson);
    GoldenSources s;

    // --- camera_calib.json（字段口径 = loadCameraCalibHandoff 实读） ---
    if (!cam.contains("imageSize") || !cam["imageSize"].is_array() ||
        cam["imageSize"].size() < 2)
        failField("imageSize: expected top-level [w,h] array", cameraJson);
    s.imageSize = cv::Size(cam["imageSize"][0].get<int>(),
                           cam["imageSize"][1].get<int>());
    s.referenceTemp = cam.value("referenceTemp", 25.0);
    const json& rec = cam.at("rectify");
    s.P1 = parseMat64F(rec.at("P1"), "rectify.P1", cameraJson);
    s.P2 = parseMat64F(rec.at("P2"), "rectify.P2", cameraJson);
    const json& vr = rec.at("validRoiLeft");
    s.roi = cv::Rect(vr.at("x").get<int>(), vr.at("y").get<int>(),
                     vr.at("w").get<int>(), vr.at("h").get<int>());

    // --- laser_calib.json（fcstepdump/fcscan 实读字段） ---
    const json& pjc = las.at("pjc");
    for (const auto& cj : pjc.at("emissionCurves")) {
        const json& co = cj.at("coeffs");
        ImplicitCurve c;   // 诊断字段（discriminant/sampsonRms/pointCount）置 0
        for (int k = 0; k < 6; ++k)
            c.coeffs[k] = co.at(k).get<double>();
        s.curves.push_back(c);
    }
    for (int i = 0; i < 3; ++i)
        s.virtualT[i] = pjc.at("projectorT").at(i).get<double>();
    s.rowStep = pjc.value("epipolarRowStep", 0.7f);
    if (pjc.contains("referenceTemp"))
        s.referenceTemp = pjc.at("referenceTemp").get<double>();
    if (pjc.contains("f") && pjc.contains("principalPoint")) {
        s.hasPjcF = true;
        s.pjcF = pjc.at("f").get<double>();
        s.pjcPP = cv::Point2d(pjc.at("principalPoint")[0].get<double>(),
                              pjc.at("principalPoint")[1].get<double>());
    }
    return s;
}

// ============================================================================
// 单条目源表（参考温各一条）
// 注: rightResult 留空——TempParamInterpolator 构造校验只读 leftResult
// （temp_param_interpolator.cpp:78），空右表被容忍。
// ============================================================================

StereoRectifyTempTableResult makeRefRectifyTable(const GoldenSources& s) {
    StereoRectifyTempTableResult r;
    r.success = true;
    r.referenceTemp = s.referenceTemp;
    r.tableSize = 1;
    StereoRectifyTempEntry e;
    e.temperature = s.referenceTemp;
    e.P1 = s.P1;
    e.P2 = s.P2;
    e.validRoiLeft = s.roi;
    r.table.push_back(std::move(e));
    return r;
}

LaserExtrinsicCompensateCPUResult makeRefLaserTable(const GoldenSources& s) {
    LaserExtrinsicCompensateCPUResult l;
    l.success = true;
    l.referenceTemp = s.referenceTemp;
    l.leftResult.success = true;
    l.leftResult.referenceTemp = s.referenceTemp;
    ExtrinsicCompensatedEntry e;
    e.temperature = s.referenceTemp;
    e.T[0] = s.virtualT[0];
    e.T[1] = s.virtualT[1];
    e.T[2] = s.virtualT[2];
    l.leftResult.table.push_back(std::move(e));
    return l;
}

CurveMapTempTableGenParams makeGenParams(const GoldenSources& s) {
    CurveMapTempTableGenParams p;
    p.referenceTemp = s.referenceTemp;
    p.tempHalfRange = 0.0f;   // 仅参考档（单 tier）
    p.tempStep = 0.5f;
    p.rowStep = s.rowStep;
    p.depthMin = 100.0f;      // 金标准头实测口径（= CurveMapParams 默认）
    p.depthMax = 700.0f;
    return p;
}

/// 组装结果 → CurveMapInput（sanity 断言用；派生配方与生成器内部 makeTierInput
/// 同式: f=P1(0,0), pp=(P1(0,2),P1(1,2)), B=|P2(0,3)|/f）
CurveMapInput toCurveMapInput(const GoldenSources& s) {
    CurveMapInput in;
    in.curves = s.curves;
    in.f = s.P1.at<double>(0, 0);
    in.principalPoint = cv::Point2d(s.P1.at<double>(0, 2), s.P1.at<double>(1, 2));
    in.baseline = std::fabs(s.P2.at<double>(0, 3)) / in.f;
    in.imageSize = s.imageSize;
    in.roi = s.roi;
    in.virtualT = s.virtualT;
    in.virtualK = cv::Matx33d(in.f, 0.0, in.principalPoint.x,
                              0.0, in.f, in.principalPoint.y,
                              0.0, 0.0, 1.0);
    return in;
}

std::vector<uint8_t> readBinFile(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
}

std::string hexHead(const uint8_t* p, size_t n) {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; ++i) {
        if (i) os << ' ';
        os << std::setw(2) << static_cast<unsigned>(p[i]);
    }
    return os.str();
}

/// 前 16 字节 hex（钳制到实际字节数——载荷 <16B 时防诊断路径越界读）
std::string hexHead16(const std::vector<uint8_t>& v) {
    return hexHead(v.data(), std::min<size_t>(16, v.size()));
}

/// 临时目录 RAII 守卫（ASSERT 失败提前返回也不泄漏 %TEMP%\jmw_cmtt_golden_*）
struct TempDirGuard {
    fs::path dir;
    ~TempDirGuard() {
        std::error_code ec;
        fs::remove_all(dir, ec);   // 清理失败不抛（析构路径）
    }
};

} // namespace

// ============================================================================
// 金标准逐字节对拍（产线/真实数据机器上期望 PASS；数据缺失 SKIP）
// ============================================================================

TEST(Golden, RefTierByteIdentical) {
    GoldenPaths gp;
    if (!findGolden(gp))
        GTEST_SKIP() << "golden data absent";

    const GoldenSources src = assembleSources(gp.camera, gp.laser);

    // 口径对照诊断: fcstepdump 直读的 pjc.f/pp 与 P1 派生应逐位一致
    // （PJC 输入即由 P1 派生，LaserChain.cpp:443-444）；不一致则对拍必败且原因前置可见
    if (src.hasPjcF) {
        EXPECT_DOUBLE_EQ(src.pjcF, src.P1.at<double>(0, 0));
        EXPECT_DOUBLE_EQ(src.pjcPP.x, src.P1.at<double>(0, 2));
        EXPECT_DOUBLE_EQ(src.pjcPP.y, src.P1.at<double>(1, 2));
    }

    CurveMapTempTableGenerator gen(makeGenParams(src));
    const auto r = gen.Generate(src.curves, makeRefRectifyTable(src),
                                makeRefLaserTable(src), src.imageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{1});   // tempHalfRange=0 → 仅参考档
    ASSERT_NE(r.tiers[0].flags & kCmttFlagReference, 0);

    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));
    const size_t refIdx = r.referenceIndex();
    ASSERT_LT(refIdx, rd.tierCount());
    const auto blob = rd.tierBytes(refIdx);
    ASSERT_TRUE(blob.has_value()) << "reference tier payload missing";
    ASSERT_FALSE(blob->empty());

    const std::vector<uint8_t> golden = readBinFile(gp.bin);
    ASSERT_EQ(blob->size(), golden.size())
        << "length mismatch: ref-tier blob " << blob->size() << " vs golden "
        << golden.size()
        << "\n  blob   head: " << hexHead16(*blob)
        << "\n  golden head: " << hexHead16(golden);
    if (!golden.empty())
        EXPECT_EQ(std::memcmp(blob->data(), golden.data(), golden.size()), 0)
            << "content mismatch at length " << golden.size()
            << "\n  blob   head: " << hexHead16(*blob)
            << "\n  golden head: " << hexHead16(golden);
}

// ============================================================================
// 组装自检（本地可跑，不依赖数据）: 合成最小 JSON → 同一组装函数 → 字段对拍
// ============================================================================

TEST(Golden, AssemblySanitySynthetic) {
    const fs::path dir = fs::temp_directory_path() /
        ("jmw_cmtt_golden_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    const TempDirGuard dirGuard{dir};
    ASSERT_TRUE(fs::create_directories(dir));
    const fs::path camPath = dir / "camera_calib.json";
    const fs::path lasPath = dir / "laser_calib.json";

    {
        json cam;
        cam["schema"] = "factory_calib.camera_calib.v1";
        cam["imageSize"] = {320, 240};
        cam["referenceTemp"] = 22.5;
        json rec;
        rec["P1"] = json::array({{400.5, 0.0, 159.25, 0.0},
                                 {0.0, 401.75, 120.5, 0.0},
                                 {0.0, 0.0, 1.0, 0.0}});
        rec["P2"] = json::array({{400.5, 0.0, 159.25, -12015.0},
                                 {0.0, 401.75, 120.5, 0.0},
                                 {0.0, 0.0, 1.0, 0.0}});
        rec["validRoiLeft"] = {{"x", 8}, {"y", 16}, {"w", 300}, {"h", 200}};
        cam["rectify"] = rec;
        std::ofstream(camPath) << cam.dump(2);

        json las;
        json pjc;
        pjc["emissionCurves"] = json::array(
            {json{{"coeffs", {1.5e-3, 2.5e-4, -3.25e-5, -0.75, 0.125, 42.5}}},
             json{{"coeffs", {0.0, 0.0, 0.0, 1.0, -2.0, -99.75}}}});
        pjc["projectorT"] = {1.25, -2.5, 3.75};
        pjc["epipolarRowStep"] = 0.5;
        pjc["f"] = 400.5;
        pjc["principalPoint"] = {159.25, 120.5};
        las["pjc"] = pjc;
        std::ofstream(lasPath) << las.dump(2);
    }

    const GoldenSources src = assembleSources(camPath, lasPath);
    const CurveMapInput in = toCurveMapInput(src);

    ASSERT_EQ(in.curves.size(), size_t{2});
    const double e0[6] = {1.5e-3, 2.5e-4, -3.25e-5, -0.75, 0.125, 42.5};
    const double e1[6] = {0.0, 0.0, 0.0, 1.0, -2.0, -99.75};
    for (int k = 0; k < 6; ++k) {
        EXPECT_DOUBLE_EQ(in.curves[0].coeffs[k], e0[k]);
        EXPECT_DOUBLE_EQ(in.curves[1].coeffs[k], e1[k]);
    }
    EXPECT_DOUBLE_EQ(in.curves[0].discriminant, 0.0);   // 诊断字段置 0
    EXPECT_DOUBLE_EQ(in.curves[0].sampsonRms, 0.0);
    EXPECT_EQ(in.curves[0].pointCount, 0);

    EXPECT_DOUBLE_EQ(in.f, 400.5);                      // P1(0,0)
    EXPECT_DOUBLE_EQ(in.principalPoint.x, 159.25);      // P1(0,2)
    EXPECT_DOUBLE_EQ(in.principalPoint.y, 120.5);       // P1(1,2)
    EXPECT_DOUBLE_EQ(in.baseline, 30.0);                // |P2(0,3)|/f = 12015/400.5
    EXPECT_EQ(in.imageSize.width, 320);
    EXPECT_EQ(in.imageSize.height, 240);
    EXPECT_EQ(in.roi, cv::Rect(8, 16, 300, 200));
    EXPECT_DOUBLE_EQ(in.virtualT[0], 1.25);
    EXPECT_DOUBLE_EQ(in.virtualT[1], -2.5);
    EXPECT_DOUBLE_EQ(in.virtualT[2], 3.75);
    EXPECT_FLOAT_EQ(src.rowStep, 0.5f);
    EXPECT_DOUBLE_EQ(src.referenceTemp, 22.5);
    EXPECT_DOUBLE_EQ(in.virtualK(0, 0), 400.5);
    EXPECT_DOUBLE_EQ(in.virtualK(0, 2), 159.25);
    EXPECT_DOUBLE_EQ(in.virtualK(1, 1), 400.5);
    EXPECT_DOUBLE_EQ(in.virtualK(1, 2), 120.5);

    // epipolarRowStep 缺省契约: 键缺失 → 0.7（fcstepdump 同款 value 默认）
    {
        json las2 = readJson(lasPath);
        las2["pjc"].erase("epipolarRowStep");
        std::ofstream(lasPath) << las2.dump(2);
        const GoldenSources src2 = assembleSources(camPath, lasPath);
        EXPECT_FLOAT_EQ(src2.rowStep, 0.7f);
    }
}
