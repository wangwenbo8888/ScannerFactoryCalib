// test_curve_map_cuda_baseline_depth.cpp — curve_map_cuda 基线景深窗黄金测试
//
// 设计文档: docs/plans/2026-10-09-curve-map-cuda-baseline-depth-design.md
// 概念: 基线景深 = 矫正系 z 坐标（平面1=两光轴所在平面 y=0；景深面 ⊥ 平面1 且 ∥ 基线
//       ⟺ z=const；到基线的投影距离 = z）——即现行 depthMin/Max 的口径（§2 推导）。
//
// 合成组场景构造四红线（§4.2，全部由测试自证）:
//   1. 直线型 conic（A=B=C=0）——CPU/CUDA 同走 |a|<1e-18 单根分支；且 ⇒ 每(行,uL)单根
//      ⇒ uLq 严格递增 ⇒ (uLq,dInt) 碰撞不可能 ⇒ CPU 去重分支恒空操作（§6 分歧绕开）；
//   2. rowStep=0.5（2 的幂）——invRowStep=2.0 精确且 u+=0.5 二进制精确步进，零舍入漂移；
//   3. 场景自证——镜像 sweep（复刻 kernel 公式、仅去 Z 窗）断言扫深跨骑窗口两端；
//   4. makeKey 位域域约束——断言 W·invRowStep≤4095（uLq 12bit）与 nRows·nLines≤65535
//      （bucket 16bit 不回卷；均为既有代码不设防前提，测试侧自查）。
// 真数据组（golden 式根探测，缺席 GTEST_SKIP）: 把 2026-10-09 对
// laser261006-1668（窗 100–500）19,157,130 条的审计固化为断言。

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curve_map_cuda.h"   // CurveMapCuda（含 SetBaselineDepthWindow）
#include "curve_map.h"        // CurveMapGenerator / CurveMapTable
#include "cmtt_container.h"   // CmttReader（可选 sidecar 交叉核对）

using namespace calib;
namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================================
// 合成场景
// ============================================================================

namespace {

constexpr int    kW = 640, kH = 480;
constexpr double kF = 500.0, kCx = 320.0, kCy = 240.0, kBasel = 60.0;
constexpr double kTx = 30.0, kTy = 10.0, kTz = 12.0;
constexpr float  kRowStep = 0.5f;          // 红线 2：2 的幂
constexpr double kD1 = 120.0, kD2 = 400.0; // 基线景深窗

ImplicitCurve lineConic(double D, double E, double F0) {
    ImplicitCurve c;   // A=B=C=0：直线型退化二次曲线（红线 1）
    c.coeffs[0] = 0.0; c.coeffs[1] = 0.0; c.coeffs[2] = 0.0;
    c.coeffs[3] = D;   c.coeffs[4] = E;   c.coeffs[5] = F0;
    return c;
}

CurveMapInput makeSynthInput() {
    CurveMapInput in;
    in.curves = { lineConic(1.0, -0.3, -50.0),    // F1: u − 0.3v − 50 = 0
                  lineConic(1.0,  0.2, -500.0) }; // F2: u + 0.2v − 500 = 0
    in.f = kF;
    in.principalPoint = cv::Point2d(kCx, kCy);
    in.baseline = kBasel;
    in.imageSize = cv::Size(kW, kH);
    in.roi = cv::Rect(16, 16, 608, 448);
    in.virtualT = cv::Vec3d(kTx, kTy, kTz);
    in.virtualK = cv::Matx33d(kF, 0.0, kCx, 0.0, kF, kCy, 0.0, 0.0, 1.0);
    return in;
}

// ---- 镜像求交（复刻 kernel 过滤链，仅去 Z 窗；直线 conic 单根路径） ----
struct MirrorCtx {
    double f, cx, cy, tx, ty, tz, B, rowStep, invRowStep;
    int W, H;
};

bool mirrorHit(const ImplicitCurve& c, double uLpx, double v, const MirrorCtx& m,
               double& Zout) {
    const double A = c.coeffs[0], Bc = c.coeffs[1], C = c.coeffs[2];
    const double D = c.coeffs[3], E = c.coeffs[4], F0 = c.coeffs[5];
    const double Ex = m.f * m.tx / m.tz + m.cx;
    const double Ey = m.f * m.ty / m.tz + m.cy;
    const double du = Ex - uLpx, dv = Ey - v;
    const double a  = A*du*du + Bc*du*dv + C*dv*dv;
    const double b  = 2*A*uLpx*du + Bc*(uLpx*dv + v*du) + 2*C*v*dv + D*du + E*dv;
    const double cc = A*uLpx*uLpx + Bc*uLpx*v + C*v*v + D*uLpx + E*v + F0;
    double q_u = 0.0, q_v = 0.0; int nh = 0;
    if (std::fabs(a) < 1e-18) {
        if (std::fabs(b) > 1e-18) {
            const double t = -cc / b;
            q_u = uLpx + t * du; q_v = v + t * dv; nh = 1;
        }
    } else {
        const double disc = b*b - 4*a*cc;
        if (disc >= 0.0) {
            const double sq = std::sqrt(disc);
            const double t1 = (-b + sq) / (2*a);
            q_u = uLpx + t1 * du; q_v = v + t1 * dv; nh = 1;   // 单根口径（本场景 a≡0）
        }
    }
    if (nh == 0) return false;
    if (q_u < 0.0 || q_u >= (double)m.W || q_v < 0.0 || q_v >= (double)m.H) return false;
    const double qc = q_u - m.cx;
    const double denom = qc - (uLpx - m.cx);
    if (std::fabs(denom) < 1e-9) return false;
    const double Z = (qc * m.tz - m.f * m.tx) / denom;
    const double uRpx = uLpx - m.f * m.B / Z;
    if (uRpx < 0.0 || uRpx >= (double)m.W) return false;
    const long uLq = (long)std::llround(uLpx * m.invRowStep);
    const long uRq = (long)std::llround(uRpx * m.invRowStep);
    if (uRq >= uLq) return false;
    Zout = Z;
    return true;
}

struct Sweep {
    double zMin = std::numeric_limits<double>::infinity();
    double zMax = -std::numeric_limits<double>::infinity();
    long   achievablePos = 0;   // Z>0 的可达条目数（诊断）
};

Sweep mirrorSweep(const CurveMapInput& in, float rowStep) {
    MirrorCtx m{kF, kCx, kCy, kTx, kTy, kTz, kBasel,
                (double)rowStep, 1.0 / (double)rowStep, kW, kH};
    const double invRowStep = 1.0 / (double)rowStep;
    const int rowCount = (int)std::round(in.imageSize.height * invRowStep);
    const int rowMin = std::max(0, (int)std::floor(in.roi.y * invRowStep));
    const int rowMax = std::min(rowCount - 1,
        (int)std::ceil((in.roi.y + in.roi.height) * invRowStep));
    const double uMin = (double)in.roi.x;
    const double uMax = (double)(in.roi.x + in.roi.width) - 1.0;
    const double uStep = (double)rowStep;   // effectiveUSampleStep()=rowStep（默认）
    Sweep s;
    for (int r = rowMin; r <= rowMax; ++r) {
        const double v = r * (double)rowStep;
        for (double u = uMin; u <= uMax + 1e-9; u += uStep) {
            for (const auto& c : in.curves) {
                double Z = 0.0;
                if (mirrorHit(c, u, v, m, Z)) {
                    if (Z > 0.0) {
                        s.zMin = std::min(s.zMin, Z);
                        s.zMax = std::max(s.zMax, Z);
                        ++s.achievablePos;
                    }
                }
            }
        }
    }
    return s;
}

// 场景构造四红线自证（§4.2）
void assertSceneRedlines(const CurveMapInput& in, const Sweep& s) {
    // 红线 4：makeKey 位域域约束
    EXPECT_LE((double)kW * (1.0 / (double)kRowStep), 4095.0) << "uLq 12bit 溢出风险";
    const double invRowStep = 1.0 / (double)kRowStep;
    const int rowCount = (int)std::round(in.imageSize.height * invRowStep);
    const int rowMin = std::max(0, (int)std::floor(in.roi.y * invRowStep));
    const int rowMax = std::min(rowCount - 1,
        (int)std::ceil((in.roi.y + in.roi.height) * invRowStep));
    const long nRows = (long)rowMax - rowMin + 1;
    EXPECT_LE(nRows * (long)in.curves.size(), 65535) << "bucket 16bit 回卷风险";
    // 红线 3：扫深跨骑窗口两端
    EXPECT_LT(s.zMin, kD1) << "扫深未达近端（场景无效）";
    EXPECT_GT(s.zMax, kD2) << "扫深未达远端（场景无效）";
    // 零窗可行前提
    EXPECT_GT(s.zMin, 2.0);
}

struct LineStat {
    long cnt = 0;
    long dMin = std::numeric_limits<long>::max();
    long dMax = 0;
};

// 逐行流式解码并验证窗界（§5.2 精确式；−1 与 0 必须区分）
void decodeAndCheckWindow(const CurveMapResult& r, double d1, double d2,
                          double fB, double gridStep,
                          long& totalOut, std::map<int, LineStat>& perLine) {
    ASSERT_TRUE(r.success) << r.message;
    CurveMapTable t;
    std::string err;
    ASSERT_TRUE(t.Load(r.tableBytes.data(), r.tableBytes.size(), err)) << err;
    // 窗口头回读（窗口写进序列化头）
    EXPECT_EQ(t.depthMin(), (float)d1);
    EXPECT_EQ(t.depthMax(), (float)d2);
    const double zLo = d1 / (1.0 + d1 * gridStep / (2.0 * fB));   // 近端精确下界
    const double zHi = d2 / (1.0 - d2 * gridStep / (2.0 * fB));   // 远端精确上界
    std::vector<CurveMapEntry> buf(262144);
    long total = 0;
    for (uint32_t row = 0; row < t.rowCount(); ++row) {
        const int n = t.EnumerateRow((int)row, buf.data(), (int)buf.size());
        ASSERT_NE(n, -1) << "EnumerateRow cap 不足 @row=" << row;
        for (int i = 0; i < n; ++i) {
            const long d = (long)buf[i].uL - (long)buf[i].uR;   // ≡ dInt（§5.1）
            ASSERT_GE(d, 1);
            const double Z = fB / ((double)d * gridStep);
            ASSERT_GE(Z, zLo - 1e-6)
                << "row=" << row << " lid=" << (int)buf[i].lid
                << " Z=" << Z << " < zLo=" << zLo;
            ASSERT_LE(Z, zHi + 1e-6)
                << "row=" << row << " lid=" << (int)buf[i].lid
                << " Z=" << Z << " > zHi=" << zHi;
            LineStat& st = perLine[(int)buf[i].lid];
            ++st.cnt;
            st.dMin = std::min(st.dMin, d);
            st.dMax = std::max(st.dMax, d);
            ++total;
        }
    }
    EXPECT_EQ(t.entryCount(), (uint32_t)total);   // 解码计数 == 表头 entryCount
    totalOut = total;
}

CurveMapResult runCuda(const CurveMapInput& in, double d1, double d2) {
    CurveMapParams prm;
    prm.epipolarRowStep = kRowStep;
    CurveMapCuda cuda(prm);
    cuda.SetBaselineDepthWindow(d1, d2);
    return cuda.Generate(in);
}

} // namespace

// ============================================================================
// A1 窗内一致性：全条目 Z′ ∈ [d1−δ₁, d2+δ₂]（精确式现算）
// ============================================================================
TEST(BaselineDepthSynthetic, A1_WindowConsistency) {
    const CurveMapInput in = makeSynthInput();
    const Sweep s = mirrorSweep(in, kRowStep);
    assertSceneRedlines(in, s);
    SCOPED_TRACE("achievablePos=" + std::to_string(s.achievablePos) +
                 " zMin=" + std::to_string(s.zMin) + " zMax=" + std::to_string(s.zMax));

    const CurveMapResult r = runCuda(in, kD1, kD2);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_GT(r.entryCount, 0u);

    const double fB = kF * kBasel;   // kernel 操作数序 f * baseline
    long total = 0;
    std::map<int, LineStat> perLine;
    decodeAndCheckWindow(r, kD1, kD2, fB, (double)kRowStep, total, perLine);

    EXPECT_EQ(perLine.size(), (size_t)in.curves.size());   // 每条线都有条目
    for (const auto& kv : perLine)
        EXPECT_GT(kv.second.cnt, 0);
    EXPECT_EQ(r.entryCount, (uint32_t)total);
}

// ============================================================================
// A2 边界剔除＋单调性＋空窗（0 条目 → Warning 且产物合法）
// ============================================================================
TEST(BaselineDepthSynthetic, A2_BoundaryAndEmptyWindow) {
    const CurveMapInput in = makeSynthInput();
    const Sweep s = mirrorSweep(in, kRowStep);
    assertSceneRedlines(in, s);

    const CurveMapResult rWide = runCuda(in, kD1, kD2);
    ASSERT_TRUE(rWide.success) << rWide.message;
    EXPECT_GT(rWide.entryCount, 0u);

    // 窄窗 (150,350) ⊂ (120,400)：条目单调不增＋同样满足窗界
    const CurveMapResult rTight = runCuda(in, 150.0, 350.0);
    ASSERT_TRUE(rTight.success) << rTight.message;
    EXPECT_LE(rTight.entryCount, rWide.entryCount);
    {
        const double fB = kF * kBasel;
        long total = 0;
        std::map<int, LineStat> perLine;
        decodeAndCheckWindow(rTight, 150.0, 350.0, fB, (double)kRowStep, total, perLine);
    }

    // 空窗：可达 Z 全体 > zw2（镜像自证）→ 0 条目 + Warning + 合法空表
    ASSERT_GT(s.zMin, 2.0);
    const double zw2 = s.zMin - 1.0;
    ASSERT_GT(zw2, 1.0);
    const CurveMapResult rEmpty = runCuda(in, 1.0, zw2);
    EXPECT_TRUE(rEmpty.success);
    EXPECT_EQ(rEmpty.entryCount, 0u);
    EXPECT_EQ(static_cast<int>(rEmpty.qualityFlag), static_cast<int>(QualityFlag::Warning));
    {
        CurveMapTable t;
        std::string err;
        ASSERT_TRUE(t.Load(rEmpty.tableBytes.data(), rEmpty.tableBytes.size(), err)) << err;
        EXPECT_EQ(t.entryCount(), 0u);   // 勿用 valid()——空表语义未定
        EXPECT_EQ(t.depthMin(), 1.0f);
        EXPECT_EQ(t.depthMax(), (float)zw2);
    }
}

// ============================================================================
// A3 非法窗口拒跑（NaN/Inf/倒置/非正）＋异常后参数不被半改
// ============================================================================
TEST(BaselineDepthSynthetic, A3_InvalidWindowRejected) {
    CurveMapCuda c;   // 编译默认 depth 100/700
    {
        const auto g = c.GetBaselineDepthWindow();
        EXPECT_EQ(g.first, 100.0);
        EXPECT_EQ(g.second, 700.0);
    }
    c.SetBaselineDepthWindow(kD1, kD2);
    {
        const auto g = c.GetBaselineDepthWindow();
        EXPECT_EQ(g.first, kD1);      // 120/400 均为 float 可精确表示
        EXPECT_EQ(g.second, kD2);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::pair<double, double> bad[] = {
        { kD2, kD1 },          // 倒置
        { 100.0, 100.0 },      // 相等
        { 0.0, 500.0 },        // 非正
        { -5.0, 500.0 },
        { nan, 500.0 },        // NaN 静默放行堵口
        { 500.0, nan },
        { inf, 500.0 },        // Inf
        { 500.0, inf },
        { 1e300, 1e301 },      // 有限 double → 窄化 float inf → validate 拒
        { 1e-300, 500.0 },     // 下溢 float 0 → validate 拒
    };
    for (const auto& w : bad) {
        EXPECT_THROW(c.SetBaselineDepthWindow(w.first, w.second), std::invalid_argument)
            << "w=(" << w.first << "," << w.second << ")";
        // 异常后参数不被半改（唯一可观测判据）
        const auto g = c.GetBaselineDepthWindow();
        EXPECT_EQ(g.first, kD1);
        EXPECT_EQ(g.second, kD2);
    }
}

// ============================================================================
// A4 CPU↔CUDA 同窗对拍：entryCount＋逐条目精确一致
//（直线 conic ⇒ 构造性等价：见文件头红线 1）
// ============================================================================
TEST(BaselineDepthSynthetic, A4_CpuCudaSameWindowEntryIdentical) {
    const CurveMapInput in = makeSynthInput();
    const Sweep s = mirrorSweep(in, kRowStep);
    assertSceneRedlines(in, s);

    CurveMapParams prm;
    prm.epipolarRowStep = kRowStep;
    prm.depthMin = (float)kD1;
    prm.depthMax = (float)kD2;

    CurveMapCuda cuda(prm);
    const CurveMapResult rc = cuda.Generate(in);
    ASSERT_TRUE(rc.success) << rc.message;

    CurveMapGenerator cpu(prm);
    const CurveMapResult rr = cpu.Generate(in);
    ASSERT_TRUE(rr.success) << rr.message;

    EXPECT_EQ(rc.entryCount, rr.entryCount);
    ASSERT_EQ(rc.tableBytes.size(), rr.tableBytes.size());   // 结构等价的前置

    CurveMapTable tc, tr;
    std::string e1, e2;
    ASSERT_TRUE(tc.Load(rc.tableBytes.data(), rc.tableBytes.size(), e1)) << e1;
    ASSERT_TRUE(tr.Load(rr.tableBytes.data(), rr.tableBytes.size(), e2)) << e2;
    ASSERT_EQ(tc.rowCount(), tr.rowCount());
    ASSERT_EQ(tc.entryCount(), tr.entryCount());

    std::vector<CurveMapEntry> bc(65536), br(65536);
    for (uint32_t row = 0; row < tc.rowCount(); ++row) {
        const int nc = tc.EnumerateRow((int)row, bc.data(), (int)bc.size());
        const int nr = tr.EnumerateRow((int)row, br.data(), (int)br.size());
        ASSERT_NE(nc, -1);
        ASSERT_NE(nr, -1);
        ASSERT_EQ(nc, nr) << "row=" << row;
        for (int i = 0; i < nc; ++i) {
            ASSERT_EQ((int)bc[i].lid, (int)br[i].lid) << "row=" << row << " i=" << i;
            ASSERT_EQ(bc[i].uL, br[i].uL) << "row=" << row << " i=" << i;
            ASSERT_EQ(bc[i].uR, br[i].uR) << "row=" << row << " i=" << i;
        }
    }
}

// ============================================================================
// A5 别名等价：SetBaselineDepthWindow ≡ SetParams 直写 depthMin/Max（逐字节一致）
// ============================================================================
TEST(BaselineDepthSynthetic, A5_AliasEquivalence) {
    const CurveMapInput in = makeSynthInput();

    CurveMapParams base;                 // rowStep=0.5，depth 保持默认
    base.epipolarRowStep = kRowStep;
    CurveMapCuda c1(base);
    c1.SetBaselineDepthWindow(kD1, kD2);

    CurveMapParams q = base;             // 同基参数，仅直写 depth 两键
    q.depthMin = (float)kD1;
    q.depthMax = (float)kD2;
    CurveMapCuda c2(q);

    // 参数真相一致（float 口径）
    EXPECT_EQ(c1.GetParams().depthMin, c2.GetParams().depthMin);
    EXPECT_EQ(c1.GetParams().depthMax, c2.GetParams().depthMax);
    {
        const auto g = c1.GetBaselineDepthWindow();   // 回读往返（float 精确值）
        EXPECT_EQ(g.first, kD1);
        EXPECT_EQ(g.second, kD2);
    }

    const CurveMapResult r1 = c1.Generate(in);
    const CurveMapResult r2 = c2.Generate(in);
    ASSERT_TRUE(r1.success) << r1.message;
    ASSERT_TRUE(r2.success) << r2.message;
    ASSERT_EQ(r1.tableBytes.size(), r2.tableBytes.size());
    EXPECT_EQ(0, std::memcmp(r1.tableBytes.data(), r2.tableBytes.data(),
                             r1.tableBytes.size()));
    // effectiveParams 回显（设置的窗就是生效的窗）
    EXPECT_EQ(r1.effectiveParams.depthMin, (float)kD1);
    EXPECT_EQ(r1.effectiveParams.depthMax, (float)kD2);
}

// ============================================================================
// 真数据组：laser261006-1668（窗 100–500）——把 2026-10-09 的审计固化为断言
// 数据在位自动跑；缺席 GTEST_SKIP（golden 式根探测）
// ============================================================================

namespace {

json readJson(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return json::parse(f);
}

cv::Mat parseMat64F(const json& j) {
    const int rows = (int)j.size();
    const int cols = (int)j[0].size();
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = j[r][c].get<double>();
    return m;
}

bool findRealData(fs::path& camJson, fs::path& lasJson, fs::path& sidecarBin) {
    std::vector<fs::path> roots;
    if (const char* env = std::getenv("FC_BASELINE_DEPTH_ROOT"))
        if (*env) roots.emplace_back(env);
    fs::path cwd = fs::current_path();
    for (int i = 0; i < 8; ++i) {        // CWD 及各级祖先
        roots.push_back(cwd);
        if (cwd == cwd.parent_path()) break;
        cwd = cwd.parent_path();
    }
    roots.emplace_back("E:/JEAMMWARE2601001/factory_calib");
    for (const auto& r : roots) {
        camJson = r / "module2_laser/test_data/laser261006-1668/camera_calib.json";
        lasJson = r / "data_out/laser261006-1668_depth100-500/laser_calib.json";
        sidecarBin = r / "data_out/laser261006-1668_depth100-500/curve_map_temp_table.bin";
        if (fs::is_regular_file(camJson) && fs::is_regular_file(lasJson))
            return true;
    }
    return false;
}

} // namespace

TEST(BaselineDepthRealData, Laser261006_1668_Depth100_500) {
    fs::path camPath, lasPath, sidecarPath;
    if (!findRealData(camPath, lasPath, sidecarPath))
        GTEST_SKIP() << "real data absent (need laser261006-1668 camera+laser json)";

    // ---- 组装（复刻 test_golden_ref_tier toCurveMapInput 配方） ----
    const json cam = readJson(camPath);
    const json las = readJson(lasPath);
    ASSERT_TRUE(cam.contains("imageSize") && cam["imageSize"].size() >= 2);
    const cv::Size imageSize(cam["imageSize"][0].get<int>(),
                             cam["imageSize"][1].get<int>());
    const json& rec = cam.at("rectify");
    const cv::Mat P1 = parseMat64F(rec.at("P1"));
    const cv::Mat P2 = parseMat64F(rec.at("P2"));
    const json& vr = rec.at("validRoiLeft");
    const cv::Rect roi(vr.at("x").get<int>(), vr.at("y").get<int>(),
                       vr.at("w").get<int>(), vr.at("h").get<int>());

    CurveMapInput in;
    const json& pjc = las.at("pjc");
    for (const auto& cj : pjc.at("emissionCurves")) {
        const json& co = cj.at("coeffs");
        ImplicitCurve c;
        for (int k = 0; k < 6; ++k) c.coeffs[k] = co.at(k).get<double>();
        in.curves.push_back(c);
    }
    ASSERT_EQ(in.curves.size(), size_t{25});
    for (int i = 0; i < 3; ++i)
        in.virtualT[i] = pjc.at("projectorT").at(i).get<double>();
    in.f = P1.at<double>(0, 0);
    in.principalPoint = cv::Point2d(P1.at<double>(0, 2), P1.at<double>(1, 2));
    in.baseline = std::fabs(P2.at<double>(0, 3)) / in.f;
    in.imageSize = imageSize;
    in.roi = roi;
    in.virtualK = cv::Matx33d(in.f, 0.0, in.principalPoint.x,
                              0.0, in.f, in.principalPoint.y,
                              0.0, 0.0, 1.0);
    const float rowStep = pjc.value("epipolarRowStep", 0.7f);
    EXPECT_FLOAT_EQ(rowStep, 0.67f) << "本数据集口径 rowStep=0.67（2026-10-12 起主库 plane_map.epipolarStep；0.5/0.7=旧口径渗入）";

    // hasPjcF 交叉诊断（口径不一致时原因前置可见；先例同款）
    ASSERT_TRUE(pjc.contains("f") && pjc.contains("principalPoint"));
    EXPECT_DOUBLE_EQ(pjc.at("f").get<double>(), P1.at<double>(0, 0));
    EXPECT_DOUBLE_EQ(pjc.at("principalPoint")[0].get<double>(), P1.at<double>(0, 2));
    EXPECT_DOUBLE_EQ(pjc.at("principalPoint")[1].get<double>(), P1.at<double>(1, 2));

    // ---- 生成（基线景深窗 100–500） ----
    CurveMapParams prm;
    prm.epipolarRowStep = rowStep;
    CurveMapCuda cuda(prm);
    cuda.SetBaselineDepthWindow(100.0, 500.0);
    const CurveMapResult r = cuda.Generate(in);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_GT(r.entryCount, 0u);

    // ---- 逐行流式解码＋断言 ----
    const double fB = in.f * in.baseline;   // kernel 操作数序
    long total = 0;
    std::map<int, LineStat> perLine;
    decodeAndCheckWindow(r, 100.0, 500.0, fB, (double)rowStep, total, perLine);

    EXPECT_EQ(perLine.size(), size_t{25});           // 25 线全部有条目
    for (int lid = 1; lid <= 25; ++lid) {
        ASSERT_GT(perLine.count(lid), 0) << "line " << lid << " 无条目";
        EXPECT_GT(perLine[lid].cnt, 0);
    }
    // 宽松护栏（0.67 口径当前盆 ref=14,827,900；PJC 双盆地历史摆幅 ~±40%——
    // 旧盆 0.5 折算 ≈10.7M, 现盆 ≈14.8M, 护栏需兼容两盆; rowStep 回归由 :504 单独钉死）
    EXPECT_GE(r.entryCount, 9000000u);
    EXPECT_LE(r.entryCount, 20000000u);

    // 可选 sidecar 交叉核对（超昂贵操作 → opt-in，先例 FC_CMTT_CUDA_FULL 定位）
    if (const char* sc = std::getenv("FC_BASELINE_DEPTH_SIDECAR")) {
        if (*sc && std::string(sc) != "0" && fs::is_regular_file(sidecarPath)) {
            std::ifstream f(sidecarPath, std::ios::binary);
            std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
            ASSERT_FALSE(data.empty());
            CmttReader rd;
            ASSERT_TRUE(rd.load(data.data(), data.size()));
            int refIdx = -1;
            for (size_t i = 0; i < rd.tierCount(); ++i)
                if (rd.isReference(i)) { refIdx = (int)i; break; }
            ASSERT_GE(refIdx, 0);
            EXPECT_EQ(rd.entryCount((size_t)refIdx), r.entryCount)
                << "CUDA 生成 entryCount 与 sidecar 参考档不一致";
        }
    }
}
