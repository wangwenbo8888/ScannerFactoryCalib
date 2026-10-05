// test_curve_map.cpp — 极线切曲线映射表单测
//
// 覆盖: 参数校验 / 几何正确性(合成曲线正反算) / 双根 / 退化分支 /
//       压缩往返(编码→解码逐条一致) / 解码符号红线(uR = uL − d) / 空输入
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <random>

#include "../curve_map.h"

using namespace calib;

namespace {

// 合成场景: 构造与设计一致的几何真值
//  - 虚拟相机 R=I, K 已知, T 已知
//  - 25 条近水平抛物线曲线（隐式化后灌入 ImplicitCurve.coeffs）
//  - 用"给定深度 Z 生成表条目"的正向几何做交叉验证
struct SyntheticScene {
    CurveMapInput input;
    CurveMapParams params;
    double f = 1284.83, cx = 999.75, cy = 755.04;
    double B = 133.1;
    cv::Vec3d t = cv::Vec3d(40.5, 8.5, 11.4);
    int W = 2048, H = 1536;
    static constexpr int kLines = 25;

    SyntheticScene() {
        input.f = f;
        input.principalPoint = cv::Point2d(cx, cy);
        input.baseline = B;
        input.imageSize = cv::Size(W, H);
        input.virtualK = cv::Matx33d(f, 0, cx, 0, f, cy, 0, 0, 1);
        input.virtualT = t;
        input.roi = cv::Rect(128, 96, W - 256, H - 192);
        params.epipolarRowStep = 0.7f;
        params.depthMin = 100.0f;
        params.depthMax = 700.0f;

        // 生成 25 条近水平曲线: v_i(u) = v0_i + sag_i·((u−cx)/halfSpan)²
        // 隐式化: F(u,v) = a·u² + b·u + (c − v)（线性于 v）, 再归一化
        input.curves.resize(kLines);
        for (int i = 0; i < kLines; ++i) {
            const double v0 = 180.0 + i * 52.0;              // 线距 52px（实测带内）
            const double sag = 20.0 + 18.0 * i / (kLines - 1);
            const double hs = 800.0;
            const double a = sag / (hs * hs);
            const double b = -2.0 * sag * cx / (hs * hs);
            const double c = v0 + sag * cx * cx / (hs * hs);
            double co[6] = {a, 0.0, 0.0, b, -1.0, c};
            double nrm = 0;
            for (double v : co) nrm += v * v;
            nrm = std::sqrt(nrm);
            for (int k = 0; k < 6; ++k)
                input.curves[static_cast<size_t>(i)].coeffs[k] = co[k] / nrm;
            input.curves[static_cast<size_t>(i)].pointCount = 0;
        }
    }

    // 正向真值: 给定左像素 (uL,v) 与线 i, 返回窗内全部交点的 (uR 栅格) 集合（≤2）。
    int groundTruth(double uL, double v, int i, int out_uRGrid[2]) const {
        const double Ex = f * t(0) / t(2) + cx;
        const double Ey = f * t(1) / t(2) + cy;
        const double du = Ex - uL, dv = Ey - v;
        const auto& c = input.curves[static_cast<size_t>(i)];
        const double A = c.coeffs[0], Bc = c.coeffs[1], C = c.coeffs[2];
        const double D = c.coeffs[3], E = c.coeffs[4], F0 = c.coeffs[5];
        const double a = A*du*du + Bc*du*dv + C*dv*dv;
        const double b = 2*A*uL*du + Bc*(uL*dv + v*du) + 2*C*v*dv + D*du + E*dv;
        const double cc = A*uL*uL + Bc*uL*v + C*v*v + D*uL + E*v + F0;
        double roots[2];
        int nr = 0;
        if (std::fabs(a) < 1e-18) {
            if (std::fabs(b) > 1e-18) roots[nr++] = -cc / b;
        } else {
            const double disc = b*b - 4*a*cc;
            if (disc < 0) return 0;
            roots[nr++] = (-b + std::sqrt(disc)) / (2*a);
            roots[nr++] = (-b - std::sqrt(disc)) / (2*a);
        }
        int nOut = 0;
        for (int r = 0; r < nr; ++r) {
            const double qu = uL + roots[r] * du;
            const double qv = v + roots[r] * dv;
            if (qu < 0 || qu >= W || qv < 0 || qv >= H) continue;
            const double qc = qu - cx;
            const double denom = qc - (uL - cx);
            if (std::fabs(denom) < 1e-9) continue;
            const double Z = (qc * t(2) - f * t(0)) / denom;
            if (Z < params.depthMin || Z > params.depthMax) continue;
            const double uRpx = uL - f * B / Z;
            if (uRpx < 0 || uRpx >= W) continue;
            out_uRGrid[nOut++] = static_cast<int>(std::lround(uRpx / 0.7));
        }
        // 去重（重根）
        if (nOut == 2 && out_uRGrid[0] == out_uRGrid[1]) nOut = 1;
        return nOut;
    }
};

} // namespace

// ============================================================================
// 参数校验
// ============================================================================

TEST(CurveMapTest, ParamsValidation) {
    CurveMapParams p;
    EXPECT_NO_THROW(p.validate());

    p.epipolarRowStep = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p = CurveMapParams{};
    p.depthMax = p.depthMin;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p = CurveMapParams{};
    p.uSampleStep = -2.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p = CurveMapParams{};
    p.uSampleStep = 1.4f;
    EXPECT_NO_THROW(p.validate());
    EXPECT_FLOAT_EQ(p.effectiveUSampleStep(), 1.4f);
    p.uSampleStep = -1.0f;
    EXPECT_FLOAT_EQ(p.effectiveUSampleStep(), p.epipolarRowStep);
}

TEST(CurveMapTest, EmptyCurves) {
    CurveMapGenerator gen;
    SyntheticScene sc;
    CurveMapInput in = sc.input;
    in.curves.clear();
    auto r = gen.Generate(in);
    EXPECT_FALSE(r.success);
}

TEST(CurveMapTest, ZeroTzRejected) {
    CurveMapGenerator gen;
    SyntheticScene sc;
    sc.input.virtualT = cv::Vec3d(40.5, 8.5, 0.0);   // 极点无穷远
    auto r = gen.Generate(sc.input);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.message.find("epipole"), std::string::npos);
}

// ============================================================================
// 几何正确性: 表值 vs 独立正向真值（B1 闭环的合成版）
// ============================================================================

TEST(CurveMapTest, GeometricClosure) {
    SyntheticScene sc;
    CurveMapGenerator gen(sc.params);
    auto r = gen.Generate(sc.input);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_GT(r.entryCount, 100000u);   // 合成全带覆盖应达十万级以上

    // 载入表
    CurveMapTable tbl;
    std::string err;
    ASSERT_TRUE(tbl.Load(r.tableBytes.data(), r.tableBytes.size(), err)) << err;
    EXPECT_FLOAT_EQ(tbl.rowStep(), 0.7f);

    // 抽样闭环: 随机采样（对齐生成器格点: uLpx = roi.x + k·uStep）
    std::mt19937 rng(42);
    const double uStep = 0.7;
    const int nSamples = static_cast<int>((sc.input.roi.width - 1) / uStep);
    std::uniform_int_distribution<int> uRow(sc.input.roi.y * 10 / 7,
                                            (sc.input.roi.y + sc.input.roi.height) * 10 / 7);
    std::uniform_int_distribution<int> uK(0, nSamples);
    int checked = 0, mismatch = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        const int row = uRow(rng);
        const double v = row * 0.7;
        const int k = uK(rng);
        const double uLpx = sc.input.roi.x + k * uStep;
        const int uLgrid = static_cast<int>(std::lround(uLpx / 0.7));

        CurveMapCand cands[64];
        const int n = tbl.Lookup(row, uLgrid, cands, 64);

        for (int i = 0; i < SyntheticScene::kLines; ++i) {
            int gtU[2];
            const int nGT = sc.groundTruth(uLpx, v, i, gtU);
            int tblU[8];
            int nTbl = 0;
            for (int q = 0; q < n && nTbl < 8; ++q)
                if (cands[q].lid == i + 1) tblU[nTbl++] = cands[q].uR;
            if (nGT == 0 && nTbl == 0) continue;
            ++checked;
            // 集合比较（顺序无关, ±1 格容差）
            bool ok = (nGT == nTbl);
            if (ok) {
                for (int g1 = 0; g1 < nGT && ok; ++g1) {
                    bool matched = false;
                    for (int g2 = 0; g2 < nTbl; ++g2)
                        if (std::abs(gtU[g1] - static_cast<int>(tblU[g2])) <= 1) {
                            matched = true; break;
                        }
                    if (!matched) ok = false;
                }
            }
            if (!ok) ++mismatch;
        }
    }
    // 允许 0.5% 边界抖动（ROI 边/量化边界/重根取舍差异）
    EXPECT_LT(mismatch, checked / 200 + 1)
        << "mismatch=" << mismatch << " checked=" << checked;
}

// ============================================================================
// 解码符号红线: uR = uL − d（恒 uR < uL）
// ============================================================================

TEST(CurveMapTest, DisparitySignInvariant) {
    SyntheticScene sc;
    CurveMapGenerator gen(sc.params);
    auto r = gen.Generate(sc.input);
    ASSERT_TRUE(r.success);
    CurveMapTable tbl;
    std::string err;
    ASSERT_TRUE(tbl.Load(r.tableBytes.data(), r.tableBytes.size(), err));

    std::mt19937 rng(7);
    for (int trial = 0; trial < 500; ++trial) {
        const int row = std::uniform_int_distribution<int>(100, 2000)(rng);
        const int uL = std::uniform_int_distribution<int>(200, 2700)(rng);
        CurveMapCand cands[64];
        const int n = tbl.Lookup(row, uL, cands, 64);
        for (int k = 0; k < n; ++k) {
            EXPECT_LT(cands[k].uR, static_cast<uint16_t>(uL))
                << "row=" << row << " uL=" << uL << " lid=" << (int)cands[k].lid
                << " uR=" << cands[k].uR << " —— 视差符号被破坏（红线）";
        }
    }
}

// ============================================================================
// 压缩往返: 生成→落盘→载入→查询一致
// ============================================================================

TEST(CurveMapTest, SerializeRoundtrip) {
    SyntheticScene sc;
    CurveMapGenerator gen(sc.params);
    auto r = gen.Generate(sc.input);
    ASSERT_TRUE(r.success);
    CurveMapTable t1, t2;
    std::string err;
    ASSERT_TRUE(t1.Load(r.tableBytes.data(), r.tableBytes.size(), err));
    ASSERT_TRUE(t1.SaveToFile("test_curve_map_roundtrip.bin", err)) << err;
    ASSERT_TRUE(t2.LoadFromFile("test_curve_map_roundtrip.bin", err)) << err;
    std::remove("test_curve_map_roundtrip.bin");

    EXPECT_EQ(t1.entryCount(), t2.entryCount());
    EXPECT_EQ(t1.rowCount(), t2.rowCount());
    EXPECT_FLOAT_EQ(t1.rowStep(), t2.rowStep());

    std::mt19937 rng(99);
    for (int trial = 0; trial < 300; ++trial) {
        const int row = std::uniform_int_distribution<int>(150, 1900)(rng);
        const int uL = std::uniform_int_distribution<int>(300, 2500)(rng);
        CurveMapCand c1[64], c2[64];
        const int n1 = t1.Lookup(row, uL, c1, 64);
        const int n2 = t2.Lookup(row, uL, c2, 64);
        ASSERT_EQ(n1, n2) << "row=" << row << " uL=" << uL;
        for (int k = 0; k < n1; ++k) {
            EXPECT_EQ(c1[k].lid, c2[k].lid);
            EXPECT_EQ(c1[k].uR, c2[k].uR);
        }
    }
}

// ============================================================================
// 双根语义: 同 (row,uL,lid) 至多 2 条
// ============================================================================

TEST(CurveMapTest, DoubleRootBound) {
    SyntheticScene sc;
    CurveMapGenerator gen(sc.params);
    auto r = gen.Generate(sc.input);
    ASSERT_TRUE(r.success);
    CurveMapTable tbl;
    std::string err;
    ASSERT_TRUE(tbl.Load(r.tableBytes.data(), r.tableBytes.size(), err));

    std::mt19937 rng(123);
    for (int trial = 0; trial < 1000; ++trial) {
        const int row = std::uniform_int_distribution<int>(100, 2000)(rng);
        const int uL = std::uniform_int_distribution<int>(200, 2700)(rng);
        CurveMapCand cands[64];
        const int n = tbl.Lookup(row, uL, cands, 64);
        int perLine[26] = {0};
        for (int k = 0; k < n; ++k) {
            ASSERT_LE(cands[k].lid, 25);
            ++perLine[cands[k].lid];
        }
        for (int lid = 1; lid <= 25; ++lid)
            EXPECT_LE(perLine[lid], 2) << "双根上限被破坏: row=" << row
                                       << " uL=" << uL << " lid=" << lid;
    }
}

// ============================================================================
// 深度窗: 全部条目反解深度落在窗内
// ============================================================================

TEST(CurveMapTest, DepthWindowInvariant) {
    SyntheticScene sc;
    CurveMapGenerator gen(sc.params);
    auto r = gen.Generate(sc.input);
    ASSERT_TRUE(r.success);
    CurveMapTable tbl;
    std::string err;
    ASSERT_TRUE(tbl.Load(r.tableBytes.data(), r.tableBytes.size(), err));

    std::mt19937 rng(2024);
    int checked = 0;
    for (int trial = 0; trial < 500; ++trial) {
        const int row = std::uniform_int_distribution<int>(150, 1900)(rng);
        for (int uL = 300; uL < 2600; uL += 37) {
            CurveMapCand cands[64];
            const int n = tbl.Lookup(row, uL, cands, 64);
            for (int k = 0; k < n; ++k) {
                const double dpx = (uL - cands[k].uR) * 0.7;
                if (dpx <= 0) continue;
                const double Z = sc.f * sc.B / dpx;
                ++checked;
                EXPECT_GE(Z, sc.params.depthMin - 1.0);   // 半格量化余量
                EXPECT_LE(Z, sc.params.depthMax + 30.0);  // 近端半格放大
            }
        }
    }
    EXPECT_GT(checked, 0);
}
