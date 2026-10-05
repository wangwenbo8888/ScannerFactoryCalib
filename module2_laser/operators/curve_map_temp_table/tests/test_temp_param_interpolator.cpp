// test_temp_param_interpolator.cpp — 温度参数插值器单测
//
// 覆盖: 节点直取逐位相等 / 线性中点（1e-12）/ atReference 零插值路径（硬规则 #4）/
//       越界 clamp＋标记 / ROI 恒冻结参考档（硬规则 #1, 含越界与 atReference）/
//       v2l 补偿 T 逐档插值（节点/中点/clamp）/ 空表·缺参考节点拒绝 / 构造期排序。
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md（Task 3）。
// 期望值全部由 TierSpec 独立派生（与 fcscan_e2e.cpp:106-133 的 CurveMapInput
// 组装配方同式）, 与实现无共享代码。

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../temp_param_interpolator.h"

using namespace calib;

namespace {

// 标准样例: 3 档（25.0 参考 / 25.2 / 25.4）。
// P1/P2 手填（对角 f＋主点的 3x4 CV_64F, P2(0,3)=-B·f）, 不依赖真实标定数据。
// 各档 validRoiLeft 故意不同 → 验证 ROI 冻结参考档而非随温取值。
struct TierSpec {
    double t, f, cx, cy, B;
    cv::Rect roi;
    double tx, ty, tz;   // v2l 补偿 T
};

const std::vector<TierSpec> kTiers = {
    {25.0, 1000.0, 320.5, 239.25, 120.5, {1, 2, 100, 200}, 10.0,  -0.5,  200.25},
    {25.2, 1001.5, 320.7, 239.35, 120.8, {2, 3, 101, 201}, 10.24, -0.44, 200.75},
    {25.4, 1003.0, 320.9, 239.45, 121.1, {3, 4, 102, 202}, 10.48, -0.38, 201.25},
};

cv::Mat makeP1(const TierSpec& s) {
    return (cv::Mat_<double>(3, 4) << s.f,  0.0, s.cx, 0.0,
                                        0.0, s.f, s.cy, 0.0,
                                        0.0, 0.0, 1.0,  0.0);
}

cv::Mat makeP2(const TierSpec& s) {
    return (cv::Mat_<double>(3, 4) << s.f, 0.0, s.cx, -s.B * s.f,   // (0,3)=-B·f（fcscan 同位）
                                        0.0, s.f, s.cy, 0.0,
                                        0.0, 0.0, 1.0, 0.0);
}

StereoRectifyTempTableResult makeRectify(std::vector<int> order = {0, 1, 2}) {
    StereoRectifyTempTableResult r;
    r.success = true;
    r.referenceTemp = 25.0;
    r.tableSize = static_cast<int>(order.size());
    for (int i : order) {
        StereoRectifyTempEntry e;
        e.temperature = kTiers[i].t;
        e.P1 = makeP1(kTiers[i]);
        e.P2 = makeP2(kTiers[i]);
        e.validRoiLeft = kTiers[i].roi;
        r.table.push_back(e);
    }
    return r;
}

LaserExtrinsicCompensateCPUResult makeLaser(std::vector<int> order = {0, 1, 2}) {
    LaserExtrinsicCompensateCPUResult l;
    l.success = true;
    l.referenceTemp = 25.0;
    l.leftResult.success = true;
    l.leftResult.referenceTemp = 25.0;
    for (int i : order) {
        ExtrinsicCompensatedEntry e;
        e.temperature = kTiers[i].t;
        e.T[0] = kTiers[i].tx;
        e.T[1] = kTiers[i].ty;
        e.T[2] = kTiers[i].tz;
        l.leftResult.table.push_back(e);
    }
    return l;
}

/// 独立派生: B = |P2(0,3)| / f（与 fcscan 配方同式）
double derivedB(const TierSpec& s) { return std::fabs(-s.B * s.f) / s.f; }

/// 期望直取值（roi 恒冻结参考档）
InterpParams expectedDirect(const TierSpec& s, bool clamped = false) {
    InterpParams p;
    p.f = s.f;
    p.principalPoint = cv::Point2d(s.cx, s.cy);
    p.baseline = derivedB(s);
    p.roi = kTiers[0].roi;
    p.virtualT = cv::Vec3d(s.tx, s.ty, s.tz);
    p.clamped = clamped;
    return p;
}

/// 逐位相等（==, 不用容差）
void expectExact(const InterpParams& got, const InterpParams& want) {
    EXPECT_EQ(got.f, want.f);
    EXPECT_EQ(got.principalPoint.x, want.principalPoint.x);
    EXPECT_EQ(got.principalPoint.y, want.principalPoint.y);
    EXPECT_EQ(got.baseline, want.baseline);
    EXPECT_EQ(got.roi.x, want.roi.x);
    EXPECT_EQ(got.roi.y, want.roi.y);
    EXPECT_EQ(got.roi.width, want.roi.width);
    EXPECT_EQ(got.roi.height, want.roi.height);
    EXPECT_EQ(got.virtualT[0], want.virtualT[0]);
    EXPECT_EQ(got.virtualT[1], want.virtualT[1]);
    EXPECT_EQ(got.virtualT[2], want.virtualT[2]);
    EXPECT_EQ(got.clamped, want.clamped);
}

/// 标量插值量容差比较（roi/clamped 不在容差语义内, 由 expectExact 覆盖）
void expectNear(const InterpParams& got, const InterpParams& want, double tol) {
    EXPECT_NEAR(got.f, want.f, tol);
    EXPECT_NEAR(got.principalPoint.x, want.principalPoint.x, tol);
    EXPECT_NEAR(got.principalPoint.y, want.principalPoint.y, tol);
    EXPECT_NEAR(got.baseline, want.baseline, tol);
    EXPECT_NEAR(got.virtualT[0], want.virtualT[0], tol);
    EXPECT_NEAR(got.virtualT[1], want.virtualT[1], tol);
    EXPECT_NEAR(got.virtualT[2], want.virtualT[2], tol);
}

double mid(double a, double b) { return (a + b) / 2.0; }

} // namespace

TEST(Interp, NodeExact) {
    TempParamInterpolator interp(makeRectify(), makeLaser());
    for (const auto& s : kTiers) {
        const InterpParams p = interp.at(s.t);
        expectExact(p, expectedDirect(s));   // 25.0/25.2/25.4 → 与表项派生值逐位相等
    }
}

TEST(Interp, MidpointLinear) {
    TempParamInterpolator interp(makeRectify(), makeLaser());
    const InterpParams p = interp.at(25.1);   // 25.0/25.2 的线性中点

    InterpParams want;
    want.f = mid(kTiers[0].f, kTiers[1].f);
    want.principalPoint = cv::Point2d(mid(kTiers[0].cx, kTiers[1].cx),
                                      mid(kTiers[0].cy, kTiers[1].cy));
    want.baseline = mid(derivedB(kTiers[0]), derivedB(kTiers[1]));
    want.virtualT = cv::Vec3d(mid(kTiers[0].tx, kTiers[1].tx),
                              mid(kTiers[0].ty, kTiers[1].ty),
                              mid(kTiers[0].tz, kTiers[1].tz));
    expectNear(p, want, 1e-12);
    EXPECT_FALSE(p.clamped);
}

TEST(Interp, ReferenceZeroInterpolation) {
    TempParamInterpolator interp(makeRectify(), makeLaser());
    EXPECT_EQ(interp.referenceTemp(), 25.0);

    // 硬规则 #4: atReference() 不走插值路径, 与参考节点项逐位相等
    expectExact(interp.atReference(), expectedDirect(kTiers[0]));

    // at() 命中参考节点允许直取 → 与 atReference() 数值一致（两条路径同落点）
    const InterpParams a = interp.at(25.0);
    const InterpParams r = interp.atReference();
    EXPECT_EQ(a.f, r.f);
    EXPECT_EQ(a.principalPoint.x, r.principalPoint.x);
    EXPECT_EQ(a.principalPoint.y, r.principalPoint.y);
    EXPECT_EQ(a.baseline, r.baseline);
    EXPECT_EQ(a.virtualT[0], r.virtualT[0]);
    EXPECT_EQ(a.virtualT[1], r.virtualT[1]);
    EXPECT_EQ(a.virtualT[2], r.virtualT[2]);
    EXPECT_FALSE(r.clamped);
}

TEST(Interp, OutOfRangeClampFlagged) {
    TempParamInterpolator interp(makeRectify(), makeLaser());

    const InterpParams lo = interp.at(24.8);   // 低于范围 → 最低档直取＋标记
    expectExact(lo, expectedDirect(kTiers[0], /*clamped=*/true));

    const InterpParams hi = interp.at(25.6);   // 高于范围 → 最高档直取＋标记
    expectExact(hi, expectedDirect(kTiers[2], /*clamped=*/true));
}

TEST(Interp, RoiAlwaysFrozenReference) {
    TempParamInterpolator interp(makeRectify(), makeLaser());
    const cv::Rect frozen = kTiers[0].roi;     // (1,2,100,200), 其余档故意不同
    for (double t : {25.0, 25.1, 25.2, 25.4, 24.8, 25.6}) {
        const InterpParams p = interp.at(t);
        EXPECT_EQ(p.roi.x, frozen.x);
        EXPECT_EQ(p.roi.y, frozen.y);
        EXPECT_EQ(p.roi.width, frozen.width);
        EXPECT_EQ(p.roi.height, frozen.height);
    }
    const InterpParams r = interp.atReference();
    EXPECT_EQ(r.roi.x, frozen.x);
    EXPECT_EQ(r.roi.y, frozen.y);
    EXPECT_EQ(r.roi.width, frozen.width);
    EXPECT_EQ(r.roi.height, frozen.height);
}

TEST(Interp, VirtualTPerTemp) {
    TempParamInterpolator interp(makeRectify(), makeLaser());

    {   // 节点直取
        const cv::Vec3d v = interp.at(25.2).virtualT;
        EXPECT_EQ(v[0], kTiers[1].tx);
        EXPECT_EQ(v[1], kTiers[1].ty);
        EXPECT_EQ(v[2], kTiers[1].tz);
    }
    {   // 中点（25.2/25.4 → 25.3）
        const cv::Vec3d v = interp.at(25.3).virtualT;
        EXPECT_NEAR(v[0], mid(kTiers[1].tx, kTiers[2].tx), 1e-12);
        EXPECT_NEAR(v[1], mid(kTiers[1].ty, kTiers[2].ty), 1e-12);
        EXPECT_NEAR(v[2], mid(kTiers[1].tz, kTiers[2].tz), 1e-12);
    }
    {   // 越界 clamp → 端点档直取
        const cv::Vec3d lo = interp.at(24.9).virtualT;
        EXPECT_EQ(lo[0], kTiers[0].tx);
        EXPECT_EQ(lo[1], kTiers[0].ty);
        EXPECT_EQ(lo[2], kTiers[0].tz);
        const cv::Vec3d hi = interp.at(25.5).virtualT;
        EXPECT_EQ(hi[0], kTiers[2].tx);
        EXPECT_EQ(hi[1], kTiers[2].ty);
        EXPECT_EQ(hi[2], kTiers[2].tz);
    }
}

TEST(Interp, EmptyTableThrows) {
    {   // 矫正表空 → 构造拒绝
        auto r = makeRectify();
        r.table.clear();
        EXPECT_THROW(TempParamInterpolator(r, makeLaser()), std::invalid_argument);
    }
    {   // v2l 补偿表空 → 构造拒绝
        auto l = makeLaser();
        l.leftResult.table.clear();
        EXPECT_THROW(TempParamInterpolator(makeRectify(), l), std::invalid_argument);
    }
    {   // 矫正表缺参考温度节点 → 构造拒绝
        auto r = makeRectify();
        r.referenceTemp = 26.0;
        EXPECT_THROW(TempParamInterpolator(r, makeLaser()), std::invalid_argument);
    }
    {   // 补偿表缺参考温度节点 → 构造拒绝
        auto l = makeLaser();
        l.referenceTemp = 26.0;
        EXPECT_THROW(TempParamInterpolator(makeRectify(), l), std::invalid_argument);
    }
}

TEST(Interp, UnsortedInputSortedAtConstruction) {
    // 构造期排序: 乱序喂入（25.4/25.0/25.2）行为与有序表一致
    TempParamInterpolator interp(makeRectify({2, 0, 1}), makeLaser({2, 0, 1}));
    EXPECT_EQ(interp.nodeCount(), size_t{3});

    expectExact(interp.at(25.0), expectedDirect(kTiers[0]));
    expectExact(interp.at(25.4), expectedDirect(kTiers[2]));
    expectExact(interp.atReference(), expectedDirect(kTiers[0]));

    const InterpParams p = interp.at(25.1);
    EXPECT_NEAR(p.f, mid(kTiers[0].f, kTiers[1].f), 1e-12);
    EXPECT_NEAR(p.virtualT[0], mid(kTiers[0].tx, kTiers[1].tx), 1e-12);
    EXPECT_FALSE(p.clamped);
}

TEST(Interp, NaNTemperatureRejected) {
    // NaN 与任何比较均为 false → 无 bracket 分支可落, 必须入口显式拒绝
    TempParamInterpolator interp(makeRectify(), makeLaser());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try {
        const InterpParams p = interp.at(nan);
        (void)p;
    } catch (const std::invalid_argument& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("NaN"), std::string::npos);   // 消息注明 NaN 输入
    }
    EXPECT_TRUE(threw);
}

TEST(Interp, MismatchedReferenceTempsRejected) {
    // 两表参考温度不一致（laser=25.2 表内存在该节点, 排除"节点缺失"干扰）→ 构造拒绝,
    // 异常消息须带两个温度值
    auto l = makeLaser();
    l.referenceTemp = 25.2;
    bool threw = false;
    try {
        TempParamInterpolator interp(makeRectify(), l);
        (void)interp;
    } catch (const std::invalid_argument& e) {
        threw = true;
        const std::string m = e.what();
        EXPECT_NE(m.find(std::to_string(25.0)), std::string::npos);
        EXPECT_NE(m.find(std::to_string(25.2)), std::string::npos);
    }
    EXPECT_TRUE(threw);
}

TEST(Interp, BadProjectionMatricesRejected) {
    {   // P1 缺失
        auto r = makeRectify();
        r.table[1].P1 = cv::Mat();
        EXPECT_THROW(TempParamInterpolator(r, makeLaser()), std::invalid_argument);
    }
    {   // P2 非 CV_64F（尺寸正确但类型错）
        auto r = makeRectify();
        r.table[1].P2 = cv::Mat(r.table[1].P2.size(), CV_32F);
        EXPECT_THROW(TempParamInterpolator(r, makeLaser()), std::invalid_argument);
    }
}

TEST(Interp, NonPositiveFocalRejected) {
    // 参考档 f≤0 → 构造拒绝（f 是 B=|P2(0,3)|/f 的分母, 非正数无意义）
    auto r = makeRectify();
    r.table[0].P1.at<double>(0, 0) = -5.0;
    EXPECT_THROW(TempParamInterpolator(r, makeLaser()), std::invalid_argument);
}
