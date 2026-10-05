// test_curve_map_temp_table.cpp — curve_map 温度表批量生成器单测
//
// 覆盖: 冻结网格（硬规则 #1, 含 header-only 断言防线直测: 不同 ROI 拒绝＋
//       rowMin↔dataRows 同尺寸重排被派生一致性关系大声拒绝）/ 参考档失败不出
//       产物（#3）/ 非参考档 missing 宽容 / 确定性（#8）/ 曲线重标（Task 0 像素
//       坐标契约, 单元级 3 组手算向量＋前置拒绝＋端到端配方复刻逐字节对拍＋
//       未重标可区分）/ clamped 比例 / sidecar CMTT 往返 / 参数拒绝（含温网整除性）/
//       内存契约（档载荷只驻 sidecarBytes, 经 CmttReader 取档）。
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md（Task 4）。
// 期望值全部由 TierSpec 独立派生; 重标公式为设计文档公式的测试内独立实现,
// 与生成器实现无共享代码。

#include <gtest/gtest.h>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../curve_map_temp_table.h"

using namespace calib;

namespace {

// ============================================================================
// 合成场景（320x240、2 条近水平抛物线、参考温 25.0C、5 档 ±1C/0.5）
// ============================================================================

constexpr int kW = 320, kH = 240;
constexpr double kRefT = 25.0;
const cv::Size kImageSize{kW, kH};
const cv::Rect kBaseRoi{16, 16, 288, 208};          // 参考节点 validRoiLeft（冻结基准）

constexpr double kFRef = 500.0, kCx = 160.0, kCy = 120.0, kBRef = 40.0;
constexpr double kFSlope = 0.01, kBSlope = 0.008;   // 合成温漂斜率（放大到量化可见, 非物理 cte）
const cv::Vec3d kVirtualT{2.0, 0.5, 1.0};

double fAt(double t) { return kFRef * (1.0 + kFSlope * (t - kRefT)); }
double bAt(double t) { return kBRef * (1.0 + kBSlope * (t - kRefT)); }

struct TierSpec {
    double t, f, cx, cy, B;
    cv::Rect roi;
    double tx, ty, tz;
};

/// 节点集 A: 网格 5 档全覆盖（全直取, 无插值误差）; 各档 ROI 故意不同 → 验证冻结到参考档
std::vector<TierSpec> tiersFull() {
    const std::vector<double> ts{24.0, 24.5, 25.0, 25.5, 26.0};
    std::vector<TierSpec> out;
    for (size_t i = 0; i < ts.size(); ++i) {
        const int d = static_cast<int>(i) * 2 - 4;   // -4..+4, 参考档（i=2）恰为 kBaseRoi
        out.push_back({ts[i], fAt(ts[i]), kCx, kCy, bAt(ts[i]),
                       {16 + d, 16 + d, 288 - 2 * d, 208 - 2 * d},
                       kVirtualT[0], kVirtualT[1], kVirtualT[2]});
    }
    return out;
}

/// 节点集 B: 只覆盖 [24.4, 25.6] → 网格端档 24.0/26.0 越界 clamp
std::vector<TierSpec> tiersNarrow() {
    const std::vector<double> ts{24.4, 25.0, 25.6};
    std::vector<TierSpec> out;
    for (double t : ts)
        out.push_back({t, fAt(t), kCx, kCy, bAt(t), kBaseRoi,
                       kVirtualT[0], kVirtualT[1], kVirtualT[2]});
    return out;
}

/// 节点集 C: tz 随温线性过零 → 25.5 档插值 tz=0（极点无穷远）→ 非参考档 Generate 失败
std::vector<TierSpec> tiersTzCrossing() {
    return {
        {24.0, fAt(24.0), kCx, kCy, bAt(24.0), kBaseRoi, 2.0, 0.5,  1.2},
        {25.0, fAt(25.0), kCx, kCy, bAt(25.0), kBaseRoi, 2.0, 0.5,  0.4},
        {26.0, fAt(26.0), kCx, kCy, bAt(26.0), kBaseRoi, 2.0, 0.5, -0.4},
    };
}

/// 节点集 D: 61 档全覆盖（±15/0.5, 全直取无 clamp; TierThreadsInvariance 用——
/// 真规模档数下小图线程扫描, 与载体级 fcstepdump --cmtt-gen 三档 sha256 对拍互补）
std::vector<TierSpec> tiersWide() {
    std::vector<TierSpec> out;
    for (int k = -30; k <= 30; ++k) {
        const double t = kRefT + k * 0.5;
        out.push_back({t, fAt(t), kCx, kCy, bAt(t), kBaseRoi,
                       kVirtualT[0], kVirtualT[1], kVirtualT[2]});
    }
    return out;
}

cv::Mat makeP1(const TierSpec& s) {
    return (cv::Mat_<double>(3, 4) << s.f,  0.0, s.cx, 0.0,
                                        0.0, s.f, s.cy, 0.0,
                                        0.0, 0.0, 1.0,  0.0);
}

cv::Mat makeP2(const TierSpec& s) {
    return (cv::Mat_<double>(3, 4) << s.f, 0.0, s.cx, -s.B * s.f,   // (0,3) = -B·f
                                        0.0, s.f, s.cy, 0.0,
                                        0.0, 0.0, 1.0,  0.0);
}

StereoRectifyTempTableResult makeRectify(const std::vector<TierSpec>& specs,
                                         double refTemp = kRefT) {
    StereoRectifyTempTableResult r;
    r.success = true;
    r.referenceTemp = refTemp;
    r.tableSize = static_cast<int>(specs.size());
    for (const auto& s : specs) {
        StereoRectifyTempEntry e;
        e.temperature = s.t;
        e.P1 = makeP1(s);
        e.P2 = makeP2(s);
        e.validRoiLeft = s.roi;
        r.table.push_back(e);
    }
    return r;
}

LaserExtrinsicCompensateCPUResult makeLaser(const std::vector<TierSpec>& specs,
                                            double refTemp = kRefT) {
    LaserExtrinsicCompensateCPUResult l;
    l.success = true;
    l.referenceTemp = refTemp;
    l.leftResult.success = true;
    l.leftResult.referenceTemp = refTemp;
    for (const auto& s : specs) {
        ExtrinsicCompensatedEntry e;
        e.temperature = s.t;
        e.T[0] = s.tx;
        e.T[1] = s.ty;
        e.T[2] = s.tz;
        l.leftResult.table.push_back(e);
    }
    return l;
}

/// 2 条近水平抛物线: v(u) = v0 + sag·((u-cx)/hs)²（隐式化灌入 coeffs）
std::vector<ImplicitCurve> makeCurves() {
    std::vector<ImplicitCurve> out;
    for (double v0 : {50.0, 150.0}) {
        const double sag = 8.0, hs = 100.0;
        ImplicitCurve c;
        c.coeffs[0] = sag / (hs * hs);
        c.coeffs[1] = 0.0;
        c.coeffs[2] = 0.0;
        c.coeffs[3] = -2.0 * sag * kCx / (hs * hs);
        c.coeffs[4] = -1.0;
        c.coeffs[5] = v0 + sag * kCx * kCx / (hs * hs);
        out.push_back(c);
    }
    return out;
}

CurveMapTempTableGenParams genParams() {
    CurveMapTempTableGenParams p;
    p.referenceTemp = kRefT;
    p.tempHalfRange = 1.0f;
    p.tempStep = 0.5f;
    p.rowStep = 0.7f;
    p.depthMin = 100.0f;
    p.depthMax = 700.0f;
    return p;
}

bool bytesEqual(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size()) == 0;
}

/// 测试内独立实现的重标公式（设计文档"温度进入点⚠节"系数表逐项转写）
ImplicitCurve testRescale(const ImplicitCurve& c, double s, double mx, double my) {
    ImplicitCurve r = c;
    const double a = c.coeffs[0], b = c.coeffs[1], cc = c.coeffs[2];
    const double d = c.coeffs[3], e = c.coeffs[4], f0 = c.coeffs[5];
    r.coeffs[3] = s * d - (2.0 * a * mx + b * my);
    r.coeffs[4] = s * e - (b * mx + 2.0 * cc * my);
    r.coeffs[5] = s * s * f0 - s * (d * mx + e * my)
                + (a * mx * mx + b * mx * my + cc * my * my);
    return r;
}

/// 直连 CurveMapGenerator（配方复刻用; 参数自 genParams() 派生, 不重复硬编码）
std::vector<uint8_t> generateDirect(const std::vector<ImplicitCurve>& curves,
                                    double f, const cv::Point2d& pp, double B,
                                    const cv::Vec3d& t, const cv::Rect& roi) {
    const CurveMapTempTableGenParams gp = genParams();
    CurveMapParams prm;
    prm.epipolarRowStep = gp.rowStep;
    prm.depthMin = gp.depthMin;
    prm.depthMax = gp.depthMax;
    CurveMapGenerator g(prm);
    CurveMapInput in;
    in.curves = curves;
    in.f = f;
    in.principalPoint = pp;
    in.baseline = B;
    in.imageSize = kImageSize;
    in.roi = roi;
    in.virtualT = t;
    auto r = g.Generate(in);
    EXPECT_TRUE(r.success) << r.message;
    return r.tableBytes;
}

/// 载入 blob 并断言网格与基准表逐字段一致（rowStep/roi 四元/rowCount）
void expectGridEquals(const std::vector<uint8_t>& blob, const CurveMapTable& base) {
    CurveMapTable t;
    std::string err;
    ASSERT_TRUE(t.Load(blob.data(), blob.size(), err)) << err;
    EXPECT_FLOAT_EQ(t.rowStep(), base.rowStep());
    EXPECT_EQ(t.roi().x, base.roi().x);
    EXPECT_EQ(t.roi().y, base.roi().y);
    EXPECT_EQ(t.roi().width, base.roi().width);
    EXPECT_EQ(t.roi().height, base.roi().height);
    EXPECT_EQ(t.rowCount(), base.rowCount());
}

/// 冻结基准六字段自 sidecar 头取（与生成器自参考档 blob 头解析值逐字段同源）
CmttFrozenGrid frozenFromHeader(const CmttHeader& h) {
    CmttFrozenGrid f;
    f.rowStep = h.rowStep;
    f.roi = cv::Rect(h.roiX, h.roiY, h.roiW, h.roiH);
    f.rowCount = h.rowCount;
    f.rowMin = h.rowMin;
    f.depthMin = h.depthMin;
    f.depthMax = h.depthMax;
    return f;
}

} // namespace

// ============================================================================
// 硬规则 #1: 冻结网格
// ============================================================================

TEST(Gen, FiveTiersFrozenGrid) {
    CurveMapTempTableGenerator gen(genParams());
    auto r = gen.Generate(makeCurves(), makeRectify(tiersFull()),
                          makeLaser(tiersFull()), kImageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{5});
    EXPECT_EQ(r.okCount(), 5);
    EXPECT_EQ(r.referenceIndex(), size_t{2});
    EXPECT_EQ(r.qualityFlag, QualityFlag::Normal);

    // 参考档 flags: bit0 置位, 无 missing/clamped
    EXPECT_NE(r.tiers[2].flags & kCmttFlagReference, 0);
    EXPECT_EQ(r.tiers[2].flags & kCmttFlagMissing, 0);
    EXPECT_EQ(r.tiers[2].flags & kCmttFlagClamped, 0);

    // 内存契约: 档载荷只驻 sidecarBytes → 经 CmttReader 取档（Task 5 消费路径同款）
    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));
    const auto refBlob = rd.tierBytes(r.referenceIndex());
    ASSERT_TRUE(refBlob.has_value());
    const CmttFrozenGrid frozen = frozenFromHeader(rd.header());

    // 各 ok 档有数据且网格与参考档逐字段一致（源表各档 ROI 不同 → 冻结到参考档）
    CurveMapTable base;
    std::string err;
    ASSERT_TRUE(base.Load(refBlob->data(), refBlob->size(), err)) << err;
    for (size_t i = 0; i < r.tiers.size(); ++i) {
        EXPECT_TRUE(r.tiers[i].ok);
        EXPECT_GT(r.tiers[i].entryCount, 0u);
        EXPECT_GT(r.tiers[i].rowCountWithData, 0u);
        const auto b = rd.tierBytes(i);
        ASSERT_TRUE(b.has_value()) << "tier " << i;
        expectGridEquals(*b, base);
        EXPECT_TRUE(cmttTierGridMatches(b->data(), b->size(), frozen, nullptr));
    }

    // 冻结断言防线①: 真实生成的不同 ROI blob → 断言拒绝（why 注明 roi）
    const auto other = generateDirect(makeCurves(), fAt(kRefT),
                                      {kCx, kCy}, bAt(kRefT), kVirtualT,
                                      cv::Rect{32, 32, 256, 192});
    std::string why;
    EXPECT_FALSE(cmttTierGridMatches(other.data(), other.size(), frozen, &why));
    EXPECT_NE(why.find("roi"), std::string::npos) << why;

    // 冻结断言防线②（派生一致性）: rowMin↔dataRows 同尺寸字段重排 → 大声拒绝。
    // BlobHeaderV1 布局 rowMin@字节 44 / dataRows@48（复刻布局契约的测试内独立钉死）;
    // 重排后 rowMin 读到 dataRows 值, 与 |rowMin − floor(roi.y/rowStep+0.5)| ≤ 1 关系冲突。
    auto hdrSwap = *refBlob;
    for (int i = 0; i < 4; ++i)
        std::swap(hdrSwap[static_cast<size_t>(44 + i)],
                  hdrSwap[static_cast<size_t>(48 + i)]);
    EXPECT_FALSE(cmttTierGridMatches(hdrSwap.data(), hdrSwap.size(), frozen, &why));
    EXPECT_NE(why.find("rowMin"), std::string::npos) << why;
}

// ============================================================================
// 硬规则 #3: 参考档失败 → 整体不出产物
// ============================================================================

TEST(Gen, ReferenceTierFailNoOutput) {
    CurveMapTempTableGenerator gen(genParams());
    auto r = gen.Generate({}, makeRectify(tiersFull()), makeLaser(tiersFull()),
                          kImageSize);
    EXPECT_FALSE(r.success);
    EXPECT_TRUE(r.sidecarBytes.empty());
    EXPECT_FALSE(r.message.empty());   // message 透传参考档失败原因
}

// ============================================================================
// 非参考档 missing 宽容: 中间档失败 → bit1、tier 数不变、其余 ok
// ============================================================================

TEST(Gen, NonRefTierMissingTolerated) {
    CurveMapTempTableGenerator gen(genParams());
    auto r = gen.Generate(makeCurves(), makeRectify(tiersFull()),
                          makeLaser(tiersTzCrossing()), kImageSize);
    ASSERT_TRUE(r.success) << r.message;          // 非参考档可缺, 整体仍成功
    ASSERT_EQ(r.tiers.size(), size_t{5});         // tier 数不变
    EXPECT_EQ(r.okCount(), 4);

    const auto& bad = r.tiers[3];                 // 25.5C: 插值 tz=0 → Generate 失败
    EXPECT_DOUBLE_EQ(bad.temperature, 25.5);
    EXPECT_FALSE(bad.ok);
    EXPECT_NE(bad.flags & kCmttFlagMissing, 0);
    EXPECT_EQ(bad.entryCount, 0u);

    for (const int i : {0, 1, 2, 4})
        EXPECT_TRUE(r.tiers[static_cast<size_t>(i)].ok) << "tier " << i;
    EXPECT_NE(r.tiers[2].flags & kCmttFlagReference, 0);   // 参考档仍在
    EXPECT_EQ(r.qualityFlag, QualityFlag::Warning);
    EXPECT_NE(r.message.find("missing"), std::string::npos);   // 计数汇总

    // 内存契约: missing 档在 sidecar 内无载荷（经 Reader 验证, 不驻 tiers）
    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));
    EXPECT_EQ(rd.tierCount(), size_t{5});
    EXPECT_TRUE(rd.isMissing(3));
    EXPECT_FALSE(rd.tierBytes(3).has_value());
}

// ============================================================================
// 硬规则 #8: 确定性
// ============================================================================

TEST(Gen, Determinism) {
    CurveMapTempTableGenerator gen(genParams());
    const auto curves = makeCurves();
    const auto rect = makeRectify(tiersFull());
    const auto laser = makeLaser(tiersFull());
    auto r1 = gen.Generate(curves, rect, laser, kImageSize);
    auto r2 = gen.Generate(curves, rect, laser, kImageSize);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    EXPECT_TRUE(bytesEqual(r1.sidecarBytes, r2.sidecarBytes));
    EXPECT_EQ(r1.gridHash, r2.gridHash);
}

// ============================================================================
// 档级并行线程数不变性: 61 档（±15/0.5）tierThreads ∈ {1,4,8} 三跑,
// sidecarBytes 逐位相等＋gridHash/tier flags 相等（硬规则 #8 的并发维）
// ============================================================================

TEST(Gen, TierThreadsInvariance) {
    const auto curves = makeCurves();
    const auto rect = makeRectify(tiersWide());
    const auto laser = makeLaser(tiersWide());

    auto run = [&](int tierThreads) {
        CurveMapTempTableGenParams p = genParams();
        p.tempHalfRange = 15.0f;
        p.tierThreads = tierThreads;
        CurveMapTempTableGenerator gen(p);
        return gen.Generate(curves, rect, laser, kImageSize);
    };
    auto r1 = run(1), r4 = run(4), r8 = run(8);
    ASSERT_TRUE(r1.success) << r1.message;
    ASSERT_TRUE(r4.success) << r4.message;
    ASSERT_TRUE(r8.success) << r8.message;

    ASSERT_EQ(r1.tiers.size(), size_t{61});
    ASSERT_EQ(r4.tiers.size(), size_t{61});
    ASSERT_EQ(r8.tiers.size(), size_t{61});
    EXPECT_EQ(r1.okCount(), 61);
    EXPECT_EQ(r8.okCount(), 61);
    EXPECT_EQ(r1.referenceIndex(), size_t{30});

    // 逐位相等（sidecar 全字节）＋冻结栅格身份一致
    EXPECT_TRUE(bytesEqual(r1.sidecarBytes, r4.sidecarBytes));
    EXPECT_TRUE(bytesEqual(r1.sidecarBytes, r8.sidecarBytes));
    EXPECT_EQ(r1.gridHash, r4.gridHash);
    EXPECT_EQ(r1.gridHash, r8.gridHash);

    // 逐档 flags 一致（含参考档位/无 missing/无 clamped）
    for (size_t i = 0; i < r1.tiers.size(); ++i) {
        ASSERT_EQ(r1.tiers[i].flags, r4.tiers[i].flags) << "tier " << i;
        ASSERT_EQ(r1.tiers[i].flags, r8.tiers[i].flags) << "tier " << i;
        EXPECT_EQ(r1.tiers[i].ok, r8.tiers[i].ok);
        EXPECT_EQ(r1.tiers[i].entryCount, r8.tiers[i].entryCount);
    }
}

// ============================================================================
// Task 0 契约: 曲线重标
// ============================================================================

TEST(Gen, CurveRescaleApplied) {
    // ---- 前置条件: s>0 且有限（s = f_T/f_ref, 两侧 f 逐档为正 → 违例即上游已坏） ----
    const ImplicitCurve anyCurve = makeCurves()[0];
    EXPECT_THROW(rescaleCurve(anyCurve, 0.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(rescaleCurve(anyCurve, -1.5, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(rescaleCurve(anyCurve, std::numeric_limits<double>::infinity(),
                              0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(rescaleCurve(anyCurve, std::numeric_limits<double>::quiet_NaN(),
                              0.0, 0.0), std::invalid_argument);

    // ---- 单元级: rescaleCurve 三组手算向量 ----
    {   // 纯缩放（m=0）: x²=100 → s=2 后 x²=400
        ImplicitCurve c;
        c.coeffs[0] = 1; c.coeffs[5] = -100;
        const auto g = rescaleCurve(c, 2.0, 0.0, 0.0);
        EXPECT_DOUBLE_EQ(g.coeffs[0], 1.0);
        EXPECT_DOUBLE_EQ(g.coeffs[3], 0.0);
        EXPECT_DOUBLE_EQ(g.coeffs[4], 0.0);
        EXPECT_DOUBLE_EQ(g.coeffs[5], -400.0);
    }
    {   // 纯平移（s=1, m=(3,-2)）: x+2y-5=0 → x+2y-4=0（代入 x=x'-3, y=y'+2 手算）
        ImplicitCurve c;
        c.coeffs[3] = 1; c.coeffs[4] = 2; c.coeffs[5] = -5;
        const auto g = rescaleCurve(c, 1.0, 3.0, -2.0);
        EXPECT_DOUBLE_EQ(g.coeffs[3], 1.0);
        EXPECT_DOUBLE_EQ(g.coeffs[4], 2.0);
        EXPECT_DOUBLE_EQ(g.coeffs[5], -4.0);
    }
    {   // 一般式: 手算 d'=-11.3, e'=-4.4, f'=24.28（a,b,c 不变）
        ImplicitCurve c;
        c.coeffs[0] = 2; c.coeffs[1] = 1.5; c.coeffs[2] = -1;
        c.coeffs[3] = -4; c.coeffs[4] = 0.5; c.coeffs[5] = 7;
        const auto g = rescaleCurve(c, 1.2, 2.0, -1.0);
        EXPECT_DOUBLE_EQ(g.coeffs[0], 2.0);
        EXPECT_DOUBLE_EQ(g.coeffs[1], 1.5);
        EXPECT_DOUBLE_EQ(g.coeffs[2], -1.0);
        EXPECT_NEAR(g.coeffs[3], -11.3, 1e-12);
        EXPECT_NEAR(g.coeffs[4], -4.4, 1e-12);
        EXPECT_NEAR(g.coeffs[5], 24.28, 1e-12);
    }

    // ---- 端到端: 26.0C 档（s=1.01≠1）配方复刻逐字节对拍 ----
    CurveMapTempTableGenerator gen(genParams());
    auto r = gen.Generate(makeCurves(), makeRectify(tiersFull()),
                          makeLaser(tiersFull()), kImageSize);
    ASSERT_TRUE(r.success) << r.message;

    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));
    const auto tierBlob = rd.tierBytes(4);        // 26.0C
    ASSERT_TRUE(tierBlob.has_value());
    ASSERT_FALSE(tierBlob->empty());

    const double s = fAt(26.0) / fAt(kRefT);
    const double mx = kCx - s * kCx;              // pp 恒 (160,120) → m≠0（一般式）
    const double my = kCy - s * kCy;
    std::vector<ImplicitCurve> resc;
    for (const auto& c : makeCurves())
        resc.push_back(testRescale(c, s, mx, my));

    // 重标配方直连生成 → 与 sidecar 档字节逐位一致
    const auto replica = generateDirect(resc, fAt(26.0), {kCx, kCy}, bAt(26.0),
                                        kVirtualT, kBaseRoi);
    EXPECT_TRUE(bytesEqual(replica, *tierBlob));

    // 未重标（原始曲线 + 温档参数）→ 与 sidecar 档可区分（重标开关生效）
    const auto raw = generateDirect(makeCurves(), fAt(26.0), {kCx, kCy}, bAt(26.0),
                                    kVirtualT, kBaseRoi);
    EXPECT_FALSE(bytesEqual(raw, *tierBlob));
}

// ============================================================================
// clamped 比例
// ============================================================================

TEST(Gen, ClampedRatioReported) {
    CurveMapTempTableGenerator gen(genParams());
    const auto narrow = tiersNarrow();            // 源表只覆盖 [24.4, 25.6]
    auto r = gen.Generate(makeCurves(), makeRectify(narrow),
                          makeLaser(narrow), kImageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{5});

    EXPECT_NE(r.tiers[0].flags & kCmttFlagClamped, 0);   // 24.0 越下界
    EXPECT_EQ(r.tiers[1].flags & kCmttFlagClamped, 0);
    EXPECT_EQ(r.tiers[2].flags & kCmttFlagClamped, 0);   // 参考档
    EXPECT_EQ(r.tiers[3].flags & kCmttFlagClamped, 0);
    EXPECT_NE(r.tiers[4].flags & kCmttFlagClamped, 0);   // 26.0 越上界

    EXPECT_FLOAT_EQ(r.clampedRatio, 0.4f);        // 2/5
    EXPECT_EQ(r.okCount(), 5);                    // clamped 档端点直取兜底仍 ok
    EXPECT_EQ(r.qualityFlag, QualityFlag::Warning);
}

// ============================================================================
// sidecar CMTT 往返
// ============================================================================

TEST(Gen, SidecarViaCmttRoundtrip) {
    CurveMapTempTableGenerator gen(genParams());
    auto r = gen.Generate(makeCurves(), makeRectify(tiersFull()),
                          makeLaser(tiersTzCrossing()), kImageSize);   // 含 1 missing 档
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_FALSE(r.sidecarBytes.empty());

    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));
    EXPECT_EQ(rd.tierCount(), size_t{5});
    EXPECT_TRUE(rd.isReference(2));
    EXPECT_FALSE(rd.isMissing(2));
    EXPECT_TRUE(rd.isMissing(3));

    for (const int i : {0, 1, 2, 4}) {            // ok 档: tierBytes 非空且 crc 通过
        const auto b = rd.tierBytes(static_cast<size_t>(i));
        ASSERT_TRUE(b.has_value()) << "tier " << i;
        EXPECT_FALSE(b->empty());
        EXPECT_EQ(rd.entryCount(static_cast<size_t>(i)),
                  r.tiers[static_cast<size_t>(i)].entryCount);
    }
    EXPECT_FALSE(rd.tierBytes(3).has_value());    // missing 档无载荷

    const auto& h = rd.header();
    EXPECT_EQ(h.tierCount, 5u);
    EXPECT_EQ(h.tempBaseX10, 250);                // 25.0C × 10
    EXPECT_EQ(h.tempStepX10, 5);                  // 0.5C × 10
    EXPECT_FLOAT_EQ(h.rowStep, 0.7f);
    EXPECT_FLOAT_EQ(h.depthMin, 100.0f);
    EXPECT_FLOAT_EQ(h.depthMax, 700.0f);
    EXPECT_EQ(h.roiX, kBaseRoi.x);
    EXPECT_EQ(h.roiY, kBaseRoi.y);
    EXPECT_EQ(h.roiW, kBaseRoi.width);
    EXPECT_EQ(h.roiH, kBaseRoi.height);
    EXPECT_GT(h.rowCount, 0u);
    EXPECT_GT(h.rowMin, 0u);
    EXPECT_EQ(h.gridHash, r.gridHash);
}

// ============================================================================
// 参数参考温与源表参考温一致性（调用方契约违约 → 显式拒绝）
// ============================================================================

TEST(Gen, RefTempMismatchRejected) {
    auto rect = makeRectify(tiersFull(), /*refTemp=*/26.0);   // 表内存在 26.0 节点
    auto laser = makeLaser(tiersFull(), /*refTemp=*/26.0);    // 排除"节点缺失"干扰
    CurveMapTempTableGenerator gen(genParams());              // params 参考温 = 25.0
    EXPECT_THROW(gen.Generate(makeCurves(), rect, laser, kImageSize),
                 std::invalid_argument);
}

// ============================================================================
// 构造参数拒绝路径（红线 #6: rowStep 必须继承; 硬规则 #2: 深度窗）
// ============================================================================

TEST(Gen, ParamsInvalidRejected) {
    {   // rowStep≤0
        EXPECT_THROW(([] {
            CurveMapTempTableGenParams p;
            p.rowStep = 0.0f;
            CurveMapTempTableGenerator g(p);
            (void)g;
        }()), std::invalid_argument);
    }
    {   // 深度窗非法（depthMax ≤ depthMin）
        EXPECT_THROW(([] {
            CurveMapTempTableGenParams p;
            p.depthMax = p.depthMin;
            CurveMapTempTableGenerator g(p);
            (void)g;
        }()), std::invalid_argument);
    }
}

// ============================================================================
// 温网整除性: halfRange 非 step 整数倍 → 端点漂移, 构造拒绝（消息含两值）
// ============================================================================

TEST(Gen, TempGridNotDivisibleRejected) {
    bool threw = false;
    try {
        CurveMapTempTableGenParams p = genParams();
        p.tempHalfRange = 1.0f;                   // 1.0/0.3 = 3.33 不可整除
        p.tempStep = 0.3f;
        CurveMapTempTableGenerator g(p);
        (void)g;
    } catch (const std::invalid_argument& e) {
        threw = true;
        const std::string m = e.what();
        EXPECT_NE(m.find(std::to_string(1.0f)), std::string::npos);
        EXPECT_NE(m.find(std::to_string(0.3f)), std::string::npos);
    }
    EXPECT_TRUE(threw);
}
