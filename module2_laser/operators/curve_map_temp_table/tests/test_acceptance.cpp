// test_acceptance.cpp — curve_map 温度表验收套件（合成口径; 设计验收 #2/#6）
//
// 对象: CurveMapTempTableGenerator 端到端验收（docs/plans/2026-09-01-curve-map-temp-table-design.md "验收"节）。
//   用例 1 SmoothnessGlobalAndPerGroup —— 验收 #2:
//     21 档（25±5°C / 0.5 步距; 源表 20..30°C 整度节点全覆盖 → 无 clamp、无 missing）
//     → Generate → CmttReader 载入 sidecarBytes;
//     全局: 相邻档 entryCount |Δ|/N < 1%（设计理论预估 0.1-0.5%）;
//     单组: 逐档 tierBytes → CurveMapTable::Load → EnumerateRow 全行枚举
//     （两段式容量模式: 返回 -1 即容量不足 → 缓冲倍增重试）→ 按 (row,lid) 聚合
//     count → 相邻档逐组变化超 10% 者进跳变清单——FOV 边缘短弦组属预期白名单
//     （设计验收 #2）, 清单仅输出（cout/SUCCEED）不判失败。
//   用例 2 SyntheticGoldenRoundtrip —— 验收 #6（合成金样读写对拍）:
//     同一合成输入 sidecar → 落盘 %TEMP%\jmw_cmtt_accept_roundtrip_* 前缀
//     （TempDirGuard RAII, 模式复制 tests/test_golden_ref_tier.cpp）→ 文件字节
//     CmttReader::load → 逐 ok 档 tierBytes(i) 与内存 sidecarBytes 对应段
//     （头+索引之后按 offset/length 切; offset 为文件绝对偏移）逐字节相等;
//     missing 档 isMissing 语义保持（主合成无 missing → 以 tz 过零变体补验该分支）。
//   用例 3 ExtremeTierNoSystematicSlope —— 验收 #3（合成口径: 端档无系统斜率）:
//     纯各向同性缩放源表（节点 {25,30}, 端档全字段 s=1.005 派生; cte 放大到 1e-3
//     是判别力设计——物理 cte 23.6e-6 下 Δd~0.1 grid 淹没于 ±0.5 格量化）→ 21 档
//     sidecar → 参考档/端档各自 CurveMapTable::Load; 对应律（各向同性缩放严格几何）
//     d_T(row,uL) = s·D_ref(uL/s, row/s)——端档栅格 (row,uL) 与参考档连续栅格
//     (uL/s,row/s) 是同一射线、深度 s 倍。模型预测取参考档视差场在 (uL/s,row/s)
//     的双线性插值（而非同格点: 本夹具视差行梯度 ~1 grid/row, 同格点公式漏掉
//     ~(s−1)·(row·D_row+uL·D_u) 的位置漂移项——终审实测 ≈2.2 grid @row≈300、
//     row 150-200 处可达 ≈6.8, 均淹没 1.1 grid 容差）, 端档 Lookup 同 lid 候选
//     对拍: 逐点 |resid| ≤ 1.1 grid（预算 = 0.5 端档量化 + s·0.5 参考角点量化
//     + 余量）, 残差均值 |mean| < 0.5 grid——
//     系统性重标错误（锚点错位 ~|s−1|·|pp|≈1 grid / 漏重标 / 基线派生错）的判别。
//   用例 4 CorruptFileCleanReject —— 验收 #4（损坏文件干净拒绝）:
//     轻量 5 档 sidecar 落盘 %TEMP%\jmw_cmtt_accept_corrupt_*（RAII）后三种破坏
//     各一拷贝: 索引区首项 offset 字段最高字节翻转（→ load 拒绝, 或按实现容忍
//     "载入成功但逐档 nullopt"）/ 载荷区首 ok 档 blob 中部 1 字节翻转（→ 该档
//     crc32 拒绝, 邻档 tierBytes 完整可用）/ 截断一半（→ load 拒绝）; 全程无崩溃。
// 合成夹具口径复制自 tests/test_curve_map_temp_table.cpp（320x240 / 2 条近水平
// 抛物线 / f=500,B=40 / 温漂斜率 0.01、0.008 放大到量化可见, 非物理 cte）。

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../curve_map_temp_table.h"

using namespace calib;
namespace fs = std::filesystem;

namespace {

// ============================================================================
// 合成夹具（口径复制自 tests/test_curve_map_temp_table.cpp）
// ============================================================================

constexpr int kW = 320, kH = 240;
constexpr double kRefT = 25.0;
const cv::Size kImageSize{kW, kH};
const cv::Rect kBaseRoi{16, 16, 288, 208};          // 参考节点 validRoiLeft（冻结基准）

constexpr double kFRef = 500.0, kCx = 160.0, kCy = 120.0, kBRef = 40.0;
constexpr double kFSlope = 0.01, kBSlope = 0.008;   // 合成温漂斜率（放大到量化可见, 非物理 cte）
// 与单测夹具（test_curve_map_temp_table.cpp）唯一实质差异: virtualT 物理化。
// 单测口径 T≈(2,0.5,1) 时左↔虚拟基线≈1mm, 深度窗 [100,700] 只留下 ~1.4k 条目的
// 窄带（worstGap~358 格, 组在行中间消失）, 量化抖动即 ±3%——1% 全局平滑门禁在该
// 条目人口下无统计意义; tz=300 使曲线整体落在深度窗内, 条目人口达到设计预估量级
// （验收 #2 的 0.1-0.5% 才可判）。
const cv::Vec3d kVirtualT{2.0, 0.5, 300.0};

double fAt(double t) { return kFRef * (1.0 + kFSlope * (t - kRefT)); }
double bAt(double t) { return kBRef * (1.0 + kBSlope * (t - kRefT)); }

struct TierSpec {
    double t, f, cx, cy, B;
    cv::Rect roi;
    double tx, ty, tz;
};

/// 验收源表: 整 °C 节点 20..30 全覆盖 21 档网格 → 无 clamp、无 missing
std::vector<TierSpec> acceptTiers() {
    std::vector<TierSpec> out;
    for (int dt = -5; dt <= 5; ++dt) {
        const double t = kRefT + dt;
        out.push_back({t, fAt(t), kCx, kCy, bAt(t), kBaseRoi,
                       kVirtualT[0], kVirtualT[1], kVirtualT[2]});
    }
    return out;
}

/// missing 变体: 节点 25/26 的 tz 过零 → 网格 25.5°C 插值 tz=0（极点无穷远）→ 该档 Generate 失败
std::vector<TierSpec> acceptTiersTzCrossing() {
    auto out = acceptTiers();
    for (auto& s : out) {
        if (s.t == 25.0) s.tz = 0.4;
        else if (s.t == 26.0) s.tz = -0.4;
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

/// 21 档验收参数（25±5°C / 0.5 步距 → 21 档; rowStep/深度窗同既有夹具口径）
CurveMapTempTableGenParams acceptParams() {
    CurveMapTempTableGenParams p;
    p.referenceTemp = kRefT;
    p.tempHalfRange = 5.0f;
    p.tempStep = 0.5f;
    p.rowStep = 0.7f;
    p.depthMin = 100.0f;
    p.depthMax = 700.0f;
    return p;
}

// ============================================================================
// 用例 3 专属: 纯各向同性缩放源表 + 参考档连续视差场
// ============================================================================

/// 端档缩放因子（cte=1e-3/°C × ΔT=5°C）。判别力设计: 物理 cte 23.6e-6 下
/// 端档 Δd ~ 0.1 grid, 淹没于 ±0.5 格量化, 无法与重标错误区分; 放大到 1e-3 后
/// 正确实现的残差仍是纯量化带, 而系统性错误（如绕错误锚点缩放 |m|~|s−1|·|pp|
/// ≈ 1 grid 量级 / 漏重标 / 基线派生错）显著越出。
constexpr double kIsoS = 1.005;

/// 各向同性缩放源表: 两节点（25.0 基准 / 30.0 端档）, 端档全字段 s 派生——
/// f′=s·f（P1 缩放）、B′=s·B（makeP2 得 (0,3)=−s²·B·f, 插值器 |P2(0,3)|/f′
/// = s·B 恰为缩放基线）、pp′=s·pp（P1 主点缩放）、T′=s·T。物理含义: 纯各向同性
/// 热膨胀下归一化像几何不变, 像素坐标整体 q′=s·q——此时重标契约 m = pp′−s·pp
/// = 0（rescaleCurve 纯缩放退化路径）; 两端节点均为精确节点（零插值直取）。
std::vector<TierSpec> isoTiers() {
    return {
        {kRefT, kFRef, kCx, kCy, kBRef, kBaseRoi,
         kVirtualT[0], kVirtualT[1], kVirtualT[2]},
        {kRefT + 5.0, kIsoS * kFRef, kIsoS * kCx, kIsoS * kCy, kIsoS * kBRef,
         kBaseRoi, kIsoS * kVirtualT[0], kIsoS * kVirtualT[1], kIsoS * kVirtualT[2]},
    };
}

/// 参考档逐行视差场: row → (lid → (uL → d=uL−uR))。解码契约 d ≡ lround(真值 d)
/// 逐条目成立（curve_map.cpp 差分取整设计）→ 每角点量化误差 ≤ 0.5 grid。
/// 双根 uL（同 lid 双候选, 分支归属在缩放对应下不唯一）整格剔除, 保证插值单值。
using DisparityRow = std::map<uint8_t, std::map<uint16_t, int>>;

std::vector<DisparityRow> disparityField(const CurveMapTable& t) {
    std::vector<DisparityRow> rows(t.rowCount());
    std::vector<CurveMapEntry> buf(256);
    for (uint32_t row = 0; row < t.rowCount(); ++row) {
        int n = t.EnumerateRow(static_cast<int>(row), buf.data(),
                               static_cast<int>(buf.size()));
        while (n < 0) {   // 容量不足: 倍增重试（两段式）
            buf.resize(buf.size() * 2);
            n = t.EnumerateRow(static_cast<int>(row), buf.data(),
                               static_cast<int>(buf.size()));
        }
        for (int k = 0; k < n; ++k) {
            const auto& e = buf[static_cast<size_t>(k)];
            auto& m = rows[row][e.lid];
            const int d = static_cast<int>(e.uL) - static_cast<int>(e.uR);
            if (!m.emplace(e.uL, d).second)
                m.erase(e.uL);   // 双根: 该 uL 分支不唯一 → 剔除
        }
    }
    return rows;
}

/// 视差场双线性采样（(gu,gr) 连续栅格坐标; 行方向 v=0.7·row 精确 → gr=row/s 无
/// 偏差, 列方向 uLpx=0.7·(uL−0.143) 的锚定偏移经 (uL−δ)/s+δ 与 uL/s 相差 <1e-3
/// grid, 忽略）。四角须同为单值条目; 任一角缺失（弦端/行端/深度窗缘）返回 false。
bool bilinearDisparity(const std::vector<DisparityRow>& rows,
                       double gu, double gr, uint8_t lid, double& dOut) {
    const int r0 = static_cast<int>(std::floor(gr));
    const int g0 = static_cast<int>(std::floor(gu));
    if (r0 < 0 || static_cast<size_t>(r0) + 1 >= rows.size()) return false;
    const double fr = gr - r0, fu = gu - g0;
    double acc[2][2];
    for (int ri = 0; ri < 2; ++ri) {
        const auto itLid = rows[static_cast<size_t>(r0 + ri)].find(lid);
        if (itLid == rows[static_cast<size_t>(r0 + ri)].end()) return false;
        for (int gi = 0; gi < 2; ++gi) {
            const auto j = itLid->second.find(static_cast<uint16_t>(g0 + gi));
            if (j == itLid->second.end()) return false;
            acc[ri][gi] = static_cast<double>(j->second);
        }
    }
    dOut = (1.0 - fr) * ((1.0 - fu) * acc[0][0] + fu * acc[0][1]) +
              fr  * ((1.0 - fu) * acc[1][0] + fu * acc[1][1]);
    return true;
}

// ============================================================================
// 临时目录 RAII + 二进制读写（模式复制自 tests/test_golden_ref_tier.cpp）
// ============================================================================

/// 临时目录 RAII 守卫（ASSERT 失败提前返回也不泄漏 %TEMP%\jmw_cmtt_accept_roundtrip_*）
struct TempDirGuard {
    fs::path dir;
    ~TempDirGuard() {
        std::error_code ec;
        fs::remove_all(dir, ec);   // 清理失败不抛（析构路径）
    }
};

std::vector<uint8_t> readBinFile(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
}

void writeBinFile(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream f(p, std::ios::binary);
    ASSERT_TRUE(f.good()) << "cannot open " << p.string();
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(f.good()) << "write failed: " << p.string();
}

// ============================================================================
// 单组统计与文件往返对拍
// ============================================================================

using GroupKey = std::pair<uint32_t, uint8_t>;   // (绝对行号, 激光线号)
using GroupCounts = std::map<GroupKey, uint32_t>;

/// EnumerateRow 全行枚举（两段式容量模式: 返回 -1 → 缓冲倍增重试）→ (row,lid) 聚合 count
GroupCounts groupCounts(const CurveMapTable& t) {
    GroupCounts m;
    std::vector<CurveMapEntry> buf(256);
    for (uint32_t row = 0; row < t.rowCount(); ++row) {
        int n = t.EnumerateRow(static_cast<int>(row), buf.data(),
                               static_cast<int>(buf.size()));
        while (n < 0) {   // 容量不足: 倍增重试（两段式）
            buf.resize(buf.size() * 2);
            n = t.EnumerateRow(static_cast<int>(row), buf.data(),
                               static_cast<int>(buf.size()));
        }
        for (int k = 0; k < n; ++k)
            ++m[{row, buf[static_cast<size_t>(k)].lid}];
    }
    return m;
}

/// 文件侧 Reader 逐档 blob 与内存 sidecar 对应段逐字节对拍:
/// 内存 sidecar 直解 96B 头 + tierCount×32B 索引, 按 offset/length 切段
/// （offset 为文件绝对偏移）; missing 档: 无载荷 + 索引三零（offset/length/crc32）。
void expectFileRoundtripEqualsMemory(const CmttReader& rd,
                                     const std::vector<uint8_t>& sidecar) {
    ASSERT_GE(sidecar.size(),
              sizeof(CmttHeader) + rd.tierCount() * sizeof(CmttTierMeta));
    std::vector<CmttTierMeta> idx(rd.tierCount());
    std::memcpy(idx.data(), sidecar.data() + sizeof(CmttHeader),
                idx.size() * sizeof(CmttTierMeta));
    for (size_t i = 0; i < rd.tierCount(); ++i) {
        SCOPED_TRACE("tier " + std::to_string(i));
        const auto b = rd.tierBytes(i);   // 文件侧取档（拷出 + crc32 验证）
        if (rd.isMissing(i)) {
            EXPECT_FALSE(b.has_value());
            EXPECT_EQ(idx[i].offset, 0u);
            EXPECT_EQ(idx[i].length, 0u);
            EXPECT_EQ(idx[i].crc32, 0u);
            continue;
        }
        ASSERT_TRUE(b.has_value());
        ASSERT_FALSE(b->empty());
        ASSERT_EQ(b->size(), idx[i].length);
        EXPECT_EQ(std::memcmp(b->data(), sidecar.data() + idx[i].offset,
                              idx[i].length), 0);
    }
}

} // namespace

// ============================================================================
// 用例 1: 全局平滑 + 单组白名单统计（设计验收 #2）
// ============================================================================

TEST(Accept, SmoothnessGlobalAndPerGroup) {
    CurveMapTempTableGenerator gen(acceptParams());
    const auto curves = makeCurves();
    const auto rect = makeRectify(acceptTiers());
    const auto laser = makeLaser(acceptTiers());
    auto r = gen.Generate(curves, rect, laser, kImageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{21});          // 25±5°C / 0.5 → 21 档
    ASSERT_EQ(r.okCount(), 21);                     // 源表全覆盖 → 无 missing
    ASSERT_EQ(r.referenceIndex(), size_t{10});      // 25.0°C 居中
    EXPECT_EQ(r.qualityFlag, QualityFlag::Normal);  // 无 missing/clamped

    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));
    ASSERT_EQ(rd.tierCount(), size_t{21});

    // ---- 全局断言: 相邻档 entryCount |Δ|/N < 1%（理论预估 0.1-0.5%） ----
    for (size_t i = 0; i + 1 < rd.tierCount(); ++i) {
        const double c1 = static_cast<double>(rd.entryCount(i));
        const double c2 = static_cast<double>(rd.entryCount(i + 1));
        const double N = std::max(1.0, std::max(c1, c2));
        EXPECT_LT(std::fabs(c2 - c1) / N, 0.01)
            << "adjacent tiers " << i << " (" << r.tiers[i].temperature
            << "C) -> " << (i + 1) << " (" << r.tiers[i + 1].temperature
            << "C): entryCount " << c1 << " vs " << c2;
    }

    // ---- 单组统计: 逐档载入 + 全行枚举 → (row,lid) 聚合 count ----
    std::vector<GroupCounts> perTier(rd.tierCount());
    for (size_t i = 0; i < rd.tierCount(); ++i) {
        const auto blob = rd.tierBytes(i);
        ASSERT_TRUE(blob.has_value()) << "tier " << i;
        CurveMapTable t;
        std::string err;
        ASSERT_TRUE(t.Load(blob->data(), blob->size(), err))
            << "tier " << i << ": " << err;
        perTier[i] = groupCounts(t);
        EXPECT_EQ(t.entryCount(), rd.entryCount(i))   // 枚举无遗漏的旁证
            << "tier " << i;
    }

    // ---- 相邻档逐组 count 变化; 超 10% 进跳变清单 ----
    // 注: FOV 边缘短弦组属预期白名单（设计验收 #2）——清单仅输出, 不判失败。
    constexpr double kGroupJumpRel = 0.10;
    constexpr size_t kMaxPrintedJumps = 32;
    size_t jumpCount = 0, printed = 0;
    for (size_t i = 0; i + 1 < perTier.size(); ++i) {
        std::set<GroupKey> keys;                     // 键并集（组出现/消失也算跳变）
        for (const auto& kv : perTier[i]) keys.insert(kv.first);
        for (const auto& kv : perTier[i + 1]) keys.insert(kv.first);
        for (const auto& k : keys) {
            const auto it1 = perTier[i].find(k);
            const auto it2 = perTier[i + 1].find(k);
            const uint32_t c1 = it1 == perTier[i].end() ? 0u : it1->second;
            const uint32_t c2 = it2 == perTier[i + 1].end() ? 0u : it2->second;
            const double N = std::max(1.0, std::max(static_cast<double>(c1),
                                                    static_cast<double>(c2)));
            const double rel = std::fabs(static_cast<double>(c2) -
                                         static_cast<double>(c1)) / N;
            if (rel <= kGroupJumpRel) continue;
            ++jumpCount;
            if (printed < kMaxPrintedJumps) {
                ++printed;
                std::cout << "[whitelist][Accept] tier " << i << "->" << (i + 1)
                          << " (" << r.tiers[i].temperature << "C->"
                          << r.tiers[i + 1].temperature << "C) row=" << k.first
                          << " lid=" << static_cast<int>(k.second)
                          << ": count " << c1 << " -> " << c2 << "\n";
            }
        }
    }
    if (jumpCount > printed)
        std::cout << "[whitelist][Accept] ... and " << (jumpCount - printed)
                  << " more jump groups (FOV edge short chords, expected)\n";
    // 记录成功（附消息）而非失败: 白名单清单语义
    SUCCEED() << "per-(row,lid) jump list: " << jumpCount
              << " groups >10% across 20 adjacent tier pairs"
                 " (FOV edge short chords whitelisted, by design)";
}

// ============================================================================
// 用例 2: 合成金样写后读回往返（设计验收 #6）
// ============================================================================

TEST(Accept, SyntheticGoldenRoundtrip) {
    // ---- 同一合成输入（用例 1 口径）→ sidecar ----
    CurveMapTempTableGenerator gen(acceptParams());
    auto r = gen.Generate(makeCurves(), makeRectify(acceptTiers()),
                          makeLaser(acceptTiers()), kImageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{21});
    ASSERT_EQ(r.okCount(), 21);
    ASSERT_FALSE(r.sidecarBytes.empty());

    // ---- 落盘临时文件（%TEMP%\jmw_cmtt_accept_roundtrip_* 前缀, TempDirGuard RAII） ----
    const fs::path dir = fs::temp_directory_path() /
        ("jmw_cmtt_accept_roundtrip_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    const TempDirGuard dirGuard{dir};
    ASSERT_TRUE(fs::create_directories(dir));
    const fs::path sidecarPath = dir / "curve_map_temp_table.cmtt";
    writeBinFile(sidecarPath, r.sidecarBytes);

    // ---- 文件字节 → CmttReader（写后读回） ----
    const std::vector<uint8_t> fileBytes = readBinFile(sidecarPath);
    ASSERT_EQ(fileBytes.size(), r.sidecarBytes.size());
    CmttReader rd;
    ASSERT_TRUE(rd.load(fileBytes.data(), fileBytes.size()));
    ASSERT_EQ(rd.tierCount(), r.tiers.size());

    // 结果簿记与文件侧一致性: ok ↔ 非 missing
    for (size_t i = 0; i < rd.tierCount(); ++i)
        EXPECT_EQ(rd.isMissing(i), !r.tiers[i].ok) << "tier " << i;

    // 逐 ok 档: 文件侧 tierBytes(i) 与内存 sidecar 对应段（按 offset/length 切）逐字节相等
    expectFileRoundtripEqualsMemory(rd, r.sidecarBytes);

    // ---- missing 档语义补验: 主合成无 missing → tz 过零变体（25.5°C 档）走同一往返 ----
    auto rm = gen.Generate(makeCurves(), makeRectify(acceptTiers()),
                           makeLaser(acceptTiersTzCrossing()), kImageSize);
    ASSERT_TRUE(rm.success) << rm.message;           // 非参考档 missing 宽容
    ASSERT_EQ(rm.tiers.size(), size_t{21});
    ASSERT_EQ(rm.okCount(), 20);
    const size_t missIdx = 11;                       // 25.5°C: 节点 25/26 tz 过零插值 tz=0
    ASSERT_DOUBLE_EQ(rm.tiers[missIdx].temperature, 25.5);
    EXPECT_FALSE(rm.tiers[missIdx].ok);
    EXPECT_NE(rm.tiers[missIdx].flags & kCmttFlagMissing, 0);
    EXPECT_TRUE(rm.tiers[rm.referenceIndex()].ok);   // 参考档仍在

    const fs::path missPath = dir / "curve_map_temp_table_missing.cmtt";
    writeBinFile(missPath, rm.sidecarBytes);
    const std::vector<uint8_t> missBytes = readBinFile(missPath);   // Reader 非拥有视图 → 须驻留
    CmttReader rdm;
    ASSERT_TRUE(rdm.load(missBytes.data(), missBytes.size()));
    ASSERT_EQ(rdm.tierCount(), size_t{21});
    EXPECT_TRUE(rdm.isMissing(missIdx));
    EXPECT_FALSE(rdm.isMissing(rm.referenceIndex()));
    EXPECT_FALSE(rdm.tierBytes(missIdx).has_value());   // missing 档无载荷
    expectFileRoundtripEqualsMemory(rdm, rm.sidecarBytes);
}

// ============================================================================
// 用例 3: 端档纯各向同性缩放 · 无系统斜率（设计验收 #3 合成口径）
// ============================================================================

TEST(Accept, ExtremeTierNoSystematicSlope) {
    // ---- 各向同性缩放源表 → 21 档 sidecar（25±5°C; 节点 {25,30} → 冷侧档 clamp
    //      至参考节点直取（s=1, 生成不受影响）, 30.0 端档为精确节点零插值 ----
    CurveMapTempTableGenerator gen(acceptParams());
    const auto iso = isoTiers();
    auto r = gen.Generate(makeCurves(), makeRectify(iso), makeLaser(iso),
                          kImageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{21});
    ASSERT_EQ(r.okCount(), 21);
    ASSERT_EQ(r.referenceIndex(), size_t{10});
    const size_t endIdx = 20;                       // 30.0°C 端档
    ASSERT_DOUBLE_EQ(r.tiers[endIdx].temperature, 30.0);
    ASSERT_TRUE(r.tiers[endIdx].ok);
    EXPECT_EQ(r.tiers[endIdx].flags & kCmttFlagClamped, 0);   // 精确节点不 clamp

    CmttReader rd;
    ASSERT_TRUE(rd.load(r.sidecarBytes.data(), r.sidecarBytes.size()));

    CurveMapTable tRef, tEnd;
    std::string err;
    const auto refBlob = rd.tierBytes(r.referenceIndex());
    const auto endBlob = rd.tierBytes(endIdx);
    ASSERT_TRUE(refBlob.has_value());
    ASSERT_TRUE(endBlob.has_value());
    ASSERT_TRUE(tRef.Load(refBlob->data(), refBlob->size(), err)) << err;
    ASSERT_TRUE(tEnd.Load(endBlob->data(), endBlob->size(), err)) << err;

    // ---- 参考档视差场 + 端档逐点对拍 ----
    // 对应律: d_T(row,uL) = s·D_ref(uL/s, row/s)（文件头注释推导）; 模型预测
    // uR_pred = uL − s·D̂(uL/s, row/s), D̂ 为视差场双线性插值。
    const auto refField = disparityField(tRef);

    struct Sample { uint32_t row; uint16_t uL; uint8_t lid; };
    std::vector<Sample> all;                        // 参考档全部单值条目坐标
    for (uint32_t row = 0; row < refField.size(); ++row)
        for (const auto& lv : refField[row])
            for (const auto& uv : lv.second)
                all.push_back({row, uv.first, lv.first});
    ASSERT_GT(all.size(), size_t{1000});
    const size_t stride = std::max<size_t>(1, all.size() / 300);

    CurveMapCand cands[64];
    size_t used = 0, skipBilin = 0, skipTier = 0;
    double sum = 0.0, sumSq = 0.0, maxAbs = 0.0;
    for (size_t i = 0; i < all.size() && used < 300; i += stride) {
        const auto& smp = all[i];

        // 端档 Lookup: 同 lid 候选恰 1 个才可对拍（双根分支归属在缩放对应下
        // 不唯一 → 跳过, 稳定规则; 本夹具近水平线束下双根罕见）
        const int n = tEnd.Lookup(static_cast<int>(smp.row), smp.uL, cands, 64);
        int hit = -1, nLid = 0;
        for (int k = 0; k < n; ++k) {
            if (cands[k].lid != smp.lid) continue;
            ++nLid;
            if (hit < 0) hit = k;
        }
        if (nLid != 1) {
            ++skipTier;
            continue;
        }

        double dHat;
        if (!bilinearDisparity(refField, smp.uL / kIsoS, smp.row / kIsoS,
                               smp.lid, dHat)) {
            ++skipBilin;
            continue;
        }

        const double uRpred = static_cast<double>(smp.uL) - kIsoS * dHat;
        const double resid = static_cast<double>(cands[hit].uR) - uRpred;
        EXPECT_LE(std::fabs(resid), 1.1)
            << "row=" << smp.row << " uL=" << smp.uL
            << " lid=" << static_cast<int>(smp.lid)
            << " uR_T=" << cands[hit].uR << " uR_pred=" << uRpred;
        sum += resid;
        sumSq += resid * resid;
        maxAbs = std::max(maxAbs, std::fabs(resid));
        ++used;
    }
    ASSERT_GE(used, size_t{100});

    // 残差均值门禁（判别核心）: 正确实现下残差 ≈ 端档量化 − s×参考角点量化,
    // 均值期望 0、σ_mean ≈ 0.3/√used ≤ 0.03 → |mean| < 0.5 有 >15σ 裕量;
    // 系统性重标错误（漏重标/反向缩放/错误锚点 |m|~|s−1|·|pp|≈1 grid/基线派生错）
    // 产生 O(1) grid 均值漂移 → 被 0.5 门限大声拒绝。
    const double mean = sum / static_cast<double>(used);
    EXPECT_LT(std::fabs(mean), 0.5);
    const double sigma = std::sqrt(std::max(0.0, sumSq / static_cast<double>(used)
                                             - mean * mean));
    std::cout << "[accept][case3] samples=" << used
              << " (skipBilin=" << skipBilin << " skipTier=" << skipTier
              << ") resid: mean=" << mean << " sigma=" << sigma
              << " maxAbs=" << maxAbs << "\n";
    SUCCEED() << "end-tier slope check over " << used
              << " samples: |mean|=" << std::fabs(mean) << " < 0.5 grid, maxAbs="
              << maxAbs << " <= 1.1 grid (quantization budget)";
}

// ============================================================================
// 用例 4: 损坏文件干净拒绝（设计验收 #4）
// ============================================================================

TEST(Accept, CorruptFileCleanReject) {
    // ---- 轻量 5 档（25±1°C; 源表 20..30°C 全覆盖 → 无 clamp/missing） ----
    CurveMapTempTableGenParams p = acceptParams();
    p.tempHalfRange = 1.0f;
    CurveMapTempTableGenerator gen(p);
    auto r = gen.Generate(makeCurves(), makeRectify(acceptTiers()),
                          makeLaser(acceptTiers()), kImageSize);
    ASSERT_TRUE(r.success) << r.message;
    ASSERT_EQ(r.tiers.size(), size_t{5});
    ASSERT_EQ(r.okCount(), 5);

    // ---- 落盘临时文件（%TEMP%\jmw_cmtt_accept_corrupt_* 前缀, TempDirGuard RAII） ----
    const fs::path dir = fs::temp_directory_path() /
        ("jmw_cmtt_accept_corrupt_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    const TempDirGuard dirGuard{dir};
    ASSERT_TRUE(fs::create_directories(dir));
    const fs::path okPath = dir / "ok.cmtt";
    writeBinFile(okPath, r.sidecarBytes);
    const std::vector<uint8_t> fileBytes = readBinFile(okPath);   // 写后读回再破坏
    ASSERT_EQ(fileBytes.size(), r.sidecarBytes.size());

    // 索引首项（载荷破坏定位用; offset 为文件绝对偏移, 同对拍函数口径）
    CmttTierMeta idx0{};
    std::memcpy(&idx0, fileBytes.data() + sizeof(CmttHeader), sizeof(idx0));
    ASSERT_EQ(idx0.flags & kCmttFlagMissing, 0);   // 首 ok 档
    ASSERT_GT(idx0.length, 0u);

    // ---- 破坏 ①: 索引区首项 offset 字段最高字节（文件偏移 96+11）翻转 ----
    // offset 变 ~2^63 → load 的逐索引项 offset/length≤size 边界检查干净拒绝;
    // 断言按两分支容错（load false, 或实现改为载入成功但逐档 nullopt）。
    {
        auto bad = fileBytes;
        bad[sizeof(CmttHeader) + 11] ^= 0x80;
        const fs::path badPath = dir / "corrupt_index.cmtt";
        writeBinFile(badPath, bad);
        const auto badBytes = readBinFile(badPath);
        CmttReader rd;
        const bool loaded = rd.load(badBytes.data(), badBytes.size());
        if (loaded) {
            for (size_t i = 0; i < rd.tierCount(); ++i)
                EXPECT_FALSE(rd.tierBytes(i).has_value()) << "tier " << i;
        } else {
            SUCCEED() << "index corruption cleanly rejected at load";
        }
    }

    // ---- 破坏 ②: 载荷区首 ok 档 blob 中部 1 字节翻转 ----
    // 索引完好 → load 成功; 该档 crc32 失配只拒该档（懒校验, 未触碰档不付费）,
    // 邻档 tierBytes 完整可用（含参考档完整 Load 的可用性实证）。
    {
        auto bad = fileBytes;
        bad[static_cast<size_t>(idx0.offset + idx0.length / 2)] ^= 0xA5;
        const fs::path badPath = dir / "corrupt_payload.cmtt";
        writeBinFile(badPath, bad);
        const auto badBytes = readBinFile(badPath);
        CmttReader rd;
        EXPECT_TRUE(rd.load(badBytes.data(), badBytes.size()));
        EXPECT_FALSE(rd.tierBytes(0).has_value());      // 损坏档: crc 拒绝
        for (size_t i = 1; i < rd.tierCount(); ++i) {
            const auto b = rd.tierBytes(i);
            EXPECT_TRUE(b.has_value()) << "neighbor tier " << i;
            EXPECT_FALSE(b->empty());
        }
        const auto refBlob = rd.tierBytes(2);           // 参考档（25.0°C）
        ASSERT_TRUE(refBlob.has_value());
        CurveMapTable t;
        std::string err;
        ASSERT_TRUE(t.Load(refBlob->data(), refBlob->size(), err)) << err;
        EXPECT_EQ(t.entryCount(), rd.entryCount(2));
    }

    // ---- 破坏 ③: 截断一半 → 头/索引之外必有档越界 → load 拒绝 ----
    {
        auto bad = fileBytes;
        bad.resize(bad.size() / 2);
        const fs::path badPath = dir / "truncated.cmtt";
        writeBinFile(badPath, bad);
        const auto badBytes = readBinFile(badPath);
        CmttReader rd;
        EXPECT_FALSE(rd.load(badBytes.data(), badBytes.size()));
    }
    // 全程无崩溃（gtest 正常走完即隐含通过）
}
