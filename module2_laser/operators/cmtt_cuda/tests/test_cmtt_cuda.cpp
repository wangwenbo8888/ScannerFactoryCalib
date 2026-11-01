// test_cmtt_cuda.cpp — CMTT CUDA 试验版测试
//
// 1) SyntheticSingleTierMatchesCpu: 合成曲线小图, CUDA vs CPU 单档等价性
//    （entryCount 相等 + CurveMapTable 双向可解析 + 逐行条目语义比对）
// 2) SyntheticCmtt9TiersVsCpu: 合成两源表, 9 档 CMTT CUDA vs 生产 CPU 生成器
//    （档数/逐档 entryCount/参考档/CmttReader 复核）
// 3) RealDataRefTierVsSidecar: 真实数据（laser261006-1668 产物在位时）
//    参考档 CUDA 生成 vs 已交付 sidecar 参考档（全尺度 3930 万条目对拍）+ 计时
// 4) RealDataFullCmttTiming: 61 档全量计时（仅环境变量 FC_CMTT_CUDA_FULL=1 时跑）

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "curve_map_cuda.h"
#include "cmtt_cuda.h"
#include "curve_map.h"
#include "curve_map_temp_table.h"
#include "cmtt_container.h"
#include "temp_param_interpolator.h"
#include "stereo_rectify_temp_table_cpu.h"
#include "laser_extrinsic_compensate_cpu.h"
#include "calib_io.h"   // fc::loadCameraCalibHandoff（生产解析器, 与 CLI 同源）

using namespace calib;
namespace fs = std::filesystem;

// ============================================================================
// 合成数据辅助
// ============================================================================

namespace {

// 椭圆带曲线族: 以 (cx0, cy) 为中心的椭圆, 系数 A·u² + C·v² − 1 归一形式
ImplicitCurve makeEllipse(double cu, double cv, double ru, double rv) {
    ImplicitCurve c;
    const double a = 1.0 / (ru * ru);
    const double d = 1.0 / (rv * rv);
    c.coeffs[0] = a;     // A u²
    c.coeffs[1] = 0.0;   // B uv
    c.coeffs[2] = d;     // C v²
    c.coeffs[3] = -2.0 * a * cu;
    c.coeffs[4] = -2.0 * d * cv;
    c.coeffs[5] = a * cu * cu + d * cv * cv - 1.0;
    return c;
}

CurveMapInput makeSyntheticInput(int nCurves = 3) {
    CurveMapInput in;
    for (int i = 0; i < nCurves; ++i)
        in.curves.push_back(makeEllipse(320.0, 120.0 + 120.0 * i, 210.0, 40.0));
    in.f = 800.0;
    in.principalPoint = cv::Point2d(320.0, 240.0);
    in.baseline = 60.0;
    in.virtualT = cv::Vec3d(40.0, 4.0, -380.0);
    in.imageSize = cv::Size(640, 480);
    in.roi = cv::Rect(10, 10, 620, 460);
    return in;
}

// 合成模块1 立体矫正温表（5 节点 ±2°C, f 随温漂移）
StereoRectifyTempTableResult makeSyntheticRectTable(double refTemp, double f0,
                                                    const cv::Rect& roi) {
    StereoRectifyTempTableResult r;
    r.success = true;
    r.referenceTemp = refTemp;
    r.cte = 2.36e-5;
    r.tableSize = 5;
    for (int k = -2; k <= 2; ++k) {
        StereoRectifyTempEntry e;
        e.temperature = refTemp + k;
        e.deltaT = k;
        const double f = f0 * (1.0 + 2.36e-5 * k);
        e.P1 = (cv::Mat_<double>(3, 3) << f, 0, 320, 0, f, 240, 0, 0, 1);
        e.P2 = (cv::Mat_<double>(3, 3) << f, 0, 320, 0, f, 240, 0, 0, 1);
        e.P2.at<double>(0, 3 - 3) = 0;   // 3x3 无平移列; 下行补 4 列形式
        e.P2 = (cv::Mat_<double>(3, 4) << f, 0, 320, -f * 60.0,
                                             0, f, 240, 0,
                                             0, 0, 1, 0);
        e.validRoiLeft = roi;
        e.validRoiRight = roi;
        r.table.push_back(e);
    }
    return r;
}

// 合成 5-3 激光外参温表（virtual→left/right 同 T, 随温小幅缩放）
LaserExtrinsicCompensateCPUResult makeSyntheticLaserTable(double refTemp,
                                                          cv::Vec3d T0) {
    LaserExtrinsicCompensateCPUResult r;
    r.success = true;
    r.referenceTemp = refTemp;
    r.cte = 2.36e-5;
    for (int side = 0; side < 2; ++side) {
        ExtrinsicCompensateCPUResult& res = side == 0 ? r.leftResult : r.rightResult;
        res.success = true;
        res.referenceTemp = refTemp;
        res.cte = 2.36e-5;
        res.baselineRef = 60.0;
        for (int k = -2; k <= 2; ++k) {
            ExtrinsicCompensatedEntry e;
            e.temperature = refTemp + k;
            e.deltaT = k;
            const double s = 1.0 + 2.36e-5 * k;
            e.T[0] = T0[0] * s; e.T[1] = T0[1] * s; e.T[2] = T0[2] * s;
            e.R[0] = e.R[4] = e.R[8] = 1.0;
            res.table.push_back(e);
        }
    }
    return r;
}

double nowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 逐行条目收集（经 CurveMapTable 解码, 语义比对用）
std::vector<std::vector<CurveMapEntry>> enumerateAll(const CurveMapResult& r) {
    std::vector<std::vector<CurveMapEntry>> out;
    std::string err;
    CurveMapTable t;
    if (!t.Load(r.tableBytes.data(), r.tableBytes.size(), err)) {
        ADD_FAILURE() << "Load failed: " << err;
        return out;
    }
    std::vector<CurveMapEntry> buf(65536);
    for (uint32_t row = 0; row < t.rowCount(); ++row) {
        const int n = t.EnumerateRow(static_cast<int>(row), buf.data(), (int)buf.size());
        if (n <= 0) continue;
        out.emplace_back(buf.begin(), buf.begin() + n);
    }
    return out;
}

} // namespace

// ============================================================================
// 1) 单档等价性（合成）
// ============================================================================

TEST(CmttCuda, SyntheticSingleTierMatchesCpu) {
    const CurveMapInput in = makeSyntheticInput();
    CurveMapParams prm;                     // 默认 rowStep 0.7 / 深度窗 100..700
    prm.depthMax = 5000.0;                  // 放宽深度窗保证合成曲线有命中

    CurveMapGenerator cpu(prm);
    const auto t0 = nowMs();
    CurveMapResult rc = cpu.Generate(in);
    const double cpuMs = nowMs() - t0;
    ASSERT_TRUE(rc.success) << rc.message;

    CurveMapCuda gpu(prm);
    const auto t1 = nowMs();
    CurveMapResult rg = gpu.Generate(in);
    const double gpuMs = nowMs() - t1;
    ASSERT_TRUE(rg.success) << rg.message;

    printf("[perf] single tier: cpu=%.1fms cuda=%.1fms entries cpu=%u cuda=%u\n",
           cpuMs, gpuMs, rc.entryCount, rg.entryCount);

    ASSERT_GT(rc.entryCount, 1000u) << "合成数据命中过少, 测试无效";
    // 容差: 浮点边界条目差 ≤ 0.1% 或 ≤3 条
    const long long diff = llabs((long long)rc.entryCount - (long long)rg.entryCount);
    EXPECT_LE(diff, std::max<long long>(3, (long long)(rc.entryCount * 0.001)))
        << "cpu=" << rc.entryCount << " cuda=" << rg.entryCount;

    // 行级语义比对
    auto rowsC = enumerateAll(rc);
    auto rowsG = enumerateAll(rg);
    ASSERT_EQ(rowsC.size(), rowsG.size());
    size_t rowMismatches = 0, entryCompared = 0, entryMismatched = 0;
    for (size_t r = 0; r < rowsC.size(); ++r) {
        if (rowsC[r].size() != rowsG[r].size()) { ++rowMismatches; continue; }
        for (size_t i = 0; i < rowsC[r].size(); ++i) {
            ++entryCompared;
            const auto& a = rowsC[r][i];
            const auto& b = rowsG[r][i];
            if (a.lid != b.lid || a.uL != b.uL
                || std::abs((int)a.uR - (int)b.uR) > 1) {
                ++entryMismatched;
            }
        }
    }
    EXPECT_EQ(rowMismatches, 0u);
    EXPECT_LE(entryMismatched, entryCompared / 1000 + 1)
        << "entryMismatched=" << entryMismatched << "/" << entryCompared;
}

// ============================================================================
// 2) 合成 CMTT 9 档: CUDA vs 生产 CPU 生成器
// ============================================================================

TEST(CmttCuda, SyntheticCmtt9TiersVsCpu) {
    const double refTemp = 22.5;
    const cv::Rect roi(10, 10, 620, 460);
    auto curves = makeSyntheticInput().curves;
    auto rectTable = makeSyntheticRectTable(refTemp, 800.0, roi);
    auto laserTable = makeSyntheticLaserTable(refTemp, cv::Vec3d(40.0, 4.0, -380.0));

    CurveMapTempTableGenParams gp;
    gp.referenceTemp = refTemp;
    gp.tempHalfRange = 2.0f;
    gp.tempStep = 0.5f;
    gp.rowStep = 0.7f;
    gp.depthMin = 100.0f;
    gp.depthMax = 5000.0f;
    gp.tierThreads = 1;

    // 生产 CPU 生成器作参照
    CurveMapTempTableGenerator cpuGen(gp);
    const auto t0 = nowMs();
    auto rc = cpuGen.Generate(curves, rectTable, laserTable, cv::Size(640, 480));
    const double cpuMs = nowMs() - t0;
    ASSERT_TRUE(rc.success) << rc.message;

    // CUDA 生成器
    CmttCudaGenerator gpuGen(gp);
    const auto t1 = nowMs();
    auto rg = gpuGen.Generate(curves, rectTable, laserTable, cv::Size(640, 480));
    const double gpuMs = nowMs() - t1;
    ASSERT_TRUE(rg.success) << rg.message;

    printf("[perf] cmtt 9 tiers: cpu=%.1fms cuda=%.1fms speedup=%.2fx\n",
           cpuMs, gpuMs, cpuMs / gpuMs);

    ASSERT_EQ(rg.tiers.size(), rc.tiers.size());
    EXPECT_EQ(rg.okCount(), rc.okCount());
    EXPECT_EQ(rg.referenceIndex(), rc.referenceIndex());

    // 逐档 entryCount（容差同单档）
    for (size_t i = 0; i < rc.tiers.size(); ++i) {
        ASSERT_EQ(rc.tiers[i].ok, rg.tiers[i].ok)
            << "tier " << i << " ok mismatch";
        if (!rc.tiers[i].ok) continue;
        const long long diff = llabs((long long)rc.tiers[i].entryCount
                                   - (long long)rg.tiers[i].entryCount);
        EXPECT_LE(diff, std::max<long long>(3,
                   (long long)(rc.tiers[i].entryCount * 0.001)))
            << "tier " << i << " temp=" << rc.tiers[i].temperature;
    }

    // sidecar 结构复核
    CmttReader rd;
    ASSERT_TRUE(rd.load(rg.sidecarBytes.data(), rg.sidecarBytes.size()));
    EXPECT_EQ(rd.tierCount(), rg.tiers.size());
    std::optional<size_t> refIdx;
    for (size_t i = 0; i < rd.tierCount(); ++i)
        if (rd.isReference(i)) refIdx = i;
    ASSERT_TRUE(refIdx.has_value());
    auto refBlob = rd.tierBytes(*refIdx);
    ASSERT_TRUE(refBlob.has_value());
    EXPECT_EQ(rd.entryCount(*refIdx), rg.tiers[*refIdx].entryCount);
}

// ============================================================================
// 2b) 非对称档位方案（2026-10-13）: tiersBelow/tiersAbove 覆盖对称半范围路径
//     结构: 下 2 档 / 上 7 档 / 步距 1.0 → 共 10 档, 参考档下标 2
// ============================================================================

TEST(CmttCuda, SyntheticAsymmetricTiers) {
    const double refTemp = 22.5;
    const cv::Rect roi(10, 10, 620, 460);
    auto curves = makeSyntheticInput().curves;
    auto rectTable = makeSyntheticRectTable(refTemp, 800.0, roi);
    auto laserTable = makeSyntheticLaserTable(refTemp, cv::Vec3d(40.0, 4.0, -380.0));

    CurveMapTempTableGenParams gp;
    gp.referenceTemp = refTemp;
    gp.tiersBelow = 2;          // 非对称模式: 覆盖 tempHalfRange 对称路径
    gp.tiersAbove = 7;
    gp.tempStep = 1.0f;
    gp.rowStep = 0.67f;
    gp.depthMin = 100.0f;
    gp.depthMax = 5000.0f;
    gp.tierThreads = 1;

    CurveMapTempTableGenerator cpuGen(gp);
    auto rc = cpuGen.Generate(curves, rectTable, laserTable, cv::Size(640, 480));
    ASSERT_TRUE(rc.success) << rc.message;

    ASSERT_EQ(rc.tiers.size(), size_t{10});
    EXPECT_DOUBLE_EQ(rc.tiers.front().temperature, refTemp - 2.0);
    EXPECT_DOUBLE_EQ(rc.tiers[2].temperature, refTemp);
    EXPECT_DOUBLE_EQ(rc.tiers.back().temperature, refTemp + 7.0);
    EXPECT_EQ(rc.referenceIndex(), size_t{2});
    ASSERT_TRUE(rc.tiers[rc.referenceIndex()].ok);

    CmttCudaGenerator gpuGen(gp);
    auto rg = gpuGen.Generate(curves, rectTable, laserTable, cv::Size(640, 480));
    ASSERT_TRUE(rg.success) << rg.message;
    ASSERT_EQ(rg.tiers.size(), rc.tiers.size());
    for (size_t i = 0; i < rc.tiers.size(); ++i) {
        EXPECT_DOUBLE_EQ(rg.tiers[i].temperature, rc.tiers[i].temperature);
        ASSERT_EQ(rg.tiers[i].ok, rc.tiers[i].ok) << "tier " << i;
        if (!rc.tiers[i].ok) continue;
        const long long diff = llabs((long long)rc.tiers[i].entryCount
                                   - (long long)rg.tiers[i].entryCount);
        EXPECT_LE(diff, std::max<long long>(3,
                   (long long)(rc.tiers[i].entryCount * 0.001)))
            << "tier " << i << " temp=" << rc.tiers[i].temperature;
    }

    CmttReader rd;
    ASSERT_TRUE(rd.load(rg.sidecarBytes.data(), rg.sidecarBytes.size()));
    EXPECT_EQ(rd.tierCount(), size_t{10});
    EXPECT_EQ(rd.header().tempStepX10, 10);   // 1.0C ×10
}

// ============================================================================
// 3) 真实数据参考档 vs 已交付 sidecar（全尺度对拍）
// ============================================================================

namespace {
bool loadRealCurves(std::vector<ImplicitCurve>& curves, cv::Vec3d& virtualT) {
    const fs::path p = "E:/JEAMMWARE2601001/factory_calib/data_out/laser261006-1668/laser_calib.json";
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    std::ifstream f(p);
    if (!f) return false;
    nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
    if (j.is_discarded() || !j.contains("pjc")) return false;
    const auto& pjc = j["pjc"];
    if (!pjc.contains("emissionCurves") || !j.contains("virtualT")) return false;
    for (const auto& c : pjc["emissionCurves"]) {
        ImplicitCurve ic;
        for (int k = 0; k < 6; ++k) ic.coeffs[k] = c["coeffs"][k].get<double>();
        curves.push_back(ic);
    }
    for (int k = 0; k < 3; ++k) virtualT[k] = j["virtualT"][k].get<double>();
    return !curves.empty();
}

// 生产 handoff 解析（含 stereoRectifyTempTable 源①）; 缺席返回 nullopt
std::optional<StereoRectifyTempTableResult> loadRectTableViaHandoff() {
    const fs::path hp = "E:/JEAMMWARE2601001/factory_calib/module2_laser/test_data/"
                        "laser261006-1668/camera_calib.json";
    std::error_code ec;
    if (!fs::exists(hp, ec)) return std::nullopt;
    auto h = fc::loadCameraCalibHandoff(hp.string());
    if (!h.has_value()) return std::nullopt;
    return std::move(h->rectifyTempTable);
}
} // namespace

TEST(CmttCuda, RealDataRefTierVsSidecar) {
    std::vector<ImplicitCurve> curves;
    cv::Vec3d virtualT;
    if (!loadRealCurves(curves, virtualT))
        GTEST_SKIP() << "真实标定产物缺席, 跳过全尺度对拍";
    auto rectOpt = loadRectTableViaHandoff();
    ASSERT_TRUE(rectOpt.has_value()) << "handoff 解析失败";
    auto rect = std::move(*rectOpt);
    ASSERT_FALSE(rect.table.empty());
    auto laserTable = makeSyntheticLaserTable(22.5, virtualT);

    TempParamInterpolator interp(rect, laserTable);
    const InterpParams refP = interp.atReference();

    CurveMapInput in;
    in.curves = curves;
    in.f = refP.f;
    in.principalPoint = refP.principalPoint;
    in.baseline = refP.baseline;
    in.virtualT = refP.virtualT;
    in.imageSize = cv::Size(1688, 1400);
    in.roi = refP.roi;

    CurveMapParams prm;
    prm.epipolarRowStep = 0.67f;  // CLI 口径: 主库 plane_map.epipolarStep=0.67（2026-10-12 起, 矫正缩放比 round2(1231.14/1839.60)）
    prm.depthMin = 100.0f;
    prm.depthMax = 5000.0f;

    CurveMapCuda gen(prm);
    const auto t0 = nowMs();
    CurveMapResult rg = gen.Generate(in);
    const double gpuMs = nowMs() - t0;
    ASSERT_TRUE(rg.success) << rg.message;
    printf("[perf] real ref tier: cuda=%.1fms entries=%u\n", gpuMs, rg.entryCount);
    ASSERT_GT(rg.entryCount, 1000000u) << "全尺度命中异常";

    // 已交付 sidecar 参考档对照（3.4GB 读入; 128GB RAM 可承受）
    const fs::path sc = "E:/JEAMMWARE2601001/factory_calib/data_out/laser261006-1668/curve_map_temp_table.bin";
    if (!fs::exists(sc)) GTEST_SKIP() << "sidecar 缺席";
    std::vector<uint8_t> side;
    {
        std::ifstream f(sc, std::ios::binary);
        f.seekg(0, std::ios::end);
        side.resize((size_t)f.tellg());
        f.seekg(0);
        f.read(reinterpret_cast<char*>(side.data()), side.size());
    }
    CmttReader rd;
    ASSERT_TRUE(rd.load(side.data(), side.size()));
    std::optional<size_t> refIdx;
    for (size_t i = 0; i < rd.tierCount(); ++i)
        if (rd.isReference(i)) refIdx = i;
    ASSERT_TRUE(refIdx.has_value());
    const uint32_t refEntries = rd.entryCount(*refIdx);
    printf("[ref ] sidecar ref entries=%u  cuda entries=%u\n", refEntries, rg.entryCount);
    const long long diff = llabs((long long)refEntries - (long long)rg.entryCount);
    EXPECT_LE(diff, std::max<long long>(3, (long long)(refEntries * 0.001)))
        << "全尺度参考档条目差超容差";

    // 冻结网格一致性（header 字段对齐）
    std::string err;
    CurveMapTable t;
    ASSERT_TRUE(t.Load(rg.tableBytes.data(), rg.tableBytes.size(), err)) << err;
    EXPECT_EQ(t.rowStep(), 0.67f);
}

// ============================================================================
// 4) 真实数据 61 档全量计时（env 门控）
// ============================================================================

TEST(CmttCuda, RealDataFullCmttTiming) {
    if (!getenv("FC_CMTT_CUDA_FULL") || std::string(getenv("FC_CMTT_CUDA_FULL")) != "1")
        GTEST_SKIP() << "设 FC_CMTT_CUDA_FULL=1 启用全量计时";
    std::vector<ImplicitCurve> curves;
    cv::Vec3d virtualT;
    if (!loadRealCurves(curves, virtualT))
        GTEST_SKIP() << "真实标定产物缺席";

    // 源表与 RealDataRefTierVsSidecar 同构（生产解析器读 handoff + 合成 laser 表）
    auto rectOpt = loadRectTableViaHandoff();
    ASSERT_TRUE(rectOpt.has_value()) << "handoff 解析失败";
    auto rect = std::move(*rectOpt);
    ASSERT_FALSE(rect.table.empty());
    auto laserTable = makeSyntheticLaserTable(22.5, virtualT);

    CurveMapTempTableGenParams gp;
    gp.referenceTemp = 22.5;
    gp.tempHalfRange = 15.0f;
    gp.tempStep = 0.5f;
    gp.rowStep = 0.67f;       // CLI 口径: 主库 plane_map.epipolarStep=0.67（2026-10-12 起）
    gp.depthMin = 100.0f;
    gp.depthMax = 5000.0f;

    CmttCudaGenerator gen(gp);
    const auto t0 = nowMs();
    auto r = gen.Generate(curves, rect, laserTable, cv::Size(1688, 1400));
    const double ms = nowMs() - t0;
    ASSERT_TRUE(r.success) << r.message;
    printf("[perf] full 61 tiers (cuda): total=%.1fs ok=%d/%zu clamped=%.4f\n",
           ms / 1000.0, r.okCount(), r.tiers.size(), r.clampedRatio);
    ASSERT_EQ(r.okCount(), (int)r.tiers.size());

    // sidecar 落盘供人工比对（试验产物, 不覆盖交付件）
    const fs::path out = "E:/JEAMMWARE2601001/factory_calib/data_out/laser261006-1668/cmtt_cuda_sidecar.bin";
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(r.sidecarBytes.data()),
            (std::streamsize)r.sidecarBytes.size());
    printf("[out ] %s (%.1f MB)\n", out.string().c_str(),
           r.sidecarBytes.size() / 1048576.0);
}
