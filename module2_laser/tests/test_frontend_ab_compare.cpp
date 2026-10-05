// test_frontend_ab_compare.cpp — 前端算子 AB 对拍诊断（2026-09-04 选项 C）
//
// 背景: 4-4 切 steger_fast＋4-6 切 dual·Labeled 后 left_skew 真实数据 rms 劣化被证伪回滚，
//       偏差集中在 pose_06/pose_07 两帧。本测试在同机同进程内做四臂对拍定位分歧环节：
//   A = 原版 steger     × 原版 interp(lineIdCheck)   （基线臂）
//   B = steger_fast     × dual·Labeled               （被证伪的切换臂）
//   C = steger_fast     × 原版 interp                （隔离 4-4）
//   D = 原版 steger     × dual·Labeled               （隔离 4-6）
// 每臂重复 kReps 次（区分系统差 vs 运行间 tie 抖动），逐级计数（steger/interp/match）
// ＋同源输入下两版算子输出点集的量化比对（1/8 px 桶）。
// 纯诊断不断言数值（只断言各步 success）；缺数据 GTEST_SKIP。
// 产物: 控制台报告（无文件输出）。

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "steger_fast.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "epipolar_interp_dual_cuda.h"
#include "laser_match_cuda.h"
#include "laser_reconstruct_cuda.h"

namespace fs = std::filesystem;
using namespace fc;
using namespace calib;

namespace {

constexpr int kReps = 3;

bool locateLeftSkew(fs::path& out) {
    if (const char* envDir = std::getenv("FC2_LEFT_SKEW")) {
        fs::path d(envDir);
        if (fs::exists(d / "config.json") && fs::exists(d / "camera_calib.json")) {
            out = d;
            return true;
        }
    }
    for (const char* rel : {"../../data_in/left_skew", "../../../data_in/left_skew",
                            "../../../../data_in/left_skew"}) {
        fs::path d(rel);
        if (fs::exists(d / "config.json") && fs::exists(d / "camera_calib.json")) {
            out = fs::absolute(d);
            return true;
        }
    }
    fs::path d("E:/JEAMMWARE260705/factory_calib/data_in/left_skew");
    if (fs::exists(d / "config.json") && fs::exists(d / "camera_calib.json")) {
        out = d;
        return true;
    }
    return false;
}

struct StageCounts {
    int stegerPtsL = 0, stegerPtsR = 0;
    int interpL = 0, interpR = 0;
    int matched = 0;
};

// 量化键: (line_id, x*8, y*8) —— 1/8 px 桶，用于点集多重集交集
struct QKey {
    long long v;
    bool operator<(const QKey& o) const { return v < o.v; }
};
QKey qkey(int id, float x, float y) {
    return QKey{ (static_cast<long long>(id) << 42)
                 | (static_cast<long long>(std::llround(x * 8.0f)) << 21)
                 | static_cast<long long>(std::llround(y * 8.0f)) };
}

std::multiset<QKey> quantizedKeys(const cv::cuda::GpuMat& dPts, const cv::cuda::GpuMat& dIds) {
    cv::Mat pts, ids;
    dPts.download(pts);
    dIds.download(ids);
    pts = pts.reshape(2, 1);
    ids = ids.reshape(1, 1);
    std::multiset<QKey> s;
    for (int i = 0; i < pts.cols; ++i) {
        cv::Point2f p = pts.at<cv::Point2f>(0, i);
        s.insert(qkey(ids.at<int>(0, i), p.x, p.y));
    }
    return s;
}

size_t intersectionSize(const std::multiset<QKey>& a, const std::multiset<QKey>& b) {
    size_t n = 0;
    for (auto it = a.begin(); it != a.end();) {
        auto range = a.equal_range(*it);
        auto brange = b.equal_range(*it);
        n += std::min<size_t>(std::distance(range.first, range.second),
                              std::distance(brange.first, brange.second));
        it = range.second;
    }
    return n;
}

// 顺序无关量化哈希: 下载 GpuMat → 量化 (x*256,y*256) int64 → 排序 → FNV 累积。
// 跨进程比对逐级输出, 首个哈希不同的步骤即非确定来源（点数可同坐标可异）。
uint64_t quantHash2f(const cv::cuda::GpuMat& dPts) {
    cv::Mat m;
    dPts.download(m);
    m = m.reshape(2, 1);
    std::vector<long long> v;
    v.reserve(m.cols);
    for (int i = 0; i < m.cols; ++i) {
        const cv::Point2f p = m.at<cv::Point2f>(0, i);
        v.push_back((static_cast<long long>(std::llround(p.x * 256.0f)) << 32)
                    | (static_cast<unsigned>(std::llround(p.y * 256.0f))));
    }
    std::sort(v.begin(), v.end());
    uint64_t h = 14695981039346656037ull;
    for (long long e : v) {
        h ^= static_cast<uint64_t>(e);
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t quantHash3f(const cv::cuda::GpuMat& dPts) {
    cv::Mat m;
    dPts.download(m);
    m = m.reshape(3, 1);
    std::vector<std::array<long long, 3>> v;
    v.reserve(m.cols);
    for (int i = 0; i < m.cols; ++i) {
        const cv::Vec3f p = m.at<cv::Vec3f>(0, i);
        v.push_back({std::llround(p[0] * 256.0f), std::llround(p[1] * 256.0f),
                     std::llround(p[2] * 256.0f)});
    }
    std::sort(v.begin(), v.end());
    uint64_t h = 14695981039346656037ull;
    for (const auto& a : v)
        for (long long e : a) {
            h ^= static_cast<uint64_t>(e);
            h *= 1099511628211ull;
        }
    return h;
}

} // namespace

TEST(FrontendABCompare, Pose06) {
    // ⚠ log 的 "pose 6/7" 是 loadLaserInput 排序后索引（pose_06..pose_27 共 22 档），
    //   实际对应目录 pose_12/pose_13 —— 分歧帧；pose_06（=索引 0）四臂逐位一致，非目标。
    for (const char* poseName : {"pose_12", "pose_13"}) {
    fs::path dataDir;
    if (!locateLeftSkew(dataDir))
        GTEST_SKIP() << "left_skew data not found; skip frontend AB compare";

    const auto cfg = LaserCalibConfig::fromJson((dataDir / "config.json").string());
    auto handoffOpt = loadCameraCalibHandoff((dataDir / "camera_calib.json").string());
    ASSERT_TRUE(handoffOpt.has_value()) << "handoff parse failed";
    const auto& h = *handoffOpt;

    cv::Mat leftGray = cv::imread((dataDir / poseName / "L_tube0.png").string(), cv::IMREAD_GRAYSCALE);
    cv::Mat rightGray = cv::imread((dataDir / poseName / "R_tube0.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(leftGray.empty()) << poseName << " L_tube0.png unreadable";
    ASSERT_FALSE(rightGray.empty()) << poseName << " R_tube0.png unreadable";

    cv::cuda::Stream stream;

    // —— 公共前级（两臂共用构造，逐次重跑 4-1~4-3）——
    MaskExtractParams mp;
    mp.threshold = 50; mp.erodeSize = 1; mp.laserDilateSize = 19; mp.postErodeSize = 13;
    RegionAnalyzerParams cp;
    cp.deviceId = cfg.deviceId; cp.minArea = 0; cp.topXCount = 27;
    LaserLabelParams lp;
    lp.deviceId = cfg.deviceId; lp.scanDirection = cfg.labelScanDirection;
    lp.centerRowOffset = cfg.labelCenterRowOffset; lp.realLineTolerance = 10;
    cv::Mat P1_3 = h.P1.clone(); if (P1_3.cols > 3) P1_3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3 = h.P2.clone(); if (P2_3.cols > 3) P2_3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL, upR;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3; upL.deviceId = cfg.deviceId;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3; upR.deviceId = cfg.deviceId;
    LaserMatchParams mtp;
    mtp.deviceId = cfg.deviceId;
    {
        const double fPx = h.Q.at<double>(2, 3);
        const double invTx = std::abs(h.Q.at<double>(3, 2));
        if (invTx > 0 && cfg.depthMin > 0)
            mtp.max_disparity = static_cast<float>(std::abs(fPx / invTx) / cfg.depthMin * 1.05);
    }

    // —— 四臂算子实例（steger/interp 每臂独立 ×2，对齐 CLI 惯例）——
    StegerParams sp;
    sp.deviceId = cfg.deviceId;
    StegerExtractorCUDA stegerOrigL(sp), stegerOrigR(sp);
    StegerExtractorFast stegerFastL(sp), stegerFastR(sp);

    EpipolarInterpParams eip;
    eip.deviceId = cfg.deviceId; eip.lineIdCheck = true;
    EpipolarInterpCuda interpOrigL(eip), interpOrigR(eip);
    EpipolarInterpDualParams edp;
    edp.deviceId = cfg.deviceId; edp.mode = InterpMode::Labeled;
    EpipolarInterpDualCuda interpDualL(edp), interpDualR(edp);

    auto runArm = [&](bool fastSteger, bool dualInterp, StageCounts& sc,
                      std::shared_ptr<cv::cuda::GpuMat> *outInterpL = nullptr) -> bool {
        MaskExtractCUDA maskL(mp), maskR(mp);
        RegionAnalyzerCUDA cclL(cp), cclR(cp);
        LaserLabelerCUDA labelL(lp), labelR(lp);
        UndistortPointsCuda undL(upL), undR(upR);
        LaserMatchCuda match(mtp);

        auto mResL = maskL.Execute(leftGray, stream);
        auto mResR = maskR.Execute(rightGray, stream);
        if (!mResL.success || !mResR.success) return false;
        auto cResL = cclL.Execute(mResL.d_cleanedMask, stream);
        auto cResR = cclR.Execute(mResR.d_cleanedMask, stream);
        if (!cResL.success || !cResR.success) return false;
        auto lResL = labelL.Execute(*cResL.d_labeledMask, stream);
        auto lResR = labelR.Execute(*cResR.d_labeledMask, stream);
        if (!lResL.success || !lResR.success) return false;

        auto sResL = fastSteger
            ? stegerFastL.Execute(*mResL.d_grayImage, *lResL.d_labeledMask, stream, GroupMode::ByLabel)
            : stegerOrigL.Execute(*mResL.d_grayImage, *lResL.d_labeledMask, stream, GroupMode::ByLabel);
        auto sResR = fastSteger
            ? stegerFastR.Execute(*mResR.d_grayImage, *lResR.d_labeledMask, stream, GroupMode::ByLabel)
            : stegerOrigR.Execute(*mResR.d_grayImage, *lResR.d_labeledMask, stream, GroupMode::ByLabel);
        if (!sResL.success || !sResR.success) return false;
        sc.stegerPtsL = sResL.totalPointCount;
        sc.stegerPtsR = sResR.totalPointCount;

        auto uResL = undL.Execute(*sResL.d_centerPoints, *sResL.d_line_ids, stream);
        auto uResR = undR.Execute(*sResR.d_centerPoints, *sResR.d_line_ids, stream);
        if (!uResL.success || !uResR.success) return false;

        auto iResL = dualInterp
            ? interpDualL.Execute(*uResL.d_rectifiedPoints, *uResL.d_line_ids, stream)
            : interpOrigL.Execute(*uResL.d_rectifiedPoints, *uResL.d_line_ids, stream);
        auto iResR = dualInterp
            ? interpDualR.Execute(*uResR.d_rectifiedPoints, *uResR.d_line_ids, stream)
            : interpOrigR.Execute(*uResR.d_rectifiedPoints, *uResR.d_line_ids, stream);
        if (!iResL.success || !iResR.success) return false;
        sc.interpL = iResL.interpCount;
        sc.interpR = iResR.interpCount;
        if (outInterpL) *outInterpL = iResL.d_interpPoints;

        auto matchRes = match.Execute(*iResL.d_interpPoints, *iResL.d_interp_line_ids,
                                      *iResR.d_interpPoints, *iResR.d_interp_line_ids, stream);
        if (!matchRes.success) return false;
        sc.matched = matchRes.matchCount;

        maskL.Destroy(); maskR.Destroy(); cclL.Destroy(); cclR.Destroy();
        labelL.Destroy(); labelR.Destroy(); undL.Destroy(); undR.Destroy(); match.Destroy();
        return true;
    };

    struct ArmDef { const char* name; bool fastSteger; bool dualInterp; };
    const ArmDef arms[] = {
        {"A orig×orig  ", false, false},
        {"B fast×dual  ", true, true},
        {"C fast×orig  ", true, false},
        {"D orig×dual  ", false, true},
    };

    std::map<std::string, std::vector<StageCounts>> results;
    for (const auto& a : arms)
        for (int r = 0; r < kReps; ++r) {
            StageCounts sc;
            ASSERT_TRUE(runArm(a.fastSteger, a.dualInterp, sc)) << a.name << " rep " << r;
            results[a.name].push_back(sc);
        }

    std::cout << "\n==== " << poseName << " frontend AB compare (" << kReps << " reps each) ====\n";
    std::cout << "arm            | stegerL/R          | interpL/R          | matched\n";
    std::cout << "---------------+--------------------+--------------------+--------\n";
    for (const auto& a : arms)
        for (size_t r = 0; r < results[a.name].size(); ++r) {
            const auto& s = results[a.name][r];
            std::ostringstream os;
            os << a.name << "r" << r << " | " << std::setw(8) << s.stegerPtsL << "/"
               << std::setw(8) << s.stegerPtsR << " | " << std::setw(8) << s.interpL << "/"
               << std::setw(8) << s.interpR << " | " << s.matched << "\n";
            std::cout << os.str();
        }
    std::cout << "baseline(full run): p6 matched=21536 (orig chain, final_run16)\n\n";

    // —— 同源输入下两版算子输出点集量化比对（判系统差 vs tie 抖动）——
    // 4-6 隔离: 原版 steger 输出 → 同一 undistort 结果分别喂 原版 interp 与 dual。
    // 注: undistort 为逐点确定映射, 两臂分别重算结果一致, 可作同源输入。
    {
        MaskExtractCUDA maskL(mp), maskR(mp);
        RegionAnalyzerCUDA cclL(cp), cclR(cp);
        LaserLabelerCUDA labelL(lp), labelR(lp);
        UndistortPointsCuda undL(upL), undR(upR);
        auto mResL = maskL.Execute(leftGray, stream);
        auto mResR = maskR.Execute(rightGray, stream);
        ASSERT_TRUE(mResL.success && mResR.success);
        auto cResL = cclL.Execute(mResL.d_cleanedMask, stream);
        auto cResR = cclR.Execute(mResR.d_cleanedMask, stream);
        ASSERT_TRUE(cResL.success && cResR.success);
        auto lResL = labelL.Execute(*cResL.d_labeledMask, stream);
        auto lResR = labelR.Execute(*cResR.d_labeledMask, stream);
        ASSERT_TRUE(lResL.success && lResR.success);

        auto sResL = stegerOrigL.Execute(*mResL.d_grayImage, *lResL.d_labeledMask, stream, GroupMode::ByLabel);
        auto sResR = stegerOrigR.Execute(*mResR.d_grayImage, *lResR.d_labeledMask, stream, GroupMode::ByLabel);
        auto sFastL = stegerFastL.Execute(*mResL.d_grayImage, *lResL.d_labeledMask, stream, GroupMode::ByLabel);
        auto sFastR = stegerFastR.Execute(*mResR.d_grayImage, *lResR.d_labeledMask, stream, GroupMode::ByLabel);
        std::cout << "---- 4-4 isolation (same mask/label input) ----\n";
        std::cout << "steger orig L/R pts: " << sResL.totalPointCount << "/" << sResR.totalPointCount << "\n";
        std::cout << "steger fast L/R pts: " << sFastL.totalPointCount << "/" << sFastR.totalPointCount << "\n";
        auto ksO = quantizedKeys(*sResL.d_centerPoints, *sResL.d_line_ids);
        auto ksF = quantizedKeys(*sFastL.d_centerPoints, *sFastL.d_line_ids);
        std::cout << "steger L quantized(1/8px) common=" << intersectionSize(ksO, ksF)
                  << " of orig=" << ksO.size() << " fast=" << ksF.size() << "\n\n";

        auto uResL = undL.Execute(*sResL.d_centerPoints, *sResL.d_line_ids, stream);
        auto uResR = undR.Execute(*sResR.d_centerPoints, *sResR.d_line_ids, stream);
        auto iOrigL = interpOrigL.Execute(*uResL.d_rectifiedPoints, *uResL.d_line_ids, stream);
        auto iDualL = interpDualL.Execute(*uResL.d_rectifiedPoints, *uResL.d_line_ids, stream);
        auto iOrigR = interpOrigR.Execute(*uResR.d_rectifiedPoints, *uResR.d_line_ids, stream);
        auto iDualR = interpDualR.Execute(*uResR.d_rectifiedPoints, *uResR.d_line_ids, stream);
        std::cout << "---- 4-6 isolation (same undistort input) ----\n";
        std::cout << "interp orig L/R cnt: " << iOrigL.interpCount << "/" << iOrigR.interpCount << "\n";
        std::cout << "interp dual L/R cnt: " << iDualL.interpCount << "/" << iDualR.interpCount << "\n";
        auto ko = quantizedKeys(*iOrigL.d_interpPoints, *iOrigL.d_interp_line_ids);
        auto kd = quantizedKeys(*iDualL.d_interpPoints, *iDualL.d_interp_line_ids);
        std::cout << "interp L quantized(1/8px) common=" << intersectionSize(ko, kd)
                  << " of orig=" << ko.size() << " dual=" << kd.size() << "\n";

        maskL.Destroy(); maskR.Destroy(); cclL.Destroy(); cclR.Destroy();
        labelL.Destroy(); labelR.Destroy(); undL.Destroy(); undR.Destroy();
    }

    SUCCEED();
    } // poseName (pose_12 / pose_13)
}

// ============================================================================
// Shuffle 探针（2026-09-04 根因实锤）:
// 同一组 undistort 输出点（内容完全相同），仅打乱数组顺序喂 原版 4-6/4-7。
// 若 matched 随顺序变化 → "点序→tie-break→输出差"链条实锤（无需抓 GPU 调度）；
// 若先按 (line_id,y,x) 确定性排序再喂 → matched 恢复一致 → 修复方案有效实锤。
// ============================================================================
TEST(FrontendABCompare, ShuffleProbePose12) {
    fs::path dataDir;
    if (!locateLeftSkew(dataDir))
        GTEST_SKIP() << "left_skew data not found; skip shuffle probe";

    const auto cfg = LaserCalibConfig::fromJson((dataDir / "config.json").string());
    auto handoffOpt = loadCameraCalibHandoff((dataDir / "camera_calib.json").string());
    ASSERT_TRUE(handoffOpt.has_value());
    const auto& h = *handoffOpt;

    cv::Mat leftGray = cv::imread((dataDir / "pose_12" / "L_tube0.png").string(), cv::IMREAD_GRAYSCALE);
    cv::Mat rightGray = cv::imread((dataDir / "pose_12" / "R_tube0.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(leftGray.empty() || rightGray.empty());

    cv::cuda::Stream stream;
    MaskExtractParams mp;
    mp.threshold = 50; mp.erodeSize = 1; mp.laserDilateSize = 19; mp.postErodeSize = 13;
    RegionAnalyzerParams cp;
    cp.deviceId = cfg.deviceId; cp.minArea = 0; cp.topXCount = 27;
    LaserLabelParams lp;
    lp.deviceId = cfg.deviceId; lp.scanDirection = cfg.labelScanDirection;
    lp.centerRowOffset = cfg.labelCenterRowOffset; lp.realLineTolerance = 10;
    cv::Mat P1_3 = h.P1.clone(); if (P1_3.cols > 3) P1_3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3 = h.P2.clone(); if (P2_3.cols > 3) P2_3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL, upR;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3; upL.deviceId = cfg.deviceId;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3; upR.deviceId = cfg.deviceId;
    EpipolarInterpParams eip;
    eip.deviceId = cfg.deviceId; eip.lineIdCheck = true;
    LaserMatchParams mtp;
    mtp.deviceId = cfg.deviceId;
    {
        const double fPx = h.Q.at<double>(2, 3);
        const double invTx = std::abs(h.Q.at<double>(3, 2));
        if (invTx > 0 && cfg.depthMin > 0)
            mtp.max_disparity = static_cast<float>(std::abs(fPx / invTx) / cfg.depthMin * 1.05);
    }

    // —— 公共前级到 4-5，取 undistort 输出（CPU 侧持有，作为可控输入）——
    cv::Mat ptsL, idsL, ptsR, idsR;
    {
        MaskExtractCUDA maskL(mp), maskR(mp);
        RegionAnalyzerCUDA cclL(cp), cclR(cp);
        LaserLabelerCUDA labelL(lp), labelR(lp);
        UndistortPointsCuda undL(upL), undR(upR);
        auto mL = maskL.Execute(leftGray, stream);
        auto mR = maskR.Execute(rightGray, stream);
        ASSERT_TRUE(mL.success && mR.success);
        auto cL = cclL.Execute(mL.d_cleanedMask, stream);
        auto cR = cclR.Execute(mR.d_cleanedMask, stream);
        ASSERT_TRUE(cL.success && cR.success);
        auto lL = labelL.Execute(*cL.d_labeledMask, stream);
        auto lR = labelR.Execute(*cR.d_labeledMask, stream);
        ASSERT_TRUE(lL.success && lR.success);
        StegerParams sp; sp.deviceId = cfg.deviceId;
        StegerExtractorCUDA stL(sp), stR(sp);
        auto sL = stL.Execute(*mL.d_grayImage, *lL.d_labeledMask, stream, GroupMode::ByLabel);
        auto sR = stR.Execute(*mR.d_grayImage, *lR.d_labeledMask, stream, GroupMode::ByLabel);
        ASSERT_TRUE(sL.success && sR.success);
        auto uL = undL.Execute(*sL.d_centerPoints, *sL.d_line_ids, stream);
        auto uR = undR.Execute(*sR.d_centerPoints, *sR.d_line_ids, stream);
        ASSERT_TRUE(uL.success && uR.success);
        uL.d_rectifiedPoints->download(ptsL); uL.d_line_ids->download(idsL);
        uR.d_rectifiedPoints->download(ptsR); uR.d_line_ids->download(idsR);
        ptsL = ptsL.reshape(2, 1); idsL = idsL.reshape(1, 1);
        ptsR = ptsR.reshape(2, 1); idsR = idsR.reshape(1, 1);
        maskL.Destroy(); maskR.Destroy(); cclL.Destroy(); cclR.Destroy();
        labelL.Destroy(); labelR.Destroy(); undL.Destroy(); undR.Destroy();
    }

    // —— 排列构造: original / shuffle(seed 1..5) / canonical sort ——
    const int nL = ptsL.cols, nR = ptsR.cols;
    auto permute = [](const cv::Mat& pts, const cv::Mat& ids, const std::vector<int>& idx) {
        cv::Mat p2(pts.size(), pts.type()), i2(ids.size(), ids.type());
        for (size_t k = 0; k < idx.size(); ++k) {
            p2.at<cv::Point2f>(0, static_cast<int>(k)) = pts.at<cv::Point2f>(0, idx[k]);
            i2.at<int>(0, static_cast<int>(k)) = ids.at<int>(0, idx[k]);
        }
        return std::make_pair(p2, i2);
    };
    auto canonicalIdx = [](const cv::Mat& pts, const cv::Mat& ids) {
        std::vector<int> idx(ids.cols);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            const int ia = ids.at<int>(0, a), ib = ids.at<int>(0, b);
            if (ia != ib) return ia < ib;
            const cv::Point2f pa = pts.at<cv::Point2f>(0, a), pb = pts.at<cv::Point2f>(0, b);
            if (std::abs(pa.y - pb.y) > 1e-9f) return pa.y < pb.y;
            return pa.x < pb.x;
        });
        return idx;
    };

    struct RunOut { int interpL = 0, interpR = 0, matched = 0; };
    auto runWithOrder = [&](const std::vector<int>& xL, const std::vector<int>& xR) {
        auto [pL, iL] = permute(ptsL, idsL, xL);
        auto [pR, iR] = permute(ptsR, idsR, xR);
        cv::cuda::GpuMat dPL(pL), dIL(iL), dPR(pR), dIR(iR);
        EpipolarInterpCuda interpL(eip), interpR(eip);
        LaserMatchCuda match(mtp);
        auto oL = interpL.Execute(dPL, dIL, stream);
        auto oR = interpR.Execute(dPR, dIR, stream);
        RunOut out;
        if (!oL.success || !oR.success) return out;
        out.interpL = oL.interpCount; out.interpR = oR.interpCount;
        auto m = match.Execute(*oL.d_interpPoints, *oL.d_interp_line_ids,
                               *oR.d_interpPoints, *oR.d_interp_line_ids, stream);
        if (m.success) out.matched = m.matchCount;
        interpL.Destroy(); interpR.Destroy(); match.Destroy();
        return out;
    };

    std::vector<int> natL(nL), natR(nR);
    std::iota(natL.begin(), natL.end(), 0);
    std::iota(natR.begin(), natR.end(), 0);

    EpipolarInterpDualParams edp;
    edp.deviceId = cfg.deviceId; edp.mode = InterpMode::Labeled;

    // 双 interp 对比: 原版 interp(lineIdCheck) 与 dual·Labeled 在相同点集不同顺序下
    // 的 matched（2026-09-04 补测——dual 顺序敏感性是切换链生产差异的最后嫌疑）
    auto runWithOrderDual = [&](bool useDual, const std::vector<int>& xL, const std::vector<int>& xR) {
        auto [pL, iL] = permute(ptsL, idsL, xL);
        auto [pR, iR] = permute(ptsR, idsR, xR);
        cv::cuda::GpuMat dPL(pL), dIL(iL), dPR(pR), dIR(iR);
        EpipolarInterpCuda interpO(eip);
        EpipolarInterpDualCuda interpD0(edp);
        LaserMatchCuda match(mtp);
        RunOut out;
        if (useDual) {
            auto oL = interpD0.Execute(dPL, dIL, stream);
            EpipolarInterpDualCuda interpD1(edp);
            auto oR = interpD1.Execute(dPR, dIR, stream);
            interpD1.Destroy();
            if (!oL.success || !oR.success) { interpO.Destroy(); interpD0.Destroy(); match.Destroy(); return out; }
            out.interpL = oL.interpCount; out.interpR = oR.interpCount;
            auto m = match.Execute(*oL.d_interpPoints, *oL.d_interp_line_ids,
                                   *oR.d_interpPoints, *oR.d_interp_line_ids, stream);
            if (m.success) out.matched = m.matchCount;
        } else {
            auto oL = interpO.Execute(dPL, dIL, stream);
            EpipolarInterpCuda interpO1(eip);
            auto oR = interpO1.Execute(dPR, dIR, stream);
            interpO1.Destroy();
            if (!oL.success || !oR.success) { interpO.Destroy(); interpD0.Destroy(); match.Destroy(); return out; }
            out.interpL = oL.interpCount; out.interpR = oR.interpCount;
            auto m = match.Execute(*oL.d_interpPoints, *oL.d_interp_line_ids,
                                   *oR.d_interpPoints, *oR.d_interp_line_ids, stream);
            if (m.success) out.matched = m.matchCount;
        }
        interpO.Destroy(); interpD0.Destroy(); match.Destroy();
        return out;
    };

    std::cout << "\n==== shuffle probe pose_12 (原版 interp+match, 同点集仅变顺序) ====\n";
    std::cout << "order        | interpL/R    | matched\n";
    std::cout << "-------------+--------------+--------\n";
    {
        auto r = runWithOrder(natL, natR);
        std::cout << "original     | " << r.interpL << "/" << r.interpR << " | " << r.matched << "\n";
    }
    std::vector<int> matchedVals;
    for (unsigned seed = 1; seed <= 5; ++seed) {
        std::mt19937 rng(seed);
        auto xL = natL, xR = natR;
        std::shuffle(xL.begin(), xL.end(), rng);
        std::shuffle(xR.begin(), xR.end(), rng);
        auto r = runWithOrder(xL, xR);
        matchedVals.push_back(r.matched);
        std::cout << "shuffle#" << seed << "    | " << r.interpL << "/" << r.interpR
                  << " | " << r.matched << "\n";
    }
    {
        auto rL = canonicalIdx(ptsL, idsL);
        auto rR = canonicalIdx(ptsR, idsR);
        auto r = runWithOrder(rL, rR);
        std::cout << "canonical    | " << r.interpL << "/" << r.interpR << " | " << r.matched
                  << "   <- 确定性排序(line_id,y,x)后\n";
        // 排序后再 shuffle（排序应使任何顺序等价——若 matched 仍变则排序不能根治）
        std::mt19937 rng(99);
        auto xL = rL, xR = rR;
        std::shuffle(xL.begin(), xL.end(), rng);
        std::shuffle(xR.begin(), xR.end(), rng);
        auto r2 = runWithOrder(xL, xR);
        std::cout << "canon+shuf   | " << r2.interpL << "/" << r2.interpR << " | " << r2.matched
                  << "   <- 排序后再乱序（应与 canonical 同→排序根治有效）\n";
    }

    std::cout << "\n==== dual·Labeled order sensitivity（同点集, 原序 vs 乱序）====\n";
    {
        auto r0 = runWithOrderDual(true, natL, natR);
        std::cout << "dual original | " << r0.interpL << "/" << r0.interpR << " | " << r0.matched << "\n";
        for (unsigned seed = 1; seed <= 5; ++seed) {
            std::mt19937 rng(seed);
            auto xL = natL, xR = natR;
            std::shuffle(xL.begin(), xL.end(), rng);
            std::shuffle(xR.begin(), xR.end(), rng);
            auto r = runWithOrderDual(true, xL, xR);
            std::cout << "dual shuf#" << seed << "  | " << r.interpL << "/" << r.interpR
                      << " | " << r.matched << "\n";
        }
    }

    SUCCEED();
}

// ============================================================================
// 顺序 22 帧探针（2026-09-04 根因取证第二轮）:
// 复刻 CLI 条件——同一组算子实例连跑全部 22 帧（跨帧复用＋满载），逐帧记录
// steger/interp/matched 计数。跑两个进程 diff 本测试输出: 若复现差异 →
// 可在此环境二分定位首分歧步骤; 若两进程完全一致 → 非确定需 CLI 特有条件。
// ============================================================================
TEST(FrontendABCompare, Sequential22Frames) {
    fs::path dataDir;
    if (!locateLeftSkew(dataDir))
        GTEST_SKIP() << "left_skew data not found; skip sequential probe";

    const auto cfg = LaserCalibConfig::fromJson((dataDir / "config.json").string());
    auto handoffOpt = loadCameraCalibHandoff((dataDir / "camera_calib.json").string());
    ASSERT_TRUE(handoffOpt.has_value());
    const auto& h = *handoffOpt;

    std::vector<fs::path> poseDirs;
    for (const auto& e : fs::directory_iterator(dataDir))
        if (e.is_directory() && e.path().filename().string().rfind("pose_", 0) == 0)
            poseDirs.push_back(e.path());
    std::sort(poseDirs.begin(), poseDirs.end());
    ASSERT_FALSE(poseDirs.empty());

    cv::cuda::Stream stream;
    MaskExtractParams mp;
    mp.threshold = 50; mp.erodeSize = 1; mp.laserDilateSize = 19; mp.postErodeSize = 13;
    RegionAnalyzerParams cp;
    cp.deviceId = cfg.deviceId; cp.minArea = 0; cp.topXCount = 27;
    LaserLabelParams lp;
    lp.deviceId = cfg.deviceId; lp.scanDirection = cfg.labelScanDirection;
    lp.centerRowOffset = cfg.labelCenterRowOffset; lp.realLineTolerance = 10;
    cv::Mat P1_3 = h.P1.clone(); if (P1_3.cols > 3) P1_3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3 = h.P2.clone(); if (P2_3.cols > 3) P2_3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL, upR;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3; upL.deviceId = cfg.deviceId;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3; upR.deviceId = cfg.deviceId;
    StegerParams sp; sp.deviceId = cfg.deviceId;
    EpipolarInterpParams eip;
    eip.deviceId = cfg.deviceId; eip.lineIdCheck = true;
    LaserMatchParams mtp;
    mtp.deviceId = cfg.deviceId;
    {
        const double fPx = h.Q.at<double>(2, 3);
        const double invTx = std::abs(h.Q.at<double>(3, 2));
        if (invTx > 0 && cfg.depthMin > 0)
            mtp.max_disparity = static_cast<float>(std::abs(fPx / invTx) / cfg.depthMin * 1.05);
    }

    // —— CLI 同款: 实例只构造一次, 跨 22 帧复用; 含 4-8 reconstruct＋主机累积 ——
    MaskExtractCUDA maskL(mp), maskR(mp);
    RegionAnalyzerCUDA cclL(cp), cclR(cp);
    LaserLabelerCUDA labelL(lp), labelR(lp);
    StegerExtractorCUDA stegerL(sp), stegerR(sp);
    UndistortPointsCuda undL(upL), undR(upR);
    EpipolarInterpCuda interpL(eip), interpR(eip);
    LaserMatchCuda match(mtp);
    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin; rp.maxDepth = cfg.depthMax; rp.deviceId = cfg.deviceId;
    LaserReconstructCuda recon(rp);
    std::vector<cv::Vec3f> accum;   // CLI 同款 host 累积

    std::cout << "\n==== sequential 22 frames (CLI-like instance reuse) ====\n";
    std::cout << "idx | stegerL/R      | interpL/R      | matched\n";
    for (size_t pi = 0; pi < poseDirs.size(); ++pi) {
        cv::Mat lg = cv::imread((poseDirs[pi] / "L_tube0.png").string(), cv::IMREAD_GRAYSCALE);
        cv::Mat rg = cv::imread((poseDirs[pi] / "R_tube0.png").string(), cv::IMREAD_GRAYSCALE);
        if (lg.empty() || rg.empty()) continue;
        auto mL = maskL.Execute(lg, stream);
        auto mR = maskR.Execute(rg, stream);
        ASSERT_TRUE(mL.success && mR.success);
        auto cL = cclL.Execute(mL.d_cleanedMask, stream);
        auto cR = cclR.Execute(mR.d_cleanedMask, stream);
        ASSERT_TRUE(cL.success && cR.success);
        auto lL = labelL.Execute(*cL.d_labeledMask, stream);
        auto lR = labelR.Execute(*cR.d_labeledMask, stream);
        ASSERT_TRUE(lL.success && lR.success);
        auto sL = stegerL.Execute(*mL.d_grayImage, *lL.d_labeledMask, stream, GroupMode::ByLabel);
        auto sR = stegerR.Execute(*mR.d_grayImage, *lR.d_labeledMask, stream, GroupMode::ByLabel);
        ASSERT_TRUE(sL.success && sR.success);
        auto uL = undL.Execute(*sL.d_centerPoints, *sL.d_line_ids, stream);
        auto uR = undR.Execute(*sR.d_centerPoints, *sR.d_line_ids, stream);
        ASSERT_TRUE(uL.success && uR.success);
        auto iL = interpL.Execute(*uL.d_rectifiedPoints, *uL.d_line_ids, stream);
        auto iR = interpR.Execute(*uR.d_rectifiedPoints, *uR.d_line_ids, stream);
        ASSERT_TRUE(iL.success && iR.success);
        auto m = match.Execute(*iL.d_interpPoints, *iL.d_interp_line_ids,
                               *iR.d_interpPoints, *iR.d_interp_line_ids, stream);
        ASSERT_TRUE(m.success);
        auto rec = recon.Execute(*m.d_matched_left, *m.d_matched_right,
                                 *m.d_matched_line_ids, h.Q, stream);
        ASSERT_TRUE(rec.success);
        // 幽灵线号检测（2026-09-04）: 各级 line_ids 值域——合法线号 1..25,
        // 出现大值即该级引入未初始化/越界数据
        if (pi == 6 || pi == 7) {
            auto idRange = [](const cv::cuda::GpuMat& dIds) {
                cv::Mat m2;
                dIds.download(m2);
                m2 = m2.reshape(1, 1);
                int mn = INT_MAX, mx = INT_MIN;
                for (int i = 0; i < m2.cols; ++i) {
                    const int v = m2.at<int>(0, i);
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                return std::make_pair(mn, mx);
            };
            auto rs = idRange(*sL.d_line_ids);
            auto ru = idRange(*uL.d_line_ids);
            auto ri = idRange(*iL.d_interp_line_ids);
            auto riR = idRange(*iR.d_interp_line_ids);   // 右路 interp（match 线号真源）
            auto rm = idRange(*m.d_matched_line_ids);
            auto rr = idRange(*rec.d_valid_line_ids);
            std::cout << "IDR p" << pi << " steger=[" << rs.first << "," << rs.second << "]"
                      << " undist=[" << ru.first << "," << ru.second << "]"
                      << " interpL=[" << ri.first << "," << ri.second << "]"
                      << " interpR=[" << riR.first << "," << riR.second << "]"
                      << " match=[" << rm.first << "," << rm.second << "]"
                      << " recon=[" << rr.first << "," << rr.second << "]"
                      << "  matched=" << m.matchCount << " reconCnt=" << rec.validCount << "\n";
            // 幽灵点样本（fid 越界）: 序号/线号/坐标——判断是"选错槽位"还是"写了垃圾"
            {
                cv::Mat mp, mf;
                m.d_matched_left->download(mp);
                m.d_matched_line_ids->download(mf);
                mp = mp.reshape(2, 1);
                mf = mf.reshape(1, 1);
                int shown = 0, totalGhost = 0;
                for (int i = 0; i < mf.cols; ++i) {
                    const int v = mf.at<int>(0, i);
                    if (v < 1 || v > 25) {
                        ++totalGhost;
                        if (shown < 5) {
                            const cv::Point2f p = mp.at<cv::Point2f>(0, i);
                            std::cout << "GHOST p" << pi << " i=" << i << " fid=" << v
                                      << " x=" << p.x << " y=" << p.y << "\n";
                            ++shown;
                        }
                    }
                }
                if (totalGhost > 0) std::cout << "GHOST p" << pi << " total=" << totalGhost
                                              << " of matched=" << mf.cols << "\n";
            }
        }
        if (rec.d_points3d && !rec.d_points3d->empty()) {
            cv::Mat hp, hi;
            rec.d_points3d->download(hp);
            hp = hp.reshape(3, 1);
            accum.insert(accum.end(), hp.begin<cv::Vec3f>(), hp.end<cv::Vec3f>());
        }
        // 目标帧逐级量化哈希（跨进程比对定位首个非确定步骤）
        if (pi == 6 || pi == 7) {
            std::cout << "H" << pi << " stegerL=" << std::hex << quantHash2f(*sL.d_centerPoints)
                      << " undistL=" << quantHash2f(*uL.d_rectifiedPoints)
                      << " interpL=" << quantHash2f(*iL.d_interpPoints)
                      << " matchL=" << quantHash2f(*m.d_matched_left)
                      << " recon=" << (rec.d_points3d && !rec.d_points3d->empty()
                                           ? quantHash3f(*rec.d_points3d) : 0ull)
                      << std::dec << "\n";
        }
        std::cout << std::setw(3) << pi << " | " << std::setw(6) << sL.totalPointCount << "/"
                  << std::setw(6) << sR.totalPointCount << " | " << std::setw(6) << iL.interpCount
                  << "/" << std::setw(6) << iR.interpCount << " | " << m.matchCount << " r"
                  << rec.validCount << "  (" << poseDirs[pi].filename().string() << ")\n";
    }
    std::cout << "TOTAL accum=" << accum.size() << "\n";

    maskL.Destroy(); maskR.Destroy(); cclL.Destroy(); cclR.Destroy();
    labelL.Destroy(); labelR.Destroy(); stegerL.Destroy(); stegerR.Destroy();
    undL.Destroy(); undR.Destroy(); interpL.Destroy(); interpR.Destroy(); match.Destroy();
    recon.Destroy();
    SUCCEED();
}

// ============================================================================
// 全 22 帧逐级对拍（2026-09-04 二次证伪根因定位）:
// 每帧同一 mask/label 输出分别喂 原版 steger 与 steger_fast——定位 steger 差异帧;
// 同一 undistort 输出分别喂 原版 interp 与 dual·Labeled——定位 interp 差异帧。
// Δ = 量化集合(1/8px)对称差元素数。确定性链下同进程对比即算法差异。
// ============================================================================
TEST(FrontendABCompare, AllFramesStageDiff) {
    fs::path dataDir;
    if (!locateLeftSkew(dataDir))
        GTEST_SKIP() << "left_skew data not found; skip all-frames diff";

    const auto cfg = LaserCalibConfig::fromJson((dataDir / "config.json").string());
    auto handoffOpt = loadCameraCalibHandoff((dataDir / "camera_calib.json").string());
    ASSERT_TRUE(handoffOpt.has_value());
    const auto& h = *handoffOpt;

    std::vector<fs::path> poseDirs;
    for (const auto& e : fs::directory_iterator(dataDir))
        if (e.is_directory() && e.path().filename().string().rfind("pose_", 0) == 0)
            poseDirs.push_back(e.path());
    std::sort(poseDirs.begin(), poseDirs.end());

    cv::cuda::Stream stream;
    MaskExtractParams mp;
    mp.threshold = 50; mp.erodeSize = 1; mp.laserDilateSize = 19; mp.postErodeSize = 13;
    RegionAnalyzerParams cp;
    cp.deviceId = cfg.deviceId; cp.minArea = 0; cp.topXCount = 27;
    LaserLabelParams lp;
    lp.deviceId = cfg.deviceId; lp.scanDirection = cfg.labelScanDirection;
    lp.centerRowOffset = cfg.labelCenterRowOffset; lp.realLineTolerance = 10;
    cv::Mat P1_3 = h.P1.clone(); if (P1_3.cols > 3) P1_3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3 = h.P2.clone(); if (P2_3.cols > 3) P2_3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL, upR;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3; upL.deviceId = cfg.deviceId;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3; upR.deviceId = cfg.deviceId;
    StegerParams sp; sp.deviceId = cfg.deviceId;
    EpipolarInterpParams eip;
    eip.deviceId = cfg.deviceId; eip.lineIdCheck = true;
    EpipolarInterpDualParams edp;
    edp.deviceId = cfg.deviceId; edp.mode = InterpMode::Labeled;

    MaskExtractCUDA maskL(mp), maskR(mp);
    RegionAnalyzerCUDA cclL(cp), cclR(cp);
    LaserLabelerCUDA labelL(lp), labelR(lp);
    StegerExtractorCUDA stO_L(sp), stO_R(sp);
    StegerExtractorFast stF_L(sp), stF_R(sp);
    UndistortPointsCuda undL(upL), undR(upR);
    EpipolarInterpCuda ipO_L(eip), ipO_R(eip);
    EpipolarInterpDualCuda ipD_L(edp), ipD_R(edp);

    auto symDiff = [](const std::multiset<QKey>& a, const std::multiset<QKey>& b) {
        return static_cast<long>(a.size()) + static_cast<long>(b.size())
               - 2 * static_cast<long>(intersectionSize(a, b));
    };

    // 全精度对比: (line_id, float 位模式) 三元组排序后逐位比较。
    // 返回 {不等元素数, 最大坐标欧氏距离(px)}（同尺寸同序下位置对位置）。
    struct ExactDiff { long nDiff = 0; float maxDist = 0.f; double sumDist = 0.0; };
    auto exactCompare = [](const cv::cuda::GpuMat& dA_pts, const cv::cuda::GpuMat& dA_ids,
                           const cv::cuda::GpuMat& dB_pts, const cv::cuda::GpuMat& dB_ids) {
        cv::Mat ap, ai, bp, bi;
        dA_pts.download(ap); dA_ids.download(ai);
        dB_pts.download(bp); dB_ids.download(bi);
        ap = ap.reshape(2, 1); ai = ai.reshape(1, 1);
        bp = bp.reshape(2, 1); bi = bi.reshape(1, 1);
        using E = std::tuple<int, unsigned int, unsigned int>;
        std::vector<E> va(ap.cols), vb(bp.cols);
        auto bits = [](float f) { unsigned int u; std::memcpy(&u, &f, 4); return u; };
        for (int i = 0; i < ap.cols; ++i) {
            const cv::Point2f p = ap.at<cv::Point2f>(0, i);
            va[i] = {ai.at<int>(0, i), bits(p.x), bits(p.y)};
        }
        for (int i = 0; i < bp.cols; ++i) {
            const cv::Point2f p = bp.at<cv::Point2f>(0, i);
            vb[i] = {bi.at<int>(0, i), bits(p.x), bits(p.y)};
        }
        std::sort(va.begin(), va.end());
        std::sort(vb.begin(), vb.end());
        ExactDiff d;
        if (va.size() != vb.size()) { d.nDiff = -1; return d; }
        for (size_t i = 0; i < va.size(); ++i) {
            if (va[i] != vb[i]) {
                ++d.nDiff;
                const cv::Point2f pa = ap.at<cv::Point2f>(0, i);   // 仅尺寸同序近似配对
                const cv::Point2f pb = bp.at<cv::Point2f>(0, i);
                const float dist = std::hypot(pa.x - pb.x, pa.y - pb.y);
                if (dist > d.maxDist) d.maxDist = dist;
                d.sumDist += dist;
            }
        }
        return d;
    };

    std::cout << "\n==== all-frames stage diff (orig vs fast steger / orig vs dual interp) ====\n";
    std::cout << "frame      | steger cnt O/F L,R     | stegerD L,R | interp cnt O/D L,R     | interpD L,R\n";
    long totStgL = 0, totStgR = 0, totIntL = 0, totIntR = 0;
    for (size_t pi = 0; pi < poseDirs.size(); ++pi) {
        cv::Mat lg = cv::imread((poseDirs[pi] / "L_tube0.png").string(), cv::IMREAD_GRAYSCALE);
        cv::Mat rg = cv::imread((poseDirs[pi] / "R_tube0.png").string(), cv::IMREAD_GRAYSCALE);
        if (lg.empty() || rg.empty()) continue;
        auto mL = maskL.Execute(lg, stream);
        auto mR = maskR.Execute(rg, stream);
        ASSERT_TRUE(mL.success && mR.success);
        auto cL = cclL.Execute(mL.d_cleanedMask, stream);
        auto cR = cclR.Execute(mR.d_cleanedMask, stream);
        ASSERT_TRUE(cL.success && cR.success);
        auto lL = labelL.Execute(*cL.d_labeledMask, stream);
        auto lR = labelR.Execute(*cR.d_labeledMask, stream);
        ASSERT_TRUE(lL.success && lR.success);

        auto soL = stO_L.Execute(*mL.d_grayImage, *lL.d_labeledMask, stream, GroupMode::ByLabel);
        auto soR = stO_R.Execute(*mR.d_grayImage, *lR.d_labeledMask, stream, GroupMode::ByLabel);
        auto sfL = stF_L.Execute(*mL.d_grayImage, *lL.d_labeledMask, stream, GroupMode::ByLabel);
        auto sfR = stF_R.Execute(*mR.d_grayImage, *lR.d_labeledMask, stream, GroupMode::ByLabel);

        auto koL = quantizedKeys(*soL.d_centerPoints, *soL.d_line_ids);
        auto koR = quantizedKeys(*soR.d_centerPoints, *soR.d_line_ids);
        auto kfL = quantizedKeys(*sfL.d_centerPoints, *sfL.d_line_ids);
        auto kfR = quantizedKeys(*sfR.d_centerPoints, *sfR.d_line_ids);
        const long dStgL = symDiff(koL, kfL), dStgR = symDiff(koR, kfR);
        totStgL += dStgL; totStgR += dStgR;

        auto uL = undL.Execute(*soL.d_centerPoints, *soL.d_line_ids, stream);
        auto uR = undR.Execute(*soR.d_centerPoints, *soR.d_line_ids, stream);

        auto ioL = ipO_L.Execute(*uL.d_rectifiedPoints, *uL.d_line_ids, stream);
        auto ioR = ipO_R.Execute(*uR.d_rectifiedPoints, *uR.d_line_ids, stream);
        auto idL = ipD_L.Execute(*uL.d_rectifiedPoints, *uL.d_line_ids, stream);
        auto idR = ipD_R.Execute(*uR.d_rectifiedPoints, *uR.d_line_ids, stream);

        // 全精度位级对比（steger 与 interp 各取左路——量化为 0 时定位亚量化微差）
        auto es = exactCompare(*soL.d_centerPoints, *soL.d_line_ids,
                               *sfL.d_centerPoints, *sfL.d_line_ids);
        auto ei = exactCompare(*ioL.d_interpPoints, *ioL.d_interp_line_ids,
                               *idL.d_interpPoints, *idL.d_interp_line_ids);

        auto qoL = quantizedKeys(*ioL.d_interpPoints, *ioL.d_interp_line_ids);
        auto qoR = quantizedKeys(*ioR.d_interpPoints, *ioR.d_interp_line_ids);
        auto qdL = quantizedKeys(*idL.d_interpPoints, *idL.d_interp_line_ids);
        auto qdR = quantizedKeys(*idR.d_interpPoints, *idR.d_interp_line_ids);
        const long dIntL = symDiff(qoL, qdL), dIntR = symDiff(qoR, qdR);
        totIntL += dIntL; totIntR += dIntR;

        std::ostringstream os;
        os << std::left << std::setw(10) << poseDirs[pi].filename().string() << "| "
           << soL.totalPointCount << "/" << sfL.totalPointCount << " "
           << soR.totalPointCount << "/" << sfR.totalPointCount << " | "
           << dStgL << "," << dStgR << " | "
           << ioL.interpCount << "/" << idL.interpCount << " "
           << ioR.interpCount << "/" << idR.interpCount << " | "
           << dIntL << "," << dIntR
           << " | EXACT stegerL: n=" << es.nDiff << " max=" << std::scientific
           << std::setprecision(3) << es.maxDist
           << " interpL: n=" << ei.nDiff << " max=" << ei.maxDist << "\n";
        std::cout << os.str();
    }
    std::cout << "TOTALS stegerD L,R = " << totStgL << "," << totStgR
              << "  interpD L,R = " << totIntL << "," << totIntR << "\n";

    maskL.Destroy(); maskR.Destroy(); cclL.Destroy(); cclR.Destroy();
    labelL.Destroy(); labelR.Destroy(); stO_L.Destroy(); stO_R.Destroy();
    stF_L.Destroy(); stF_R.Destroy(); undL.Destroy(); undR.Destroy();
    ipO_L.Destroy(); ipO_R.Destroy(); ipD_L.Destroy(); ipD_R.Destroy();
    SUCCEED();
}

// ============================================================================
// 顺序回放（2026-09-04 终极定位）: 读生产环境 CLI dump 的 CSV（同点集,
// orig=行扫描序 / fast=分组聚集序）, 用真实顺序分别喂 原版 interp 与 dual·Labeled。
// 若 (fast序, dual) 复现生产 matched 21600 而 (orig序, dual) 为 21536 →
// dual 对"分组聚集序"敏感＝切换链生产差异根因实锤。
// ============================================================================
TEST(FrontendABCompare, ReplayRealOrder) {
    fs::path dataDir;
    if (!locateLeftSkew(dataDir))
        GTEST_SKIP() << "left_skew data not found; skip order replay";
    const fs::path dumpDir = dataDir.parent_path().parent_path() / "data_out" / "e2e_fast";
    const std::array<fs::path, 4> need = {
        dumpDir / "orig_p6_L.csv", dumpDir / "orig_p6_R.csv",
        dumpDir / "fast_p6_L.csv", dumpDir / "fast_p6_R.csv"};
    for (const auto& p : need)
        if (!fs::exists(p))
            GTEST_SKIP() << "dump CSVs not found at " << dumpDir.string();

    auto loadCsv = [](const fs::path& p, cv::Mat& pts, cv::Mat& ids) {
        std::ifstream f(p);
        std::vector<float> xs, ys;
        std::vector<int> idv;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const auto c1 = line.find(','), c2 = line.find(',', c1 + 1);
            idv.push_back(std::stoi(line.substr(0, c1)));
            xs.push_back(std::stof(line.substr(c1 + 1, c2 - c1 - 1)));
            ys.push_back(std::stof(line.substr(c2 + 1)));
        }
        pts = cv::Mat(1, static_cast<int>(xs.size()), CV_32FC2);
        ids = cv::Mat(1, static_cast<int>(xs.size()), CV_32SC1);
        for (size_t i = 0; i < xs.size(); ++i) {
            pts.at<cv::Point2f>(0, static_cast<int>(i)) = {xs[i], ys[i]};
            ids.at<int>(0, static_cast<int>(i)) = idv[i];
        }
    };
    cv::Mat pO_L, iO_L, pF_L, iF_L, pO_R, iO_R, pF_R, iF_R;
    loadCsv(need[0], pO_L, iO_L);   // orig 序（行扫描）
    loadCsv(need[1], pO_R, iO_R);
    loadCsv(need[2], pF_L, iF_L);   // fast 序（分组聚集）
    loadCsv(need[3], pF_R, iF_R);
    ASSERT_EQ(pO_L.cols, pF_L.cols);

    const auto cfg = LaserCalibConfig::fromJson((dataDir / "config.json").string());
    auto handoffOpt = loadCameraCalibHandoff((dataDir / "camera_calib.json").string());
    ASSERT_TRUE(handoffOpt.has_value());
    EpipolarInterpParams eip;
    eip.deviceId = cfg.deviceId; eip.lineIdCheck = true;
    EpipolarInterpDualParams edp;
    edp.deviceId = cfg.deviceId; edp.mode = InterpMode::Labeled;
    LaserMatchParams mtp;
    mtp.deviceId = cfg.deviceId;
    {
        const double fPx = handoffOpt->Q.at<double>(2, 3);
        const double invTx = std::abs(handoffOpt->Q.at<double>(3, 2));
        if (invTx > 0 && cfg.depthMin > 0)
            mtp.max_disparity = static_cast<float>(std::abs(fPx / invTx) / cfg.depthMin * 1.05);
    }

    cv::cuda::Stream stream;
    auto runCase = [&](const char* name, bool useDual, const cv::Mat& pL, const cv::Mat& iL,
                       const cv::Mat& pR, const cv::Mat& iR) {
        cv::cuda::GpuMat dPL(pL), dIL(iL), dPR(pR), dIR(iR);
        LaserMatchCuda match(mtp);
        int mL = -1, mR = -1, matched = -1;
        if (useDual) {
            EpipolarInterpDualCuda a(edp), b(edp);
            auto oL = a.Execute(dPL, dIL, stream);
            auto oR = b.Execute(dPR, dIR, stream);
            a.Destroy(); b.Destroy();
            mL = oL.interpCount; mR = oR.interpCount;
            auto m = match.Execute(*oL.d_interpPoints, *oL.d_interp_line_ids,
                                   *oR.d_interpPoints, *oR.d_interp_line_ids, stream);
            if (m.success) matched = m.matchCount;
        } else {
            EpipolarInterpCuda a(eip), b(eip);
            auto oL = a.Execute(dPL, dIL, stream);
            auto oR = b.Execute(dPR, dIR, stream);
            a.Destroy(); b.Destroy();
            mL = oL.interpCount; mR = oR.interpCount;
            auto m = match.Execute(*oL.d_interpPoints, *oL.d_interp_line_ids,
                                   *oR.d_interpPoints, *oR.d_interp_line_ids, stream);
            if (m.success) matched = m.matchCount;
        }
        match.Destroy();
        std::cout << name << " | interp " << mL << "/" << mR << " | matched " << matched << "\n";
    };

    std::cout << "\n==== replay real production order (pose_12, same point set) ====\n";
    runCase("origI x origOrd", false, pO_L, iO_L, pO_R, iO_R);
    runCase("origI x fastOrd", false, pF_L, iF_L, pF_R, iF_R);
    runCase("dual  x origOrd", true,  pO_L, iO_L, pO_R, iO_R);
    runCase("dual  x fastOrd", true,  pF_L, iF_L, pF_R, iF_R);

    SUCCEED();
}
