// test_laser_match_scan_v2.cpp — V1 vs V2 对比基准（逐位结果一致性 + CUDA event 计时）
//
// 用法: 直接运行（或 ctest）。数据目录解析顺序:
//   1) 环境变量 FC2_LEFT_SKEW / FC2_CALIB_JSON
//   2) 相对路径候选（ctest cwd = build_fc2_rel/module2_laser; 手动 cwd = .../Release）
//   3) 缺数据 → GTEST_SKIP（保持无数据机器回归绿）
//
// 判定:
//   - 一致性: 每帧 matchedCount / matched(left,right,lineIds) / status(L,R) / excluded 与 v1 逐位一致
//   - 性能: v2 计时中位数; 目标 ≤ 3ms（基线 v1 ≈ 7.5ms）

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "epipolar_interp_dual_cuda.h"    // dual 有号模式回归对比
#include "curve_map.h"
#include "laser_match_scan_cuda.h"       // v1 基准
#include "laser_match_scan_v2_cuda.h"    // v2 优化版

using namespace fc;
using namespace calib;

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

bool findData(fs::path& dataDir, fs::path& calibJson) {
    const char* envDir = std::getenv("FC2_LEFT_SKEW");
    const char* envJson = std::getenv("FC2_CALIB_JSON");
    std::vector<fs::path> dirCandidates;
    if (envDir) dirCandidates.emplace_back(envDir);
    for (const char* rel : {"../../data_in/left_skew", "../../../data_in/left_skew",
                            "../../../../data_in/left_skew"}) {
        dirCandidates.emplace_back(fs::path(rel));
    }
    dirCandidates.emplace_back("E:/JEAMMWARE260705/factory_calib/data_in/left_skew");
    for (auto& d : dirCandidates) {
        if (fs::exists(d / "config.json") && fs::exists(d / "camera_calib.json")) {
            dataDir = d;
            break;
        }
    }
    if (dataDir.empty()) return false;

    std::vector<fs::path> jsonCandidates;
    if (envJson) jsonCandidates.emplace_back(envJson);
    jsonCandidates.emplace_back(dataDir.parent_path().parent_path() / "data_out" / "left_skew_calib.json");
    jsonCandidates.emplace_back(dataDir.parent_path().parent_path() / "data_out" / "laser_calib.json");
    for (auto& j : jsonCandidates) {
        if (fs::exists(j)) { calibJson = j; return true; }
    }
    return false;
}

template <typename F>
float timedMatchMs(F&& execFn, cv::cuda::Stream& stream,
                   cudaEvent_t ev0, cudaEvent_t ev1) {
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    cudaEventRecord(ev0, cs);
    execFn();
    cudaEventRecord(ev1, cs);
    cudaEventSynchronize(ev1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    return ms;
}

} // namespace

TEST(LaserMatchScanV2Bench, CompareAndTime) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir, calibJson;
    if (!findData(dataDir, calibJson)) {
        GTEST_SKIP() << "left_skew data not found; skip v1-vs-v2 bench";
    }
    std::cout << "data: " << dataDir.string() << "\ncalib: " << calibJson.string() << "\n";

    // ---------- 1. 载入标定产物（同 fcscan_e2e） ----------
    json j;
    {
        std::ifstream f(calibJson.string());
        ASSERT_TRUE(f.good()) << "cannot open " << calibJson.string();
        j = json::parse(f);
    }
    auto inputOpt = loadLaserInput(dataDir.string());
    ASSERT_TRUE(inputOpt.has_value()) << "loadLaserInput failed";
    const auto* input = &inputOpt.value();
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    const double fpx = j["pjc"]["f"].get<double>();
    const cv::Point2d pp(j["pjc"]["principalPoint"][0].get<double>(),
                         j["pjc"]["principalPoint"][1].get<double>());
    const cv::Vec3d tRect(j["pjc"]["projectorT"][0].get<double>(),
                          j["pjc"]["projectorT"][1].get<double>(),
                          j["pjc"]["projectorT"][2].get<double>());
    const float rowStep = j["pjc"].value("epipolarRowStep", 0.7f);

    // ---------- 2. curve_map 生成 + 表适配（同 fcscan_e2e） ----------
    CurveMapInput cmi;
    for (const auto& cj : j["pjc"]["emissionCurves"]) {
        ImplicitCurve c;
        for (int k = 0; k < 6; ++k)
            c.coeffs[k] = cj["coeffs"][k].get<double>();
        cmi.curves.push_back(c);
    }
    cmi.f = fpx;
    cmi.principalPoint = pp;
    cv::Matx34d P2(h.P2);
    cmi.baseline = std::fabs(P2(0, 3)) / fpx;
    cmi.imageSize = h.imageSize;
    cmi.virtualT = tRect;
    {
        std::ifstream hf(dataDir.string() + "/camera_calib.json");
        json hj = json::parse(hf);
        const auto& vr = hj["rectify"]["validRoiLeft"];
        cmi.roi = cv::Rect(vr["x"].get<int>(), vr["y"].get<int>(),
                           vr["w"].get<int>(), vr["h"].get<int>());
    }
    CurveMapParams cmp;
    cmp.epipolarRowStep = rowStep;
    cmp.depthMin = cfg.depthMin;
    cmp.depthMax = std::min<float>(cfg.depthMax, 1500.0f);
    CurveMapGenerator cmGen(cmp);
    auto cmr = cmGen.Generate(cmi);
    ASSERT_TRUE(cmr.success) << "curve_map failed: " << cmr.message;

    CurveMapTable cmt;
    std::string lerr;
    ASSERT_TRUE(cmt.Load(cmr.tableBytes.data(), cmr.tableBytes.size(), lerr)) << lerr;
    auto tempTable = std::make_shared<LaserPlaneMapTempTable>();
    {
        std::vector<cv::Vec4f> rows;
        for (uint32_t row = 0; row < cmt.rowCount(); ++row) {
            std::vector<CurveMapEntry> tmp;
            int cap = 4096;
            tmp.resize(cap);
            int n = cmt.EnumerateRow(static_cast<int>(row), tmp.data(), cap);
            while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(static_cast<int>(row), tmp.data(), cap); }
            for (int q = 0; q < n; ++q) {
                rows.push_back(cv::Vec4f(tmp[static_cast<size_t>(q)].uL * rowStep,
                                         static_cast<float>(row) * rowStep,
                                         tmp[static_cast<size_t>(q)].uR * rowStep,
                                         static_cast<float>(tmp[static_cast<size_t>(q)].lid)));
            }
        }
        LaserPlaneMap pm;
        cv::Mat m(static_cast<int>(rows.size()), 1, CV_32FC4, rows.data());
        pm.leftToRightMap = m.clone();
        pm.totalPairs = static_cast<int>(rows.size());
        tempTable->referenceTemperature = cfg.referenceTemp;
        tempTable->table[cfg.referenceTemp] = std::move(pm);
    }

    // ---------- 3. 双算子装配 ----------
    LaserMatchScanParams msp;
    msp.epipolar_row_step = rowStep;
    msp.match_threshold = 2.0f;
    msp.vL_tolerance = rowStep;
    LaserMatchScanCuda matcherV1(msp);

    LaserMatchScanParamsV2 msp2;
    msp2.epipolar_row_step = rowStep;
    msp2.match_threshold = 2.0f;
    msp2.vL_tolerance = rowStep;
    LaserMatchScanCudaV2 matcherV2(msp2);

    ASSERT_TRUE(matcherV1.SetTempTable(tempTable));
    ASSERT_TRUE(matcherV2.SetTempTable(tempTable));
    matcherV1.SetCurrentTemperature(cfg.referenceTemp);
    matcherV2.SetCurrentTemperature(cfg.referenceTemp);
    matcherV1.Warmup(65536, 65536);
    matcherV2.Warmup(65536, 65536);

    // ---------- 4. 前段流水线（同 fcscan_e2e 4-1..4-6） ----------
    cv::cuda::Stream stream;
    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
    MaskExtractCUDA maskL(mp), maskR(mp);
    RegionAnalyzerParams cp; cp.minArea = 0; cp.topXCount = 27;
    RegionAnalyzerCUDA cclL(cp), cclR(cp);
    LaserLabelParams lp; lp.scanDirection = cfg.labelScanDirection;
    lp.centerRowOffset = cfg.labelCenterRowOffset; lp.realLineTolerance = 10;
    LaserLabelerCUDA labL(lp), labR(lp);
    StegerParams sp;
    StegerExtractorCUDA stgL(sp), stgR(sp);
    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3x3; upL.validate();
    UndistortPointsParams upR;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3x3; upR.validate();
    UndistortPointsCuda unL(upL), unR(upR);
    EpipolarInterpParams eip; eip.lineIdCheck = true;
    EpipolarInterpCuda epiL(eip), epiR(eip);

    // dual 有号模式回归（与 opt 逐位一致）
    EpipolarInterpDualParams dip;
    dip.epipolar_row_step = rowStep;
    EpipolarInterpDualCuda depiL(dip), depiR(dip);
    int interpMismatch = 0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    struct FrameStat { std::string stem; int nIn; int matched;
                       double v1Ms = 0, v2Ms = 0; };
    std::vector<FrameStat> stats;
    int mismatchFrames = 0;
    constexpr int kTimedRuns = 5;
    constexpr int kWarmRuns = 2;

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const auto& fr = input->poseFrames[pi];
        for (size_t ti = 0; ti < fr.size(); ++ti) {
            const auto& img = fr[ti];
            const std::string stem =
                input->poseDirs[pi] + "_tube" + std::to_string(ti);

            auto m1 = maskL.Execute(img.leftGray, stream);
            auto m2 = maskR.Execute(img.rightGray, stream);
            if (!m1.success || !m2.success || !m1.d_cleanedMask || !m2.d_cleanedMask)
                { std::cout << stem << ": 4-1 skip\n"; continue; }
            auto c1 = cclL.Execute(*m1.d_cleanedMask, stream);
            auto c2 = cclR.Execute(*m2.d_cleanedMask, stream);
            if (!c1.success || !c2.success || !c1.d_labeledMask || !c2.d_labeledMask)
                { std::cout << stem << ": 4-2 skip\n"; continue; }
            auto l1 = labL.Execute(*c1.d_labeledMask, stream);
            auto l2 = labR.Execute(*c2.d_labeledMask, stream);
            if (!l1.success || !l2.success || !l1.d_labeledMask || !l2.d_labeledMask)
                { std::cout << stem << ": 4-3 skip\n"; continue; }
            auto s1 = stgL.Execute(*m1.d_grayImage, *l1.d_labeledMask, stream, GroupMode::ByLabel);
            auto s2 = stgR.Execute(*m2.d_grayImage, *l2.d_labeledMask, stream, GroupMode::ByLabel);
            if (!s1.success || !s2.success || !s1.d_centerPoints || !s2.d_centerPoints)
                { std::cout << stem << ": 4-4 skip\n"; continue; }
            auto u1 = unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream);
            auto u2 = unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream);
            if (!u1.success || !u2.success || !u1.d_rectifiedPoints || !u2.d_rectifiedPoints)
                { std::cout << stem << ": 4-5 skip\n"; continue; }
            auto e1 = epiL.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream);
            auto e2 = epiR.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream);
            if (!e1.success || !e2.success || !e1.d_interpPoints || !e2.d_interpPoints)
                { std::cout << stem << ": 4-6 skip\n"; continue; }

            // ── dual(labeled) vs opt 逐位一致性回归 ──
            {
                auto d1 = depiL.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream);
                auto d2 = depiR.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream);
                ASSERT_TRUE(d1.success && d2.success) << stem << ": dual interp failed";
                bool ok = (d1.interpCount == e1.interpCount) && (d2.interpCount == e2.interpCount);
                if (ok && d1.interpCount > 0) {
                    cv::Mat a1, b1, a1i, b1i, a2, b2, a2i, b2i;
                    e1.d_interpPoints->download(a1);   d1.d_interpPoints->download(b1);
                    e1.d_interp_line_ids->download(a1i); d1.d_interp_line_ids->download(b1i);
                    e2.d_interpPoints->download(a2);   d2.d_interpPoints->download(b2);
                    e2.d_interp_line_ids->download(a2i); d2.d_interp_line_ids->download(b2i);
                    stream.waitForCompletion();
                    const size_t nb = static_cast<size_t>(d1.interpCount);
                    ok = ok && memcmp(a1.data, b1.data, nb * 8) == 0
                            && memcmp(a1i.data, b1i.data, nb * 4) == 0;
                    const size_t nb2 = static_cast<size_t>(d2.interpCount);
                    ok = ok && memcmp(a2.data, b2.data, nb2 * 8) == 0
                            && memcmp(a2i.data, b2i.data, nb2 * 4) == 0;
                }
                if (!ok) {
                    ++interpMismatch;
                    std::cout << stem << ": dual-vs-opt interp MISMATCH ("
                              << d1.interpCount << " vs " << e1.interpCount << " / "
                              << d2.interpCount << " vs " << e2.interpCount << ")\n";
                }
            }
            if (!e1.d_interp_line_ids || !e2.d_interp_line_ids)
                { std::cout << stem << ": no interp ids skip\n"; continue; }

            const int nL = e1.d_interpPoints->cols;
            const int nR = e2.d_interpPoints->cols;

            // ── 正确性对比（第 1 次运行的结果）──
            auto r1 = matcherV1.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                        *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
            auto r2 = matcherV2.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                        *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
            ASSERT_TRUE(r1.success) << stem << ": v1 failed: " << r1.message;
            ASSERT_TRUE(r2.success) << stem << ": v2 failed: " << r2.message;

            cv::Mat mL1, mR1, mI1, mL2, mR2, mI2, stL1, stR1, stL2, stR2;
            r1.d_matched_left->download(mL1);
            r1.d_matched_right->download(mR1);
            r1.d_matched_line_ids->download(mI1);
            r1.d_left_status->download(stL1);
            r1.d_right_status->download(stR1);
            r2.d_matched_left->download(mL2);
            r2.d_matched_right->download(mR2);
            r2.d_matched_line_ids->download(mI2);
            r2.d_left_status->download(stL2);
            r2.d_right_status->download(stR2);
            stream.waitForCompletion();

            bool ok = (r1.matchedCount == r2.matchedCount) &&
                      (r1.excludedLeftCount == r2.excludedLeftCount) &&
                      (r1.excludedRightCount == r2.excludedRightCount);
            if (ok && r1.matchedCount > 0) {
                // v1 匹配对顺序由块调度决定（atomicAdd 竞态序），跨行拼接非确定；
                // 逐行贪心本身确定 → 配对多重集应严格相等。按 (L, R, lid) 全元组排序后比对。
                const int n = r1.matchedCount;
                using Tuple = std::tuple<float, float, float, float, float, float, int>;
                auto buildSorted = [&](const cv::Mat& L, const cv::Mat& R, const cv::Mat& I) {
                    std::vector<Tuple> v(n);
                    for (int k = 0; k < n; ++k) {
                        const cv::Vec2f l = L.at<cv::Vec2f>(k);
                        const cv::Vec2f r = R.at<cv::Vec2f>(k);
                        v[k] = {l[0], l[1], r[0], r[1], 0.f, 0.f, I.at<int>(k)};
                    }
                    std::sort(v.begin(), v.end());
                    return v;
                };
                auto s1 = buildSorted(mL1, mR1, mI1);
                auto s2 = buildSorted(mL2, mR2, mI2);
                ok = (s1 == s2);
                if (!ok) {
                    int firstDiff = -1;
                    for (int k = 0; k < n && firstDiff < 0; ++k)
                        if (!(s1[k] == s2[k])) firstDiff = k;
                    std::cout << stem << ": first sorted-pair diff at " << firstDiff << "/" << n << "\n";
                }
            }
            if (ok) {
                ok = ok && (memcmp(stL1.data, stL2.data, static_cast<size_t>(nL) * sizeof(int)) == 0);
                ok = ok && (memcmp(stR1.data, stR2.data, static_cast<size_t>(nR) * sizeof(int)) == 0);
            }
            if (!ok) {
                ++mismatchFrames;
                std::cout << stem << ": MISMATCH v1(matched=" << r1.matchedCount
                          << ",exL=" << r1.excludedLeftCount << ",exR=" << r1.excludedRightCount
                          << ") v2(matched=" << r2.matchedCount
                          << ",exL=" << r2.excludedLeftCount << ",exR=" << r2.excludedRightCount << ")\n";
            }

            // ── 计时（预热 + 中位数）──
            for (int k = 0; k < kWarmRuns; ++k) {
                matcherV1.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                  *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
                matcherV2.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                  *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
            }
            std::vector<float> t1(kTimedRuns), t2(kTimedRuns);
            for (int k = 0; k < kTimedRuns; ++k) {
                t1[k] = timedMatchMs([&] {
                    matcherV1.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                      *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
                }, stream, ev0, ev1);
                t2[k] = timedMatchMs([&] {
                    matcherV2.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                      *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
                }, stream, ev0, ev1);
            }
            std::sort(t1.begin(), t1.end());
            std::sort(t2.begin(), t2.end());

            FrameStat st{stem, nL, r1.matchedCount, t1[kTimedRuns / 2], t2[kTimedRuns / 2]};
            stats.push_back(st);
            std::cout << stem << ": in=" << nL << " matched=" << r1.matchedCount
                      << (ok ? " [bit-equal]" : " [MISMATCH]")
                      << "  v1=" << std::fixed << std::setprecision(2) << st.v1Ms
                      << "ms  v2=" << st.v2Ms << "ms\n";
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    // ---------- 5. 汇总 ----------
    ASSERT_FALSE(stats.empty()) << "no frames processed";
    double s1 = 0, s2 = 0;
    double maxV2 = 0;
    for (auto& st : stats) { s1 += st.v1Ms; s2 += st.v2Ms; maxV2 = std::max(maxV2, st.v2Ms); }
    const double avgV1 = s1 / stats.size();
    const double avgV2 = s2 / stats.size();

    std::cout << "\n==== summary over " << stats.size() << " frames ====\n"
              << "v1 avg " << avgV1 << " ms | v2 avg " << avgV2
              << " ms | v2 max " << maxV2 << " ms | speedup "
              << (avgV1 / avgV2) << "x | mismatch frames: " << mismatchFrames << "\n";

    EXPECT_EQ(mismatchFrames, 0) << "v2 results diverge from v1 baseline";
    EXPECT_LE(avgV2, 3.0) << "v2 target <= 3ms not met (avg " << avgV2 << " ms)";
    EXPECT_EQ(interpMismatch, 0) << "dual labeled-mode diverges from opt";
}
