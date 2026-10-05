// test_pose06_lid_vote.cpp - track-connectivity + majority-vote lid disambiguation (pose_06)
// (analysis test, no operator modification; same chains/data as test_scan_vs_calib_analysis)
//
// 2026-09-01 增补（人工指令）：
//   - 扫描链前端并行运行 CCL（RegionAnalyzer，仅计时/可视化对照，不进链——steger_fast
//     仍吃 cleanedMask Flat，链语义与金标准对拍不变）
//   - 每步骤产物 → data_out/analysis/steps/：
//       01_mask_{L,R}.png / 02_ccl_{L,R}.png / 03_steger_{L,R}.png /
//       04_undistort_{L,R}.png / 05_interp_tracks_{L,R}.png
//       06_recon_calib_truth.ply（标定链真值云）/ 06_recon_scan_voted.ply（扫描链投票云）
//       summary_timing.txt（含 interp_dual 内段链接连通计算耗时 scan_link_ms）
//   - EpipolarInterpDualCuda::Execute 的 Scan 段链接相位新增 GPU 计时（cudaEvent，
//     经 EpipolarInterpResult::scan_link_ms 暴露；Labeled 路径恒 0）
//
// Pipeline (params validated against 2026-08-30 data run):
//   1. per scan-left-point HIT lid candidates: table entries with |uL-lx|<=0.7 whose
//      predicted uR has an actual right interp point within +-2px
//      (line uR gap min 116px, table within2px=95.8%, identity-miss=0)
//   2. 1D row tracking: each line is single-valued x(row) (skew form, scanDirection=1);
//      link consecutive rows with |dx|<=12px, tolerate <=3-row gaps
//   3. per-track majority vote; gate: size>=5 AND winner votes >= 50% of track points
//      (expected true-lid HIT rate ~81% due to 19% right-side leak)
//   4. gate-failed tracks: per-point depth tiebreak vs bootstrap Zmed
//      (bootstrap w=30 phantom kill rate measured 100%)
//   5. match voted lid (nearest-uL table pred uR -> nearest right pt +-2px),
//      reconstruct, PLY; stats in V0-V8 format
// Baselines to beat (pose_06): V8d correct=18550 wrong=0 lost=1222 of 19772.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
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
#include "curve_map.h"
#include "laser_match_scan_cuda.h"
#include "laser_match_scan_v3_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "common/calib_result_types.h"

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
        if (fs::exists(d / "config.json") && fs::exists(d / "camera_calib.json")) { dataDir = d; break; }
    }
    if (dataDir.empty()) return false;
    std::vector<fs::path> jsonCandidates;
    if (envJson) jsonCandidates.emplace_back(envJson);
    jsonCandidates.emplace_back(dataDir.parent_path().parent_path() / "data_out" / "left_skew_calib.json");
    jsonCandidates.emplace_back(dataDir.parent_path().parent_path() / "data_out" / "laser_calib.json");
    for (auto& j : jsonCandidates) { if (fs::exists(j)) { calibJson = j; return true; } }
    return false;
}

void writePly(const std::string& path, const std::vector<cv::Vec3f>& pts,
              const std::vector<char>& ok) {
    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
      << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < pts.size(); ++i)
        f << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << " "
          << (ok[i] ? "0 200 0" : "255 0 0") << "\n";
}

// ── 步骤产物辅助（同 fcchain_v3）──
cv::Vec3b labelColor(int label) {
    if (label <= 0) return {0, 0, 0};
    const int h = (label * 47) % 180;
    cv::Mat m(1, 1, CV_8UC3, cv::Scalar(h, 255, 255));
    cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
    return m.at<cv::Vec3b>(0, 0);
}

void savePts(const std::string& path, const cv::Mat& base,
             const std::vector<cv::Point2f>& pts, const cv::Scalar& color,
             bool blackBg = false) {
    cv::Mat img;
    if (blackBg) img = cv::Mat::zeros(base.size(), CV_8UC3);
    else if (base.channels() == 1) cv::cvtColor(base, img, cv::COLOR_GRAY2BGR);
    else img = base.clone();
    for (auto& p : pts)
        cv::circle(img, {cvRound(p.x), cvRound(p.y)}, 1, color, -1);
    cv::imwrite(path, img);
}

struct MatchPairRec {
    float lx = 0, ly = 0, rx = 0, ry = 0;
    int lid = -1;
    cv::Vec3f p3;
};

std::vector<cv::Point2f> downloadPoints(const cv::cuda::GpuMat& d_pts) {
    cv::Mat m;
    d_pts.download(m);
    if (m.empty()) return {};
    m = m.reshape(2, 1);
    return std::vector<cv::Point2f>(m.begin<cv::Point2f>(), m.end<cv::Point2f>());
}

} // namespace

TEST(Pose06TrackVoteLid, Run) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir, calibJson;
    if (!findData(dataDir, calibJson)) {
        GTEST_SKIP() << "left_skew data not found; skip analysis";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "analysis";
    fs::create_directories(outDir);
    std::ofstream summary((outDir / "summary_trackvote.txt").string());
    auto logLine = [&](const std::string& s) { std::cout << s << "\n"; summary << s << "\n"; };

    // ---------- load calibration + build table (same as scan_vs_calib) ----------
    json j;
    {
        std::ifstream f(calibJson.string());
        ASSERT_TRUE(f.good());
        j = json::parse(f);
    }
    auto inputOpt = loadLaserInput(dataDir.string());
    ASSERT_TRUE(inputOpt.has_value());
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
    const float HIT_WIN = [] {
        const char* e = std::getenv("TV_HITWIN");
        return e ? static_cast<float>(std::atof(e)) : 3.0f;
    }();
    const float UL_TOL = [] {
        const char* e = std::getenv("TV_ULTOL");
        return e ? static_cast<float>(std::atof(e)) : 0.7f;
    }();
    const int NEARN = [] {
        const char* e = std::getenv("TV_NEARN");
        return e ? std::atoi(e) : 0;
    }();
    std::cout << "HIT_WIN = " << HIT_WIN << " UL_TOL = " << UL_TOL
              << " NEARN = " << NEARN << "\n";

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
    ASSERT_TRUE(cmr.success) << cmr.message;
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
            for (int q = 0; q < n; ++q)
                rows.push_back(cv::Vec4f(tmp[static_cast<size_t>(q)].uL * rowStep,
                                         static_cast<float>(row) * rowStep,
                                         tmp[static_cast<size_t>(q)].uR * rowStep,
                                         static_cast<float>(tmp[static_cast<size_t>(q)].lid)));
        }
        LaserPlaneMap pm;
        cv::Mat m(static_cast<int>(rows.size()), 1, CV_32FC4, rows.data());
        pm.leftToRightMap = m.clone();
        pm.totalPairs = static_cast<int>(rows.size());
        tempTable->referenceTemperature = cfg.referenceTemp;
        tempTable->table[cfg.referenceTemp] = std::move(pm);
    }

    // ---------- operators ----------
    cv::cuda::Stream stream;

    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
    MaskExtractCUDA maskC1(mp), maskC2(mp), maskS1(mp), maskS2(mp);

    RegionAnalyzerParams cp; cp.minArea = 0; cp.topXCount = 27;
    RegionAnalyzerCUDA cclL(cp), cclR(cp);
    RegionAnalyzerCUDA cclSL(cp), cclSR(cp);   // 扫描链前端 CCL（计时/可视化对照，不进链——保持链语义与金标准对拍不变）
    LaserLabelParams lp; lp.scanDirection = cfg.labelScanDirection;
    lp.centerRowOffset = cfg.labelCenterRowOffset; lp.realLineTolerance = 10;
    LaserLabelerCUDA labL(lp), labR(lp);

    StegerParams sp;
    StegerExtractorCUDA stgC1(sp), stgC2(sp);
    StegerExtractorFast stgS1(sp), stgS2(sp);

    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3x3; upL.validate();
    UndistortPointsParams upR;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3x3; upR.validate();
    UndistortPointsCuda unL(upL), unR(upR);

    EpipolarInterpParams eip; eip.epipolar_row_step = rowStep; eip.lineIdCheck = true;
    EpipolarInterpCuda epiC1(eip), epiC2(eip);
    EpipolarInterpDualParams dip; dip.epipolar_row_step = rowStep;
    EpipolarInterpDualCuda epiS1(dip), epiS2(dip);

    LaserMatchScanParams msp;
    msp.epipolar_row_step = rowStep;
    msp.match_threshold = 2.0f;
    msp.vL_tolerance = rowStep;
    LaserMatchScanCuda matcherV1(msp);
    ASSERT_TRUE(matcherV1.SetTempTable(tempTable));
    matcherV1.SetCurrentTemperature(cfg.referenceTemp);
    matcherV1.Warmup(65536, 65536);

    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    auto runCalib = [&](size_t pi, size_t ti) {
        const auto& img = input->poseFrames[pi][ti];
        auto m1 = maskC1.Execute(img.leftGray, stream);
        auto m2 = maskC2.Execute(img.rightGray, stream);
        auto c1 = cclL.Execute(*m1.d_cleanedMask, stream);
        auto c2 = cclR.Execute(*m2.d_cleanedMask, stream);
        auto l1 = labL.Execute(*c1.d_labeledMask, stream);
        auto l2 = labR.Execute(*c2.d_labeledMask, stream);
        auto s1 = stgC1.Execute(*m1.d_grayImage, *l1.d_labeledMask, stream, GroupMode::ByLabel);
        auto s2 = stgC2.Execute(*m2.d_grayImage, *l2.d_labeledMask, stream, GroupMode::ByLabel);
        auto u1 = unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream);
        auto u2 = unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream);
        auto e1 = epiC1.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream);
        auto e2 = epiC2.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream);
        stream.waitForCompletion();
        auto mr = matcherV1.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                     *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
        std::vector<MatchPairRec> out;
        if (!mr.success || mr.matchedCount <= 0) return out;
        auto rr = recon.Execute(*mr.d_matched_left, *mr.d_matched_right,
                                *mr.d_matched_line_ids, h.Q, stream);
        if (!rr.success || !rr.d_points3d) return out;
        cv::Mat L, R, I, P3, I3;
        mr.d_matched_left->download(L);
        mr.d_matched_right->download(R);
        mr.d_matched_line_ids->download(I);
        rr.d_points3d->download(P3);
        rr.d_valid_line_ids->download(I3);
        stream.waitForCompletion();
        L = L.reshape(2, 1); R = R.reshape(2, 1);
        P3 = P3.reshape(3, 1); I3 = I3.reshape(1, 1);
        const int n = mr.matchedCount;
        out.resize(n);
        for (int k = 0; k < n; ++k) {
            const cv::Vec2f l = L.at<cv::Vec2f>(k), r = R.at<cv::Vec2f>(k);
            out[k] = {l[0], l[1], r[0], r[1], I3.at<int>(k), P3.at<cv::Vec3f>(k)};
        }
        return out;
    };

    auto runScanPts = [&](size_t pi, size_t ti,
                          std::vector<cv::Point2f>& outL,
                          std::vector<cv::Point2f>& outR) {
        const auto& img = input->poseFrames[pi][ti];
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
        std::vector<std::pair<std::string, float>> stepMs;
        auto timed = [&](const char* name, auto&& fn) {
            cudaEvent_t s, e;
            cudaEventCreate(&s);
            cudaEventCreate(&e);
            cudaEventRecord(s, cs);
            auto r = fn();
            cudaEventRecord(e, cs);
            cudaEventSynchronize(e);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, s, e);
            cudaEventDestroy(s);
            cudaEventDestroy(e);
            stepMs.push_back({name, ms});
            return r;
        };
        // 扫描链（前端 CCL 为对照运行，不计入链语义：steger_fast 仍吃 cleanedMask Flat）
        auto m1 = timed("mask(L)", [&] { return maskS1.Execute(img.leftGray, stream); });
        auto m2 = timed("mask(R)", [&] { return maskS2.Execute(img.rightGray, stream); });
        auto c1 = timed("ccl(L)", [&] { return cclSL.Execute(*m1.d_cleanedMask, stream); });
        auto c2 = timed("ccl(R)", [&] { return cclSR.Execute(*m2.d_cleanedMask, stream); });
        auto s1 = timed("steger_fast(L)", [&] {
            return stgS1.Execute(*m1.d_grayImage, *m1.d_cleanedMask, stream, GroupMode::Flat); });
        auto s2 = timed("steger_fast(R)", [&] {
            return stgS2.Execute(*m2.d_grayImage, *m2.d_cleanedMask, stream, GroupMode::Flat); });
        auto u1 = timed("undistort(L)", [&] {
            return unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream); });
        auto u2 = timed("undistort(R)", [&] {
            return unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream); });
        auto e1 = timed("interp_dual(L)", [&] {
            return epiS1.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream); });
        auto e2 = timed("interp_dual(R)", [&] {
            return epiS2.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream); });
        outL = downloadPoints(*e1.d_interpPoints);
        outR = downloadPoints(*e2.d_interpPoints);
        stream.waitForCompletion();
        // GPU 段链接输出验证：interp_line_ids = 段 ID
        {
            cv::Mat f1, f2;
            e1.d_interp_line_ids->download(f1);
            e2.d_interp_line_ids->download(f2);
            f1 = f1.reshape(1, 1);
            f2 = f2.reshape(1, 1);
            auto segStat = [](const cv::Mat& f, const char* tag) {
                std::map<int, int> cnt;
                const int n = f.cols;
                for (int k = 0; k < n; ++k) ++cnt[f.at<int>(k)];
                size_t mx = 0;
                for (auto& kv : cnt) mx = std::max(mx, (size_t)kv.second);
                std::ostringstream os;
                os << "GPU track ids [" << tag << "]: pts=" << n
                   << " segments=" << cnt.size() << " maxSeg=" << mx
                   << " (ids: cluster-code roots)";
                return os.str();
            };
            logLine(segStat(f1, "left"));
            logLine(segStat(f2, "right"));
        }
        // ── 每步骤产物：图像（01~05，L/R）＋计时表（含 interp_dual 内段链接连通计算耗时）──
        const fs::path stepsDir = outDir / "steps";
        fs::create_directories(stepsDir);
        const std::string pre = (stepsDir / "").string();
        {   // 01 mask
            cv::Mat mL, mR;
            m1.d_cleanedMask->download(mL, stream);
            m2.d_cleanedMask->download(mR, stream);
            stream.waitForCompletion();
            cv::imwrite(pre + "01_mask_L.png", mL);
            cv::imwrite(pre + "01_mask_R.png", mR);
        }
        {   // 02 ccl colored（前端 CCL 对照）
            for (int side = 0; side < 2; ++side) {
                cv::Mat lab;
                if (side == 0) c1.d_labeledMask->download(lab, stream);
                else c2.d_labeledMask->download(lab, stream);
                stream.waitForCompletion();
                cv::Mat col(lab.size(), CV_8UC3, cv::Scalar(0, 0, 0));
                for (int y = 0; y < lab.rows; ++y)
                    for (int x = 0; x < lab.cols; ++x) {
                        const int l = lab.at<int>(y, x);
                        if (l > 0) col.at<cv::Vec3b>(y, x) = labelColor(l);
                    }
                cv::imwrite(pre + "02_ccl_" + std::string(side ? "R" : "L") + ".png", col);
            }
        }
        auto dlPts = [&](const cv::cuda::GpuMat& g) {
            cv::Mat m;
            g.download(m, stream);
            stream.waitForCompletion();
            m = m.reshape(2, 1);
            return std::vector<cv::Point2f>(m.begin<cv::Point2f>(), m.end<cv::Point2f>());
        };
        {   // 03 steger centers on raw gray
            savePts(pre + "03_steger_L.png", img.leftGray, dlPts(*s1.d_centerPoints), {0, 0, 255});
            savePts(pre + "03_steger_R.png", img.rightGray, dlPts(*s2.d_centerPoints), {0, 0, 255});
        }
        {   // 04 undistorted pts on rectified image
            cv::Mat mxL, myL, mxR, myR, recL, recR;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        img.leftGray.size(), CV_32FC1, mxL, myL);
            cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                        img.rightGray.size(), CV_32FC1, mxR, myR);
            cv::remap(img.leftGray, recL, mxL, myL, cv::INTER_LINEAR);
            cv::remap(img.rightGray, recR, mxR, myR, cv::INTER_LINEAR);
            savePts(pre + "04_undistort_L.png", recL, dlPts(*u1.d_rectifiedPoints), {255, 0, 0});
            savePts(pre + "04_undistort_R.png", recR, dlPts(*u2.d_rectifiedPoints), {255, 0, 0});
        }
        {   // 05 interp pts colored by segment id
            for (int side = 0; side < 2; ++side) {
                const auto& e = side ? e2 : e1;
                auto pts = dlPts(*e.d_interpPoints);
                cv::Mat fids;
                e.d_interp_line_ids->download(fids, stream);
                stream.waitForCompletion();
                fids = fids.reshape(1, 1);
                cv::Mat imgv(img.leftGray.size(), CV_8UC3, cv::Scalar(0, 0, 0));
                for (size_t i = 0; i < pts.size(); ++i) {
                    const int sid = fids.at<int>(static_cast<int>(i));
                    const cv::Vec3b c = labelColor((sid * 7) % 180 + 1);
                    cv::circle(imgv, {cvRound(pts[i].x), cvRound(pts[i].y)}, 1,
                               cv::Scalar(c[0], c[1], c[2]), -1);
                }
                cv::imwrite(pre + "05_interp_tracks_" + std::string(side ? "R" : "L") + ".png", imgv);
            }
        }
        {   // 计时表（终端+summary+steps/summary_timing.txt；三维点云 PLY 见 06_recon_*）
            std::ostringstream os;
            os << "scan chain step timing (cuda events, ms):\n";
            for (auto& kv : stepMs) os << "  " << kv.first << " = " << kv.second << "\n";
            os << "  interp_dual(L) internal scan-link = " << e1.scan_link_ms << "\n"
               << "  interp_dual(R) internal scan-link = " << e2.scan_link_ms << "\n"
               << "  steps output -> " << pre;
            logLine(os.str());
            std::ofstream tf((stepsDir / "summary_timing.txt").string());
            tf << os.str();
        }
    };

    auto calibPairs = runCalib(0, 0);
    ASSERT_FALSE(calibPairs.empty()) << "calib chain empty";
    // [消融实验开关] FC2_ABLATE_CALIB_LID=1：打乱标定链逐点线号（几何不变）——
    // 验证扫描链匹配是否依赖本帧标定链的激光编号数据（预期：matched 不变）
    if (std::getenv("FC2_ABLATE_CALIB_LID")) {
        std::mt19937 rng(12345);
        std::vector<int> lids;
        lids.reserve(calibPairs.size());
        for (auto& r : calibPairs) lids.push_back(r.lid);
        std::shuffle(lids.begin(), lids.end(), rng);
        size_t ki = 0;
        for (auto& r : calibPairs) r.lid = lids[ki++];
        logLine("ABLATION: calib chain per-point lids scrambled (geometry untouched)");
    }
    std::vector<cv::Point2f> sL, sR;
    runScanPts(0, 0, sL, sR);
    ASSERT_FALSE(sL.empty());
    {   // 步骤产物：标定链（真值）三维点云
        const fs::path stepsDir = outDir / "steps";
        fs::create_directories(stepsDir);
        std::vector<cv::Vec3f> cloud;
        std::vector<char> ok;
        cloud.reserve(calibPairs.size());
        ok.reserve(calibPairs.size());
        for (auto& r : calibPairs) { cloud.push_back(r.p3); ok.push_back(1); }
        writePly((stepsDir / "06_recon_calib_truth.ply").string(), cloud, ok);
        logLine("steps cloud 06_recon_calib_truth.ply: " + std::to_string(cloud.size())
                + " pts (calib chain ground truth)");
    }

    const double fB = fpx * cmi.baseline;
    const int calibTotal = static_cast<int>(calibPairs.size());

    // truth index by row
    std::map<int, std::vector<const MatchPairRec*>> truthByRow;
    for (auto& r : calibPairs)
        truthByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
    auto isTruePair = [&](int row, float lx, float rx) {
        auto it = truthByRow.find(row);
        if (it == truthByRow.end()) return false;
        for (auto* c : it->second)
            if (std::fabs(c->lx - lx) < 0.6f && std::fabs(c->rx - rx) < 0.5f) return true;
        return false;
    };
    auto trueLidOf = [&](int row, float lx) -> int {
        auto it = truthByRow.find(row);
        if (it == truthByRow.end()) return -1;
        for (auto* c : it->second)
            if (std::fabs(c->lx - lx) < 0.6f) return c->lid;
        return -1;
    };

    // right pts by row (sorted)
    std::map<int, std::vector<float>> rByRow;
    for (auto& p : sR)
        rByRow[static_cast<int>(std::lround(p.y / rowStep))].push_back(p.x);
    for (auto& kv : rByRow) std::sort(kv.second.begin(), kv.second.end());
    auto nearestRight = [&](int row, float x, float win) -> float {
        auto it = rByRow.find(row);
        if (it == rByRow.end()) return -1.0f;
        auto& v = it->second;
        int lo = static_cast<int>(std::lower_bound(v.begin(), v.end(), x - win) - v.begin());
        float best = -1.0f, bestD = win + 1.0f;
        for (int k = lo; k < static_cast<int>(v.size()) && v[k] <= x + win; ++k) {
            float d = std::fabs(v[k] - x);
            if (d < bestD) { bestD = d; best = v[k]; }
        }
        return best;
    };

    // table cache: row -> lid -> sweep (uL px, uR px) sorted by uL
    std::map<int, std::map<int, std::vector<std::pair<float, float>>>> tabCache;
    auto getTab = [&](int row) -> std::map<int, std::vector<std::pair<float, float>>>& {
        auto it = tabCache.find(row);
        if (it != tabCache.end()) return it->second;
        std::vector<CurveMapEntry> tmp;
        int cap = 4096;
        tmp.resize(cap);
        int n = cmt.EnumerateRow(row, tmp.data(), cap);
        while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
        std::map<int, std::vector<std::pair<float, float>>> byLid;
        for (int q = 0; q < n; ++q)
            byLid[tmp[static_cast<size_t>(q)].lid].emplace_back(
                tmp[static_cast<size_t>(q)].uL * rowStep,
                tmp[static_cast<size_t>(q)].uR * rowStep);
        for (auto& kv : byLid) std::sort(kv.second.begin(), kv.second.end());
        tabCache[row] = std::move(byLid);
        return tabCache[row];
    };

    // ---------- step 1: per-left-point HIT lid candidates ----------
    struct Cand { int lid; float predU; float hitRx; };
    struct LPt {
        float x; int row; int trueLid;
        std::vector<Cand> cands;
        int trackId = -1, lid = -1;
    };
    std::vector<LPt> pts;
    pts.reserve(sL.size());
    int ptsWithTruth = 0, ptsWithCand = 0;
    for (auto& p : sL) {
        const int row = static_cast<int>(std::lround(p.y / rowStep));
        if (row < 0 || static_cast<uint32_t>(row) >= cmt.rowCount()) continue;
        LPt pt;
        pt.x = p.x; pt.row = row; pt.trueLid = trueLidOf(row, p.x);
        if (pt.trueLid > 0) ++ptsWithTruth;
        auto& tab = getTab(row);
        std::map<long long, std::pair<float, int>> cand;   // (lid, uRbin) -> (uR, lid)
        for (auto& bl : tab) {
            auto& v = bl.second;
            int lo = 0, hi = static_cast<int>(v.size());
            while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < p.x) lo = mid + 1; else hi = mid; }
            const int down = static_cast<int>(std::ceil(UL_TOL / rowStep));
            if (NEARN > 0) {
                // take the NEARN nearest sweep entries (each independently HIT-checked)
                std::vector<std::pair<float, int>> byDist;
                for (int k = std::max(0, lo - 3);
                     k <= std::min(static_cast<int>(v.size()) - 1, lo + 2); ++k)
                    byDist.push_back({std::fabs(v[static_cast<size_t>(k)].first - p.x), k});
                std::sort(byDist.begin(), byDist.end());
                const int take = std::min(NEARN, static_cast<int>(byDist.size()));
                for (int i = 0; i < take; ++i) {
                    const int k = byDist[static_cast<size_t>(i)].second;
                    const long long key = (static_cast<long long>(bl.first) << 20)
                                        | static_cast<long long>(std::lround(v[static_cast<size_t>(k)].second / rowStep));
                    if (!cand.count(key)) cand[key] = {v[static_cast<size_t>(k)].second, bl.first};
                }
            } else {
                for (int k = std::max(0, lo - down);
                     k <= std::min(static_cast<int>(v.size()) - 1, lo + down - 1); ++k) {
                    if (std::fabs(v[k].first - p.x) > UL_TOL) continue;
                    const long long key = (static_cast<long long>(bl.first) << 20)
                                        | static_cast<long long>(std::lround(v[k].second / rowStep));
                    if (!cand.count(key)) cand[key] = {v[k].second, bl.first};
                }
            }
        }
        for (auto& kv : cand) {
            const float predU = kv.second.first;
            const float hitRx = nearestRight(row, predU, HIT_WIN);
            if (hitRx < 0) continue;
            pt.cands.push_back({kv.second.second, predU, hitRx});
        }
        if (!pt.cands.empty()) ++ptsWithCand;
        pts.push_back(std::move(pt));
    }

    // ---------- table-lookup export: ALL left/right points with candidate laser ids ----------
    {
        const fs::path csvPath = outDir / "table_lookup_points.csv";
        std::ofstream cf(csvPath.string());
        cf << "side,row,x,y,trueLid,candLids,candCount\n";
        int l1 = 0, l2 = 0, l3 = 0, l4 = 0, lHasTrue = 0, lTrueIn = 0, lNoCand = 0;
        for (auto& pt : pts) {
            std::string ids;
            int n = 0;
            for (auto& c : pt.cands) {
                if (!ids.empty()) ids += " ";
                ids += std::to_string(c.lid);
                ++n;
            }
            if (n == 0) ++lNoCand; else if (n == 1) ++l1; else if (n == 2) ++l2;
            else if (n == 3) ++l3; else ++l4;
            if (pt.trueLid > 0) {
                ++lHasTrue;
                bool in = false;
                for (auto& c : pt.cands)
                    if (c.lid == pt.trueLid) { in = true; break; }
                if (in) ++lTrueIn;
            }
            cf << "L," << pt.row << "," << std::fixed << std::setprecision(2)
               << pt.x << "," << static_cast<float>(pt.row) * rowStep << ","
               << pt.trueLid << ",\"" << ids << "\"," << n << "\n";
        }
        // right points: reverse table lookup (all lids whose sweep uR within 0.7px)
        std::map<int, std::vector<std::pair<float, int>>> rowEnt;   // row -> (uRpx, lid)
        auto getRowEnt = [&](int row) -> std::vector<std::pair<float, int>>& {
            auto it = rowEnt.find(row);
            if (it != rowEnt.end()) return it->second;
            std::vector<CurveMapEntry> tmp;
            int cap = 4096;
            tmp.resize(cap);
            int n = cmt.EnumerateRow(row, tmp.data(), cap);
            while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
            std::vector<std::pair<float, int>> es;
            for (int q = 0; q < n; ++q)
                es.push_back({tmp[static_cast<size_t>(q)].uR * rowStep,
                              static_cast<int>(tmp[static_cast<size_t>(q)].lid)});
            rowEnt[row] = std::move(es);
            return rowEnt[row];
        };
        std::map<int, std::vector<std::pair<float, int>>> rTr;
        for (auto& r : calibPairs)
            rTr[static_cast<int>(std::lround(r.ly / rowStep))].push_back({r.rx, r.lid});
        int r1 = 0, r2 = 0, r3 = 0, r4 = 0, rHasTrue = 0, rTrueIn = 0, rNoCand = 0;
        for (auto& p : sR) {
            const int row = static_cast<int>(std::lround(p.y / rowStep));
            int trueLid = -1;
            float bd = 2.0f;
            auto tit = rTr.find(row);
            if (tit != rTr.end())
                for (auto& e : tit->second) {
                    const float d = std::fabs(e.first - p.x);
                    if (d < bd) { bd = d; trueLid = e.second; }
                }
            std::set<int> ids;
            if (row >= 0 && static_cast<uint32_t>(row) < cmt.rowCount())
                for (auto& e : getRowEnt(row))
                    if (std::fabs(e.first - p.x) <= 0.7f) ids.insert(e.second);
            const int n = static_cast<int>(ids.size());
            if (n == 0) ++rNoCand; else if (n == 1) ++r1; else if (n == 2) ++r2;
            else if (n == 3) ++r3; else ++r4;
            if (trueLid > 0) {
                ++rHasTrue;
                if (ids.count(trueLid)) ++rTrueIn;
            }
            std::string idstr;
            for (int id : ids) {
                if (!idstr.empty()) idstr += " ";
                idstr += std::to_string(id);
            }
            cf << "R," << row << "," << std::fixed << std::setprecision(2)
               << p.x << "," << p.y << "," << trueLid << ",\"" << idstr << "\"," << n << "\n";
        }
        std::ostringstream ose;
        ose << "table-lookup export -> " << csvPath.string() << "\n";
        ose << "LEFT pts=" << pts.size() << " candCount hist: 0=" << lNoCand
            << " 1=" << l1 << " 2=" << l2 << " 3=" << l3 << " 4+=" << l4
            << " | truth-known=" << lHasTrue << " true-in-cands=" << lTrueIn
            << " (" << (lHasTrue ? 100.0 * lTrueIn / lHasTrue : 0.0) << "%)";
        logLine(ose.str());
        std::ostringstream ose2;
        ose2 << "RIGHT pts=" << sR.size() << " candCount hist: 0=" << rNoCand
             << " 1=" << r1 << " 2=" << r2 << " 3=" << r3 << " 4+=" << r4
             << " | truth-known=" << rHasTrue << " true-in-cands=" << rTrueIn
             << " (" << (rHasTrue ? 100.0 * rTrueIn / rHasTrue : 0.0) << "%)";
        logLine(ose2.str());
    }

    // ---------- step 2: 1D row tracking ----------
    // diagnostic first: per-lid slope + adjacent-line gap/crossing from calib truth
    {
        std::map<int, std::map<int, float>> lidRowX;   // lid -> row -> x
        for (auto& r : calibPairs)
            lidRowX[r.lid][static_cast<int>(std::lround(r.ly / rowStep))] = r.lx;
        std::ostringstream os;
        os << "calib trace stats per lid (maxDxPerRow / xRange):";
        logLine(os.str());
        for (auto& kv : lidRowX) {
            float maxDx = 0;
            auto it1 = kv.second.begin();
            for (auto it2 = std::next(it1); it2 != kv.second.end(); ++it1, ++it2) {
                if (it2->first - it1->first == 1)
                    maxDx = std::max(maxDx, std::fabs(it2->second - it1->second));
            }
            std::ostringstream os2;
            os2 << "  lid=" << kv.first << " maxDx/row=" << maxDx
                << " xRange=[" << kv.second.begin()->second << ","
                << kv.second.rbegin()->second << "] rows=" << kv.second.size();
            logLine(os2.str());
        }
        int crossings = 0;
        float minGapOverall = 1e9f;
        std::map<int, std::vector<std::pair<int, int>>> rowsSorted;
        for (auto& kv : lidRowX)
            for (auto& rx2 : kv.second) rowsSorted[rx2.first].push_back({kv.first, rx2.second});
        std::map<int, int> prevOrder;
        for (auto& kv : rowsSorted) {
            std::sort(kv.second.begin(), kv.second.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
            std::map<int, int> order;
            for (size_t i = 0; i < kv.second.size(); ++i)
                order[kv.second[i].first] = static_cast<int>(i);
            if (!prevOrder.empty()) {
                for (size_t i = 0; i + 1 < kv.second.size(); ++i) {
                    const float gap = kv.second[i + 1].second - kv.second[i].second;
                    minGapOverall = std::min(minGapOverall, gap);
                }
                for (auto& o : order) {
                    auto it = prevOrder.find(o.first);
                    if (it != prevOrder.end() && std::fabs(it->second - o.second) >= 2) ++crossings;
                }
            }
            prevOrder = order;
        }
        std::ostringstream os3;
        os3 << "adjacent-line min x-gap (calib, left image) = " << minGapOverall
            << "px | order-swaps across rows = " << crossings;
        logLine(os3.str());
    }

    std::map<int, std::vector<int>> lByRow;
    for (size_t i = 0; i < pts.size(); ++i)
        lByRow[pts[i].row].push_back(static_cast<int>(i));
    for (auto& kv : lByRow)
        std::sort(kv.second.begin(), kv.second.end(),
                  [&](int a, int b) { return pts[a].x < pts[b].x; });

    struct Track { std::vector<int> idx; };
    std::vector<Track> tracks;
    struct Tail { float lastX; int lastRow; int trackId; };
    std::vector<Tail> tails;
    const float LINK_DX = 3.0f;
    const int LINK_GAP_ROWS = 10;
    std::map<std::string, int> linkGapHist;
    auto gapBucket = [](int g) -> std::string {
        if (g <= 1) return "1";
        if (g <= 3) return "2-3";
        if (g <= 5) return "4-5";
        if (g <= 10) return "6-10";
        if (g <= 30) return "11-30";
        return ">30";
    };
    // nearest-neighbor linking: each point joins the CLOSEST tail within LINK_DX
    for (auto& kv : lByRow) {
        const int row = kv.first;
        tails.erase(std::remove_if(tails.begin(), tails.end(),
                    [&](const Tail& t) { return row - t.lastRow > LINK_GAP_ROWS; }),
                    tails.end());
        std::vector<char> used(tails.size(), 0);
        const size_t nTails0 = tails.size();
        for (int pi : kv.second) {
            const float x = pts[static_cast<size_t>(pi)].x;
            int best = -1;
            float bestDx = LINK_DX;
            for (size_t t = 0; t < nTails0; ++t) {
                if (used[t]) continue;
                const float dx = std::fabs(tails[t].lastX - x);
                if (dx < bestDx) { bestDx = dx; best = static_cast<int>(t); }
            }
            if (best >= 0) {
                Tail& tl = tails[static_cast<size_t>(best)];
                used[static_cast<size_t>(best)] = 1;
                ++linkGapHist[gapBucket(row - tl.lastRow)];
                tracks[static_cast<size_t>(tl.trackId)].idx.push_back(pi);
                pts[static_cast<size_t>(pi)].trackId = tl.trackId;
                tl.lastX = x;
                tl.lastRow = row;
            } else {
                pts[static_cast<size_t>(pi)].trackId = static_cast<int>(tracks.size());
                tracks.push_back({{pi}});
                tails.push_back({x, row, static_cast<int>(tracks.size()) - 1});
            }
        }
    }
    {
        std::ostringstream os;
        os << "link gap hist (bridged breaks):";
        for (auto& kv2 : linkGapHist) os << " " << kv2.first << ":" << kv2.second;
        logLine(os.str());
    }

    // ---------- step 3: bootstrap Zmed from unique-HIT points ----------
    std::vector<float> uniqDepths;
    for (auto& pt : pts)
        if (pt.cands.size() == 1)
            uniqDepths.push_back(static_cast<float>(
                fB / std::max(1e-3, static_cast<double>(pt.x) - pt.cands[0].hitRx)));
    std::sort(uniqDepths.begin(), uniqDepths.end());
    const float Zmed = uniqDepths.empty() ? 260.0f : uniqDepths[uniqDepths.size() / 2];

    // ---------- step 4: per-track vote + gate; tiebreak for failed tracks ----------
    const size_t MIN_TRACK = 20;
    const float VOTE_GATE = 0.5f;
    const float DEPTH_WIN = 30.0f;
    auto candDepthOk = [&](const LPt& pt, const Cand& c) {
        const float Z = static_cast<float>(
            fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
        return std::fabs(Z - Zmed) <= DEPTH_WIN;
    };
    // diagnostic: per true-lid candidate availability (raw vs depth-gated)
    {
        std::map<int, std::array<int, 3>> m;   // lid -> {truthPts, hasTrueCand, hasTrueCandInWin}
        for (auto& pt : pts) {
            if (pt.trueLid <= 0) continue;
            auto& e = m[pt.trueLid];
            ++e[0];
            for (auto& c : pt.cands) {
                if (c.lid == pt.trueLid) { ++e[1]; if (candDepthOk(pt, c)) { ++e[2]; break; } break; }
            }
        }
        logLine("true-cand availability per lid (truthPts / hasTrueCand / trueCandInDepthWin):");
        for (auto& kv : m) {
            std::ostringstream os;
            os << "  lid=" << kv.first << " " << kv.second[0] << " / " << kv.second[1]
               << " / " << kv.second[2];
            logLine(os.str());
        }
    }
    int keptTracks = 0, gatePass = 0, gateFail = 0;
    int smallTracks = 0;
    std::map<int, std::pair<int, int>> lidAgg;   // winner lid -> (tracks, pts)
    std::map<int, std::pair<int, int>> lidTrue;  // winner lid -> (pts with truth, truth==winner)
    int lidAccTot = 0, lidAccOk = 0;
    int tiebreakPts = 0, tiebreakOk = 0;
    for (auto& tr : tracks) {
        if (tr.idx.size() < MIN_TRACK) { ++smallTracks; continue; }
        ++keptTracks;
        std::map<int, int> votes;
        std::set<std::pair<int, int>> seenRL;
        for (int pi : tr.idx)
            for (auto& c : pts[static_cast<size_t>(pi)].cands)
                if (seenRL.insert({pts[static_cast<size_t>(pi)].row, c.lid}).second
                    && candDepthOk(pts[static_cast<size_t>(pi)], c)) ++votes[c.lid];
        int winner = -1, winnerVotes = 0;
        for (auto& kv : votes)
            if (kv.second > winnerVotes) { winnerVotes = kv.second; winner = kv.first; }
        if (winner < 0) continue;
        const float share = static_cast<float>(winnerVotes) / static_cast<float>(tr.idx.size());
        const bool pass = share >= VOTE_GATE;
        if (pass) {
            ++gatePass;
            for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = winner;
        } else {
            ++gateFail;
            for (int pi : tr.idx) {
                LPt& pt = pts[static_cast<size_t>(pi)];
                int bestLid = -1;
                float bestD = 1e9f;
                for (auto& c : pt.cands) {
                    const float Z = static_cast<float>(
                        fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
                    const float d = std::fabs(Z - Zmed);
                    if (d < bestD) { bestD = d; bestLid = c.lid; }
                }
                pt.lid = bestLid;
                if (bestLid >= 0) {
                    ++tiebreakPts;
                    if (bestLid == pt.trueLid) ++tiebreakOk;
                }
            }
        }
        lidAgg[winner].first += 1;
        lidAgg[winner].second += static_cast<int>(tr.idx.size());
        for (int pi : tr.idx) {
            const LPt& pt = pts[static_cast<size_t>(pi)];
            if (pt.trueLid > 0) {
                lidTrue[winner].first += 1;
                if (winner == pt.trueLid) lidTrue[winner].second += 1;
                ++lidAccTot;
                if (pt.lid == pt.trueLid) ++lidAccOk;
            }
        }
    }

    // ---------- per-segment ballot summary (all left-camera segments, CSV) ----------
    {
        const fs::path csvPath2 = outDir / "segment_ballots.csv";
        std::ofstream cf2(csvPath2.string());
        cf2 << "segId,pts,rowMin,rowMax,xMin,xMax,winner,winVotes,trueMode,ballots\n";
        int nSeg = 0, nOk = 0;
        long long okPts = 0, totPts = 0;
        for (size_t t = 0; t < tracks.size(); ++t) {
            auto& tr = tracks[t];
            if (tr.idx.size() < MIN_TRACK) continue;
            std::map<int, int> votes;
            std::map<int, int> trueCnt;
            int rMin = 1 << 30, rMax = -(1 << 30);
            float xMin = 1e9f, xMax = -1e9f;
            for (int pi : tr.idx) {
                const LPt& p2 = pts[static_cast<size_t>(pi)];
                rMin = std::min(rMin, p2.row);
                rMax = std::max(rMax, p2.row);
                xMin = std::min(xMin, p2.x);
                xMax = std::max(xMax, p2.x);
                for (auto& c : p2.cands) ++votes[c.lid];
                if (p2.trueLid > 0) ++trueCnt[p2.trueLid];
            }
            std::vector<std::pair<int, int>> vv(votes.begin(), votes.end());
            std::sort(vv.begin(), vv.end(), [](auto& a, auto& b) { return a.second > b.second; });
            int tm = -1, tc = 0;
            for (auto& kv : trueCnt)
                if (kv.second > tc) { tc = kv.second; tm = kv.first; }
            std::string bal;
            for (auto& kv : vv) {
                if (!bal.empty()) bal += " ";
                bal += std::to_string(kv.first) + ":" + std::to_string(kv.second);
            }
            const int winner = vv.empty() ? -1 : vv[0].first;
            const int wv = vv.empty() ? 0 : vv[0].second;
            ++nSeg;
            totPts += static_cast<long long>(tr.idx.size());
            if (winner == tm) { ++nOk; okPts += static_cast<long long>(tr.idx.size()); }
            cf2 << t << "," << tr.idx.size() << "," << rMin << "," << rMax << ","
                << std::fixed << std::setprecision(2) << xMin << "," << xMax << ","
                << winner << "," << wv << "," << tm << ",\"" << bal << "\"\n";
        }
        std::ostringstream osb;
        osb << "segment ballots export -> " << csvPath2.string()
            << " | segs=" << nSeg << " winner==trueMode: " << nOk
            << " segs (" << (nSeg ? 100.0 * nOk / nSeg : 0.0) << "%), "
            << okPts << "/" << totPts << " pts";
        logLine(osb.str());
    }

    // ---------- forensic: plain-vote failure autopsy ----------
    // re-run UNGATED voting; for tracks whose winner != majority-true lid, dump:
    //   winner/true vote counts (tie vs strict loss), phantom HIT right-pt ownership
    {
        std::map<int, std::vector<std::pair<float, int>>> rTrace;   // calib right trace
        for (auto& r : calibPairs)
            rTrace[static_cast<int>(std::lround(r.ly / rowStep))].push_back({r.rx, r.lid});
        for (auto& kv : rTrace)
            std::sort(kv.second.begin(), kv.second.end());
        auto rightOwner = [&](int row, float rx) -> int {
            auto it = rTrace.find(row);
            if (it == rTrace.end()) return -1;
            auto& v = it->second;
            int lo = static_cast<int>(std::lower_bound(v.begin(), v.end(), std::make_pair(rx - 2.0f, 0)) - v.begin());
            int best = -1;
            float bestD = 2.0f;
            for (int k = lo; k < static_cast<int>(v.size()) && v[k].first <= rx + 2.0f; ++k) {
                float d = std::fabs(v[k].first - rx);
                if (d < bestD) { bestD = d; best = v[k].second; }
            }
            return best;
        };

        // right-image adjacent-line spacing (same row, calib rx sorted)
        {
            std::vector<float> gapsR, gapsL;
            for (auto& kv : rTrace) {
                auto& v = kv.second;
                for (size_t k = 1; k < v.size(); ++k)
                    if (v[k].second != v[k - 1].second)
                        gapsR.push_back(v[k].first - v[k - 1].first);
            }
            std::map<int, std::vector<float>> lTrace;
            for (auto& r : calibPairs)
                lTrace[static_cast<int>(std::lround(r.ly / rowStep))].push_back(r.lx);
            for (auto& kv : lTrace) {
                auto& v = kv.second;
                std::sort(v.begin(), v.end());
                for (size_t k = 1; k < v.size(); ++k) gapsL.push_back(v[k] - v[k - 1]);
            }
            std::sort(gapsR.begin(), gapsR.end());
            std::sort(gapsL.begin(), gapsL.end());
            std::ostringstream osr;
            osr << "same-row adjacent-line spacing RIGHT image: n=" << gapsR.size()
                << " min=" << gapsR.front() << " P50=" << gapsR[gapsR.size() / 2]
                << " | LEFT image: min=" << gapsL.front()
                << " P50=" << gapsL[gapsL.size() / 2]
                << " | table min adjacent-hyp uR gap 116.2px / rightP50 = "
                << 116.2 / gapsR[gapsR.size() / 2];
            logLine(osr.str());
        }

        int nFail = 0, nTieLoss = 0, nStrictLoss = 0, nSuccess = 0;
        int nDedupFixes = 0, nDedupStill = 0, failIdx = 0;
        int totWinLine = 0, totWinOther = 0, totTrueLine = 0, totTrueOther = 0;
        // non-line-point forensics (aggregated over FAIL segs)
        int nlClose = 0, nlMid = 0, nlFar = 0;       // dist to true-line interp: <0.3 / <1 / rest
        int nlNoTruth = 0, nlOtherLid = 0;
        int nlRowNoCalib = 0, nlHaveTrueCand = 0, nlTotal = 0;
        // mode-sweep distance buckets for non-line pts
        int swA = 0, swB = 0, swC = 0, swD = 0, swE = 0, swNone = 0;
        std::map<int, int> entryHist, hitHist;   // mode sweep entries within tol / HIT-pass count
        std::map<int, std::vector<std::pair<float, int>>> lTraceF;
        for (auto& r : calibPairs)
            lTraceF[static_cast<int>(std::lround(r.ly / rowStep))].push_back({r.lx, r.lid});
        cv::Mat rectLF;
        {
            cv::Mat mx, my;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        h.imageSize, CV_32FC1, mx, my);
            cv::remap(input->poseFrames[0][0].leftGray, rectLF, mx, my, cv::INTER_LINEAR);
        }
        const fs::path nlDir = outDir / "rows";
        fs::create_directories(nlDir);
        std::ofstream nlf((nlDir / "nonline_pts.txt").string());
        nlf << "non-line pts on FAIL segs: seg / row / lx / dev-to-true-line / trueLid /"
            << " rowHasCalib(mode) / hasModeCand / intensity / cands\n";
        int phantomTot = 0, phantomOnTrace = 0, phantomOnNoise = 0;
        long long successTrueVotes = 0, successPts = 0;
        std::map<int, int> phantomQdiff;      // ownerQ - winnerLid histogram
        std::map<int, int> failPairs;         // winner -> true mode lid
        logLine("forensic plain-vote failure autopsy:");
        for (auto& tr : tracks) {
            if (tr.idx.size() < MIN_TRACK) continue;
                std::map<int, int> votes;
                std::set<std::pair<int, int>> seenRL4;
                for (int pi : tr.idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        if (seenRL4.insert({pts[static_cast<size_t>(pi)].row, c.lid}).second)
                            ++votes[c.lid];
            int winner = -1, winnerVotes = 0;
            for (auto& kv : votes)
                if (kv.second > winnerVotes) { winnerVotes = kv.second; winner = kv.first; }
            std::map<int, int> trueCnt;
            int known = 0;
            for (int pi : tr.idx) {
                const int t = pts[static_cast<size_t>(pi)].trueLid;
                if (t > 0) { ++trueCnt[t]; ++known; }
            }
            int mode = -1, modeCnt = 0;
            for (auto& kv : trueCnt)
                if (kv.second > modeCnt) { modeCnt = kv.second; mode = kv.first; }
            if (winner < 0 || mode < 0) continue;
            if (winner == mode) {
                ++nSuccess;
                successTrueVotes += votes[mode];
                successPts += static_cast<long long>(tr.idx.size());
                continue;
            }
            ++nFail;
            const int trueVotes = votes.count(mode) ? votes[mode] : 0;
            if (winnerVotes > trueVotes) ++nStrictLoss;
            else if (winnerVotes == trueVotes) ++nTieLoss;
            {
                int rMin = 1 << 30, rMax = -(1 << 30);
                std::set<int> rowsWithData;
                float xMin = 1e9f, xMax = -1e9f;
                for (int pi : tr.idx) {
                    const LPt& p2 = pts[static_cast<size_t>(pi)];
                    rMin = std::min(rMin, p2.row);
                    rMax = std::max(rMax, p2.row);
                    rowsWithData.insert(p2.row);
                    xMin = std::min(xMin, p2.x);
                    xMax = std::max(xMax, p2.x);
                }
                std::ostringstream os;
                os << "  FAIL seg#" << failIdx << ": pts=" << tr.idx.size()
                   << " rowSpan=" << (rMax - rMin + 1)
                   << " rowsWithData=" << rowsWithData.size()
                   << " x[" << xMin << "-" << xMax << "]"
                   << " winner=lid" << winner << "(" << winnerVotes << ")"
                   << " trueMode=lid" << mode << "(" << trueVotes << ")"
                   << (winnerVotes == trueVotes ? " TIE" : " OVER");
                logLine(os.str());
                std::vector<std::pair<int, int>> vv(votes.begin(), votes.end());
                std::sort(vv.begin(), vv.end(), [](auto& a, auto& b) { return a.second > b.second; });
                std::ostringstream os2;
                os2 << "    ballots: ";
                for (auto& kv2 : vv)
                    os2 << "lid" << kv2.first << ":" << kv2.second << "  ";
                logLine(os2.str());
                // ballot-source breakdown for winner & true lid: how many pts cast 1/2/3+ ballots
                auto breakdown = [&](int lid) -> std::pair<std::map<int, int>, int> {
                    std::map<int, int> cnt;
                    int total = 0;
                    for (int pi : tr.idx) {
                        int b = 0;
                        for (auto& c : pts[static_cast<size_t>(pi)].cands)
                            if (c.lid == lid) ++b;
                        ++cnt[b];
                        total += b;
                    }
                    return {cnt, total};
                };
                for (int lid : {winner, mode}) {
                    auto bd = breakdown(lid);
                    std::ostringstream os3;
                    os3 << "    lid" << lid << " total=" << bd.second
                        << " | pts casting 0/1/2/3+ ballots:";
                    std::string parts;
                    for (int b = 0; b <= 3; ++b) {
                        int n = bd.first.count(b) ? bd.first[b] : 0;
                        char buf[64];
                        snprintf(buf, sizeof(buf), " %d:%d", b, n);
                        parts += buf;
                    }
                    os3 << parts;
                    logLine(os3.str());
                    int lineP = 0, otherP = 0;
                    for (int pi : tr.idx) {
                        bool sup = false;
                        for (auto& c : pts[static_cast<size_t>(pi)].cands)
                            if (c.lid == lid) { sup = true; break; }
                        if (!sup) continue;
                        if (pts[static_cast<size_t>(pi)].trueLid == mode) ++lineP;
                        else ++otherP;
                    }
                    std::ostringstream os4;
                    os4 << "      supporters: line-pts=" << lineP
                        << " non-line-pts=" << otherP;
                    logLine(os4.str());
                    if (lid == winner) { totWinLine += lineP; totWinOther += otherP; }
                    else { totTrueLine += lineP; totTrueOther += otherP; }
                }
                // per non-line pt: deviation to true-line interpolation + identity + calib row
                {
                    std::map<int, float> lineRowX;
                    for (int pi : tr.idx) {
                        const LPt& p2 = pts[static_cast<size_t>(pi)];
                        if (p2.trueLid == mode) lineRowX[p2.row] = p2.x;
                    }
                    for (int pi : tr.idx) {
                        const LPt& p2 = pts[static_cast<size_t>(pi)];
                        if (p2.trueLid == mode) continue;
                        ++nlTotal;
                        // deviation: interpolate true-line x at p2.row
                        float dev = -1.0f;
                        auto it2 = lineRowX.lower_bound(p2.row);
                        if (!lineRowX.empty()) {
                            if (it2 == lineRowX.end()) dev = std::fabs(p2.x - lineRowX.rbegin()->second);
                            else if (it2 == lineRowX.begin()) dev = std::fabs(p2.x - it2->second);
                            else {
                                auto prev = std::prev(it2);
                                const float t = static_cast<float>(p2.row - prev->first)
                                    / static_cast<float>(it2->first - prev->first);
                                const float xi = prev->second + t * (it2->second - prev->second);
                                dev = std::fabs(p2.x - xi);
                            }
                        }
                        if (dev >= 0 && dev < 0.3f) ++nlClose;
                        else if (dev >= 0 && dev < 1.0f) ++nlMid;
                        else ++nlFar;
                        if (p2.trueLid < 0) ++nlNoTruth; else ++nlOtherLid;
                        bool rowHasCalib = false;
                        {
                            auto lt = lTraceF.find(p2.row);
                            if (lt != lTraceF.end())
                                for (auto& e : lt->second)
                                    if (e.second == mode) { rowHasCalib = true; break; }
                        }
                        if (!rowHasCalib) ++nlRowNoCalib;
                        bool hasMode = false;
                        for (auto& c : p2.cands)
                            if (c.lid == mode) { hasMode = true; break; }
                        if (hasMode) ++nlHaveTrueCand;
                        // mode-sweep nearest-entry distance + entry-level HIT audit
                        {
                            auto& tab2 = getTab(p2.row);
                            auto it3 = tab2.find(mode);
                            if (it3 == tab2.end() || it3->second.empty()) {
                                ++swNone;
                            } else {
                                auto& v3 = it3->second;
                                int lo3 = 0, hi3 = static_cast<int>(v3.size());
                                while (lo3 < hi3) {
                                    int mid3 = (lo3 + hi3) / 2;
                                    if (v3[static_cast<size_t>(mid3)].first < p2.x) lo3 = mid3 + 1;
                                    else hi3 = mid3;
                                }
                                float bd3 = 1e9f;
                                for (int k3 = std::max(0, lo3 - 3);
                                     k3 <= std::min(static_cast<int>(v3.size()) - 1, lo3 + 2); ++k3)
                                    bd3 = std::min(bd3, std::fabs(v3[static_cast<size_t>(k3)].first - p2.x));
                                if (bd3 < 0.7f) ++swA;
                                else if (bd3 < 1.4f) ++swB;
                                else if (bd3 < 5.0f) ++swC;
                                else if (bd3 < 50.0f) ++swD;
                                else ++swE;
                                if (bd3 < 0.7f) {
                                    // enumerate ALL entries within +-0.7 and test each HIT
                                    int nEnt = 0, nHit = 0;
                                    for (int k3 = std::max(0, lo3 - 3);
                                         k3 <= std::min(static_cast<int>(v3.size()) - 1, lo3 + 2); ++k3) {
                                        if (std::fabs(v3[static_cast<size_t>(k3)].first - p2.x) > UL_TOL) continue;
                                        ++nEnt;
                                        const float rx3 = nearestRight(p2.row, v3[static_cast<size_t>(k3)].second, HIT_WIN);
                                        if (rx3 >= 0) ++nHit;
                                    }
                                    ++entryHist[nEnt];
                                    ++hitHist[nHit];
                                }
                            }
                        }
                        const int yy = cvRound(static_cast<float>(p2.row) * rowStep);
                        const int xx = cvRound(p2.x);
                        int inten = -1;
                        if (yy >= 0 && yy < rectLF.rows && xx >= 0 && xx < rectLF.cols)
                            inten = rectLF.at<uchar>(yy, xx);
                        nlf << "seg" << failIdx << " row=" << p2.row << " lx=" << p2.x
                            << " dev=" << dev << " trueLid=" << p2.trueLid
                            << " rowCalib=" << (rowHasCalib ? 1 : 0)
                            << " modeCand=" << (hasMode ? 1 : 0)
                            << " I=" << inten << " cands:";
                        for (auto& c : p2.cands)
                            nlf << " [l" << c.lid << " r" << c.hitRx << "]";
                        nlf << "\n";
                    }
                }
                if (failIdx == 6) {
                    const fs::path rowsDir2 = outDir / "rows";
                    fs::create_directories(rowsDir2);
                    std::ofstream df((rowsDir2 / "fail_seg6_points.txt").string());
                    df << "seg#6 point dump: row / lx (left rect) / trueLid / candidates [lid, predU(right), hitRx(right)]\n";
                    for (int pi : tr.idx) {
                        const LPt& p2 = pts[static_cast<size_t>(pi)];
                        df << "row=" << p2.row << " lx=" << std::fixed << std::setprecision(2)
                           << p2.x << " trueLid=" << p2.trueLid << " cands:";
                        for (auto& c : p2.cands)
                            df << " [l" << c.lid << " u=" << c.predU << " r=" << c.hitRx << "]";
                        df << "\n";
                    }
                    logLine("    seg#6 per-point dump -> data_out/analysis/rows/fail_seg6_points.txt");
                }
                ++failIdx;
            }
            // dedup counterfactual: 1 ballot per lid per point
            std::map<int, std::set<int>> dv;
            for (int pi : tr.idx) {
                std::set<int> lids;
                for (auto& c : pts[static_cast<size_t>(pi)].cands) lids.insert(c.lid);
                for (int l : lids) dv[l].insert(pi);
            }
            int dWinner = -1, dWv = -1, dTrue = 0;
            for (auto& kv : dv) {
                const int cnt = static_cast<int>(kv.second.size());
                if (cnt > dWv) { dWv = cnt; dWinner = kv.first; }
                if (kv.first == mode) dTrue = cnt;
            }
            if (dWinner == mode) ++nDedupFixes;
            else if (nDedupStill <= 5 && dWinner >= 0) {
                std::ostringstream osd;
                osd << "  dedup-still-fails: pts=" << tr.idx.size()
                    << " dedupWinner=lid" << dWinner << "(" << dWv << ")"
                    << " dedupTrue=lid" << mode << "(" << dTrue << ")";
                logLine(osd.str());
                ++nDedupStill;
            }
            failPairs[winner] = mode;
            for (int pi : tr.idx) {
                const LPt& pt = pts[static_cast<size_t>(pi)];
                for (auto& c : pt.cands) {
                    if (c.lid != winner || c.lid == pt.trueLid) continue;
                    ++phantomTot;
                    const int q = rightOwner(pt.row, c.hitRx);
                    if (q > 0) { ++phantomOnTrace; ++phantomQdiff[q - winner]; }
                    else ++phantomOnNoise;
                }
            }
        }
        std::ostringstream os;
        os << "forensic: success=" << nSuccess << " (avg true-vote share "
           << (successPts ? 100.0 * successTrueVotes / successPts : 0.0) << "%)"
           << " | fail=" << nFail << " [tie-loss(small-lid wins)=" << nTieLoss
           << " strict-loss=" << nStrictLoss << "]"
            << " | dedup counterfactual: fixes=" << nDedupFixes
            << " still-fails=" << (nFail - nDedupFixes)
            << " | supporter heads on FAIL segs: PHANTOM(win)=" << (totWinLine + totWinOther)
            << " [line " << totWinLine << " + nonline " << totWinOther << "]"
            << " vs TRUE=" << (totTrueLine + totTrueOther)
            << " [line " << totTrueLine << " + nonline " << totTrueOther << "]";
        logLine(os.str());
        std::ostringstream osn;
        osn << "non-line pt forensics: total=" << nlTotal
            << " | dev-to-true-line: <0.3px=" << nlClose << " 0.3-1px=" << nlMid
            << " >1px=" << nlFar
            << " | trueLid: none=" << nlNoTruth << " other-line=" << nlOtherLid
            << " | row-missing-calib(mode)=" << nlRowNoCalib
            << " | has-mode-cand=" << nlHaveTrueCand;
        logLine(osn.str());
        std::ostringstream osn2;
        osn2 << "mode-sweep dist for non-line pts: <0.7=" << swA << " 0.7-1.4=" << swB
             << " 1.4-5=" << swC << " 5-50=" << swD << " >50=" << swE
             << " no-entry=" << swNone;
        logLine(osn2.str());
        std::ostringstream osn3;
        osn3 << "entry audit (sweep<0.7 pts): entries-in-tol per pt:";
        for (auto& kv : entryHist) osn3 << " " << kv.first << "ent:" << kv.second;
        osn3 << " | HIT-pass per pt:";
        for (auto& kv : hitHist) osn3 << " " << kv.first << "hit:" << kv.second;
        logLine(osn3.str());
        std::ostringstream os2;
        os2 << "phantom winner-HITs: total=" << phantomTot << " on-other-line-trace="
            << phantomOnTrace << " on-noise=" << phantomOnNoise;
        logLine(os2.str());
        logLine("phantom right-pt owner (ownerLid - winnerLid -> count):");
        for (auto& kv : phantomQdiff) {
            std::ostringstream os3;
            os3 << "  diff=" << kv.first << " count=" << kv.second;
            logLine(os3.str());
        }
        logLine("fail winner->trueMode pairs (top):");
        std::vector<std::pair<int, int>> fv(failPairs.begin(), failPairs.end());
        std::sort(fv.begin(), fv.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < fv.size() && i < 10; ++i) {
            std::ostringstream os4;
            os4 << "  winner=lid" << fv[i].first << " trueMode=lid" << fv[i].second;
            logLine(os4.str());
        }
    }

    // ---------- visual vote audit: annotate per-point ballots on rectified images ----------
    {
        const auto& img0 = input->poseFrames[0][0];
        cv::Mat mapXL, mapYL, mapXR, mapYR, recL, recR;
        cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                    h.imageSize, CV_32FC1, mapXL, mapYL);
        cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                    h.imageSize, CV_32FC1, mapXR, mapYR);
        cv::remap(img0.leftGray, recL, mapXL, mapYL, cv::INTER_LINEAR);
        cv::remap(img0.rightGray, recR, mapXR, mapYR, cv::INTER_LINEAR);
        cv::cvtColor(recL, recL, cv::COLOR_GRAY2BGR);
        cv::cvtColor(recR, recR, cv::COLOR_GRAY2BGR);

        // right pts by row (for annotation + claim counting)
        std::map<int, std::vector<float>> srX;   // row -> sorted x (reuse rByRow)
        (void)srX;
        std::map<std::pair<int, long long>, int> claims;   // (row, round(rx)) -> ballots hitting it
        for (auto& pt : pts)
            for (auto& c : pt.cands)
                ++claims[{pt.row, std::lround(c.hitRx)}];

        // ballot-count histogram + same-lid multi-ballot evidence
        int n1 = 0, n2 = 0, n3 = 0, n4 = 0, multiSameLid = 0;
        const LPt* ex[3] = {nullptr, nullptr, nullptr};
        for (auto& pt : pts) {
            const int m = static_cast<int>(pt.cands.size());
            if (m <= 1) ++n1; else if (m == 2) ++n2; else if (m == 3) ++n3; else ++n4;
            std::map<int, int> per;
            for (auto& c : pt.cands) ++per[c.lid];
            for (auto& kv : per)
                if (kv.second >= 2) {
                    ++multiSameLid;
                    for (auto& e : ex) if (!e) { e = &pt; break; }
                    break;
                }
        }
        std::ostringstream osb;
        osb << "ballots per point hist: 1=" << n1 << " 2=" << n2 << " 3=" << n3
            << " 4+=" << n4 << " | pts casting SAME-lid multiple ballots=" << multiSameLid;
        logLine(osb.str());
        for (auto* e : ex) {
            if (!e) break;
            std::ostringstream os2;
            os2 << "  multi-ballot example @row=" << e->row << " x=" << e->x
                << " trueLid=" << e->trueLid << " cands:";
            for (auto& c : e->cands)
                os2 << " [lid" << c.lid << " predU=" << c.predU << " hitRx=" << c.hitRx << "]";
            logLine(os2.str());
        }

        // pick 5 rows spaced >=320, 6..20 left pts, right pts present
        std::vector<int> pool;
        for (auto& kv : lByRow)
            if (kv.second.size() >= 6 && kv.second.size() <= 20 && rByRow.count(kv.first))
                pool.push_back(kv.first);
        std::mt19937 rng(20260831);
        std::shuffle(pool.begin(), pool.end(), rng);
        std::vector<int> rows;
        for (int r : pool) {
            bool ok = true;
            for (int t : rows)
                if (std::abs(t - r) < 320) { ok = false; break; }
            if (ok) rows.push_back(r);
            if (rows.size() >= 5) break;
        }
        std::sort(rows.begin(), rows.end());

        const fs::path rowsDir = outDir / "rows";
        fs::create_directories(rowsDir);
        std::ofstream vtxt((rowsDir / "vote_audit.txt").string());
        const int W = h.imageSize.width, H = h.imageSize.height;

        for (int row : rows) {
            const int y = static_cast<int>(row * rowStep);
            cv::line(recL, {0, y}, {W, y}, cv::Scalar(0, 255, 255), 1);
            cv::line(recR, {0, y}, {W, y}, cv::Scalar(0, 255, 255), 1);
            cv::putText(recL, "row " + std::to_string(row), {8, y - 5}, 0, 0.55,
                        cv::Scalar(0, 255, 255), 1);
            vtxt << "\n==== row " << row << " ====\n";
            for (int pi : lByRow[row]) {
                const LPt& pt = pts[static_cast<size_t>(pi)];
                const int x = static_cast<int>(pt.x);
                cv::circle(recL, {x, y}, 4, cv::Scalar(0, 0, 255), -1);
                // per-lid ballot stack above point
                std::map<int, int> per;
                for (auto& c : pt.cands) ++per[c.lid];
                vtxt << "  L x=" << pt.x << " trueLid=" << pt.trueLid << " ballots=";
                int sy = y - 10;
                for (auto it2 = per.rbegin(); it2 != per.rend(); ++it2, sy -= 14) {
                    const bool isTrue = (it2->first == pt.trueLid);
                    const std::string s = (isTrue ? "[" : "") + std::to_string(it2->first)
                                        + (isTrue ? "]" : "") + "x" + std::to_string(it2->second);
                    cv::putText(recL, s, {x - 12, sy}, 0, 0.4,
                                isTrue ? cv::Scalar(255, 80, 80) : cv::Scalar(0, 140, 255), 1);
                    vtxt << " lid" << it2->first << "x" << it2->second;
                }
                vtxt << " | cands:";
                for (auto& c : pt.cands) {
                    vtxt << " (l" << c.lid << ",u" << c.predU << ",r" << c.hitRx << ")";
                    const bool isTrue = (c.lid == pt.trueLid);
                    cv::line(recL, {x, y}, {static_cast<int>(c.hitRx), y},
                             isTrue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 130, 255),
                             isTrue ? 2 : 1);
                }
                vtxt << "\n";
            }
            // right pts: claims count
            for (float rx : rByRow[row]) {
                const int xr = static_cast<int>(rx);
                const int n = claims.count({row, std::lround(rx)}) ? claims[{row, std::lround(rx)}] : 0;
                cv::circle(recR, {xr, y}, 4, cv::Scalar(0, 140, 0), -1);
                cv::putText(recR, std::to_string(n), {xr - 4, y + 22}, 0, 0.4,
                            cv::Scalar(0, 220, 0), 1);
            }
        }
        cv::Mat canvas(H, W * 2 + 10, recL.type(), cv::Scalar(0, 0, 0));
        recL.copyTo(canvas(cv::Rect(0, 0, W, H)));
        recR.copyTo(canvas(cv::Rect(W + 10, 0, W, H)));
        cv::putText(canvas, "LEFT: red pt + ballot stack 'lid x N' ([]=true lid), blue=true ballot, orange=phantom ballot",
                    {10, 22}, 0, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(canvas, "RIGHT: green pt + claim count", {W + 20, 22}, 0, 0.6,
                    cv::Scalar(255, 255, 255), 2);
        cv::imwrite((rowsDir / "vote_audit.png").string(), canvas);
        std::cout << "vote audit image -> " << (rowsDir / "vote_audit.png").string() << "\n";
    }

    // ---------- visual: connected components (1D-tracked point sets), one color each ----------
    {
        const int W = h.imageSize.width, H = h.imageSize.height;
        cv::Mat imL(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat imR(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
        for (auto& p : sR) {
            const int x = cvRound(p.x), y = cvRound(p.y);
            if (x >= 0 && x < W && y >= 0 && y < H)
                cv::circle(imR, {x, y}, 1, cv::Scalar(150, 150, 150), -1);
        }
        // per-track color: golden-angle hue for kept tracks, dark gray for fragments
        std::vector<cv::Vec3b> trackColor(tracks.size(), cv::Vec3b(45, 45, 45));
        for (size_t t = 0; t < tracks.size(); ++t) {
            if (tracks[t].idx.size() < MIN_TRACK) continue;
            cv::Mat m(1, 1, CV_8UC3, cv::Scalar((t * 47) % 180, 255, 255));
            cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
            trackColor[t] = m.at<cv::Vec3b>(0, 0);
        }
        for (auto& pt : pts) {
            const int x = cvRound(pt.x), y = cvRound(static_cast<float>(pt.row) * rowStep);
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            const cv::Vec3b& c = pt.trackId >= 0
                ? trackColor[static_cast<size_t>(pt.trackId)] : cv::Vec3b(45, 45, 45);
            cv::circle(imL, {x, y}, 1, cv::Scalar(c[0], c[1], c[2]), -1);
        }
        std::map<std::string, int> hist;
        auto bucket = [](size_t n) -> std::string {
            if (n == 1) return "1";
            if (n == 2) return "2";
            if (n == 3) return "3";
            if (n == 4) return "4";
            if (n <= 10) return "5-10";
            if (n <= 50) return "11-50";
            if (n <= 200) return "51-200";
            return ">200";
        };
        for (auto& tr : tracks) ++hist[bucket(tr.idx.size())];
        size_t nFrag = 0;
        for (auto& tr : tracks) if (tr.idx.size() < MIN_TRACK) ++nFrag;
        std::ostringstream os;
        os << "connected components: total=" << tracks.size()
           << " kept(>=" << MIN_TRACK << ")=" << (tracks.size() - nFrag)
           << " fragment(<" << MIN_TRACK << ")=" << nFrag;
        logLine(os.str());
        std::ostringstream os2;
        os2 << "component size hist:";
        for (auto& kv : hist) os2 << " " << kv.first << ":" << kv.second;
        logLine(os2.str());

        cv::Mat canvas(H, W * 2 + 10, imL.type(), cv::Scalar(0, 0, 0));
        imL.copyTo(canvas(cv::Rect(0, 0, W, H)));
        imR.copyTo(canvas(cv::Rect(W + 10, 0, W, H)));
        cv::putText(canvas, "LEFT: connected point sets (each color = one component, dark gray = fragments <5 pts)",
                    {10, 22}, 0, 0.55, cv::Scalar(255, 255, 255), 2);
        cv::putText(canvas, "RIGHT: raw interp points (gray)", {W + 20, 22}, 0, 0.55,
                    cv::Scalar(255, 255, 255), 2);
        cv::imwrite((outDir / "pose_06_tracks.png").string(), canvas);
        std::cout << "tracks image -> " << (outDir / "pose_06_tracks.png").string() << "\n";
    }

    // ---------- step 5: match by voted lid + reconstruct ----------
    struct PairRec { int ptIdx; float lx, rx; };
    std::vector<PairRec> pairs;
    int lostNoTrack = 0, lostNoLid = 0, lostNoSweep = 0, lostNoRight = 0;
    auto matchAll = [&](bool verbose) {
        pairs.clear();
        if (verbose) { lostNoTrack = lostNoLid = lostNoSweep = lostNoRight = 0; }
        for (size_t i = 0; i < pts.size(); ++i) {
            LPt& pt = pts[i];
            const bool known = pt.trueLid > 0;
            if (known && (pt.trackId < 0 ||
                          tracks[static_cast<size_t>(pt.trackId)].idx.size() < MIN_TRACK)) {
                ++lostNoTrack;
                continue;
            }
            if (pt.lid < 0) { if (known) ++lostNoLid; continue; }
            auto& tab = getTab(pt.row);
            auto it = tab.find(pt.lid);
            if (it == tab.end() || it->second.empty()) { if (known) ++lostNoSweep; continue; }
            auto& v = it->second;
            int lo = 0, hi = static_cast<int>(v.size());
            while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < pt.x) lo = mid + 1; else hi = mid; }
            float bestDx = 1e9f, predU = -1;
            const int down = static_cast<int>(std::ceil(UL_TOL / rowStep));
            for (int k = std::max(0, lo - down);
                 k <= std::min(static_cast<int>(v.size()) - 1, lo + down - 1); ++k) {
                float dx = std::fabs(v[k].first - pt.x);
                if (dx < bestDx) { bestDx = dx; predU = v[k].second; }
            }
            if (bestDx > UL_TOL || predU < 0) { if (known) ++lostNoSweep; continue; }
            const float rx = nearestRight(pt.row, predU, HIT_WIN);
            if (rx < 0) { if (known) ++lostNoRight; continue; }
            pairs.push_back({static_cast<int>(i), pt.x, rx});
        }
        if (verbose) {
            std::ostringstream os;
            os << "lost breakdown (truth-known pts): small-track=" << lostNoTrack
               << " no-lid=" << lostNoLid << " no-sweep=" << lostNoSweep
               << " no-right=" << lostNoRight;
            logLine(os.str());
        }
    };
    matchAll(true);

    auto statsOf = [&](const char* tag, const std::vector<PairRec>& pv0,
                       bool postDepthFilter) {
        std::vector<PairRec> pv = pv0;
        if (postDepthFilter) {
            std::vector<float> zs, zs2;
            zs.reserve(pv.size());
            for (auto& pr : pv)
                zs.push_back(static_cast<float>(
                    fB / std::max(1e-3, static_cast<double>(pr.lx) - pr.rx)));
            zs2 = zs;
            std::sort(zs2.begin(), zs2.end());
            const float med = zs2[zs2.size() / 2];
            std::vector<PairRec> keep;
            keep.reserve(pv.size());
            for (size_t i = 0; i < pv.size(); ++i)
                if (std::fabs(zs[i] - med) <= 30.0f) keep.push_back(pv[i]);
            pv = keep;
        }
        int matched = static_cast<int>(pv.size()), correct = 0;
        for (auto& pr : pv)
            if (isTruePair(pts[static_cast<size_t>(pr.ptIdx)].row, pr.lx, pr.rx)) ++correct;
        std::ostringstream os;
        os << tag << ": matched=" << matched << " correct=" << correct
           << " wrong=" << (matched - correct)
           << " lost=" << (calibTotal - correct);
        logLine(os.str());
    };

    statsOf("TV2 vote-depth-gated (raw)    ", pairs, false);
    statsOf("TV2 vote-depth-gated + filter ", pairs, true);
    logLine("TV0 plain-vote (run-2 hist)   : matched=19544 correct=15276 wrong=4268 lost=4496");
    logLine("GPU reference                  : matched=13244 correct=10680 wrong=2564 lost=9092");
    logLine("V8d  CSP+depth+post-filter     : matched=18550 correct=18550 wrong=0 lost=1222");

    // PLY
    {
        const int n = static_cast<int>(pairs.size());
        if (n > 0) {
            cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
            std::vector<char> ok(static_cast<size_t>(n), 0);
            for (int k = 0; k < n; ++k) {
                const PairRec& pr = pairs[static_cast<size_t>(k)];
                const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                const float y = static_cast<float>(pt.row) * rowStep;
                L.at<cv::Vec2f>(k) = cv::Vec2f(pr.lx, y);
                R.at<cv::Vec2f>(k) = cv::Vec2f(pr.rx, y);
                I.at<int>(k) = pt.lid;
                ok[static_cast<size_t>(k)] =
                    isTruePair(pt.row, pr.lx, pr.rx) ? 1 : 0;
            }
            cv::cuda::GpuMat dL, dR, dI;
            dL.upload(L);
            dR.upload(R);
            dI.upload(I);
            auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
            if (rr.success && rr.d_points3d) {
                cv::Mat P3;
                rr.d_points3d->download(P3);
                stream.waitForCompletion();
                P3 = P3.reshape(3, 1);
                std::vector<cv::Vec3f> cloud(static_cast<size_t>(n));
                for (int k = 0; k < n; ++k) cloud[static_cast<size_t>(k)] = P3.at<cv::Vec3f>(k);
                writePly((outDir / "pose_06_trackvote.ply").string(), cloud, ok);
                {   // 步骤产物：扫描链（投票后）三维点云
                    const fs::path stepsDir = outDir / "steps";
                    fs::create_directories(stepsDir);
                    writePly((stepsDir / "06_recon_scan_voted.ply").string(), cloud, ok);
                }
                int nOk = 0;
                for (char c : ok) nOk += c;
                std::ostringstream os;
                os << "cloud pose_06_trackvote.ply: " << n << " pts (green=" << nOk
                   << " red=" << (n - nOk) << ")";
                logLine(os.str());
            }
        }
    }

    // ---------- 扫描匹配左右图标注（编号=投票线号，红=错配）＋ 与标定链对比 ----------
    {
        const int W = h.imageSize.width, H = h.imageSize.height;
        cv::Mat imL(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat imR(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
        auto lidColor = [](int lid) -> cv::Vec3b {
            cv::Mat m(1, 1, CV_8UC3, cv::Scalar((lid * 47) % 180, 255, 255));
            cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
            return m.at<cv::Vec3b>(0, 0);
        };
        // lid -> (sxL, sy, sxR, n)：正确匹配才计入质心，避免错配点拉偏标号位置
        std::map<int, std::array<double, 4>> cen;
        int matched = 0, correct = 0;
        for (auto& pr : pairs) {
            const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
            const int y = cvRound(static_cast<float>(pt.row) * rowStep);
            const bool good = isTruePair(pt.row, pr.lx, pr.rx);
            ++matched;
            if (good) ++correct;
            const cv::Vec3b c = good ? lidColor(pt.lid) : cv::Vec3b(0, 0, 255);
            cv::circle(imL, {cvRound(pr.lx), y}, 1, cv::Scalar(c[0], c[1], c[2]), -1);
            cv::circle(imR, {cvRound(pr.rx), y}, 1, cv::Scalar(c[0], c[1], c[2]), -1);
            if (good) {
                auto& a = cen[pt.lid];
                a[0] += pr.lx;
                a[1] += y;
                a[2] += pr.rx;
                a[3] += 1;
            }
        }
        for (auto& kv : cen) {
            const auto& a = kv.second;
            const std::string s = std::to_string(kv.first);
            cv::putText(imL, s, {cvRound(a[0] / a[3]) - 8, cvRound(a[1] / a[3]) - 6},
                        0, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(imR, s, {cvRound(a[2] / a[3]) - 8, cvRound(a[1] / a[3]) - 6},
                        0, 0.8, cv::Scalar(255, 255, 255), 2);
        }
        cv::putText(imL, "LEFT: scan matched (color/number = voted lid, red = wrong)",
                    {8, 20}, 0, 0.55, cv::Scalar(200, 200, 200), 1);
        cv::putText(imR, "RIGHT: matched pts (number = voted lid, red = wrong)",
                    {8, 20}, 0, 0.55, cv::Scalar(200, 200, 200), 1);
        const fs::path stepsDir = outDir / "steps";
        fs::create_directories(stepsDir);
        cv::imwrite((stepsDir / "07_match_labeled_L.png").string(), imL);
        cv::imwrite((stepsDir / "07_match_labeled_R.png").string(), imR);
        logLine("labeled match images -> steps/07_match_labeled_{L,R}.png");

        std::ostringstream os2;
        os2 << "SCAN vs CALIB comparison (voted-lid matching, TV2-gated raw):\n"
            << "  calib chain truth pairs : " << calibTotal << "\n"
            << "  scan  matched=" << matched << " correct=" << correct
            << " wrong=" << (matched - correct) << " lost=" << (calibTotal - correct) << "\n"
            << "  accuracy(correct/matched) = " << (100.0 * correct / std::max(1, matched))
            << "%  coverage(correct/truth) = " << (100.0 * correct / std::max(1, calibTotal)) << "%";
        logLine(os2.str());
    }

    // ---------- right-side consistency audit (parameterized: gate on/off) ----------
    std::map<int, std::vector<std::array<float, 3>>> revCache;   // row -> (uR,uL,lid) by uR
    auto getRev = [&](int row) -> std::vector<std::array<float, 3>>& {
            auto it = revCache.find(row);
            if (it != revCache.end()) return it->second;
            std::vector<CurveMapEntry> tmp;
            int cap = 4096;
            tmp.resize(cap);
            int n = cmt.EnumerateRow(row, tmp.data(), cap);
            while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
            std::vector<std::array<float, 3>> es;
            es.reserve(static_cast<size_t>(n));
            for (int q = 0; q < n; ++q)
                es.push_back({tmp[static_cast<size_t>(q)].uR * rowStep,
                              tmp[static_cast<size_t>(q)].uL * rowStep,
                              static_cast<float>(tmp[static_cast<size_t>(q)].lid)});
            std::sort(es.begin(), es.end(),
                      [](const std::array<float, 3>& a, const std::array<float, 3>& b) { return a[0] < b[0]; });
            revCache[row] = std::move(es);
            return revCache[row];
        };
        auto nearestLeftX = [&](int row, float x, float win) -> float {
            auto it = lByRow.find(row);
            if (it == lByRow.end()) return -1.0f;
            float best = -1.0f, bestD = win + 1.0f;
            for (int pi : it->second) {
                const float d = std::fabs(pts[static_cast<size_t>(pi)].x - x);
                if (d < bestD) { bestD = d; best = pts[static_cast<size_t>(pi)].x; }
            }
            return bestD <= win ? best : -1.0f;
        };
        auto rightAudit = [&](bool useGate, const char* tag) {
            std::map<int, std::vector<const PairRec*>> pairsByTrack;
            for (auto& pr : pairs) pairsByTrack[pts[static_cast<size_t>(pr.ptIdx)].trackId].push_back(&pr);
            int segAgree = 0, segDis = 0, segNoMode = 0;
            long long ptAgree = 0, ptTot = 0;
            int agrCorrect = 0, agrWrong = 0, disCorrect = 0, disWrong = 0;
            std::map<std::pair<int, int>, int> disList;
            for (size_t ti = 0; ti < tracks.size(); ++ti) {
                if (tracks[ti].idx.size() < MIN_TRACK) continue;
                std::map<int, int> lmode;
                for (int pi : tracks[ti].idx) {
                    const int l = pts[static_cast<size_t>(pi)].lid;
                    if (l >= 0) ++lmode[l];
                }
                int leftMode = -1, mc = 0;
                for (auto& kv : lmode)
                    if (kv.second > mc) { mc = kv.second; leftMode = kv.first; }
                auto pit = pairsByTrack.find(static_cast<int>(ti));
                if (pit == pairsByTrack.end() || leftMode < 0) continue;
                std::map<int, int> rvotes;
                for (auto* pr : pit->second) {
                    const int row = pts[static_cast<size_t>(pr->ptIdx)].row;
                    auto& es = getRev(row);
                    std::map<int, float> perLid;
                    int lo = 0, hi = static_cast<int>(es.size());
                    while (lo < hi) {
                        int mid = (lo + hi) / 2;
                        if (es[static_cast<size_t>(mid)][0] < pr->rx - 0.7f) lo = mid + 1; else hi = mid;
                    }
                    for (int k = lo; k < static_cast<int>(es.size())
                                    && es[static_cast<size_t>(k)][0] <= pr->rx + 0.7f; ++k) {
                        const int lid = static_cast<int>(es[static_cast<size_t>(k)][2]);
                        if (!perLid.count(lid))
                            perLid[lid] = es[static_cast<size_t>(k)][1];
                    }
                    std::map<int, int> ptVotes;
                    for (auto& kv : perLid) {
                        const float pred_uL = kv.second;
                        if (nearestLeftX(row, pred_uL, 2.0f) < 0) continue;
                        if (useGate) {
                            const float Z = static_cast<float>(
                                fB / std::max(1e-3, static_cast<double>(pred_uL) - pr->rx));
                            if (std::fabs(Z - Zmed) > DEPTH_WIN) continue;
                        }
                        ptVotes[kv.first] = 1;
                    }
                    if (!ptVotes.empty()) {
                        ++ptTot;
                        if (ptVotes.count(leftMode)) ++ptAgree;
                        for (auto& kv : ptVotes) ++rvotes[kv.first];
                    }
                }
                int rightMode = -1, rc = 0;
                for (auto& kv : rvotes)
                    if (kv.second > rc) { rc = kv.second; rightMode = kv.first; }
                if (rightMode < 0) { ++segNoMode; continue; }
                const bool agree = (rightMode == leftMode);
                if (agree) ++segAgree; else ++segDis;
                for (auto* pr : pit->second) {
                    const bool ok = isTruePair(pts[static_cast<size_t>(pr->ptIdx)].row, pr->lx, pr->rx);
                    if (agree) { ok ? ++agrCorrect : ++agrWrong; }
                    else { ok ? ++disCorrect : ++disWrong; ++disList[{leftMode, rightMode}]; }
                }
            }
            std::ostringstream osr;
            osr << "right-side consistency [" << tag << "]: segments agree=" << segAgree
                << " disagree=" << segDis << " no-right-mode=" << segNoMode
                << " | point-level agree=" << ptAgree << "/" << ptTot
                << " (" << (ptTot ? 100.0 * ptAgree / ptTot : 0.0) << "%)";
            logLine(osr.str());
            std::ostringstream osr2;
            osr2 << "  agree-seg pairs: correct=" << agrCorrect << " wrong=" << agrWrong
                 << " | disagree-seg pairs: correct=" << disCorrect << " wrong=" << disWrong;
            logLine(osr2.str());
            if (!disList.empty()) {
                logLine("  disagree (leftMode -> rightMode): count");
                std::vector<std::pair<std::pair<int, int>, int>> dv(disList.begin(), disList.end());
                std::sort(dv.begin(), dv.end(), [](auto& a, auto& b) { return a.second > b.second; });
                for (size_t i = 0; i < dv.size() && i < 10; ++i) {
                    std::ostringstream osr3;
                    osr3 << "    lid" << dv[i].first.first << " -> lid" << dv[i].first.second
                         << " : " << dv[i].second;
                    logLine(osr3.str());
                }
            }
        };
    rightAudit(true, "TV2 lids, gated reverse votes");
    rightAudit(false, "TV2 lids, UNGATED reverse votes");

    // forward-label audit: right lids inherited from left pts' cands (claim-based)
    auto forwardAudit = [&](bool useGate, const char* tag) {
        std::map<std::pair<int, long long>, std::map<int, int>> labels;   // (row, rx) -> lid -> claims
        for (auto& pt : pts) {
            for (auto& c : pt.cands) {
                if (useGate) {
                    const float Z = static_cast<float>(
                        fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
                    if (std::fabs(Z - Zmed) > DEPTH_WIN) continue;
                }
                ++labels[{pt.row, std::lround(c.hitRx)}][c.lid];
            }
        }
        std::map<int, std::vector<const PairRec*>> pairsByTrack;
        for (auto& pr : pairs) pairsByTrack[pts[static_cast<size_t>(pr.ptIdx)].trackId].push_back(&pr);
        int segAgree = 0, segDis = 0, segNoLabel = 0;
        long long ptAgree = 0, ptTot = 0;
        int agrCorrect = 0, agrWrong = 0, disCorrect = 0, disWrong = 0;
        std::map<std::pair<int, int>, int> disList;
        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            if (tracks[ti].idx.size() < MIN_TRACK) continue;
            std::map<int, int> lmode;
            for (int pi : tracks[ti].idx) {
                const int l = pts[static_cast<size_t>(pi)].lid;
                if (l >= 0) ++lmode[l];
            }
            int leftMode = -1, mc = 0;
            for (auto& kv : lmode)
                if (kv.second > mc) { mc = kv.second; leftMode = kv.first; }
            auto pit = pairsByTrack.find(static_cast<int>(ti));
            if (pit == pairsByTrack.end() || leftMode < 0) continue;
            std::map<int, int> rvotes;
            for (auto* pr : pit->second) {
                const int row = pts[static_cast<size_t>(pr->ptIdx)].row;
                auto lit = labels.find({row, std::lround(pr->rx)});
                if (lit == labels.end() || lit->second.empty()) continue;
                ++ptTot;
                if (lit->second.count(leftMode)) ++ptAgree;
                for (auto& kv : lit->second) rvotes[kv.first] += kv.second;
            }
            int rightMode = -1, rc = 0;
            for (auto& kv : rvotes)
                if (kv.second > rc) { rc = kv.second; rightMode = kv.first; }
            if (rightMode < 0) { ++segNoLabel; continue; }
            const bool agree = (rightMode == leftMode);
            if (agree) ++segAgree; else ++segDis;
            for (auto* pr : pit->second) {
                const bool ok = isTruePair(pts[static_cast<size_t>(pr->ptIdx)].row, pr->lx, pr->rx);
                if (agree) { ok ? ++agrCorrect : ++agrWrong; }
                else { ok ? ++disCorrect : ++disWrong; ++disList[{leftMode, rightMode}]; }
            }
        }
        std::ostringstream osf;
        osf << "forward-label consistency [" << tag << "]: segments agree=" << segAgree
            << " disagree=" << segDis << " no-label=" << segNoLabel
            << " | point-level agree=" << ptAgree << "/" << ptTot
            << " (" << (ptTot ? 100.0 * ptAgree / ptTot : 0.0) << "%)";
        logLine(osf.str());
        std::ostringstream osf2;
        osf2 << "  agree-seg pairs: correct=" << agrCorrect << " wrong=" << agrWrong
             << " | disagree-seg pairs: correct=" << disCorrect << " wrong=" << disWrong;
        logLine(osf2.str());
        if (!disList.empty()) {
            logLine("  disagree (leftMode -> rightMode): count");
            std::vector<std::pair<std::pair<int, int>, int>> dv(disList.begin(), disList.end());
            std::sort(dv.begin(), dv.end(), [](auto& a, auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < dv.size() && i < 10; ++i) {
                std::ostringstream osf3;
                osf3 << "    lid" << dv[i].first.first << " -> lid" << dv[i].first.second
                     << " : " << dv[i].second;
                logLine(osf3.str());
            }
        }
    };
    forwardAudit(true, "TV2 lids, gated claims");
    forwardAudit(false, "TV2 lids, UNGATED claims");

    // ---------- TV3 experiment: per-lid dedup voting (1 ballot per lid per point) ----------
    // tests whether the multi-ballot rule (same lid counted once per uR-bin) was the
    // decisive failure cause of plain voting
    {
        for (auto& pt : pts) pt.lid = -1;
        for (auto& tr : tracks) {
            if (tr.idx.size() < MIN_TRACK) continue;
            std::map<int, int> votes;
            for (int pi : tr.idx) {
                std::set<int> lids;
                for (auto& c : pts[static_cast<size_t>(pi)].cands) lids.insert(c.lid);
                for (int l : lids) ++votes[l];
            }
            std::map<int, double> dev;
            for (auto& kv : votes) {
                double s = 0;
                int n = 0;
                for (int pi : tr.idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        if (c.lid == kv.first) {
                            s += std::fabs(fB / std::max(1e-3, static_cast<double>(pts[static_cast<size_t>(pi)].x) - c.hitRx) - Zmed);
                            ++n;
                        }
                dev[kv.first] = n ? s / n : 1e18;
            }
            int winner = -1, wv = 0;
            double bestDev = 1e18;
            for (auto& kv : votes)
                if (kv.second > wv || (kv.second == wv && dev[kv.first] < bestDev)) {
                    wv = kv.second;
                    winner = kv.first;
                    bestDev = dev[kv.first];
                }
            const float share = static_cast<float>(wv) / static_cast<float>(tr.idx.size());
            if (share >= VOTE_GATE) {
                for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = winner;
            } else {
                for (int pi : tr.idx) {
                    LPt& pt = pts[static_cast<size_t>(pi)];
                    int bl = -1;
                    float bd = 1e9f;
                    for (auto& c : pt.cands) {
                        const float Z = static_cast<float>(
                            fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
                        if (std::fabs(Z - Zmed) < bd) { bd = std::fabs(Z - Zmed); bl = c.lid; }
                    }
                    pt.lid = bl;
                }
            }
        }
        matchAll(false);
        statsOf("TV3 per-lid-dedup vote (raw)  ", pairs, false);
        statsOf("TV3 per-lid-dedup + filter    ", pairs, true);
    }

    // ---------- TV4 experiment: plain vote with depth gate DISABLED ----------
    // isolates the gate's contribution under the new long-gap tracking
    {
        for (auto& pt : pts) pt.lid = -1;
        for (auto& tr : tracks) {
            if (tr.idx.size() < MIN_TRACK) continue;
                std::map<int, int> votes;
                std::set<std::pair<int, int>> seenRL4;
                for (int pi : tr.idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        if (seenRL4.insert({pts[static_cast<size_t>(pi)].row, c.lid}).second)
                            ++votes[c.lid];
            int winner = -1, wv = 0;
            for (auto& kv : votes)
                if (kv.second > wv) { wv = kv.second; winner = kv.first; }
            const float share = static_cast<float>(wv) / static_cast<float>(tr.idx.size());
            if (winner >= 0 && share >= VOTE_GATE) {
                for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = winner;
            } else {
                for (int pi : tr.idx) {
                    LPt& pt = pts[static_cast<size_t>(pi)];
                    int bl = -1;
                    float bd = 1e9f;
                    for (auto& c : pt.cands) {
                        const float Z = static_cast<float>(
                            fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
                        if (std::fabs(Z - Zmed) < bd) { bd = std::fabs(Z - Zmed); bl = c.lid; }
                    }
                    pt.lid = bl;
                }
            }
        }
        matchAll(false);
        statsOf("TV4 plain vote NO gate (raw)  ", pairs, false);
        statsOf("TV4 plain vote NO gate +filter", pairs, true);
        // PLY for the no-gate variant (green=correct red=wrong)
        {
            const int n = static_cast<int>(pairs.size());
            if (n > 0) {
                cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
                std::vector<char> ok(static_cast<size_t>(n), 0);
                for (int k = 0; k < n; ++k) {
                    const PairRec& pr = pairs[static_cast<size_t>(k)];
                    const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                    const float y = static_cast<float>(pt.row) * rowStep;
                    L.at<cv::Vec2f>(k) = cv::Vec2f(pr.lx, y);
                    R.at<cv::Vec2f>(k) = cv::Vec2f(pr.rx, y);
                    I.at<int>(k) = pt.lid;
                    ok[static_cast<size_t>(k)] =
                        isTruePair(pt.row, pr.lx, pr.rx) ? 1 : 0;
                }
                cv::cuda::GpuMat dL, dR, dI;
                dL.upload(L);
                dR.upload(R);
                dI.upload(I);
                auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
                if (rr.success && rr.d_points3d) {
                    cv::Mat P3;
                    rr.d_points3d->download(P3);
                    stream.waitForCompletion();
                    P3 = P3.reshape(3, 1);
                    std::vector<cv::Vec3f> cloud(static_cast<size_t>(n));
                    for (int k = 0; k < n; ++k) cloud[static_cast<size_t>(k)] = P3.at<cv::Vec3f>(k);
                    writePly((outDir / "pose_06_trackvote_nogate.ply").string(), cloud, ok);
                    int nOk = 0;
                    for (char c : ok) nOk += c;
                    std::ostringstream os;
                    os << "cloud pose_06_trackvote_nogate.ply: " << n << " pts (green="
                       << nOk << " red=" << (n - nOk) << ")";
                    logLine(os.str());
                }
            }
        }
        // ---------- per-lid profile: image intensity + TV4 matching behavior ----------
        {
            const auto& imgL0 = input->poseFrames[0][0].leftGray;
            cv::Mat rectL0;
            {
                cv::Mat mx, my;
                cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                            h.imageSize, CV_32FC1, mx, my);
                cv::remap(imgL0, rectL0, mx, my, cv::INTER_LINEAR);
            }
            std::map<int, std::vector<int>> inten;
            std::map<int, std::vector<int>> widthAboveHalf;
            std::map<int, int> calibCnt;
            for (auto& r : calibPairs) {
                ++calibCnt[r.lid];
                const int yy = cvRound(r.ly), xx = cvRound(r.lx);
                if (yy >= 0 && yy < rectL0.rows && xx >= 0 && xx < rectL0.cols) {
                    inten[r.lid].push_back(static_cast<int>(rectL0.at<uchar>(yy, xx)));
                    const int pk = rectL0.at<uchar>(yy, xx);
                    int w = 1;
                    for (int dx = 1; xx - dx >= 0 && dx < 30; ++dx) {
                        if (rectL0.at<uchar>(yy, xx - dx) > pk / 2) ++w; else break;
                    }
                    for (int dx = 1; xx + dx < rectL0.cols && dx < 30; ++dx) {
                        if (rectL0.at<uchar>(yy, xx + dx) > pk / 2) ++w; else break;
                    }
                    widthAboveHalf[r.lid].push_back(w);
                }
            }
            std::map<int, std::vector<std::pair<float, int>>> rTrace;
            for (auto& r : calibPairs)
                rTrace[static_cast<int>(std::lround(r.ly / rowStep))].push_back({r.rx, r.lid});
            for (auto& kv : rTrace) std::sort(kv.second.begin(), kv.second.end());
            std::map<int, std::array<int, 4>> tv4p;   // lid -> (matched, correct, ownerSelf, ownerOther)
            for (auto& pr : pairs) {
                const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                const int lid = pt.lid;
                if (lid < 0) continue;
                auto& e = tv4p[lid];
                ++e[0];
                if (isTruePair(pt.row, pr.lx, pr.rx)) ++e[1];
                auto it = rTrace.find(pt.row);
                if (it != rTrace.end()) {
                    int owner = -1;
                    float bd = 3.0f;
                    for (auto& en : it->second) {
                        const float d = std::fabs(en.first - pr.rx);
                        if (d < bd) { bd = d; owner = en.second; }
                    }
                    if (owner == lid) ++e[2]; else ++e[3];
                }
            }
            logLine("per-lid profile (calib n / center-pixel intensity med / half-width med | TV4 matched/correct/ownerSelf/ownerOther):");
            for (int lid = 1; lid <= 25; ++lid) {
                auto& iv = inten[lid];
                auto& wv = widthAboveHalf[lid];
                int med = 0, wmed = 0;
                if (!iv.empty()) {
                    std::sort(iv.begin(), iv.end());
                    med = iv[iv.size() / 2];
                    std::sort(wv.begin(), wv.end());
                    wmed = wv[wv.size() / 2];
                }
                auto& e = tv4p[lid];
                std::ostringstream os;
                os << "  lid=" << lid << " calib=" << calibCnt[lid]
                   << " intensMed=" << med << " widthMed=" << wmed
                   << " | TV4 m=" << e[0] << " c=" << e[1]
                   << " ownerSelf=" << e[2] << " ownerOther=" << e[3];
                logLine(os.str());
            }
        }
        rightAudit(false, "TV4 lids, NO gate anywhere");
        forwardAudit(false, "TV4 lids, NO gate anywhere");


            // TV5 veto: no-gate vote + forward-claim veto, 3D output
        {
            std::map<std::pair<int, long long>, std::map<int, int>> labels;
            std::map<std::pair<int, long long>, std::map<int, int>> labelsG;   // depth-gated claims
            for (auto& pt : pts)
                for (auto& c : pt.cands) {
                    ++labels[{pt.row, std::lround(c.hitRx)}][c.lid];
                    const float Z = static_cast<float>(
                        fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
                    if (std::fabs(Z - Zmed) <= DEPTH_WIN)
                        ++labelsG[{pt.row, std::lround(c.hitRx)}][c.lid];
                }
            std::map<int, std::vector<const PairRec*>> pbt;
            for (auto& pr : pairs) pbt[pts[static_cast<size_t>(pr.ptIdx)].trackId].push_back(&pr);
            std::set<int> veto;
            std::map<int, std::pair<int, int>> vetoInfo;   // ti -> (leftMode, rightMode)
            for (size_t ti = 0; ti < tracks.size(); ++ti) {
                if (tracks[ti].idx.size() < MIN_TRACK) continue;
                std::map<int, int> lmode;
                for (int pi : tracks[ti].idx) {
                    const int l = pts[static_cast<size_t>(pi)].lid;
                    if (l >= 0) ++lmode[l];
                }
                int leftMode = -1, mc = 0;
                for (auto& kv : lmode)
                    if (kv.second > mc) { mc = kv.second; leftMode = kv.first; }
                auto pit = pbt.find(static_cast<int>(ti));
                if (pit == pbt.end() || leftMode < 0) continue;
                std::map<int, int> rvotes;
                for (auto* pr : pit->second) {
                    auto lit = labels.find({pts[static_cast<size_t>(pr->ptIdx)].row,
                                            std::lround(pr->rx)});
                    if (lit == labels.end() || lit->second.empty()) continue;
                    for (auto& kv : lit->second) rvotes[kv.first] += kv.second;
                }
                int rightMode = -1, rc = 0;
                for (auto& kv : rvotes)
                    if (kv.second > rc) { rc = kv.second; rightMode = kv.first; }
                if (rightMode >= 0 && rightMode != leftMode) {
                    veto.insert(static_cast<int>(ti));
                    vetoInfo[static_cast<int>(ti)] = {leftMode, rightMode};
                }
            }
            std::vector<PairRec> kept;
            kept.reserve(pairs.size());
            int dropped = 0;
            for (auto& pr : pairs) {
                if (veto.count(pts[static_cast<size_t>(pr.ptIdx)].trackId)) { ++dropped; continue; }
                kept.push_back(pr);
            }
            pairs = kept;
            std::ostringstream osv;
            osv << "TV5 veto: " << veto.size() << " segments dropped, " << dropped << " pairs removed";
            logLine(osv.str());
            for (auto& kv : vetoInfo) {
                auto& tr = tracks[static_cast<size_t>(kv.first)];
                int rMin = 1 << 30, rMax = -(1 << 30);
                float xMin = 1e9f, xMax = -1e9f;
                for (int pi : tr.idx) {
                    const LPt& p2 = pts[static_cast<size_t>(pi)];
                    rMin = std::min(rMin, p2.row);
                    rMax = std::max(rMax, p2.row);
                    xMin = std::min(xMin, p2.x);
                    xMax = std::max(xMax, p2.x);
                }
                std::ostringstream osv2;
                osv2 << "  VETO seg#" << kv.first << ": pts=" << tr.idx.size()
                     << " rows[" << rMin << "-" << rMax << "] x[" << xMin << "-" << xMax << "]"
                     << " voted=lid" << kv.second.first
                     << " claimMode=lid" << kv.second.second;
                logLine(osv2.str());
            }
            statsOf("TV5 no-gate + fwd-veto (raw) ", pairs, false);
            statsOf("TV5 no-gate + fwd-veto +filter", pairs, true);
            const int n = static_cast<int>(pairs.size());
            if (n > 0) {
                cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
                std::vector<char> ok(static_cast<size_t>(n), 0);
                for (int k = 0; k < n; ++k) {
                    const PairRec& pr = pairs[static_cast<size_t>(k)];
                    const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                    const float y = static_cast<float>(pt.row) * rowStep;
                    L.at<cv::Vec2f>(k) = cv::Vec2f(pr.lx, y);
                    R.at<cv::Vec2f>(k) = cv::Vec2f(pr.rx, y);
                    I.at<int>(k) = pt.lid;
                    ok[static_cast<size_t>(k)] =
                        isTruePair(pt.row, pr.lx, pr.rx) ? 1 : 0;
                }
                cv::cuda::GpuMat dL, dR, dI;
                dL.upload(L);
                dR.upload(R);
                dI.upload(I);
                auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
                if (rr.success && rr.d_points3d) {
                    cv::Mat P3;
                    rr.d_points3d->download(P3);
                    stream.waitForCompletion();
                    P3 = P3.reshape(3, 1);
                    std::vector<cv::Vec3f> cloud(static_cast<size_t>(n));
                    for (int k = 0; k < n; ++k) cloud[static_cast<size_t>(k)] = P3.at<cv::Vec3f>(k);
                    writePly((outDir / "pose_06_trackvote_fwdveto.ply").string(), cloud, ok);
                    int nOk = 0;
                    for (char c : ok) nOk += c;
                    std::ostringstream os;
                    os << "cloud pose_06_trackvote_fwdveto.ply: " << n << " pts (green="
                       << nOk << " red=" << (n - nOk) << ")";
                    logLine(os.str());
                }
            }
            // labeled tracks image: color by voted LID, label at centroid, veto marks
            {
                const int W = h.imageSize.width, H = h.imageSize.height;
                cv::Mat imL2(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
                cv::Mat imR2(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
                std::vector<cv::Vec3b> trackColor(tracks.size(), cv::Vec3b(45, 45, 45));
                std::map<size_t, std::pair<int, char>> segLid;   // ti -> (lid, vetoed)
                auto lidColor = [](int lid) -> cv::Vec3b {
                    cv::Mat m(1, 1, CV_8UC3, cv::Scalar((lid * 47) % 180, 255, 255));
                    cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
                    return m.at<cv::Vec3b>(0, 0);
                };
                for (size_t t = 0; t < tracks.size(); ++t) {
                    if (tracks[t].idx.size() < MIN_TRACK) continue;
                    std::map<int, int> m;
                    for (int pi : tracks[static_cast<size_t>(t)].idx) {
                        const int l = pts[static_cast<size_t>(pi)].lid;
                        if (l >= 0) ++m[l];
                    }
                    int lid = -1, mc = 0;
                    for (auto& kv : m)
                        if (kv.second > mc) { mc = kv.second; lid = kv.first; }
                    segLid[t] = {lid, veto.count(static_cast<int>(t)) ? 1 : 0};
                    if (lid >= 0) trackColor[t] = lidColor(lid);
                }
                for (auto& pt : pts) {
                    const int x = cvRound(pt.x), y = cvRound(static_cast<float>(pt.row) * rowStep);
                    if (x < 0 || x >= W || y < 0 || y >= H) continue;
                    const cv::Vec3b& c = pt.trackId >= 0
                        ? trackColor[static_cast<size_t>(pt.trackId)] : cv::Vec3b(45, 45, 45);
                    cv::circle(imL2, {x, y}, 1, cv::Scalar(c[0], c[1], c[2]), -1);
                }
                for (auto& kv : segLid) {
                    if (kv.second.first < 0) continue;
                    double sx = 0, sy = 0;
                    for (int pi : tracks[kv.first].idx) {
                        sx += pts[static_cast<size_t>(pi)].x;
                        sy += static_cast<double>(pts[static_cast<size_t>(pi)].row) * rowStep;
                    }
                    const int cx = static_cast<int>(sx / tracks[kv.first].idx.size());
                    const int cy = static_cast<int>(sy / tracks[kv.first].idx.size());
                    const std::string s = std::to_string(kv.second.first)
                                        + (kv.second.second ? " X" : "");
                    cv::putText(imL2, s, {cx - 10, cy - 6}, 0, 0.7,
                                kv.second.second ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255), 2);
                }
                // right image: right pts colored by claim-label majority lid
                std::map<int, std::array<double, 3>> lidSum;   // lid -> (sx, sy, n)
                std::map<int, std::vector<std::pair<int, float>>> lidPts;   // lid -> (row, x)
                std::map<int, std::vector<std::pair<float, float>>> purePts;   // lid -> (x, y), margin>=2 claims
                for (auto& p : sR) {
                    const int x = cvRound(p.x), y = cvRound(p.y);
                    if (x < 0 || x >= W || y < 0 || y >= H) continue;
                    const int row = static_cast<int>(std::lround(p.y / rowStep));
                    auto lit = labels.find({row, std::lround(p.x)});
                    if (lit == labels.end() || lit->second.empty()) {
                        cv::circle(imR2, {x, y}, 1, cv::Scalar(45, 45, 45), -1);
                        continue;
                    }
                    int lid = -1, best = -1, second = -1;
                    for (auto& kv2 : lit->second) {
                        if (kv2.second > best) { second = best; best = kv2.second; lid = kv2.first; }
                        else if (kv2.second > second) second = kv2.second;
                    }
                    if (lid < 0) {
                        cv::circle(imR2, {x, y}, 1, cv::Scalar(45, 45, 45), -1);
                        continue;
                    }
                    // purity from GATED claims (label placement source)
                    {
                        auto git = labelsG.find({row, std::lround(p.x)});
                        if (git != labelsG.end() && !git->second.empty()) {
                            int gl2 = -1, gb = -1, gs = -1;
                            for (auto& kv3 : git->second) {
                                if (kv3.second > gb) { gs = gb; gb = kv3.second; gl2 = kv3.first; }
                                else if (kv3.second > gs) gs = kv3.second;
                            }
                            if (gl2 == lid && gb - gs >= 2)
                                purePts[lid].push_back({p.x, p.y});
                        }
                    }
                    auto& acc = lidSum[lid];
                    acc[0] += x;
                    acc[1] += y;
                    acc[2] += 1;
                    lidPts[lid].push_back({row, p.x});
                    const cv::Vec3b c = lidColor(lid);
                    cv::circle(imR2, {x, y}, 1, cv::Scalar(c[0], c[1], c[2]), -1);
                }
                // label placement: centroid of high-purity (margin>=2) claims per lid
                {
                    std::map<int, std::vector<std::pair<float, int>>> rTrace;
                    for (auto& r : calibPairs)
                        rTrace[static_cast<int>(std::lround(r.ly / rowStep))].push_back({r.rx, r.lid});
                    for (auto& kv : rTrace)
                        std::sort(kv.second.begin(), kv.second.end());
                    logLine("right-label placement (groupLid: n grp / n pure, pure centroid vs TRUE centroid):");
                    std::map<int, std::array<double, 3>> trueC;
                    for (auto& r : calibPairs) {
                        auto& a = trueC[r.lid];
                        a[0] += r.rx;
                        a[1] += r.ry;
                        a[2] += 1;
                    }
                    for (auto& kv : lidPts) {
                        const int gl = kv.first;
                        if (kv.second.size() < 50) continue;
                        std::map<int, int> comp;
                        for (auto& rx2 : kv.second) {
                            auto it = rTrace.find(rx2.first);
                            int owner = -1;
                            float bd = 2.0f;
                            if (it != rTrace.end())
                                for (auto& e : it->second) {
                                    const float d = std::fabs(e.first - rx2.second);
                                    if (d < bd) { bd = d; owner = e.second; }
                                }
                            ++comp[owner];
                        }
                        double ox = 0, oy = 0;
                        size_t pn = 0;
                        auto pit = purePts.find(gl);
                        if (pit != purePts.end()) {
                            pn = pit->second.size();
                            for (auto& pp : pit->second) { ox += pp.first; oy += pp.second; }
                            if (pn) { ox /= pn; oy /= pn; }
                        }
                        std::ostringstream osc;
                        osc << "  grp lid" << gl << " n=" << kv.second.size();
                        std::vector<std::pair<int, int>> cv2(comp.begin(), comp.end());
                        std::sort(cv2.begin(), cv2.end(), [](auto& a, auto& b) { return a.second > b.second; });
                        for (size_t i = 0; i < cv2.size() && i < 2; ++i)
                            osc << " " << (cv2[i].first < 0 ? std::string("noise") : std::to_string(cv2[i].first))
                                << ":" << cv2[i].second;
                        osc << " | pure=" << pn
                            << " TRUEcentroid(" << (trueC.count(gl) ? trueC[gl][0] / trueC[gl][2] : 0)
                            << "," << (trueC.count(gl) ? trueC[gl][1] / trueC[gl][2] : 0) << ")";
                        logLine(osc.str());
                        if (pn >= 50) {
                            auto& pp = pit->second;
                            // support filter: keep pts with >=10 neighbors within 40px
                            // (line pts chain along the line; isolated noise has none)
                            std::vector<std::pair<float, float>> sup;
                            for (size_t a = 0; a < pp.size(); ++a) {
                                int s = 0;
                                for (size_t b = 0; b < pp.size(); ++b) {
                                    if (a == b) continue;
                                    if (std::fabs(pp[a].first - pp[b].first) < 40.0f
                                        && std::fabs(pp[a].second - pp[b].second) < 40.0f) ++s;
                                    if (s >= 10) break;
                                }
                                if (s >= 10) sup.push_back(pp[a]);
                            }
                            if (sup.size() >= 30) {
                                std::vector<int> rows;
                                for (auto& p2 : sup)
                                    rows.push_back(static_cast<int>(std::lround(p2.second / rowStep)));
                                std::sort(rows.begin(), rows.end());
                                const int R = rows[rows.size() / 2];
                                std::vector<float> xs;
                                for (auto& p2 : sup) {
                                    const int r2 = static_cast<int>(std::lround(p2.second / rowStep));
                                    if (std::abs(r2 - R) <= 30) xs.push_back(p2.first);
                                }
                                if (xs.size() >= 10) {
                                    std::sort(xs.begin(), xs.end());
                                    const float lx = xs[xs.size() / 2];
                                    const float ly = static_cast<float>(R) * rowStep;
                                    double tx = 0, ty = 0;
                                    int tn = 0;
                                    auto tit = rTrace.upper_bound(R + 30);
                                    for (auto it3 = rTrace.lower_bound(R - 30); it3 != tit; ++it3)
                                        for (auto& e : it3->second)
                                            if (e.second == gl) {
                                                tx += e.first;
                                                ty += static_cast<double>(it3->first) * rowStep;
                                                ++tn;
                                }
                                    std::ostringstream osv2;
                                    osv2 << "    label lid" << gl << " @(" << lx << "," << ly
                                         << ")  true@sameRows(" << (tn ? tx / tn : 0)
                                         << "," << (tn ? ty / tn : 0) << ")  dx="
                                         << (tn ? lx - tx / tn : 0);
                                    logLine(osv2.str());
                                    cv::putText(imR2, std::to_string(gl),
                                                {static_cast<int>(lx) - 8, static_cast<int>(ly) - 6}, 0, 0.7,
                                                cv::Scalar(255, 255, 255), 2);
                                }
                            }
                        }
                    }
                }
                cv::Mat canvas2(H, W * 2 + 10, imL2.type(), cv::Scalar(0, 0, 0));
                imL2.copyTo(canvas2(cv::Rect(0, 0, W, H)));
                imR2.copyTo(canvas2(cv::Rect(W + 10, 0, W, H)));
                cv::putText(canvas2, "LEFT: tracks, color+label = voted lid, red 'X' = fwd-vetoed",
                            {10, 22}, 0, 0.55, cv::Scalar(255, 255, 255), 2);
                cv::putText(canvas2, "RIGHT: right pts, color = claim-label majority lid",
                            {W + 20, 22}, 0, 0.55, cv::Scalar(255, 255, 255), 2);
                cv::imwrite((outDir / "pose_06_tracks_labeled.png").string(), canvas2);
                std::cout << "labeled tracks image -> "
                          << (outDir / "pose_06_tracks_labeled.png").string() << "\n";
            }
        }
        // ---------- TV6: no-gate vote + disagreement repair via 2nd-place lid ----------
        {
            matchAll(false);   // full pairs under TV4 lids
            std::map<std::pair<int, long long>, std::map<int, int>> labels;
            for (auto& pt : pts)
                for (auto& c : pt.cands)
                    ++labels[{pt.row, std::lround(c.hitRx)}][c.lid];
            std::map<int, std::vector<const PairRec*>> pbt;
            for (auto& pr : pairs) pbt[pts[static_cast<size_t>(pr.ptIdx)].trackId].push_back(&pr);
            int nRepaired = 0, nDropped = 0;
            long long repPts = 0;
            for (size_t ti = 0; ti < tracks.size(); ++ti) {
                auto& tr = tracks[ti];
                if (tr.idx.size() < MIN_TRACK) continue;
                std::map<int, int> lmode;
                for (int pi : tr.idx) {
                    const int l = pts[static_cast<size_t>(pi)].lid;
                    if (l >= 0) ++lmode[l];
                }
                int voted = -1, vmc = 0;
                for (auto& kv : lmode)
                    if (kv.second > vmc) { vmc = kv.second; voted = kv.first; }
                auto pit = pbt.find(static_cast<int>(ti));
                if (pit == pbt.end() || voted < 0) continue;
                // claim mode over matched right pts
                std::map<int, int> rvotes;
                for (auto* pr : pit->second) {
                    auto lit = labels.find({pts[static_cast<size_t>(pr->ptIdx)].row,
                                            std::lround(pr->rx)});
                    if (lit == labels.end() || lit->second.empty()) continue;
                    for (auto& kv : lit->second) rvotes[kv.first] += kv.second;
                }
                int claimMode = -1, rc = 0;
                for (auto& kv : rvotes)
                    if (kv.second > rc) { rc = kv.second; claimMode = kv.first; }
                if (claimMode < 0 || claimMode == voted) continue;   // no disagreement
                // plain-vote 2nd place (from raw candidates, ungated)
                std::map<int, int> pv;
                std::set<std::pair<int, int>> seenRL5;
                for (int pi : tr.idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        if (seenRL5.insert({pts[static_cast<size_t>(pi)].row, c.lid}).second)
                            ++pv[c.lid];
                int first = -1, fvc = 0, second = -1, svc = 0;
                for (auto& kv : pv) {
                    if (kv.second > fvc) { second = first; svc = fvc; first = kv.first; fvc = kv.second; }
                    else if (kv.second > svc) { second = kv.first; svc = kv.second; }
                }
                const size_t nPts = tr.idx.size();
                int chosen = -1;
                const char* how = "none";
                if (second == claimMode) { chosen = second; how = "2nd==claim(double-evidence)"; }
                else if (second >= 0 && svc * 10 >= static_cast<int>(nPts) * 3) {
                    chosen = second; how = "2nd-majority(30%+)";
                }
                int rMin = 1 << 30, rMax = -(1 << 30);
                for (int pi : tr.idx) {
                    rMin = std::min(rMin, pts[static_cast<size_t>(pi)].row);
                    rMax = std::max(rMax, pts[static_cast<size_t>(pi)].row);
                }
                std::ostringstream osr2;
                osr2 << "  REPAIR seg#" << ti << ": pts=" << nPts << " rows[" << rMin << "-" << rMax << "]"
                     << " voted=lid" << voted << " 2nd=lid" << second << "(" << svc << ")"
                     << " claim=lid" << claimMode << " -> ";
                if (chosen >= 0) {
                    for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = chosen;
                    ++nRepaired;
                    repPts += static_cast<long long>(nPts);
                    osr2 << "lid" << chosen << " [" << how << "]";
                } else {
                    for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = -1;
                    ++nDropped;
                    osr2 << "DROP (no trusted 2nd)";
                }
                logLine(osr2.str());
            }
            matchAll(false);
            std::ostringstream osr3;
            osr3 << "TV6 repair: " << nRepaired << " segments re-lidded (" << repPts
                 << " pts), " << nDropped << " dropped";
            logLine(osr3.str());
            statsOf("TV6 2nd-place repair (raw)     ", pairs, false);
            statsOf("TV6 2nd-place repair + filter  ", pairs, true);
            const int n = static_cast<int>(pairs.size());
            if (n > 0) {
                cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
                std::vector<char> ok(static_cast<size_t>(n), 0);
                for (int k = 0; k < n; ++k) {
                    const PairRec& pr = pairs[static_cast<size_t>(k)];
                    const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                    const float y = static_cast<float>(pt.row) * rowStep;
                    L.at<cv::Vec2f>(k) = cv::Vec2f(pr.lx, y);
                    R.at<cv::Vec2f>(k) = cv::Vec2f(pr.rx, y);
                    I.at<int>(k) = pt.lid;
                    ok[static_cast<size_t>(k)] =
                        isTruePair(pt.row, pr.lx, pr.rx) ? 1 : 0;
                }
                cv::cuda::GpuMat dL, dR, dI;
                dL.upload(L);
                dR.upload(R);
                dI.upload(I);
                auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
                if (rr.success && rr.d_points3d) {
                    cv::Mat P3;
                    rr.d_points3d->download(P3);
                    stream.waitForCompletion();
                    P3 = P3.reshape(3, 1);
                    std::vector<cv::Vec3f> cloud(static_cast<size_t>(n));
                    for (int k = 0; k < n; ++k) cloud[static_cast<size_t>(k)] = P3.at<cv::Vec3f>(k);
                    writePly((outDir / "pose_06_trackvote_second.ply").string(), cloud, ok);
                    int nOk = 0;
                    for (char c : ok) nOk += c;
                    std::ostringstream os;
                    os << "cloud pose_06_trackvote_second.ply: " << n << " pts (green="
                       << nOk << " red=" << (n - nOk) << ")";
                    logLine(os.str());
                }
            }
        }

        // ---------- TV7: iterative rank repair (2nd/3rd/4th...), NO depth gate ----------
        // trigger: claim-disagreement OR narrow margin (<10%); accept rank r iff
        // re-matching the segment under lid_r yields claim-mode == lid_r
        {
            for (auto& pt : pts) pt.lid = -1;
            for (auto& tr : tracks) {
                if (tr.idx.size() < MIN_TRACK) continue;
            std::map<int, int> votes;
            std::set<std::pair<int, int>> seenRL;
            for (int pi : tr.idx)
                for (auto& c : pts[static_cast<size_t>(pi)].cands)
                    if (seenRL.insert({pts[static_cast<size_t>(pi)].row, c.lid}).second)
                        ++votes[c.lid];
                int winner = -1, wv = 0;
                for (auto& kv : votes)
                    if (kv.second > wv) { wv = kv.second; winner = kv.first; }
                const float share = static_cast<float>(wv) / static_cast<float>(tr.idx.size());
                if (winner >= 0 && share >= VOTE_GATE) {
                    for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = winner;
                } else {
                    for (int pi : tr.idx) {
                        LPt& pt = pts[static_cast<size_t>(pi)];
                        int bl = -1;
                        float bd = 1e9f;
                        for (auto& c : pt.cands) {
                            const float Z = static_cast<float>(
                                fB / std::max(1e-3, static_cast<double>(pt.x) - c.hitRx));
                            if (std::fabs(Z - Zmed) < bd) { bd = std::fabs(Z - Zmed); bl = c.lid; }
                        }
                        pt.lid = bl;
                    }
                }
            }
            std::map<std::pair<int, long long>, std::map<int, int>> labels;
            for (auto& pt : pts)
                for (auto& c : pt.cands)
                    ++labels[{pt.row, std::lround(c.hitRx)}][c.lid];
            auto tryMatchTrack = [&](std::vector<int>& idx, int L) -> std::array<int, 3> {
                // returns (matchedCount, claimMode, containCount)
                std::map<int, int> rv;
                int cnt = 0, contain = 0;
                for (int pi : idx) {
                    LPt& pt = pts[static_cast<size_t>(pi)];
                    auto& tab = getTab(pt.row);
                    auto it = tab.find(L);
                    if (it == tab.end() || it->second.empty()) continue;
                    auto& v = it->second;
                    int lo = 0, hi = static_cast<int>(v.size());
                    while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < pt.x) lo = mid + 1; else hi = mid; }
                    float bestDx = 1e9f, predU = -1;
                    const int down = static_cast<int>(std::ceil(UL_TOL / rowStep));
                    for (int k = std::max(0, lo - down);
                         k <= std::min(static_cast<int>(v.size()) - 1, lo + down - 1); ++k) {
                        float dx = std::fabs(v[k].first - pt.x);
                        if (dx < bestDx) { bestDx = dx; predU = v[k].second; }
                    }
                    if (bestDx > UL_TOL || predU < 0) continue;
                    const float rx = nearestRight(pt.row, predU, HIT_WIN);
                    if (rx < 0) continue;
                    ++cnt;
                    auto lit = labels.find({pt.row, std::lround(rx)});
                    if (lit != labels.end() && !lit->second.empty()) {
                        if (lit->second.count(L)) ++contain;
                        for (auto& kv : lit->second) rv[kv.first] += kv.second;
                    }
                }
                int cm = -1, best = 0;
                for (auto& kv : rv)
                    if (kv.second > best) { best = kv.second; cm = kv.first; }
                return {cnt, cm, contain};
            };
            int nFixed = 0, nReverted = 0;
            long long fixedPts = 0;
            std::map<int, int> fixedTo;   // trackId -> chosen lid (TV7 audit)
            for (size_t ti = 0; ti < tracks.size(); ++ti) {
                auto& tr = tracks[ti];
                if (tr.idx.size() < MIN_TRACK) continue;
                std::map<int, int> pv;
                std::set<std::pair<int, int>> seenRL5;
                for (int pi : tr.idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        if (seenRL5.insert({pts[static_cast<size_t>(pi)].row, c.lid}).second)
                            ++pv[c.lid];
                std::vector<std::pair<int, int>> ranked(pv.begin(), pv.end());
                std::sort(ranked.begin(), ranked.end(),
                          [](auto& a, auto& b) { return a.second > b.second; });
                if (ranked.size() < 2) continue;
                const int voted = ranked[0].first;
                const int v1 = ranked[0].second, v2 = ranked[1].second;
                const auto t0 = tryMatchTrack(tr.idx, voted);
                const bool disagree = (t0[1] >= 0 && t0[1] != voted);
                const bool narrow = (v1 - v2) * 10 < static_cast<int>(tr.idx.size());
                if (!disagree && !narrow) continue;
                int chosen = -1;
                std::ostringstream trlog;
                trlog << "  RANK seg#" << ti << ": pts=" << tr.idx.size()
                      << " voted=lid" << voted << "(" << v1 << ") trials:";
                for (size_t r = 1; r < ranked.size() && r < 4; ++r) {
                    const int L = ranked[r].first;
                    const auto tc = tryMatchTrack(tr.idx, L);
                    trlog << " lid" << L << "->m" << tc[0] << "/contain" << tc[2];
                    if (tc[0] * 10 >= static_cast<int>(tr.idx.size()) * 3
                        && tc[2] * 2 >= tc[0]) {
                        chosen = L;
                        break;
                    }
                }
                if (chosen >= 0) {
                    for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = chosen;
                    ++nFixed;
                    fixedPts += static_cast<long long>(tr.idx.size());
                    fixedTo[static_cast<int>(ti)] = chosen;
                    trlog << " => FIXED to lid" << chosen;
                    logLine(trlog.str());
                } else {
                    ++nReverted;
                    trlog << " => keep voted";
                    logLine(trlog.str());
                }
            }
            matchAll(false);
            // ---------- TV7 wrong-pair autopsy (empirical attribution) ----------
            {
                std::map<int, std::vector<std::pair<float, int>>> rTrace;
                for (auto& r : calibPairs)
                    rTrace[static_cast<int>(std::lround(r.ly / rowStep))].push_back({r.rx, r.lid});
                for (auto& kv : rTrace) std::sort(kv.second.begin(), kv.second.end());
                // per-track true mode
                std::map<int, int> trackTrue;
                for (size_t ti2 = 0; ti2 < tracks.size(); ++ti2) {
                    if (tracks[ti2].idx.size() < MIN_TRACK) continue;
                    std::map<int, int> tc2;
                    for (int pi : tracks[ti2].idx) {
                        const int t = pts[static_cast<size_t>(pi)].trueLid;
                        if (t > 0) ++tc2[t];
                    }
                    int m2 = -1, c2 = 0;
                    for (auto& kv : tc2)
                        if (kv.second > c2) { c2 = kv.second; m2 = kv.first; }
                    trackTrue[static_cast<int>(ti2)] = m2;
                }
                int srcFixedWrong = 0, srcKeptWrong = 0, srcUntrig = 0,
                    srcNoTruth = 0, srcTrueSegBad = 0;
                int depthIn = 0, wrongTot = 0;
                std::map<int, int> ownerHist;
                std::map<int, int> survivorSrc;   // depth-in-window survivors by source
                for (auto& pr : pairs) {
                    const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                    if (isTruePair(pt.row, pr.lx, pr.rx)) continue;
                    ++wrongTot;
                    const int tid = pt.trackId;
                    const int tmode = trackTrue.count(tid) ? trackTrue[tid] : -2;
                    const int src =
                        (tmode == -2) ? 4 :
                        (fixedTo.count(tid) ? (fixedTo[tid] == tmode ? 5 : 1)
                                            : (pt.lid == tmode ? 5 : 2));
                    switch (src) {
                        case 1: ++srcFixedWrong; break;
                        case 2: {
                            // kept-wrong vs untriggered: kept segs had trigger fired
                            // (they are in RANK log); approximate: lid != mode and not fixed
                            ++srcKeptWrong;
                            break;
                        }
                        case 4: ++srcNoTruth; break;
                        case 5: ++srcTrueSegBad; break;
                    }
                    const float Z = static_cast<float>(
                        fB / std::max(1e-3, static_cast<double>(pr.lx) - pr.rx));
                    const bool inWin = std::fabs(Z - Zmed) <= DEPTH_WIN;
                    if (inWin) {
                        ++depthIn;
                        ++survivorSrc[src];
                    }
                    auto it = rTrace.find(pt.row);
                    if (it != rTrace.end()) {
                        int owner = -1;
                        float bd = 3.0f;
                        for (auto& e : it->second) {
                            const float d = std::fabs(e.first - pr.rx);
                            if (d < bd) { bd = d; owner = e.second; }
                        }
                        ++ownerHist[owner];
                    } else {
                        ++ownerHist[-1];
                    }
                }
                std::ostringstream osa;
                osa << "TV7 wrong-pair autopsy (total=" << wrongTot << "):"
                    << " fixed-to-WRONG-lid=" << srcFixedWrong
                    << " kept/other-wrong-lid=" << srcKeptWrong
                    << " no-truth-seg(noise)=" << srcNoTruth
                    << " true-seg-bad-pair=" << srcTrueSegBad
                    << " | depth-in-window(survive filter)=" << depthIn;
                logLine(osa.str());
                logLine("  wrong-pair right-pt true owner (lid -> count, -1=noise):");
                std::vector<std::pair<int, int>> ov(ownerHist.begin(), ownerHist.end());
                std::sort(ov.begin(), ov.end(), [](auto& a, auto& b) { return a.second > b.second; });
                for (size_t i = 0; i < ov.size() && i < 8; ++i) {
                    std::ostringstream osb;
                    osb << "    owner=" << ov[i].first << " count=" << ov[i].second;
                    logLine(osb.str());
                }
                logLine("  filter survivors by source (1=fixedWrong 2=keptWrong 4=noTruth 5=trueSegBad):");
                for (auto& kv : survivorSrc) {
                    std::ostringstream osb;
                    osb << "    src=" << kv.first << " count=" << kv.second;
                    logLine(osb.str());
                }
            }
            std::ostringstream os7;
            os7 << "TV7 rank-repair: fixed=" << nFixed << " segs (" << fixedPts
                << " pts), reverted=" << nReverted;
            logLine(os7.str());
            statsOf("TV7 rank-repair (raw)         ", pairs, false);
            statsOf("TV7 rank-repair + filter      ", pairs, true);
            const int n = static_cast<int>(pairs.size());
            if (n > 0) {
                cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
                std::vector<char> ok(static_cast<size_t>(n), 0);
                for (int k = 0; k < n; ++k) {
                    const PairRec& pr = pairs[static_cast<size_t>(k)];
                    const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                    const float y = static_cast<float>(pt.row) * rowStep;
                    L.at<cv::Vec2f>(k) = cv::Vec2f(pr.lx, y);
                    R.at<cv::Vec2f>(k) = cv::Vec2f(pr.rx, y);
                    I.at<int>(k) = pt.lid;
                    ok[static_cast<size_t>(k)] =
                        isTruePair(pt.row, pr.lx, pr.rx) ? 1 : 0;
                }
                cv::cuda::GpuMat dL, dR, dI;
                dL.upload(L);
                dR.upload(R);
                dI.upload(I);
                auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
                if (rr.success && rr.d_points3d) {
                    cv::Mat P3;
                    rr.d_points3d->download(P3);
                    stream.waitForCompletion();
                    P3 = P3.reshape(3, 1);
                    std::vector<cv::Vec3f> cloud(static_cast<size_t>(n));
                    for (int k = 0; k < n; ++k) cloud[static_cast<size_t>(k)] = P3.at<cv::Vec3f>(k);
                    writePly((outDir / "pose_06_trackvote_rank.ply").string(), cloud, ok);
                    int nOk = 0;
                    for (char c : ok) nOk += c;
                    std::ostringstream os;
                    os << "cloud pose_06_trackvote_rank.ply: " << n << " pts (green="
                       << nOk << " red=" << (n - nOk) << ")";
                    logLine(os.str());
                }
            }
        }
    }

        // ---------- TV8: intra-row duplicate winner -> flip to runner-up ----------
        // trigger: the vote winner occurs >=2 times in some row (one of them must be
        // fake; the non-line pt claims the same phantom card as the true-line pt)
        {
            for (auto& pt : pts) pt.lid = -1;
            int nFixed8 = 0, nKept8 = 0;
            long long fixedPts8 = 0;
            int totMultiRows = 0, totRows = 0;
            // pass 1: per-seg votes + winner occupancy of (row, winner-lid)
            struct SegV { int winner = -1, fv = 0, second = -1, sv = 0, trueMode = -1; };
            std::map<size_t, SegV> segV;
            std::map<std::pair<int, int>, std::set<size_t>> winOcc;
            for (size_t ti8 = 0; ti8 < tracks.size(); ++ti8) {
                auto& tr = tracks[ti8];
                if (tr.idx.size() < MIN_TRACK) continue;
                std::map<int, int> votes;
                std::map<int, int> rowCount;
                std::map<int, int> trueCnt;
                for (int pi : tr.idx) {
                    ++rowCount[pts[static_cast<size_t>(pi)].row];
                    for (auto& c : pts[static_cast<size_t>(pi)].cands) ++votes[c.lid];
                    const int t = pts[static_cast<size_t>(pi)].trueLid;
                    if (t > 0) ++trueCnt[t];
                }
                for (auto& kv : rowCount) {
                    ++totRows;
                    if (kv.second >= 2) ++totMultiRows;
                }
                SegV sv8;
                for (auto& kv : votes) {
                    if (kv.second > sv8.fv) { sv8.second = sv8.winner; sv8.sv = sv8.fv; sv8.winner = kv.first; sv8.fv = kv.second; }
                    else if (kv.second > sv8.sv) { sv8.second = kv.first; sv8.sv = kv.second; }
                }
                int tm = -1, tc = 0;
                for (auto& kv : trueCnt)
                    if (kv.second > tc) { tc = kv.second; tm = kv.first; }
                sv8.trueMode = tm;
                if (sv8.winner < 0) continue;
                segV[ti8] = sv8;
                for (int pi : tr.idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        if (c.lid == sv8.winner)
                            winOcc[{pts[static_cast<size_t>(pi)].row, sv8.winner}].insert(ti8);
            }
            // pass 2: conflicts + self-bootstrapped normal disparity band
            // repair: re-elect the highest-vote lid whose OWN card disparity is in-band
            std::map<size_t, float> segDisp;
            std::map<size_t, std::map<int, float>> segLidDisp;
            for (auto& kv : segV) {
                std::map<int, std::vector<float>> perLid;
                for (int pi : tracks[kv.first].idx)
                    for (auto& c : pts[static_cast<size_t>(pi)].cands)
                        perLid[c.lid].push_back(pts[static_cast<size_t>(pi)].x - c.hitRx);
                for (auto& kv2 : perLid) {
                    std::sort(kv2.second.begin(), kv2.second.end());
                    segLidDisp[kv.first][kv2.first] = kv2.second[kv2.second.size() / 2];
                    if (kv2.first == kv.second.winner)
                        segDisp[kv.first] = kv2.second[kv2.second.size() / 2];
                }
            }
            std::vector<float> allDisp;
            for (auto& kv : segDisp) allDisp.push_back(kv.second);
            std::sort(allDisp.begin(), allDisp.end());
            const float bandC = allDisp[allDisp.size() / 2];
            const float bandW = 35.0f;
            auto inBand = [&](float d) { return std::fabs(d - bandC) <= bandW; };
            std::set<size_t> toFlip;
            std::map<std::pair<size_t, size_t>, int> conflictPairs;
            for (auto& kv : winOcc) {
                if (kv.second.size() < 2) continue;
                std::vector<size_t> v(kv.second.begin(), kv.second.end());
                for (size_t a = 0; a < v.size(); ++a)
                    for (size_t b = a + 1; b < v.size(); ++b)
                        ++conflictPairs[{v[a], v[b]}];
                size_t out = *(kv.second.begin());
                float worstDev = -1.0f;
                for (size_t s : kv.second) {
                    auto it8 = segDisp.find(s);
                    if (it8 == segDisp.end()) continue;
                    const float dev = std::fabs(it8->second - bandC);
                    if (dev > worstDev) { worstDev = dev; out = s; }
                }
                if (worstDev > bandW) toFlip.insert(out);   // only if truly outside band
            }
            {
                std::ostringstream osb;
                osb << "TV8c band: center=" << bandC << " width=+-" << bandW;
                logLine(osb.str());
            }
            logLine("cross-seg (row,lid) conflicts - WRONG/RIGHT breakdown (1st/2nd votes + disparity):");
            std::vector<std::pair<std::pair<size_t, size_t>, int>> cp(conflictPairs.begin(), conflictPairs.end());
            std::sort(cp.begin(), cp.end(), [](auto& a, auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < cp.size() && i < 20; ++i) {
                for (int side = 0; side < 2; ++side) {
                    const size_t sid = side == 0 ? cp[i].first.first : cp[i].first.second;
                    auto sS = segV.find(sid);
                    if (sS == segV.end()) continue;
                    const SegV& sv9 = sS->second;
                    const bool wrong = (sv9.trueMode > 0 && sv9.winner != sv9.trueMode);
                    std::ostringstream osc;
                    osc << (side == 0 ? "  [" : "     ") << (wrong ? "WRONG" : "RIGHT")
                        << "] seg" << sid << " pts=" << tracks[sid].idx.size()
                        << " 1st=lid" << sv9.winner << "(" << sv9.fv << ")"
                        << " 2nd=lid" << sv9.second << "(" << sv9.sv << ")"
                        << " disp=" << (segDisp.count(sid) ? segDisp[sid] : 0.0f)
                        << " trueLid=" << sv9.trueMode
                        << (side == 1 ? " sharedRows=" + std::to_string(cp[i].second) : "");
                    logLine(osc.str());
                }
            }
            const fs::path errCsv = outDir / "error_segments.csv";
            std::ofstream cfe(errCsv.string());
            cfe << "segId,pts,rowMin,rowMax,xMin,xMax,action,fakeLid,fakeVotes,fakeDisp,trueLid,reElectLid,reElectDisp,opponentSeg,sharedRows\n";
            for (size_t ti8 = 0; ti8 < tracks.size(); ++ti8) {
                auto& tr = tracks[ti8];
                if (tr.idx.size() < MIN_TRACK) continue;
                auto it8 = segV.find(ti8);
                if (it8 == segV.end()) continue;
                const SegV& sv8 = it8->second;
                if (toFlip.count(ti8)) {
                    // re-elect: highest-vote lid whose own card disparity is in-band
                    int chosen = -1, cv = 0;
                    float cdisp = 0;
                    auto ld = segLidDisp.find(ti8);
                    std::map<int, int> votes2;
                    for (int pi : tr.idx)
                        for (auto& c : pts[static_cast<size_t>(pi)].cands) ++votes2[c.lid];
                    if (ld != segLidDisp.end())
                        for (auto& kv2 : votes2) {
                            auto d2 = ld->second.find(kv2.first);
                            if (d2 == ld->second.end() || !inBand(d2->second)) continue;
                            if (kv2.second > cv) { cv = kv2.second; chosen = kv2.first; cdisp = d2->second; }
                        }
                    for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = chosen;
                    ++nFixed8;
                    fixedPts8 += static_cast<long long>(tr.idx.size());
                    std::ostringstream os8;
                    os8 << "  BANDFIX seg#" << ti8 << ": pts=" << tr.idx.size()
                        << " was 1st=lid" << sv8.winner << "(" << sv8.fv
                        << ") disp=" << (segDisp.count(ti8) ? segDisp[ti8] : 0.0f)
                        << " -> re-elect lid" << chosen << "(" << cv << ") disp=" << cdisp;
                    logLine(os8.str());
                    // CSV row: geometry + ids + disparities + biggest conflict opponent
                    int rMin = 1 << 30, rMax = -(1 << 30);
                    float xMin = 1e9f, xMax = -1e9f;
                    for (int pi : tr.idx) {
                        const LPt& p2 = pts[static_cast<size_t>(pi)];
                        rMin = std::min(rMin, p2.row);
                        rMax = std::max(rMax, p2.row);
                        xMin = std::min(xMin, p2.x);
                        xMax = std::max(xMax, p2.x);
                    }
                    size_t opp = ti8;
                    int shared = 0;
                    for (auto& kv : conflictPairs) {
                        if (kv.first.first == ti8 || kv.first.second == ti8) {
                            if (kv.second > shared) {
                                shared = kv.second;
                                opp = kv.first.first == ti8 ? kv.first.second : kv.first.first;
                            }
                        }
                    }
                    cfe << ti8 << "," << tr.idx.size() << "," << rMin << "," << rMax << ","
                        << std::fixed << std::setprecision(2) << xMin << "," << xMax << ","
                        << (chosen >= 0 ? "FIX" : "DROP") << ","
                        << sv8.winner << "," << sv8.fv << ","
                        << (segDisp.count(ti8) ? segDisp[ti8] : 0.0f) << ","
                        << sv8.trueMode << "," << chosen << "," << cdisp << ","
                        << (opp != ti8 ? static_cast<long long>(opp) : -1) << "," << shared << "\n";
                } else {
                    for (int pi : tr.idx) pts[static_cast<size_t>(pi)].lid = sv8.winner;
                    ++nKept8;
                }
            }
            cfe.close();
            {
                std::ostringstream ose3;
                ose3 << "error segments export -> " << errCsv.string();
                logLine(ose3.str());
            }
            // SIMULATION: user's drop-both policy (conflict pair members all dropped)
            {
                std::set<size_t> dropBoth;
                for (auto& kv : conflictPairs) {
                    dropBoth.insert(kv.first.first);
                    dropBoth.insert(kv.first.second);
                }
                for (size_t ti8 = 0; ti8 < tracks.size(); ++ti8) {
                    if (dropBoth.count(ti8))
                        for (int pi : tracks[ti8].idx)
                            pts[static_cast<size_t>(pi)].lid = -1;
                }
                std::ostringstream osd;
                osd << "SIM drop-both: " << dropBoth.size() << " segments dropped";
                logLine(osd.str());
            }
            matchAll(false);
            std::ostringstream os8s;
            os8s << "TV8 dup-flip: fixed=" << nFixed8 << " segs (" << fixedPts8
                 << " pts), kept=" << nKept8
                 << " | rows-with-multiple-pts: " << totMultiRows << " / " << totRows;
            logLine(os8s.str());
            statsOf("TV8 dup-flip (raw)           ", pairs, false);
            statsOf("TV8 dup-flip + filter        ", pairs, true);
            {
                const int n = static_cast<int>(pairs.size());
                if (n > 0) {
                    cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
                    std::vector<char> ok(static_cast<size_t>(n), 0);
                    for (int k = 0; k < n; ++k) {
                        const PairRec& pr = pairs[static_cast<size_t>(k)];
                        const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
                        const float y = static_cast<float>(pt.row) * rowStep;
                        L.at<cv::Vec2f>(k) = cv::Vec2f(pr.lx, y);
                        R.at<cv::Vec2f>(k) = cv::Vec2f(pr.rx, y);
                        I.at<int>(k) = pt.lid;
                        ok[static_cast<size_t>(k)] =
                            isTruePair(pt.row, pr.lx, pr.rx) ? 1 : 0;
                    }
                    cv::cuda::GpuMat dL, dR, dI;
                    dL.upload(L);
                    dR.upload(R);
                    dI.upload(I);
                    auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
                    if (rr.success && rr.d_points3d) {
                        cv::Mat P3;
                        rr.d_points3d->download(P3);
                        stream.waitForCompletion();
                        P3 = P3.reshape(3, 1);
                        std::vector<cv::Vec3f> cloud(static_cast<size_t>(n));
                        for (int k = 0; k < n; ++k) cloud[static_cast<size_t>(k)] = P3.at<cv::Vec3f>(k);
                        writePly((outDir / "pose_06_trackvote_band.ply").string(), cloud, ok);
                        int nOk = 0;
                        for (char c : ok) nOk += c;
                        std::ostringstream os;
                        os << "cloud pose_06_trackvote_band.ply: " << n << " pts (green="
                           << nOk << " red=" << (n - nOk) << ")";
                        logLine(os.str());
                    }
                }
            }
        }
    // ---------- SIM v3-goldstandard: v2 match semantics + pair-vote winner + drop-both ----------
    {
        const float THR = 2.0f;
        // candidates per left pt: full-entry |uL-x|<=0.7, dedup (lid,uRbin), nearest-uL first
        auto v2cands = [&](int row, float xL, std::vector<std::pair<float, int>>& out) {
            out.clear();
            auto& tab = getTab(row);
            std::map<long long, std::pair<float, int>> seen;   // (lid,uRbin) -> (uR,lid)
            for (auto& bl : tab) {
                auto& v = bl.second;
                int lo = 0, hi = static_cast<int>(v.size());
                while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < xL) lo = mid + 1; else hi = mid; }
                for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                    if (std::fabs(v[k].first - xL) > 0.7f) continue;
                    const long long key = (static_cast<long long>(bl.first) << 20)
                                        | static_cast<long long>(std::lround(v[k].second / rowStep));
                    if (!seen.count(key)) seen[key] = {v[k].second, bl.first};
                }
            }
            for (auto& kv : seen) out.push_back(kv.second);
        };
        // per-row matching (row-serial, in-row left pts by x order; right greedy occupied)
        struct V2Pair { int ptIdx; float lx, rx; int lid; };
        std::vector<V2Pair> v2pairs;
        for (auto& kv : lByRow) {
            const int row = kv.first;
            auto& lidx = kv.second;                     // x-sorted left pt indices
            std::vector<float> rxs;
            auto rit = rByRow.find(row);
            if (rit != rByRow.end()) rxs = rit->second; // x-sorted right xs
            std::vector<char> occ(rxs.size(), 0);
            for (int pi : lidx) {
                const LPt& pt = pts[static_cast<size_t>(pi)];
                std::vector<std::pair<float, int>> cands;
                v2cands(row, pt.x, cands);
                if (cands.empty()) continue;
                int hitSm = -1, hitCand = -1, totalHits = 0;
                for (int c = 0; c < static_cast<int>(cands.size()) && totalHits < 2; ++c) {
                    const float loV = cands[static_cast<size_t>(c)].first - THR;
                    const float hiV = cands[static_cast<size_t>(c)].first + THR;
                    for (size_t ri = 0; ri < rxs.size() && totalHits < 2; ++ri) {
                        if (rxs[ri] < loV) continue;
                        if (rxs[ri] > hiV) break;
                        if (occ[ri]) continue;
                        if (static_cast<int>(ri) == hitSm) continue;
                        ++totalHits;
                        hitSm = static_cast<int>(ri);
                        hitCand = c;
                    }
                }
                if (totalHits == 1) {
                    occ[static_cast<size_t>(hitSm)] = 1;
                    v2pairs.push_back({pi, pt.x, rxs[static_cast<size_t>(hitSm)],
                                       cands[static_cast<size_t>(hitCand)].second});
                }
            }
        }
        // pair-vote winner per track + conflict (row, winner) + drop-both
        std::map<int, std::map<int, int>> segLid;        // track -> lid -> pairs
        for (auto& pr : v2pairs)
            segLid[pts[static_cast<size_t>(pr.ptIdx)].trackId][pr.lid]++;
        std::map<int, int> segWin;
        for (auto& kv : segLid) {
            int w = -1, best = 0;
            for (auto& kv2 : kv.second)
                if (kv2.second > best) { best = kv2.second; w = kv2.first; }
            segWin[kv.first] = w;
        }
        std::map<std::pair<int, int>, std::set<int>> rowWinSeg;
        for (auto& pr : v2pairs) {
            const int tid = pts[static_cast<size_t>(pr.ptIdx)].trackId;
            const int row = pts[static_cast<size_t>(pr.ptIdx)].row;
            if (segWin[tid] == pr.lid)
                rowWinSeg[{row, segWin[tid]}].insert(tid);
        }
        std::set<int> dropSegs;
        for (auto& kv : rowWinSeg)
            if (kv.second.size() >= 2)
                for (int s : kv.second) dropSegs.insert(s);
        int keptC = 0, keptW = 0, dropC = 0, dropW = 0;
        for (auto& pr : v2pairs) {
            const LPt& pt = pts[static_cast<size_t>(pr.ptIdx)];
            const bool ok = isTruePair(pt.row, pr.lx, pr.rx);
            if (dropSegs.count(pt.trackId)) { ok ? ++dropC : ++dropW; }
            else { ok ? ++keptC : ++keptW; }
        }
        std::ostringstream os9;
        os9 << "SIM v3 goldstandard (v2-unique-hit + pair-vote + drop-both):"
            << " v2pairs=" << v2pairs.size()
            << " tracksWithPairs=" << segLid.size()
            << " conflictDropped=" << dropSegs.size()
            << " | keep: correct=" << keptC << " wrong=" << keptW
            << " | dropped side: correct=" << dropC << " wrong=" << dropW
            << " | FINAL=" << keptC << "/" << keptW << "/" << (calibTotal - keptC);
        logLine(os9.str());
    }

    // ---------- V3 operator e2e verification (gold standard: 13602/0) ----------
    {
        LaserMatchScanParamsV3 vp3;
        vp3.epipolar_row_step = rowStep;
        vp3.vL_tolerance = 0.7f;
        vp3.match_threshold = 2.0f;
        LaserMatchScanCudaV3 v3(vp3);
        if (!v3.SetTempTable(tempTable)) {
            logLine("V3: SetTempTable failed");
        } else {
            std::cout << "V3DBG: before SetCurrentTemperature\n";
            v3.SetCurrentTemperature(cfg.referenceTemp);
            std::cout << "V3DBG: before chain\n";
            const auto& img0 = input->poseFrames[0][0];
            auto m1 = maskS1.Execute(img0.leftGray, stream);
            auto m2 = maskS2.Execute(img0.rightGray, stream);
            auto s1 = stgS1.Execute(*m1.d_grayImage, *m1.d_cleanedMask, stream, GroupMode::Flat);
            auto s2 = stgS2.Execute(*m2.d_grayImage, *m2.d_cleanedMask, stream, GroupMode::Flat);
            auto u1 = unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream);
            auto u2 = unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream);
            auto e1 = epiS1.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream);
            auto e2 = epiS2.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream);
            std::cout << "V3DBG: before Execute\n";
            auto mr3 = v3.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                  *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
            std::cout << "V3DBG: after Execute\n";
            stream.waitForCompletion();
            std::ostringstream osv;
            osv << "V3 operator: success=" << mr3.success << " msg=" << mr3.message;
            logLine(osv.str());
            if (mr3.success && mr3.matchedCount > 0) {
                cv::Mat mL, mR, mI, mT;
                mr3.d_matched_left->download(mL);
                mr3.d_matched_right->download(mR);
                mr3.d_matched_line_ids->download(mI);
                mr3.d_matched_track_ids->download(mT);
                stream.waitForCompletion();
                mL = mL.reshape(2, 1);
                mR = mR.reshape(2, 1);
                const int n3 = mr3.matchedCount;
                int correct = 0;
                for (int k = 0; k < n3; ++k) {
                    const cv::Vec2f l = mL.at<cv::Vec2f>(k);
                    const cv::Vec2f r = mR.at<cv::Vec2f>(k);
                    const int row = static_cast<int>(std::lround(l[1] / rowStep));
                    if (isTruePair(row, l[0], r[0])) ++correct;
                }
                std::ostringstream osv2;
                osv2 << "V3 gold check: matched=" << n3 << " correct=" << correct
                     << " wrong=" << (n3 - correct)
                     << " lost=" << (calibTotal - correct)
                     << " | tracks total=" << mr3.trackTotal
                     << " dropped=" << mr3.trackDropped
                     << " (target 13602/0, sim drop-both)";
                logLine(osv2.str());
            }
        }
    }

    // ---------- diagnostics ----------
    {
        std::ostringstream os;
        os << "\ntracks: total=" << tracks.size() << " small(<" << MIN_TRACK << ")=" << smallTracks
           << " kept=" << keptTracks << " gate-pass=" << gatePass << " gate-fail=" << gateFail;
        logLine(os.str());
        std::ostringstream os2;
        os2 << "bootstrap Zmed (unique-HIT median, n=" << uniqDepths.size() << ") = "
            << Zmed << "mm";
        logLine(os2.str());
        std::ostringstream os3;
        os3 << "left pts=" << pts.size() << " with-truth=" << ptsWithTruth
            << " with-cand=" << ptsWithCand;
        logLine(os3.str());
        std::ostringstream os4;
        os4 << "per-point voted-lid accuracy (truth-known pts in kept tracks): "
            << lidAccOk << "/" << lidAccTot;
        logLine(os4.str());
        std::ostringstream os5;
        os5 << "gate-fail tiebreak pts=" << tiebreakPts << " correct-lid=" << tiebreakOk;
        logLine(os5.str());
        logLine("per-track winner-lid aggregate (lid: tracks/pts/true-rate):");
        for (auto& kv : lidAgg) {
            auto& t = lidTrue[kv.first];
            std::ostringstream os6;
            os6 << "  lid=" << kv.first << " tracks=" << kv.second.first
                << " pts=" << kv.second.second
                << " true-rate=" << (t.first ? 100.0 * t.second / t.first : 0.0) << "%";
            logLine(os6.str());
        }
    }

    std::cout << "\ntrackvote outputs -> " << outDir.string() << "\n";
    SUCCEED();
}
