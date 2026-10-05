// test_scan_vs_calib_analysis.cpp - scan-chain vs calib-chain matching quality analysis
// (analysis only, no operator modification)
//
// Chains on pose_06/pose_07:
//   calib: mask -> ccl -> label -> steger -> undistort -> interp -> match_v1 -> recon  (ground truth)
//   scan : mask -> steger_fast(Flat) -> undistort -> interp_dual(scan) -> match_v2 -> recon
// Sections:
//   per-frame: plane fit (calib cloud), scan outliers, pair association classification, PLYs
//   decisive dump: table entries rows 440/445/450, chain interp/matched dumps
//   (e) table prediction accuracy vs calib ground truth
//   (f) row-445 hypothesis-vs-truth matrix
//   (h) table structure stats + interval exclusion audits (depth window / row-monotone lid)
//   (g) LR cross-check audit (rejected as primary adjudicator)
// Outputs: data_out/analysis/{summary.txt, per-frame PLYs}

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
#include <cstdlib>
#include <cmath>
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
#include "laser_match_scan_v2_cuda.h"
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
              const cv::Scalar& color) {
    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
      << std::fixed << std::setprecision(3);
    for (const auto& p : pts)
        f << p[0] << " " << p[1] << " " << p[2] << " "
          << (int)color[0] << " " << (int)color[1] << " " << (int)color[2] << "\n";
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

TEST(ScanVsCalibAnalysis, TwoFrames) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir, calibJson;
    if (!findData(dataDir, calibJson)) {
        GTEST_SKIP() << "left_skew data not found; skip analysis";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "analysis";
    fs::create_directories(outDir);
    std::ofstream summary((outDir / "summary.txt").string());

    // ---------- load calibration + build table (same as fcscan_e2e) ----------
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

    // ---------- operators (both chains) ----------
    cv::cuda::Stream stream;

    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
    MaskExtractCUDA maskC1(mp), maskC2(mp), maskS1(mp), maskS2(mp);

    RegionAnalyzerParams cp; cp.minArea = 0; cp.topXCount = 27;
    RegionAnalyzerCUDA cclL(cp), cclR(cp);
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

    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    // download matched pairs + 3d points
    auto collect = [&](auto& mr, cv::cuda::Stream& st) -> std::vector<MatchPairRec> {
        std::vector<MatchPairRec> out;
        if (!mr.success || mr.matchedCount <= 0) return out;
        auto rr = recon.Execute(*mr.d_matched_left, *mr.d_matched_right,
                                *mr.d_matched_line_ids, h.Q, st);
        if (!rr.success || !rr.d_points3d) return out;
        cv::Mat L, R, I, P3, I3;
        mr.d_matched_left->download(L);
        mr.d_matched_right->download(R);
        mr.d_matched_line_ids->download(I);
        rr.d_points3d->download(P3);
        rr.d_valid_line_ids->download(I3);
        st.waitForCompletion();
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

    auto runCalib = [&](size_t pi, size_t ti,
                        std::vector<cv::Point2f>* dumpL = nullptr,
                        std::vector<cv::Point2f>* dumpR = nullptr,
                        std::vector<int>* dumpLI = nullptr,
                        std::vector<int>* dumpRI = nullptr) {
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
        if (dumpL) *dumpL = downloadPoints(*e1.d_interpPoints);
        if (dumpR) *dumpR = downloadPoints(*e2.d_interpPoints);
        if (dumpLI) {
            cv::Mat m;
            e1.d_interp_line_ids->download(m);
            m = m.reshape(1, 1);
            *dumpLI = std::vector<int>(m.begin<int>(), m.end<int>());
        }
        if (dumpRI) {
            cv::Mat m;
            e2.d_interp_line_ids->download(m);
            m = m.reshape(1, 1);
            *dumpRI = std::vector<int>(m.begin<int>(), m.end<int>());
        }
        stream.waitForCompletion();
        auto mr = matcherV1.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                     *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
        return collect(mr, stream);
    };

    auto runScan = [&](size_t pi, size_t ti,
                       std::vector<cv::Point2f>* dumpL = nullptr,
                       std::vector<cv::Point2f>* dumpR = nullptr) {
        const auto& img = input->poseFrames[pi][ti];
        auto m1 = maskS1.Execute(img.leftGray, stream);
        auto m2 = maskS2.Execute(img.rightGray, stream);
        auto s1 = stgS1.Execute(*m1.d_grayImage, *m1.d_cleanedMask, stream, GroupMode::Flat);
        auto s2 = stgS2.Execute(*m2.d_grayImage, *m2.d_cleanedMask, stream, GroupMode::Flat);
        auto u1 = unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream);
        auto u2 = unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream);
        auto e1 = epiS1.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream);
        auto e2 = epiS2.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream);
        if (dumpL) *dumpL = downloadPoints(*e1.d_interpPoints);
        if (dumpR) *dumpR = downloadPoints(*e2.d_interpPoints);
        stream.waitForCompletion();
        auto mr = matcherV2.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                     *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
        return collect(mr, stream);
    };

    // plane fit
    auto fitPlane = [](const std::vector<cv::Vec3f>& pts, cv::Vec3f& centroid, cv::Vec3f& normal) {
        cv::Mat A(static_cast<int>(pts.size()), 3, CV_32F);
        centroid = cv::Vec3f(0, 0, 0);
        for (size_t i = 0; i < pts.size(); ++i) {
            centroid += pts[i];
            A.at<float>(static_cast<int>(i), 0) = pts[i][0];
            A.at<float>(static_cast<int>(i), 1) = pts[i][1];
            A.at<float>(static_cast<int>(i), 2) = pts[i][2];
        }
        centroid *= 1.0f / static_cast<float>(pts.size());
        for (int i = 0; i < A.rows; ++i) {
            A.at<float>(i, 0) -= centroid[0];
            A.at<float>(i, 1) -= centroid[1];
            A.at<float>(i, 2) -= centroid[2];
        }
        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt);
        normal = cv::Vec3f(vt.at<float>(2, 0), vt.at<float>(2, 1), vt.at<float>(2, 2));
    };

    ASSERT_GE(input->poseFrames.size(), 2u);
    (void)runCalib(0, 0);
    (void)runScan(0, 0);

    for (size_t fi = 0; fi < 2; ++fi) {
        const std::string stem = input->poseDirs[fi] + "_tube0";
        auto calib = runCalib(fi, 0);
        auto scan = runScan(fi, 0);
        ASSERT_FALSE(calib.empty()) << stem << ": calib chain empty";
        std::cout << "\n==== " << stem << " ====\n";
        summary << "\n==== " << stem << " ====\n";

        std::vector<cv::Vec3f> cpts;
        cpts.reserve(calib.size());
        for (auto& r : calib) cpts.push_back(r.p3);
        cv::Vec3f centroid, normal;
        fitPlane(cpts, centroid, normal);
        double calibRms = 0;
        for (size_t i = 0; i < cpts.size(); ++i) {
            float d = (cpts[i] - centroid).dot(normal);
            calibRms += d * d;
        }
        calibRms = std::sqrt(calibRms / cpts.size());
        std::cout << "calib: matched=" << calib.size() << " planeRMS=" << std::fixed
                  << std::setprecision(4) << calibRms << "mm\n";
        summary << "calib matched=" << calib.size() << " planeRMS=" << calibRms << "mm\n";

        std::vector<float> sD(scan.size());
        int out1 = 0, out2 = 0, out5 = 0;
        for (size_t i = 0; i < scan.size(); ++i) {
            sD[i] = (scan[i].p3 - centroid).dot(normal);
            float a = std::fabs(sD[i]);
            if (a > 1.0f) ++out1;
            if (a > 2.0f) ++out2;
            if (a > 5.0f) ++out5;
        }
        std::cout << "scan : matched=" << scan.size() << " |planeDist| >1mm: " << out1
                  << "  >2mm: " << out2 << "  >5mm: " << out5 << "\n";
        summary << "scan matched=" << scan.size() << " out>1mm=" << out1
                << " out>2mm=" << out2 << " out>5mm=" << out5 << "\n";

        auto buildIndex = [&](const std::vector<MatchPairRec>& v) {
            std::multimap<int, int> idx;
            for (size_t i = 0; i < v.size(); ++i)
                idx.emplace(static_cast<int>(std::lround(v[i].ly / rowStep)), static_cast<int>(i));
            return idx;
        };
        auto scanIdx = buildIndex(scan);
        std::vector<char> scanPaired(scan.size(), 0);
        int commonSame = 0, commonDiff = 0, calibOnly = 0;
        std::vector<int> diffIdx, onlyIdx;
        for (size_t ci = 0; ci < calib.size(); ++ci) {
            const int row = static_cast<int>(std::lround(calib[ci].ly / rowStep));
            auto [it, itEnd] = scanIdx.equal_range(row);
            int best = -1;
            float bestDx = 0.6f;
            for (auto k = it; k != itEnd; ++k) {
                const float dx = std::fabs(scan[k->second].lx - calib[ci].lx);
                if (dx < bestDx) { bestDx = dx; best = k->second; }
            }
            if (best < 0) {
                ++calibOnly;
                onlyIdx.push_back(static_cast<int>(ci));
                continue;
            }
            scanPaired[best] = 1;
            if (std::fabs(scan[best].rx - calib[ci].rx) <= 0.5f) {
                ++commonSame;
            } else {
                ++commonDiff;
                diffIdx.push_back(best);
            }
        }
        int scanOnly = 0;
        for (size_t i = 0; i < scan.size(); ++i)
            if (!scanPaired[i]) ++scanOnly;

        std::cout << "assoc: common-same=" << commonSame << " common-diff=" << commonDiff
                  << " calib-only=" << calibOnly << " scan-only=" << scanOnly << "\n";
        summary << "common-same=" << commonSame << " common-diff=" << commonDiff
                << " calib-only=" << calibOnly << " scan-only=" << scanOnly << "\n";

        int outFromDiff = 0, outFromOnly = 0, outFromSame = 0;
        std::vector<int> outIdx;
        for (size_t i = 0; i < scan.size(); ++i) {
            if (std::fabs(sD[i]) > 2.0f) {
                outIdx.push_back(static_cast<int>(i));
                if (std::find(diffIdx.begin(), diffIdx.end(), static_cast<int>(i)) != diffIdx.end())
                    ++outFromDiff;
                else if (!scanPaired[i])
                    ++outFromOnly;
                else
                    ++outFromSame;
            }
        }
        std::cout << "outliers(>2mm)=" << outIdx.size() << " source: common-diff=" << outFromDiff
                  << " scan-only=" << outFromOnly << " common-same=" << outFromSame << "\n";
        summary << "outliers=" << outIdx.size() << " (diff=" << outFromDiff
                << " only=" << outFromOnly << " same=" << outFromSame << ")\n";

        std::vector<cv::Vec3f> inliers, outliers;
        for (size_t i = 0; i < scan.size(); ++i)
            (std::fabs(sD[i]) > 2.0f ? outliers : inliers).push_back(scan[i].p3);
        writePly((outDir / (stem + "_calib.ply")).string(), cpts, cv::Scalar(0, 200, 0));
        writePly((outDir / (stem + "_scan_inliers.ply")).string(), inliers, cv::Scalar(220, 220, 220));
        writePly((outDir / (stem + "_scan_outliers.ply")).string(), outliers, cv::Scalar(0, 0, 255));
        summary << "inliers=" << inliers.size() << " outliers=" << outliers.size() << "\n";
    }

    // ================= decisive dump (pose_06, rows 440-450) =================
    {
        std::vector<cv::Point2f> cL, cR, sL, sR;
        runCalib(0, 0, &cL, &cR);
        runScan(0, 0, &sL, &sR);
        auto calibPairs = runCalib(0, 0);
        auto scanPairs = runScan(0, 0);

        auto dumpPts = [&](const char* tag, const std::vector<cv::Point2f>& pts,
                           float x0, float x1) {
            std::cout << "-- " << tag << " rows 440-450 --\n";
            for (auto& p : pts) {
                const int row = static_cast<int>(std::lround(p.y / rowStep));
                if (row >= 440 && row <= 450 && p.x >= x0 && p.x <= x1)
                    std::cout << "    row=" << row << " x=" << p.x << "\n";
            }
        };
        auto dumpPairs = [&](const char* tag, const std::vector<MatchPairRec>& v) {
            std::cout << "-- " << tag << " rows 440-450 --\n";
            for (auto& r : v) {
                const int row = static_cast<int>(std::lround(r.ly / rowStep));
                if (row >= 440 && row <= 450 && r.lx >= 1100 && r.lx <= 1560)
                    std::cout << "    row=" << row << " Lx=" << r.lx
                              << " Rx=" << r.rx << " lid=" << r.lid << "\n";
            }
        };
        dumpPts("calib LEFT interp", cL, 0, 2048);
        dumpPts("scan  LEFT interp", sL, 0, 2048);
        dumpPts("calib RIGHT interp", cR, 0, 2048);
        dumpPts("scan  RIGHT interp", sR, 0, 2048);
        dumpPairs("calib matched", calibPairs);
        dumpPairs("scan  matched", scanPairs);

        // (e) table prediction accuracy vs calib ground truth
        {
            std::cout << "\n==== table accuracy vs calib ground truth (pose_06) ====\n";
            std::map<int, std::vector<const MatchPairRec*>> byRow;
            for (auto& r : calibPairs)
                byRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
            double sumErr = 0, sumErr2 = 0;
            float maxErr = 0;
            int nOk = 0, nMiss = 0, nBad = 0;
            std::vector<float> errs;
            for (auto& kv : byRow) {
                const int row = kv.first;
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
                for (auto* r : kv.second) {
                    auto it = byLid.find(r->lid);
                    if (it == byLid.end() || it->second.empty()) { ++nMiss; continue; }
                    auto& v = it->second;
                    int lo = 0, hi = static_cast<int>(v.size());
                    while (lo < hi) {
                        int mid = (lo + hi) / 2;
                        if (v[mid].first < r->lx) lo = mid + 1; else hi = mid;
                    }
                    float bestDx = 1e9f, pred = -1;
                    for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                        float dx = std::fabs(v[k].first - r->lx);
                        if (dx < bestDx) { bestDx = dx; pred = v[k].second; }
                    }
                    const float err = pred - r->rx;
                    errs.push_back(err);
                    sumErr += err; sumErr2 += static_cast<double>(err) * err;
                    maxErr = std::max(maxErr, std::fabs(err));
                    if (std::fabs(err) <= 2.0f) ++nOk; else ++nBad;
                }
            }
            const int n = static_cast<int>(errs.size());
            std::sort(errs.begin(), errs.end());
            auto pct = [&](double p) { return errs[static_cast<size_t>(n * p)]; };
            int b0 = 0, b1 = 0, b2 = 0, b5 = 0, bn1 = 0, bn2 = 0, bn5 = 0;
            for (float e : errs) {
                if (e >= 0 && e < 0.35f) ++b0; else if (e >= 0.35f && e < 1.0f) ++b1;
                else if (e >= 1.0f && e < 2.0f) ++b2; else if (e >= 2.0f && e < 5.0f) ++b5;
                else if (e < 0 && e > -0.35f) ++bn1; else if (e <= -0.35f && e > -1.0f) ++bn1;
                else if (e <= -1.0f && e > -2.0f) ++bn2; else if (e <= -2.0f && e > -5.0f) ++bn5;
            }
            std::cout << "err distribution: [0,0.35): " << b0 << " | [0.35,1): " << b1
                      << " | [1,2): " << b2 << " | [2,5): " << b5
                      << " | (-0.35,0): " << 0 << " | (-1,-0.35]: " << bn1
                      << " | (-2,-1]: " << bn2 << " | (-5,-2]: " << bn5 << "\n";
            std::cout << "percentiles: P50=" << pct(0.50) << " P90=" << pct(0.90)
                      << " P95=" << pct(0.95) << " P99=" << pct(0.99)
                      << " P99.5=" << pct(0.995) << " max=" << errs[n - 1]
                      << " min=" << errs[0] << "\n";
            std::cout << "calib pairs checked=" << calibPairs.size()
                      << " (miss lid=" << nMiss << ")"
                      << " | pred err: mean=" << (n ? sumErr / n : 0)
                      << " rms=" << (n ? std::sqrt(sumErr2 / n) : 0)
                      << " |err|max=" << maxErr
                      << " P99=" << (n ? errs[static_cast<size_t>(n * 99 / 100)] : 0)
                      << " | within 2px: " << nOk << ", beyond 2px: " << nBad << "\n";
        }

        // (f) row 445 hypothesis-vs-truth matrix
        {
            std::cout << "\n-- row 445 hypothesis-vs-truth matrix --\n";
            const int row = 445;
            std::vector<CurveMapEntry> tmp;
            int cap = 4096;
            tmp.resize(cap);
            int n2 = cmt.EnumerateRow(row, tmp.data(), cap);
            while (n2 < 0) { cap *= 2; tmp.resize(cap); n2 = cmt.EnumerateRow(row, tmp.data(), cap); }
            std::map<int, std::vector<std::pair<float, float>>> byLid;
            for (int q = 0; q < n2; ++q)
                byLid[tmp[static_cast<size_t>(q)].lid].emplace_back(
                    tmp[static_cast<size_t>(q)].uL * rowStep,
                    tmp[static_cast<size_t>(q)].uR * rowStep);
            std::map<int, std::pair<float, float>> truth;
            for (auto& r : calibPairs)
                if (static_cast<int>(std::lround(r.ly / rowStep)) == row)
                    truth[r.lid] = {r.lx, r.rx};
            for (auto& t : truth) {
                std::cout << "  line " << t.first << " trueLx=" << t.second.first
                          << " trueRx=" << t.second.second << " | hypotheses@Lx: ";
                for (auto& bl : byLid) {
                    auto& v = bl.second;
                    int lo = 0, hi = static_cast<int>(v.size());
                    while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < t.second.first) lo = mid + 1; else hi = mid; }
                    float bestDx = 1e9f, pred = -1;
                    for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                        float dx = std::fabs(v[k].first - t.second.first);
                        if (dx < bestDx) { bestDx = dx; pred = v[k].second; }
                    }
                    if (bestDx <= 0.35f) {
                        float nearest = 1e9f; int nl = -1;
                        for (auto& t2 : truth)
                            if (std::fabs(pred - t2.second.second) < nearest) {
                                nearest = std::fabs(pred - t2.second.second); nl = t2.first;
                            }
                        std::cout << "[lid" << bl.first << "->uR" << pred
                                  << (std::fabs(pred - t.second.second) <= 2.f ? "(SELF)" : "")
                                  << (nearest <= 2.f ? "(HIT line" + std::to_string(nl) + ")" : "") << "] ";
                    }
                }
                std::cout << "\n";
            }
        }

        // (f2) reverse query: at given uR, which lines' sweeps pass it and from which xL
        {
            std::vector<CurveMapEntry> rtmp;
            int rcap = 4096;
            rtmp.resize(rcap);
            int rn = cmt.EnumerateRow(445, rtmp.data(), rcap);
            while (rn < 0) { rcap *= 2; rtmp.resize(rcap); rn = cmt.EnumerateRow(445, rtmp.data(), rcap); }
            for (float uRt : {832.78f, 683.56f}) {
                std::cout << "\n-- row 445 reverse query: sweeps crossing uR=" << uRt << " --\n";
                std::map<int, std::pair<float, float>> best;
                for (int q = 0; q < rn; ++q) {
                    const float uR = rtmp[static_cast<size_t>(q)].uR * rowStep;
                    const float uL = rtmp[static_cast<size_t>(q)].uL * rowStep;
                    const float dur = std::fabs(uR - uRt);
                    if (dur > 0.35f) continue;
                    const int lid = rtmp[static_cast<size_t>(q)].lid;
                    auto it = best.find(lid);
                    if (it == best.end() || dur < it->second.first)
                        best[lid] = {dur, uL};
                }
                for (auto& kv : best)
                    std::cout << "    lid=" << kv.first << " uL=" << kv.second.second
                              << " (duR=" << kv.second.first << ")\n";
            }
        }

        // (h) table structure stats + interval exclusion audits
        {
            std::cout << "\n==== table structure & interval-exclusion audit (pose_06) ====\n";
            const double fB = fpx * cmi.baseline;

            std::map<int, long long> cntHist;
            long long totalGrid = 0, contigBad = 0, monoBad = 0;
            std::map<int, std::pair<double, long long>> binAcc;
            for (int row = 0; row < static_cast<int>(cmt.rowCount()); ++row) {
                std::vector<CurveMapEntry> tmp;
                int cap = 4096;
                tmp.resize(cap);
                int n = cmt.EnumerateRow(row, tmp.data(), cap);
                while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
                if (n <= 0) continue;
                std::vector<std::pair<float, std::pair<int, float>>> es;
                es.reserve(n);
                for (int q = 0; q < n; ++q)
                    es.push_back({tmp[static_cast<size_t>(q)].uL * rowStep,
                                  {tmp[static_cast<size_t>(q)].lid,
                                   tmp[static_cast<size_t>(q)].uR * rowStep}});
                std::sort(es.begin(), es.end());
                size_t i = 0;
                while (i < es.size()) {
                    size_t j = i;
                    while (j < es.size() && es[j].first == es[i].first) ++j;
                    const int cnt = static_cast<int>(j - i);
                    ++cntHist[cnt];
                    ++totalGrid;
                    const float uL = es[i].first;
                    const int bin = static_cast<int>(uL / 128.0f);
                    binAcc[bin].first += cnt;
                    binAcc[bin].second += 1;
                    std::map<int, float> lidU;
                    for (size_t k = i; k < j; ++k)
                        lidU.emplace(es[k].second.first, es[k].second.second);
                    if (lidU.size() >= 2) {
                        int prevL = -999; float prevZ = -1;
                        for (auto& lu : lidU) {
                            if (prevL >= 0 && lu.first != prevL + 1) ++contigBad;
                            const double Z = fB / std::max(1e-3, static_cast<double>(uL) - lu.second);
                            if (prevZ >= 0 && Z <= prevZ) ++monoBad;
                            prevL = lu.first; prevZ = static_cast<float>(Z);
                        }
                    }
                    i = j;
                }
            }
            std::cout << "hypothesis-count per (row,xL-grid) over " << totalGrid << " grid points:\n";
            for (auto& kv : cntHist)
                std::cout << "  " << kv.first << " hyp: " << std::fixed << std::setprecision(1)
                          << 100.0 * kv.second / totalGrid << "%\n";
            std::cout << "contiguity violations: " << contigBad
                      << " | depth-monotonicity violations: " << monoBad << "\n";
            std::cout << "mean hypothesis count by xL bin (128px):\n";
            for (auto& kv : binAcc)
                std::cout << "  [" << kv.first * 128 << "," << (kv.first + 1) * 128 << "): "
                          << std::fixed << std::setprecision(2)
                          << kv.second.first / kv.second.second << "\n";

            // exclusion D: depth window (dedup per lid: nearest-uL entry)
            auto passDHist = [&](double w, const char* tag) {
                int tot = 0, one = 0, zero = 0, multi = 0;
                for (auto& r : calibPairs) {
                    const int row = static_cast<int>(std::lround(r.ly / rowStep));
                    std::vector<CurveMapEntry> tmp;
                    int cap = 4096;
                    tmp.resize(cap);
                    int n = cmt.EnumerateRow(row, tmp.data(), cap);
                    while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
                    const double Zb = fB / std::max(1e-3, static_cast<double>(r.lx) - r.rx);
                    std::map<int, std::pair<float, float>> perLid;   // lid -> (|duL|, uR)
                    for (int q = 0; q < n; ++q) {
                        const float uL = tmp[static_cast<size_t>(q)].uL * rowStep;
                        const float uR = tmp[static_cast<size_t>(q)].uR * rowStep;
                        const float dx = std::fabs(uL - r.lx);
                        if (dx > 0.7f) continue;
                        const int lid = tmp[static_cast<size_t>(q)].lid;
                        auto it2 = perLid.find(lid);
                        if (it2 == perLid.end() || dx < it2->second.first)
                            perLid[lid] = {dx, uR};
                    }
                    int c = 0;
                    for (auto& kv2 : perLid) {
                        const double Z = fB / std::max(1e-3, static_cast<double>(r.lx) - kv2.second.second);
                        if (std::fabs(Z - Zb) <= w) ++c;
                    }
                    ++tot;
                    if (c == 1) ++one; else if (c == 0) ++zero; else ++multi;
                }
                std::cout << "depth-window w=" << tag << ": true pairs -> exactly-1 hyp: "
                          << 100.0 * one / tot << "% | 0 hyp: " << 100.0 * zero / tot
                          << "% | >=2 hyp: " << 100.0 * multi / tot << "%\n";
            };
            passDHist(10.0, "10mm");
            passDHist(30.0, "30mm");
            passDHist(50.0, "50mm");

            // exclusion D2: GLOBAL depth center (non-oracle) + center-offset sensitivity
            //                + kill-rate on wrong/noise points
            {
                // global center = median depth of calib pairs
                std::vector<float> depths;
                depths.reserve(calibPairs.size());
                for (auto& r : calibPairs)
                    depths.push_back(static_cast<float>(fB / std::max(1e-3, static_cast<double>(r.lx) - r.rx)));
                std::sort(depths.begin(), depths.end());
                const float Zmed = depths[depths.size() / 2];
                std::cout << "global depth center (calib median) = " << Zmed
                          << "mm  (calib depth range " << depths.front() << ".." << depths.back() << ")\n";

                // classify scan pairs true/wrong (same rule as LR audit)
                std::map<int, std::vector<const MatchPairRec*>> cbr2;
                for (auto& r : calibPairs)
                    cbr2[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
                auto isTruePair = [&](const MatchPairRec* pr) {
                    auto it = cbr2.find(static_cast<int>(std::lround(pr->ly / rowStep)));
                    if (it == cbr2.end()) return false;
                    for (auto* c : it->second)
                        if (std::fabs(c->lx - pr->lx) < 0.6f && std::fabs(c->rx - pr->rx) < 0.5f) return true;
                    return false;
                };

                // per-row table cache: row -> (lid -> sweep entries)
                std::map<int, std::map<int, std::vector<std::pair<float, float>>>> rowSweeps;

                auto countInWindow = [&](int row, float xL, double zc, double w, int& c) {
                    auto it = rowSweeps.find(row);
                    if (it == rowSweeps.end()) {
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
                        rowSweeps[row] = byLid;
                        it = rowSweeps.find(row);
                    }
                    c = 0;
                    for (auto& bl : it->second) {
                        // nearest-uL entry of this lid
                        auto& v = bl.second;
                        int lo = 0, hi = static_cast<int>(v.size());
                        while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < xL) lo = mid + 1; else hi = mid; }
                        float bestDx = 1e9f, uR = -1;
                        for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                            float dx = std::fabs(v[k].first - xL);
                            if (dx < bestDx) { bestDx = dx; uR = v[k].second; }
                        }
                        if (bestDx > 0.7f) continue;
                        const double Z = fB / std::max(1e-3, static_cast<double>(xL) - uR);
                        if (std::fabs(Z - zc) <= w) ++c;
                    }
                };

                // (a) true pairs: exactly-1 / 0 / >=2 with global center, w & offsets
                for (double w : {15.0, 30.0}) {
                    for (double off : {0.0, 10.0, 20.0}) {
                        int tot = 0, one = 0, zero = 0, multi = 0;
                        for (auto& r : calibPairs) {
                            int c = 0;
                            countInWindow(static_cast<int>(std::lround(r.ly / rowStep)), r.lx,
                                          Zmed + off, w, c);
                            ++tot;
                            if (c == 1) ++one; else if (c == 0) ++zero; else ++multi;
                        }
                        std::cout << "global-center w=" << w << " off=" << off
                                  << ": true pairs exactly-1 " << 100.0 * one / tot
                                  << "% | 0 hyp " << 100.0 * zero / tot
                                  << "% | >=2 " << 100.0 * multi / tot << "%\n";
                    }
                }

                // (b) wrong scan pairs: is the WINNING (phantom) hypothesis in-window? (want 0%)
                //     and is the TRUE lid in-window? (recovery potential)
                {
                    int tot = 0, phantomIn = 0, trueIn = 0;
                    for (auto& r : scanPairs) {
                        if (isTruePair(&r)) continue;
                        const int row = static_cast<int>(std::lround(r.ly / rowStep));
                        // winning hypothesis depth at left pt
                        auto& sw = rowSweeps[row];
                        auto wl = sw.find(r.lid);
                        if (wl == sw.end()) continue;
                        auto& v = wl->second;
                        int lo = 0, hi = static_cast<int>(v.size());
                        while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < r.lx) lo = mid + 1; else hi = mid; }
                        float bestDx = 1e9f, uRw = -1;
                        for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                            float dx = std::fabs(v[k].first - r.lx);
                            if (dx < bestDx) { bestDx = dx; uRw = v[k].second; }
                        }
                        const double Zw = fB / std::max(1e-3, static_cast<double>(r.lx) - uRw);
                        ++tot;
                        if (std::fabs(Zw - Zmed) <= 30.0) ++phantomIn;
                        // true lid at same left position
                        auto cit = cbr2.find(row);
                        if (cit != cbr2.end())
                            for (auto* c : cit->second)
                                if (std::fabs(c->lx - r.lx) < 0.6f) {
                                    const double Zt = fB / std::max(1e-3, static_cast<double>(c->lx) - c->rx);
                                    if (std::fabs(Zt - Zmed) <= 30.0) ++trueIn;
                                    break;
                                }
                    }
                    std::cout << "wrong pairs under global w=30: PHANTOM(win-lid) in-window(survives) "
                              << 100.0 * phantomIn / tot << "% -> killed " << 100.0 * (tot - phantomIn) / tot
                              << "% | true lid in-window(recoverable) "
                              << 100.0 * trueIn / tot << "%  (n=" << tot << ")\n";
                }

                // (i) bootstrap center audit: no depth prior, center from pass-1 matches
                {
                    std::cout << "\n-- bootstrap depth-center audit (no prior) --\n";
                    // pass-1 depth from scan pairs (mixed true+wrong)
                    std::vector<float> d1;
                    d1.reserve(scanPairs.size());
                    for (auto& r : scanPairs)
                        d1.push_back(static_cast<float>(fB / std::max(1e-3, static_cast<double>(r.lx) - r.rx)));
                    std::vector<float> ds = d1;
                    std::sort(ds.begin(), ds.end());
                    const float Zboot = ds[ds.size() / 2];
                    // truth center for comparison
                    std::cout << "bootstrap center (pass-1 median, n=" << d1.size() << ") = "
                              << Zboot << "mm  vs truth center 254.54mm"
                              << "  -> error " << (Zboot - 254.54f) << "mm\n";
                    // robustness: median of TRUE-only matches for reference
                    {
                        std::vector<float> dt;
                        for (auto& r : scanPairs)
                            if (isTruePair(&r))
                                dt.push_back(static_cast<float>(fB / std::max(1e-3, static_cast<double>(r.lx) - r.rx)));
                        std::sort(dt.begin(), dt.end());
                        std::cout << "  (true-only median = " << dt[dt.size() / 2]
                                  << "mm, true range " << dt.front() << ".." << dt.back() << ")\n";
                    }
                    // metrics with bootstrap global center w=30
                    {
                        int tot = 0, one = 0, zero = 0, multi = 0;
                        for (auto& r : calibPairs) {
                            int c = 0;
                            countInWindow(static_cast<int>(std::lround(r.ly / rowStep)), r.lx,
                                          Zboot, 30.0, c);
                            ++tot;
                            if (c == 1) ++one; else if (c == 0) ++zero; else ++multi;
                        }
                        std::cout << "BOOTSTRAP global w=30: true pairs exactly-1 "
                                  << 100.0 * one / tot << "% | 0 hyp " << 100.0 * zero / tot
                                  << "% | >=2 " << 100.0 * multi / tot << "%\n";
                    }
                    // phantom kill with bootstrap center
                    {
                        int tot = 0, phantomIn = 0;
                        for (auto& r : scanPairs) {
                            if (isTruePair(&r)) continue;
                            const int row = static_cast<int>(std::lround(r.ly / rowStep));
                            auto& sw = rowSweeps[row];
                            auto wl = sw.find(r.lid);
                            if (wl == sw.end()) continue;
                            auto& v = wl->second;
                            int lo = 0, hi = static_cast<int>(v.size());
                            while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < r.lx) lo = mid + 1; else hi = mid; }
                            float bestDx = 1e9f, uRw = -1;
                            for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                                float dx = std::fabs(v[k].first - r.lx);
                                if (dx < bestDx) { bestDx = dx; uRw = v[k].second; }
                            }
                            const double Zw = fB / std::max(1e-3, static_cast<double>(r.lx) - uRw);
                            ++tot;
                            if (std::fabs(Zw - Zboot) <= 30.0) ++phantomIn;
                        }
                        std::cout << "BOOTSTRAP phantom kill rate = "
                                  << 100.0 * (tot - phantomIn) / tot << "% (n=" << tot << ")\n";
                    }
                    // per-row-band local centers (handle tilt): band = 128 rows
                    {
                        std::map<int, std::vector<float>> bandD;
                        for (auto& r : scanPairs) {
                            const int row = static_cast<int>(std::lround(r.ly / rowStep));
                            bandD[row / 128].push_back(
                                static_cast<float>(fB / std::max(1e-3, static_cast<double>(r.lx) - r.rx)));
                        }
                        std::map<int, float> bandC;
                        for (auto& kv : bandD) {
                            auto v = kv.second;
                            std::sort(v.begin(), v.end());
                            bandC[kv.first] = v[v.size() / 2];
                        }
                        int tot = 0, one = 0, zero = 0, multi = 0;
                        for (auto& r : calibPairs) {
                            const int row = static_cast<int>(std::lround(r.ly / rowStep));
                            auto it = bandC.find(row / 128);
                            if (it == bandC.end()) continue;
                            int c = 0;
                            countInWindow(row, r.lx, it->second, 15.0, c);
                            ++tot;
                            if (c == 1) ++one; else if (c == 0) ++zero; else ++multi;
                        }
                        std::cout << "BOOTSTRAP band-local w=15: true pairs exactly-1 "
                                  << 100.0 * one / tot << "% | 0 hyp " << 100.0 * zero / tot
                                  << "% | >=2 " << 100.0 * multi / tot << "%\n";
                    }
                }

                // (c) noise interp left pts (no calib truth): fraction with 0 in-window
                {
                    std::map<int, std::vector<const MatchPairRec*>> scanByRow;
                    for (auto& r : scanPairs)
                        scanByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
                    int tot = 0, zero = 0;
                    for (auto& p : sL) {
                        const int row = static_cast<int>(std::lround(p.y / rowStep));
                        // has truth?
                        bool hasT = false;
                        auto cit = cbr2.find(row);
                        if (cit != cbr2.end())
                            for (auto* c : cit->second)
                                if (std::fabs(c->lx - p.x) < 0.6f) { hasT = true; break; }
                        if (hasT) continue;
                        int c = 0;
                        countInWindow(row, p.x, Zmed, 30.0, c);
                        ++tot;
                        if (c == 0) ++zero;
                    }
                    std::cout << "noise interp pts under global w=30: 0-in-window(excluded) "
                              << 100.0 * zero / std::max(1, tot) << "%  (n=" << tot << ")\n";
                }
            }

            // exclusion O: row-monotone lid assignment via DP (LIS-style, skips allowed)
            {
                std::map<int, std::vector<float>> lbr;
                for (auto& p : sL)
                    lbr[static_cast<int>(std::lround(p.y / rowStep))].push_back(p.x);
                std::map<int, std::vector<const MatchPairRec*>> cbr;
                for (auto& r : calibPairs)
                    cbr[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
                int withTruth = 0, correct = 0, wrongAssign = 0, unassigned = 0;
                int noisePts = 0, noiseAssigned = 0;
                for (auto& kv : lbr) {
                    const int row = kv.first;
                    auto& pts = kv.second;
                    std::sort(pts.begin(), pts.end());
                    std::vector<std::vector<int>> cand(pts.size());
                    {
                        std::vector<CurveMapEntry> tmp;
                        int cap = 4096;
                        tmp.resize(cap);
                        int n = cmt.EnumerateRow(row, tmp.data(), cap);
                        while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
                        for (size_t pi = 0; pi < pts.size(); ++pi) {
                            std::set<int> s;
                            for (int q = 0; q < n; ++q)
                                if (std::fabs(tmp[static_cast<size_t>(q)].uL * rowStep - pts[pi]) <= 0.7f)
                                    s.insert(tmp[static_cast<size_t>(q)].lid);
                            cand[pi].assign(s.begin(), s.end());
                        }
                    }
                    const int M = static_cast<int>(pts.size());
                    const int MAXL = 32;
                    // dp[i][l] = max assigned among pts[0..i] with pts[i] assigned lid l
                    std::vector<std::vector<int>> dp(M, std::vector<int>(MAXL + 1, -1));
                    std::vector<std::vector<int>> pa(M, std::vector<int>(MAXL + 1, -1));
                    // bestBelow[l] = max dp over j<i with lid l, and its (j)
                    std::vector<int> bestBelow(MAXL + 2, -1), bestBelowIdx(MAXL + 2, -1);
                    for (int i = 0; i < M; ++i) {
                        for (int l : cand[i]) {
                            if (l > MAXL) continue;
                            int best = 0, bp = -1;   // option: skip all previous
                            int pref = -1, prefIdx = -1;
                            for (int l2 = 1; l2 < l; ++l2)
                                if (bestBelow[l2] > pref) { pref = bestBelow[l2]; prefIdx = bestBelowIdx[l2]; }
                            if (pref >= 0 && pref + 1 > best) { best = pref + 1; bp = prefIdx; }
                            dp[i][l] = best;
                            pa[i][l] = bp;
                        }
                        for (int l = 1; l <= MAXL; ++l)
                            if (dp[i][l] > bestBelow[l]) { bestBelow[l] = dp[i][l]; bestBelowIdx[l] = i; }
                    }
                    // backtrack best
                    int bi = -1, bl = -1, bc = 0;
                    for (int i = 0; i < M; ++i)
                        for (int l = 1; l <= MAXL; ++l)
                            if (dp[i][l] > bc) { bc = dp[i][l]; bi = i; bl = l; }
                    std::vector<int> assign(M, -1);
                    while (bi >= 0 && bl > 0) {
                        assign[bi] = bl;
                        int p = pa[bi][bl];
                        // predecessor's lid unknown during backtrack via bestBelow composite... recompute:
                        // fallback: mark chain by scanning predecessor dp consistency
                        if (p < 0) break;
                        int nl = -1;
                        for (int l2 = 1; l2 < bl; ++l2)
                            if (dp[p][l2] == dp[bi][bl] - 1) { nl = l2; break; }
                        if (nl < 0) break;
                        bi = p; bl = nl;
                    }
                    auto cit = cbr.find(row);
                    for (int pi = 0; pi < M; ++pi) {
                        const MatchPairRec* t = nullptr;
                        if (cit != cbr.end())
                            for (auto* c : cit->second)
                                if (std::fabs(c->lx - pts[pi]) < 0.6f) { t = c; break; }
                        if (t) {
                            ++withTruth;
                            if (assign[pi] < 0) ++unassigned;
                            else if (assign[pi] == t->lid) ++correct;
                            else ++wrongAssign;
                        } else {
                            ++noisePts;
                            if (assign[pi] >= 0) ++noiseAssigned;
                        }
                    }
                }
                std::cout << "row-monotone DP lid assignment: with-truth pts=" << withTruth
                          << " correct=" << std::fixed << std::setprecision(1)
                          << 100.0 * correct / withTruth << "%"
                          << " wrong=" << 100.0 * wrongAssign / withTruth << "%"
                          << " unassigned=" << 100.0 * unassigned / withTruth << "%"
                          << " | noise pts=" << noisePts
                          << " noise-assigned=" << 100.0 * noiseAssigned / std::max(1, noisePts) << "%\n";
            }
        }

        // (j) scheme simulation: zigzag two-phase deferred matching (+depth variants)
        {
            std::cout << "\n==== scheme simulation (pose_06) ====\n";
            const double fB2 = fpx * cmi.baseline;

            std::map<int, std::vector<const MatchPairRec*>> truthByRow;
            for (auto& r : calibPairs)
                truthByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);

            std::map<int, std::vector<float>> lRow, rRow;
            for (auto& p : sL) lRow[static_cast<int>(std::lround(p.y / rowStep))].push_back(p.x);
            for (auto& p : sR) rRow[static_cast<int>(std::lround(p.y / rowStep))].push_back(p.x);
            for (auto& kv : lRow) std::sort(kv.second.begin(), kv.second.end());
            for (auto& kv : rRow) std::sort(kv.second.begin(), kv.second.end());

            struct RowData {
                std::vector<float> lx, rx;
                std::vector<std::vector<std::pair<float, int>>> cand;
            };
            std::map<int, RowData> RD;
            for (auto& kv : lRow) {
                const int row = kv.first;
                RowData d;
                d.lx = kv.second;
                auto rit = rRow.find(row);
                if (rit != rRow.end()) d.rx = rit->second;
                std::vector<std::pair<float, std::pair<int, float>>> es;
                {
                    std::vector<CurveMapEntry> tmp;
                    int cap = 4096;
                    tmp.resize(cap);
                    int n = cmt.EnumerateRow(row, tmp.data(), cap);
                    while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
                    es.reserve(n);
                    for (int q = 0; q < n; ++q)
                        es.push_back({tmp[static_cast<size_t>(q)].uL * rowStep,
                                      {tmp[static_cast<size_t>(q)].lid,
                                       tmp[static_cast<size_t>(q)].uR * rowStep}});
                    std::sort(es.begin(), es.end());
                }
                d.cand.resize(d.lx.size());
                for (size_t i = 0; i < d.lx.size(); ++i) {
                    const float x = d.lx[i];
                    size_t lo = 0, hi = es.size();
                    while (lo < hi) { size_t mid = (lo + hi) / 2; if (es[mid].first < x) lo = mid + 1; else hi = mid; }
                    std::set<long long> seen;
                    auto& out = d.cand[i];
                    auto tryAdd = [&](size_t idx) -> bool {
                        if (std::fabs(es[idx].first - x) > 0.7f) return false;
                        const long long key = (static_cast<long long>(es[idx].second.first) << 20)
                                            | static_cast<long long>(std::lround(es[idx].second.second / rowStep));
                        if (!seen.count(key) && out.size() < 64) {
                            out.push_back({es[idx].second.second, es[idx].second.first});
                            seen.insert(key);
                        }
                        return true;
                    };
                    for (size_t p2 = lo; p2 < es.size(); ++p2) if (!tryAdd(p2)) break;
                    for (size_t p2 = lo; p2-- > 0;) if (!tryAdd(p2)) break;
                }
                RD[row] = std::move(d);
            }

            auto evalHits = [](const RowData& d, const std::vector<char>& occ, size_t li,
                               std::vector<int>& hits, float thr,
                               const std::set<int>* locked, std::vector<int>& hitLids) {
                hits.clear();
                hitLids.clear();
                int hitSm = -1;
                for (size_t c = 0; c < d.cand[li].size() && hits.size() < 2; ++c) {
                    if (locked && locked->count(d.cand[li][c].second)) continue;
                    const float lov = d.cand[li][c].first - thr;
                    const float hiv = d.cand[li][c].first + thr;
                    const int sB = static_cast<int>(std::lower_bound(d.rx.begin(), d.rx.end(), lov) - d.rx.begin());
                    const int eB = static_cast<int>(std::upper_bound(d.rx.begin(), d.rx.end(), hiv) - d.rx.begin());
                    for (int ri = sB; ri < eB && hits.size() < 2; ++ri) {
                        if (occ[static_cast<size_t>(ri)]) continue;
                        if (ri == hitSm) continue;
                        hits.push_back(ri);
                        hitSm = ri;
                        hitLids.push_back(d.cand[li][c].second);
                    }
                }
            };

            // variant: 0=greedy L2R single-phase; 1=zigzag two-phase;
            //          2=1 + phase2 depth tiebreak; (3/4 post-filter applied outside)
            auto runScheme = [&](int variant, float thr, bool hardLock) -> std::vector<std::pair<int, std::pair<float, float>>> {
                std::vector<std::pair<int, std::pair<float, float>>> pairs;   // (row, (lx, rx))
                std::vector<int> hits;
                for (auto& kv : RD) {
                    const int row = kv.first;
                    RowData& d = const_cast<RowData&>(kv.second);
                    std::vector<char> occ(d.rx.size(), 0), state(d.lx.size(), 0); // 0 pending 1 matched 2 dead 3 deferred
                    std::vector<float> rowZ;
                    std::set<int> locked;
                    std::vector<int> hitLids;
                    auto doMatch = [&](size_t li, int ri, int lid) {
                        occ[static_cast<size_t>(ri)] = 1;
                        state[li] = 1;
                        if (hardLock) locked.insert(lid);
                        pairs.push_back({row, {d.lx[li], d.rx[static_cast<size_t>(ri)]}});
                        rowZ.push_back(static_cast<float>(fB2 / std::max(1e-3, static_cast<double>(d.lx[li]) - d.rx[static_cast<size_t>(ri)])));
                    };
                    if (variant == 0) {
                        for (size_t i = 0; i < d.lx.size(); ++i) {
                            evalHits(d, occ, i, hits, thr, hardLock ? &locked : nullptr, hitLids);
                            if (hits.size() == 1) doMatch(i, hits[0], hitLids[0]);
                            else state[i] = 2;
                        }
                    } else {
                        std::vector<size_t> order;
                        {
                            size_t a = 0, b = d.lx.size();
                            bool left = true;
                            while (a < b) {
                                if (left) { order.push_back(a++); }
                                else { order.push_back(--b); }
                                left = !left;
                            }
                        }
                        for (size_t i : order) {
                            evalHits(d, occ, i, hits, thr, hardLock ? &locked : nullptr, hitLids);
                            if (hits.size() == 1) doMatch(i, hits[0], hitLids[0]);
                            else if (hits.empty()) state[i] = 2;
                            else state[i] = 3;
                        }
                        for (size_t i = 0; i < d.lx.size(); ++i) {
                            if (state[i] != 3) continue;
                            evalHits(d, occ, i, hits, thr, hardLock ? &locked : nullptr, hitLids);
                            if (hits.size() == 1) { doMatch(i, hits[0], hitLids[0]); continue; }
                            if (hits.empty()) { state[i] = 2; continue; }
                            if (variant >= 2) {
                                std::vector<float> zs = rowZ;
                                std::sort(zs.begin(), zs.end());
                                const float med = zs.empty() ? 260.0f : zs[zs.size() / 2];
                                float z0 = static_cast<float>(fB2 / std::max(1e-3, static_cast<double>(d.lx[i]) - d.rx[static_cast<size_t>(hits[0])]));
                                float z1 = static_cast<float>(fB2 / std::max(1e-3, static_cast<double>(d.lx[i]) - d.rx[static_cast<size_t>(hits[1])]));
                                int pick = std::fabs(z0 - med) <= std::fabs(z1 - med) ? hits[0] : hits[1];
                                doMatch(i, pick, pick == hits[0] ? hitLids[0] : hitLids[1]);
                            } else {
                                state[i] = 2;
                            }
                        }
                    }
                }
                return pairs;
            };

            auto statsOf = [&](const char* tag, std::vector<std::pair<int, std::pair<float, float>>>& pairs,
                               bool postDepthFilter) {
                if (postDepthFilter) {
                    std::vector<float> zs;
                    zs.reserve(pairs.size());
                    for (auto& pr : pairs)
                        zs.push_back(static_cast<float>(fB2 / std::max(1e-3, static_cast<double>(pr.second.first) - pr.second.second)));
                    std::vector<float> zs2 = zs;
                    std::sort(zs2.begin(), zs2.end());
                    const float med = zs2[zs2.size() / 2];
                    std::vector<std::pair<int, std::pair<float, float>>> keep;
                    keep.reserve(pairs.size());
                    for (size_t i = 0; i < pairs.size(); ++i)
                        if (std::fabs(zs[i] - med) <= 30.0f) keep.push_back(pairs[i]);
                    pairs = keep;
                }
                int matched = static_cast<int>(pairs.size()), correct = 0;
                for (auto& pr : pairs) {
                    auto it = truthByRow.find(pr.first);
                    if (it == truthByRow.end()) continue;
                    for (auto* c : it->second)
                        if (std::fabs(c->lx - pr.second.first) < 0.6f &&
                            std::fabs(c->rx - pr.second.second) < 0.5f) { ++correct; break; }
                }
                std::cout << tag << ": matched=" << matched << " correct=" << correct
                          << " wrong=" << (matched - correct)
                          << " lost=" << (static_cast<int>(calibPairs.size()) - correct) << "\n";
            };

            std::cout << "GPU reference: matched=13244 correct=10680 wrong=2564 lost=9092 | calib total="
                      << calibPairs.size() << "\n";
            auto v0 = runScheme(0, 2.0f, false); statsOf("V0 greedy-L2R           (fidelity check) ", v0, false);
            auto v1 = runScheme(1, 2.0f, false); statsOf("V1 zigzag+defer         (no lid lock)    ", v1, false);
            auto v5 = runScheme(1, 2.0f, true);  statsOf("V5 zigzag+defer+HARD-LID-LOCK            ", v5, false);
            auto v2 = runScheme(2, 2.0f, false); statsOf("V2 V1+depth-tiebreak                      ", v2, false);
            auto v6 = runScheme(2, 2.0f, true);  statsOf("V6 hard-lock+depth-tiebreak               ", v6, false);
            auto v3 = runScheme(1, 2.0f, false); statsOf("V3 V1+post-filter                         ", v3, true);
            auto v4 = runScheme(2, 2.0f, false); statsOf("V4 V2+post-filter                         ", v4, true);
            auto v7 = runScheme(2, 2.0f, true);  statsOf("V7 hard-lock+depth+post-filter            ", v7, true);

            // V8: lid-set constraint propagation (CSP naked-single)
            long long v8ChainMatches = 0;
            int v8MaxRound = 0;
            //     V8a: user rule (right-uniqueness only)
            //     V8b: + left-exclusivity (bidirectional uniqueness)
            //     depthLeftover: adjudicate fixpoint leftovers by row-median depth
            auto runV8 = [&](bool bidirectional, bool depthLeftover) -> std::vector<std::pair<int, std::pair<float, float>>> {
                v8ChainMatches = 0;
                v8MaxRound = 0;
                std::vector<std::pair<int, std::pair<float, float>>> pairs;
                for (auto& kv : RD) {
                    const int row = kv.first;
                    RowData& d = const_cast<RowData&>(kv.second);
                    const size_t NL = d.lx.size(), NR = d.rx.size();
                    std::vector<char> occ(NR, 0), ls(NL, 0);   // ls: 0 pending 1 matched 2 dead
                    struct Cand { int lid; int ri; };
                    std::vector<std::vector<Cand>> cd(NL);
                    for (size_t i = 0; i < NL; ++i) {
                        for (size_t c = 0; c < d.cand[i].size(); ++c) {
                            const float lov = d.cand[i][c].first - 2.0f;
                            const float hiv = d.cand[i][c].first + 2.0f;
                            const int sB = static_cast<int>(std::lower_bound(d.rx.begin(), d.rx.end(), lov) - d.rx.begin());
                            const int eB = static_cast<int>(std::upper_bound(d.rx.begin(), d.rx.end(), hiv) - d.rx.begin());
                            for (int ri = sB; ri < eB; ++ri)
                                cd[i].push_back({d.cand[i][c].second, ri});
                        }
                        if (cd[i].empty()) ls[i] = 2;
                    }
                    std::vector<float> rowZ;
                    auto doMatch = [&](size_t i, int ri) {
                        occ[static_cast<size_t>(ri)] = 1;
                        ls[i] = 1;
                        pairs.push_back({row, {d.lx[i], d.rx[static_cast<size_t>(ri)]}});
                        rowZ.push_back(static_cast<float>(fB2 / std::max(1e-3, static_cast<double>(d.lx[i]) - d.rx[static_cast<size_t>(ri)])));
                    };
                    bool prog = true;
                    int round = 0;
                    while (prog) {
                        prog = false;
                        ++round;
                        // 1) void candidates pointing to occupied right points
                        for (size_t i = 0; i < NL; ++i) {
                            if (ls[i] != 0) continue;
                            auto& v = cd[i];
                            size_t w = 0;
                            for (size_t k = 0; k < v.size(); ++k)
                                if (!occ[static_cast<size_t>(v[k].ri)]) v[w++] = v[k];
                            v.resize(w);
                            if (v.empty()) { ls[i] = 2; prog = true; }
                        }
                        // 2) naked single: left point with exactly one candidate
                        for (size_t i = 0; i < NL; ++i) {
                            if (ls[i] != 0 || cd[i].size() != 1) continue;
                            const int L = cd[i][0].lid;
                            const int R = cd[i][0].ri;
                            std::set<int> holders;
                            int claimants = 0;
                            for (size_t j = 0; j < NL; ++j) {
                                if (ls[j] != 0) continue;
                                bool has = false;
                                for (auto& cc : cd[j])
                                    if (cc.lid == L) { holders.insert(cc.ri); has = true; }
                                if (has) ++claimants;
                            }
                            if (holders.size() != 1 || *holders.begin() != R) continue;
                            if (bidirectional && claimants != 1) continue;
                            if (round > 1) { ++v8ChainMatches; }
                            if (round > v8MaxRound) v8MaxRound = round;
                            doMatch(i, R);
                            for (size_t j = 0; j < NL; ++j) {
                                if (ls[j] != 0) continue;
                                auto& v = cd[j];
                                size_t w = 0;
                                for (size_t k = 0; k < v.size(); ++k)
                                    if (v[k].lid != L) v[w++] = v[k];
                                v.resize(w);
                                if (v.empty()) ls[j] = 2;
                            }
                            prog = true;
                            break;
                        }
                    }
                    // 3) leftovers: depth adjudication (row median)
                    if (depthLeftover) {
                        for (size_t i = 0; i < NL; ++i) {
                            if (ls[i] != 0 || cd[i].empty()) continue;
                            std::vector<float> zs = rowZ;
                            std::sort(zs.begin(), zs.end());
                            const float med = zs.empty() ? 260.0f : zs[zs.size() / 2];
                            int bestRi = -1;
                            float bestD = 1e9f;
                            for (auto& cc : cd[i]) {
                                const float z = static_cast<float>(fB2 / std::max(1e-3, static_cast<double>(d.lx[i]) - d.rx[static_cast<size_t>(cc.ri)]));
                                const float dd = std::fabs(z - med);
                                if (dd < bestD) { bestD = dd; bestRi = cc.ri; }
                            }
                            if (bestRi >= 0) doMatch(i, bestRi);
                        }
                    }
                }
                return pairs;
            };
            auto v8a = runV8(false, false); statsOf("V8a CSP(user rule, right-unique only)      ", v8a, false);
            std::cout << "   V8a propagation: chain-triggered matches (round>=2) = " << v8ChainMatches
                      << ", max rounds = " << v8MaxRound << "\n";
            auto v8b = runV8(true, false);  statsOf("V8b CSP bidirectional uniqueness           ", v8b, false);
            std::cout << "   V8b propagation: chain-triggered matches (round>=2) = " << v8ChainMatches
                      << ", max rounds = " << v8MaxRound << "\n";
            auto v8c = runV8(true, true);   statsOf("V8c V8b + depth-leftover                   ", v8c, false);
            auto v8d = runV8(true, true);   statsOf("V8d V8c + post-filter                      ", v8d, true);
            {
                auto renderCloud = [&](const std::string& name,
                                       std::vector<std::pair<int, std::pair<float, float>>>& pairs) {
                    const int n = static_cast<int>(pairs.size());
                    if (n <= 0) return;
                    cv::Mat L(1, n, CV_32FC2), R(1, n, CV_32FC2), I(1, n, CV_32SC1);
                    std::vector<char> ok(n, 0);
                    for (int k = 0; k < n; ++k) {
                        const float y = pairs[static_cast<size_t>(k)].first * rowStep;
                        const float lx = pairs[static_cast<size_t>(k)].second.first;
                        const float rx = pairs[static_cast<size_t>(k)].second.second;
                        L.at<cv::Vec2f>(k) = cv::Vec2f(lx, y);
                        R.at<cv::Vec2f>(k) = cv::Vec2f(rx, y);
                        I.at<int>(k) = 0;
                        auto it = truthByRow.find(pairs[static_cast<size_t>(k)].first);
                        if (it != truthByRow.end())
                            for (auto* c : it->second)
                                if (std::fabs(c->lx - lx) < 0.6f && std::fabs(c->rx - rx) < 0.5f) {
                                    ok[static_cast<size_t>(k)] = 1; break;
                                }
                    }
                    cv::cuda::GpuMat dL, dR, dI;
                    dL.upload(L);
                    dR.upload(R);
                    dI.upload(I);
                    auto rr = recon.Execute(dL, dR, dI, h.Q, stream);
                    if (!rr.success || !rr.d_points3d) {
                        std::cout << "renderCloud " << name << ": recon failed\n";
                        return;
                    }
                    cv::Mat P3;
                    rr.d_points3d->download(P3);
                    stream.waitForCompletion();
                    P3 = P3.reshape(3, 1);
                    std::ofstream f((outDir / name).string());
                    f << "ply\nformat ascii 1.0\nelement vertex " << n << "\n"
                      << "property float x\nproperty float y\nproperty float z\n"
                      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                      << std::fixed << std::setprecision(3);
                    int nOk = 0;
                    for (int k = 0; k < n; ++k) {
                        const cv::Vec3f p = P3.at<cv::Vec3f>(k);
                        nOk += ok[static_cast<size_t>(k)];
                        f << p[0] << " " << p[1] << " " << p[2] << " "
                          << (ok[static_cast<size_t>(k)] ? "0 200 0" : "255 0 0") << "\n";
                    }
                    std::cout << "cloud " << name << ": " << n << " pts (green=" << nOk
                              << " red=" << (n - nOk) << ")\n";
                };
                renderCloud("pose_06_simV0_greedy.ply", v0);
                renderCloud("pose_06_simV4_best.ply", v4);
                renderCloud("pose_06_simV7_hardlock.ply", v7);
            }
        }

        // (k) table correctness verification: labeled calib interp as ground truth
        //     for each calib left point (row, xL, true lid): table-predict uR,
        //     compare with actual right point of the SAME lid; also identity check
        //     (nearest right point to prediction should carry the same lid)
        {
            std::cout << "\n==== table verification via labeled interp (pose_06) ====\n";
            std::vector<cv::Point2f> cL2, cR2;
            std::vector<int> cLi, cRi;
            runCalib(0, 0, &cL2, &cR2, &cLi, &cRi);

            // right points by row: (x, lid)
            std::map<int, std::vector<std::pair<float, int>>> rByRow;
            for (size_t i = 0; i < cR2.size(); ++i)
                rByRow[static_cast<int>(std::lround(cR2[i].y / rowStep))].push_back({cR2[i].x, cRi[i]});
            for (auto& kv : rByRow) std::sort(kv.second.begin(), kv.second.end());

            // row table cache: row -> lid -> sweep (uL, uR) sorted by uL
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

            int nTot = 0, nNoEntry = 0, nNoRight = 0, nOk = 0, idMiss = 0;
            double sumE = 0, sumE2 = 0;
            float maxE = 0;
            std::vector<float> errs;
            std::map<int, std::pair<double, int>> byLidErr;
            for (size_t i = 0; i < cL2.size(); ++i) {
                const int row = static_cast<int>(std::lround(cL2[i].y / rowStep));
                const float lx = cL2[i].x;
                const int lid = cLi[i];
                ++nTot;
                auto& tab = getTab(row);
                auto it = tab.find(lid);
                if (it == tab.end() || it->second.empty()) { ++nNoEntry; continue; }
                auto& v = it->second;
                int lo = 0, hi = static_cast<int>(v.size());
                while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < lx) lo = mid + 1; else hi = mid; }
                float bestDx = 1e9f, pred = -1;
                for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                    float dx = std::fabs(v[k].first - lx);
                    if (dx < bestDx) { bestDx = dx; pred = v[k].second; }
                }
                if (bestDx > 0.7f) { ++nNoEntry; continue; }
                // actual right point of same lid at this row
                auto rit = rByRow.find(row);
                if (rit == rByRow.end()) { ++nNoRight; continue; }
                auto& rv = rit->second;
                float bestSame = 1e9f, rxSame = -1;
                float bestAny = 1e9f; int lidAny = -1; float rxAny = -1;
                for (auto& e : rv) {
                    float dx = std::fabs(e.first - pred);
                    if (dx < bestAny) { bestAny = dx; lidAny = e.second; rxAny = e.first; }
                    if (e.second == lid && dx < bestSame) { bestSame = dx; rxSame = e.first; }
                }
                if (rxSame < 0 || bestSame > 3.0f) { ++nNoRight; continue; }
                if (lidAny != lid) ++idMiss;
                const float err = pred - rxSame;
                ++nOk;
                sumE += err; sumE2 += static_cast<double>(err) * err;
                maxE = std::max(maxE, std::fabs(err));
                errs.push_back(err);
                byLidErr[lid].first += err;
                byLidErr[lid].second += 1;
            }
            std::sort(errs.begin(), errs.end());
            std::cout << "left pts=" << nTot << " no-table-entry=" << nNoEntry
                      << " no-same-lid-right=" << nNoRight << " compared=" << nOk
                      << " | identity-miss(nearest-right-lid != true-lid): " << idMiss << "\n";
            std::cout << "pred err (vs same-lid actual right): mean=" << (nOk ? sumE / nOk : 0)
                      << " rms=" << (nOk ? std::sqrt(sumE2 / nOk) : 0)
                      << " P50=" << errs[errs.size() / 2]
                      << " P99=" << errs[errs.size() * 99 / 100]
                      << " |err|max=" << maxE
                      << " within1px=" << 100.0 * (std::count_if(errs.begin(), errs.end(),
                           [](float e) { return std::fabs(e) <= 1.0f; })) / errs.size()
                      << "% within2px=" << 100.0 * (std::count_if(errs.begin(), errs.end(),
                           [](float e) { return std::fabs(e) <= 2.0f; })) / errs.size() << "%\n";
            std::cout << "per-lid mean err (px):\n";
            for (auto& kv : byLidErr)
                std::cout << "  lid=" << kv.first << " n=" << kv.second.second
                          << " mean=" << std::fixed << std::setprecision(3)
                          << kv.second.first / kv.second.second << "\n";
        }

        // (l) hypothesis separation: how often do two different lines' predicted uR
        //     land within +-2px of each other (window collision), and how often is a
        //     phantom hypothesis armed on another line's actual right point
        {
            std::cout << "\n==== hypothesis-separation audit (pose_06) ====\n";
            // (a) over all multi-hypothesis grid points of the table
            long long gridMulti = 0, g2 = 0, g4 = 0, g8 = 0;
            std::vector<float> gaps;
            gaps.reserve(1 << 20);
            for (int row = 0; row < static_cast<int>(cmt.rowCount()); ++row) {
                std::vector<CurveMapEntry> tmp;
                int cap = 4096;
                tmp.resize(cap);
                int n = cmt.EnumerateRow(row, tmp.data(), cap);
                while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
                if (n <= 0) continue;
                std::vector<std::pair<float, std::pair<int, float>>> es;
                for (int q = 0; q < n; ++q)
                    es.push_back({tmp[static_cast<size_t>(q)].uL * rowStep,
                                  {tmp[static_cast<size_t>(q)].lid,
                                   tmp[static_cast<size_t>(q)].uR * rowStep}});
                std::sort(es.begin(), es.end());
                size_t i = 0;
                while (i < es.size()) {
                    size_t j = i;
                    while (j < es.size() && es[j].first == es[i].first) ++j;
                    std::map<int, float> lu;   // lid -> first uR
                    for (size_t k2 = i; k2 < j; ++k2) lu.emplace(es[k2].second.first, es[k2].second.second);
                    if (lu.size() >= 2) {
                        ++gridMulti;
                        float mg = 1e9f;
                        auto p1 = lu.begin();
                        auto p2 = std::next(p1);
                        for (; p2 != lu.end(); ++p1, ++p2) {
                            const float g = p2->second - p1->second;
                            gaps.push_back(g);
                            mg = std::min(mg, g);
                        }
                        if (mg <= 2.0f) ++g2;
                        if (mg <= 4.0f) ++g4;
                        if (mg <= 8.0f) ++g8;
                    }
                    i = j;
                }
            }
            {
                std::vector<float> gs = gaps;
                std::sort(gs.begin(), gs.end());
                std::cout << "(a) adjacent-hypothesis uR gap distribution over " << gs.size()
                          << " gaps: min=" << gs.front()
                          << " P1=" << gs[gs.size() / 100]
                          << " P5=" << gs[gs.size() / 20]
                          << " P25=" << gs[gs.size() / 4]
                          << " P50=" << gs[gs.size() / 2]
                          << " P75=" << gs[gs.size() * 3 / 4]
                          << " P95=" << gs[gs.size() * 95 / 100]
                          << " max=" << gs.back() << "\n";
            }
            std::cout << "(a) table grid pts with >=2 hyps: " << gridMulti
                      << " | min adjacent-uR gap <=2px: " << 100.0 * g2 / gridMulti
                      << "% | <=4px: " << 100.0 * g4 / gridMulti
                      << "% | <=8px: " << 100.0 * g8 / gridMulti << "%\n";

            // (a2)+(b) at calib left points (matcher-style candidate set)
            std::vector<cv::Point2f> cL2, cR2;
            std::vector<int> cLi, cRi;
            runCalib(0, 0, &cL2, &cR2, &cLi, &cRi);
            std::map<int, std::vector<std::pair<float, int>>> rByRow;
            for (size_t i = 0; i < cR2.size(); ++i)
                rByRow[static_cast<int>(std::lround(cR2[i].y / rowStep))].push_back({cR2[i].x, cRi[i]});
            for (auto& kv : rByRow) std::sort(kv.second.begin(), kv.second.end());

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

            int nPts = 0, ptsGap2 = 0, ptsArmed = 0, ptsArmedDistinct = 0;
            long long hypTot = 0, hypArmed = 0;
            for (size_t i = 0; i < cL2.size(); ++i) {
                const int row = static_cast<int>(std::lround(cL2[i].y / rowStep));
                const float lx = cL2[i].x;
                const int tlid = cLi[i];
                auto& tab = getTab(row);
                // matcher-style candidates: entries of all lids within 0.7 of lx, dedup (lid, uRg)
                std::map<long long, std::pair<float, int>> cand;
                float predTrue = -1;
                for (auto& bl : tab) {
                    auto& v = bl.second;
                    int lo = 0, hi = static_cast<int>(v.size());
                    while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < lx) lo = mid + 1; else hi = mid; }
                    for (int k2 = std::max(0, lo - 1); k2 <= std::min(lo, static_cast<int>(v.size()) - 1); ++k2) {
                        if (std::fabs(v[k2].first - lx) > 0.7f) continue;
                        const long long key = (static_cast<long long>(bl.first) << 20)
                                            | static_cast<long long>(std::lround(v[k2].second / rowStep));
                        auto it2 = cand.find(key);
                        if (it2 == cand.end()) cand[key] = {v[k2].second, bl.first};
                    }
                }
                if (cand.size() < 2) continue;
                ++nPts;
                hypTot += static_cast<long long>(cand.size());
                // collect per-lid prediction (nearest-entry dedup already)
                std::vector<std::pair<float, int>> hv;   // (uR, lid)
                for (auto& kv2 : cand) hv.push_back({kv2.second.first, kv2.second.second});
                std::sort(hv.begin(), hv.end());
                float mg = 1e9f;
                for (size_t k2 = 1; k2 < hv.size(); ++k2)
                    if (hv[k2].second != hv[k2 - 1].second)
                        mg = std::min(mg, hv[k2].first - hv[k2 - 1].first);
                if (mg <= 2.0f) ++ptsGap2;
                // (b) armed phantoms: other-lid pred within 2px of an actual right point
                auto rit = rByRow.find(row);
                if (rit == rByRow.end()) continue;
                auto& rv = rit->second;
                bool armed = false, armedDistinct = false;
                float truePred = -1e9f;
                for (auto& h : hv) if (h.second == tlid) truePred = h.first;
                for (auto& h : hv) {
                    if (h.second == tlid) continue;
                    for (auto& e : rv) {
                        if (std::fabs(h.first - e.first) <= 2.0f) {
                            armed = true;
                            if (!(std::fabs(e.first - truePred) <= 2.0f)) armedDistinct = true;
                        }
                    }
                }
                if (armed) ++ptsArmed;
                if (armedDistinct) ++ptsArmedDistinct;
            }
            std::cout << "(a2) calib left pts with >=2 hyps: " << nPts
                      << " | two-hyps-uR-within-2px: " << 100.0 * ptsGap2 / nPts << "%\n";
            std::cout << "(b)  phantom hyp lands within 2px of an actual right point: "
                      << 100.0 * ptsArmed / nPts << "% of pts"
                      << " | lands on a point NOT covered by true hyp's window (true ambiguity): "
                      << 100.0 * ptsArmedDistinct / nPts << "%\n";
        }

        // (m) visual candidate audit: 5 random rows, annotate points + hypothesis hits
        {
            std::vector<cv::Point2f> cL2, cR2;
            std::vector<int> cLi, cRi;
            runCalib(0, 0, &cL2, &cR2, &cLi, &cRi);
            std::map<int, std::vector<size_t>> lIdx, rIdx;
            for (size_t i = 0; i < cL2.size(); ++i)
                lIdx[static_cast<int>(std::lround(cL2[i].y / rowStep))].push_back(i);
            for (size_t i = 0; i < cR2.size(); ++i)
                rIdx[static_cast<int>(std::lround(cR2[i].y / rowStep))].push_back(i);

            std::map<int, std::map<int, std::vector<std::pair<float, float>>>> tabCache2;
            auto getTab2 = [&](int row) -> std::map<int, std::vector<std::pair<float, float>>>& {
                auto it = tabCache2.find(row);
                if (it != tabCache2.end()) return it->second;
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
                tabCache2[row] = std::move(byLid);
                return tabCache2[row];
            };

            std::vector<int> candRows;
            for (auto& kv : lIdx)
                if (kv.second.size() >= 6 && kv.second.size() <= 20)
                    if (rIdx.count(kv.first)) candRows.push_back(kv.first);
            std::mt19937 rng(20260830);
            std::shuffle(candRows.begin(), candRows.end(), rng);
            if (candRows.size() > 5) candRows.resize(5);
            std::sort(candRows.begin(), candRows.end());

            const fs::path rowsDir = outDir / "rows";
            fs::create_directories(rowsDir);
            std::ofstream txt((rowsDir / "candidate_audit.txt").string());

            for (int row : candRows) {
                auto& lv = lIdx[row];
                auto& rv = rIdx[row];
                cv::Mat img(580, 2100, CV_8UC3, cv::Scalar(255, 255, 255));
                cv::putText(img, "row " + std::to_string(row) +
                            "  (top=LEFT pts, bottom=RIGHT pts, blue=TRUE pair, orange=hypothesis HIT)",
                            {20, 40}, 0, 0.7, cv::Scalar(0, 0, 0), 2);
                cv::line(img, {20, 150}, {2080, 150}, cv::Scalar(180, 180, 180), 1);
                cv::line(img, {20, 440}, {2080, 440}, cv::Scalar(180, 180, 180), 1);

                for (auto& r : calibPairs)
                    if (static_cast<int>(std::lround(r.ly / rowStep)) == row)
                        cv::line(img, {static_cast<int>(r.lx) + 20, 152},
                                 {static_cast<int>(r.rx) + 20, 438}, cv::Scalar(255, 0, 0), 2);

                for (size_t i : lv) {
                    int x = static_cast<int>(cL2[i].x) + 20;
                    cv::circle(img, {x, 150}, 5, cv::Scalar(0, 0, 255), -1);
                    cv::putText(img, "L" + std::to_string(cLi[i]), {x - 10, 125}, 0, 0.5, cv::Scalar(0, 0, 255), 1);
                }
                for (size_t i : rv) {
                    int x = static_cast<int>(cR2[i].x) + 20;
                    cv::circle(img, {x, 440}, 5, cv::Scalar(0, 140, 0), -1);
                    cv::putText(img, "R" + std::to_string(cRi[i]), {x - 10, 470}, 0, 0.5, cv::Scalar(0, 140, 0), 1);
                }

                txt << "\n==== row " << row << " ====\n";
                txt << "LEFT : ";
                for (size_t i : lv) txt << cL2[i].x << "(l" << cLi[i] << ") ";
                txt << "\nRIGHT: ";
                for (size_t i : rv) txt << cR2[i].x << "(l" << cRi[i] << ") ";
                txt << "\n";

                auto& tab = getTab2(row);
                for (size_t i : lv) {
                    const float lx = cL2[i].x;
                    const int tlid = cLi[i];
                    std::map<long long, std::pair<float, int>> cand;
                    for (auto& bl : tab) {
                        auto& v = bl.second;
                        int lo = 0, hi = static_cast<int>(v.size());
                        while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < lx) lo = mid + 1; else hi = mid; }
                        for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                            if (std::fabs(v[k].first - lx) > 0.7f) continue;
                            const long long key = (static_cast<long long>(bl.first) << 20)
                                                | static_cast<long long>(std::lround(v[k].second / rowStep));
                            if (!cand.count(key)) cand[key] = {v[k].second, bl.first};
                        }
                    }
                    txt << "  L x=" << lx << " trueLid=" << tlid << " | hyps=" << cand.size() << ":";
                    for (auto& kv2 : cand) {
                        const float pred = kv2.second.first;
                        const int hl = kv2.second.second;
                        int hitIdx = -1;
                        float hitX = 0;
                        for (size_t j : rv)
                            if (std::fabs(cR2[j].x - pred) <= 2.0f) { hitIdx = static_cast<int>(j); hitX = cR2[j].x; break; }
                        if (hitIdx >= 0) {
                            txt << " [lid" << hl << "->" << pred << " HIT R" << cRi[hitIdx] << "@" << hitX << "]";
                            cv::line(img, {static_cast<int>(lx) + 20, 152},
                                     {static_cast<int>(hitX) + 20, 438}, cv::Scalar(0, 130, 255), 1);
                            cv::putText(img, "h" + std::to_string(hl),
                                        {(static_cast<int>(lx) + static_cast<int>(hitX)) / 2 + 8, 295},
                                        0, 0.45, cv::Scalar(0, 130, 255), 1);
                        } else {
                            txt << " [lid" << hl << "->" << pred << " miss]";
                        }
                    }
                    txt << "\n";
                }
                cv::imwrite((rowsDir / ("row_" + std::to_string(row) + ".png")).string(), img);
                std::cout << "row audit image: " << (rowsDir / ("row_" + std::to_string(row) + ".png")).string() << "\n";
            }
        }

        // (m2) annotate RECTIFIED stereo images: 5 random rows, lid-labeled interp points
        {
            const auto& img0 = input->poseFrames[0][0];
            const cv::Mat gL = img0.leftGray;
            const cv::Mat gR = img0.rightGray;
            cv::Mat mapXL, mapYL, mapXR, mapYR, rectL, rectR;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        h.imageSize, CV_32FC1, mapXL, mapYL);
            cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                        h.imageSize, CV_32FC1, mapXR, mapYR);
            cv::remap(gL, rectL, mapXL, mapYL, cv::INTER_LINEAR);
            cv::remap(gR, rectR, mapXR, mapYR, cv::INTER_LINEAR);
            cv::cvtColor(rectL, rectL, cv::COLOR_GRAY2BGR);
            cv::cvtColor(rectR, rectR, cv::COLOR_GRAY2BGR);

            std::vector<cv::Point2f> cL2, cR2;
            std::vector<int> cLi, cRi;
            runCalib(0, 0, &cL2, &cR2, &cLi, &cRi);
            std::map<int, std::vector<size_t>> lIdx2, rIdx2;
            for (size_t i = 0; i < cL2.size(); ++i)
                lIdx2[static_cast<int>(std::lround(cL2[i].y / rowStep))].push_back(i);
            for (size_t i = 0; i < cR2.size(); ++i)
                rIdx2[static_cast<int>(std::lround(cR2[i].y / rowStep))].push_back(i);

            std::vector<int> candRows2;
            for (auto& kv : lIdx2)
                if (kv.second.size() >= 6 && kv.second.size() <= 20)
                    if (rIdx2.count(kv.first)) candRows2.push_back(kv.first);
            std::mt19937 rng2(20260830);
            std::shuffle(candRows2.begin(), candRows2.end(), rng2);
            if (candRows2.size() > 5) candRows2.resize(5);
            std::sort(candRows2.begin(), candRows2.end());

            const int W = h.imageSize.width, H = h.imageSize.height;
            for (int row : candRows2) {
                const int y = static_cast<int>(row * rowStep);
                cv::line(rectL, {0, y}, {W, y}, cv::Scalar(0, 255, 255), 1);
                cv::line(rectR, {0, y}, {W, y}, cv::Scalar(0, 255, 255), 1);
                cv::putText(rectL, "row " + std::to_string(row), {8, y - 5}, 0, 0.55,
                            cv::Scalar(0, 255, 255), 1);
                cv::putText(rectR, "row " + std::to_string(row), {8, y - 5}, 0, 0.55,
                            cv::Scalar(0, 255, 255), 1);
                auto itL = lIdx2.find(row);
                if (itL != lIdx2.end())
                    for (size_t i : itL->second) {
                        const int x = static_cast<int>(cL2[i].x);
                        cv::circle(rectL, {x, y}, 4, cv::Scalar(0, 0, 255), -1);
                        cv::putText(rectL, "L" + std::to_string(cLi[i]), {x - 8, y - 8},
                                    0, 0.45, cv::Scalar(0, 0, 255), 1);
                    }
                auto itR = rIdx2.find(row);
                if (itR != rIdx2.end())
                    for (size_t i : itR->second) {
                        const int x = static_cast<int>(cR2[i].x);
                        cv::circle(rectR, {x, y}, 4, cv::Scalar(0, 220, 0), -1);
                        cv::putText(rectR, "R" + std::to_string(cRi[i]), {x - 8, y + 20},
                                    0, 0.45, cv::Scalar(0, 220, 0), 1);
                    }
            }
            cv::Mat canvas(H, W * 2 + 10, rectL.type(), cv::Scalar(0, 0, 0));
            rectL.copyTo(canvas(cv::Rect(0, 0, W, H)));
            rectR.copyTo(canvas(cv::Rect(W + 10, 0, W, H)));
            cv::putText(canvas, "LEFT (rectified)", {10, 22}, 0, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(canvas, "RIGHT (rectified)", {W + 20, 22}, 0, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::imwrite((outDir / "rows" / "rect_rows_annotated.png").string(), canvas);
            std::cout << "rectified annotated image -> "
                      << (outDir / "rows" / "rect_rows_annotated.png").string() << "\n";
        }

        // (m3) matcher-output lids (RAW, no post-filter) annotated on rectified images
        {
            const cv::Mat gL2 = input->poseFrames[0][0].leftGray;
            const cv::Mat gR2 = input->poseFrames[0][0].rightGray;
            cv::Mat mapXL2, mapYL2, mapXR2, mapYR2, recL, recR;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        h.imageSize, CV_32FC1, mapXL2, mapYL2);
            cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                        h.imageSize, CV_32FC1, mapXR2, mapYR2);
            cv::remap(gL2, recL, mapXL2, mapYL2, cv::INTER_LINEAR);
            cv::remap(gR2, recR, mapXR2, mapYR2, cv::INTER_LINEAR);
            cv::cvtColor(recL, recL, cv::COLOR_GRAY2BGR);
            cv::cvtColor(recR, recR, cv::COLOR_GRAY2BGR);

            std::map<int, std::vector<const MatchPairRec*>> spByRow, cpByRow;
            for (auto& r : scanPairs)
                spByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
            for (auto& r : calibPairs)
                cpByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);

            std::map<int, std::vector<size_t>> lIdx3;
            for (size_t i = 0; i < sL.size(); ++i)
                lIdx3[static_cast<int>(std::lround(sL[i].y / rowStep))].push_back(i);
            std::vector<int> candRows3;
            for (auto& kv : lIdx3)
                if (kv.second.size() >= 6 && kv.second.size() <= 20)
                    if (spByRow.count(kv.first)) candRows3.push_back(kv.first);
            std::mt19937 rng3(20260830);
            std::shuffle(candRows3.begin(), candRows3.end(), rng3);
            if (candRows3.size() > 5) candRows3.resize(5);
            std::sort(candRows3.begin(), candRows3.end());

            std::ofstream mtxt((outDir / "rows" / "matched_lids_raw.txt").string());
            mtxt << "matcher-assigned lids (RAW GPU output, no post-filter) vs calib truth\n";

            const int W3 = h.imageSize.width, H3 = h.imageSize.height;
            for (int row : candRows3) {
                const int y = static_cast<int>(row * rowStep);
                cv::line(recL, {0, y}, {W3, y}, cv::Scalar(0, 255, 255), 1);
                cv::line(recR, {0, y}, {W3, y}, cv::Scalar(0, 255, 255), 1);
                cv::putText(recL, "row " + std::to_string(row), {8, y - 5}, 0, 0.55,
                            cv::Scalar(0, 255, 255), 1);
                cv::putText(recR, "row " + std::to_string(row), {8, y - 5}, 0, 0.55,
                            cv::Scalar(0, 255, 255), 1);
                mtxt << "\n==== row " << row << " ====\n";
                auto it = spByRow.find(row);
                if (it == spByRow.end()) continue;
                for (auto* pr : it->second) {
                    const MatchPairRec* truth = nullptr;
                    auto cit = cpByRow.find(row);
                    if (cit != cpByRow.end())
                        for (auto* c : cit->second)
                            if (std::fabs(c->lx - pr->lx) < 0.6f) { truth = c; break; }
                    const bool ok = truth && std::fabs(truth->rx - pr->rx) < 0.5f;
                    const cv::Scalar col = ok ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
                    const int xl = static_cast<int>(pr->lx);
                    const int xr = static_cast<int>(pr->rx);
                    cv::circle(recL, {xl, y}, 4, col, -1);
                    cv::circle(recR, {xr, y}, 4, col, -1);
                    cv::putText(recL, "M" + std::to_string(pr->lid), {xl - 8, y - 8}, 0, 0.45, col, 1);
                    cv::putText(recR, "M" + std::to_string(pr->lid), {xr - 8, y + 20}, 0, 0.45, col, 1);
                    mtxt << "  Lx=" << pr->lx << " Rx=" << pr->rx
                         << " matcherLid=" << pr->lid
                         << " truthLid=" << (truth ? std::to_string(truth->lid) : std::string("none"))
                         << (ok ? "  OK" : "  WRONG") << "\n";
                }
            }
            cv::Mat canvas3(H3, W3 * 2 + 10, recL.type(), cv::Scalar(0, 0, 0));
            recL.copyTo(canvas3(cv::Rect(0, 0, W3, H3)));
            recR.copyTo(canvas3(cv::Rect(W3 + 10, 0, W3, H3)));
            cv::putText(canvas3, "LEFT  (blue=pair OK, red=WRONG, Mxx=matcher lid RAW)",
                        {10, 22}, 0, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(canvas3, "RIGHT", {W3 + 20, 22}, 0, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::imwrite((outDir / "rows" / "rect_rows_matched_lids.png").string(), canvas3);
            std::cout << "matcher-lid annotated image -> "
                      << (outDir / "rows" / "rect_rows_matched_lids.png").string() << "\n";
        }

        // (m4) ALL computed candidate lids per point (V8-style, no post-filter)
        //      annotated on rectified images; vertical lid stack per left point
        {
            const cv::Mat gL4 = input->poseFrames[0][0].leftGray;
            const cv::Mat gR4 = input->poseFrames[0][0].rightGray;
            cv::Mat mapXL4, mapYL4, mapXR4, mapYR4, recL4, recR4;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        h.imageSize, CV_32FC1, mapXL4, mapYL4);
            cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                        h.imageSize, CV_32FC1, mapXR4, mapYR4);
            cv::remap(gL4, recL4, mapXL4, mapYL4, cv::INTER_LINEAR);
            cv::remap(gR4, recR4, mapXR4, mapYR4, cv::INTER_LINEAR);
            cv::cvtColor(recL4, recL4, cv::COLOR_GRAY2BGR);
            cv::cvtColor(recR4, recR4, cv::COLOR_GRAY2BGR);

            // scan left/right by row (matcher input)
            std::map<int, std::vector<size_t>> slIdx, srIdx;
            for (size_t i = 0; i < sL.size(); ++i)
                slIdx[static_cast<int>(std::lround(sL[i].y / rowStep))].push_back(i);
            for (size_t i = 0; i < sR.size(); ++i)
                srIdx[static_cast<int>(std::lround(sR[i].y / rowStep))].push_back(i);
            // calib truth by row
            std::map<int, std::vector<const MatchPairRec*>> cp4;
            for (auto& r : calibPairs)
                cp4[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);

            // table cache
            std::map<int, std::map<int, std::vector<std::pair<float, float>>>> tab4;
            auto getTab4 = [&](int row) -> std::map<int, std::vector<std::pair<float, float>>>& {
                auto it = tab4.find(row);
                if (it != tab4.end()) return it->second;
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
                tab4[row] = std::move(byLid);
                return tab4[row];
            };

            // pick 5 rows, enforce >=320-row spacing (stack height)
            std::vector<int> pool4;
            for (auto& kv : slIdx)
                if (kv.second.size() >= 6 && kv.second.size() <= 20 && srIdx.count(kv.first))
                    pool4.push_back(kv.first);
            std::mt19937 rng4(20260830);
            std::shuffle(pool4.begin(), pool4.end(), rng4);
            std::vector<int> rows4;
            for (int r : pool4) {
                bool ok = true;
                for (int t : rows4)
                    if (std::abs(t - r) < 320) { ok = false; break; }
                if (ok) rows4.push_back(r);
                if (rows4.size() >= 5) break;
            }
            std::sort(rows4.begin(), rows4.end());

            std::ofstream t4((outDir / "rows" / "all_candidate_lids.txt").string());
            t4 << "ALL computed hypothesis lids per point (V8-style collection, no filter)\n"
               << "lid* = HIT (right point exists within +-2px), lid = miss, [lid]=true lid\n";

            const int W4 = h.imageSize.width, H4 = h.imageSize.height;
            for (int row : rows4) {
                const int y = static_cast<int>(row * rowStep);
                cv::line(recL4, {0, y}, {W4, y}, cv::Scalar(0, 255, 255), 1);
                cv::line(recR4, {0, y}, {W4, y}, cv::Scalar(0, 255, 255), 1);
                cv::putText(recL4, "row " + std::to_string(row), {8, y + 18}, 0, 0.55,
                            cv::Scalar(0, 255, 255), 1);
                t4 << "\n==== row " << row << " ====\n";
                auto& rv = srIdx[row];
                for (size_t i : slIdx[row]) {
                    const float lx = sL[i].x;
                    // truth lid
                    int tlid = -1;
                    auto cit = cp4.find(row);
                    if (cit != cp4.end())
                        for (auto* c : cit->second)
                            if (std::fabs(c->lx - lx) < 0.6f) { tlid = c->lid; break; }
                    // matcher-style candidates
                    std::map<long long, std::pair<float, int>> cand;
                    for (auto& bl : getTab4(row)) {
                        auto& v = bl.second;
                        int lo = 0, hi = static_cast<int>(v.size());
                        while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < lx) lo = mid + 1; else hi = mid; }
                        for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                            if (std::fabs(v[k].first - lx) > 0.7f) continue;
                            const long long key = (static_cast<long long>(bl.first) << 20)
                                                | static_cast<long long>(std::lround(v[k].second / rowStep));
                            if (!cand.count(key)) cand[key] = {v[k].second, bl.first};
                        }
                    }
                    // hits among scan right points
                    std::vector<std::pair<int, float>> hits;   // (lid, hitRx)
                    for (auto& kv2 : cand) {
                        for (size_t j : rv)
                            if (std::fabs(sR[j].x - kv2.second.first) <= 2.0f) {
                                hits.push_back({kv2.second.second, sR[j].x});
                                break;
                            }
                    }
                    // draw left point + vertical lid stack
                    const int x = static_cast<int>(lx);
                    cv::circle(recL4, {x, y}, 4,
                               tlid > 0 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), -1);
                    int sy = y - 10;
                    t4 << "  Lx=" << lx << " true=" << (tlid > 0 ? std::to_string(tlid) : "-")
                       << " hyps=" << cand.size() << ":";
                    std::map<int, bool> lidHit;
                    for (auto& h2 : hits) lidHit[h2.first] = true;
                    for (auto& kv2 : cand) {
                        const int hl = kv2.second.second;
                        if (lidHit.count(hl)) continue;   // drawn later
                        cv::putText(recL4, std::to_string(hl), {x - 5, sy}, 0, 0.35,
                                    cv::Scalar(120, 120, 120), 1);
                        sy -= 13;
                        t4 << " " << hl;
                    }
                    for (auto& h2 : hits) {
                        const bool isTrue = (h2.first == tlid);
                        cv::putText(recL4, (isTrue ? "[" : "") + std::to_string(h2.first) + (isTrue ? "]" : "*"),
                                    {x - 6, sy}, 0, 0.4,
                                    isTrue ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 140, 0), 1);
                        sy -= 14;
                        t4 << " " << h2.first << (isTrue ? "[]" : "*");
                        // orange line to hit right point
                        const int xr = static_cast<int>(h2.second);
                        cv::line(recL4, {x, y}, {xr, y}, cv::Scalar(0, 130, 255), 1);
                        cv::circle(recR4, {xr, y}, 4, cv::Scalar(0, 140, 0), -1);
                        cv::putText(recR4, "h" + std::to_string(h2.first), {xr - 6, y + 18}, 0, 0.4,
                                    cv::Scalar(0, 140, 0), 1);
                    }
                    t4 << "\n";
                }
            }
            cv::Mat canvas4(H4, W4 * 2 + 10, recL4.type(), cv::Scalar(0, 0, 0));
            recL4.copyTo(canvas4(cv::Rect(0, 0, W4, H4)));
            recR4.copyTo(canvas4(cv::Rect(W4 + 10, 0, W4, H4)));
            cv::putText(canvas4, "LEFT: gray lid=miss(no right pt), green lid*=HIT, red [lid]*=TRUE, orange line=hit target",
                        {10, 22}, 0, 0.7, cv::Scalar(255, 255, 255), 2);
            cv::putText(canvas4, "RIGHT: hit points + claiming lid", {W4 + 20, 22}, 0, 0.7,
                        cv::Scalar(255, 255, 255), 2);
            cv::imwrite((outDir / "rows" / "rect_rows_all_lids.png").string(), canvas4);
            std::cout << "all-lids annotated image -> "
                      << (outDir / "rows" / "rect_rows_all_lids.png").string() << "\n";
        }

        // (m5) clean annotation: HIT lids only, left points + right points labeled
        {
            const cv::Mat gL5 = input->poseFrames[0][0].leftGray;
            const cv::Mat gR5 = input->poseFrames[0][0].rightGray;
            cv::Mat mapXL5, mapYL5, mapXR5, mapYR5, recL5, recR5;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        h.imageSize, CV_32FC1, mapXL5, mapYL5);
            cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                        h.imageSize, CV_32FC1, mapXR5, mapYR5);
            cv::remap(gL5, recL5, mapXL5, mapYL5, cv::INTER_LINEAR);
            cv::remap(gR5, recR5, mapXR5, mapYR5, cv::INTER_LINEAR);
            cv::cvtColor(recL5, recL5, cv::COLOR_GRAY2BGR);
            cv::cvtColor(recR5, recR5, cv::COLOR_GRAY2BGR);

            std::map<int, std::vector<size_t>> sl5, sr5;
            for (size_t i = 0; i < sL.size(); ++i)
                sl5[static_cast<int>(std::lround(sL[i].y / rowStep))].push_back(i);
            for (size_t i = 0; i < sR.size(); ++i)
                sr5[static_cast<int>(std::lround(sR[i].y / rowStep))].push_back(i);

            std::map<int, std::map<int, std::vector<std::pair<float, float>>>> tab5;
            auto getTab5 = [&](int row) -> std::map<int, std::vector<std::pair<float, float>>>& {
                auto it = tab5.find(row);
                if (it != tab5.end()) return it->second;
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
                tab5[row] = std::move(byLid);
                return tab5[row];
            };

            std::vector<int> pool5;
            for (auto& kv : sl5)
                if (kv.second.size() >= 6 && kv.second.size() <= 20 && sr5.count(kv.first))
                    pool5.push_back(kv.first);
            std::mt19937 rng5(20260830);
            std::shuffle(pool5.begin(), pool5.end(), rng5);
            std::vector<int> rows5;
            for (int r : pool5) {
                bool ok = true;
                for (int t : rows5)
                    if (std::abs(t - r) < 320) { ok = false; break; }
                if (ok) rows5.push_back(r);
                if (rows5.size() >= 5) break;
            }
            std::sort(rows5.begin(), rows5.end());

            const int W5 = h.imageSize.width, H5 = h.imageSize.height;
            for (int row : rows5) {
                const int y = static_cast<int>(row * rowStep);
                cv::line(recL5, {0, y}, {W5, y}, cv::Scalar(0, 255, 255), 1);
                cv::line(recR5, {0, y}, {W5, y}, cv::Scalar(0, 255, 255), 1);
                auto& rv = sr5[row];
                // collect right-point claims: rx -> lids
                std::map<float, std::set<int>> rClaims;
                for (size_t i : sl5[row]) {
                    const float lx = sL[i].x;
                    std::map<long long, std::pair<float, int>> cand;
                    for (auto& bl : getTab5(row)) {
                        auto& v = bl.second;
                        int lo = 0, hi = static_cast<int>(v.size());
                        while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < lx) lo = mid + 1; else hi = mid; }
                        for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                            if (std::fabs(v[k].first - lx) > 0.7f) continue;
                            const long long key = (static_cast<long long>(bl.first) << 20)
                                                | static_cast<long long>(std::lround(v[k].second / rowStep));
                            if (!cand.count(key)) cand[key] = {v[k].second, bl.first};
                        }
                    }
                    // HIT lids
                    std::set<int> hitLids;
                    for (auto& kv2 : cand)
                        for (size_t j : rv)
                            if (std::fabs(sR[j].x - kv2.second.first) <= 2.0f) {
                                hitLids.insert(kv2.second.second);
                                rClaims[sR[j].x].insert(kv2.second.second);
                                break;
                            }
                    if (!hitLids.empty()) {
                        const int x = static_cast<int>(lx);
                        cv::circle(recL5, {x, y}, 4, cv::Scalar(0, 0, 255), -1);
                        int sy = y - 10;
                        for (auto it2 = hitLids.rbegin(); it2 != hitLids.rend(); ++it2, sy -= 14)
                            cv::putText(recL5, std::to_string(*it2), {x - 4, sy}, 0, 0.45,
                                        cv::Scalar(0, 255, 0), 1);
                    }
                }
                // right points: circles + claiming lids
                for (auto& rc : rClaims) {
                    const int x = static_cast<int>(rc.first);
                    cv::circle(recR5, {x, y}, 4, cv::Scalar(0, 140, 0), -1);
                    int sy = y + 22;
                    for (int l : rc.second) {
                        cv::putText(recR5, std::to_string(l), {x - 4, sy}, 0, 0.45,
                                    cv::Scalar(0, 140, 0), 1);
                        sy += 14;
                    }
                }
            }
            cv::Mat canvas5(H5, W5 * 2 + 10, recL5.type(), cv::Scalar(0, 0, 0));
            recL5.copyTo(canvas5(cv::Rect(0, 0, W5, H5)));
            recR5.copyTo(canvas5(cv::Rect(W5 + 10, 0, W5, H5)));
            cv::imwrite((outDir / "rows" / "rect_rows_hit_lids.png").string(), canvas5);
            std::cout << "hit-lids clean image -> "
                      << (outDir / "rows" / "rect_rows_hit_lids.png").string() << "\n";
        }

        // (m6) interp-points image (no photo, dots on black, natural line breaks visible)
        //      + 5 rows annotated with vertical HIT-lid stacks
        {
            const int W6 = h.imageSize.width, H6 = h.imageSize.height;
            cv::Mat imL6(H6, W6, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Mat imR6(H6, W6, CV_8UC3, cv::Scalar(0, 0, 0));

            // draw all interp points as dots (natural line traces with gaps)
            for (auto& p : sL) {
                int x = cvRound(p.x), y = cvRound(p.y);
                if (x >= 0 && x < W6 && y >= 0 && y < H6)
                    imL6.at<cv::Vec3b>(y, x) = cv::Vec3b(180, 180, 180);
            }
            for (auto& p : sR) {
                int x = cvRound(p.x), y = cvRound(p.y);
                if (x >= 0 && x < W6 && y >= 0 && y < H6)
                    imR6.at<cv::Vec3b>(y, x) = cv::Vec3b(180, 180, 180);
            }

            std::map<int, std::vector<size_t>> sl6, sr6;
            for (size_t i = 0; i < sL.size(); ++i)
                sl6[static_cast<int>(std::lround(sL[i].y / rowStep))].push_back(i);
            for (size_t i = 0; i < sR.size(); ++i)
                sr6[static_cast<int>(std::lround(sR[i].y / rowStep))].push_back(i);

            std::map<int, std::map<int, std::vector<std::pair<float, float>>>> tab6;
            auto getTab6 = [&](int row) -> std::map<int, std::vector<std::pair<float, float>>>& {
                auto it = tab6.find(row);
                if (it != tab6.end()) return it->second;
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
                tab6[row] = std::move(byLid);
                return tab6[row];
            };

            std::vector<int> pool6;
            for (auto& kv : sl6)
                if (kv.second.size() >= 6 && kv.second.size() <= 20 && sr6.count(kv.first))
                    pool6.push_back(kv.first);
            std::mt19937 rng6(20260830);
            std::shuffle(pool6.begin(), pool6.end(), rng6);
            // pick 10 groups of 5 consecutive rows, groups >= 320 apart
            std::vector<int> groups;   // starting row of each group
            for (int r : pool6) {
                // check 5 consecutive rows all valid
                bool ok5 = true;
                for (int d = 0; d < 5; ++d)
                    if (!sl6.count(r + d) || !sr6.count(r + d) ||
                        sl6[r + d].size() < 6 || sl6[r + d].size() > 20) { ok5 = false; break; }
                if (!ok5) continue;
                bool ok = true;
                for (int t : groups)
                    if (std::abs(t - r) < 320) { ok = false; break; }
                if (ok) groups.push_back(r);
                if (groups.size() >= 10) break;
            }
            std::vector<int> rows6;
            for (int g : groups)
                for (int d = 0; d < 5; ++d)
                    rows6.push_back(g + d);
            std::sort(rows6.begin(), rows6.end());

            for (int row : rows6) {
                const int y = static_cast<int>(row * rowStep);
                cv::line(imL6, {0, y}, {W6, y}, cv::Scalar(0, 255, 255), 1);
                cv::line(imR6, {0, y}, {W6, y}, cv::Scalar(0, 255, 255), 1);
                auto& rv = sr6[row];
                std::map<float, std::set<int>> rClaims;
                for (size_t i : sl6[row]) {
                    const float lx = sL[i].x;
                    std::map<long long, std::pair<float, int>> cand;
                    for (auto& bl : getTab6(row)) {
                        auto& v = bl.second;
                        int lo = 0, hi = static_cast<int>(v.size());
                        while (lo < hi) { int mid = (lo + hi) / 2; if (v[mid].first < lx) lo = mid + 1; else hi = mid; }
                        for (int k = std::max(0, lo - 1); k <= std::min(lo, static_cast<int>(v.size()) - 1); ++k) {
                            if (std::fabs(v[k].first - lx) > 0.7f) continue;
                            const long long key = (static_cast<long long>(bl.first) << 20)
                                                | static_cast<long long>(std::lround(v[k].second / rowStep));
                            if (!cand.count(key)) cand[key] = {v[k].second, bl.first};
                        }
                    }
                    std::set<int> hitLids;
                    for (auto& kv2 : cand)
                        for (size_t j : rv)
                            if (std::fabs(sR[j].x - kv2.second.first) <= 2.0f) {
                                hitLids.insert(kv2.second.second);
                                rClaims[sR[j].x].insert(kv2.second.second);
                                break;
                            }
                    if (!hitLids.empty()) {
                        const int x = static_cast<int>(lx);
                        cv::circle(imL6, {x, y}, 4, cv::Scalar(0, 0, 255), -1);
                        int sy = y - 10;
                        for (auto it2 = hitLids.rbegin(); it2 != hitLids.rend(); ++it2, sy -= 14)
                            cv::putText(imL6, std::to_string(*it2), {x - 4, sy}, 0, 0.45,
                                        cv::Scalar(0, 255, 0), 1);
                    } else {
                        // zero-hit: draw with distinct marker + "X"
                        const int x = static_cast<int>(lx);
                        cv::circle(imL6, {x, y}, 4, cv::Scalar(0, 0, 255), 2);
                        cv::putText(imL6, "X", {x - 4, y - 8}, 0, 0.45,
                                    cv::Scalar(0, 0, 255), 1);
                    }
                }
                for (auto& rc : rClaims) {
                    const int x = static_cast<int>(rc.first);
                    cv::circle(imR6, {x, y}, 4, cv::Scalar(0, 140, 0), -1);
                    int sy = y + 22;
                    for (int l : rc.second) {
                        cv::putText(imR6, std::to_string(l), {x - 4, sy}, 0, 0.45,
                                    cv::Scalar(0, 140, 0), 1);
                        sy += 14;
                    }
                }
            }
            cv::Mat canvas6(H6, W6 * 2 + 10, imL6.type(), cv::Scalar(0, 0, 0));
            imL6.copyTo(canvas6(cv::Rect(0, 0, W6, H6)));
            imR6.copyTo(canvas6(cv::Rect(W6 + 10, 0, W6, H6)));
            cv::imwrite((outDir / "rows" / "interp_points_hit_lids.png").string(), canvas6);
            std::cout << "interp-points hit-lids image -> "
                      << (outDir / "rows" / "interp_points_hit_lids.png").string() << "\n";
        }

        // (g) LR cross-check audit
        {
            std::cout << "\n==== LR cross-check audit (pose_06) ====\n";
            std::map<int, std::vector<float>> leftByRow;
            for (auto& p : sL)
                leftByRow[static_cast<int>(std::lround(p.y / rowStep))].push_back(p.x);
            for (auto& kv : leftByRow) std::sort(kv.second.begin(), kv.second.end());

            std::map<int, std::vector<const MatchPairRec*>> pairByRow;
            for (auto& r : scanPairs)
                pairByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);
            std::map<int, std::vector<const MatchPairRec*>> calibByRow;
            for (auto& r : calibPairs)
                calibByRow[static_cast<int>(std::lround(r.ly / rowStep))].push_back(&r);

            auto hasLeftHit = [&](int row, float xL, float& nearestDx, float& hitX) {
                nearestDx = 1e9f; hitX = -1;
                auto it = leftByRow.find(row);
                if (it == leftByRow.end()) return false;
                auto& v = it->second;
                auto lb = std::lower_bound(v.begin(), v.end(), xL - 2.0f);
                bool any = false;
                for (auto p = lb; p != v.end() && *p <= xL + 2.0f; ++p) {
                    const float dx = std::fabs(*p - xL);
                    if (dx < nearestDx) { nearestDx = dx; hitX = *p; }
                    any = true;
                }
                return any;
            };

            int nTrue = 0, nWrong = 0;
            int trueKeptUniq = 0, trueKilledUniq = 0;
            int wrongKilledUniq = 0, wrongKeptUniq = 0;
            int trueKeptNear = 0, trueKilledNear = 0;
            int wrongKilledNear = 0, wrongKeptNear = 0;

            for (auto& kv : pairByRow) {
                const int row = kv.first;
                std::vector<CurveMapEntry> tmp;
                int cap = 4096;
                tmp.resize(cap);
                int n = cmt.EnumerateRow(row, tmp.data(), cap);
                while (n < 0) { cap *= 2; tmp.resize(cap); n = cmt.EnumerateRow(row, tmp.data(), cap); }
                std::vector<const MatchPairRec*> ct;
                auto cit = calibByRow.find(row);
                if (cit != calibByRow.end()) ct = cit->second;

                for (auto* pr : kv.second) {
                    bool isTrue = false;
                    for (auto* c : ct)
                        if (std::fabs(c->lx - pr->lx) < 0.6f && std::fabs(c->rx - pr->rx) < 0.5f) { isTrue = true; break; }
                    isTrue ? ++nTrue : ++nWrong;

                    std::map<long long, std::pair<float, float>> revCand;
                    for (int q = 0; q < n; ++q) {
                        const float uRe = tmp[static_cast<size_t>(q)].uR * rowStep;
                        const float uLe = tmp[static_cast<size_t>(q)].uL * rowStep;
                        const float dur = std::fabs(uRe - pr->rx);
                        if (dur > 0.7f) continue;
                        const long long key = (static_cast<long long>(tmp[static_cast<size_t>(q)].lid) << 20)
                                            | static_cast<long long>(std::lround(uLe / rowStep));
                        auto it2 = revCand.find(key);
                        if (it2 == revCand.end() || dur < it2->second.second)
                            revCand[key] = {uLe, dur};
                    }
                    int hits = 0;
                    float bestDx = 1e9f, bestX = -1;
                    for (auto& rc : revCand) {
                        float ndx, hx;
                        if (hasLeftHit(row, rc.second.first, ndx, hx)) {
                            ++hits;
                            if (ndx < bestDx) { bestDx = ndx; bestX = hx; }
                        }
                    }
                    const bool selfInRev = (bestX > 0 && std::fabs(bestX - pr->lx) <= 0.6f);
                    const bool passUniq = (hits == 1 && selfInRev);
                    const bool passNear = selfInRev;

                    if (isTrue) {
                        passUniq ? ++trueKeptUniq : ++trueKilledUniq;
                        passNear ? ++trueKeptNear : ++trueKilledNear;
                    } else {
                        passUniq ? ++wrongKeptUniq : ++wrongKilledUniq;
                        passNear ? ++wrongKeptNear : ++wrongKilledNear;
                    }
                }
            }
            std::cout << "scan pairs: true=" << nTrue << " wrong=" << nWrong << "\n";
            std::cout << "LR [A: reverse-unique + self]:  kill wrong " << wrongKilledUniq << "/" << nWrong
                      << " | collateral kill true " << trueKilledUniq << "/" << nTrue << "\n";
            std::cout << "LR [B: reverse-nearest + self]: kill wrong " << wrongKilledNear << "/" << nWrong
                      << " | collateral kill true " << trueKilledNear << "/" << nTrue << "\n";
        }
    }

    std::cout << "\nanalysis outputs -> " << outDir.string() << "\n";
    SUCCEED();
}
