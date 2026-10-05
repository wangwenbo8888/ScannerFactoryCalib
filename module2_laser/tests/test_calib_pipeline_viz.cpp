// test_calib_pipeline_viz.cpp — 标定流程逐步骤可视化（前 2 帧, 原版算子链）
//
// 链路: mask_extract → region_analyze(ccl) → laser_label(编号) → steger(中心提取)
//       → undistort_cuda → epipolar_interp(编号插值) → laser_match_scan(v1 编号匹配)
//       → laser_reconstruct
// 输入: data_in/left_skew 前 2 帧 + data_out/left_skew_calib.json
// 输出: data_out/calib_viz/
//   {stem}_{L|R}_mask.png     清洗后掩膜
//   {stem}_{L|R}_ccl.png      连通区域标记（按区域号着色）
//   {stem}_{L|R}_label.png    激光线编号掩膜（按线号着色）
//   {stem}_{L|R}_steger.png   灰度图 + 亚像素中心点(红)
//   {stem}_{L|R}_undist.png   矫正后点集(白)
//   {stem}_{L|R}_interp.png   编号插值点集(绿)
//   {stem}_recon.ply          三维点云(按线号着色)
//   summary.txt               各步骤点数/匹配率/耗时
// 对照: 扫描流程工具见 test_scan_pipeline_viz.cpp（mask→steger_fast Flat→…→match_v2）
// 缺数据 → GTEST_SKIP。

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "curve_map.h"
#include "laser_match_scan_cuda.h"
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

void writePlyAscii(const std::string& path,
                   const std::vector<cv::Vec3f>& pts,
                   const std::vector<int>& ids,
                   const std::function<cv::Scalar(int)>& colorOf) {
    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";
    f << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < pts.size(); ++i) {
        const cv::Scalar c = colorOf(ids[i]);
        f << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << " "
          << (int)c[0] << " " << (int)c[1] << " " << (int)c[2] << "\n";
    }
}

cv::Scalar lidColor(int id) {
    const int h = ((id * 137) % 360 + 360) % 360;
    return cv::Scalar(h, 180, 255);
}

cv::Scalar hsvToBgr(const cv::Scalar& hsv) {
    cv::Mat m(1, 1, CV_8UC3);
    m.at<cv::Vec3b>(0, 0) = cv::Vec3b((uchar)hsv[0], (uchar)hsv[1], (uchar)hsv[2]);
    cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = m.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

// 标记掩膜（CV_32SC1）→ 按标号着色可视化
void saveLabelVis(const std::string& path, const cv::Mat& labels) {
    cv::Mat vis(labels.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    int maxL = 0;
    for (int r = 0; r < labels.rows; ++r) {
        const int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c)
            if (row[c] > maxL) maxL = row[c];
    }
    for (int r = 0; r < labels.rows; ++r) {
        const int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            if (row[c] > 0) {
                const cv::Scalar cBgr = hsvToBgr(lidColor(row[c] % std::max(1, maxL)));
                vis.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar)cBgr[0], (uchar)cBgr[1], (uchar)cBgr[2]);
            }
        }
    }
    cv::imwrite(path, vis);
}

void savePointsOnGray(const std::string& path, const cv::Mat& gray,
                      const std::vector<cv::Point2f>& pts) {
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto& p : pts) {
        const int x = cvRound(p.x), y = cvRound(p.y);
        if (x >= 0 && x < vis.cols && y >= 0 && y < vis.rows)
            cv::circle(vis, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    cv::imwrite(path, vis);
}

void savePointsOnCanvas(const std::string& path, const cv::Size& size,
                        const std::vector<cv::Point2f>& pts, const cv::Scalar& color) {
    cv::Mat vis(size, CV_8UC3, cv::Scalar(0, 0, 0));
    for (const auto& p : pts) {
        const int x = cvRound(p.x), y = cvRound(p.y);
        if (x >= 0 && x < vis.cols && y >= 0 && y < vis.rows)
            cv::circle(vis, cv::Point(x, y), 1, color, cv::FILLED);
    }
    cv::imwrite(path, vis);
}

std::vector<cv::Point2f> downloadPoints(const cv::cuda::GpuMat& d_pts) {
    cv::Mat m;
    d_pts.download(m);
    if (m.empty()) return {};
    m = m.reshape(2, 1);
    return std::vector<cv::Point2f>(m.begin<cv::Point2f>(), m.end<cv::Point2f>());
}

} // namespace

TEST(CalibPipelineViz, TwoFramesStepOutputs) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir, calibJson;
    if (!findData(dataDir, calibJson)) {
        GTEST_SKIP() << "left_skew data not found; skip calib viz test";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "calib_viz";
    fs::create_directories(outDir);
    std::ofstream summary((outDir / "summary.txt").string());

    // ---------- 载入标定 + 表（同 fcscan_e2e） ----------
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

    // ---------- 算子装配（原版标定链） ----------
    cv::cuda::Stream stream;
    cudaStream_t cuStream = cv::cuda::StreamAccessor::getStream(stream);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
    if (const char* env = std::getenv("FC_MASK_APPROX"))
        mp.morphApprox = std::atoi(env);
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

    EpipolarInterpParams eip; eip.epipolar_row_step = rowStep; eip.lineIdCheck = true;
    EpipolarInterpCuda epiL(eip), epiR(eip);

    LaserMatchScanParams msp;                    // v1 编号匹配
    msp.epipolar_row_step = rowStep;
    msp.match_threshold = 2.0f;
    msp.vL_tolerance = rowStep;
    LaserMatchScanCuda matcher(msp);
    ASSERT_TRUE(matcher.SetTempTable(tempTable));
    matcher.SetCurrentTemperature(cfg.referenceTemp);
    matcher.Warmup(65536, 65536);

    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    auto timedMs = [&](auto&& fn) {
        cudaEventRecord(ev0, cuStream);
        auto r = fn();
        cudaEventRecord(ev1, cuStream);
        cudaEventSynchronize(ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        return std::make_pair(std::move(r), ms);
    };

    const cv::Size canvas(h.imageSize.width, h.imageSize.height);

    auto processFrame = [&](size_t pi, size_t ti, bool warmup) {
        const auto& img = input->poseFrames[pi][ti];
        const std::string stem = input->poseDirs[pi] + "_tube" + std::to_string(ti);
        const std::string o = (outDir / stem).string();
        std::ostringstream log;
        double tMask = 0, tCcl = 0, tLab = 0, tStg = 0, tUn = 0, tEpi = 0, tMatch = 0, tRecon = 0;

        // ── 1. mask ──
        auto [m1, m1ms] = timedMs([&] { return maskL.Execute(img.leftGray, stream); });
        auto [m2, m2ms] = timedMs([&] { return maskR.Execute(img.rightGray, stream); });
        tMask = m1ms + m2ms;
        ASSERT_TRUE(m1.success && m2.success) << stem << ": mask fail";
        if (!warmup) {
            cv::Mat vL, vR;
            m1.d_cleanedMask->download(vL);
            m2.d_cleanedMask->download(vR);
            cv::imwrite(o + "_L_mask.png", vL);
            cv::imwrite(o + "_R_mask.png", vR);
        }

        // ── 2. ccl 连通区域 ──
        auto [c1, c1ms] = timedMs([&] { return cclL.Execute(*m1.d_cleanedMask, stream); });
        auto [c2, c2ms] = timedMs([&] { return cclR.Execute(*m2.d_cleanedMask, stream); });
        tCcl = c1ms + c2ms;
        ASSERT_TRUE(c1.success && c2.success) << stem << ": ccl fail";
        if (!warmup) {
            cv::Mat vL, vR;
            c1.d_labeledMask->download(vL);
            c2.d_labeledMask->download(vR);
            saveLabelVis(o + "_L_ccl.png", vL);
            saveLabelVis(o + "_R_ccl.png", vR);
        }

        // ── 3. label 编号 ──
        auto [l1, l1ms] = timedMs([&] { return labL.Execute(*c1.d_labeledMask, stream); });
        auto [l2, l2ms] = timedMs([&] { return labR.Execute(*c2.d_labeledMask, stream); });
        tLab = l1ms + l2ms;
        ASSERT_TRUE(l1.success && l2.success) << stem << ": label fail";
        if (!warmup) {
            cv::Mat vL, vR;
            l1.d_labeledMask->download(vL);
            l2.d_labeledMask->download(vR);
            saveLabelVis(o + "_L_label.png", vL);
            saveLabelVis(o + "_R_label.png", vR);
        }

        // ── 4. steger 中心提取 ──
        auto [s1, s1ms] = timedMs([&] {
            return stgL.Execute(*m1.d_grayImage, *l1.d_labeledMask, stream, GroupMode::ByLabel); });
        auto [s2, s2ms] = timedMs([&] {
            return stgR.Execute(*m2.d_grayImage, *l2.d_labeledMask, stream, GroupMode::ByLabel); });
        tStg = s1ms + s2ms;
        ASSERT_TRUE(s1.success && s2.success) << stem << ": steger fail: " << s1.message;
        log << "steger: L=" << s1.totalPointCount << " R=" << s2.totalPointCount
            << " lines=" << s1.lineCount << "/" << s2.lineCount << "\n";
        if (!warmup) {
            cv::Mat gL, gR;
            m1.d_grayImage->download(gL);
            m2.d_grayImage->download(gR);
            savePointsOnGray(o + "_L_steger.png", gL, downloadPoints(*s1.d_centerPoints));
            savePointsOnGray(o + "_R_steger.png", gR, downloadPoints(*s2.d_centerPoints));
        }

        // ── 5. undistort ──
        auto [u1, u1ms] = timedMs([&] { return unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream); });
        auto [u2, u2ms] = timedMs([&] { return unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream); });
        tUn = u1ms + u2ms;
        ASSERT_TRUE(u1.success && u2.success) << stem << ": undistort fail";
        log << "undistort: L=" << u1.d_rectifiedPoints->cols << " R=" << u2.d_rectifiedPoints->cols << "\n";
        if (!warmup) {
            savePointsOnCanvas(o + "_L_undist.png", canvas, downloadPoints(*u1.d_rectifiedPoints), cv::Scalar(255, 255, 255));
            savePointsOnCanvas(o + "_R_undist.png", canvas, downloadPoints(*u2.d_rectifiedPoints), cv::Scalar(255, 255, 255));
        }

        // ── 6. interp 编号插值 ──
        auto [e1, e1ms] = timedMs([&] { return epiL.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream); });
        auto [e2, e2ms] = timedMs([&] { return epiR.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream); });
        tEpi = e1ms + e2ms;
        ASSERT_TRUE(e1.success && e2.success) << stem << ": interp fail: " << e1.message;
        log << "interp: L=" << e1.interpCount << " R=" << e2.interpCount << "\n";
        if (!warmup) {
            savePointsOnCanvas(o + "_L_interp.png", canvas, downloadPoints(*e1.d_interpPoints), cv::Scalar(0, 255, 0));
            savePointsOnCanvas(o + "_R_interp.png", canvas, downloadPoints(*e2.d_interpPoints), cv::Scalar(0, 255, 0));
        }

        // ── 7. match v1 编号匹配（仅输出匹配率） ──
        const int nL = e1.d_interpPoints->cols;
        auto [mr, mms] = timedMs([&] {
            return matcher.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                    *e2.d_interpPoints, *e2.d_interp_line_ids, stream); });
        tMatch = mms;
        ASSERT_TRUE(mr.success) << stem << ": match fail: " << mr.message;
        const double rate = nL > 0 ? 100.0 * mr.matchedCount / nL : 0.0;
        log << "match_v1: in=" << nL << " matched=" << mr.matchedCount
            << " rate=" << std::fixed << std::setprecision(2) << rate << "%"
            << " (exclL=" << mr.excludedLeftCount << ", exclR=" << mr.excludedRightCount << ")\n";

        // ── 8. recon ──
        if (mr.matchedCount > 0) {
            auto [rr, rms] = timedMs([&] {
                return recon.Execute(*mr.d_matched_left, *mr.d_matched_right,
                                      *mr.d_matched_line_ids, h.Q, stream); });
            tRecon = rms;
            ASSERT_TRUE(rr.success) << stem << ": recon fail: " << rr.message;
            if (!warmup) {
                cv::Mat pts3, ids3;
                rr.d_points3d->download(pts3);
                rr.d_valid_line_ids->download(ids3);
                stream.waitForCompletion();
                pts3 = pts3.reshape(3, 1);
                ids3 = ids3.reshape(1, 1);
                std::vector<cv::Vec3f> vp(pts3.begin<cv::Vec3f>(), pts3.end<cv::Vec3f>());
                std::vector<int> vi(ids3.begin<int>(), ids3.end<int>());
                writePlyAscii(o + "_recon.ply", vp, vi,
                              [](int id) { return hsvToBgr(lidColor(id)); });
                log << "recon: pts3d=" << vp.size() << "\n";
            }
        } else {
            log << "recon: skipped (0 matched)\n";
        }

        log << std::fixed << std::setprecision(2)
            << "timing_ms: mask=" << tMask << " ccl=" << tCcl << " label=" << tLab
            << " steger=" << tStg << " undist=" << tUn << " interp=" << tEpi
            << " match=" << tMatch << " recon=" << tRecon << "\n";
        if (!warmup) {
            std::cout << "\n[" << stem << "]\n" << log.str();
            summary << "[" << stem << "]\n" << log.str();
        }
    };

    ASSERT_GE(input->poseFrames.size(), 2u) << "need >=2 pose frames";
    processFrame(0, 0, true);   // 预热
    processFrame(0, 0, false);  // 帧 1: pose_06
    processFrame(1, 0, false);  // 帧 2: pose_07

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    std::cout << "\ncalib viz outputs -> " << outDir.string() << "\n";
    SUCCEED();
}
