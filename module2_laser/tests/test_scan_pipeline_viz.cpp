// test_scan_pipeline_viz.cpp — 逐步骤可视化诊断测试（无 ccl/label 链, 仅前 2 帧）
//
// 链路: mask_extract → steger_fast(Flat) → undistort_cuda → epipolar_interp_opt
//       → laser_match_scan_v2(仅输出匹配率) → laser_reconstruct
// 输入: data_in/left_skew 前 2 帧 + data_out/left_skew_calib.json
// 输出: data_out/scane2e_viz/
//   {stem}_{L|R}_mask.png     清洗后掩膜
//   {stem}_{L|R}_steger.png   灰度图 + 亚像素中心点(红)
//   {stem}_{L|R}_undist.png   矫正后点集(白)
//   {stem}_{L|R}_interp.png   极线插值点集(绿)
//   {stem}_recon.ply          三维点云(按线号着色)
//   summary.txt               各步骤点数/匹配率/耗时
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
#include "steger_fast.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_dual_cuda.h"    // ★ 双模式 interp（无号→扫描模式）
#include "curve_map.h"
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

// 灰度图上画点集（红色）
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

// 黑 canvas 上画点集
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

TEST(ScanPipelineViz, TwoFramesStepOutputs) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir, calibJson;
    if (!findData(dataDir, calibJson)) {
        GTEST_SKIP() << "left_skew data not found; skip viz test";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "scane2e_viz";
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

    // ---------- 算子装配 ----------
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

    StegerParams sp;
    StegerExtractorFast stgL(sp), stgR(sp);

    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3x3; upL.validate();
    UndistortPointsParams upR;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3x3; upR.validate();
    UndistortPointsCuda unL(upL), unR(upR);

    EpipolarInterpDualParams dip;   // ★ Auto 模式：Flat 单线号 → 扫描（无号聚类插值）
    dip.epipolar_row_step = rowStep;
    EpipolarInterpDualCuda epiL(dip), epiR(dip);

    LaserMatchScanParamsV2 msp;
    msp.epipolar_row_step = rowStep;
    msp.match_threshold = 2.0f;
    msp.vL_tolerance = rowStep;
    LaserMatchScanCudaV2 matcher(msp);
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

        // ── Step 1: mask ──
        auto [m1, tMaskL] = timedMs([&] { return maskL.Execute(img.leftGray, stream); });
        auto [m2, tMaskR] = timedMs([&] { return maskR.Execute(img.rightGray, stream); });
        ASSERT_TRUE(m1.success && m2.success) << stem << ": mask fail";
        long maskPxL = -1, maskPxR = -1;
        if (!warmup) {
            cv::Mat maskVisL, maskVisR;
            m1.d_cleanedMask->download(maskVisL);
            m2.d_cleanedMask->download(maskVisR);
            maskPxL = cv::countNonZero(maskVisL);
            maskPxR = cv::countNonZero(maskVisR);
            cv::imwrite(o + "_L_mask.png", maskVisL);
            cv::imwrite(o + "_R_mask.png", maskVisR);
        }

        // ── Step 2: steger_fast (Flat, 直接吃二值掩膜) ──
        auto [s1, tStgL] = timedMs([&] { return stgL.Execute(*m1.d_grayImage, *m1.d_cleanedMask, stream, GroupMode::Flat); });
        auto [s2, tStgR] = timedMs([&] { return stgR.Execute(*m2.d_grayImage, *m2.d_cleanedMask, stream, GroupMode::Flat); });
        ASSERT_TRUE(s1.success && s2.success) << stem << ": steger fail: " << s1.message;
        log << "steger: L=" << s1.totalPointCount << " R=" << s2.totalPointCount << "\n";
        if (!warmup) {
            cv::Mat grayL, grayR;
            m1.d_grayImage->download(grayL);
            m2.d_grayImage->download(grayR);
            savePointsOnGray(o + "_L_steger.png", grayL, downloadPoints(*s1.d_centerPoints));
            savePointsOnGray(o + "_R_steger.png", grayR, downloadPoints(*s2.d_centerPoints));
        }

        // ── Step 3: undistort ──
        auto [u1, tUnL] = timedMs([&] { return unL.Execute(*s1.d_centerPoints, *s1.d_line_ids, stream); });
        auto [u2, tUnR] = timedMs([&] { return unR.Execute(*s2.d_centerPoints, *s2.d_line_ids, stream); });
        ASSERT_TRUE(u1.success && u2.success) << stem << ": undistort fail";
        log << "undistort: L=" << u1.d_rectifiedPoints->cols << " R=" << u2.d_rectifiedPoints->cols << "\n";
        if (!warmup) {
            savePointsOnCanvas(o + "_L_undist.png", canvas, downloadPoints(*u1.d_rectifiedPoints), cv::Scalar(255, 255, 255));
            savePointsOnCanvas(o + "_R_undist.png", canvas, downloadPoints(*u2.d_rectifiedPoints), cv::Scalar(255, 255, 255));
        }

        // ── Step 4: interp_opt ──
        auto [e1, tEpiL] = timedMs([&] { return epiL.Execute(*u1.d_rectifiedPoints, *u1.d_line_ids, stream); });
        auto [e2, tEpiR] = timedMs([&] { return epiR.Execute(*u2.d_rectifiedPoints, *u2.d_line_ids, stream); });
        ASSERT_TRUE(e1.success && e2.success) << stem << ": interp fail: " << e1.message;
        log << "interp: L=" << e1.interpCount << " R=" << e2.interpCount << "\n";
        if (!warmup) {
            savePointsOnCanvas(o + "_L_interp.png", canvas, downloadPoints(*e1.d_interpPoints), cv::Scalar(0, 255, 0));
            savePointsOnCanvas(o + "_R_interp.png", canvas, downloadPoints(*e2.d_interpPoints), cv::Scalar(0, 255, 0));
        }

        // ── Step 5: match_v2 —— 仅输出匹配率 ──
        const int nL = e1.d_interpPoints->cols;
        auto [mr, tMatch] = timedMs([&] {
            return matcher.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                                    *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
        });
        ASSERT_TRUE(mr.success) << stem << ": match fail: " << mr.message;
        const double rate = nL > 0 ? 100.0 * mr.matchedCount / nL : 0.0;
        log << "match_v2: in=" << nL << " matched=" << mr.matchedCount
            << " rate=" << std::fixed << std::setprecision(2) << rate << "%"
            << " (exclL=" << mr.excludedLeftCount << ", exclR=" << mr.excludedRightCount << ")\n";

        // ── Step 6: recon ──
        double tRecon = 0;
        if (mr.matchedCount > 0) {
            auto [rr, ms] = timedMs([&] {
                return recon.Execute(*mr.d_matched_left, *mr.d_matched_right,
                                      *mr.d_matched_line_ids, h.Q, stream);
            });
            tRecon = ms;
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
            << "timing_ms: mask=" << (tMaskL + tMaskR)
            << " steger=" << (tStgL + tStgR)
            << " undist=" << (tUnL + tUnR)
            << " interp=" << (tEpiL + tEpiR)
            << " match=" << tMatch << " recon=" << tRecon << "\n";
        if (!warmup) {
            std::cout << "\n[" << stem << "]\n" << log.str();
            summary << "[" << stem << "]\n" << log.str();
        }
    };

    // 预热 1 帧后正式处理前 2 帧
    ASSERT_GE(input->poseFrames.size(), 2u) << "need >=2 pose frames";
    processFrame(0, 0, true);
    processFrame(0, 0, false);
    processFrame(1, 0, false);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    std::cout << "\nviz outputs -> " << outDir.string() << "\n";
    SUCCEED();
}
