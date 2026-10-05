// test_pose06_chain_v3_viz.cpp — pose_06 V3 全链逐步骤可视化测试（无 CCL）
//
// 链路: mask_extract → steger_fast(Flat) → undistort_cuda → epipolar_interp_dual
//       (Auto→Scan, 段ID) → laser_match_scan_v3(段级投票) → laser_reconstruct
// 输入: data_in/left_skew（自包含：config.json + camera_calib.json
//       + curve_map_temp_table.bin CMTT sidecar——按温选档，无现场生成，
//       不依赖 data_out 标定 JSON）
// 输出: data_out/pose06_chainv3/
//   01_mask_{L|R}.png        清洗后掩膜
//   02_steger_{L|R}.png      灰度图 + 亚像素中心点(红)
//   03_undistort_{L|R}.png   矫正图 + 矫正后点集(蓝)
//   04_interp_{L|R}.png      插值点集(黑底, 按段/线 ID 着色)
//   05_match_pairs.png       并排矫正图匹配连线(track ID 着色)
//   06_cloud.ply             三维点云(track ID 着色)
//   timing.txt               各步 GPU event 计时
// 门禁: matchedCount/cloudPts >= 8000 (pose_06 基线 14004)
// 缺数据 → GTEST_SKIP。

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "steger_fast.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_dual_cuda.h"
#include "curve_map.h"
#include "laser_match_scan_v3_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "common/calib_result_types.h"
#include "common/cmtt_tier_select.h"

using namespace fc;
using namespace calib;

namespace fs = std::filesystem;

namespace {

bool findData(fs::path& dataDir) {
    const char* envDir = std::getenv("FC2_LEFT_SKEW");
    std::vector<fs::path> dirCandidates;
    if (envDir) dirCandidates.emplace_back(envDir);
    for (const char* rel : {"../../data_in/left_skew", "../../../data_in/left_skew",
                            "../../../../data_in/left_skew"}) {
        dirCandidates.emplace_back(fs::path(rel));
    }
    dirCandidates.emplace_back("E:/JEAMMWARE260705/factory_calib/data_in/left_skew");
    for (auto& d : dirCandidates) {
        if (fs::exists(d / "config.json") && fs::exists(d / "camera_calib.json") &&
            fs::exists(d / "curve_map_temp_table.bin")) {
            dataDir = d;
            return true;
        }
    }
    return false;
}

cv::Vec3b labelColor(int label) {
    if (label <= 0) return {0, 0, 0};
    const int h = (label * 47) % 180;
    cv::Mat m(1, 1, CV_8UC3, cv::Scalar(h, 255, 255));
    cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
    return m.at<cv::Vec3b>(0, 0);
}

void savePts(const std::string& path, const cv::Mat& base,
             const std::vector<cv::Point2f>& pts, const cv::Scalar& color) {
    cv::Mat img;
    if (base.channels() == 1) cv::cvtColor(base, img, cv::COLOR_GRAY2BGR);
    else img = base.clone();
    for (auto& p : pts)
        cv::circle(img, {cvRound(p.x), cvRound(p.y)}, 1, color, -1);
    cv::imwrite(path, img);
}

std::vector<cv::Point2f> dlPts(const cv::cuda::GpuMat& g, cv::cuda::Stream& s) {
    cv::Mat m;
    g.download(m, s);
    s.waitForCompletion();
    if (m.empty()) return {};
    m = m.reshape(2, 1);
    return std::vector<cv::Point2f>(m.begin<cv::Point2f>(), m.end<cv::Point2f>());
}

struct StepTime { std::string name; float ms; };

} // namespace

TEST(Pose06ChainV3Viz, StepOutputsAndLooseGate) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir;
    if (!findData(dataDir)) {
        GTEST_SKIP() << "left_skew data not found; skip pose_06 chain viz test";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "pose06_chainv3";
    fs::create_directories(outDir);
    const std::string pre = (outDir / "").string();

    // ---------- 载入输入 + pose_06 定位（按名查找, 不写死下标） ----------
    auto inputOpt = loadLaserInput(dataDir.string());
    ASSERT_TRUE(inputOpt.has_value());
    const auto& input = inputOpt.value();
    const auto& cfg = input.config;
    const auto& h = input.handoff;
    size_t p06 = SIZE_MAX;
    for (size_t i = 0; i < input.poseDirs.size(); ++i)
        if (input.poseDirs[i] == "pose_06") { p06 = i; break; }
    ASSERT_NE(p06, SIZE_MAX) << "pose_06 not found under " << dataDir.string();
    ASSERT_FALSE(input.poseFrames[p06].empty());
    const auto& img = input.poseFrames[p06][0];

    // ---------- curve map 表：CMTT sidecar 按温选档（同 fcchain_v3 首选路径） ----------
    const fs::path sidecar = dataDir / "curve_map_temp_table.bin";
    CmttTierChoice choice = selectCmttTier(sidecar.string(), cfg.referenceTemp);
    ASSERT_TRUE(choice.ok) << "CMTT sidecar: " << choice.why;
    CurveMapTable cmt;
    std::string lerr;
    ASSERT_TRUE(cmt.Load(choice.blob.data(), choice.blob.size(), lerr)) << lerr;
    const float rowStep = cmt.rowStep();
    const double tableTempC = choice.tierTemp;   // 注入档温（选档结果）
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
        tempTable->referenceTemperature = tableTempC;
        tempTable->table[tableTempC] = std::move(pm);
    }

    // ---------- 算子装配（参数逐字取自 fcchain_v3） ----------
    cv::cuda::Stream stream;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
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

    EpipolarInterpDualParams dip;   // Auto→Scan（Flat 单线号）; scan_link_dx 默认 3.0 → 段 ID
    dip.epipolar_row_step = rowStep;
    EpipolarInterpDualCuda epiL(dip), epiR(dip);

    LaserMatchScanParamsV3 vp3;
    vp3.epipolar_row_step = rowStep;
    vp3.vL_tolerance = 0.7f;
    vp3.match_threshold = 2.0f;
    LaserMatchScanCudaV3 matcher(vp3);
    ASSERT_TRUE(matcher.SetTempTable(tempTable));
    matcher.SetCurrentTemperature(tableTempC);

    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    // ---------- 链路执行（warmup 不落盘 → 正式跑落盘） ----------
    std::vector<StepTime> times;
    auto timed = [&times](const char* name, cudaStream_t cs_, auto&& fn) {
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        cudaEventRecord(s, cs_);
        auto r = fn();
        cudaEventRecord(e, cs_);
        cudaEventSynchronize(e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        times.push_back({name, ms});
        return r;
    };

    struct ChainOut {
        bool ok = false;
        std::string err;
        int stgL = 0, stgR = 0;
        int intL = 0, intR = 0;
        int matchedCount = 0, trackTotal = 0, trackDropped = 0;
        int cloudPts = 0;
    };

    auto runChain = [&](bool save) -> ChainOut {
        ChainOut o;
        auto tMask  = timed("mask", cs, [&] { return maskL.Execute(img.leftGray, stream); });
        auto tMaskR = timed("mask(R)", cs, [&] { return maskR.Execute(img.rightGray, stream); });
        if (!tMask.success || !tMaskR.success) { o.err = "mask failed"; return o; }

        auto tStg = timed("steger_fast", cs, [&] {
            return stgL.Execute(*tMask.d_grayImage, *tMask.d_cleanedMask, stream, GroupMode::Flat); });
        auto tStgR = timed("steger_fast(R)", cs, [&] {
            return stgR.Execute(*tMaskR.d_grayImage, *tMaskR.d_cleanedMask, stream, GroupMode::Flat); });
        if (!tStg.success || !tStgR.success) {
            o.err = "steger_fast: " + tStg.message + " / " + tStgR.message; return o;
        }
        o.stgL = tStg.totalPointCount;
        o.stgR = tStgR.totalPointCount;

        auto tUn = timed("undistort", cs, [&] {
            return unL.Execute(*tStg.d_centerPoints, *tStg.d_line_ids, stream); });
        auto tUnR = timed("undistort(R)", cs, [&] {
            return unR.Execute(*tStgR.d_centerPoints, *tStgR.d_line_ids, stream); });
        if (!tUn.success || !tUnR.success) { o.err = "undistort failed"; return o; }

        auto tEpi = timed("interp_dual", cs, [&] {
            return epiL.Execute(*tUn.d_rectifiedPoints, *tUn.d_line_ids, stream); });
        auto tEpiR = timed("interp_dual(R)", cs, [&] {
            return epiR.Execute(*tUnR.d_rectifiedPoints, *tUnR.d_line_ids, stream); });
        if (!tEpi.success || !tEpiR.success) {
            o.err = "interp_dual: " + tEpi.message + " / " + tEpiR.message; return o;
        }
        o.intL = tEpi.interpCount;
        o.intR = tEpiR.interpCount;

        auto tMatch = timed("match_v3", cs, [&] {
            return matcher.Execute(*tEpi.d_interpPoints, *tEpi.d_interp_line_ids,
                                   *tEpiR.d_interpPoints, *tEpiR.d_interp_line_ids, stream); });
        if (!tMatch.success) { o.err = "match_v3: " + tMatch.message; return o; }
        o.matchedCount = tMatch.matchedCount;
        o.trackTotal = tMatch.trackTotal;
        o.trackDropped = tMatch.trackDropped;

        auto tRec = timed("reconstruct", cs, [&] {
            return recon.Execute(*tMatch.d_matched_left, *tMatch.d_matched_right,
                                 *tMatch.d_matched_line_ids, h.Q, stream); });
        if (!tRec.success) { o.err = "reconstruct: " + tRec.message; return o; }

        if (!save) { o.ok = true; return o; }

        // ---- 01 mask ----
        {
            cv::Mat mL, mR;
            tMask.d_cleanedMask->download(mL, stream);
            tMaskR.d_cleanedMask->download(mR, stream);
            stream.waitForCompletion();
            cv::imwrite(pre + "01_mask_L.png", mL);
            cv::imwrite(pre + "01_mask_R.png", mR);
        }
        // ---- 02 steger centers on raw gray ----
        {
            cv::Mat gL, gR;
            tMask.d_grayImage->download(gL, stream);
            tMaskR.d_grayImage->download(gR, stream);
            stream.waitForCompletion();
            savePts(pre + "02_steger_L.png", gL, dlPts(*tStg.d_centerPoints, stream), {0, 0, 255});
            savePts(pre + "02_steger_R.png", gR, dlPts(*tStgR.d_centerPoints, stream), {0, 0, 255});
        }
        // ---- 03 undistorted pts on rectified image ----
        cv::Mat recL, recR;
        {
            cv::Mat mxL, myL, mxR, myR;
            cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                        h.imageSize, CV_32FC1, mxL, myL);
            cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                        h.imageSize, CV_32FC1, mxR, myR);
            cv::remap(img.leftGray, recL, mxL, myL, cv::INTER_LINEAR);
            cv::remap(img.rightGray, recR, mxR, myR, cv::INTER_LINEAR);
            savePts(pre + "03_undistort_L.png", recL, dlPts(*tUn.d_rectifiedPoints, stream), {255, 0, 0});
            savePts(pre + "03_undistort_R.png", recR, dlPts(*tUnR.d_rectifiedPoints, stream), {255, 0, 0});
        }
        // ---- 04 interp pts colored by segment/line id ----
        for (int side = 0; side < 2; ++side) {
            const auto& e = side ? tEpiR : tEpi;
            auto pts = dlPts(*e.d_interpPoints, stream);
            cv::Mat fids;
            e.d_interp_line_ids->download(fids, stream);
            stream.waitForCompletion();
            fids = fids.reshape(1, 1);
            cv::Mat image(h.imageSize, CV_8UC3, cv::Scalar(0, 0, 0));
            for (size_t i = 0; i < pts.size(); ++i) {
                const int tid = fids.at<int>(static_cast<int>(i));
                const cv::Vec3b c = labelColor((tid * 7) % 180 + 1);
                cv::circle(image, {cvRound(pts[i].x), cvRound(pts[i].y)}, 1,
                           cv::Scalar(c[0], c[1], c[2]), -1);
            }
            cv::imwrite(pre + "04_interp_" + std::string(side ? "R" : "L") + ".png", image);
        }
        // ---- 05 match pairs: rectified L/R side-by-side with lines ----
        {
            cv::Mat mL, mR, mT;
            tMatch.d_matched_left->download(mL, stream);
            tMatch.d_matched_right->download(mR, stream);
            tMatch.d_matched_track_ids->download(mT, stream);
            stream.waitForCompletion();
            mL = mL.reshape(2, 1);
            mR = mR.reshape(2, 1);
            mT = mT.reshape(1, 1);
            const int W = h.imageSize.width, H = h.imageSize.height;
            cv::Mat canvas(H, W * 2 + 10, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::cvtColor(recL, canvas(cv::Rect(0, 0, W, H)), cv::COLOR_GRAY2BGR);
            cv::cvtColor(recR, canvas(cv::Rect(W + 10, 0, W, H)), cv::COLOR_GRAY2BGR);
            const int n = tMatch.matchedCount;
            const int stride = std::max(1, n / 1500);
            for (int k = 0; k < n; k += stride) {
                const cv::Vec2f l = mL.at<cv::Vec2f>(k);
                const cv::Vec2f r = mR.at<cv::Vec2f>(k);
                const cv::Vec3b c = labelColor((mT.at<int>(k) * 7) % 180 + 1);
                cv::line(canvas, {cvRound(l[0]), cvRound(l[1])},
                         {W + 10 + cvRound(r[0]), cvRound(r[1])},
                         cv::Scalar(c[0], c[1], c[2]), 1);
            }
            cv::imwrite(pre + "05_match_pairs.png", canvas);
        }
        // ---- 06 cloud.ply colored by track id ----
        {
            cv::Mat P3, mT;
            tRec.d_points3d->download(P3, stream);
            tMatch.d_matched_track_ids->download(mT, stream);
            stream.waitForCompletion();
            P3 = P3.reshape(3, 1);
            mT = mT.reshape(1, 1);
            const int n = P3.cols;
            std::ofstream f(pre + "06_cloud.ply");
            f << "ply\nformat ascii 1.0\nelement vertex " << n << "\n"
              << "property float x\nproperty float y\nproperty float z\n"
              << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
              << std::fixed << std::setprecision(3);
            for (int k = 0; k < n; ++k) {
                const cv::Vec3f p = P3.at<cv::Vec3f>(k);
                const cv::Vec3b c = labelColor((mT.at<int>(k) * 7) % 180 + 1);
                f << p[0] << " " << p[1] << " " << p[2] << " "
                  << (int)c[0] << " " << (int)c[1] << " " << (int)c[2] << "\n";
            }
            o.cloudPts = n;
        }
        o.ok = true;
        return o;
    };

    ChainOut w = runChain(false);
    ASSERT_TRUE(w.ok) << "warmup chain failed: " << w.err;
    times.clear();
    ChainOut o = runChain(true);
    ASSERT_TRUE(o.ok) << "chain failed: " << o.err;

    // ---------- 宽松数值门禁 ----------
    EXPECT_GT(o.stgL, 0);
    EXPECT_GT(o.stgR, 0);
    EXPECT_GT(o.intL, 0);
    EXPECT_GT(o.intR, 0);
    EXPECT_GE(o.matchedCount, 8000);
    EXPECT_GE(o.cloudPts, 8000);

    // ---------- 产物存在且非空 ----------
    for (const char* name : {"01_mask_L.png", "01_mask_R.png",
                             "02_steger_L.png", "02_steger_R.png",
                             "03_undistort_L.png", "03_undistort_R.png",
                             "04_interp_L.png", "04_interp_R.png",
                             "05_match_pairs.png", "06_cloud.ply"}) {
        const fs::path p = outDir / name;
        EXPECT_TRUE(fs::exists(p)) << name;
        if (fs::exists(p)) EXPECT_GT(fs::file_size(p), 0u) << name;
    }

    // ---------- timing 落盘 + 汇总 ----------
    {
        std::ofstream tf(pre + "timing.txt");
        tf << "step,ms\n";
        float total = 0.f;
        for (auto& t : times) { tf << t.name << "," << t.ms << "\n"; total += t.ms; }
        tf << "TOTAL," << total << "\n";
        std::cout << "\n==== pose_06 chain v3 timing (GPU events) ====\n";
        for (auto& t : times)
            std::cout << std::setw(16) << t.name << " : " << std::fixed
                      << std::setprecision(3) << t.ms << " ms\n";
        std::cout << std::setw(16) << "TOTAL" << " : " << total << " ms\n";
        std::cout << "steger: L=" << o.stgL << " R=" << o.stgR
                  << " interp: L=" << o.intL << " R=" << o.intR << "\n"
                  << "matched=" << o.matchedCount
                  << " tracks=" << o.trackTotal
                  << " dropped=" << o.trackDropped
                  << " cloudPts=" << o.cloudPts << "\n"
                  << "outputs -> " << outDir.string() << "\n";
    }
    SUCCEED();
}
