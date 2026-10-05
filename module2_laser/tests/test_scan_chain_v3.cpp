// test_scan_chain_v3.cpp — V3 全链扫描测试（全姿态纯点云版）
//
// 链路: mask_extract → steger_fast(Flat) → undistort_cuda → epipolar_interp_dual
//       (Auto→Scan, 段ID) → laser_match_scan_v3(段级投票, HIT窗口=4.0) →
//       laser_reconstruct
// 输入: data_in/left_skew（自包含：config.json + camera_calib.json
//       + curve_map_temp_table.bin CMTT sidecar——按温选档, 无现场生成）
// 输出: data_out/scan_chain_v3/{每帧 PLY, all_scan.ply}——仅点云
// 门禁: accFrames>0 且 ≥1 帧 matched>0 且总点数 ≥8000;
//       失败帧仅计数报告不判失败。缺数据 → GTEST_SKIP。
// 历史: HIT 窗口对比 2.0/3.0/4.0（详见 docs/plans/2026-09-04-scan-chain-v3-test-design.md）:
//       2.0=错配搬家+大弃段, 3.0=段1040 邻线错配漏出, 4.0=双弃兜底零错配且量最大。

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

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

} // namespace

TEST(ScanChainV3, FullSweepPointCloudOnly) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir;
    if (!findData(dataDir)) {
        GTEST_SKIP() << "left_skew data not found; skip v3 scan chain test";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "scan_chain_v3";
    fs::create_directories(outDir);
    std::cout << "data: " << dataDir.string() << "\nout: " << outDir.string() << "\n";

    // ---------- 1. 载入输入（图片 + 相机标定参数 via handoff） ----------
    auto inputOpt = loadLaserInput(dataDir.string());
    ASSERT_TRUE(inputOpt.has_value()) << "loadLaserInput failed";
    const auto& input = inputOpt.value();
    const auto& cfg = input.config;
    const auto& h = input.handoff;

    // ---------- 2. 激光映射表: CMTT sidecar 按温选档 ----------
    const fs::path sidecar = dataDir / "curve_map_temp_table.bin";
    CmttTierChoice choice = selectCmttTier(sidecar.string(), cfg.referenceTemp);
    ASSERT_TRUE(choice.ok) << "CMTT sidecar: " << choice.why;
    CurveMapTable cmt;
    std::string lerr;
    ASSERT_TRUE(cmt.Load(choice.blob.data(), choice.blob.size(), lerr)) << lerr;
    const float rowStep = cmt.rowStep();
    const double tableTempC = choice.tierTemp;
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

    // ---------- 3. 算子装配（参数同 fcchain_v3, 唯 HIT 窗口=4.0） ----------
    cv::cuda::Stream stream;
    cudaStream_t cuStream = cv::cuda::StreamAccessor::getStream(stream);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

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
    vp3.match_threshold = 5.0f;   // HIT 窗口（人工指令: 2.0/3.0/4.0 对比后调 5.0）
    LaserMatchScanCudaV3 matcher(vp3);
    ASSERT_TRUE(matcher.SetTempTable(tempTable));
    matcher.SetCurrentTemperature(tableTempC);

    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    // ---------- 4. 逐帧处理（首帧预热; 全链出点云, 无图片/CSV） ----------
    enum OpIdx { OP_MASK = 0, OP_STEGER, OP_UNDIST, OP_INTERP, OP_MATCH, OP_RECON, OP_COUNT };
    const char* opNames[OP_COUNT] = { "mask", "steger_fast", "undistort", "interp_dual", "match_v3", "recon" };

    double accMs[OP_COUNT] = { 0 };
    int accFrames = 0, failedFrames = 0, zeroMatchFrames = 0;
    long totalIn = 0, totalMatched = 0, totalPts = 0, totalDropped = 0;
    std::vector<cv::Vec3f> allPts;
    std::vector<int> allIds;

    auto processFrame = [&](size_t pi, size_t ti, bool timed) -> bool {
        const auto& img = input.poseFrames[pi][ti];
        const std::string stem = input.poseDirs[pi] + "_tube" + std::to_string(ti);
        double sm[OP_COUNT] = { 0 };

        auto tExec = [&](int k, auto&& op, auto&&... args) {
            cudaEventRecord(ev0, cuStream);
            auto r = op.Execute(args..., stream);
            cudaEventRecord(ev1, cuStream);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            sm[k] += ms;
            return r;
        };

        auto m1 = tExec(OP_MASK, maskL, img.leftGray);
        auto m2 = tExec(OP_MASK, maskR, img.rightGray);
        if (!m1.success || !m2.success || !m1.d_cleanedMask || !m2.d_cleanedMask) {
            std::cout << stem << ": mask fail\n";
            ++failedFrames;
            return false;
        }

        cudaEventRecord(ev0, cuStream);
        auto s1 = stgL.Execute(*m1.d_grayImage, *m1.d_cleanedMask, stream, GroupMode::Flat);
        auto s2 = stgR.Execute(*m2.d_grayImage, *m2.d_cleanedMask, stream, GroupMode::Flat);
        cudaEventRecord(ev1, cuStream);
        cudaEventSynchronize(ev1);
        {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            sm[OP_STEGER] += ms;
        }
        if (!s1.success || !s2.success || !s1.d_centerPoints || !s2.d_centerPoints) {
            std::cout << stem << ": steger_fast fail (" << s1.message << " | " << s2.message << ")\n";
            ++failedFrames;
            return false;
        }

        auto u1 = tExec(OP_UNDIST, unL, *s1.d_centerPoints, *s1.d_line_ids);
        auto u2 = tExec(OP_UNDIST, unR, *s2.d_centerPoints, *s2.d_line_ids);
        if (!u1.success || !u2.success || !u1.d_rectifiedPoints || !u2.d_rectifiedPoints) {
            std::cout << stem << ": undistort fail (" << u1.message << " | " << u2.message << ")\n";
            ++failedFrames;
            return false;
        }

        auto e1 = tExec(OP_INTERP, epiL, *u1.d_rectifiedPoints, *u1.d_line_ids);
        auto e2 = tExec(OP_INTERP, epiR, *u2.d_rectifiedPoints, *u2.d_line_ids);
        if (!e1.success || !e2.success || !e1.d_interpPoints || !e2.d_interpPoints ||
            !e1.d_interp_line_ids || !e2.d_interp_line_ids) {
            std::cout << stem << ": interp_dual fail (" << e1.message << " | " << e2.message << ")\n";
            ++failedFrames;
            return false;
        }
        const int nL = e1.d_interpPoints->cols;

        auto mr = tExec(OP_MATCH, matcher,
                        *e1.d_interpPoints, *e1.d_interp_line_ids,
                        *e2.d_interpPoints, *e2.d_interp_line_ids);
        if (!mr.success) {
            std::cout << stem << ": match_v3 fail (msg=" << mr.message << ")\n";
            ++failedFrames;
            return false;
        }
        if (mr.matchedCount == 0) {
            if (timed) {
                ++zeroMatchFrames;
                totalIn += nL;
                std::cout << stem << ": in=" << nL << " matched=0 (tracks=" << mr.trackTotal
                          << " dropped=" << mr.trackDropped << ")\n";
            }
            return false;
        }

        auto rr = tExec(OP_RECON, recon, *mr.d_matched_left, *mr.d_matched_right,
                        *mr.d_matched_line_ids, h.Q);
        if (!rr.success || !rr.d_points3d) {
            std::cout << stem << ": recon fail (" << rr.message << ")\n";
            ++failedFrames;
            return false;
        }

        if (!timed) return true;   // 预热帧: 不统计不落盘

        cv::Mat pts3, tids;
        rr.d_points3d->download(pts3, stream);
        mr.d_matched_track_ids->download(tids, stream);
        stream.waitForCompletion();
        pts3 = pts3.reshape(3, 1);
        tids = tids.reshape(1, 1);
        std::vector<cv::Vec3f> vp(pts3.begin<cv::Vec3f>(), pts3.end<cv::Vec3f>());
        std::vector<int> vi(tids.begin<int>(), tids.end<int>());

        totalIn += nL;
        totalMatched += mr.matchedCount;
        totalPts += static_cast<long>(vp.size());
        totalDropped += mr.trackDropped;
        allPts.insert(allPts.end(), vp.begin(), vp.end());
        allIds.insert(allIds.end(), vi.begin(), vi.end());
        writePlyAscii((outDir / (stem + "_scan.ply")).string(), vp, vi,
                      [](int id) { return hsvToBgr(lidColor(id)); });
        for (int k = 0; k < OP_COUNT; ++k) accMs[k] += sm[k];
        ++accFrames;

        std::cout << stem << ": in=" << nL << " matched=" << mr.matchedCount
                  << " pts=" << vp.size()
                  << " tracks=" << mr.trackTotal << " dropped=" << mr.trackDropped << "\n";
        return true;
    };

    // 预热：首帧全链跑一遍（分配/JIT/首次触达）
    if (!input.poseFrames.empty() && !input.poseFrames[0].empty())
        processFrame(0, 0, false);

    for (size_t pi = 0; pi < input.poseFrames.size(); ++pi)
        for (size_t ti = 0; ti < input.poseFrames[pi].size(); ++ti)
            processFrame(pi, ti, true);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    // ---------- 5. 门禁 + 汇总 ----------
    ASSERT_GT(accFrames, 0) << "no frames processed";
    ASSERT_EQ(failedFrames, 0) << failedFrames << " frames failed at operator stage";
    EXPECT_GT(totalMatched, 0) << "no frame produced any match";
    EXPECT_GE(totalPts, 8000) << "merged cloud too small";

    if (!allPts.empty())
        writePlyAscii((outDir / "all_scan.ply").string(), allPts, allIds,
                      [](int id) { return hsvToBgr(lidColor(id)); });

    std::cout << "\n==== avg gpu-compute ms over " << accFrames << " frames (v3 chain, HIT="
              << vp3.match_threshold << ") ====\n";
    double grand = 0;
    for (int k = 0; k < OP_COUNT; ++k) {
        std::cout << "  " << std::left << std::setw(11) << opNames[k]
                  << std::fixed << std::setprecision(2) << accMs[k] / accFrames << " ms\n";
        grand += accMs[k];
    }
    std::cout << "  " << std::left << std::setw(11) << "TOTAL"
              << grand / accFrames << " ms\n";
    std::cout << "==== v3 full sweep: input=" << totalIn << " matched=" << totalMatched
              << " (" << (totalIn > 0 ? 100.0 * totalMatched / totalIn : 0.0) << "%) pts3d="
              << totalPts << " tracksDropped=" << totalDropped
              << " | zeroMatchFrames=" << zeroMatchFrames
              << " failedFrames=" << failedFrames << " ====\n";
    std::cout << "merged cloud -> " << (outDir / "all_scan.ply").string()
              << " (" << allPts.size() << " pts)\n";
}
