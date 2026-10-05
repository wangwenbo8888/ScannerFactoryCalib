// test_scan_pipeline_fast.cpp — 扫描模式全链流水线测试（无 ccl/label, 双模式 interp）
//
// 链路: mask_extract → steger_fast(Flat) → undistort_cuda → epipolar_interp_dual(扫描模式)
//       → laser_match_scan_v2 → laser_reconstruct
// 输入: data_in/left_skew + data_out/left_skew_calib.json（参数设置同 tools/fcscan_e2e.cpp）
// 输出: data_out/scane2e_fast/{每帧 PLY, all_scan.ply, timing.csv}
// 说明: 无线号链——interp 按行 x 聚类+就近配对（输出 fid=0），match_v2 纯几何查表匹配。
//       与 fcscan 基线（有号链 63.5%）的差异仅报告不判失败（无 labeler 噪声过滤+无线号辅助）。
// 缺数据 → GTEST_SKIP。

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "steger_fast.h"                 // 优化版（同契约, 逐位一致）
#include "undistort_points_cuda.h"
#include "epipolar_interp_dual_cuda.h"    // ★ 双模式 interp（无号→扫描模式）
#include "curve_map.h"
#include "laser_match_scan_v2_cuda.h"    // ★ 优化版 v2
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

} // namespace

TEST(ScanPipelineFast, EndToEndWithOptimizedOps) {
    spdlog::set_level(spdlog::level::warn);

    fs::path dataDir, calibJson;
    if (!findData(dataDir, calibJson)) {
        GTEST_SKIP() << "left_skew data not found; skip fast pipeline e2e";
    }
    const fs::path outDir = dataDir.parent_path().parent_path() / "data_out" / "scane2e_fast";
    fs::create_directories(outDir);
    std::cout << "data: " << dataDir.string() << "\nout: " << outDir.string() << "\n";

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

    // ---------- 3. 算子装配（优化版） ----------
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
    StegerExtractorFast stgL(sp), stgR(sp);   // ★ 优化版

    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams upL;
    upL.cameraMatrix = h.cameraMatrixL; upL.distCoeffs = h.distCoeffsL;
    upL.R = h.R1; upL.P = P1_3x3; upL.validate();
    UndistortPointsParams upR;
    upR.cameraMatrix = h.cameraMatrixR; upR.distCoeffs = h.distCoeffsR;
    upR.R = h.R2; upR.P = P2_3x3; upR.validate();
    UndistortPointsCuda unL(upL), unR(upR);

    EpipolarInterpDualParams dip;   // ★ Auto：Flat 单线号 → 扫描模式
    dip.epipolar_row_step = rowStep;
    EpipolarInterpDualCuda epiL(dip), epiR(dip);

    LaserMatchScanParamsV2 msp;             // ★ v2
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

    // ---------- 4. 逐帧处理（首帧预热一遍全链后再计时） ----------
    enum OpIdx { OP_MASK = 0, OP_STEGER, OP_UNDIST, OP_INTERP, OP_MATCH, OP_RECON, OP_COUNT };
    const char* opNames[OP_COUNT] = { "mask", "steger_fast", "undistort", "interp_dual", "match_v2", "recon" };

    double accMs[OP_COUNT] = { 0 };
    int accFrames = 0;
    long totalIn = 0, totalMatched = 0, totalPts = 0;
    std::vector<cv::Vec3f> allPts;
    std::vector<int> allIds;
    std::ofstream tcsv((outDir / "timing.csv").string());
    tcsv << "frame,mask_ms,steger_ms,undist_ms,interp_ms,match_ms,recon_ms,matched\n";

    // 基线 matched（fcscan_e2e 输出, 存在则逐帧校验）
    std::map<std::string, int> baseline;
    {
        const fs::path baseCsv = dataDir.parent_path().parent_path() / "data_out" / "scane2e" / "timing.csv";
        std::ifstream bf(baseCsv.string());
        if (bf.good()) {
            std::string line;
            std::getline(bf, line);   // header
            while (std::getline(bf, line)) {
                std::stringstream ss(line);
                std::string stem; std::string v;
                std::vector<std::string> cols;
                std::getline(ss, stem, ',');
                while (std::getline(ss, v, ',')) cols.push_back(v);
                if (!cols.empty() && cols.size() >= 9)
                    baseline[stem] = std::atoi(cols[8].c_str());
            }
            std::cout << "baseline csv loaded: " << baseline.size() << " frames\n";
        } else {
            std::cout << "no baseline csv; skip per-frame matched check\n";
        }
    }

    int mismatchFrames = 0;

    auto processFrame = [&](size_t pi, size_t ti, bool timed) -> bool {
        const auto& img = input->poseFrames[pi][ti];
        const std::string stem = input->poseDirs[pi] + "_tube" + std::to_string(ti);
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
        if (!m1.success || !m2.success || !m1.d_cleanedMask || !m2.d_cleanedMask) return false;

        // 无 ccl/label：steger Flat 直接消费二值掩膜
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
            std::cout << stem << ": steger fail (" << s1.message << " | " << s2.message << ")\n"; return false; }

        auto u1 = tExec(OP_UNDIST, unL, *s1.d_centerPoints, *s1.d_line_ids);
        auto u2 = tExec(OP_UNDIST, unR, *s2.d_centerPoints, *s2.d_line_ids);
        if (!u1.success || !u2.success || !u1.d_rectifiedPoints || !u2.d_rectifiedPoints) {
            std::cout << stem << ": undistort fail (" << u1.message << " | " << u2.message << ")\n"; return false; }
        auto e1 = tExec(OP_INTERP, epiL, *u1.d_rectifiedPoints, *u1.d_line_ids);
        auto e2 = tExec(OP_INTERP, epiR, *u2.d_rectifiedPoints, *u2.d_line_ids);
        if (!e1.success || !e2.success || !e1.d_interpPoints || !e2.d_interpPoints) {
            std::cout << stem << ": interp fail (" << e1.message << " | " << e2.message << ")\n"; return false; }
        if (!e1.d_interp_line_ids || !e2.d_interp_line_ids) {
            std::cout << stem << ": interp no line ids\n"; return false; }

        const int nL = e1.d_interpPoints->cols;

        auto mr = tExec(OP_MATCH, matcher,
                        *e1.d_interpPoints, *e1.d_interp_line_ids,
                        *e2.d_interpPoints, *e2.d_interp_line_ids);
        if (!mr.success) {
            std::cout << stem << ": match fail (msg=" << mr.message << ")\n"; return false;
        }
        if (mr.matchedCount == 0) {
            // 如实记录前段耗时（recon 未执行计 0）
            if (timed) {
                tcsv << stem;
                for (int k = 0; k < OP_COUNT; ++k) tcsv << ',' << sm[k];
                tcsv << ',' << mr.matchedCount << '\n';
                for (int k = 0; k < OP_COUNT; ++k) accMs[k] += sm[k];
                ++accFrames;
                double tot = 0;
                for (int k = 0; k < OP_COUNT; ++k) tot += sm[k];
                std::cout << stem << ": in=" << nL << " interp="
                          << e1.d_interpPoints->cols << "/" << e2.d_interpPoints->cols
                          << " matched=0 (exclL=" << mr.excludedLeftCount
                          << ") | total(partial)=" << tot << "ms\n";
            }
            return false;
        }

        auto rr = tExec(OP_RECON, recon, *mr.d_matched_left, *mr.d_matched_right,
                        *mr.d_matched_line_ids, h.Q);
        if (!rr.success || !rr.d_points3d) {
            std::cout << stem << ": recon fail (" << rr.message << ")\n"; return false; }

        cv::Mat pts3, ids3;
        rr.d_points3d->download(pts3);
        rr.d_valid_line_ids->download(ids3);
        stream.waitForCompletion();
        pts3 = pts3.reshape(3, 1);
        ids3 = ids3.reshape(1, 1);
        std::vector<cv::Vec3f> vp(pts3.begin<cv::Vec3f>(), pts3.end<cv::Vec3f>());
        std::vector<int> vi(ids3.begin<int>(), ids3.end<int>());

        if (timed) {
            totalIn += nL;
            totalMatched += mr.matchedCount;
            totalPts += vp.size();
            allPts.insert(allPts.end(), vp.begin(), vp.end());
            allIds.insert(allIds.end(), vi.begin(), vi.end());
            writePlyAscii((outDir / (stem + "_scan.ply")).string(), vp, vi,
                          [](int id) { return hsvToBgr(lidColor(id)); });
            tcsv << stem;
            for (int k = 0; k < OP_COUNT; ++k) tcsv << ',' << sm[k];
            tcsv << ',' << mr.matchedCount << '\n';
            for (int k = 0; k < OP_COUNT; ++k) accMs[k] += sm[k];
            ++accFrames;

            double tot = 0;
            for (int k = 0; k < OP_COUNT; ++k) tot += sm[k];
            std::ostringstream oss;
            for (int k = 0; k < OP_COUNT; ++k) oss << opNames[k] << '=' << sm[k] << ' ';
            std::cout << stem << ": in=" << nL << " matched=" << mr.matchedCount
                      << " pts=" << vp.size() << " | " << oss.str() << "| total=" << tot << "ms\n";

            auto it = baseline.find(stem);
            if (it != baseline.end() && it->second != mr.matchedCount) {
                ++mismatchFrames;
                std::cout << stem << ": matched diff vs fcscan baseline (no-label="
                          << mr.matchedCount << " vs " << it->second
                          << ", delta=" << (mr.matchedCount - it->second) << ")\n";
            }
        }
        return true;
    };

    // 预热：首帧全链跑一遍（分配/JIT/首次触达）
    if (!input->poseFrames.empty() && !input->poseFrames[0].empty())
        processFrame(0, 0, false);

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi)
        for (size_t ti = 0; ti < input->poseFrames[pi].size(); ++ti)
            processFrame(pi, ti, true);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    // ---------- 5. 汇总 ----------
    ASSERT_GT(accFrames, 0) << "no frames processed";
    if (!allPts.empty())
        writePlyAscii((outDir / "all_scan.ply").string(), allPts, allIds,
                      [](int id) { return hsvToBgr(lidColor(id)); });

    std::cout << "\n==== avg gpu-compute ms over " << accFrames << " frames (optimized ops) ====\n";
    double grand = 0;
    for (int k = 0; k < OP_COUNT; ++k) {
        std::cout << "  " << std::left << std::setw(11) << opNames[k]
                  << std::fixed << std::setprecision(2) << accMs[k] / accFrames << " ms\n";
        grand += accMs[k];
    }
    std::cout << "  " << std::left << std::setw(11) << "TOTAL"
              << grand / accFrames << " ms\n";
    std::cout << "==== e2e: input=" << totalIn << " matched=" << totalMatched
              << " (" << (totalIn > 0 ? 100.0 * totalMatched / totalIn : 0.0) << "%) pts3d=" << totalPts
              << " ====\n";
    std::cout << "merged cloud -> " << (outDir / "all_scan.ply").string()
              << " (" << allPts.size() << " pts)\n";
    if (!baseline.empty())
        std::cout << "baseline mismatch frames: " << mismatchFrames << "\n";

    if (!baseline.empty())
        std::cout << "frames differing from fcscan baseline (expected: label removed "
                  << "-> line-id semantics changed): " << mismatchFrames << "/" << baseline.size() << "\n";
}
