// fcchain_v3.cpp - full scan-chain tool with per-step timing, images and 3D cloud
// chain: mask -> CCL -> steger_fast -> undistort -> interp_dual(track ids)
//        -> laser_match_scan_v3 (card vote + conflict drop-both) -> reconstruct
// outputs: data_out/chainv3/{01..06 step PNGs, cloud.ply, timing.txt}
// table: calib JSON curveMapTempTable meta -> CMTT sidecar tier (selectCmttTier)
//        -> CurveMapTable::Load in-memory (generation skipped); no meta / tier
//        fail -> warn + on-the-fly generation (ref-temp single tier); chain temp
//        injection unified to selected tier temp

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <cctype>
#include <cmath>
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
using json = nlohmann::json;

namespace {

struct StepTime { std::string name; float ms; };
std::vector<StepTime> g_times;

template <typename F>
auto timed(const char* name, cudaStream_t cs, F&& fn) -> decltype(fn()) {
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
    g_times.push_back({name, ms});
    return r;
}

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

} // namespace

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::warn);
    const std::string dataDir = argc > 1 ? argv[1]
        : "E:/JEAMMWARE260705/factory_calib/data_in/left_skew";
    const std::string calibJsonPath = argc > 2 ? argv[2]
        : "E:/JEAMMWARE260705/factory_calib/data_out/laser_calib.json";
    const std::string outDirS = argc > 3 ? argv[3]
        : "E:/JEAMMWARE260705/factory_calib/data_out/chainv3";
    // 可选第 4 参: 目标温度（C）——sidecar 按温选档用; 非数字 → usage 报错
    bool hasTempArg = false;
    double cliTempC = 0.0;
    if (argc >= 5) {
        const std::string s = argv[4];
        try {
            size_t pos = 0;
            cliTempC = std::stod(s, &pos);
            bool tailWs = true;
            for (size_t k = pos; k < s.size(); ++k)
                if (!std::isspace(static_cast<unsigned char>(s[k]))) { tailWs = false; break; }
            if (pos == 0 || !tailWs || !std::isfinite(cliTempC))
                throw std::invalid_argument("not a finite number");
            hasTempArg = true;
        } catch (const std::exception&) {
            std::cerr << "error: temp_c must be a number, got '" << s << "'\n"
                      << "usage: fcchain_v3 <input_dir> <calib_json> <out_dir> [temp_c]\n"
                      << "  temp_c: 可选目标温度C（缺省=curveMapTempTable.tempBase; 仅 sidecar 路径生效）\n";
            return 2;
        }
    }
    const fs::path outDir(outDirS);
    fs::create_directories(outDir);

    std::cout << "fcchain_v3: data=" << dataDir << "\n out=" << outDirS << "\n";

    // ---------- load ----------
    auto inputOpt = loadLaserInput(dataDir);
    if (!inputOpt) { std::cerr << "loadLaserInput failed\n"; return 1; }
    const auto& input = inputOpt.value();
    const auto& cfg = input.config;
    const auto& h = input.handoff;
    const auto& img0 = input.poseFrames[0][0];

    json j;
    {
        std::ifstream f(calibJsonPath);
        if (!f.good()) { std::cerr << "cannot open " << calibJsonPath << "\n"; return 1; }
        j = json::parse(f);
    }

    // ---------- curve map table: CMTT sidecar tier select (preferred) / on-the-fly (fallback) ----------
    const double fpx = j["pjc"]["f"].get<double>();
    const cv::Point2d pp(j["pjc"]["principalPoint"][0].get<double>(),
                         j["pjc"]["principalPoint"][1].get<double>());
    const cv::Vec3d tRect(j["pjc"]["projectorT"][0].get<double>(),
                          j["pjc"]["projectorT"][1].get<double>(),
                          j["pjc"]["projectorT"][2].get<double>());
    const float rowStep = j["pjc"].value("epipolarRowStep", 0.7f);

    CurveMapTable cmt;
    double tableTempC = cfg.referenceTemp;   // 注入档温（sidecar 档温或参考温）
    {
        std::string why;
        bool fromSidecar = false;
        if (!j.contains("curveMapTempTable"))
            why = "calib JSON has no curveMapTempTable meta";
        else if (!j["curveMapTempTable"].value("success", false))
            why = "curveMapTempTable.success != true";
        else if (j["curveMapTempTable"].value("path", std::string()).empty())
            why = "curveMapTempTable.path empty";
        else {
            const fs::path sidecar = fs::path(calibJsonPath).parent_path() /
                j["curveMapTempTable"].value("path", std::string());
            const double tempC = hasTempArg
                ? cliTempC
                : j["curveMapTempTable"].value("tempBase", cfg.referenceTemp);
            CmttTierChoice choice = selectCmttTier(sidecar.string(), tempC);
            if (!choice.ok)
                why = choice.why;
            else {
                std::string lerr;
                if (!cmt.Load(choice.blob.data(), choice.blob.size(), lerr))
                    why = "tier blob CMPU load failed: " + lerr;
                else {
                    fromSidecar = true;
                    tableTempC = choice.tierTemp;
                    spdlog::info("sidecar tier selected: T={:.1f} ({}), entries from blob",
                                 choice.tierTemp, choice.why);
                }
            }
        }

        if (!fromSidecar) {
            spdlog::warn("sidecar unavailable ({}), fallback to on-the-fly generation", why);
            CurveMapInput cmi;
            for (const auto& cj : j["pjc"]["emissionCurves"]) {
                ImplicitCurve c;
                for (int k = 0; k < 6; ++k) c.coeffs[k] = cj["coeffs"][k].get<double>();
                cmi.curves.push_back(c);
            }
            cmi.f = fpx;
            cmi.principalPoint = pp;
            cv::Matx34d P2m(h.P2);
            cmi.baseline = std::fabs(P2m(0, 3)) / fpx;
            cmi.imageSize = h.imageSize;
            cmi.virtualT = tRect;
            {
                std::ifstream hf(dataDir + "/camera_calib.json");
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
            if (!cmr.success) { std::cerr << "curve map: " << cmr.message << "\n"; return 1; }
            std::string lerr;
            if (!cmt.Load(cmr.tableBytes.data(), cmr.tableBytes.size(), lerr)) {
                std::cerr << "table load: " << lerr << "\n"; return 1;
            }
        }
    }
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

    cv::cuda::Stream stream;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    // ---------- operators ----------
    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
    MaskExtractCUDA maskL(mp), maskR(mp);

    RegionAnalyzerParams cp; cp.minArea = 0; cp.topXCount = 27;
    RegionAnalyzerCUDA cclL(cp), cclR(cp);

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

    EpipolarInterpDualParams dip;
    dip.epipolar_row_step = rowStep;
    EpipolarInterpDualCuda epiL(dip), epiR(dip);

    LaserMatchScanParamsV3 vp3;
    vp3.epipolar_row_step = rowStep;
    vp3.vL_tolerance = 0.7f;
    vp3.match_threshold = 2.0f;
    LaserMatchScanCudaV3 matcher(vp3);
    if (!matcher.SetTempTable(tempTable)) { std::cerr << "v3 SetTempTable failed\n"; return 1; }
    matcher.SetCurrentTemperature(tableTempC);

    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    // ---------- chain ----------
    auto tMask = timed("mask", cs, [&] { return maskL.Execute(img0.leftGray, stream); });
    auto tMaskR = timed("mask(R)", cs, [&] { return maskR.Execute(img0.rightGray, stream); });
    auto tCcl = timed("ccl", cs, [&] { return cclL.Execute(*tMask.d_cleanedMask, stream); });
    auto tCclR = timed("ccl(R)", cs, [&] { return cclR.Execute(*tMaskR.d_cleanedMask, stream); });
    auto tStg = timed("steger_fast", cs, [&] {
        return stgL.Execute(*tMask.d_grayImage, *tMask.d_cleanedMask, stream, GroupMode::Flat); });
    auto tStgR = timed("steger_fast(R)", cs, [&] {
        return stgR.Execute(*tMaskR.d_grayImage, *tMaskR.d_cleanedMask, stream, GroupMode::Flat); });
    if (!tStg.success || !tStgR.success) {
        std::cerr << "steger_fast failed\n"; return 1;
    }
    auto tUn = timed("undistort", cs, [&] {
        return unL.Execute(*tStg.d_centerPoints, *tStg.d_line_ids, stream); });
    auto tUnR = timed("undistort(R)", cs, [&] {
        return unR.Execute(*tStgR.d_centerPoints, *tStgR.d_line_ids, stream); });
    auto tEpi = timed("interp_dual", cs, [&] {
        return epiL.Execute(*tUn.d_rectifiedPoints, *tUn.d_line_ids, stream); });
    auto tEpiR = timed("interp_dual(R)", cs, [&] {
        return epiR.Execute(*tUnR.d_rectifiedPoints, *tUnR.d_line_ids, stream); });
    auto tMatch = timed("match_v3", cs, [&] {
        return matcher.Execute(*tEpi.d_interpPoints, *tEpi.d_interp_line_ids,
                               *tEpiR.d_interpPoints, *tEpiR.d_interp_line_ids, stream); });
    auto tRec = timed("reconstruct", cs, [&] {
        return recon.Execute(*tMatch.d_matched_left, *tMatch.d_matched_right,
                             *tMatch.d_matched_line_ids, h.Q, stream); });
    stream.waitForCompletion();

    if (!tMatch.success) { std::cerr << "match_v3: " << tMatch.message << "\n"; return 1; }
    if (!tRec.success) { std::cerr << "reconstruct: " << tRec.message << "\n"; return 1; }

    // ---------- step images ----------
    const std::string pre = (outDir / "").string();
    {   // 01 mask
        cv::Mat mL, mR;
        tMask.d_cleanedMask->download(mL, stream);
        tMaskR.d_cleanedMask->download(mR, stream);
        stream.waitForCompletion();
        cv::imwrite(pre + "01_mask_L.png", mL);
        cv::imwrite(pre + "01_mask_R.png", mR);
    }
    {   // 02 ccl colored
        for (int side = 0; side < 2; ++side) {
            cv::Mat lab;
            if (side == 0) tCcl.d_labeledMask->download(lab, stream);
            else tCclR.d_labeledMask->download(lab, stream);
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
    auto dlPts = [](const cv::cuda::GpuMat& g, cv::cuda::Stream& s) {
        cv::Mat m;
        g.download(m, s);
        s.waitForCompletion();
        m = m.reshape(2, 1);
        return std::vector<cv::Point2f>(m.begin<cv::Point2f>(), m.end<cv::Point2f>());
    };
    {   // 03 steger centers on raw gray
        savePts(pre + "03_steger_L.png", img0.leftGray,
                dlPts(*tStg.d_centerPoints, stream), {0, 0, 255});
        savePts(pre + "03_steger_R.png", img0.rightGray,
                dlPts(*tStgR.d_centerPoints, stream), {0, 0, 255});
    }
    {   // 04 undistorted pts on rectified image
        cv::Mat mxL, myL, mxR, myR, recL, recR;
        cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                    h.imageSize, CV_32FC1, mxL, myL);
        cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                    h.imageSize, CV_32FC1, mxR, myR);
        cv::remap(img0.leftGray, recL, mxL, myL, cv::INTER_LINEAR);
        cv::remap(img0.rightGray, recR, mxR, myR, cv::INTER_LINEAR);
        savePts(pre + "04_undistort_L.png", recL,
                dlPts(*tUn.d_rectifiedPoints, stream), {255, 0, 0});
        savePts(pre + "04_undistort_R.png", recR,
                dlPts(*tUnR.d_rectifiedPoints, stream), {255, 0, 0});
    }
    {   // 05 interp pts colored by track id
        for (int side = 0; side < 2; ++side) {
            const auto& e = side ? tEpiR : tEpi;
            auto pts = dlPts(*e.d_interpPoints, stream);
            cv::Mat fids;
            e.d_interp_line_ids->download(fids, stream);
            stream.waitForCompletion();
            fids = fids.reshape(1, 1);
            cv::Mat img(h.imageSize, CV_8UC3, cv::Scalar(0, 0, 0));
            for (size_t i = 0; i < pts.size(); ++i) {
                const int tid = fids.at<int>(static_cast<int>(i));
                const cv::Vec3b c = labelColor((tid * 7) % 180 + 1);
                cv::circle(img, {cvRound(pts[i].x), cvRound(pts[i].y)}, 1,
                           cv::Scalar(c[0], c[1], c[2]), -1);
            }
            cv::imwrite(pre + "05_interp_tracks_" + std::string(side ? "R" : "L") + ".png", img);
        }
    }
    {   // 06 match pairs: rectified L/R side-by-side with lines
        cv::Mat mL, mR, mI, mT;
        tMatch.d_matched_left->download(mL, stream);
        tMatch.d_matched_right->download(mR, stream);
        tMatch.d_matched_line_ids->download(mI, stream);
        tMatch.d_matched_track_ids->download(mT, stream);
        stream.waitForCompletion();
        mL = mL.reshape(2, 1);
        mR = mR.reshape(2, 1);
        mI = mI.reshape(1, 1);
        mT = mT.reshape(1, 1);
        cv::Mat mxL, myL, mxR, myR, recL, recR;
        cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, P1_3x3,
                                    h.imageSize, CV_32FC1, mxL, myL);
        cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, P2_3x3,
                                    h.imageSize, CV_32FC1, mxR, myR);
        cv::remap(img0.leftGray, recL, mxL, myL, cv::INTER_LINEAR);
        cv::remap(img0.rightGray, recR, mxR, myR, cv::INTER_LINEAR);
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
        cv::imwrite(pre + "06_match_pairs.png", canvas);
    }
    int cloudPts = 0;
    {   // cloud.ply colored by track
        cv::Mat P3, I3;
        tRec.d_points3d->download(P3, stream);
        tRec.d_valid_line_ids->download(I3, stream);
        cv::Mat mT;
        tMatch.d_matched_track_ids->download(mT, stream);
        stream.waitForCompletion();
        P3 = P3.reshape(3, 1);
        I3 = I3.reshape(1, 1);
        mT = mT.reshape(1, 1);
        const int n = P3.cols;
        std::ofstream f(pre + "cloud.ply");
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
        cloudPts = n;
    }

    // ---------- timing report ----------
    std::ofstream tf(pre + "timing.txt");
    float total = 0.f;
    std::cout << "\n==== per-step timing (GPU events) ====\n";
    tf << "step,ms\n";
    for (auto& t : g_times) {
        std::cout << std::setw(16) << t.name << " : " << std::fixed
                  << std::setprecision(3) << t.ms << " ms\n";
        tf << t.name << "," << t.ms << "\n";
        total += t.ms;
    }
    std::cout << std::setw(16) << "TOTAL" << " : " << total << " ms\n";
    std::cout << "\nmatched=" << tMatch.matchedCount
              << " tracks=" << tMatch.trackTotal
              << " kept=" << (tMatch.trackTotal - tMatch.trackDropped)
              << " cloudPts=" << cloudPts << "\n";
    std::cout << "outputs -> " << outDirS << "\n";
    return 0;
}
