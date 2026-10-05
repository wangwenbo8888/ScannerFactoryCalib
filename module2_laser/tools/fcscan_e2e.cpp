// fcscan_e2e.cpp — 扫描式端到端测试: 表消费 → match_scan 匹配 → 三维重建 → PLY
//
// 链路: left_skew 图像 → 4-1..4-6(标定链取点) → curve_map 表(查表预测)
//       → laser_match_scan(匹配+线号输出) → laser_reconstruct(Q) → PLY 点云
// 表注入(优先): calib JSON curveMapTempTable meta → CMTT sidecar 按温选档
//       (selectCmttTier) → CurveMapTable::Load 内存段; 无 meta/选档失败回落
//       现场生成(ref 温单档)。两路统一: curve_map 条目 →
//       LaserPlaneMapTempTable.leftToRightMap → SetTempTable(档温注入)
#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "curve_map.h"
#include "laser_match_scan_cuda.h"
#include "common/calib_result_types.h"
#include "common/cmtt_tier_select.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using namespace fc;
using namespace calib;

namespace {

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
    return cv::Scalar(h, 180, 255);   // HSV 近似（仅区分用）
}

cv::Scalar hsvToBgr(const cv::Scalar& hsv) {
    cv::Mat m(1, 1, CV_8UC3);
    m.at<cv::Vec3b>(0, 0) = cv::Vec3b((uchar)hsv[0], (uchar)hsv[1], (uchar)hsv[2]);
    cv::cvtColor(m, m, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = m.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: fcscan_e2e <calib_json> <input_dir> <out_dir> [temp_c]\n"
                  << "  calib_json: laser_calib.json（pjc 节: 曲线/tRect/rowStep; curveMapTempTable 节: sidecar）\n"
                  << "  input_dir : left_skew 图像目录\n"
                  << "  out_dir   : PLY/统计输出目录\n"
                  << "  temp_c    : 可选目标温度C（缺省=curveMapTempTable.tempBase 参考温; 仅 sidecar 路径生效）\n";
        return 2;
    }
    const std::string calibJson = argv[1];
    const std::string inDir = argv[2];
    const std::string outDir = argv[3];

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
                      << "usage: fcscan_e2e <calib_json> <input_dir> <out_dir> [temp_c]\n";
            return 2;
        }
    }
    std::filesystem::create_directories(outDir);

    spdlog::info("=== fcscan_e2e: 表消费→匹配→重建→点云 ===");
    try {

    // ---------- 1. 载入标定产物 ----------
    nlohmann::json j;
    {
        std::ifstream f(calibJson);
        if (!f) { spdlog::error("cannot open {}", calibJson); return 1; }
        j = nlohmann::json::parse(f);
    }
    auto input = loadLaserInput(inDir);
    if (!input) { spdlog::error("load input failed"); return 1; }
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    const double fpx = j["pjc"]["f"].get<double>();
    const cv::Point2d pp(j["pjc"]["principalPoint"][0].get<double>(),
                         j["pjc"]["principalPoint"][1].get<double>());
    const cv::Vec3d tRect(j["pjc"]["projectorT"][0].get<double>(),
                          j["pjc"]["projectorT"][1].get<double>(),
                          j["pjc"]["projectorT"][2].get<double>());
    const float rowStep = j["pjc"].value("epipolarRowStep", 0.7f);

    // ---------- 2. 表获取: CMTT sidecar 按温选档（优先）/ curve_map 现场生成（回落） ----------
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
            const std::filesystem::path sidecar =
                std::filesystem::path(calibJson).parent_path() /
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
                std::ifstream hf(inDir + "/camera_calib.json");
                nlohmann::json hj = nlohmann::json::parse(hf);
                const auto& vr = hj["rectify"]["validRoiLeft"];
                cmi.roi = cv::Rect(vr["x"].get<int>(), vr["y"].get<int>(),
                                   vr["w"].get<int>(), vr["h"].get<int>());
            }
            CurveMapParams cmp;
            cmp.epipolarRowStep = rowStep;
            cmp.depthMin = cfg.depthMin;      // 用 config 的深度窗(100-5000)
            cmp.depthMax = std::min<float>(cfg.depthMax, 1500.0f);  // 收窄防表爆炸
            CurveMapGenerator cmGen(cmp);
            auto cmr = cmGen.Generate(cmi);
            if (!cmr.success) { spdlog::error("curve_map failed: {}", cmr.message); return 1; }
            spdlog::info("curve_map: entries={} bytes={:.1f}MB", cmr.entryCount,
                         cmr.tableBytesSize / 1048576.0);
            std::string lerr;
            if (!cmt.Load(cmr.tableBytes.data(), cmr.tableBytes.size(), lerr)) {
                spdlog::error("curve table reload: {}", lerr);
                return 1;
            }
        }
    }

    // ---------- 3. 表注入 match_scan（LaserPlaneMapTempTable 适配）----------
    auto tempTable = std::make_shared<LaserPlaneMapTempTable>();
    {
        std::vector<cv::Vec4f> rows;
        std::vector<CurveMapEntry> buf;
        buf.reserve(4096);
        for (uint32_t row = 0; row < cmt.rowCount(); ++row) {
            if (row % 64 == 0) buf.reserve(buf.capacity());   // no-op 防优化告警
            const int need = cmt.EnumerateRow(static_cast<int>(row), nullptr, 0);
            (void)need;
            std::vector<CurveMapEntry> tmp;
            // 两段式: 先取容量
            int cap = 4096;
            tmp.resize(cap);
            int n = cmt.EnumerateRow(static_cast<int>(row), tmp.data(), cap);
            while (n < 0) {
                cap *= 2;
                tmp.resize(cap);
                n = cmt.EnumerateRow(static_cast<int>(row), tmp.data(), cap);
            }
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
        tempTable->referenceTemperature = tableTempC;
        tempTable->table[tableTempC] = std::move(pm);
        spdlog::info("temp table adapted: {} entries @ ref={:.1f}C",
                     rows.size(), tableTempC);
    }

    LaserMatchScanParams msp;
    msp.epipolar_row_step = rowStep;          // F4 红线: 与表同栅格!
    msp.match_threshold = 2.0f;               // 表量化 0.35 + 噪声余量
    msp.vL_tolerance = rowStep;               // x 最近邻容差 = 半栅格级（表 x 量化 0.7）
    LaserMatchScanCuda matcher(msp);
    if (!matcher.SetTempTable(tempTable)) {
        spdlog::error("SetTempTable failed");
        return 1;
    }
    matcher.SetCurrentTemperature(tableTempC);

    // ---------- 4. 重建用 4-8（Q 用参考档） ----------
    LaserReconstructParams rp;
    rp.minDepth = cfg.depthMin;
    rp.maxDepth = 1500.0f;
    LaserReconstructCuda recon(rp);

    // ---------- 5. 逐姿态: 4-1..4-6 → match → reconstruct → PLY ----------
    cv::cuda::Stream stream;
    MaskExtractParams mp; mp.threshold = 50; mp.erodeSize = 1;
    mp.laserDilateSize = 19; mp.postErodeSize = 13;
    // A/B 开关: FC_MASK_APPROX=1 启用线核近似(默认 0=精确椭圆核)
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
    EpipolarInterpParams eip; eip.lineIdCheck = true;
    EpipolarInterpCuda epiL(eip), epiR(eip);

    long totalIn = 0, totalMatched = 0, totalPts = 0;
    std::vector<cv::Vec3f> allPts;
    std::vector<int> allIds;

    std::ofstream tcsv(outDir + "/timing.csv");
    tcsv << "frame,mask_ms,ccl_ms,label_ms,steger_ms,undist_ms,interp_ms,match_ms,recon_ms,matched\n";
    double accMs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int accFrames = 0;

    // 纯计算计时: CUDA 事件在 GPU 时间线上卡点(不含 CPU 等待/外部同步;
    // mask 含算子内部图像 H2D ~1-2ms, 其余步骤输入输出均已在 GPU)
    cudaStream_t cuStream = cv::cuda::StreamAccessor::getStream(stream);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const auto& fr = input->poseFrames[pi];
        for (size_t ti = 0; ti < fr.size(); ++ti) {
            const auto& img = fr[ti];
            const std::string stem =
                input->poseDirs[pi] + "_tube" + std::to_string(ti);

            double sm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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

            auto m1 = tExec(0, maskL, img.leftGray);
            auto m2 = tExec(0, maskR, img.rightGray);
            if (!m1.success || !m2.success || !m1.d_cleanedMask || !m2.d_cleanedMask)
                { spdlog::warn("{}: 4-1 fail, skip", stem); continue; }
            auto c1 = tExec(1, cclL, *m1.d_cleanedMask);
            auto c2 = tExec(1, cclR, *m2.d_cleanedMask);
            if (!c1.success || !c2.success || !c1.d_labeledMask || !c2.d_labeledMask)
                { spdlog::warn("{}: 4-2 fail, skip", stem); continue; }
            auto l1 = tExec(2, labL, *c1.d_labeledMask);
            auto l2 = tExec(2, labR, *c2.d_labeledMask);
            if (!l1.success || !l2.success || !l1.d_labeledMask || !l2.d_labeledMask)
                { spdlog::warn("{}: 4-3 fail, skip", stem); continue; }
            cudaEventRecord(ev0, cuStream);
            auto s1 = stgL.Execute(*m1.d_grayImage, *l1.d_labeledMask, stream, GroupMode::ByLabel);
            auto s2 = stgR.Execute(*m2.d_grayImage, *l2.d_labeledMask, stream, GroupMode::ByLabel);
            cudaEventRecord(ev1, cuStream);
            cudaEventSynchronize(ev1);
            {
                float ms = 0.f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                sm[3] += ms;
            }
            if (!s1.success || !s2.success || !s1.d_centerPoints || !s2.d_centerPoints)
                { spdlog::warn("{}: 4-4 fail, skip", stem); continue; }
            auto u1 = tExec(4, unL, *s1.d_centerPoints, *s1.d_line_ids);
            auto u2 = tExec(4, unR, *s2.d_centerPoints, *s2.d_line_ids);
            if (!u1.success || !u2.success || !u1.d_rectifiedPoints || !u2.d_rectifiedPoints)
                { spdlog::warn("{}: 4-5 fail, skip", stem); continue; }
            auto e1 = tExec(5, epiL, *u1.d_rectifiedPoints, *u1.d_line_ids);
            auto e2 = tExec(5, epiR, *u2.d_rectifiedPoints, *u2.d_line_ids);
            if (!e1.success || !e2.success || !e1.d_interpPoints || !e2.d_interpPoints)
                { spdlog::warn("{}: 4-6 fail, skip", stem); continue; }

            const int nL = e1.d_interpPoints->cols;
            const int nR = e2.d_interpPoints->cols;
            totalIn += nL;

            if (!e1.d_interp_line_ids || !e2.d_interp_line_ids) {
                spdlog::warn("{}: no interp line ids, skip", stem);
                continue;
            }

            auto mr = tExec(6, matcher, *e1.d_interpPoints, *e1.d_interp_line_ids,
                            *e2.d_interpPoints, *e2.d_interp_line_ids);
            if (!mr.success) {
                spdlog::warn("{}: match fail ({}), skip", stem, mr.message);
                continue;
            }
            totalMatched += mr.matchedCount;

            if (mr.matchedCount == 0) {
                spdlog::warn("{}: 0 matched (excluded L/R={}/{})",
                             stem, mr.excludedLeftCount, mr.excludedRightCount);
                continue;
            }

            auto rr = tExec(7, recon, *mr.d_matched_left, *mr.d_matched_right,
                            *mr.d_matched_line_ids, h.Q);
            if (!rr.success || !rr.d_points3d) {
                spdlog::warn("{}: recon fail ({})", stem, rr.message);
                continue;
            }
            cv::Mat pts3, ids3;
            rr.d_points3d->download(pts3);
            rr.d_valid_line_ids->download(ids3);
            pts3 = pts3.reshape(3, 1);
            ids3 = ids3.reshape(1, 1);
            std::vector<cv::Vec3f> vp(pts3.begin<cv::Vec3f>(), pts3.end<cv::Vec3f>());
            std::vector<int> vi(ids3.begin<int>(), ids3.end<int>());
            totalPts += vp.size();
            allPts.insert(allPts.end(), vp.begin(), vp.end());
            allIds.insert(allIds.end(), vi.begin(), vi.end());

            writePlyAscii(outDir + "/" + stem + "_scan.ply", vp, vi,
                          [](int id) { return hsvToBgr(lidColor(id)); });
            // 线号分布
            std::map<int, int> dist;
            for (int id : vi) ++dist[id];
            std::ostringstream ss;
            for (auto& kv : dist) ss << kv.first << ":" << kv.second << " ";
            spdlog::info("{}: in={} matched={} pts={} lines[{}]", stem, nL,
                         mr.matchedCount, vp.size(), ss.str());
            tcsv << stem << ',' << sm[0] << ',' << sm[1] << ',' << sm[2] << ','
                 << sm[3] << ',' << sm[4] << ',' << sm[5] << ',' << sm[6] << ','
                 << sm[7] << ',' << vp.size() << '\n';
            for (int k = 0; k < 8; ++k) accMs[k] += sm[k];
            ++accFrames;
            spdlog::info("{}: gpu-compute ms | mask={:.2f} ccl={:.2f} label={:.2f} steger={:.2f} "
                         "undist={:.2f} interp={:.2f} match={:.2f} recon={:.2f} | total={:.2f}",
                         stem, sm[0], sm[1], sm[2], sm[3], sm[4], sm[5], sm[6], sm[7],
                         sm[0]+sm[1]+sm[2]+sm[3]+sm[4]+sm[5]+sm[6]+sm[7]);
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    if (accFrames > 0) {
        spdlog::info("avg gpu-compute ms over {} frames | mask={:.2f} ccl={:.2f} label={:.2f} "
                     "steger={:.2f} undist={:.2f} interp={:.2f} match={:.2f} recon={:.2f} "
                     "| total={:.2f}",
                     accFrames,
                     accMs[0]/accFrames, accMs[1]/accFrames, accMs[2]/accFrames,
                     accMs[3]/accFrames, accMs[4]/accFrames, accMs[5]/accFrames,
                     accMs[6]/accFrames, accMs[7]/accFrames,
                     (accMs[0]+accMs[1]+accMs[2]+accMs[3]+accMs[4]+accMs[5]+accMs[6]+accMs[7])/accFrames);
    }

    writePlyAscii(outDir + "/all_scan.ply", allPts, allIds,
                  [](int id) { return hsvToBgr(lidColor(id)); });
    spdlog::info("=== e2e done: input={} matched={} ({:.1f}%) pts3d={} ===",
                 totalIn, totalMatched,
                 totalIn > 0 ? 100.0 * totalMatched / totalIn : 0.0, totalPts);
    spdlog::info("merged cloud -> {}/all_scan.ply ({} pts)", outDir, allPts.size());
    return totalMatched > 0 ? 0 : 1;

    } catch (const std::exception& e) {
        spdlog::error("exception: {}", e.what());
        return 1;
    }
}
