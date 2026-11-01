#include "calib_io.h"
#include "chessboard_corner.h"

#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <map>
#include <array>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cmath>

using namespace fc;
using namespace calib;

namespace {

IntrinsicCalibParams makeIntrinParams(const CameraCalibConfig& c) {
    IntrinsicCalibParams p;
    p.chessboard_width = c.chessWidth;
    p.chessboard_height = c.chessHeight;
    p.square_size_mm = c.squareSizeMm;
    p.image_width = c.imageWidth;
    p.image_height = c.imageHeight;
    p.use_calibrateCameraRO = c.useCalibrateCameraRO;
    p.calib_flags = c.intrinsicFlags;
    p.reproj_error_threshold = c.reprojErrorThreshold;
    p.temperature_coeff = c.plateTempCoeff;
    p.plate_temp = c.plateTemp;
    return p;
}

// 注意: ExtrinsicCalibCpuParams 不持有 cameraMatrixL/distCoeffsL/R —— 这些通过
// Execute(KL,DL,KR,DR) 重载在调用点直接传入（见 step 3）。本函数只填观测点+板参数。
ExtrinsicCalibCpuParams makeExtrinParams(const CameraCalibConfig& c,
    const CameraOpParams& ops,
    const std::vector<std::vector<cv::Point2f>>& lpts,
    const std::vector<std::vector<cv::Point2f>>& rpts)
{
    ExtrinsicCalibCpuParams p;
    p.leftPointsPerView = lpts;
    p.rightPointsPerView = rpts;
    p.imageSize = cv::Size(c.imageWidth, c.imageHeight);
    p.patternSize = cv::Size(c.chessWidth, c.chessHeight);
    p.squareSize = static_cast<float>(c.squareSizeMm);
    p.maxReprojError = c.reprojErrorThreshold * ops.extrinsicMaxReprojFactor;
    p.minViewCount = ops.extrinsicMinViewCount;
    // 物点网格与 IntrinsicCalibCPU::generateObjectPoints 同约定（行主序+板热膨胀）
    const double actualSize = c.squareSizeMm *
        (1.0 + c.plateTempCoeff * (c.plateTemp - 20.0));
    for (int i = 0; i < c.chessHeight; ++i)
        for (int j = 0; j < c.chessWidth; ++j)
            p.objectPoints.emplace_back(
                static_cast<float>(j * actualSize),
                static_cast<float>(i * actualSize), 0.0f);
    return p;
}

struct FrameMetaInfo {
    std::optional<double> temp;
    std::optional<double> exposure;
};

struct FrameRec {
    size_t origIndex = 0;
    std::string leftFile, rightFile;
    bool valid = false;
    std::string skipReason;
    double subpixL = 0.0, subpixR = 0.0;
    std::vector<cv::Point2f> cornersL, cornersR;
    std::optional<double> temp, exposure;
};

// 侧车 frames_meta.csv: file,temp_c,exposure_us（首行表头；file 匹配左目 stem）
std::map<std::string, FrameMetaInfo> parseFramesMeta(const std::string& dir) {
    std::map<std::string, FrameMetaInfo> out;
    std::ifstream ifs(dir + "/frames_meta.csv");
    if (!ifs.is_open()) return out;
    std::string line;
    if (!std::getline(ifs, line)) return out;
    size_t rows = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string f, t, e;
        std::getline(ss, f, ',');
        std::getline(ss, t, ',');
        std::getline(ss, e, ',');
        if (f.empty()) continue;
        FrameMetaInfo m;
        try { if (!t.empty()) m.temp = std::stod(t); } catch (const std::exception&) {}
        try { if (!e.empty()) m.exposure = std::stod(e); } catch (const std::exception&) {}
        out[std::filesystem::path(f).stem().string()] = m;
        ++rows;
    }
    spdlog::info("frames_meta.csv loaded: {} rows", rows);
    return out;
}

std::string csvEsc(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string r = "\"";
    for (char c : s) { if (c == '"') r += "\"\""; else r += c; }
    return r + "\"";
}

std::string fmtN(double v, int prec) {
    if (!std::isfinite(v)) return "";
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string optStr(const std::optional<double>& v, int prec) {
    return v.has_value() ? fmtN(*v, prec) : std::string();
}

std::string gridRegion(double u, double v, int W, int H) {
    static const char* rows[3] = {"T", "M", "B"};
    static const char* cols[3] = {"L", "C", "R"};
    int ci = u < W / 3.0 ? 0 : (u < 2.0 * W / 3.0 ? 1 : 2);
    int ri = v < H / 3.0 ? 0 : (v < 2.0 * H / 3.0 ? 1 : 2);
    return std::string(rows[ri]) + cols[ci];
}

// 逐帧/逐角点 CSV 落盘（写失败仅 warn；epi_raw=原始像素×F 口径，row_align=矫正域口径）
void emitFrameCsvs(const std::string& outPath,
                   const std::vector<FrameRec>& recs,
                   const std::vector<std::vector<cv::Point2f>>& lpts,
                   const std::vector<std::vector<cv::Point2f>>& rpts,
                   const calib::IntrinsicCalibResult& intrinRes,
                   const calib::ExtrinsicCalibCpuResult& extrinRes,
                   const calib::StereoRectifyCpuResult& rectifyRes,
                   const CameraCalibConfig& cfg,
                   const ExtrinsicCalibCpuParams& ep)
{
    namespace fs = std::filesystem;
    const cv::Mat &KL = intrinRes.left.camera_matrix, &DL = intrinRes.left.dist_coeffs;
    const cv::Mat &KR = intrinRes.right.camera_matrix, &DR = intrinRes.right.dist_coeffs;
    const cv::Mat &F = extrinRes.F, &GR = extrinRes.R, &GT = extrinRes.T;
    const cv::Mat &R1 = rectifyRes.R1, &R2 = rectifyRes.R2, &P1 = rectifyRes.P1, &P2 = rectifyRes.P2;
    const int W = cfg.imageWidth, H = cfg.imageHeight;
    const int NX = cfg.chessWidth, NY = cfg.chessHeight;
    const double fxL = KL.at<double>(0, 0), fyL = KL.at<double>(1, 1);
    const double cxL = KL.at<double>(0, 2), cyL = KL.at<double>(1, 2);

    fs::path base(outPath);
    if (base.has_extension()) base.replace_extension();
    std::string pfPath = base.string() + "_perframe.csv";
    std::string pcPath = base.string() + "_percorner.csv";
    std::ofstream pf(pfPath), pc(pcPath);
    if (!pf.is_open() || !pc.is_open()) {
        spdlog::warn("per-frame csv not written (cannot open {} / {})", pfPath, pcPath);
        return;
    }

    char tbuf[32];
    std::time_t tt = std::time(nullptr);
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(1);
        os << "# schema=camera_calib_perframe.v1 generated=" << tbuf
           << " board=" << NX << "x" << NY << "x" << cfg.squareSizeMm << "mm"
           << " image=" << W << "x" << H << " ref_temp=" << cfg.referenceTemp
           << " note=epi_raw uses raw-pixel F domain (has ~0.3px baseline artifact); row_align uses rectified domain";
        pf << os.str() << "\n";
        pc << os.str() << "\n";
    }

    pf << "frame_id,orig_index,left_file,right_file,valid,skip_reason,corner_count,"
          "mean_subpix_delta_px,reproj_L_px,reproj_R_px,reproj_ratio,epi_raw_px,row_align_px,"
          "stereo_reproj_px,extrin_deg,extrin_mm,dist_L_mm,dist_R_mm,tilt_deg,cover_pct,"
          "board_center_u_px,board_center_v_px,region_L,region_R,"
          "rvec_Lx,rvec_Ly,rvec_Lz,tvec_Lx_mm,tvec_Ly_mm,tvec_Lz_mm,"
          "rvec_Rx,rvec_Ry,rvec_Rz,tvec_Rx_mm,tvec_Ry_mm,tvec_Rz_mm,"
          "temp_c,exposure_us\n";
    pc << "frame_id,corner_idx,grid_row,grid_col,x_L_px,y_L_px,x_R_px,y_R_px,"
          "resid_dx_L_px,resid_dy_L_px,resid_dx_R_px,resid_dy_R_px,"
          "epi_raw_px,row_align_dy_px,radius_norm\n";

    size_t validSeen = 0;
    for (const auto& rec : recs) {
        std::ostringstream row;
        row << (rec.valid ? validSeen + 1 : 0) << "," << rec.origIndex << ","
            << csvEsc(rec.leftFile) << "," << csvEsc(rec.rightFile) << ","
            << (rec.valid ? 1 : 0) << "," << csvEsc(rec.skipReason) << ",";
        if (!rec.valid) {
            row << "0";
            for (int k = 0; k < 27; ++k) row << ",";
            row << "," << optStr(rec.temp, 2)
                << "," << optStr(rec.exposure, 1) << "\n";
            pf << row.str();
            continue;
        }
        const size_t vi = validSeen++;
        const auto& cl = lpts[vi];
        const auto& cr = rpts[vi];
        const cv::Mat rvecL = intrinRes.left.rvecs[vi], tvecL = intrinRes.left.tvecs[vi];
        const cv::Mat rvecR = intrinRes.right.rvecs[vi], tvecR = intrinRes.right.tvecs[vi];

        std::vector<cv::Point2f> projL, projR, undL, undR;
        cv::projectPoints(ep.objectPoints, rvecL, tvecL, KL, DL, projL);
        cv::projectPoints(ep.objectPoints, rvecR, tvecR, KR, DR, projR);
        cv::undistortPoints(cl, undL, KL, DL, R1, P1);
        cv::undistortPoints(cr, undR, KR, DR, R2, P2);

        const size_t n = cl.size();
        double epiRawSum = 0.0, rowSum = 0.0, sqL = 0.0, sqR = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double xl = cl[j].x, yl = cl[j].y, xr = cr[j].x, yr = cr[j].y;
            const double a = F.at<double>(0, 0) * xl + F.at<double>(0, 1) * yl + F.at<double>(0, 2);
            const double b = F.at<double>(1, 0) * xl + F.at<double>(1, 1) * yl + F.at<double>(1, 2);
            const double c = F.at<double>(2, 0) * xl + F.at<double>(2, 1) * yl + F.at<double>(2, 2);
            const double epiJ = std::abs(a * xr + b * yr + c) / std::hypot(a, b);
            const double dy = undL[j].y - undR[j].y;
            epiRawSum += epiJ;
            rowSum += std::abs(dy);
            const double dxL = xl - projL[j].x, dyL = yl - projL[j].y;
            const double dxR = xr - projR[j].x, dyR = yr - projR[j].y;
            sqL += dxL * dxL + dyL * dyL;
            sqR += dxR * dxR + dyR * dyR;
            std::ostringstream prow;
            prow << (vi + 1) << "," << j << "," << (j / NX) << "," << (j % NX) << ","
                 << fmtN(xl, 2) << "," << fmtN(yl, 2) << "," << fmtN(xr, 2) << "," << fmtN(yr, 2) << ","
                 << fmtN(dxL, 4) << "," << fmtN(dyL, 4) << "," << fmtN(dxR, 4) << "," << fmtN(dyR, 4) << ","
                 << fmtN(epiJ, 4) << "," << fmtN(dy, 4) << ","
                 << fmtN(std::hypot((xl - cxL) / fxL, (yl - cyL) / fyL), 4) << "\n";
            pc << prow.str();
        }
        const double stReproj = std::sqrt((sqL + sqR) / (2.0 * static_cast<double>(n)));

        cv::Mat Rl, Rr;
        cv::Rodrigues(rvecL, Rl);
        cv::Rodrigues(rvecR, Rr);
        cv::Mat Rrel = Rr * Rl.t();
        double trc = (cv::trace(Rrel * GR.t()).val[0] - 1.0) / 2.0;
        trc = std::max(-1.0, std::min(1.0, trc));
        const double edeg = std::acos(trc) * 180.0 / CV_PI;
        cv::Mat trel = tvecR - Rrel * tvecL;
        const double emm = cv::norm(trel - GT);

        std::vector<cv::Point3f> ext = {ep.objectPoints[0], ep.objectPoints[NX - 1],
                                        ep.objectPoints[static_cast<size_t>(NY - 1) * NX],
                                        ep.objectPoints[static_cast<size_t>(NY) * NX - 1]};
        std::vector<cv::Point2f> pixL, pixR;
        cv::projectPoints(ext, rvecL, tvecL, KL, DL, pixL);
        cv::projectPoints(ext, rvecR, tvecR, KR, DR, pixR);
        auto bbox = [](const std::vector<cv::Point2f>& p) {
            float u0 = p[0].x, v0 = p[0].y, u1 = p[0].x, v1 = p[0].y;
            for (const auto& q : p) {
                u0 = std::min(u0, q.x); v0 = std::min(v0, q.y);
                u1 = std::max(u1, q.x); v1 = std::max(v1, q.y);
            }
            return std::array<float, 4>{u0, v0, u1, v1};
        };
        const auto bl = bbox(pixL), br = bbox(pixR);
        const double cover = (bl[2] - bl[0]) * (bl[3] - bl[1]) / (double(W) * H) * 100.0;
        const double cuL = (bl[0] + bl[2]) / 2.0, cvL = (bl[1] + bl[3]) / 2.0;
        const double cuR = (br[0] + br[2]) / 2.0, cvR = (br[1] + br[3]) / 2.0;

        const double reprojL = intrinRes.left.per_view_errors[vi];
        const double reprojR = intrinRes.right.per_view_errors[vi];

        row << n << "," << fmtN((rec.subpixL + rec.subpixR) / 2.0, 4) << ","
            << fmtN(reprojL, 4) << "," << fmtN(reprojR, 4) << ","
            << (reprojR > 0 ? fmtN(reprojL / reprojR, 3) : std::string()) << ","
            << fmtN(epiRawSum / n, 4) << "," << fmtN(rowSum / n, 4) << ","
            << fmtN(stReproj, 4) << "," << fmtN(edeg, 4) << "," << fmtN(emm, 3) << ","
            << fmtN(cv::norm(tvecL), 3) << "," << fmtN(cv::norm(tvecR), 3) << ","
            << fmtN(std::acos(std::max(-1.0, std::min(1.0, Rl.at<double>(2, 2)))) * 180.0 / CV_PI, 3) << ","
            << fmtN(cover, 2) << "," << fmtN(cuL, 1) << "," << fmtN(cvL, 1) << ","
            << gridRegion(cuL, cvL, W, H) << "," << gridRegion(cuR, cvR, W, H) << ","
            << fmtN(rvecL.at<double>(0), 6) << "," << fmtN(rvecL.at<double>(1), 6) << ","
            << fmtN(rvecL.at<double>(2), 6) << ","
            << fmtN(tvecL.at<double>(0), 3) << "," << fmtN(tvecL.at<double>(1), 3) << ","
            << fmtN(tvecL.at<double>(2), 3) << ","
            << fmtN(rvecR.at<double>(0), 6) << "," << fmtN(rvecR.at<double>(1), 6) << ","
            << fmtN(rvecR.at<double>(2), 6) << ","
            << fmtN(tvecR.at<double>(0), 3) << "," << fmtN(tvecR.at<double>(1), 3) << ","
            << fmtN(tvecR.at<double>(2), 3) << ","
            << optStr(rec.temp, 2) << "," << optStr(rec.exposure, 1) << "\n";
        pf << row.str();
    }
    spdlog::info("per-frame csv -> {} / {}", pfPath, pcPath);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: camera_calib <input_dir> [output_json]\n"
                  << "  input_dir 含 config.json + left/ + right/ + temps.txt\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "camera_calib.json";

    // ------------------------------------------------------------------
    // 0. 参数四层合并: 编译内置 ← camera_calib_params.json(主库) ← 数据集 config.json
    //    ← temps.txt(ref_temp)
    //    主库定位: exe 旁(POST_BUILD 拷贝) → cwd → 源码树兜底; 全缺＝内置(等值, warn)
    // ------------------------------------------------------------------
    CameraCalibConfig baseCfg;
    CameraOpParams    baseOps;
    std::string paramSource = "builtin defaults (master json not found)";
    {
        namespace fs = std::filesystem;
        std::vector<std::filesystem::path> cand;
        std::error_code ec;
        if (argc > 0) {
            auto exeDir = std::filesystem::absolute(std::filesystem::path(argv[0]), ec).parent_path();
            if (!ec) cand.push_back(exeDir / "camera_calib_params.json");
        }
        cand.push_back(std::filesystem::path("camera_calib_params.json"));
        cand.push_back(std::filesystem::path(
            "E:/JEAMMWARE2601001/factory_calib/module1_camera/camera_calib_params.json"));
        for (const auto& p : cand) {
            std::error_code ec2;
            if (!std::filesystem::exists(p, ec2)) continue;
            std::ifstream ifs(p);
            if (!ifs.is_open()) continue;
            try {
                nlohmann::json jm = nlohmann::json::parse(ifs, nullptr, true);
                fc::applyCameraParamsJson(jm, baseCfg, baseOps);
                paramSource = p.string();
            } catch (const std::exception& e) {
                spdlog::warn("master params parse failed ({}), fallback to builtin", e.what());
            }
            break;
        }
    }
    spdlog::info("params source: {}", paramSource);

    auto input = loadCameraInput(inDir, &baseCfg, &baseOps);
    if (!input) { spdlog::error("load input failed"); return 1; }
    const auto& cfg = input->config;
    const auto& ops = input->ops;

    // 1. 逐帧提取棋盘角点（全帧留痕：失败帧也进 FrameRec，供逐帧 CSV 用）
    // 批次二（2026-10-14）：corner.workers>1 时按帧并行提取；
    // 提取期间将 OpenCV 内部线程压为 1（嵌套并行闸会退化帧级并行），
    // 结果与串行逐字节一致（opt_bench test_opt_corner_10s 臂二实证）。
    ChessboardCornerParams cp;
    cp.patternSize = cv::Size(cfg.chessWidth, cfg.chessHeight);
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    auto metaMap = parseFramesMeta(inDir);
    std::vector<FrameRec> recs(input->frames.size());
    const size_t frameCount = input->frames.size();
    const int cornerWorkers = std::max(1, ops.cornerWorkers);

    auto processFrame = [&](size_t i) {
        FrameRec& rec = recs[i];
        rec.origIndex = i;
        rec.leftFile = input->frames[i].leftFile;
        rec.rightFile = input->frames[i].rightFile;
        auto it = metaMap.find(std::filesystem::path(rec.leftFile).stem().string());
        if (it != metaMap.end()) {
            rec.temp = it->second.temp;
            rec.exposure = it->second.exposure;
        }
        ChessboardCornerResult rl, rr;
        bool okL = extractChessboardCorners(input->frames[i].leftGray, cp, rl);
        bool okR = extractChessboardCorners(input->frames[i].rightGray, cp, rr);
        if (!okL || !okR) {
            rec.skipReason = !okL ? (!okR ? "corner_L+corner_R" : "corner_L") : "corner_R";
            spdlog::warn("frame {}: corner extraction failed, skip ({})", i, rec.skipReason);
            return;
        }
        normalizeLRCornerOrder(rl, rr);
        rec.valid = true;
        rec.subpixL = rl.meanSubpixDelta;
        rec.subpixR = rr.meanSubpixDelta;
        rec.cornersL = std::move(rl.corners);
        rec.cornersR = std::move(rr.corners);
    };

    if (cornerWorkers <= 1 || frameCount == 0) {
        for (size_t i = 0; i < frameCount; ++i) processFrame(i);
    } else {
        const int savedNumThreads = cv::getNumThreads();
        cv::setNumThreads(1);
        const int nt = static_cast<int>(
            std::min(static_cast<size_t>(cornerWorkers), frameCount));
        std::atomic<size_t> next(0);
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(nt));
        for (int t = 0; t < nt; ++t) {
            pool.emplace_back([&] {
                for (;;) {
                    size_t i = next.fetch_add(1);
                    if (i >= frameCount) break;
                    processFrame(i);
                }
            });
        }
        for (auto& th : pool) th.join();
        cv::setNumThreads(savedNumThreads);
    }

    for (const auto& rec : recs) {
        if (!rec.valid) continue;
        lpts.push_back(rec.cornersL);
        rpts.push_back(rec.cornersR);
    }
    if (lpts.size() < static_cast<size_t>(ops.framesMinValidFrames)) {
        spdlog::error("too few valid frames: {}", lpts.size()); return 1;
    }

    // 2. 内参
    IntrinsicCalibCPU intrin(makeIntrinParams(cfg));
    IntrinsicCalibResult intrinRes;
    if (!intrin.Execute(lpts, rpts, intrinRes) || !intrinRes.success) {
        spdlog::error("intrinsic calib failed: {}", intrinRes.message); return 1;
    }
    spdlog::info("intrinsic OK, reproj_mean={}", intrinRes.reproj_error_mean);

    // 3. 外参
    ExtrinsicCalibCpuParams ep = makeExtrinParams(cfg, ops, lpts, rpts);
    ExtrinsicCalibCpu extrin(ep);
    ExtrinsicCalibCpuResult extrinRes = extrin.Execute(
        intrinRes.left.camera_matrix, intrinRes.left.dist_coeffs,
        intrinRes.right.camera_matrix, intrinRes.right.dist_coeffs);
    if (!extrinRes.success) { spdlog::error("extrinsic failed: {}", extrinRes.message); return 1; }

    // 4. 立体矫正
    StereoRectifyCpuParams rp;
    rp.cameraMatrixL = intrinRes.left.camera_matrix;
    rp.distCoeffsL   = intrinRes.left.dist_coeffs;
    rp.cameraMatrixR = intrinRes.right.camera_matrix;
    rp.distCoeffsR   = intrinRes.right.dist_coeffs;
    rp.imageSize     = cv::Size(cfg.imageWidth, cfg.imageHeight);
    rp.R = extrinRes.R; rp.T = extrinRes.T;
    rp.alpha = cfg.rectifyAlpha; rp.flags = cfg.rectifyFlags;
    StereoRectifyCpu rectify(rp);
    StereoRectifyCpuResult rectifyRes = rectify.Execute();
    if (!rectifyRes.success) { spdlog::error("rectify failed: {}", rectifyRes.message); return 1; }

    // 5. 三张温度表
    CameraIntrinsics cL{intrinRes.left.camera_matrix.at<double>(0,0),
                        intrinRes.left.camera_matrix.at<double>(1,1),
                        intrinRes.left.camera_matrix.at<double>(0,2),
                        intrinRes.left.camera_matrix.at<double>(1,2),
                        cfg.referenceTemp};
    CameraIntrinsics cR{intrinRes.right.camera_matrix.at<double>(0,0),
                        intrinRes.right.camera_matrix.at<double>(1,1),
                        intrinRes.right.camera_matrix.at<double>(0,2),
                        intrinRes.right.camera_matrix.at<double>(1,2),
                        cfg.referenceTemp};
    IntrinsicCompensateCPUParams icp; icp.cte=cfg.cte; icp.tempStep=cfg.tempStep;
    icp.tempRangeMin=cfg.tempRangeMin; icp.tempRangeMax=cfg.tempRangeMax;
    IntrinsicCompensateCPU icomp(icp);
    auto tableL = icomp.Execute(cL);
    auto tableR = icomp.Execute(cR);

    CameraExtrinsics ce; ce.referenceTemp = cfg.referenceTemp;
    for (int i=0;i<3;++i) ce.T[i]=extrinRes.T.at<double>(i);
    for (int i=0;i<9;++i) ce.R[i]=extrinRes.R.at<double>(i/3,i%3);
    ExtrinsicCompensateCPUParams ecp; ecp.cte=cfg.cte; ecp.tempStep=cfg.tempStep;
    ecp.tempRangeMin=cfg.tempRangeMin; ecp.tempRangeMax=cfg.tempRangeMax;
    ExtrinsicCompensateCPU ecomp(ecp);
    auto tableE = ecomp.Execute(ce);

    StereoRectifyTempTableParams strp;
    strp.cameraMatrixL=rp.cameraMatrixL; strp.distCoeffsL=rp.distCoeffsL;
    strp.cameraMatrixR=rp.cameraMatrixR; strp.distCoeffsR=rp.distCoeffsR;
    strp.imageSize=rp.imageSize; strp.R=rp.R; strp.T=rp.T;
    strp.referenceTemp=cfg.referenceTemp; strp.cte=cfg.cte;
    strp.tempStep=cfg.tempStep; strp.tempRangeMin=cfg.tempRangeMin; strp.tempRangeMax=cfg.tempRangeMax;
    strp.alpha=cfg.rectifyAlpha; strp.flags=cfg.rectifyFlags;
    StereoRectifyTempTableCpu strtab(strp);
    auto tableR2 = strtab.Execute();

    // 6. 写交接文件
    auto j = buildCameraCalibJson(cfg, intrinRes, extrinRes, rectifyRes,
                                  tableL, tableR, tableE, tableR2);
    if (!writeJson(outPath, j)) return 1;
    spdlog::info("camera_calib done -> {}", outPath);

    try {
        emitFrameCsvs(outPath, recs, lpts, rpts, intrinRes, extrinRes, rectifyRes, cfg, ep);
    } catch (const std::exception& e) {
        spdlog::warn("per-frame csv emit failed: {}", e.what());
    }
    return 0;
}
