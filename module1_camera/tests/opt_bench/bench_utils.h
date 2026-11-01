#pragma once

#include "chessboard_corner.h"
#include "calib_io.h"

#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fcbench {

namespace fs = std::filesystem;

inline fs::path sourceDir() {
    return fs::path(__FILE__).parent_path();
}

inline std::string datasetDir() {
    const char* env = std::getenv("FC_OPT_BENCH_DATA");
    if (env && *env) return env;
    return (sourceDir() / ".." / ".." / "test_data" / "camera261006-1688").lexically_normal().string();
}

inline std::string outDir() {
    const char* env = std::getenv("FC_OPT_BENCH_OUT");
    fs::path p = (env && *env) ? fs::path(env) : sourceDir() / "out";
    std::error_code ec;
    fs::create_directories(p, ec);
    return p.string();
}

inline bool datasetAvailable() {
    std::error_code ec;
    return fs::exists(fs::path(datasetDir()) / "left", ec)
        && fs::exists(fs::path(datasetDir()) / "right", ec);
}

inline std::optional<fc::CameraInput> loadDataset() {
    fc::CameraCalibConfig cfg;
    fc::CameraOpParams ops;
    fs::path master = sourceDir() / ".." / ".." / "camera_calib_params.json";
    std::error_code ec;
    if (fs::exists(master, ec)) {
        std::ifstream ifs(master.string());
        if (ifs.is_open()) {
            try {
                nlohmann::json jm = nlohmann::json::parse(ifs, nullptr, true);
                fc::applyCameraParamsJson(jm, cfg, ops);
            } catch (const std::exception& e) {
                spdlog::warn("opt_bench: master params parse failed: {}", e.what());
            }
        }
    }
    return fc::loadCameraInput(datasetDir(), &cfg, &ops);
}

inline fs::path datasetHistoryJson() {
    return fs::path(datasetDir()) / "camera_calib.json";
}

class Stopwatch {
public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

struct StepTimings {
    std::vector<std::pair<std::string, double>> steps;
    void add(const std::string& name, double ms) { steps.emplace_back(name, ms); }
    double total() const {
        double s = 0.0;
        for (auto& kv : steps) if (kv.second > 0.0) s += kv.second;
        return s;
    }
    nlohmann::json toJson() const {
        nlohmann::json j = nlohmann::json::object();
        for (auto& kv : steps) j[kv.first] = kv.second;
        return j;
    }
};

using CornerFn = std::function<bool(const cv::Mat&,
                                    const fc::ChessboardCornerParams&,
                                    fc::ChessboardCornerResult&)>;

struct CornerFrameResult {
    size_t origIndex = 0;
    std::string leftFile, rightFile;
    bool valid = false;
    std::string skipReason;
    double subpixL = 0.0, subpixR = 0.0;
    double msL = 0.0, msR = 0.0, msNorm = 0.0;
    std::vector<cv::Point2f> l, r;
};

inline void extractAllCorners(const fc::CameraInput& in,
                              const fc::ChessboardCornerParams& cp,
                              const CornerFn& fnIn,
                              int threads,
                              std::vector<CornerFrameResult>& out) {
    const size_t n = in.frames.size();
    out.assign(n, CornerFrameResult{});
    CornerFn fn = fnIn
        ? fnIn
        : CornerFn([](const cv::Mat& g, const fc::ChessboardCornerParams& p,
                      fc::ChessboardCornerResult& r) {
              return fc::extractChessboardCorners(g, p, r);
          });

    auto process = [&](size_t i) {
        CornerFrameResult rec;
        rec.origIndex = i;
        rec.leftFile = in.frames[i].leftFile;
        rec.rightFile = in.frames[i].rightFile;
        fc::ChessboardCornerResult rl, rr;
        Stopwatch sw;
        bool okL = fn(in.frames[i].leftGray, cp, rl);
        rec.msL = sw.ms();
        sw.reset();
        bool okR = fn(in.frames[i].rightGray, cp, rr);
        rec.msR = sw.ms();
        if (!okL || !okR) {
            rec.skipReason = !okL ? (!okR ? "corner_L+corner_R" : "corner_L") : "corner_R";
        } else {
            sw.reset();
            normalizeLRCornerOrder(rl, rr);
            rec.msNorm = sw.ms();
            rec.valid = true;
            rec.subpixL = rl.meanSubpixDelta;
            rec.subpixR = rr.meanSubpixDelta;
            rec.l = std::move(rl.corners);
            rec.r = std::move(rr.corners);
        }
        out[i] = std::move(rec);
    };

    int nt = std::max(1, threads);
    if (nt == 1 || n == 0) {
        for (size_t i = 0; i < n; ++i) process(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        pool.emplace_back([&] {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= n) break;
                process(i);
            }
        });
    }
    for (auto& th : pool) th.join();
}

inline calib::IntrinsicCalibParams makeIntrinsicParams(const fc::CameraCalibConfig& c) {
    calib::IntrinsicCalibParams p;
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

inline std::vector<cv::Point3f> buildObjectPoints(const fc::CameraCalibConfig& c) {
    std::vector<cv::Point3f> pts;
    const double actualSize = c.squareSizeMm *
        (1.0 + c.plateTempCoeff * (c.plateTemp - 20.0));
    for (int i = 0; i < c.chessHeight; ++i)
        for (int j = 0; j < c.chessWidth; ++j)
            pts.emplace_back(static_cast<float>(j * actualSize),
                             static_cast<float>(i * actualSize), 0.0f);
    return pts;
}

inline calib::ExtrinsicCalibCpuParams makeExtrinsicParams(
    const fc::CameraCalibConfig& c,
    const fc::CameraOpParams& ops,
    const std::vector<std::vector<cv::Point2f>>& lpts,
    const std::vector<std::vector<cv::Point2f>>& rpts) {
    calib::ExtrinsicCalibCpuParams p;
    p.leftPointsPerView = lpts;
    p.rightPointsPerView = rpts;
    p.imageSize = cv::Size(c.imageWidth, c.imageHeight);
    p.patternSize = cv::Size(c.chessWidth, c.chessHeight);
    p.squareSize = static_cast<float>(c.squareSizeMm);
    p.maxReprojError = c.reprojErrorThreshold * ops.extrinsicMaxReprojFactor;
    p.minViewCount = ops.extrinsicMinViewCount;
    p.objectPoints = buildObjectPoints(c);
    return p;
}

struct PipelineOptions {
    int cornerThreads = 1;
    CornerFn cornerFn = nullptr;
    std::vector<int> rejectValidIdx;
    std::optional<int> intrinsicFlags;
};

struct PipelineResults {
    std::vector<CornerFrameResult> frames;
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    std::vector<int> validOrigIdx;
    calib::IntrinsicCalibResult intrin;
    calib::ExtrinsicCalibCpuResult extrin;
    calib::StereoRectifyCpuResult rectify;
    calib::IntrinsicCompensateCPUResult tableIL, tableIR;
    calib::ExtrinsicCompensateCPUResult tableE;
    calib::StereoRectifyTempTableResult tableR2;
    nlohmann::json outputJson;
    StepTimings timing;
};

inline bool runPipeline(const fc::CameraInput& in,
                        const PipelineOptions& opt,
                        PipelineResults& R) {
    fc::ChessboardCornerParams cp;
    cp.patternSize = cv::Size(in.config.chessWidth, in.config.chessHeight);
    CornerFn fn = opt.cornerFn
        ? opt.cornerFn
        : CornerFn([](const cv::Mat& g, const fc::ChessboardCornerParams& p,
                      fc::ChessboardCornerResult& r) {
              return fc::extractChessboardCorners(g, p, r);
          });

    Stopwatch sw;
    extractAllCorners(in, cp, fn, opt.cornerThreads, R.frames);
    R.timing.add("corner_extract", sw.ms());

    for (const auto& f : R.frames) {
        if (f.valid) {
            R.lpts.push_back(f.l);
            R.rpts.push_back(f.r);
            R.validOrigIdx.push_back(static_cast<int>(f.origIndex));
        }
    }
    if (static_cast<int>(R.lpts.size()) < in.ops.framesMinValidFrames) {
        spdlog::error("opt_bench: too few valid frames: {}", R.lpts.size());
        return false;
    }

    if (!opt.rejectValidIdx.empty()) {
        std::set<int> rej(opt.rejectValidIdx.begin(), opt.rejectValidIdx.end());
        std::vector<std::vector<cv::Point2f>> l2, r2;
        std::vector<int> vi2;
        for (size_t v = 0; v < R.lpts.size(); ++v) {
            if (rej.count(static_cast<int>(v)) == 0) {
                l2.push_back(std::move(R.lpts[v]));
                r2.push_back(std::move(R.rpts[v]));
                vi2.push_back(R.validOrigIdx[v]);
            }
        }
        R.lpts = std::move(l2);
        R.rpts = std::move(r2);
        R.validOrigIdx = std::move(vi2);
    }

    sw.reset();
    calib::IntrinsicCalibParams ip = makeIntrinsicParams(in.config);
    if (opt.intrinsicFlags.has_value()) ip.calib_flags = *opt.intrinsicFlags;
    calib::IntrinsicCalibCPU intrin(ip);
    if (!intrin.Execute(R.lpts, R.rpts, R.intrin) || !R.intrin.success) {
        spdlog::error("opt_bench: intrinsic failed: {}", R.intrin.message);
        return false;
    }
    R.timing.add("intrinsic", sw.ms());

    sw.reset();
    calib::ExtrinsicCalibCpuParams ep = makeExtrinsicParams(in.config, in.ops, R.lpts, R.rpts);
    calib::ExtrinsicCalibCpu extrin(ep);
    R.extrin = extrin.Execute(R.intrin.left.camera_matrix, R.intrin.left.dist_coeffs,
                              R.intrin.right.camera_matrix, R.intrin.right.dist_coeffs);
    if (!R.extrin.success) {
        spdlog::error("opt_bench: extrinsic failed: {}", R.extrin.message);
        return false;
    }
    R.timing.add("extrinsic", sw.ms());

    sw.reset();
    calib::StereoRectifyCpuParams rp;
    rp.cameraMatrixL = R.intrin.left.camera_matrix;
    rp.distCoeffsL = R.intrin.left.dist_coeffs;
    rp.cameraMatrixR = R.intrin.right.camera_matrix;
    rp.distCoeffsR = R.intrin.right.dist_coeffs;
    rp.imageSize = cv::Size(in.config.imageWidth, in.config.imageHeight);
    rp.R = R.extrin.R;
    rp.T = R.extrin.T;
    rp.alpha = in.config.rectifyAlpha;
    rp.flags = in.config.rectifyFlags;
    calib::StereoRectifyCpu rectify(rp);
    R.rectify = rectify.Execute();
    if (!R.rectify.success) {
        spdlog::error("opt_bench: rectify failed: {}", R.rectify.message);
        return false;
    }
    R.timing.add("rectify", sw.ms());

    sw.reset();
    calib::CameraIntrinsics cL{R.intrin.left.camera_matrix.at<double>(0, 0),
                               R.intrin.left.camera_matrix.at<double>(1, 1),
                               R.intrin.left.camera_matrix.at<double>(0, 2),
                               R.intrin.left.camera_matrix.at<double>(1, 2),
                               in.config.referenceTemp};
    calib::CameraIntrinsics cR{R.intrin.right.camera_matrix.at<double>(0, 0),
                               R.intrin.right.camera_matrix.at<double>(1, 1),
                               R.intrin.right.camera_matrix.at<double>(0, 2),
                               R.intrin.right.camera_matrix.at<double>(1, 2),
                               in.config.referenceTemp};
    calib::IntrinsicCompensateCPUParams icp;
    icp.cte = in.config.cte;
    icp.tempStep = in.config.tempStep;
    icp.tempRangeMin = in.config.tempRangeMin;
    icp.tempRangeMax = in.config.tempRangeMax;
    calib::IntrinsicCompensateCPU icomp(icp);
    R.tableIL = icomp.Execute(cL);
    R.tableIR = icomp.Execute(cR);

    calib::CameraExtrinsics ce;
    ce.referenceTemp = in.config.referenceTemp;
    for (int i = 0; i < 3; ++i) ce.T[i] = R.extrin.T.at<double>(i);
    for (int i = 0; i < 9; ++i) ce.R[i] = R.extrin.R.at<double>(i / 3, i % 3);
    calib::ExtrinsicCompensateCPUParams ecp;
    ecp.cte = in.config.cte;
    ecp.tempStep = in.config.tempStep;
    ecp.tempRangeMin = in.config.tempRangeMin;
    ecp.tempRangeMax = in.config.tempRangeMax;
    calib::ExtrinsicCompensateCPU ecomp(ecp);
    R.tableE = ecomp.Execute(ce);

    calib::StereoRectifyTempTableParams strp;
    strp.cameraMatrixL = rp.cameraMatrixL;
    strp.distCoeffsL = rp.distCoeffsL;
    strp.cameraMatrixR = rp.cameraMatrixR;
    strp.distCoeffsR = rp.distCoeffsR;
    strp.imageSize = rp.imageSize;
    strp.R = rp.R;
    strp.T = rp.T;
    strp.referenceTemp = in.config.referenceTemp;
    strp.cte = in.config.cte;
    strp.tempStep = in.config.tempStep;
    strp.tempRangeMin = in.config.tempRangeMin;
    strp.tempRangeMax = in.config.tempRangeMax;
    strp.alpha = in.config.rectifyAlpha;
    strp.flags = in.config.rectifyFlags;
    calib::StereoRectifyTempTableCpu strtab(strp);
    R.tableR2 = strtab.Execute();
    R.timing.add("temp_tables", sw.ms());

    sw.reset();
    R.outputJson = fc::buildCameraCalibJson(in.config, R.intrin, R.extrin, R.rectify,
                                            R.tableIL, R.tableIR, R.tableE, R.tableR2);
    R.timing.add("json_build", sw.ms());
    return true;
}

struct AggMetrics {
    int validFrames = 0;
    double rmsL = 0.0, rmsR = 0.0;
    double reprojLMean = 0.0, reprojRMean = 0.0;
    double epiRawMean = 0.0, rowAlignMean = 0.0;
    double stereoReprojMean = 0.0;
    int stereoViews = 0;
    double subpixMean = 0.0;
};

inline AggMetrics computeMetrics(const calib::IntrinsicCalibResult& intrin,
                                 const calib::ExtrinsicCalibCpuResult& extrin,
                                 const calib::StereoRectifyCpuResult& rectify,
                                 const std::vector<std::vector<cv::Point2f>>& lpts,
                                 const std::vector<std::vector<cv::Point2f>>& rpts,
                                 const fc::CameraCalibConfig& cfg) {
    AggMetrics m;
    m.validFrames = static_cast<int>(lpts.size());
    m.rmsL = intrin.left.rms_error;
    m.rmsR = intrin.right.rms_error;
    if (!intrin.left.per_view_errors.empty()) {
        double s = 0.0;
        for (double v : intrin.left.per_view_errors) s += v;
        m.reprojLMean = s / intrin.left.per_view_errors.size();
    }
    if (!intrin.right.per_view_errors.empty()) {
        double s = 0.0;
        for (double v : intrin.right.per_view_errors) s += v;
        m.reprojRMean = s / intrin.right.per_view_errors.size();
    }

    const cv::Mat& F = extrin.F;
    const cv::Mat& R1 = rectify.R1, &R2 = rectify.R2;
    const cv::Mat& P1 = rectify.P1, &P2 = rectify.P2;
    std::vector<cv::Point3f> obj = buildObjectPoints(cfg);

    const bool haveViews = intrin.left.rvecs.size() == lpts.size()
        && intrin.right.rvecs.size() == rpts.size()
        && !lpts.empty();

    double epiSum = 0.0, rowSum = 0.0, stSum = 0.0;
    size_t cornerTotal = 0;
    for (size_t vi = 0; vi < lpts.size(); ++vi) {
        const auto& cl = lpts[vi];
        const auto& cr = rpts[vi];
        const size_t n = cl.size();
        std::vector<cv::Point2f> undL, undR;
        cv::undistortPoints(cl, undL, intrin.left.camera_matrix, intrin.left.dist_coeffs, R1, P1);
        cv::undistortPoints(cr, undR, intrin.right.camera_matrix, intrin.right.dist_coeffs, R2, P2);
        for (size_t j = 0; j < n; ++j) {
            const double xl = cl[j].x, yl = cl[j].y, xr = cr[j].x, yr = cr[j].y;
            const double a = F.at<double>(0, 0) * xl + F.at<double>(0, 1) * yl + F.at<double>(0, 2);
            const double b = F.at<double>(1, 0) * xl + F.at<double>(1, 1) * yl + F.at<double>(1, 2);
            const double c = F.at<double>(2, 0) * xl + F.at<double>(2, 1) * yl + F.at<double>(2, 2);
            epiSum += std::abs(a * xr + b * yr + c) / std::hypot(a, b);
            rowSum += std::abs(undL[j].y - undR[j].y);
        }
        cornerTotal += n;
        if (haveViews) {
            std::vector<cv::Point2f> projL, projR;
            cv::projectPoints(obj, intrin.left.rvecs[vi], intrin.left.tvecs[vi],
                              intrin.left.camera_matrix, intrin.left.dist_coeffs, projL);
            cv::projectPoints(obj, intrin.right.rvecs[vi], intrin.right.tvecs[vi],
                              intrin.right.camera_matrix, intrin.right.dist_coeffs, projR);
            double sqL = 0.0, sqR = 0.0;
            for (size_t j = 0; j < n; ++j) {
                const double dxL = cl[j].x - projL[j].x, dyL = cl[j].y - projL[j].y;
                const double dxR = cr[j].x - projR[j].x, dyR = cr[j].y - projR[j].y;
                sqL += dxL * dxL + dyL * dyL;
                sqR += dxR * dxR + dyR * dyR;
            }
            stSum += std::sqrt((sqL + sqR) / (2.0 * static_cast<double>(n)));
        }
    }
    if (cornerTotal > 0) {
        m.epiRawMean = epiSum / static_cast<double>(cornerTotal);
        m.rowAlignMean = rowSum / static_cast<double>(cornerTotal);
    }
    if (haveViews) {
        m.stereoReprojMean = stSum / static_cast<double>(lpts.size());
        m.stereoViews = static_cast<int>(lpts.size());
    }
    return m;
}

inline AggMetrics computeMetrics(const PipelineResults& R,
                                 const fc::CameraCalibConfig& cfg) {
    return computeMetrics(R.intrin, R.extrin, R.rectify, R.lpts, R.rpts, cfg);
}

inline nlohmann::json metricsJson(const AggMetrics& m) {
    nlohmann::json j;
    j["validFrames"] = m.validFrames;
    j["rmsL"] = m.rmsL;
    j["rmsR"] = m.rmsR;
    j["reprojLMean"] = m.reprojLMean;
    j["reprojRMean"] = m.reprojRMean;
    j["epiRawMean"] = m.epiRawMean;
    j["rowAlignMean"] = m.rowAlignMean;
    j["stereoReprojMean"] = m.stereoReprojMean;
    j["stereoViews"] = m.stereoViews;
    j["subpixMean"] = m.subpixMean;
    return j;
}

inline void writeTextFile(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p, std::ios::binary);
    if (!ofs.is_open()) {
        spdlog::warn("opt_bench: cannot write {}", p.string());
        return;
    }
    ofs << content;
}

inline void writeTimingArtifacts(const std::string& tag,
                                 const StepTimings& st,
                                 const std::vector<CornerFrameResult>& frames) {
    fs::path dir(outDir());
    {
        std::ostringstream os;
        os << "step,ms\n";
        for (auto& kv : st.steps) os << kv.first << "," << kv.second << "\n";
        writeTextFile(dir / ("timing_" + tag + "_steps.csv"), os.str());
    }
    {
        std::ostringstream os;
        os << "orig_index,left_file,valid,skip_reason,ms_left,ms_right,ms_norm,subpixL,subpixR\n";
        for (const auto& f : frames) {
            os << f.origIndex << "," << f.leftFile << "," << (f.valid ? 1 : 0) << ","
               << f.skipReason << "," << f.msL << "," << f.msR << "," << f.msNorm << ","
               << f.subpixL << "," << f.subpixR << "\n";
        }
        writeTextFile(dir / ("timing_" + tag + "_frames.csv"), os.str());
    }
}

inline double relDiff(double a, double b) {
    if (a == 0.0 && b == 0.0) return 0.0;
    const double denom = std::max(std::abs(a), std::abs(b));
    if (denom == 0.0) return 0.0;
    return std::abs(a - b) / denom;
}

inline double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline bool matsByteEqual(const cv::Mat& a, const cv::Mat& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type()) return false;
    if (a.empty() && b.empty()) return true;
    const cv::Mat ac = a.isContinuous() ? a : a.clone();
    const cv::Mat bc = b.isContinuous() ? b : b.clone();
    return 0 == std::memcmp(ac.data, bc.data, ac.total() * ac.elemSize());
}

inline double matMaxAbsDiff(const cv::Mat& a, const cv::Mat& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
        return std::numeric_limits<double>::infinity();
    if (a.empty()) return 0.0;
    cv::Mat d;
    cv::absdiff(a, b, d);
    double m = 0.0;
    cv::minMaxLoc(d, nullptr, &m);
    return m;
}

inline bool monoResultsByteEqual(const calib::MonocularCalibResult& a,
                                 const calib::MonocularCalibResult& b) {
    if (!matsByteEqual(a.camera_matrix, b.camera_matrix)) return false;
    if (!matsByteEqual(a.dist_coeffs, b.dist_coeffs)) return false;
    if (a.rms_error != b.rms_error) return false;
    if (a.rvecs.size() != b.rvecs.size() || a.tvecs.size() != b.tvecs.size()) return false;
    for (size_t i = 0; i < a.rvecs.size(); ++i) {
        if (!matsByteEqual(a.rvecs[i], b.rvecs[i])) return false;
        if (!matsByteEqual(a.tvecs[i], b.tvecs[i])) return false;
    }
    if (a.per_view_errors.size() != b.per_view_errors.size()) return false;
    for (size_t i = 0; i < a.per_view_errors.size(); ++i)
        if (a.per_view_errors[i] != b.per_view_errors[i]) return false;
    return true;
}

inline std::map<std::string, std::string> loadMarkedGates(const fs::path& csv) {
    std::map<std::string, std::string> out;
    std::error_code ec;
    if (!fs::exists(csv, ec)) return out;
    std::ifstream ifs(csv.string());
    if (!ifs.is_open()) return out;
    std::string line;
    std::getline(ifs, line);
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string f, hit;
        std::getline(ss, f, ',');
        std::getline(ss, hit, ',');
        std::getline(ss, hit, ',');
        std::getline(ss, hit, ',');
        std::getline(ss, hit, ',');
        if (!f.empty()) out[f] = hit;
    }
    return out;
}

} // namespace fcbench
