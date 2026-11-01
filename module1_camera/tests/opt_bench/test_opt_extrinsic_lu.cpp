#include <gtest/gtest.h>
#include "bench_utils.h"

using namespace fcbench;

namespace {

struct StereoOut {
    cv::Mat R, T, E, F;
    double rms = 0.0;
    double ms = 0.0;
};

StereoOut runStereo(const calib::ExtrinsicCalibCpuParams& ep,
                    const cv::Mat& KL, const cv::Mat& DL,
                    const cv::Mat& KR, const cv::Mat& DR,
                    int extraFlags) {
    StereoOut o;
    std::vector<std::vector<cv::Point3f>> objPerView(
        ep.leftPointsPerView.size(), ep.objectPoints);
    Stopwatch sw;
    o.rms = cv::stereoCalibrate(
        objPerView, ep.leftPointsPerView, ep.rightPointsPerView,
        KL, DL, KR, DR, ep.imageSize,
        o.R, o.T, o.E, o.F,
        cv::CALIB_FIX_INTRINSIC | extraFlags,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6));
    o.ms = sw.ms();
    return o;
}

double epiRawMeanWithF(const cv::Mat& F,
                       const std::vector<std::vector<cv::Point2f>>& lpts,
                       const std::vector<std::vector<cv::Point2f>>& rpts) {
    double sum = 0.0;
    size_t cnt = 0;
    for (size_t vi = 0; vi < lpts.size(); ++vi) {
        for (size_t j = 0; j < lpts[vi].size(); ++j) {
            const double xl = lpts[vi][j].x, yl = lpts[vi][j].y;
            const double xr = rpts[vi][j].x, yr = rpts[vi][j].y;
            const double a = F.at<double>(0, 0) * xl + F.at<double>(0, 1) * yl + F.at<double>(0, 2);
            const double b = F.at<double>(1, 0) * xl + F.at<double>(1, 1) * yl + F.at<double>(1, 2);
            const double c = F.at<double>(2, 0) * xl + F.at<double>(2, 1) * yl + F.at<double>(2, 2);
            sum += std::abs(a * xr + b * yr + c) / std::hypot(a, b);
            ++cnt;
        }
    }
    return cnt ? sum / cnt : 0.0;
}

} // namespace

TEST(OptExtrinsicLu, SameResultMuchFaster) {
    if (!datasetAvailable()) GTEST_SKIP() << "dataset camera261006-1688 absent";

    auto inOpt = loadDataset();
    ASSERT_TRUE(inOpt.has_value());
    const fc::CameraInput& in = *inOpt;

    fc::ChessboardCornerParams cp;
    cp.patternSize = cv::Size(in.config.chessWidth, in.config.chessHeight);
    int cornerThreads = static_cast<int>(
        std::min(8u, std::thread::hardware_concurrency()));
    std::vector<CornerFrameResult> frames;
    extractAllCorners(in, cp, nullptr, cornerThreads, frames);
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    for (const auto& f : frames) {
        if (f.valid) { lpts.push_back(f.l); rpts.push_back(f.r); }
    }
    ASSERT_GT(lpts.size(), 0u);

    calib::IntrinsicCalibParams ip = makeIntrinsicParams(in.config);
    ip.calib_flags = cv::CALIB_USE_LU;
    calib::IntrinsicCalibCPU cal(ip);
    calib::IntrinsicCalibResult intrin;
    ASSERT_TRUE(cal.Execute(lpts, rpts, intrin) && intrin.success);

    calib::ExtrinsicCalibCpuParams ep = makeExtrinsicParams(in.config, in.ops, lpts, rpts);

    StereoOut refSvd, refLu;
    std::vector<double> svdMs, luMs;
    const std::vector<int> order = {0, 1, 1, 0, 0, 1};
    for (int arm : order) {
        StereoOut o = runStereo(ep,
            intrin.left.camera_matrix, intrin.left.dist_coeffs,
            intrin.right.camera_matrix, intrin.right.dist_coeffs,
            arm == 0 ? 0 : static_cast<int>(cv::CALIB_USE_LU));
        if (arm == 0) {
            svdMs.push_back(o.ms);
            if (svdMs.size() == 1) refSvd = o;
        } else {
            luMs.push_back(o.ms);
            if (luMs.size() == 1) refLu = o;
        }
        spdlog::info("opt_bench E6: arm={} ms={:.0f} rms={:.8f}", arm, o.ms, o.rms);
    }

    const double medSvd = medianOf(svdMs);
    const double medLu = medianOf(luMs);
    const double speedup = medLu > 0.0 ? medSvd / medLu : 0.0;

    auto matTol = [](const cv::Mat& a, const cv::Mat& b) {
        double worst = 0.0;
        for (int r = 0; r < a.rows; ++r)
            for (int c = 0; c < a.cols; ++c) {
                const double x = a.at<double>(r, c), y = b.at<double>(r, c);
                const double tol = 1e-12 + 1e-9 * std::max(std::abs(x), std::abs(y));
                worst = std::max(worst, std::abs(x - y) / tol);
            }
        return worst;
    };
    const double rTol = matTol(refSvd.R, refLu.R);
    const double tTol = matTol(refSvd.T, refLu.T);
    const double rmsRel = relDiff(refLu.rms, refSvd.rms);
    const double epiSvd = epiRawMeanWithF(refSvd.F, lpts, rpts);
    const double epiLu = epiRawMeanWithF(refLu.F, lpts, rpts);
    const double epiRel = relDiff(epiLu, epiSvd);

    EXPECT_LT(rTol, 1.0) << "R delta/tol=" << rTol;
    EXPECT_LT(tTol, 1.0) << "T delta/tol=" << tTol;
    EXPECT_LT(rmsRel, 1e-6) << "rmsRel=" << rmsRel;
    EXPECT_LT(epiRel, 1e-6) << "epiRel=" << epiRel;

    spdlog::info("opt_bench E6: medSvd={:.0f}ms medLu={:.0f}ms speedup={:.2f}x "
                 "rTol={:g} tTol={:g} rmsRel={:g} epiRel={:g}",
                 medSvd, medLu, speedup, rTol, tTol, rmsRel, epiRel);

    nlohmann::json rep;
    rep["experiment"] = "E6_extrinsic_LU";
    rep["protocol"] = "3+3 reps ABBA interleaved, median; flags=CALIB_FIX_INTRINSIC(+LU)";
    rep["svdMs"] = svdMs;
    rep["luMs"] = luMs;
    rep["medianSvdMs"] = medSvd;
    rep["medianLuMs"] = medLu;
    rep["speedup"] = speedup;
    rep["rDeltaOverTol"] = rTol;
    rep["tDeltaOverTol"] = tTol;
    rep["rmsRel"] = rmsRel;
    rep["epiRel"] = epiRel;
    const bool pass = rTol < 1.0 && tTol < 1.0 && rmsRel < 1e-6 && epiRel < 1e-6;
    rep["verdict"] = pass ? "pass" : "FAIL";
    writeTextFile(fs::path(outDir()) / "exp_extrinsic_lu_report.json", rep.dump(2));
}
