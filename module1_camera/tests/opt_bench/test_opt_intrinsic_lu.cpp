#include <gtest/gtest.h>
#include "bench_utils.h"

using namespace fcbench;

TEST(OptIntrinsicLu, FasterAndNumericallySafe) {
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

    auto runIntrinsic = [&](int flags, calib::IntrinsicCalibResult& res) {
        calib::IntrinsicCalibParams ip = makeIntrinsicParams(in.config);
        ip.calib_flags = flags;
        calib::IntrinsicCalibCPU cal(ip);
        Stopwatch sw;
        const bool ok = cal.Execute(lpts, rpts, res);
        const double ms = sw.ms();
        EXPECT_TRUE(ok && res.success);
        return ms;
    };

    calib::IntrinsicCalibResult resSvd, resLu;
    std::vector<double> svdMs, luMs;
    const std::vector<int> order = {0, 1, 1, 0, 0, 1};
    for (int arm : order) {
        if (arm == 0) svdMs.push_back(runIntrinsic(0, resSvd));
        else luMs.push_back(runIntrinsic(cv::CALIB_USE_LU, resLu));
        spdlog::info("opt_bench E1: arm={} svdMs={} luMs={}",
                     arm, svdMs.empty() ? "-" : std::to_string(svdMs.back()),
                     luMs.empty() ? "-" : std::to_string(luMs.back()));
    }

    const double medSvd = medianOf(svdMs);
    const double medLu = medianOf(luMs);
    const double speedup = medLu > 0.0 ? medSvd / medLu : 0.0;

    double maxKRel = 0.0;
    for (const calib::MonocularCalibResult* m :
         {&resSvd.left, &resSvd.right}) {
        const calib::MonocularCalibResult* n =
            (m == &resSvd.left) ? &resLu.left : &resLu.right;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                const double a = m->camera_matrix.at<double>(r, c);
                const double b = n->camera_matrix.at<double>(r, c);
                maxKRel = std::max(maxKRel, relDiff(a, b));
            }
    }
    double maxDDelta = 0.0;
    for (const calib::MonocularCalibResult* m :
         {&resSvd.left, &resSvd.right}) {
        const calib::MonocularCalibResult* n =
            (m == &resSvd.left) ? &resLu.left : &resLu.right;
        const int nD = std::min(m->dist_coeffs.cols, n->dist_coeffs.cols);
        for (int c = 0; c < nD; ++c) {
            const double a = m->dist_coeffs.at<double>(c);
            const double b = n->dist_coeffs.at<double>(c);
            const double tol = 1e-9 + 1e-7 * std::max(std::abs(a), std::abs(b));
            maxDDelta = std::max(maxDDelta, std::abs(a - b) / tol);
        }
    }
    const double rmsLRel = relDiff(resLu.left.rms_error, resSvd.left.rms_error);
    const double rmsRRel = relDiff(resLu.right.rms_error, resSvd.right.rms_error);

    EXPECT_LT(maxKRel, 1e-8) << "maxKRel=" << maxKRel;
    EXPECT_LT(maxDDelta, 1.0) << "distDelta/tol=" << maxDDelta;
    EXPECT_LT(rmsLRel, 1e-6);
    EXPECT_LT(rmsRRel, 1e-6);

    PipelineOptions optA;
    optA.cornerThreads = cornerThreads;
    PipelineResults RA;
    ASSERT_TRUE(runPipeline(in, optA, RA));
    PipelineOptions optB;
    optB.cornerThreads = cornerThreads;
    optB.intrinsicFlags = static_cast<int>(cv::CALIB_USE_LU);
    PipelineResults RB;
    ASSERT_TRUE(runPipeline(in, optB, RB));

    AggMetrics MA = computeMetrics(RA, in.config);
    AggMetrics MB = computeMetrics(RB, in.config);
    const double epiRel = relDiff(MB.epiRawMean, MA.epiRawMean);
    const double rowRel = relDiff(MB.rowAlignMean, MA.rowAlignMean);
    EXPECT_LT(epiRel, 1e-3) << "epiRel=" << epiRel;
    EXPECT_LT(rowRel, 1e-3) << "rowRel=" << rowRel;

    spdlog::info("opt_bench E1 LU: medSvd={:.0f}ms medLu={:.0f}ms speedup={:.2f}x "
                 "maxKRel={:g} maxDTol={:g} rmsLRel={:g} rmsRRel={:g} "
                 "epiRel={:g} rowRel={:g}",
                 medSvd, medLu, speedup, maxKRel, maxDDelta, rmsLRel, rmsRRel,
                 epiRel, rowRel);

    nlohmann::json rep;
    rep["experiment"] = "E1_intrinsic_LU";
    rep["protocol"] = "3+3 reps ABBA interleaved, median";
    rep["svdMs"] = svdMs;
    rep["luMs"] = luMs;
    rep["medianSvdMs"] = medSvd;
    rep["medianLuMs"] = medLu;
    rep["speedup"] = speedup;
    rep["maxKRel"] = maxKRel;
    rep["distDeltaOverTol"] = maxDDelta;
    rep["rmsLRel"] = rmsLRel;
    rep["rmsRRel"] = rmsRRel;
    rep["downstream"]["epiRel"] = epiRel;
    rep["downstream"]["rowAlignRel"] = rowRel;
    const bool pass = maxKRel < 1e-8 && maxDDelta < 1.0 && rmsLRel < 1e-6
        && rmsRRel < 1e-6 && epiRel < 1e-3 && rowRel < 1e-3;
    rep["verdict"] = pass ? "pass" : "FAIL";
    writeTextFile(fs::path(outDir()) / "exp_intrinsic_lu_report.json", rep.dump(2));
}
