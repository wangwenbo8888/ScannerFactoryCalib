#include <gtest/gtest.h>
#include "bench_utils.h"

using namespace fcbench;

namespace {

struct DiagRun {
    bool singleThread;
    double ms;
    double rms;
};

DiagRun runOne(const std::vector<std::vector<cv::Point2f>>& lpts,
               const fc::CameraCalibConfig& cfg) {
    calib::IntrinsicCalibParams ip = makeIntrinsicParams(cfg);
    calib::IntrinsicCalibCPU cal(ip);
    calib::MonocularCalibResult res;
    Stopwatch sw;
    const bool ok = cal.Execute(lpts, res);
    const double ms = sw.ms();
    EXPECT_TRUE(ok && res.success);
    return DiagRun{false, ms, res.rms_error};
}

} // namespace

TEST(OptDiagIntrinsicThreads, AbbaMedianDefaultVsSingle) {
    if (!datasetAvailable()) GTEST_SKIP() << "dataset camera261006-1688 absent";

    auto inOpt = loadDataset();
    ASSERT_TRUE(inOpt.has_value());
    const fc::CameraInput& in = *inOpt;
    ASSERT_GT(in.frames.size(), 0u);

    fc::ChessboardCornerParams cp;
    cp.patternSize = cv::Size(in.config.chessWidth, in.config.chessHeight);
    int cornerThreads = static_cast<int>(
        std::min(8u, std::thread::hardware_concurrency()));
    std::vector<CornerFrameResult> frames;
    extractAllCorners(in, cp, nullptr, cornerThreads, frames);
    std::vector<std::vector<cv::Point2f>> lpts;
    for (const auto& f : frames) if (f.valid) lpts.push_back(f.l);
    ASSERT_GT(lpts.size(), 0u);

    const unsigned hw = std::thread::hardware_concurrency();
    const int nDefault = cv::getNumThreads();

    std::vector<double> defMs, sngMs;
    double rmsDef = 0.0, rmsSng = 0.0;
    const std::vector<bool> order = {false, true, true, false, false, true};
    for (bool single : order) {
        cv::setNumThreads(single ? 1 : 0);
        DiagRun r = runOne(lpts, in.config);
        r.singleThread = single;
        if (single) {
            sngMs.push_back(r.ms);
            rmsSng = r.rms;
        } else {
            defMs.push_back(r.ms);
            rmsDef = r.rms;
        }
        spdlog::info("opt_bench diag: mode={} ms={:.0f} rms={:.8f}",
                     single ? "single" : "default", r.ms, r.rms);
    }
    cv::setNumThreads(0);

    const double medDef = medianOf(defMs);
    const double medSng = medianOf(sngMs);
    const double ratio = medSng > 0.0 ? medDef / medSng : 0.0;
    const double spreadDef = *std::max_element(defMs.begin(), defMs.end())
        - *std::min_element(defMs.begin(), defMs.end());
    const double spreadSng = *std::max_element(sngMs.begin(), sngMs.end())
        - *std::min_element(sngMs.begin(), sngMs.end());

    spdlog::info("opt_bench diag: hw={} opencvThreads={} medDefault={:.0f} "
                 "medSingle={:.0f} ratio={:.3f} spreadDef={:.0f}s spreadSng={:.0f}s",
                 hw, nDefault, medDef, medSng, ratio, spreadDef / 1000.0,
                 spreadSng / 1000.0);

    EXPECT_EQ(rmsDef, rmsSng);

    std::string conclusion;
    if (ratio > 0.9) conclusion = "internally_serial_or_weak";
    else if (ratio > 0.6) conclusion = "partial_internal_parallel";
    else conclusion = "internally_parallel";

    nlohmann::json rep;
    rep["experiment"] = "DIAG_intrinsic_threads_abba";
    rep["protocol"] = "3+3 reps, ABBA interleaved, median";
    rep["hwThreads"] = hw;
    rep["opencvDefaultThreads"] = nDefault;
    rep["defaultMs"] = defMs;
    rep["singleThreadMs"] = sngMs;
    rep["medianDefaultMs"] = medDef;
    rep["medianSingleMs"] = medSng;
    rep["ratio"] = ratio;
    rep["spreadDefaultMs"] = spreadDef;
    rep["spreadSingleMs"] = spreadSng;
    rep["conclusion"] = conclusion;
    writeTextFile(fs::path(outDir()) / "exp_diag_intrinsic_threads_report.json",
                  rep.dump(2));
}
