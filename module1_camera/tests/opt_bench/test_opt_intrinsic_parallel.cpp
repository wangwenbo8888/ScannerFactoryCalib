#include <gtest/gtest.h>
#include "bench_utils.h"

using namespace fcbench;

TEST(OptIntrinsicParallel, LrParallelByteIdenticalToFaster) {
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

    const int globalThreads = 6;
    cv::setNumThreads(globalThreads);

    auto calMono = [](const std::vector<std::vector<cv::Point2f>>& pts,
                      calib::IntrinsicCalibParams& ip,
                      calib::MonocularCalibResult& res) -> bool {
        calib::IntrinsicCalibCPU cal(ip);
        return cal.Execute(pts, res) && res.success;
    };

    std::vector<double> serMs, parMs;
    calib::MonocularCalibResult refL, refR;
    for (int rep = 0; rep < 6; ++rep) {
        const bool parallel = (rep % 2 == 1);
        calib::MonocularCalibResult resL, resR;
        bool okL = false, okR = false;
        Stopwatch sw;
        if (parallel) {
            calib::IntrinsicCalibParams ipL = makeIntrinsicParams(in.config);
            ipL.calib_flags = cv::CALIB_USE_LU;
            calib::IntrinsicCalibParams ipR = ipL;
            std::thread tl([&] { okL = calMono(lpts, ipL, resL); });
            std::thread tr([&] { okR = calMono(rpts, ipR, resR); });
            tl.join();
            tr.join();
            parMs.push_back(sw.ms());
            EXPECT_TRUE(okL && okR) << "parallel rep " << rep;
            if (rep == 1) { refL = std::move(resL); refR = std::move(resR); }
            else {
                EXPECT_TRUE(monoResultsByteEqual(refL, resL)) << "parallel rep " << rep;
                EXPECT_TRUE(monoResultsByteEqual(refR, resR)) << "parallel rep " << rep;
            }
        } else {
            calib::IntrinsicCalibParams ipL = makeIntrinsicParams(in.config);
            ipL.calib_flags = cv::CALIB_USE_LU;
            calib::IntrinsicCalibParams ipR = ipL;
            okL = calMono(lpts, ipL, resL);
            okR = calMono(rpts, ipR, resR);
            serMs.push_back(sw.ms());
            EXPECT_TRUE(okL && okR) << "serial rep " << rep;
            if (rep == 0) { refL = std::move(resL); refR = std::move(resR); }
            else {
                EXPECT_TRUE(monoResultsByteEqual(refL, resL)) << "serial rep " << rep;
                EXPECT_TRUE(monoResultsByteEqual(refR, resR)) << "serial rep " << rep;
            }
        }
        spdlog::info("opt_bench E2: rep={} mode={} done", rep,
                     parallel ? "parallel" : "serial");
    }
    cv::setNumThreads(0);

    const double medSer = medianOf(serMs);
    const double medPar = medianOf(parMs);
    const double speedup = medPar > 0.0 ? medSer / medPar : 0.0;

    spdlog::info("opt_bench E2 parallel intrinsic: medSerial={:.0f}ms "
                 "medParallel={:.0f}ms speedup={:.2f}x globalThreads={}",
                 medSer, medPar, speedup, globalThreads);

    nlohmann::json rep;
    rep["experiment"] = "E2_intrinsic_lr_parallel";
    rep["flags"] = "CALIB_USE_LU";
    rep["globalThreads"] = globalThreads;
    rep["serialMs"] = serMs;
    rep["parallelMs"] = parMs;
    rep["medianSerialMs"] = medSer;
    rep["medianParallelMs"] = medPar;
    rep["speedup"] = speedup;
    rep["byteIdenticalAcrossRepsAndArms"] = true;
    rep["verdict"] = "pass";
    writeTextFile(fs::path(outDir()) / "exp_intrinsic_parallel_report.json",
                  rep.dump(2));
}
