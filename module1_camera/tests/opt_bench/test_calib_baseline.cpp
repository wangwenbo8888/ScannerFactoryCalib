#include <gtest/gtest.h>
#include "bench_utils.h"

#include <fstream>

using namespace fcbench;

namespace {

std::string readFileText(const fs::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs.is_open()) return {};
    std::ostringstream os;
    os << ifs.rdbuf();
    return os.str();
}

} // namespace

TEST(OptBenchBaseline, RunSixStepsAndValidateAgainstHistory) {
    if (!datasetAvailable()) GTEST_SKIP() << "dataset camera261006-1688 absent";

    Stopwatch swAll;

    Stopwatch swLoad;
    auto inOpt = loadDataset();
    ASSERT_TRUE(inOpt.has_value()) << "loadCameraInput failed";
    const double loadMs = swLoad.ms();
    const fc::CameraInput& in = *inOpt;
    ASSERT_GT(in.frames.size(), 0u);

    PipelineOptions opt;
    opt.cornerThreads = 1;
    PipelineResults R;
    ASSERT_TRUE(runPipeline(in, opt, R));
    R.timing.add("load_input", loadMs);

    AggMetrics M = computeMetrics(R, in.config);

    EXPECT_EQ(R.intrin.valid_frames_count, static_cast<int>(R.lpts.size()));
    EXPECT_GT(R.intrin.valid_frames_count, 0);
    EXPECT_TRUE(std::isfinite(M.rmsL) && M.rmsL > 0.0);
    EXPECT_TRUE(std::isfinite(M.rmsR) && M.rmsR > 0.0);
    EXPECT_TRUE(std::isfinite(M.epiRawMean) && M.epiRawMean > 0.0);
    EXPECT_GT(M.stereoViews, 0);

    fs::path hist = datasetHistoryJson();
    std::error_code ec;
    if (fs::exists(hist, ec)) {
        std::ifstream ifs(hist.string());
        ASSERT_TRUE(ifs.is_open());
        nlohmann::json hj = nlohmann::json::parse(ifs, nullptr, true);
        const double hRmsL = hj["intrinsic"]["left"]["rms_error"].get<double>();
        const double hRmsR = hj["intrinsic"]["right"]["rms_error"].get<double>();
        const int hFrames = hj["intrinsic"]["valid_frames_count"].get<int>();
        const double hStereo = hj["extrinsic"]["stereoReprojError"].get<double>();
        const double hEpi = hj["extrinsic"]["epipolarErrorMean"].get<double>();

        EXPECT_EQ(R.intrin.valid_frames_count, hFrames);
        EXPECT_LT(relDiff(M.rmsL, hRmsL), 1e-3) << "rmsL=" << M.rmsL << " hist=" << hRmsL;
        EXPECT_LT(relDiff(M.rmsR, hRmsR), 1e-3) << "rmsR=" << M.rmsR << " hist=" << hRmsR;
        EXPECT_LT(relDiff(R.extrin.stereoReprojError, hStereo), 1e-2)
            << "stereo=" << R.extrin.stereoReprojError << " hist=" << hStereo;
        EXPECT_LT(relDiff(R.extrin.epipolarErrorMean, hEpi), 1e-2)
            << "epi=" << R.extrin.epipolarErrorMean << " hist=" << hEpi;

        const bool byteIdentical = (readFileText(hist) == R.outputJson.dump(2));
        spdlog::info("opt_bench baseline: byte-identical with history json = {}",
                     byteIdentical);
    } else {
        spdlog::warn("opt_bench baseline: history json absent, skip cross-check");
    }

    nlohmann::json snap;
    snap["dataset"] = {{"dir", datasetDir()}, {"frames_input", in.frames.size()},
                       {"frames_valid", R.lpts.size()}};
    snap["params"] = {{"chess", {in.config.chessWidth, in.config.chessHeight}},
                      {"square_mm", in.config.squareSizeMm},
                      {"image", {in.config.imageWidth, in.config.imageHeight}},
                      {"referenceTemp", in.config.referenceTemp},
                      {"rectifyAlpha", in.config.rectifyAlpha}};
    snap["metrics"] = metricsJson(M);
    snap["metrics"]["extrinStereoReproj"] = R.extrin.stereoReprojError;
    snap["metrics"]["extrinEpiMean"] = R.extrin.epipolarErrorMean;
    snap["timing_steps"] = R.timing.toJson();
    snap["timing_total_ms"] = R.timing.total();
    snap["output"] = R.outputJson;
    writeTextFile(fs::path(outDir()) / "baseline_snapshot.json", snap.dump(2));

    writeTimingArtifacts("baseline", R.timing, R.frames);

    spdlog::info("opt_bench baseline done: frames={}/{} rmsL={:.4f} rmsR={:.4f} "
                 "epiRaw={:.4f} rowAlign={:.4f} stereoReproj={:.4f} totalMs={:.0f}",
                 R.lpts.size(), in.frames.size(), M.rmsL, M.rmsR,
                 M.epiRawMean, M.rowAlignMean, M.stereoReprojMean, swAll.ms());
}
