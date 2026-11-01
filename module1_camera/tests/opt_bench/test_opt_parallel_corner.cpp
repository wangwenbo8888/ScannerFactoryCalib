#include <gtest/gtest.h>
#include "bench_utils.h"

#include <algorithm>

using namespace fcbench;

TEST(OptParallelCorner, SameResultsAsSequential) {
    if (!datasetAvailable()) GTEST_SKIP() << "dataset camera261006-1688 absent";

    auto inOpt = loadDataset();
    ASSERT_TRUE(inOpt.has_value());
    const fc::CameraInput& in = *inOpt;
    ASSERT_GT(in.frames.size(), 0u);

    fc::ChessboardCornerParams cp;
    cp.patternSize = cv::Size(in.config.chessWidth, in.config.chessHeight);

    unsigned hw = std::thread::hardware_concurrency();
    int threads = static_cast<int>(std::min(8u, hw == 0 ? 1u : hw));

    std::vector<CornerFrameResult> seq, par;
    Stopwatch sw;
    extractAllCorners(in, cp, nullptr, 1, seq);
    const double seqMs = sw.ms();
    sw.reset();
    extractAllCorners(in, cp, nullptr, threads, par);
    const double parMs = sw.ms();

    ASSERT_EQ(seq.size(), par.size());
    ASSERT_EQ(seq.size(), in.frames.size());

    size_t validSeq = 0, validPar = 0;
    size_t patternMismatch = 0, pointMismatch = 0, countMismatch = 0;
    double maxDelta = 0.0;
    for (size_t i = 0; i < seq.size(); ++i) {
        if (seq[i].valid) ++validSeq;
        if (par[i].valid) ++validPar;
        if (seq[i].valid != par[i].valid) {
            ++patternMismatch;
            continue;
        }
        if (!seq[i].valid) continue;
        if (seq[i].l.size() != par[i].l.size() || seq[i].r.size() != par[i].r.size()) {
            ++countMismatch;
            continue;
        }
        for (size_t j = 0; j < seq[i].l.size(); ++j) {
            if (seq[i].l[j].x != par[i].l[j].x || seq[i].l[j].y != par[i].l[j].y
                || seq[i].r[j].x != par[i].r[j].x || seq[i].r[j].y != par[i].r[j].y) {
                ++pointMismatch;
                maxDelta = std::max({maxDelta,
                    static_cast<double>(std::abs(seq[i].l[j].x - par[i].l[j].x)),
                    static_cast<double>(std::abs(seq[i].l[j].y - par[i].l[j].y)),
                    static_cast<double>(std::abs(seq[i].r[j].x - par[i].r[j].x)),
                    static_cast<double>(std::abs(seq[i].r[j].y - par[i].r[j].y))});
            }
        }
    }

    EXPECT_EQ(patternMismatch, 0u);
    EXPECT_EQ(countMismatch, 0u);
    EXPECT_EQ(pointMismatch, 0u);
    EXPECT_EQ(validSeq, validPar);
    EXPECT_GT(validSeq, 0u);

    const double speedup = parMs > 0.0 ? seqMs / parMs : 0.0;
    spdlog::info("opt_bench S1 parallel corner: frames={} valid={} threads={} "
                 "seqMs={:.0f} parMs={:.0f} speedup={:.2f}x maxPointDelta={:g}",
                 in.frames.size(), validSeq, threads, seqMs, parMs, speedup, maxDelta);

    nlohmann::json rep;
    rep["experiment"] = "S1_parallel_corner";
    rep["threads"] = threads;
    rep["frames"] = in.frames.size();
    rep["validFrames"] = validSeq;
    rep["seqMs"] = seqMs;
    rep["parMs"] = parMs;
    rep["speedup"] = speedup;
    rep["pointMismatch"] = pointMismatch;
    rep["patternMismatch"] = patternMismatch;
    rep["maxPointDelta"] = maxDelta;
    rep["verdict"] = (pointMismatch == 0 && patternMismatch == 0)
        ? "identical" : "MISMATCH";
    writeTextFile(fs::path(outDir()) / "exp_parallel_corner_report.json", rep.dump(2));
}
