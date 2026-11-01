#include <gtest/gtest.h>
#include "bench_utils.h"

#include <algorithm>
#include <numeric>

using namespace fcbench;

TEST(OptFrameReject, TopKByReprojNotWorse) {
    if (!datasetAvailable()) GTEST_SKIP() << "dataset camera261006-1688 absent";

    auto inOpt = loadDataset();
    ASSERT_TRUE(inOpt.has_value());
    const fc::CameraInput& in = *inOpt;

    PipelineOptions optA;
    optA.cornerThreads = 1;
    PipelineResults RA;
    ASSERT_TRUE(runPipeline(in, optA, RA));
    const int nViews = static_cast<int>(RA.lpts.size());
    ASSERT_GT(nViews, 0);
    ASSERT_EQ(static_cast<int>(RA.intrin.left.per_view_errors.size()), nViews);
    ASSERT_EQ(static_cast<int>(RA.intrin.right.per_view_errors.size()), nViews);

    std::vector<double> score(static_cast<size_t>(nViews));
    for (int v = 0; v < nViews; ++v) {
        score[static_cast<size_t>(v)] = std::max(
            RA.intrin.left.per_view_errors[static_cast<size_t>(v)],
            RA.intrin.right.per_view_errors[static_cast<size_t>(v)]);
    }
    std::vector<int> order(static_cast<size_t>(nViews));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return score[static_cast<size_t>(a)] > score[static_cast<size_t>(b)];
    });

    const int K = std::max(4, static_cast<int>(std::lround(nViews * 0.05)));
    std::vector<int> rejectIdx(order.begin(), order.begin() + std::min(K, nViews - 1));
    std::vector<int> rejectedOrig;
    for (int v : rejectIdx) rejectedOrig.push_back(RA.validOrigIdx[static_cast<size_t>(v)]);

    PipelineOptions optB;
    optB.cornerThreads = 1;
    optB.rejectValidIdx = rejectIdx;
    PipelineResults RB;
    ASSERT_TRUE(runPipeline(in, optB, RB));
    ASSERT_EQ(static_cast<int>(RB.lpts.size()), nViews - static_cast<int>(rejectIdx.size()));

    AggMetrics MA = computeMetrics(RA, in.config);
    AggMetrics MB = computeMetrics(RB, in.config);

    AggMetrics GA = computeMetrics(RA.intrin, RA.extrin, RA.rectify,
                                   RA.lpts, RA.rpts, in.config);
    AggMetrics GB = computeMetrics(RB.intrin, RB.extrin, RB.rectify,
                                   RA.lpts, RA.rpts, in.config);

    EXPECT_LT(GB.epiRawMean, GA.epiRawMean * 1.05)
        << "epi all-frames: base=" << GA.epiRawMean << " rejected=" << GB.epiRawMean;
    EXPECT_LT(GB.rowAlignMean, GA.rowAlignMean * 1.05)
        << "rowAlign all-frames: base=" << GA.rowAlignMean
        << " rejected=" << GB.rowAlignMean;

    std::map<std::string, std::string> gates = loadMarkedGates(
        fs::path(datasetDir()) / "marked_gates.csv");
    int overlap = 0;
    nlohmann::json rejArr = nlohmann::json::array();
    for (int oi : rejectedOrig) {
        std::string stem = fs::path(in.frames[static_cast<size_t>(oi)].leftFile)
                               .stem().string();
        auto it = gates.find(stem);
        const bool marked = it != gates.end() && it->second != "-";
        if (marked) ++overlap;
        rejArr.push_back({{"origIndex", oi}, {"stem", stem},
                          {"score", score[static_cast<size_t>(
                              static_cast<int>(
                                  std::find(RA.validOrigIdx.begin(),
                                            RA.validOrigIdx.end(), oi)
                                  - RA.validOrigIdx.begin()))]},
                          {"markedGate", marked}});
    }

    spdlog::info("opt_bench A1 frame reject: views={} K={} keptRmsL={:.4f}->{:.4f} "
                 "keptRmsR={:.4f}->{:.4f} epiAll={:.4f}->{:.4f} rowAll={:.4f}->{:.4f} "
                 "gatesOverlap={}/{}",
                 nViews, rejectIdx.size(), MA.rmsL, MB.rmsL, MA.rmsR, MB.rmsR,
                 GA.epiRawMean, GB.epiRawMean, GA.rowAlignMean, GB.rowAlignMean,
                 overlap, rejectIdx.size());

    nlohmann::json rep;
    rep["experiment"] = "A1_frame_reject";
    rep["views"] = nViews;
    rep["K"] = rejectIdx.size();
    rep["keptMetrics"] = metricsJson(MB);
    rep["allFrameMetrics"]["base"] = {{"epiRawMean", GA.epiRawMean},
                                      {"rowAlignMean", GA.rowAlignMean}};
    rep["allFrameMetrics"]["rejected"] = {{"epiRawMean", GB.epiRawMean},
                                          {"rowAlignMean", GB.rowAlignMean}};
    rep["rejectedFrames"] = rejArr;
    rep["gatesOverlap"] = overlap;
    const bool pass = GB.epiRawMean < GA.epiRawMean * 1.05
        && GB.rowAlignMean < GA.rowAlignMean * 1.05;
    rep["verdict"] = pass ? "pass" : "FAIL";
    writeTextFile(fs::path(outDir()) / "exp_frame_reject_report.json", rep.dump(2));
}
