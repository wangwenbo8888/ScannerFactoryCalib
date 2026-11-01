#include <gtest/gtest.h>
#include "bench_utils.h"

#include <opencv2/imgproc.hpp>
#include <atomic>

using namespace fcbench;

namespace {

bool extractFast(const cv::Mat& gray,
                 const fc::ChessboardCornerParams& params,
                 fc::ChessboardCornerResult& result) {
    result.found = false;
    result.corners.clear();
    result.meanSubpixDelta = 0.0;
    if (gray.empty()) return false;
    cv::Mat gray8;
    if (gray.type() != CV_8U) gray.convertTo(gray8, CV_8U);
    else gray8 = gray;

    const int fastFlags = cv::CALIB_CB_NORMALIZE_IMAGE;
    std::vector<cv::Point2f> coarse;
    bool found = cv::findChessboardCornersSB(gray8, params.patternSize, coarse, fastFlags);
    if (!found || static_cast<int>(coarse.size()) != params.patternSize.area()) return false;

    std::vector<cv::Point2f> before = coarse;
    cv::cornerSubPix(gray8, coarse, params.subpixWin, params.subpixZeroZone, params.subpixTerm);
    double acc = 0.0;
    for (size_t i = 0; i < coarse.size(); ++i)
        acc += std::hypot(coarse[i].x - before[i].x, coarse[i].y - before[i].y);
    result.meanSubpixDelta = acc / coarse.size();
    result.corners = std::move(coarse);
    result.found = true;
    return true;
}

} // namespace

TEST(OptSbTwoStage, NotWorseThanExhaustive) {
    if (!datasetAvailable()) GTEST_SKIP() << "dataset camera261006-1688 absent";

    auto inOpt = loadDataset();
    ASSERT_TRUE(inOpt.has_value());
    const fc::CameraInput& in = *inOpt;
    ASSERT_GT(in.frames.size(), 0u);

    fc::ChessboardCornerParams cp;
    cp.patternSize = cv::Size(in.config.chessWidth, in.config.chessHeight);

    std::atomic<int> fastHits{0};
    std::atomic<int> fastCalls{0};
    CornerFn twoStage = [&](const cv::Mat& g, const fc::ChessboardCornerParams& p,
                            fc::ChessboardCornerResult& r) -> bool {
        fastCalls.fetch_add(1);
        if (extractFast(g, p, r)) {
            fastHits.fetch_add(1);
            return true;
        }
        return fc::extractChessboardCorners(g, p, r);
    };

    std::vector<CornerFrameResult> base, two;
    Stopwatch sw;
    extractAllCorners(in, cp, nullptr, 1, base);
    const double baseMs = sw.ms();
    sw.reset();
    extractAllCorners(in, cp, twoStage, 1, two);
    const double twoMs = sw.ms();

    ASSERT_EQ(base.size(), two.size());

    size_t validBase = 0, validTwo = 0, setMismatch = 0, countMismatch = 0;
    double maxDelta = 0.0, sumDelta = 0.0;
    size_t deltaPoints = 0;
    for (size_t i = 0; i < base.size(); ++i) {
        if (base[i].valid) ++validBase;
        if (two[i].valid) ++validTwo;
        if (base[i].valid != two[i].valid) {
            ++setMismatch;
            continue;
        }
        if (!base[i].valid) continue;
        if (base[i].l.size() != two[i].l.size()
            || base[i].r.size() != two[i].r.size()) {
            ++countMismatch;
            continue;
        }
        for (size_t j = 0; j < base[i].l.size(); ++j) {
            const double dl = std::hypot(base[i].l[j].x - two[i].l[j].x,
                                         base[i].l[j].y - two[i].l[j].y);
            const double dr = std::hypot(base[i].r[j].x - two[i].r[j].x,
                                         base[i].r[j].y - two[i].r[j].y);
            const double d = std::max(dl, dr);
            if (d > 0.0) ++deltaPoints;
            maxDelta = std::max(maxDelta, d);
            sumDelta += d;
        }
    }

    EXPECT_EQ(setMismatch, 0u);
    EXPECT_EQ(countMismatch, 0u);
    EXPECT_EQ(validBase, validTwo);
    EXPECT_GT(validBase, 0u);

    std::vector<std::vector<cv::Point2f>> lBase, rBase, lTwo, rTwo;
    for (size_t i = 0; i < base.size(); ++i) {
        if (!base[i].valid) continue;
        lBase.push_back(base[i].l);
        rBase.push_back(base[i].r);
        lTwo.push_back(two[i].l);
        rTwo.push_back(two[i].r);
    }

    calib::IntrinsicCalibParams ip = makeIntrinsicParams(in.config);
    calib::IntrinsicCalibCPU calBase(ip);
    calib::IntrinsicCalibResult resBase;
    ASSERT_TRUE(calBase.Execute(lBase, rBase, resBase) && resBase.success);
    calib::IntrinsicCalibCPU calTwo(ip);
    calib::IntrinsicCalibResult resTwo;
    ASSERT_TRUE(calTwo.Execute(lTwo, rTwo, resTwo) && resTwo.success);

    const double rmsLd = relDiff(resTwo.left.rms_error, resBase.left.rms_error);
    const double rmsRd = relDiff(resTwo.right.rms_error, resBase.right.rms_error);
    EXPECT_LT(rmsLd, 0.02) << "rmsL two=" << resTwo.left.rms_error
                           << " base=" << resBase.left.rms_error;
    EXPECT_LT(rmsRd, 0.02) << "rmsR two=" << resTwo.right.rms_error
                           << " base=" << resBase.right.rms_error;

    const double coverage = fastCalls > 0
        ? static_cast<double>(fastHits.load()) / fastCalls.load() : 0.0;
    const double meanDelta = deltaPoints > 0 ? sumDelta / deltaPoints : 0.0;
    const double speedup = twoMs > 0.0 ? baseMs / twoMs : 0.0;

    spdlog::info("opt_bench S2 two-stage: valid={}/{} fastCoverage={:.1f}% "
                 "maxCornerDelta={:.4f}px meanCornerDelta={:.4f}px "
                 "baseMs={:.0f} twoMs={:.0f} speedup={:.2f}x rmsLrel={:.4f} rmsRrel={:.4f}",
                 validTwo, base.size(), coverage * 100.0, maxDelta, meanDelta,
                 baseMs, twoMs, speedup, rmsLd, rmsRd);

    nlohmann::json rep;
    rep["experiment"] = "S2_sb_two_stage";
    rep["validFrames"] = validTwo;
    rep["setMismatch"] = setMismatch;
    rep["fastCoverage"] = coverage;
    rep["maxCornerDeltaPx"] = maxDelta;
    rep["meanCornerDeltaPx"] = meanDelta;
    rep["baseMs"] = baseMs;
    rep["twoStageMs"] = twoMs;
    rep["speedup"] = speedup;
    rep["rmsL"]["base"] = resBase.left.rms_error;
    rep["rmsL"]["two"] = resTwo.left.rms_error;
    rep["rmsR"]["base"] = resBase.right.rms_error;
    rep["rmsR"]["two"] = resTwo.right.rms_error;
    rep["verdict"] = (setMismatch == 0 && rmsLd < 0.02 && rmsRd < 0.02)
        ? "pass" : "FAIL";
    writeTextFile(fs::path(outDir()) / "exp_sb_two_stage_report.json", rep.dump(2));
}
