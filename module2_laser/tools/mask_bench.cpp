// mask_bench.cpp — mask_extract 单算子分步计时 + 优化对照基准
// 输出: <outDir>/timing.csv + 每图掩膜对照 PNG (base/approx/diff)
// 计时: CUDA 事件; 精确路径(e3d19p13 椭圆核) vs 近似路径(线核两趟) + 整算子对照
#include "calib_io.h"
#include "mask_extract_cuda.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace fc;
using namespace calib;

namespace {

// 步骤索引
enum {
    S_UPLOAD = 0, S_THR, S_ERODE1, S_DIL19, S_POST13,
    S_DIL_H, S_DIL_V, S_POST_H, S_POST_V,
    S_COPYPOOL, S_EXEC_EXACT, S_EXEC_APPROX,
    S_COUNT
};
const char* kNames[S_COUNT] = {
    "upload(H2D)", "threshold", "erode1x1(old)", "dilate19(ellipse)", "post13(ellipse)",
    "dilateH(19)", "dilateV(19)", "postH(13)", "postV(13)",
    "poolCopy(3x)", "Execute(exact)", "Execute(approx)"
};

} // namespace

int runMaskBench(const std::string& inDir, const std::string& outDir, int iters) {
    auto input = loadLaserInput(inDir);
    if (!input) { spdlog::error("load input failed"); return 1; }
    std::filesystem::create_directories(outDir);

    MaskExtractParams mp;
    mp.threshold = 50;
    mp.erodeSize = 1;
    mp.laserDilateSize = 19;
    mp.postErodeSize = 13;
    MaskExtractParams mpA = mp;
    mpA.morphApprox = 1;

    cv::cuda::Stream stream;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    cudaEvent_t ev[2];
    cudaEventCreate(&ev[0]);
    cudaEventCreate(&ev[1]);

    MaskExtractCUDA opExact(mp);
    MaskExtractCUDA opApprox(mpA);
    opExact.Warmup(1536, 2048);
    opApprox.Warmup(1536, 2048);

    // 复刻缓冲 (分步计时用)
    auto mkLine = [](int morph, int w, int h) {
        return cv::cuda::createMorphologyFilter(
            morph, CV_8UC1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h)));
    };
    auto fErode1 = cv::cuda::createMorphologyFilter(
        cv::MORPH_ERODE, CV_8UC1,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1)));
    auto fDil19 = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8UC1,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19)));
    auto fPost13 = cv::cuda::createMorphologyFilter(
        cv::MORPH_ERODE, CV_8UC1,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));
    auto fDilH = mkLine(cv::MORPH_DILATE, 19, 1), fDilV = mkLine(cv::MORPH_DILATE, 1, 19);
    auto fPoH = mkLine(cv::MORPH_ERODE, 13, 1), fPoV = mkLine(cv::MORPH_ERODE, 1, 13);

    cv::cuda::GpuMat dIn, dTh, dEr, dDi, dPo, dTmp, dP0, dP1, dP2;
    for (auto* m : {&dIn, &dTh, &dEr, &dDi, &dPo, &dTmp, &dP0, &dP1, &dP2})
        cv::cuda::createContinuous(1536, 2048, CV_8UC1, *m);

    double acc[S_COUNT] = {0};
    long accCalls = 0;

    std::ofstream csv(outDir + "/timing.csv");
    csv << "image";
    for (int i = 0; i < S_COUNT; ++i) csv << ',' << kNames[i];
    csv << ",diff_px,diff_pct\n";

    auto stepMs = [&](int k, double& row) {
        cudaEventRecord(ev[1], cs);
        cudaEventSynchronize(ev[1]);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev[0], ev[1]);
        acc[k] += ms;
        row += ms;
    };

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        for (size_t ti = 0; ti < input->poseFrames[pi].size(); ++ti) {
            const auto& frame = input->poseFrames[pi][ti];
            const std::string stem =
                input->poseDirs[pi] + "_tube" + std::to_string(ti);
            const cv::Mat imgs[2] = {frame.leftGray, frame.rightGray};
            const char* side[2] = {"L", "R"};

            for (int s = 0; s < 2; ++s) {
                const std::string name = stem + "_" + side[s];
                for (int w = 0; w < 3; ++w) {
                    dIn.upload(imgs[s], stream);
                    cv::cuda::threshold(dIn, dTh, 50.0, 255.0, cv::THRESH_BINARY, stream);
                    fErode1->apply(dTh, dEr, stream);
                    fDil19->apply(dEr, dDi, stream);
                    fPost13->apply(dDi, dPo, stream);
                    opExact.Execute(imgs[s], stream);
                    opApprox.Execute(imgs[s], stream);
                    stream.waitForCompletion();
                }

                double row[S_COUNT] = {0};
                for (int it = 0; it < iters; ++it) {
                    cudaEventRecord(ev[0], cs);
                    dIn.upload(imgs[s], stream);
                    stepMs(S_UPLOAD, row[S_UPLOAD]);

                    cudaEventRecord(ev[0], cs);
                    cv::cuda::threshold(dIn, dTh, 50.0, 255.0, cv::THRESH_BINARY, stream);
                    stepMs(S_THR, row[S_THR]);

                    // 旧精确序列 (含恒等腐蚀)
                    cudaEventRecord(ev[0], cs);
                    fErode1->apply(dTh, dEr, stream);
                    stepMs(S_ERODE1, row[S_ERODE1]);
                    cudaEventRecord(ev[0], cs);
                    fDil19->apply(dTh, dDi, stream);
                    stepMs(S_DIL19, row[S_DIL19]);
                    cudaEventRecord(ev[0], cs);
                    fPost13->apply(dDi, dPo, stream);
                    stepMs(S_POST13, row[S_POST13]);

                    // 新近似序列
                    cudaEventRecord(ev[0], cs);
                    fDilH->apply(dTh, dTmp, stream);
                    stepMs(S_DIL_H, row[S_DIL_H]);
                    cudaEventRecord(ev[0], cs);
                    fDilV->apply(dTmp, dDi, stream);
                    stepMs(S_DIL_V, row[S_DIL_V]);
                    cudaEventRecord(ev[0], cs);
                    fPoH->apply(dDi, dTmp, stream);
                    stepMs(S_POST_H, row[S_POST_H]);
                    cudaEventRecord(ev[0], cs);
                    fPoV->apply(dTmp, dPo, stream);
                    stepMs(S_POST_V, row[S_POST_V]);

                    // 整算子
                    cudaEventRecord(ev[0], cs);
                    auto rE = opExact.Execute(imgs[s], stream);
                    stepMs(S_EXEC_EXACT, row[S_EXEC_EXACT]);
                    cudaEventRecord(ev[0], cs);
                    auto rA = opApprox.Execute(imgs[s], stream);
                    stepMs(S_EXEC_APPROX, row[S_EXEC_APPROX]);
                    stream.waitForCompletion();
                    if (!rE.success || !rA.success)
                        spdlog::warn("{}: Execute fail", name);
                }

                // 掩膜差异 (精确 vs 近似) + PNG
                cv::Mat mBase, mApx;
                {
                    auto rE = opExact.Execute(imgs[s], stream);
                    auto rA = opApprox.Execute(imgs[s], stream);
                    stream.waitForCompletion();
                    rE.d_cleanedMask->download(mBase);
                    rA.d_cleanedMask->download(mApx);
                }
                cv::Mat diff;
                cv::absdiff(mBase, mApx, diff);
                const int diffPx = cv::countNonZero(diff);
                const int totalPx = mBase.rows * mBase.cols;
                const double diffPct = 100.0 * diffPx / totalPx;

                cv::imwrite(outDir + "/" + name + "_mask_base.png", mBase);
                cv::imwrite(outDir + "/" + name + "_mask_approx.png", mApx);
                cv::imwrite(outDir + "/" + name + "_mask_diff.png", diff);

                csv << name;
                for (int k = 0; k < S_COUNT; ++k) csv << ',' << std::fixed
                                                      << std::setprecision(3)
                                                      << row[k] / iters;
                csv << ',' << diffPx << ',' << std::setprecision(4) << diffPct << '\n';
                ++accCalls;
                spdlog::info("{}: old(thr+er+dil+post)={:.2f} new(thr+dilHV+postHV)={:.2f} "
                             "| ExecExact={:.2f} ExecApprox={:.2f} | diff={}/{} ({:.3}%)",
                             name,
                             row[S_THR]+row[S_ERODE1]+row[S_DIL19]+row[S_POST13],
                             row[S_THR]+row[S_DIL_H]+row[S_DIL_V]+row[S_POST_H]+row[S_POST_V],
                             row[S_EXEC_EXACT], row[S_EXEC_APPROX],
                             diffPx, totalPx, diffPct);
            }
        }
    }

    spdlog::info("=== avg over {} images x {} iters (ms) ===", accCalls, iters);
    for (int k = 0; k < S_COUNT; ++k)
        spdlog::info("  {:<18} {:>8.3f}", kNames[k], acc[k] / (accCalls * iters));
    const double d = accCalls * iters;
    spdlog::info("  形态学旧序列(thr+er+dil+post)={:.3f}  新序列(thr+dilHV+postHV)={:.3f}  加速 {:.1f}x",
                 (acc[S_THR]+acc[S_ERODE1]+acc[S_DIL19]+acc[S_POST13]) / d,
                 (acc[S_THR]+acc[S_DIL_H]+acc[S_DIL_V]+acc[S_POST_H]+acc[S_POST_V]) / d,
                 (acc[S_ERODE1]+acc[S_DIL19]+acc[S_POST13]) /
                 std::max(1e-9, (acc[S_DIL_H]+acc[S_DIL_V]+acc[S_POST_H]+acc[S_POST_V])));
    spdlog::info("  整算子: exact={:.3f} approx={:.3f} 加速 {:.1f}x",
                 acc[S_EXEC_EXACT] / d, acc[S_EXEC_APPROX] / d,
                 acc[S_EXEC_EXACT] / std::max(1e-9, acc[S_EXEC_APPROX]));
    return 0;
}
