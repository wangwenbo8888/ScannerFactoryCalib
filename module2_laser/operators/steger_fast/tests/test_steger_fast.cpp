/**
 * @file test_steger_fast.cpp
 * @brief Steger激光中心亚像素提取算子（优化版）- 等价性 + 基准测试
 *
 * 等价性：与原版 StegerExtractorCUDA 在相同输入下逐位比对点坐标（label 内点序
 * 原本就由原子时序决定、非确定，故按 (x,y) 排序后比对）。
 * 基准：2048x1536 全分辨率，原版为基准，优化版门槛 Total 中位 ≤ 3.0ms。
 *   - 合成 25 线：ByLabel 模式（int 标签 1..25，贴近激光标定链路）
 *   - 真实样本（data_in/left_skew）：Flat 模式（阈值掩膜）
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "steger_fast.h"
#include "steger_extract_cuda.h"   // 参照算子（等价性对拍）——原为经 steger_fast.h 传递引入，抽公共类型头后改显式

using namespace calib;

namespace {

void DrawGaussianLine(cv::Mat& gray, int y0, int x0, int x1,
                      cv::Mat* intLabels, int intLabel,
                      cv::Mat* binMask) {
    for (int x = x0; x < x1; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            int y = y0 + dy;
            if (y < 0 || y >= gray.rows) continue;
            float dist = std::abs(static_cast<float>(dy));
            gray.at<uchar>(y, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            if (intLabels) intLabels->at<int>(y, x) = intLabel;
            if (binMask) binMask->at<uchar>(y, x) = 255;
        }
    }
}

void AssertSamePoints(const StegerResult& ref, const StegerResult& fast) {
    ASSERT_EQ(ref.success, fast.success);
    if (!ref.success) return;
    ASSERT_EQ(ref.lineCount, fast.lineCount);
    ASSERT_EQ(ref.totalPointCount, fast.totalPointCount);
    ASSERT_EQ(ref.centerPoints.size(), fast.centerPoints.size());

    auto sorted = [](std::vector<cv::Point2f> v) {
        std::sort(v.begin(), v.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });
        return v;
    };

    for (const auto& [lb, refPts] : ref.centerPoints) {
        ASSERT_TRUE(fast.centerPoints.count(lb) > 0) << "label " << lb << " missing in fast";
        auto fastPts = fast.centerPoints.at(lb);
        ASSERT_EQ(refPts.size(), fastPts.size());
        auto a = sorted(refPts);
        auto b = sorted(fastPts);
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_FLOAT_EQ(a[i].x, b[i].x) << "label " << lb << " pt " << i;
            EXPECT_FLOAT_EQ(a[i].y, b[i].y) << "label " << lb << " pt " << i;
        }
    }
}

double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename T>
double TimeExecuteMs(T& op, const cv::cuda::GpuMat& gray, const cv::cuda::GpuMat& mask, bool flat) {
    auto t0 = std::chrono::steady_clock::now();
    StegerResult r = flat ? op.Execute(gray, mask, GroupMode::Flat) : op.Execute(gray, mask);
    auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(r.success);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::string FindRealSample() {
    const std::string base = "E:/JEAMMWARE260705/factory_calib/data_in/left_skew";
    std::error_code ec;
    if (!std::filesystem::exists(base, ec)) return "";
    for (const auto& entry : std::filesystem::recursive_directory_iterator(base, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.rfind("L_", 0) == 0 && entry.path().extension() == ".png") {
            return entry.path().string();
        }
    }
    return "";
}

void RunBench(const cv::Mat& gray, const cv::Mat& mask, bool flat, const char* tag) {
    const int kWarm = 5;
    const int kIters = 30;

    StegerParams params;
    StegerExtractorCUDA ref(params);
    StegerExtractorFast fast(params);
    ref.Warmup(gray.rows, gray.cols);
    fast.Warmup(gray.rows, gray.cols);

    cv::cuda::GpuMat d_gray, d_mask;
    d_gray.upload(gray);
    d_mask.upload(mask);

    std::vector<double> t_ref, t_fast;
    t_ref.reserve(kIters);
    t_fast.reserve(kIters);
    for (int i = 0; i < kWarm; ++i) {
        if (flat) {
            ref.Execute(d_gray, d_mask, GroupMode::Flat);
            fast.Execute(d_gray, d_mask, GroupMode::Flat);
        } else {
            ref.Execute(d_gray, d_mask);
            fast.Execute(d_gray, d_mask);
        }
    }
    for (int i = 0; i < kIters; ++i) {
        t_fast.push_back(TimeExecuteMs(fast, d_gray, d_mask, flat));
        t_ref.push_back(TimeExecuteMs(ref, d_gray, d_mask, flat));
    }

    double med_ref = Median(t_ref);
    double med_fast = Median(t_fast);
    double min_fast = *std::min_element(t_fast.begin(), t_fast.end());
    double max_fast = *std::max_element(t_fast.begin(), t_fast.end());

    std::cout << "[BENCH-SUMMARY " << tag << "] " << gray.cols << "x" << gray.rows
              << " | baseline(original) median=" << med_ref << "ms"
              << " | fast median=" << med_fast << "ms (min=" << min_fast
              << " max=" << max_fast << ") | speedup=" << (med_ref / med_fast) << "x\n";

    EXPECT_LT(med_fast, med_ref) << "fast should beat baseline";
    EXPECT_LE(med_fast, 3.0) << "target: fast median total <= 3ms";
}

} // namespace

class StegerFastTest : public ::testing::Test {
protected:
    void SetUp() override { params_ = StegerParams{}; }

    StegerParams params_;
};

TEST_F(StegerFastTest, EquivalenceByLabelSingleLine) {
    StegerExtractorCUDA ref(params_);
    StegerExtractorFast fast(params_);
    ref.Warmup(100, 200);
    fast.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(100, 200, CV_32SC1);
    DrawGaussianLine(gray, 50, 20, 180, &labels, 1, nullptr);

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto r_ref = ref.Execute(d_gray, d_labels);
    auto r_fast = fast.Execute(d_gray, d_labels);

    ASSERT_TRUE(r_ref.success);
    AssertSamePoints(r_ref, r_fast);
}

TEST_F(StegerFastTest, EquivalenceByLabelMultipleLines) {
    StegerExtractorCUDA ref(params_);
    StegerExtractorFast fast(params_);
    ref.Warmup(200, 200);
    fast.Warmup(200, 200);

    cv::Mat gray = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(200, 200, CV_32SC1);
    DrawGaussianLine(gray, 50, 20, 180, &labels, 3, nullptr);
    DrawGaussianLine(gray, 150, 20, 180, &labels, 7, nullptr);
    DrawGaussianLine(gray, 100, 20, 180, &labels, 11, nullptr);

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto r_ref = ref.Execute(d_gray, d_labels);
    auto r_fast = fast.Execute(d_gray, d_labels);

    ASSERT_TRUE(r_ref.success);
    AssertSamePoints(r_ref, r_fast);
    EXPECT_EQ(r_fast.lineCount, 3);
}

TEST_F(StegerFastTest, EquivalenceFlat) {
    StegerExtractorCUDA ref(params_);
    StegerExtractorFast fast(params_);
    ref.Warmup(100, 200);
    fast.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat binMask = cv::Mat::zeros(100, 200, CV_8UC1);
    DrawGaussianLine(gray, 50, 20, 180, nullptr, 0, &binMask);

    cv::cuda::GpuMat d_gray, d_mask;
    d_gray.upload(gray);
    d_mask.upload(binMask);

    auto r_ref = ref.Execute(d_gray, d_mask, GroupMode::Flat);
    auto r_fast = fast.Execute(d_gray, d_mask, GroupMode::Flat);

    ASSERT_TRUE(r_ref.success);
    AssertSamePoints(r_ref, r_fast);
}

TEST_F(StegerFastTest, EquivalenceAllBackground) {
    StegerExtractorCUDA ref(params_);
    StegerExtractorFast fast(params_);
    ref.Warmup(100, 100);
    fast.Warmup(100, 100);

    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_labels(100, 100, CV_32SC1, cv::Scalar(0));

    auto r_ref = ref.Execute(d_gray, d_labels);
    auto r_fast = fast.Execute(d_gray, d_labels);

    ASSERT_TRUE(r_ref.success);
    ASSERT_TRUE(r_fast.success);
    EXPECT_EQ(r_fast.totalPointCount, 0);
}

TEST_F(StegerFastTest, EquivalenceNoWarmup) {
    StegerExtractorCUDA ref(params_);
    StegerExtractorFast fast(params_);

    cv::Mat gray = cv::Mat::zeros(50, 100, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(50, 100, CV_32SC1);
    DrawGaussianLine(gray, 25, 10, 90, &labels, 1, nullptr);

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto r_ref = ref.Execute(d_gray, d_labels);
    auto r_fast = fast.Execute(d_gray, d_labels);

    ASSERT_TRUE(r_ref.success);
    AssertSamePoints(r_ref, r_fast);
}

TEST_F(StegerFastTest, EquivalenceAfterSetParams) {
    StegerParams p;
    p.sigma = 2.5f;
    p.kernelSize = 7;
    StegerExtractorCUDA ref(p);
    StegerExtractorFast fast(p);
    ref.Warmup(100, 200);
    fast.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(100, 200, CV_32SC1);
    DrawGaussianLine(gray, 50, 20, 180, &labels, 5, nullptr);

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    AssertSamePoints(ref.Execute(d_gray, d_labels), fast.Execute(d_gray, d_labels));
}

TEST_F(StegerFastTest, BenchFullResSynthetic) {
    const int rows = 1536, cols = 2048;
    cv::Mat gray = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(rows, cols, CV_32SC1);
    for (int i = 0; i < 25; ++i) {
        DrawGaussianLine(gray, 60 + i * 60, 50, cols - 50, &labels, i + 1, nullptr);
    }
    RunBench(gray, labels, false, "synthetic-25lines");
}

TEST_F(StegerFastTest, BenchRealImage) {
    std::string path = FindRealSample();
    if (path.empty()) GTEST_SKIP() << "no real sample found";

    cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty()) << "failed to read " << path;
    cv::Mat binMask;
    cv::threshold(gray, binMask, 20, 255, cv::THRESH_BINARY);
    RunBench(gray, binMask, true, ("real:" + std::filesystem::path(path).filename().string()).c_str());
}

TEST_F(StegerFastTest, VisualizeRealFrames) {
    const std::string base = "E:/JEAMMWARE260705/factory_calib/data_in/left_skew";
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& e : std::filesystem::recursive_directory_iterator(base, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        std::string name = e.path().filename().string();
        if (name.rfind("L_", 0) == 0 && e.path().extension() == ".png") {
            files.push_back(e.path().string());
        }
    }
    ASSERT_FALSE(files.empty()) << "no L_*.png under " << base;
    std::sort(files.begin(), files.end());
    if (files.size() > 24) files.resize(24);

    const std::string outdir = "C:/Users/DUUMM/AppData/Local/Temp/opencode/steger_fast_vis";
    std::filesystem::create_directories(outdir);

    StegerParams params;
    StegerExtractorCUDA ref(params);
    StegerExtractorFast fast(params);

    std::ofstream csv(outdir + "/timing.csv");
    csv << "frame,file,points,orig_ms,fast_ms,speedup\n";

    auto drawOverlay = [](const cv::Mat& gray, const StegerResult& r) {
        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        for (const auto& [lb, pts] : r.centerPoints) {
            float hue = std::fmod(lb * 14.0f, 180.0f);
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(cvRound(hue), 255, 255));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            cv::Scalar c = bgr.at<cv::Vec3b>(0, 0);
            for (const auto& p : pts) {
                int x = cvRound(p.x), y = cvRound(p.y);
                if (x < 1 || y < 1 || x >= vis.cols - 1 || y >= vis.rows - 1) continue;
                cv::rectangle(vis, cv::Rect(x - 1, y - 1, 3, 3), c, cv::FILLED);
            }
        }
        return vis;
    };

    double sum_orig = 0, sum_fast = 0;
    std::vector<cv::Mat> montageFrames;

    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat gray = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(gray.empty()) << "failed to read " << files[i];
        cv::Mat binMask;
        cv::threshold(gray, binMask, 20, 255, cv::THRESH_BINARY);

        if (i == 0) {
            ref.Warmup(gray.rows, gray.cols);
            fast.Warmup(gray.rows, gray.cols);
        }

        cv::cuda::GpuMat d_gray, d_mask;
        d_gray.upload(gray);
        d_mask.upload(binMask);

        const int kReps = 3;
        std::vector<double> to, tf;
        to.reserve(kReps);
        tf.reserve(kReps);
        StegerResult rr, rf;
        for (int k = 0; k < kReps; ++k) {
            auto t0 = std::chrono::steady_clock::now();
            rr = ref.Execute(d_gray, d_mask, GroupMode::Flat);
            auto t1 = std::chrono::steady_clock::now();
            to.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        for (int k = 0; k < kReps; ++k) {
            auto t0 = std::chrono::steady_clock::now();
            rf = fast.Execute(d_gray, d_mask, GroupMode::Flat);
            auto t1 = std::chrono::steady_clock::now();
            tf.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        ASSERT_TRUE(rr.success);
        ASSERT_TRUE(rf.success);
        AssertSamePoints(rr, rf);

        double mo = Median(to);
        double mf = Median(tf);
        sum_orig += mo;
        sum_fast += mf;

        std::string stem = std::filesystem::path(files[i]).stem().string();
        std::string idx = std::to_string(i);
        if (idx.size() < 2) idx = "0" + idx;
        cv::imwrite(outdir + "/frame_" + idx + "_" + stem + "_orig.png", drawOverlay(gray, rr));
        cv::imwrite(outdir + "/frame_" + idx + "_" + stem + "_fast.png", drawOverlay(gray, rf));

        std::cout << "[FRAME] " << idx << " " << stem
                  << " pts=" << rr.totalPointCount
                  << " orig=" << mo << "ms fast=" << mf << "ms"
                  << " speedup=" << (mo / mf) << "x\n";
        csv << idx << "," << stem << "," << rr.totalPointCount << ","
            << mo << "," << mf << "," << (mo / mf) << "\n";

        if (montageFrames.size() < 6) {
            cv::Mat small;
            cv::resize(drawOverlay(gray, rf), small, cv::Size(680, 510));
            montageFrames.push_back(small);
        }
    }

    if (!montageFrames.empty()) {
        int cols = 3;
        int rows = static_cast<int>((montageFrames.size() + cols - 1) / cols);
        cv::Mat montage(rows * 510, cols * 680, CV_8UC3, cv::Scalar(30, 30, 30));
        for (size_t k = 0; k < montageFrames.size(); ++k) {
            int r = static_cast<int>(k) / cols;
            int c = static_cast<int>(k) % cols;
            montageFrames[k].copyTo(montage(cv::Rect(c * 680, r * 510, 680, 510)));
        }
        cv::imwrite(outdir + "/overview_fast_first6.png", montage);
    }

    std::cout << "[FRAME-SUMMARY] frames=" << files.size()
              << " orig_mean=" << (sum_orig / files.size()) << "ms"
              << " fast_mean=" << (sum_fast / files.size()) << "ms"
              << " speedup=" << (sum_orig / sum_fast) << "x"
              << " | images at " << outdir << "\n";
    EXPECT_LE(sum_fast / files.size(), 3.0);
}
