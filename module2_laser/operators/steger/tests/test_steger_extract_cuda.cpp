/**
 * @file test_steger_extract_cuda.cpp
 * @brief Steger激光中心亚像素提取算子 - 单元测试
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <map>

#include "../steger_extract_cuda.h"
#include "common/calib_warmup_config.h"

using namespace calib;


// ============================================================================
// 测试夹具
// ============================================================================
class StegerExtractTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = StegerParams{};
    }

    StegerParams params_;
};

// ============================================================================
// 参数校验测试
// ============================================================================
TEST_F(StegerExtractTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_FLOAT_EQ(params_.sigma, 1.5f);
    EXPECT_EQ(params_.kernelSize, 0);
    EXPECT_FLOAT_EQ(params_.lowThreshold, 2.0f);
    EXPECT_EQ(params_.maxLabels, 256);
    EXPECT_EQ(params_.deviceId, 0);
}

TEST_F(StegerExtractTest, SigmaTooSmallThrows) {
    params_.sigma = 0.1f;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(StegerExtractTest, SigmaTooLargeThrows) {
    params_.sigma = 15.0f;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(StegerExtractTest, InvalidKernelSizeThrows) {
    params_.kernelSize = 4;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(StegerExtractTest, ValidKernelSizes) {
    for (int ks : {0, 3, 5, 7, 9}) {
        params_.kernelSize = ks;
        EXPECT_NO_THROW(params_.validate());
    }
}

TEST_F(StegerExtractTest, NegativeLowThresholdThrows) {
    params_.lowThreshold = -1.0f;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(StegerExtractTest, MaxLabelsOutOfRangeThrows) {
    params_.maxLabels = 0;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_.maxLabels = 5000;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(StegerExtractTest, NegativeDeviceIdThrows) {
    params_.deviceId = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(StegerExtractTest, ValidBoundaryParams) {
    params_.sigma = 0.5f;
    EXPECT_NO_THROW(params_.validate());
    params_.sigma = 10.0f;
    EXPECT_NO_THROW(params_.validate());
    params_.maxLabels = 1;
    EXPECT_NO_THROW(params_.validate());
    params_.maxLabels = 4096;
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================================
// JSON 序列化测试
// ============================================================================
TEST_F(StegerExtractTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = StegerParams::fromJson(j);

    EXPECT_FLOAT_EQ(restored.sigma, params_.sigma);
    EXPECT_EQ(restored.kernelSize, params_.kernelSize);
    EXPECT_FLOAT_EQ(restored.lowThreshold, params_.lowThreshold);
    EXPECT_FLOAT_EQ(restored.highThreshold, params_.highThreshold);
    EXPECT_EQ(restored.maxLabels, params_.maxLabels);
    EXPECT_EQ(restored.deviceId, params_.deviceId);
}

TEST_F(StegerExtractTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"sigma", 2.5f}};
    auto restored = StegerParams::fromJson(j);
    EXPECT_FLOAT_EQ(restored.sigma, 2.5f);
    EXPECT_FLOAT_EQ(restored.lowThreshold, 2.0f);
}

TEST_F(StegerExtractTest, JsonUnknownFieldsIgnored) {
    nlohmann::json j = {{"sigma", 1.0f}, {"unknownField", 999}};
    EXPECT_NO_THROW(StegerParams::fromJson(j));
}

TEST_F(StegerExtractTest, JsonInvalidValueThrows) {
    nlohmann::json j = {{"sigma", 0.01f}};
    EXPECT_THROW(StegerParams::fromJson(j), std::invalid_argument);
}

// ============================================================================
// StegerResult 结构体测试
// ============================================================================
TEST_F(StegerExtractTest, ResultDefaultValues) {
    StegerResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_TRUE(result.centerPoints.empty());
    EXPECT_EQ(result.totalPointCount, 0);
    EXPECT_EQ(result.lineCount, 0);
}

TEST_F(StegerExtractTest, ResultMoveSemantics) {
    StegerResult result1;
    result1.success = true;
    result1.message = "test";
    result1.totalPointCount = 100;
    result1.lineCount = 3;
    result1.centerPoints[1] = {cv::Point2f(1.0f, 2.0f)};

    StegerResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
    EXPECT_EQ(result2.totalPointCount, 100);
    EXPECT_EQ(result2.lineCount, 3);
    EXPECT_EQ(result2.centerPoints.size(), 1u);
}

// ============================================================================
// WarmupConfig 测试
// ============================================================================
TEST_F(StegerExtractTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================================
// CUDA 测试（需要 GPU）
// ============================================================================
#ifdef WITH_CUDA_TESTS

TEST_F(StegerExtractTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        StegerExtractorCUDA extractor(params_);
    });
}

TEST_F(StegerExtractTest, ConstructWithInvalidParamsThrows) {
    StegerParams badParams;
    badParams.sigma = 0.01f;
    EXPECT_THROW({
        StegerExtractorCUDA extractor(badParams);
    }, std::invalid_argument);
}

TEST_F(StegerExtractTest, WarmupBasic) {
    StegerExtractorCUDA extractor(params_);
    EXPECT_NO_THROW(extractor.Warmup(100, 100));
}

TEST_F(StegerExtractTest, WarmupWithConfig) {
    StegerExtractorCUDA extractor(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(extractor.Warmup(config));
}

TEST_F(StegerExtractTest, ExtractEmptyGrayReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_empty;
    cv::cuda::GpuMat d_labels(100, 100, CV_32SC1, cv::Scalar(0));
    auto result = extractor.Execute(d_empty, d_labels);
    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractEmptyLabelsReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_empty;
    auto result = extractor.Execute(d_gray, d_empty);
    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractSizeMismatchReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_labels(200, 200, CV_32SC1, cv::Scalar(0));
    auto result = extractor.Execute(d_gray, d_labels);
    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractWrongGrayTypeReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_gray(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::cuda::GpuMat d_labels(100, 100, CV_32SC1, cv::Scalar(0));
    auto result = extractor.Execute(d_gray, d_labels);
    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractWrongLabelTypeReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_labels(100, 100, CV_8UC1, cv::Scalar(0));
    auto result = extractor.Execute(d_gray, d_labels);
    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractHorizontalLine) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(100, 200, CV_32SC1);

    for (int x = 20; x < 180; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            int y = 50 + dy;
            float dist = std::abs(dy);
            gray.at<uchar>(y, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            labels.at<int>(y, x) = 1;
        }
    }

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto result = extractor.Execute(d_gray, d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.lineCount, 1);
    EXPECT_GT(result.totalPointCount, 0);
    ASSERT_TRUE(result.centerPoints.count(1) > 0);

    const auto& pts = result.centerPoints.at(1);
    for (const auto& pt : pts) {
        EXPECT_NEAR(pt.y, 50.0f, 3.0f) << "Subpixel y should be near 50";
    }
}

TEST_F(StegerExtractTest, ExtractMultipleLines) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(200, 200);

    cv::Mat gray = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(200, 200, CV_32SC1);

    for (int x = 20; x < 180; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            float dist = std::abs(dy);
            uchar val = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            gray.at<uchar>(50 + dy, x) = val;
            labels.at<int>(50 + dy, x) = 1;
            gray.at<uchar>(150 + dy, x) = val;
            labels.at<int>(150 + dy, x) = 2;
        }
    }

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto result = extractor.Execute(d_gray, d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.lineCount, 2);
    EXPECT_TRUE(result.centerPoints.count(1) > 0);
    EXPECT_TRUE(result.centerPoints.count(2) > 0);
}

TEST_F(StegerExtractTest, ExtractAllBackgroundReturnsSuccessWithNoPoints) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(100, 100);

    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_labels(100, 100, CV_32SC1, cv::Scalar(0));

    auto result = extractor.Execute(d_gray, d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.totalPointCount, 0);
}

TEST_F(StegerExtractTest, SetParamsAndGetParams) {
    StegerExtractorCUDA extractor(params_);

    StegerParams newParams;
    newParams.sigma = 2.5f;
    newParams.kernelSize = 7;
    newParams.lowThreshold = 3.0f;

    extractor.SetParams(newParams);
    const auto& current = extractor.GetParams();
    EXPECT_FLOAT_EQ(current.sigma, 2.5f);
    EXPECT_EQ(current.kernelSize, 7);
    EXPECT_FLOAT_EQ(current.lowThreshold, 3.0f);
}

TEST_F(StegerExtractTest, SetInvalidParamsThrows) {
    StegerExtractorCUDA extractor(params_);
    StegerParams badParams;
    badParams.sigma = 100.0f;
    EXPECT_THROW(extractor.SetParams(badParams), std::invalid_argument);
}

TEST_F(StegerExtractTest, ExtractWithoutWarmupStillWorks) {
    StegerExtractorCUDA extractor(params_);

    cv::Mat gray = cv::Mat::zeros(50, 100, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(50, 100, CV_32SC1);

    for (int x = 10; x < 90; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            float dist = std::abs(dy);
            gray.at<uchar>(25 + dy, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            labels.at<int>(25 + dy, x) = 1;
        }
    }

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto result = extractor.Execute(d_gray, d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.lineCount, 1);
}

// ============================================================================
// Flat 模式测试 (GroupMode::Flat - 扫描管线 CV_8UC1 二值掩码)
// ============================================================================
TEST_F(StegerExtractTest, ExtractFlatHorizontalLine) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat binMask = cv::Mat::zeros(100, 200, CV_8UC1);

    for (int x = 20; x < 180; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            int y = 50 + dy;
            float dist = std::abs(dy);
            gray.at<uchar>(y, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            binMask.at<uchar>(y, x) = 255;
        }
    }

    cv::cuda::GpuMat d_gray, d_mask;
    d_gray.upload(gray);
    d_mask.upload(binMask);

    auto result = extractor.Execute(d_gray, d_mask, GroupMode::Flat);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.totalPointCount, 0);
    ASSERT_TRUE(result.d_line_ids != nullptr);
    ASSERT_TRUE(result.d_centerPoints != nullptr);

    cv::Mat h_line_ids;
    result.d_line_ids->download(h_line_ids);
    ASSERT_EQ(h_line_ids.type(), CV_32SC1);
    const int* ids = h_line_ids.ptr<int>();
    for (int i = 0; i < result.totalPointCount; ++i) {
        EXPECT_EQ(ids[i], 1) << "All line_ids must be uniform (1) in Flat mode";
    }

    ASSERT_TRUE(result.centerPoints.count(1) > 0);
    const auto& pts = result.centerPoints.at(1);
    for (const auto& pt : pts) {
        EXPECT_NEAR(pt.y, 50.0f, 3.0f) << "Subpixel y should be near 50";
    }
}

TEST_F(StegerExtractTest, ExtractFlatWithoutWarmupStillWorks) {
    StegerExtractorCUDA extractor(params_);

    cv::Mat gray = cv::Mat::zeros(50, 100, CV_8UC1);
    cv::Mat binMask = cv::Mat::zeros(50, 100, CV_8UC1);

    for (int x = 10; x < 90; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            float dist = std::abs(dy);
            gray.at<uchar>(25 + dy, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            binMask.at<uchar>(25 + dy, x) = 255;
        }
    }

    cv::cuda::GpuMat d_gray, d_mask;
    d_gray.upload(gray);
    d_mask.upload(binMask);

    auto result = extractor.Execute(d_gray, d_mask, GroupMode::Flat);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.lineCount, 1);
    EXPECT_GT(result.totalPointCount, 0);
}

TEST_F(StegerExtractTest, ExtractFlatAllBackgroundReturnsSuccessWithNoPoints) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(100, 100);

    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_mask(100, 100, CV_8UC1, cv::Scalar(0));

    auto result = extractor.Execute(d_gray, d_mask, GroupMode::Flat);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.totalPointCount, 0);
}

TEST_F(StegerExtractTest, ExtractFlatWrongMaskTypeReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_mask(100, 100, CV_32SC1, cv::Scalar(0));

    auto result = extractor.Execute(d_gray, d_mask, GroupMode::Flat);

    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractByLabelOverloadWrongMaskTypeReturnsError) {
    StegerExtractorCUDA extractor(params_);
    cv::cuda::GpuMat d_gray(100, 100, CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat d_mask(100, 100, CV_8UC1, cv::Scalar(0));

    auto result = extractor.Execute(d_gray, d_mask, GroupMode::ByLabel);

    EXPECT_FALSE(result.success);
}

TEST_F(StegerExtractTest, ExtractByLabelOverloadMatchesLegacyPath) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat labels = cv::Mat::zeros(100, 200, CV_32SC1);

    for (int x = 20; x < 180; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            int y = 50 + dy;
            float dist = std::abs(dy);
            gray.at<uchar>(y, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            labels.at<int>(y, x) = 7;
        }
    }

    cv::cuda::GpuMat d_gray, d_labels;
    d_gray.upload(gray);
    d_labels.upload(labels);

    auto result = extractor.Execute(d_gray, d_labels, GroupMode::ByLabel);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.lineCount, 1);
    EXPECT_GT(result.totalPointCount, 0);
    ASSERT_TRUE(result.centerPoints.count(7) > 0);
}

TEST_F(StegerExtractTest, ExtractFlatNoStreamOverload) {
    StegerExtractorCUDA extractor(params_);
    extractor.Warmup(100, 200);

    cv::Mat gray = cv::Mat::zeros(100, 200, CV_8UC1);
    cv::Mat binMask = cv::Mat::zeros(100, 200, CV_8UC1);

    for (int x = 20; x < 180; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            float dist = std::abs(dy);
            gray.at<uchar>(50 + dy, x) = static_cast<uchar>(255.0f * std::exp(-dist * dist / 0.5f));
            binMask.at<uchar>(50 + dy, x) = 255;
        }
    }

    cv::cuda::GpuMat d_gray, d_mask;
    d_gray.upload(gray);
    d_mask.upload(binMask);

    auto result = extractor.Execute(d_gray, d_mask, GroupMode::Flat);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.totalPointCount, 0);
}

#endif // WITH_CUDA_TESTS
