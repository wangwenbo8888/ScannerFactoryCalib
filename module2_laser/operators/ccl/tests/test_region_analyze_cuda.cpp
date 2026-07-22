/**
 * @file test_region_analyze_cuda.cpp
 * @brief 激光连通域分析算子 - 单元测试（全 GPU, conn=8 专用）
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

#include "../region_analyze_cuda.h"
#include "common/calib_warmup_config.h"

using namespace calib;


// ============================================================
// 测试夹具
// ============================================================
class RegionAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = RegionAnalyzerParams{};
    }

    RegionAnalyzerParams params_;
};

// ============================================================
// 参数校验测试
// ============================================================
TEST_F(RegionAnalyzerTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_EQ(params_.minArea, 100);
    EXPECT_EQ(params_.maxArea, 100000);
}

TEST_F(RegionAnalyzerTest, NegativeMinAreaThrows) {
    params_.minArea = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(RegionAnalyzerTest, MaxAreaLessThanOrEqualMinAreaThrows) {
    params_.minArea = 100;
    params_.maxArea = 100;
    EXPECT_THROW(params_.validate(), std::invalid_argument);

    params_.maxArea = 50;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(RegionAnalyzerTest, NegativeDeviceIdThrows) {
    params_.deviceId = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(RegionAnalyzerTest, MinAreaZeroIsValid) {
    params_.minArea = 0;
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================
// JSON 序列化测试
// ============================================================
TEST_F(RegionAnalyzerTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = RegionAnalyzerParams::fromJson(j);

    EXPECT_EQ(restored.minArea, params_.minArea);
    EXPECT_EQ(restored.maxArea, params_.maxArea);
}

TEST_F(RegionAnalyzerTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"minArea", 200}};
    auto restored = RegionAnalyzerParams::fromJson(j);
    EXPECT_EQ(restored.minArea, 200);
    EXPECT_EQ(restored.maxArea, 100000);
}

TEST_F(RegionAnalyzerTest, JsonUnknownFieldsIgnored) {
    nlohmann::json j = {{"minArea", 50}, {"unknownField", 999}};
    EXPECT_NO_THROW(RegionAnalyzerParams::fromJson(j));
}

TEST_F(RegionAnalyzerTest, JsonInvalidValueThrows) {
    nlohmann::json j = {{"minArea", -1}};
    EXPECT_THROW(RegionAnalyzerParams::fromJson(j), std::invalid_argument);
}

TEST_F(RegionAnalyzerTest, JsonRoundtripAllFields) {
    params_.minArea = 50;
    params_.maxArea = 80000;
    params_.deviceId = 1;

    auto j = params_.toJson();
    EXPECT_TRUE(j.contains("minArea"));
    EXPECT_TRUE(j.contains("maxArea"));
    EXPECT_TRUE(j.contains("deviceId"));

    auto restored = RegionAnalyzerParams::fromJson(j);
    EXPECT_EQ(restored.minArea, 50);
    EXPECT_EQ(restored.maxArea, 80000);
    EXPECT_EQ(restored.deviceId, 1);
}

TEST_F(RegionAnalyzerTest, JsonEmptyObjectUsesDefaults) {
    nlohmann::json j = nlohmann::json::object();
    auto restored = RegionAnalyzerParams::fromJson(j);
    EXPECT_EQ(restored.minArea, 100);
    EXPECT_EQ(restored.maxArea, 100000);
    EXPECT_EQ(restored.deviceId, 0);
}

// ============================================================
// RegionAnalysisResult 结构体测试
// ============================================================
TEST_F(RegionAnalyzerTest, ResultDefaultValues) {
    RegionAnalysisResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.d_labeledMask, nullptr);
    EXPECT_EQ(result.componentCount, 0);
    EXPECT_TRUE(result.components.empty());
}

TEST_F(RegionAnalyzerTest, ResultMoveSemantics) {
    RegionAnalysisResult result1;
    result1.success = true;
    result1.message = "test";
    result1.componentCount = 5;

    RegionAnalysisResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
    EXPECT_EQ(result2.componentCount, 5);
}

// ============================================================
// ComponentStats 测试
// ============================================================
TEST_F(RegionAnalyzerTest, ComponentStatsFields) {
    ComponentStats stats;
    stats.label = 1;
    stats.boundingBoxX = 50;
    stats.boundingBoxY = 100;
    stats.boundingBoxWidth = 101;
    stats.boundingBoxHeight = 201;

    EXPECT_EQ(stats.label, 1);
    EXPECT_EQ(stats.boundingBoxX, 50);
    EXPECT_EQ(stats.boundingBoxY, 100);
    EXPECT_EQ(stats.boundingBoxWidth, 101);
    EXPECT_EQ(stats.boundingBoxHeight, 201);
}

TEST_F(RegionAnalyzerTest, ToRectListEmpty) {
    RegionAnalysisResult result;
    auto rects = result.toRectList();
    EXPECT_TRUE(rects.empty());
}

TEST_F(RegionAnalyzerTest, ToRectListConversion) {
    RegionAnalysisResult result;
    result.components.resize(3);

    result.components[0].boundingBoxX = 10;
    result.components[0].boundingBoxY = 20;
    result.components[0].boundingBoxWidth = 30;
    result.components[0].boundingBoxHeight = 40;

    result.components[1].boundingBoxX = 50;
    result.components[1].boundingBoxY = 60;
    result.components[1].boundingBoxWidth = 15;
    result.components[1].boundingBoxHeight = 25;

    result.components[2].boundingBoxX = 100;
    result.components[2].boundingBoxY = 200;
    result.components[2].boundingBoxWidth = 50;
    result.components[2].boundingBoxHeight = 50;

    auto rects = result.toRectList();
    ASSERT_EQ(rects.size(), 3u);

    EXPECT_EQ(rects[0], cv::Rect(10, 20, 30, 40));
    EXPECT_EQ(rects[1], cv::Rect(50, 60, 15, 25));
    EXPECT_EQ(rects[2], cv::Rect(100, 200, 50, 50));
}

// ============================================================
// WarmupConfig 测试
// ============================================================
TEST_F(RegionAnalyzerTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================
// CUDA 测试（需要 GPU）
// ============================================================
#ifdef WITH_CUDA_TESTS

// ============================================================
// 构造/析构测试
// ============================================================
TEST_F(RegionAnalyzerTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        RegionAnalyzerCUDA analyzer(params_);
    });
}

TEST_F(RegionAnalyzerTest, ConstructWithInvalidParamsThrows) {
    RegionAnalyzerParams badParams;
    badParams.minArea = -1;
    EXPECT_THROW({
        RegionAnalyzerCUDA analyzer(badParams);
    }, std::invalid_argument);
}

// ============================================================
// warmup 测试
// ============================================================
TEST_F(RegionAnalyzerTest, WarmupBasic) {
    RegionAnalyzerCUDA analyzer(params_);
    EXPECT_NO_THROW(analyzer.Warmup(100, 100));
}

TEST_F(RegionAnalyzerTest, WarmupWithConfig) {
    RegionAnalyzerCUDA analyzer(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(analyzer.Warmup(config));
}

// ============================================================
// analyze 测试 - 正常场景
// ============================================================
TEST_F(RegionAnalyzerTest, AnalyzeNormalMask) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(20, 20), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(40, 40), cv::Point(50, 50), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(70, 70), cv::Point(80, 80), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_NE(result.d_labeledMask, nullptr);
    EXPECT_EQ(result.componentCount, 3);
    EXPECT_EQ(result.components.size(), 3u);

    for (const auto& comp : result.components) {
        EXPECT_GT(comp.label, 0);
        EXPECT_GT(comp.boundingBoxWidth, 0);
        EXPECT_GT(comp.boundingBoxHeight, 0);
    }
}

TEST_F(RegionAnalyzerTest, AnalyzeZeroMaskReturnsWarning) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(100, 100);

    cv::cuda::GpuMat d_mask(100, 100, CV_8UC1, cv::Scalar(0));

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
    EXPECT_EQ(result.componentCount, 0);
    EXPECT_NE(result.d_labeledMask, nullptr);
}

TEST_F(RegionAnalyzerTest, AnalyzeFullWhiteMask) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(50, 50);

    cv::cuda::GpuMat d_mask(50, 50, CV_8UC1, cv::Scalar(255));

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
    EXPECT_EQ(result.components.size(), 1u);
    EXPECT_EQ(result.components[0].boundingBoxX, 0);
    EXPECT_EQ(result.components[0].boundingBoxY, 0);
    EXPECT_EQ(result.components[0].boundingBoxWidth, 50);
    EXPECT_EQ(result.components[0].boundingBoxHeight, 50);
}

TEST_F(RegionAnalyzerTest, AnalyzeSinglePixel) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    mask.at<uchar>(50, 50) = 255;

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
    EXPECT_EQ(result.components[0].boundingBoxX, 50);
    EXPECT_EQ(result.components[0].boundingBoxY, 50);
    EXPECT_EQ(result.components[0].boundingBoxWidth, 1);
    EXPECT_EQ(result.components[0].boundingBoxHeight, 1);
}

TEST_F(RegionAnalyzerTest, AreaFiltering) {
    RegionAnalyzerParams p;
    p.minArea = 50;
    p.maxArea = 200;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(200, 200);

    cv::Mat mask = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::rectangle(mask, cv::Point(0, 0), cv::Point(30, 30), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(50, 50), cv::Point(60, 60), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(100, 100), cv::Point(102, 102), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
    EXPECT_EQ(result.componentCount, 1);
}

TEST_F(RegionAnalyzerTest, AnalyzeEmptyInputReturnsError) {
    RegionAnalyzerCUDA analyzer(params_);
    cv::cuda::GpuMat d_empty;
    auto result = analyzer.Execute(d_empty);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(RegionAnalyzerTest, AnalyzeWrongTypeReturnsError) {
    RegionAnalyzerCUDA analyzer(params_);
    cv::cuda::GpuMat d_color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = analyzer.Execute(d_color);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

// ============================================================
// setParams / getParams 测试
// ============================================================
TEST_F(RegionAnalyzerTest, SetParamsAndGetParams) {
    RegionAnalyzerCUDA analyzer(params_);

    RegionAnalyzerParams newParams;
    newParams.minArea = 200;
    newParams.maxArea = 50000;

    analyzer.SetParams(newParams);
    const auto& current = analyzer.GetParams();
    EXPECT_EQ(current.minArea, 200);
    EXPECT_EQ(current.maxArea, 50000);
}

TEST_F(RegionAnalyzerTest, SetInvalidParamsThrows) {
    RegionAnalyzerCUDA analyzer(params_);
    RegionAnalyzerParams badParams;
    badParams.minArea = -1;
    EXPECT_THROW(analyzer.SetParams(badParams), std::invalid_argument);
}

// ============================================================
// 重编号验证
// ============================================================
TEST_F(RegionAnalyzerTest, RelabelingIsConsecutive) {
    RegionAnalyzerParams p;
    p.minArea = 50;
    p.maxArea = 200;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(200, 200);

    cv::Mat mask = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(20, 20), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(40, 40), cv::Point(42, 42), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(60, 60), cv::Point(70, 70), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(90, 90), cv::Point(120, 120), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(140, 140), cv::Point(150, 150), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 3);

    EXPECT_EQ(result.components[0].label, 1);
    EXPECT_EQ(result.components[1].label, 2);
    EXPECT_EQ(result.components[2].label, 3);
}

TEST_F(RegionAnalyzerTest, AnalyzeWithoutWarmupStillWorks) {
    RegionAnalyzerCUDA analyzer(params_);

    cv::Mat mask = cv::Mat::zeros(50, 50, CV_8UC1);
    cv::rectangle(mask, cv::Point(20, 20), cv::Point(30, 30), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
}

// ============================================================
// shared_ptr 重载测试
// ============================================================
TEST_F(RegionAnalyzerTest, AnalyzeSharedPtrNormal) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(20, 20), cv::Scalar(255), -1);

    auto d_mask = std::make_shared<cv::cuda::GpuMat>();
    d_mask->upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
    EXPECT_NE(result.d_labeledMask, nullptr);
}

TEST_F(RegionAnalyzerTest, AnalyzeSharedPtrNullReturnsError) {
    RegionAnalyzerCUDA analyzer(params_);
    std::shared_ptr<cv::cuda::GpuMat> null_ptr;

    auto result = analyzer.Execute(null_ptr);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(RegionAnalyzerTest, AnalyzeSharedPtrEmptyReturnsError) {
    RegionAnalyzerCUDA analyzer(params_);
    auto d_mask = std::make_shared<cv::cuda::GpuMat>();

    auto result = analyzer.Execute(d_mask);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(RegionAnalyzerTest, AnalyzeSharedPtrWrongTypeReturnsError) {
    RegionAnalyzerCUDA analyzer(params_);
    auto d_mask = std::make_shared<cv::cuda::GpuMat>(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));

    auto result = analyzer.Execute(d_mask);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

// ============================================================
// 包围盒精度测试 - L 形区域
// ============================================================
TEST_F(RegionAnalyzerTest, BoundingBoxLShapedRegion) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(20, 20), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(21, 10), cv::Point(30, 15), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.componentCount, 1);

    EXPECT_EQ(result.components[0].boundingBoxX, 10);
    EXPECT_EQ(result.components[0].boundingBoxY, 10);
    EXPECT_EQ(result.components[0].boundingBoxWidth, 21);
    EXPECT_EQ(result.components[0].boundingBoxHeight, 11);
}

// ============================================================
// 边界接触区域测试
// ============================================================
TEST_F(RegionAnalyzerTest, RegionTouchingImageBorders) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(0, 0), cv::Point(10, 10), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(89, 89), cv::Point(99, 99), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 2);

    bool found_top_left = false, found_bottom_right = false;
    for (const auto& comp : result.components) {
        if (comp.boundingBoxX == 0 && comp.boundingBoxY == 0)
            found_top_left = true;
        if (comp.boundingBoxX == 89 && comp.boundingBoxY == 89)
            found_bottom_right = true;
    }
    EXPECT_TRUE(found_top_left);
    EXPECT_TRUE(found_bottom_right);
}

// ============================================================
// 薄线区域测试
// ============================================================
TEST_F(RegionAnalyzerTest, HorizontalLineRegion) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    for (int x = 10; x < 60; ++x) {
        mask.at<uchar>(50, x) = 255;
    }

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.componentCount, 1);
    EXPECT_EQ(result.components[0].boundingBoxHeight, 1);
    EXPECT_EQ(result.components[0].boundingBoxWidth, 50);
}

TEST_F(RegionAnalyzerTest, VerticalLineRegion) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    for (int y = 20; y < 70; ++y) {
        mask.at<uchar>(y, 50) = 255;
    }

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.componentCount, 1);
    EXPECT_EQ(result.components[0].boundingBoxWidth, 1);
    EXPECT_EQ(result.components[0].boundingBoxHeight, 50);
}

TEST_F(RegionAnalyzerTest, LargeImageSize) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(1024, 1024);

    cv::Mat mask = cv::Mat::zeros(1024, 1024, CV_8UC1);
    cv::rectangle(mask, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(500, 500), cv::Point(600, 600), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 2);

    for (const auto& comp : result.components) {
        EXPECT_EQ(comp.boundingBoxWidth, 101);
        EXPECT_EQ(comp.boundingBoxHeight, 101);
    }
}

TEST_F(RegionAnalyzerTest, NonSquareImage) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(200, 500);

    cv::Mat mask = cv::Mat::zeros(200, 500, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(60, 60), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(300, 100), cv::Point(400, 150), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 2);
    EXPECT_NE(result.d_labeledMask, nullptr);
    EXPECT_EQ(result.d_labeledMask->rows, 200);
    EXPECT_EQ(result.d_labeledMask->cols, 500);
}

// ============================================================
// Degraded 质量标记测试（>200 个连通域）
// ============================================================
TEST_F(RegionAnalyzerTest, DegradedQualityFlag) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(640, 480);

    cv::Mat mask = cv::Mat::zeros(480, 640, CV_8UC1);
    int count = 0;
    for (int y = 0; y < 480 && count < 250; y += 30) {
        for (int x = 0; x < 640 && count < 250; x += 30) {
            mask.at<uchar>(y, x) = 255;
            count++;
        }
    }

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.componentCount, 200);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Degraded);
}

// ============================================================
// 重复调用一致性测试
// ============================================================
TEST_F(RegionAnalyzerTest, RepeatedAnalyzeConsistent) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(20, 20), cv::Point(40, 40), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(60, 60), cv::Point(80, 80), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result1 = analyzer.Execute(d_mask);
    auto result2 = analyzer.Execute(d_mask);
    auto result3 = analyzer.Execute(d_mask);

    EXPECT_TRUE(result1.success);
    EXPECT_TRUE(result2.success);
    EXPECT_TRUE(result3.success);

    EXPECT_EQ(result1.componentCount, result2.componentCount);
    EXPECT_EQ(result2.componentCount, result3.componentCount);

    // GPU atomicAdd assigns labels non-deterministically, so sort by bbox before comparing
    auto sortComponents = [](std::vector<ComponentStats>& comps) {
        std::sort(comps.begin(), comps.end(), [](const ComponentStats& a, const ComponentStats& b) {
            if (a.boundingBoxX != b.boundingBoxX) return a.boundingBoxX < b.boundingBoxX;
            return a.boundingBoxY < b.boundingBoxY;
        });
    };
    sortComponents(result1.components);
    sortComponents(result2.components);
    sortComponents(result3.components);

    ASSERT_EQ(result1.components.size(), result2.components.size());
    for (size_t i = 0; i < result1.components.size(); ++i) {
        EXPECT_EQ(result1.components[i].boundingBoxX, result2.components[i].boundingBoxX);
        EXPECT_EQ(result1.components[i].boundingBoxY, result2.components[i].boundingBoxY);
        EXPECT_EQ(result1.components[i].boundingBoxWidth, result2.components[i].boundingBoxWidth);
        EXPECT_EQ(result1.components[i].boundingBoxHeight, result2.components[i].boundingBoxHeight);
    }
}

// ============================================================
// setParams 后重新分析测试
// ============================================================
TEST_F(RegionAnalyzerTest, SetParamsThenReanalyze) {
    RegionAnalyzerParams p1;
    p1.minArea = 1;
    p1.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p1);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(0, 0), cv::Point(2, 2), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(50, 50), cv::Point(70, 70), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result1 = analyzer.Execute(d_mask);
    EXPECT_TRUE(result1.success);
    EXPECT_EQ(result1.componentCount, 2);

    RegionAnalyzerParams p2;
    p2.minArea = 50;
    p2.maxArea = 100000;
    analyzer.SetParams(p2);

    auto result2 = analyzer.Execute(d_mask);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.componentCount, 1);
    EXPECT_GE(result2.components[0].boundingBoxWidth, 7);
    EXPECT_GE(result2.components[0].boundingBoxHeight, 7);
}

TEST_F(RegionAnalyzerTest, WarmupDifferentSizeThenAnalyze) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(200, 200);

    cv::Mat mask = cv::Mat::zeros(50, 50, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(20, 20), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
}

// ============================================================
// 对角线连通性测试 (conn=8)
// ============================================================
TEST_F(RegionAnalyzerTest, DiagonalConnected8) {
    cv::Mat mask = cv::Mat::zeros(20, 20, CV_8UC1);
    mask.at<uchar>(5, 5) = 255;
    mask.at<uchar>(6, 6) = 255;
    mask.at<uchar>(7, 7) = 255;
    mask.at<uchar>(8, 8) = 255;

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(20, 20);
    auto result = analyzer.Execute(d_mask);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
    EXPECT_EQ(result.components[0].boundingBoxX, 5);
    EXPECT_EQ(result.components[0].boundingBoxY, 5);
    EXPECT_EQ(result.components[0].boundingBoxWidth, 4);
    EXPECT_EQ(result.components[0].boundingBoxHeight, 4);
}

TEST_F(RegionAnalyzerTest, ConsecutiveDifferentMasks) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    {
        cv::Mat mask1 = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::rectangle(mask1, cv::Point(10, 10), cv::Point(30, 30), cv::Scalar(255), -1);
        cv::cuda::GpuMat d_mask1;
        d_mask1.upload(mask1);
        auto r1 = analyzer.Execute(d_mask1);
        EXPECT_TRUE(r1.success);
        EXPECT_EQ(r1.componentCount, 1);
    }

    {
        cv::Mat mask2 = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::rectangle(mask2, cv::Point(5, 5), cv::Point(15, 15), cv::Scalar(255), -1);
        cv::rectangle(mask2, cv::Point(50, 50), cv::Point(60, 60), cv::Scalar(255), -1);
        cv::rectangle(mask2, cv::Point(80, 80), cv::Point(90, 90), cv::Scalar(255), -1);
        cv::cuda::GpuMat d_mask2;
        d_mask2.upload(mask2);
        auto r2 = analyzer.Execute(d_mask2);
        EXPECT_TRUE(r2.success);
        EXPECT_EQ(r2.componentCount, 3);
    }

    {
        cv::Mat mask3 = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::cuda::GpuMat d_mask3;
        d_mask3.upload(mask3);
        auto r3 = analyzer.Execute(d_mask3);
        EXPECT_TRUE(r3.success);
        EXPECT_EQ(r3.componentCount, 0);
        EXPECT_EQ(r3.qualityFlag, calib::QualityFlag::Warning);
    }
}

// ============================================================
// 标记掩膜下载验证
// ============================================================
TEST_F(RegionAnalyzerTest, LabeledMaskDownloadCorrect) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(50, 50);

    cv::Mat mask = cv::Mat::zeros(50, 50, CV_8UC1);
    cv::rectangle(mask, cv::Point(5, 5), cv::Point(15, 15), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(30, 30), cv::Point(40, 40), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    ASSERT_NE(result.d_labeledMask, nullptr);
    EXPECT_EQ(result.d_labeledMask->type(), CV_32SC1);

    cv::Mat h_labeled;
    result.d_labeledMask->download(h_labeled);

    EXPECT_EQ(h_labeled.rows, 50);
    EXPECT_EQ(h_labeled.cols, 50);

    double min_val, max_val;
    cv::minMaxLoc(h_labeled, &min_val, &max_val);
    EXPECT_EQ(min_val, 0);
    EXPECT_EQ(max_val, 2);

    EXPECT_EQ(h_labeled.at<int>(10, 10), 1);
    EXPECT_EQ(h_labeled.at<int>(0, 0), 0);
}

// ============================================================
// toRectList 从实际分析结果转换
// ============================================================
TEST_F(RegionAnalyzerTest, ToRectListFromActualAnalysis) {
    RegionAnalyzerParams p;
    p.minArea = 1;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(20, 20), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Point(50, 50), cv::Point(60, 60), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);
    EXPECT_TRUE(result.success);

    auto rects = result.toRectList();
    ASSERT_EQ(rects.size(), 2u);

    EXPECT_EQ(rects[0].width, 11);
    EXPECT_EQ(rects[0].height, 11);
    EXPECT_EQ(rects[1].width, 11);
    EXPECT_EQ(rects[1].height, 11);
}

TEST_F(RegionAnalyzerTest, AnalyzeWithExplicitStream) {
    RegionAnalyzerCUDA analyzer(params_);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(20, 20), cv::Point(40, 40), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    cv::cuda::Stream stream;
    auto result = analyzer.Execute(d_mask, stream);
    stream.waitForCompletion();

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
}

TEST_F(RegionAnalyzerTest, MinAreaBoundaryExact) {
    RegionAnalyzerParams p;
    p.minArea = 50;
    p.maxArea = 100000;

    RegionAnalyzerCUDA analyzer(p);
    analyzer.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(16, 16), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = analyzer.Execute(d_mask);

    EXPECT_TRUE(result.success);
    int area = 7 * 7;
    if (area >= p.minArea) {
        EXPECT_EQ(result.componentCount, 1);
    } else {
        EXPECT_EQ(result.componentCount, 0);
    }
}

#endif // WITH_CUDA_TESTS
