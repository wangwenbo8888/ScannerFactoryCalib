/**
 * @file test_mask_extract_cuda.cpp
 * @brief 激光掩膜提取算子 - 单元测试
 *
 * 测试覆盖：
 * - 参数校验（合法/非法参数）
 * - 构造/析构
 * - warmup 预热
 * - extract 正常处理
 * - extract 空图像/错误类型输入
 * - setParams 动态更新
 * - getParams 参数查询
 * - warmup(WarmupConfig) 统一配置接口
 * - Result 结构体移动语义
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include "../mask_extract_cuda.h"
#include "common/calib_warmup_config.h"

using namespace calib;


// ============================================================
// 测试夹具
// ============================================================
class MaskExtractTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 使用默认参数
        params_ = MaskExtractParams{};
    }

    MaskExtractParams params_;
};

// ============================================================
// 参数校验测试
// ============================================================
TEST_F(MaskExtractTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_EQ(params_.threshold, 80);
    EXPECT_EQ(params_.erodeSize, 5);
    EXPECT_EQ(params_.laserDilateSize, 3);
}

TEST_F(MaskExtractTest, InvalidThresholdThrows) {
    params_.threshold = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);

    params_.threshold = 256;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(MaskExtractTest, EvenErodeSizeThrows) {
    params_.erodeSize = 4;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(MaskExtractTest, ZeroErodeSizeThrows) {
    params_.erodeSize = 0;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(MaskExtractTest, EvenDilateSizeThrows) {
    params_.laserDilateSize = 2;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(MaskExtractTest, NegativeMinAreaThrows) {
    params_.minArea = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(MaskExtractTest, MaxAreaLessThanOrEqualMinAreaThrows) {
    params_.minArea = 100;
    params_.maxArea = 100;
    EXPECT_THROW(params_.validate(), std::invalid_argument);

    params_.maxArea = 50;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化测试
// ============================================================
TEST_F(MaskExtractTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = MaskExtractParams::fromJson(j);

    EXPECT_EQ(restored.threshold, params_.threshold);
    EXPECT_EQ(restored.erodeSize, params_.erodeSize);
    EXPECT_EQ(restored.laserDilateSize, params_.laserDilateSize);
    EXPECT_EQ(restored.minArea, params_.minArea);
    EXPECT_EQ(restored.maxArea, params_.maxArea);
}

TEST_F(MaskExtractTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"threshold", 120}};
    auto restored = MaskExtractParams::fromJson(j);
    EXPECT_EQ(restored.threshold, 120);
    // 其他字段保持默认值
    EXPECT_EQ(restored.erodeSize, 5);
}

// ============================================================
// Result 结构体测试
// ============================================================
TEST_F(MaskExtractTest, ResultDefaultValues) {
    MaskExtractResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.d_grayImage, nullptr);
    EXPECT_EQ(result.d_laserMask, nullptr);
    EXPECT_EQ(result.d_cleanedMask, nullptr);
}

TEST_F(MaskExtractTest, ResultMoveSemantics) {
    MaskExtractResult result1;
    result1.success = true;
    result1.message = "test";

    MaskExtractResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
}

// ============================================================
// WarmupConfig 测试
// ============================================================
TEST_F(MaskExtractTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================
// 构造/析构测试（需要 GPU）
// ============================================================
#ifdef WITH_CUDA_TESTS
TEST_F(MaskExtractTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        MaskExtractCUDA extractor(params_);
    });
}

TEST_F(MaskExtractTest, ConstructWithInvalidParamsThrows) {
    MaskExtractParams badParams;
    badParams.threshold = 999;
    EXPECT_THROW({
        MaskExtractCUDA extractor(badParams);
    }, std::invalid_argument);
}

TEST_F(MaskExtractTest, WarmupAndExtract) {
    MaskExtractCUDA extractor(params_);
    extractor.Warmup(100, 100);

    // 创建测试图像（黑色背景 + 白色矩形模拟激光）
    cv::Mat gray = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(gray, cv::Point(40, 45), cv::Point(60, 55), cv::Scalar(200), -1);

    auto result = extractor.Execute(gray);
    EXPECT_TRUE(result.success);
    EXPECT_NE(result.d_grayImage, nullptr);
    EXPECT_NE(result.d_laserMask, nullptr);
    EXPECT_NE(result.d_cleanedMask, nullptr);

    // 验证 d_grayImage 格式和尺寸
    EXPECT_EQ(result.d_grayImage->type(), CV_8UC1);
    EXPECT_EQ(result.d_grayImage->rows, 100);
    EXPECT_EQ(result.d_grayImage->cols, 100);

    // 韩证 d_grayImage 内容与输入一致
    cv::Mat downloadedGray;
    result.d_grayImage->download(downloadedGray);
    EXPECT_EQ(cv::countNonZero(gray != downloadedGray), 0);
}

TEST_F(MaskExtractTest, ExtractEmptyImageReturnsError) {
    MaskExtractCUDA extractor(params_);
    cv::Mat empty;
    auto result = extractor.Execute(empty);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(MaskExtractTest, ExtractWrongTypeReturnsError) {
    MaskExtractCUDA extractor(params_);
    cv::Mat color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = extractor.Execute(color);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(MaskExtractTest, WarmupWithConfig) {
    MaskExtractCUDA extractor(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(extractor.Warmup(config));
}

TEST_F(MaskExtractTest, SetParamsAndGetParams) {
    MaskExtractCUDA extractor(params_);

    MaskExtractParams newParams;
    newParams.threshold = 120;
    newParams.erodeSize = 7;
    newParams.laserDilateSize = 5;
    newParams.minArea = 50;
    newParams.maxArea = 200000;

    extractor.SetParams(newParams);
    const auto& current = extractor.GetParams();
    EXPECT_EQ(current.threshold, 120);
    EXPECT_EQ(current.erodeSize, 7);
    EXPECT_EQ(current.laserDilateSize, 5);
}

TEST_F(MaskExtractTest, SetInvalidParamsThrows) {
    MaskExtractCUDA extractor(params_);
    MaskExtractParams badParams;
    badParams.threshold = 999;
    EXPECT_THROW(extractor.SetParams(badParams), std::invalid_argument);
}
#endif // WITH_CUDA_TESTS
