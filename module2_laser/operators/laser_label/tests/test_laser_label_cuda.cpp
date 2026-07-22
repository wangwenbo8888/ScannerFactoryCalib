/**
 * @file test_laser_label_cuda.cpp
 * @brief 激光线编号算子 - 单元测试
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

#include "../laser_label_cuda.h"
#include "common/calib_warmup_config.h"

using namespace calib;


// ============================================================================
// 测试夹具
// ============================================================================
class LaserLabelTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = LaserLabelParams{};
    }

    LaserLabelParams params_;
};

// ============================================================================
// 参数校验测试
// ============================================================================
TEST_F(LaserLabelTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_EQ(params_.maxLabels, 256);
    EXPECT_EQ(params_.centerColOffset, 0);
    EXPECT_EQ(params_.deviceId, 0);
}

TEST_F(LaserLabelTest, MaxLabelsTooSmallThrows) {
    params_.maxLabels = 0;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserLabelTest, MaxLabelsTooLargeThrows) {
    params_.maxLabels = 5000;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserLabelTest, CenterColOffsetOutOfRangeThrows) {
    params_.centerColOffset = -501;
    EXPECT_THROW(params_.validate(), std::invalid_argument);

    params_.centerColOffset = 501;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserLabelTest, NegativeDeviceIdThrows) {
    params_.deviceId = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserLabelTest, ValidBoundaryParams) {
    params_.maxLabels = 1;
    EXPECT_NO_THROW(params_.validate());

    params_.maxLabels = 4096;
    EXPECT_NO_THROW(params_.validate());

    params_.centerColOffset = -500;
    EXPECT_NO_THROW(params_.validate());

    params_.centerColOffset = 500;
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================================
// JSON 序列化测试
// ============================================================================
TEST_F(LaserLabelTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = LaserLabelParams::fromJson(j);

    EXPECT_EQ(restored.maxLabels, params_.maxLabels);
    EXPECT_EQ(restored.centerColOffset, params_.centerColOffset);
    EXPECT_EQ(restored.deviceId, params_.deviceId);
}

TEST_F(LaserLabelTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"maxLabels", 512}};
    auto restored = LaserLabelParams::fromJson(j);
    EXPECT_EQ(restored.maxLabels, 512);
    EXPECT_EQ(restored.centerColOffset, 0);
    EXPECT_EQ(restored.deviceId, 0);
}

TEST_F(LaserLabelTest, JsonUnknownFieldsIgnored) {
    nlohmann::json j = {{"maxLabels", 100}, {"unknownField", 999}};
    EXPECT_NO_THROW(LaserLabelParams::fromJson(j));
}

TEST_F(LaserLabelTest, JsonInvalidValueThrows) {
    nlohmann::json j = {{"maxLabels", 0}};
    EXPECT_THROW(LaserLabelParams::fromJson(j), std::invalid_argument);
}

// ============================================================================
// LaserLabelResult 结构体测试
// ============================================================================
TEST_F(LaserLabelTest, ResultDefaultValues) {
    LaserLabelResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.d_labeledMask, nullptr);
    EXPECT_EQ(result.componentCount, 0);
}

TEST_F(LaserLabelTest, ResultMoveSemantics) {
    LaserLabelResult result1;
    result1.success = true;
    result1.message = "test";
    result1.componentCount = 5;

    LaserLabelResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
    EXPECT_EQ(result2.componentCount, 5);
}

// ============================================================================
// WarmupConfig 测试
// ============================================================================
TEST_F(LaserLabelTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================================
// CUDA 测试（需要 GPU）
// ============================================================================
#ifdef WITH_CUDA_TESTS

// ============================================================================
// 构造/析构测试
// ============================================================================
TEST_F(LaserLabelTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        LaserLabelerCUDA labeler(params_);
    });
}

TEST_F(LaserLabelTest, ConstructWithInvalidParamsThrows) {
    LaserLabelParams badParams;
    badParams.maxLabels = 0;
    EXPECT_THROW({
        LaserLabelerCUDA labeler(badParams);
    }, std::invalid_argument);
}

// ============================================================================
// warmup 测试
// ============================================================================
TEST_F(LaserLabelTest, WarmupBasic) {
    LaserLabelerCUDA labeler(params_);
    EXPECT_NO_THROW(labeler.Warmup(100, 100));
}

TEST_F(LaserLabelTest, WarmupWithConfig) {
    LaserLabelerCUDA labeler(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(labeler.Warmup(config));
}

// ============================================================================
// label 测试 - CV_32SC1 输入（标记掩膜）
// ============================================================================
TEST_F(LaserLabelTest, LabelWith32SC1Input) {
    LaserLabelerCUDA labeler(params_);
    labeler.Warmup(100, 100);

    cv::Mat labels_cpu = cv::Mat::zeros(100, 100, CV_32SC1);
    cv::rectangle(labels_cpu, cv::Point(10, 10), cv::Point(90, 15), cv::Scalar(1), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 40), cv::Point(90, 45), cv::Scalar(2), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 70), cv::Point(90, 75), cv::Scalar(3), -1);

    cv::cuda::GpuMat d_labels;
    d_labels.upload(labels_cpu);

    auto result = labeler.Execute(d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_NE(result.d_labeledMask, nullptr);
    EXPECT_EQ(result.componentCount, 3);

    cv::Mat h_output;
    result.d_labeledMask->download(h_output);

    int center_x = 50;
    int label_at_12 = h_output.at<int>(12, center_x);
    int label_at_42 = h_output.at<int>(42, center_x);
    int label_at_72 = h_output.at<int>(72, center_x);

    EXPECT_EQ(label_at_12, 1);
    EXPECT_EQ(label_at_42, 2);
    EXPECT_EQ(label_at_72, 3);
}

// ============================================================================
// label 测试 - CV_8UC1 输入不再支持
// ============================================================================
TEST_F(LaserLabelTest, LabelWith8UC1InputReturnsError) {
    LaserLabelerCUDA labeler(params_);
    labeler.Warmup(100, 100);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Point(10, 10), cv::Point(90, 15), cv::Scalar(255), -1);

    cv::cuda::GpuMat d_mask;
    d_mask.upload(mask);

    auto result = labeler.Execute(d_mask);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("CV_32SC1"), std::string::npos);
}

// ============================================================================
// label 测试 - 重编号正确性
// ============================================================================
TEST_F(LaserLabelTest, RelabelingByCenterColumnMinY) {
    LaserLabelerCUDA labeler(params_);
    labeler.Warmup(200, 200);

    cv::Mat labels_cpu = cv::Mat::zeros(200, 200, CV_32SC1);
    cv::rectangle(labels_cpu, cv::Point(10, 100), cv::Point(190, 105), cv::Scalar(3), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 50), cv::Point(190, 55), cv::Scalar(1), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 150), cv::Point(190, 155), cv::Scalar(2), -1);

    cv::cuda::GpuMat d_labels;
    d_labels.upload(labels_cpu);

    auto result = labeler.Execute(d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 3);

    cv::Mat h_output;
    result.d_labeledMask->download(h_output);

    int center_x = 100;
    EXPECT_EQ(h_output.at<int>(52, center_x), 1);
    EXPECT_EQ(h_output.at<int>(102, center_x), 2);
    EXPECT_EQ(h_output.at<int>(152, center_x), 3);
}

// ============================================================================
// label 测试 - 空输入
// ============================================================================
TEST_F(LaserLabelTest, LabelEmptyInputReturnsError) {
    LaserLabelerCUDA labeler(params_);
    cv::cuda::GpuMat d_empty;
    auto result = labeler.Execute(d_empty);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

// ============================================================================
// label 测试 - 全背景
// ============================================================================
TEST_F(LaserLabelTest, LabelAllBackgroundReturnsError) {
    LaserLabelerCUDA labeler(params_);
    labeler.Warmup(100, 100);

    cv::cuda::GpuMat d_labels(100, 100, CV_32SC1, cv::Scalar(0));
    auto result = labeler.Execute(d_labels);

    EXPECT_FALSE(result.success);
}

// ============================================================================
// label 测试 - 单连通域
// ============================================================================
TEST_F(LaserLabelTest, LabelSingleComponent) {
    LaserLabelerCUDA labeler(params_);
    labeler.Warmup(100, 100);

    cv::Mat labels_cpu = cv::Mat::zeros(100, 100, CV_32SC1);
    cv::rectangle(labels_cpu, cv::Point(20, 20), cv::Point(80, 25), cv::Scalar(1), -1);

    cv::cuda::GpuMat d_labels;
    d_labels.upload(labels_cpu);

    auto result = labeler.Execute(d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);

    cv::Mat h_output;
    result.d_labeledMask->download(h_output);

    int center_x = 50;
    EXPECT_EQ(h_output.at<int>(22, center_x), 1);
}

// ============================================================================
// label 测试 - 中心列偏移
// ============================================================================
TEST_F(LaserLabelTest, LabelWithCenterColOffset) {
    LaserLabelParams offsetParams;
    offsetParams.centerColOffset = 10;
    LaserLabelerCUDA labeler(offsetParams);
    labeler.Warmup(100, 100);

    cv::Mat labels_cpu = cv::Mat::zeros(100, 100, CV_32SC1);
    cv::rectangle(labels_cpu, cv::Point(10, 20), cv::Point(90, 25), cv::Scalar(1), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 60), cv::Point(90, 65), cv::Scalar(2), -1);

    cv::cuda::GpuMat d_labels;
    d_labels.upload(labels_cpu);

    auto result = labeler.Execute(d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 2);
}

// ============================================================================
// label 测试 - 类型不匹配
// ============================================================================
TEST_F(LaserLabelTest, LabelWrongTypeReturnsError) {
    LaserLabelerCUDA labeler(params_);
    cv::cuda::GpuMat d_color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = labeler.Execute(d_color);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

// ============================================================================
// setParams / getParams 测试
// ============================================================================
TEST_F(LaserLabelTest, SetParamsAndGetParams) {
    LaserLabelerCUDA labeler(params_);

    LaserLabelParams newParams;
    newParams.maxLabels = 512;
    newParams.centerColOffset = 5;

    labeler.SetParams(newParams);
    const auto& current = labeler.GetParams();
    EXPECT_EQ(current.maxLabels, 512);
    EXPECT_EQ(current.centerColOffset, 5);
}

TEST_F(LaserLabelTest, SetInvalidParamsThrows) {
    LaserLabelerCUDA labeler(params_);
    LaserLabelParams badParams;
    badParams.maxLabels = 0;
    EXPECT_THROW(labeler.SetParams(badParams), std::invalid_argument);
}

// ============================================================================
// 未经 warmup 直接调用
// ============================================================================
TEST_F(LaserLabelTest, LabelWithoutWarmupStillWorks) {
    LaserLabelerCUDA labeler(params_);

    cv::Mat labels_cpu = cv::Mat::zeros(50, 50, CV_32SC1);
    cv::rectangle(labels_cpu, cv::Point(10, 20), cv::Point(40, 25), cv::Scalar(1), -1);

    cv::cuda::GpuMat d_labels;
    d_labels.upload(labels_cpu);

    auto result = labeler.Execute(d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 1);
}

// ============================================================================
// 重编号连续性验证
// ============================================================================
TEST_F(LaserLabelTest, RelabelingIsConsecutive) {
    LaserLabelerCUDA labeler(params_);
    labeler.Warmup(200, 200);

    cv::Mat labels_cpu = cv::Mat::zeros(200, 200, CV_32SC1);
    cv::rectangle(labels_cpu, cv::Point(10, 120), cv::Point(190, 125), cv::Scalar(5), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 80), cv::Point(190, 85), cv::Scalar(3), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 40), cv::Point(190, 45), cv::Scalar(1), -1);
    cv::rectangle(labels_cpu, cv::Point(10, 160), cv::Point(190, 165), cv::Scalar(7), -1);

    cv::cuda::GpuMat d_labels;
    d_labels.upload(labels_cpu);

    auto result = labeler.Execute(d_labels);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.componentCount, 4);

    cv::Mat h_output;
    result.d_labeledMask->download(h_output);

    int center_x = 100;
    std::set<int> labels_found;
    for (int y = 0; y < 200; ++y) {
        int val = h_output.at<int>(y, center_x);
        if (val > 0) labels_found.insert(val);
    }

    EXPECT_EQ(labels_found.size(), 4u);
    EXPECT_NE(labels_found.find(1), labels_found.end());
    EXPECT_NE(labels_found.find(2), labels_found.end());
    EXPECT_NE(labels_found.find(3), labels_found.end());
    EXPECT_NE(labels_found.find(4), labels_found.end());

    EXPECT_EQ(h_output.at<int>(42, center_x), 1);
    EXPECT_EQ(h_output.at<int>(82, center_x), 2);
    EXPECT_EQ(h_output.at<int>(122, center_x), 3);
    EXPECT_EQ(h_output.at<int>(162, center_x), 4);
}

#endif // WITH_CUDA_TESTS
