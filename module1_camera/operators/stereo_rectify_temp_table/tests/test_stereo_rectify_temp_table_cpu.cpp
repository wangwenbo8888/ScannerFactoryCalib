#include <gtest/gtest.h>
#include "stereo_rectify_temp_table_cpu.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <cmath>

using namespace calib;

class StereoRectifyTempTableCpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.cameraMatrixL = (cv::Mat_<double>(3, 3) <<
            2500.0, 0.0, 1280.0,
            0.0, 2500.0, 720.0,
            0.0, 0.0, 1.0);
        params_.distCoeffsL = (cv::Mat_<double>(1, 5) << -0.1, 0.05, 0.0, 0.0, 0.0);
        params_.cameraMatrixR = (cv::Mat_<double>(3, 3) <<
            2500.0, 0.0, 1280.0,
            0.0, 2500.0, 720.0,
            0.0, 0.0, 1.0);
        params_.distCoeffsR = (cv::Mat_<double>(1, 5) << -0.1, 0.05, 0.0, 0.0, 0.0);
        params_.R = cv::Mat::eye(3, 3, CV_64F);
        params_.T = (cv::Mat_<double>(3, 1) << 120.0, 0.0, 0.0);
        params_.imageSize = cv::Size(2560, 1440);
        params_.referenceTemp = 25.0;
        params_.cte = 23.6e-6;
        params_.tempStep = 1.0;
        params_.tempRangeMin = -5.0;
        params_.tempRangeMax = 5.0;
        params_.alpha = 0.0;
        params_.flags = cv::CALIB_ZERO_DISPARITY;
    }

    StereoRectifyTempTableParams params_;
};

TEST_F(StereoRectifyTempTableCpuTest, BasicCompute) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.referenceTemp, 25.0);
    EXPECT_GT(result.table.size(), 0u);

    double tMin = 25.0 + (-5.0);
    double tMax = 25.0 + 5.0;
    int expectedCount = static_cast<int>(std::round((tMax - tMin) / 1.0)) + 1;
    EXPECT_EQ(result.table.size(), static_cast<size_t>(expectedCount));
}

TEST_F(StereoRectifyTempTableCpuTest, EntryHasValidRectifyResult) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    ASSERT_GT(result.table.size(), 0u);

    const auto& entry = result.table[0];
    EXPECT_EQ(entry.R1.rows, 3);
    EXPECT_EQ(entry.R1.cols, 3);
    EXPECT_EQ(entry.R2.rows, 3);
    EXPECT_EQ(entry.R2.cols, 3);
    EXPECT_EQ(entry.P1.rows, 3);
    EXPECT_EQ(entry.P1.cols, 4);
    EXPECT_EQ(entry.P2.rows, 3);
    EXPECT_EQ(entry.P2.cols, 4);
    EXPECT_EQ(entry.Q.rows, 4);
    EXPECT_EQ(entry.Q.cols, 4);
}

TEST_F(StereoRectifyTempTableCpuTest, ReferenceTempEntryMatchesOriginal) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    bool foundRef = false;
    for (const auto& entry : result.table) {
        if (std::abs(entry.deltaT) < 1e-9) {
            foundRef = true;
            double dFx = std::abs(entry.compensatedCameraMatrixL.at<double>(0, 0)
                                  - params_.cameraMatrixL.at<double>(0, 0));
            EXPECT_LT(dFx, 1e-6);
            break;
        }
    }
    EXPECT_TRUE(foundRef);
}

TEST_F(StereoRectifyTempTableCpuTest, CompensatedIntrinsicsVaryWithTemp) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    ASSERT_GE(result.table.size(), 2u);

    const auto& cold = result.table.front();
    const auto& hot = result.table.back();

    double fx0 = params_.cameraMatrixL.at<double>(0, 0);
    double fxCold = cold.compensatedCameraMatrixL.at<double>(0, 0);
    double fxHot = hot.compensatedCameraMatrixL.at<double>(0, 0);

    EXPECT_LT(fxCold, fx0);
    EXPECT_GT(fxHot, fx0);
}

TEST_F(StereoRectifyTempTableCpuTest, CompensatedBaselineVaryWithTemp) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    ASSERT_GE(result.table.size(), 2u);

    const auto& cold = result.table.front();
    const auto& hot = result.table.back();

    double baselineRef = params_.T.at<double>(0, 0);
    double baselineCold = cold.compensatedT.at<double>(0, 0);
    double baselineHot = hot.compensatedT.at<double>(0, 0);

    EXPECT_LT(baselineCold, baselineRef);
    EXPECT_GT(baselineHot, baselineRef);
}

TEST_F(StereoRectifyTempTableCpuTest, ToJsonProducesValidOutput) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    auto j = result.toJson();
    EXPECT_TRUE(j["success"].get<bool>());
    EXPECT_TRUE(j.contains("table"));
    EXPECT_TRUE(j["table"].is_array());
    EXPECT_EQ(j["table"].size(), result.table.size());

    const auto& entry0 = j["table"][0];
    EXPECT_TRUE(entry0.contains("R1"));
    EXPECT_TRUE(entry0.contains("P1"));
    EXPECT_TRUE(entry0.contains("Q"));
    EXPECT_TRUE(entry0.contains("temperature"));
    EXPECT_TRUE(entry0.contains("deltaT"));
}

TEST_F(StereoRectifyTempTableCpuTest, InvalidParamsThrows) {
    auto badParams = params_;
    badParams.cameraMatrixL = cv::Mat();
    EXPECT_THROW(StereoRectifyTempTableCpu op(badParams), std::invalid_argument);
}

TEST_F(StereoRectifyTempTableCpuTest, SmallStepProducesMoreEntries) {
    auto smallParams = params_;
    smallParams.tempStep = 0.5;
    smallParams.tempRangeMin = -2.0;
    smallParams.tempRangeMax = 2.0;

    StereoRectifyTempTableCpu op(smallParams);
    auto result = op.Execute();

    EXPECT_TRUE(result.success);
    int expectedCount = static_cast<int>(std::round(4.0 / 0.5)) + 1;
    EXPECT_EQ(result.table.size(), static_cast<size_t>(expectedCount));
}

TEST_F(StereoRectifyTempTableCpuTest, QMatrixDriftsWithTemperature) {
    StereoRectifyTempTableCpu op(params_);
    auto result = op.Execute();

    ASSERT_GE(result.table.size(), 3u);

    const auto& entry0 = result.table.front();
    const auto& entryN = result.table.back();

    double qDiff = cv::norm(entryN.Q - entry0.Q);
    EXPECT_GT(qDiff, 0.0);
}
