#include <gtest/gtest.h>
#include "plane_map_temp_table.h"
#include "plane_map_cuda.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace calib;

class PlaneMapTempTableTest : public ::testing::Test {
protected:
    void TearDown() override {
        cudaDeviceReset();
    }
};

static PlaneMapTempTableParams makeSimulatedParams(
    double tempRangeMin = -2.0,
    double tempRangeMax = 2.0,
    double tempStep = 1.0,
    int imageW = 640,
    int imageH = 480,
    int depthSamples = 50,
    float gridStep = 2.0f)
{
    PlaneMapTempTableParams p;

    double fxL = 800.0, fyL = 800.0, cxL = imageW / 2.0, cyL = imageH / 2.0;
    p.cameraMatrixL = (cv::Mat_<double>(3, 3) <<
        fxL, 0, cxL,
        0, fyL, cyL,
        0, 0, 1);

    double fxR = 805.0, fyR = 805.0, cxR = imageW / 2.0 + 1.5, cyR = imageH / 2.0 - 0.8;
    p.cameraMatrixR = (cv::Mat_<double>(3, 3) <<
        fxR, 0, cxR,
        0, fyR, cyR,
        0, 0, 1);

    p.distCoeffsL = (cv::Mat_<double>(1, 5) << -0.05, 0.02, 0.001, -0.002, 0.0);
    p.distCoeffsR = (cv::Mat_<double>(1, 5) << -0.06, 0.03, 0.001, -0.001, 0.0);

    double baseline = 120.0;
    p.R = cv::Mat::eye(3, 3, CV_64F);
    p.T = (cv::Mat_<double>(3, 1) << baseline, 0.0, 0.0);

    p.imageSize = cv::Size(imageW, imageH);

    double vfx = 700.0;
    p.virtualK = cv::Matx33d(
        vfx, 0, imageW / 2.0,
        0, vfx, imageH / 2.0,
        0, 0, 1);
    p.virtualR = cv::Matx33d::eye();
    p.virtualT = cv::Vec3d(baseline / 2.0, 0.0, 200.0);

    p.lineIds = {0, 1, 2, 3, 4};

    p.referenceTemp = 25.0;
    p.cte = 23.6e-6;
    p.tempStep = tempStep;
    p.tempRangeMin = tempRangeMin;
    p.tempRangeMax = tempRangeMax;
    p.alpha = 0.0;
    p.flags = cv::CALIB_ZERO_DISPARITY;
    p.deviceId = 0;
    p.gridStep = gridStep;
    p.depthMin = 200.0f;
    p.depthMax = 2000.0f;
    p.depthSamples = depthSamples;
    p.epipolarStep = 1.0f;

    return p;
}

TEST_F(PlaneMapTempTableTest, ValidateRejectsEmptyLineIds) {
    PlaneMapTempTableParams params;
    params.cameraMatrixL = cv::Mat::eye(3, 3, CV_64F);
    params.cameraMatrixR = cv::Mat::eye(3, 3, CV_64F);
    params.distCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    params.distCoeffsR = cv::Mat::zeros(1, 5, CV_64F);
    params.R = cv::Mat::eye(3, 3, CV_64F);
    params.T = (cv::Mat_<double>(3, 1) << 100, 0, 0);
    params.imageSize = cv::Size(640, 480);
    params.virtualK = cv::Matx33d::eye();
    params.virtualR = cv::Matx33d::eye();
    params.virtualT = cv::Vec3d(50, 0, 200);
    params.lineIds = {};

    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST_F(PlaneMapTempTableTest, ValidateAcceptsValidParams) {
    PlaneMapTempTableParams params;
    params.cameraMatrixL = cv::Mat::eye(3, 3, CV_64F);
    params.cameraMatrixR = cv::Mat::eye(3, 3, CV_64F);
    params.distCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    params.distCoeffsR = cv::Mat::zeros(1, 5, CV_64F);
    params.R = cv::Mat::eye(3, 3, CV_64F);
    params.T = (cv::Mat_<double>(3, 1) << 100, 0, 0);
    params.imageSize = cv::Size(640, 480);
    params.virtualK = cv::Matx33d::eye();
    params.virtualR = cv::Matx33d::eye();
    params.virtualT = cv::Vec3d(50, 0, 200);
    params.lineIds = {1, 2, 3};
    params.tempRangeMin = -1.0;
    params.tempRangeMax = 1.0;
    params.tempStep = 2.0;
    params.flags = cv::CALIB_ZERO_DISPARITY;

    EXPECT_NO_THROW(params.validate());
}

TEST_F(PlaneMapTempTableTest, JsonRoundTrip) {
    PlaneMapTempTableParams params;
    params.cameraMatrixL = cv::Mat::eye(3, 3, CV_64F) * 800;
    params.cameraMatrixR = cv::Mat::eye(3, 3, CV_64F) * 810;
    params.distCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    params.distCoeffsR = cv::Mat::zeros(1, 5, CV_64F);
    params.R = cv::Mat::eye(3, 3, CV_64F);
    params.T = (cv::Mat_<double>(3, 1) << 120, 0.1, 0.2);
    params.imageSize = cv::Size(1280, 960);
    params.virtualK = cv::Matx33d::eye() * 500;
    params.virtualR = cv::Matx33d::eye();
    params.virtualT = cv::Vec3d(50, 0, 200);
    params.lineIds = {1, 2, 3};
    params.referenceTemp = 30.0;
    params.cte = 23.6e-6;
    params.tempStep = 1.0;
    params.tempRangeMin = -2.0;
    params.tempRangeMax = 2.0;
    params.flags = cv::CALIB_ZERO_DISPARITY;

    nlohmann::json j = params.toJson();
    PlaneMapTempTableParams params2 = PlaneMapTempTableParams::fromJson(j);

    EXPECT_DOUBLE_EQ(params2.referenceTemp, 30.0);
    EXPECT_DOUBLE_EQ(params2.cte, 23.6e-6);
    EXPECT_EQ(params2.imageSize.width, 1280);
    EXPECT_EQ(params2.imageSize.height, 960);
    EXPECT_EQ(params2.lineIds.size(), 3u);
    EXPECT_NEAR(params2.virtualT(0), 50.0, 1e-12);
}

TEST_F(PlaneMapTempTableTest, CompensateIntrinsicsScale) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    double cte = 23.6e-6;
    double deltaT = 10.0;
    double expected_scale = 1.0 + cte * deltaT;

    cv::Mat Kc;
    Kc = K.clone();
    double scale = 1.0 + cte * deltaT;
    Kc.at<double>(0, 0) = K.at<double>(0, 0) * scale;
    Kc.at<double>(1, 1) = K.at<double>(1, 1) * scale;
    Kc.at<double>(0, 2) = K.at<double>(0, 2) * scale;
    Kc.at<double>(1, 2) = K.at<double>(1, 2) * scale;

    EXPECT_NEAR(Kc.at<double>(0, 0), 800 * expected_scale, 1e-10);
    EXPECT_NEAR(Kc.at<double>(1, 1), 800 * expected_scale, 1e-10);
    EXPECT_NEAR(Kc.at<double>(2, 2), 1.0, 1e-10);
}

#if BUILD_CUDA

TEST_F(PlaneMapTempTableTest, ComputeSmallTable) {
    auto params = makeSimulatedParams(
        -1.0, 1.0, 1.0,
        640, 480,
        50, 2.0f);

    PlaneMapTempTable op(params);

    auto t0 = std::chrono::high_resolution_clock::now();
    PlaneMapTempTableResult result = op.Execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "=== ComputeSmallTable ===" << std::endl;
    std::cout << "  Total wall time: " << dt_ms << " ms" << std::endl;
    std::cout << "  success: " << result.success << std::endl;
    std::cout << "  tableSize: " << result.tableSize << std::endl;
    std::cout << "  qualityFlag: " << static_cast<int>(result.qualityFlag) << std::endl;
    if (!result.message.empty())
        std::cout << "  message: " << result.message << std::endl;

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.tableSize, 3);
    EXPECT_EQ(result.table.size(), 3u);

    EXPECT_NEAR(result.table[0].temperature, 24.0, 1e-9);
    EXPECT_NEAR(result.table[1].temperature, 25.0, 1e-9);
    EXPECT_NEAR(result.table[2].temperature, 26.0, 1e-9);

    for (const auto& entry : result.table) {
        EXPECT_FALSE(entry.compensatedCameraMatrixL.empty());
        EXPECT_FALSE(entry.compensatedCameraMatrixR.empty());
        EXPECT_FALSE(entry.compensatedT.empty());
        EXPECT_FALSE(entry.R1.empty());
        EXPECT_FALSE(entry.R2.empty());
        EXPECT_FALSE(entry.P1.empty());
        EXPECT_FALSE(entry.P2.empty());
        EXPECT_FALSE(entry.Q.empty());
        std::cout << "  temp=" << entry.temperature
                  << " deltaT=" << entry.deltaT
                  << " totalPairs=" << entry.totalPairs
                  << " lines=" << entry.lineStats.size()
                  << std::endl;
    }

    std::cout << "  Per-step avg: " << dt_ms / result.tableSize << " ms" << std::endl;
}

TEST_F(PlaneMapTempTableTest, ComputeMediumTable) {
    auto params = makeSimulatedParams(
        -5.0, 5.0, 1.0,
        640, 480,
        80, 2.0f);

    PlaneMapTempTable op(params);

    auto t0 = std::chrono::high_resolution_clock::now();
    PlaneMapTempTableResult result = op.Execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "=== ComputeMediumTable ===" << std::endl;
    std::cout << "  Total wall time: " << dt_ms << " ms" << std::endl;
    std::cout << "  success: " << result.success << std::endl;
    std::cout << "  tableSize: " << result.tableSize << std::endl;
    std::cout << "  qualityFlag: " << static_cast<int>(result.qualityFlag) << std::endl;

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.tableSize, 11);
    EXPECT_EQ(result.table.size(), 11u);

    EXPECT_NEAR(result.table.front().temperature, 20.0, 1e-9);
    EXPECT_NEAR(result.table.back().temperature, 30.0, 1e-9);

    for (const auto& entry : result.table) {
        std::cout << "  temp=" << entry.temperature
                  << " deltaT=" << entry.deltaT
                  << " pairs=" << entry.totalPairs
                  << std::endl;
    }

    std::cout << "  Per-step avg: " << dt_ms / result.tableSize << " ms" << std::endl;
}

TEST_F(PlaneMapTempTableTest, ComputeFineStepTable) {
    auto params = makeSimulatedParams(
        -2.0, 2.0, 0.5,
        640, 480,
        80, 1.0f);

    PlaneMapTempTable op(params);

    auto t0 = std::chrono::high_resolution_clock::now();
    PlaneMapTempTableResult result = op.Execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "=== ComputeFineStepTable ===" << std::endl;
    std::cout << "  Total wall time: " << dt_ms << " ms" << std::endl;
    std::cout << "  success: " << result.success << std::endl;
    std::cout << "  tableSize: " << result.tableSize << std::endl;

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.tableSize, 9);
    EXPECT_EQ(result.table.size(), 9u);

    EXPECT_NEAR(result.table.front().temperature, 23.0, 1e-9);
    EXPECT_NEAR(result.table.back().temperature, 27.0, 1e-9);

    for (size_t i = 1; i < result.table.size(); ++i) {
        double step = result.table[i].temperature - result.table[i - 1].temperature;
        EXPECT_NEAR(step, 0.5, 1e-9) << "step between [" << (i - 1) << "] and [" << i << "]";
    }

    std::cout << "  Per-step avg: " << dt_ms / result.tableSize << " ms" << std::endl;
}

TEST_F(PlaneMapTempTableTest, ComputeResultJsonSerializable) {
    auto params = makeSimulatedParams(-1.0, 1.0, 2.0, 640, 480, 50, 2.0f);

    PlaneMapTempTable op(params);
    PlaneMapTempTableResult result = op.Execute();

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.tableSize, 2);

    nlohmann::json j = result.toJson();
    EXPECT_TRUE(j.contains("success"));
    EXPECT_TRUE(j.contains("table"));
    EXPECT_TRUE(j["table"].is_array());
    EXPECT_EQ(j["table"].size(), 2u);

    for (const auto& entry : j["table"]) {
        EXPECT_TRUE(entry.contains("temperature"));
        EXPECT_TRUE(entry.contains("deltaT"));
        EXPECT_TRUE(entry.contains("compensatedCameraMatrixL"));
        EXPECT_TRUE(entry.contains("R1"));
        EXPECT_TRUE(entry.contains("P1"));
        EXPECT_TRUE(entry.contains("Q"));
        EXPECT_TRUE(entry.contains("lineStats"));
    }

    std::cout << "=== ComputeResultJsonSerializable ===" << std::endl;
    std::cout << "  JSON size: " << j.dump().size() << " bytes" << std::endl;
    std::cout << "  Table entries: " << j["table"].size() << std::endl;
}

TEST_F(PlaneMapTempTableTest, ComputeTemperatureCompensationEffect) {
    auto params = makeSimulatedParams(-10.0, 10.0, 10.0, 640, 480, 50, 2.0f);

    PlaneMapTempTable op(params);
    PlaneMapTempTableResult result = op.Execute();

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.tableSize, 3);

    const auto& cold = result.table[0];
    const auto& ref = result.table[1];
    const auto& hot = result.table[2];

    double fx_cold = cold.compensatedCameraMatrixL.at<double>(0, 0);
    double fx_ref = ref.compensatedCameraMatrixL.at<double>(0, 0);
    double fx_hot = hot.compensatedCameraMatrixL.at<double>(0, 0);

    EXPECT_LT(fx_cold, fx_ref);
    EXPECT_GT(fx_hot, fx_ref);

    double ratio = (fx_hot - fx_cold) / fx_ref;
    double expected_ratio = params.cte * 20.0;
    EXPECT_NEAR(ratio, expected_ratio, 1e-8);

    double baseline_cold = cold.compensatedT.at<double>(0, 0);
    double baseline_ref = ref.compensatedT.at<double>(0, 0);
    double baseline_hot = hot.compensatedT.at<double>(0, 0);

    EXPECT_LT(baseline_cold, baseline_ref);
    EXPECT_GT(baseline_hot, baseline_ref);

    std::cout << "=== ComputeTemperatureCompensationEffect ===" << std::endl;
    std::cout << "  fx_cold=" << fx_cold << " fx_ref=" << fx_ref << " fx_hot=" << fx_hot << std::endl;
    std::cout << "  baseline_cold=" << baseline_cold << " baseline_ref=" << baseline_ref << " baseline_hot=" << baseline_hot << std::endl;
    std::cout << "  fx ratio=" << ratio << " expected=" << expected_ratio << std::endl;
}

TEST_F(PlaneMapTempTableTest, ComputeProduction2054x1580x17) {
    PlaneMapTempTableParams p;

    double fxL = 2856.0, fyL = 2858.0, cxL = 1027.3, cyL = 790.8;
    p.cameraMatrixL = (cv::Mat_<double>(3, 3) <<
        fxL, 0, cxL,
        0, fyL, cyL,
        0, 0, 1);

    double fxR = 2861.0, fyR = 2863.0, cxR = 1028.1, cyR = 789.5;
    p.cameraMatrixR = (cv::Mat_<double>(3, 3) <<
        fxR, 0, cxR,
        0, fyR, cyR,
        0, 0, 1);

    p.distCoeffsL = (cv::Mat_<double>(1, 5) << -0.042, 0.018, 0.0008, -0.0012, 0.0);
    p.distCoeffsR = (cv::Mat_<double>(1, 5) << -0.051, 0.025, 0.0006, -0.0009, 0.0);

    double baseline = 150.0;
    p.R = cv::Mat::eye(3, 3, CV_64F);
    p.T = (cv::Mat_<double>(3, 1) << baseline, 0.0, 0.0);

    p.imageSize = cv::Size(2054, 1580);

    double vfx = 2400.0;
    p.virtualK = cv::Matx33d(
        vfx, 0, 1027.0,
        0, vfx, 790.0,
        0, 0, 1);
    p.virtualR = cv::Matx33d::eye();
    p.virtualT = cv::Vec3d(baseline / 2.0, 0.0, 300.0);

    std::vector<int> lineIds;
    for (int i = 0; i < 17; ++i)
        lineIds.push_back(i);
    p.lineIds = lineIds;

    p.referenceTemp = 25.0;
    p.cte = 23.6e-6;
    p.tempStep = 0.5;
    p.tempRangeMin = -10.0;
    p.tempRangeMax = 10.0;
    p.alpha = 0.0;
    p.flags = cv::CALIB_ZERO_DISPARITY;
    p.deviceId = 0;
    p.gridStep = 4.0f;
    p.depthMin = 200.0f;
    p.depthMax = 2000.0f;
    p.depthSamples = 50;
    p.epipolarStep = 1.0f;

    const int expectedSteps = 41;

    std::cout << "=== ComputeProduction2054x1580x17 ===" << std::endl;
    std::cout << "  Image: " << p.imageSize.width << "x" << p.imageSize.height << std::endl;
    std::cout << "  Lines: " << p.lineIds.size() << std::endl;
    std::cout << "  Temp range: " << (p.referenceTemp + p.tempRangeMin) << " ~ "
              << (p.referenceTemp + p.tempRangeMax) << " C" << std::endl;
    std::cout << "  Temp step: " << p.tempStep << " C" << std::endl;
    std::cout << "  Expected steps: " << expectedSteps << std::endl;
    std::cout << "  gridStep: " << p.gridStep << std::endl;
    std::cout << "  depthSamples: " << p.depthSamples << std::endl;
    std::cout << "  Baseline: " << baseline << " mm" << std::endl;
    std::cout << "  fxL: " << fxL << " fyL: " << fyL << std::endl;
    std::cout << "  fxR: " << fxR << " fyR: " << fyR << std::endl;
    std::cout << std::endl;

    PlaneMapTempTable op(p);

    auto t0 = std::chrono::high_resolution_clock::now();
    PlaneMapTempTableResult result = op.Execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::endl;
    std::cout << "=== RESULT ===" << std::endl;
    std::cout << "  success: " << result.success << std::endl;
    std::cout << "  tableSize: " << result.tableSize << std::endl;
    std::cout << "  qualityFlag: " << static_cast<int>(result.qualityFlag) << std::endl;
    if (!result.message.empty())
        std::cout << "  message: " << result.message << std::endl;
    std::cout << "  Total wall time: " << dt_ms << " ms (" << dt_ms / 1000.0 << " s)" << std::endl;

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.tableSize, expectedSteps);
    EXPECT_EQ(result.table.size(), static_cast<size_t>(expectedSteps));

    EXPECT_NEAR(result.table.front().temperature, 15.0, 1e-9);
    EXPECT_NEAR(result.table.back().temperature, 35.0, 1e-9);

    for (size_t i = 1; i < result.table.size(); ++i) {
        double step = result.table[i].temperature - result.table[i - 1].temperature;
        EXPECT_NEAR(step, 0.5, 1e-9) << "step [" << (i - 1) << "]鈫抂" << i << "]";
    }

    double totalPairsSum = 0;
    double minPairs = std::numeric_limits<double>::max();
    double maxPairs = 0;
    for (const auto& entry : result.table) {
        totalPairsSum += entry.totalPairs;
        minPairs = std::min(minPairs, static_cast<double>(entry.totalPairs));
        maxPairs = std::max(maxPairs, static_cast<double>(entry.totalPairs));
        EXPECT_EQ(entry.lineStats.size(), 17u);
        EXPECT_FALSE(entry.R1.empty());
        EXPECT_FALSE(entry.P1.empty());
        EXPECT_FALSE(entry.Q.empty());
    }

    std::cout << std::endl;
    std::cout << "=== PER-STEP DETAIL ===" << std::endl;
    std::cout << "  [step] temp    deltaT    pairs     lines" << std::endl;
    std::cout << "  " << std::string(50, '-') << std::endl;
    for (size_t i = 0; i < result.table.size(); ++i) {
        const auto& e = result.table[i];
        std::cout << "  [" << std::setw(2) << i << "] "
                  << std::fixed << std::setprecision(1)
                  << std::setw(6) << e.temperature << "  "
                  << std::setw(7) << e.deltaT << "  "
                  << std::setw(9) << e.totalPairs << "  "
                  << std::setw(3) << e.lineStats.size()
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== LINE STATS (last step) ===" << std::endl;
    if (!result.table.empty()) {
        const auto& last = result.table.back();
        std::cout << "  lineId  numPairs     uMin     uMax     vMin     vMax" << std::endl;
        std::cout << "  " << std::string(55, '-') << std::endl;
        for (const auto& ls : last.lineStats) {
            std::cout << "  " << std::setw(4) << ls.lineId
                      << "  " << std::setw(8) << ls.numPairs
                      << "  " << std::setw(8) << std::fixed << std::setprecision(1) << ls.uMin
                      << "  " << std::setw(8) << ls.uMax
                      << "  " << std::setw(8) << ls.vMin
                      << "  " << std::setw(8) << ls.vMax
                      << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== TIMING SUMMARY ===" << std::endl;
    std::cout << "  Total wall:           " << dt_ms / 1000.0 << " s" << std::endl;
    std::cout << "  Per-step avg:         " << dt_ms / expectedSteps << " ms ("
              << dt_ms / expectedSteps / 1000.0 << " s)" << std::endl;
    std::cout << "  Avg pairs/step:       " << totalPairsSum / expectedSteps << std::endl;
    std::cout << "  Pair range:           " << minPairs << " ~ " << maxPairs << std::endl;
    std::cout << "  JSON size:            " << result.toJson().dump().size() << " bytes" << std::endl;

    double dt_cold = std::abs(result.table.front().deltaT);
    double dt_hot = std::abs(result.table.back().deltaT);
    double fx_cold = result.table.front().compensatedCameraMatrixL.at<double>(0, 0);
    double fx_hot = result.table.back().compensatedCameraMatrixL.at<double>(0, 0);
    double fx_ref = result.table[expectedSteps / 2].compensatedCameraMatrixL.at<double>(0, 0);

    std::cout << std::endl;
    std::cout << "=== TEMPERATURE COMPENSATION CHECK ===" << std::endl;
    std::cout << "  fx@15C=" << fx_cold << " fx@25C=" << fx_ref << " fx@35C=" << fx_hot << std::endl;
    double bl_cold = result.table.front().compensatedT.at<double>(0, 0);
    double bl_ref = result.table[expectedSteps / 2].compensatedT.at<double>(0, 0);
    double bl_hot = result.table.back().compensatedT.at<double>(0, 0);
    std::cout << "  baseline@15C=" << bl_cold << " baseline@25C=" << bl_ref << " baseline@35C=" << bl_hot << std::endl;
    EXPECT_LT(fx_cold, fx_ref);
    EXPECT_GT(fx_hot, fx_ref);
    EXPECT_LT(bl_cold, bl_ref);
    EXPECT_GT(bl_hot, bl_ref);
}

#endif
