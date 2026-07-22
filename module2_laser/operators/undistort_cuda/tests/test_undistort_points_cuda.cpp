/**
 * @file test_undistort_points_cuda.cpp
 * @brief 激光中心亚像素点集去畸变+立体矫正算子 - 单元测试
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <vector>

#include "../undistort_points_cuda.h"
#include "common/calib_warmup_config.h"

using namespace calib;


// ============================================================================
// Helper: 构建测试用标定参数
// ============================================================================

static UndistortPointsParams makeIdentityParams() {
    UndistortPointsParams p;
    p.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        1000.0, 0.0, 640.0,
        0.0, 1000.0, 480.0,
        0.0, 0.0, 1.0);
    p.distCoeffs = (cv::Mat_<double>(1, 8) <<
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    p.R = (cv::Mat_<double>(3, 3) <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
    p.P = (cv::Mat_<double>(3, 4) <<
        1000.0, 0.0, 640.0, 0.0,
        0.0, 1000.0, 480.0, 0.0,
        0.0, 0.0, 1.0, 0.0);
    return p;
}

static UndistortPointsParams makeDistortedParams() {
    UndistortPointsParams p;
    p.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, 320.0,
        0.0, 800.0, 240.0,
        0.0, 0.0, 1.0);
    p.distCoeffs = (cv::Mat_<double>(1, 8) <<
        -0.1, 0.01, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0);
    p.R = (cv::Mat_<double>(3, 3) <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
    p.P = (cv::Mat_<double>(3, 4) <<
        800.0, 0.0, 320.0, 0.0,
        0.0, 800.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0);
    return p;
}

// ============================================================================
// 测试夹具
// ============================================================================

class UndistortPointsTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = makeIdentityParams();
    }

    UndistortPointsParams params_;
};

// ============================================================================
// 参数校验测试 (CPU-only)
// ============================================================================

TEST_F(UndistortPointsTest, DefaultParamsMissingMatrixThrows) {
    UndistortPointsParams p;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, IdentityParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(UndistortPointsTest, EmptyCameraMatrixThrows) {
    params_.cameraMatrix = cv::Mat();
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, WrongCameraMatrixSizeThrows) {
    params_.cameraMatrix = cv::Mat::eye(4, 4, CV_64F);
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, EmptyDistCoeffsThrows) {
    params_.distCoeffs = cv::Mat();
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, WrongDistCoeffsCountThrows) {
    params_.distCoeffs = (cv::Mat_<double>(1, 3) << 0.0, 0.0, 0.0);
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, WrongRSizeThrows) {
    params_.R = cv::Mat::eye(2, 2, CV_64F);
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, WrongPSizeThrows) {
    params_.P = cv::Mat::eye(3, 3, CV_64F);
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, NegativeDeviceIdThrows) {
    params_.deviceId = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(UndistortPointsTest, EmptyRAndPAreValid) {
    params_.R = cv::Mat();
    params_.P = cv::Mat();
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(UndistortPointsTest, FourDistCoeffsIsValid) {
    params_.distCoeffs = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 0.0);
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(UndistortPointsTest, EightDistCoeffsIsValid) {
    params_.distCoeffs = (cv::Mat_<double>(1, 8) <<
        -0.1, 0.01, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0);
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(UndistortPointsTest, CV32FCameraMatrixIsValid) {
    params_.cameraMatrix = (cv::Mat_<float>(3, 3) <<
        1000.0f, 0.0f, 640.0f,
        0.0f, 1000.0f, 480.0f,
        0.0f, 0.0f, 1.0f);
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================================
// JSON 序列化测试 (CPU-only)
// ============================================================================

TEST_F(UndistortPointsTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = UndistortPointsParams::fromJson(j);
    EXPECT_EQ(restored.deviceId, params_.deviceId);
    EXPECT_NEAR(restored.cameraMatrix.at<double>(0, 0),
                params_.cameraMatrix.at<double>(0, 0), 1e-9);
}

TEST_F(UndistortPointsTest, JsonPartialDeserialization) {
    nlohmann::json j;
    j["cameraMatrix"] = std::vector<double>{1000,0,640, 0,1000,480, 0,0,1};
    j["distCoeffs"] = std::vector<double>(8, 0.0);
    EXPECT_NO_THROW(UndistortPointsParams::fromJson(j));
}

TEST_F(UndistortPointsTest, JsonUnknownFieldsIgnored) {
    auto j = params_.toJson();
    j["unknownField"] = 999;
    EXPECT_NO_THROW(UndistortPointsParams::fromJson(j));
}

// ============================================================================
// Result 结构体测试 (CPU-only)
// ============================================================================

TEST_F(UndistortPointsTest, ResultDefaultValues) {
    UndistortPointsResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.d_rectifiedPoints, nullptr);
}

TEST_F(UndistortPointsTest, ResultMoveSemantics) {
    UndistortPointsResult result1;
    result1.success = true;
    result1.message = "test";

    UndistortPointsResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
}

// ============================================================================
// WarmupConfig 测试 (CPU-only)
// ============================================================================

TEST_F(UndistortPointsTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================================
// CUDA 测试（需要 GPU）
// ============================================================================
#ifdef WITH_CUDA_TESTS

TEST_F(UndistortPointsTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        UndistortPointsCuda undistorter(params_);
    });
}

TEST_F(UndistortPointsTest, ConstructWithInvalidParamsThrows) {
    UndistortPointsParams badParams;
    EXPECT_THROW({
        UndistortPointsCuda undistorter(badParams);
    }, std::invalid_argument);
}

TEST_F(UndistortPointsTest, WarmupBasic) {
    UndistortPointsCuda undistorter(params_);
    EXPECT_NO_THROW(undistorter.Warmup(10000));
}

TEST_F(UndistortPointsTest, WarmupWithConfig) {
    UndistortPointsCuda undistorter(params_);
    auto config = calib::WarmupConfig::forPointCloud(10000);
    EXPECT_NO_THROW(undistorter.Warmup(config));
}

TEST_F(UndistortPointsTest, ProcessEmptyInputReturnsSuccess) {
    UndistortPointsCuda undistorter(params_);
    cv::cuda::GpuMat d_empty;
    auto result = undistorter.Execute(d_empty);
    EXPECT_TRUE(result.success);
}

TEST_F(UndistortPointsTest, ProcessWrongTypeReturnsError) {
    UndistortPointsCuda undistorter(params_);
    cv::cuda::GpuMat d_wrong(1, 10, CV_32FC1);
    auto result = undistorter.Execute(d_wrong);
    EXPECT_FALSE(result.success);
}

TEST_F(UndistortPointsTest, ProcessIdentityNoDistortion) {
    UndistortPointsCuda undistorter(params_);
    undistorter.Warmup(10);

    std::vector<cv::Point2f> points = {
        {100.0f, 200.0f}, {640.0f, 480.0f}, {320.0f, 240.0f}
    };
    cv::Mat h_points(points, true);
    h_points = h_points.reshape(2, 1);

    cv::cuda::GpuMat d_points;
    d_points.upload(h_points);

    auto result = undistorter.Execute(d_points);

    EXPECT_TRUE(result.success);
    ASSERT_NE(result.d_rectifiedPoints, nullptr);

    cv::Mat h_output;
    result.d_rectifiedPoints->download(h_output);
    h_output = h_output.reshape(2, 1);

    for (int i = 0; i < 3; ++i) {
        cv::Point2f outPt = h_output.at<cv::Point2f>(0, i);
        EXPECT_NEAR(outPt.x, points[i].x, 0.1f);
        EXPECT_NEAR(outPt.y, points[i].y, 0.1f);
    }
}

TEST_F(UndistortPointsTest, ProcessWithDistortionMatchesOpenCV) {
    auto distParams = makeDistortedParams();
    UndistortPointsCuda undistorter(distParams);
    undistorter.Warmup(10);

    std::vector<cv::Point2f> points = {
        {100.0f, 100.0f}, {200.0f, 150.0f}, {300.0f, 200.0f},
        {400.0f, 250.0f}, {500.0f, 300.0f}
    };
    cv::Mat h_points(points, true);
    h_points = h_points.reshape(2, 1);

    cv::cuda::GpuMat d_points;
    d_points.upload(h_points);

    auto result = undistorter.Execute(d_points);

    EXPECT_TRUE(result.success);

    cv::Mat h_opencv;
    cv::undistortPoints(h_points, h_opencv,
                        distParams.cameraMatrix, distParams.distCoeffs,
                        distParams.R, distParams.P);
    h_opencv = h_opencv.reshape(2, 1);

    cv::Mat h_cuda;
    result.d_rectifiedPoints->download(h_cuda);
    h_cuda = h_cuda.reshape(2, 1);

    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        cv::Point2f opencvPt = h_opencv.at<cv::Point2f>(0, i);
        cv::Point2f cudaPt = h_cuda.at<cv::Point2f>(0, i);
        EXPECT_NEAR(cudaPt.x, opencvPt.x, 0.01f)
            << "Point " << i << " x mismatch";
        EXPECT_NEAR(cudaPt.y, opencvPt.y, 0.01f)
            << "Point " << i << " y mismatch";
    }
}

TEST_F(UndistortPointsTest, ProcessLargeBatch) {
    UndistortPointsCuda undistorter(params_);

    int N = 1000000;
    cv::Mat h_points(1, N, CV_32FC2);
    auto* ptr = h_points.ptr<cv::Point2f>();
    for (int i = 0; i < N; ++i) {
        ptr[i] = cv::Point2f(
            static_cast<float>(i % 1280),
            static_cast<float>(i % 960));
    }

    cv::cuda::GpuMat d_points;
    d_points.upload(h_points);

    auto result = undistorter.Execute(d_points);

    EXPECT_TRUE(result.success);
    ASSERT_NE(result.d_rectifiedPoints, nullptr);
    EXPECT_EQ(result.d_rectifiedPoints->cols, N);
}

TEST_F(UndistortPointsTest, SetParamsAndGetParams) {
    UndistortPointsCuda undistorter(params_);

    auto newParams = makeDistortedParams();
    undistorter.SetParams(newParams);

    const auto& current = undistorter.GetParams();
    EXPECT_EQ(current.deviceId, newParams.deviceId);
}

TEST_F(UndistortPointsTest, SetInvalidParamsThrows) {
    UndistortPointsCuda undistorter(params_);
    UndistortPointsParams badParams;
    EXPECT_THROW(undistorter.SetParams(badParams), std::invalid_argument);
}

TEST_F(UndistortPointsTest, ProcessWithStream) {
    UndistortPointsCuda undistorter(params_);
    undistorter.Warmup(10);

    std::vector<cv::Point2f> points = {{320.0f, 240.0f}};
    cv::Mat h_points(points, true);
    h_points = h_points.reshape(2, 1);

    cv::cuda::GpuMat d_points;
    d_points.upload(h_points);

    cv::cuda::Stream stream;
    auto result = undistorter.Execute(d_points, stream);

    EXPECT_TRUE(result.success);
}

TEST_F(UndistortPointsTest, ProcessWithoutWarmupStillWorks) {
    UndistortPointsCuda undistorter(params_);

    std::vector<cv::Point2f> points = {{320.0f, 240.0f}};
    cv::Mat h_points(points, true);
    h_points = h_points.reshape(2, 1);

    cv::cuda::GpuMat d_points;
    d_points.upload(h_points);

    auto result = undistorter.Execute(d_points);

    EXPECT_TRUE(result.success);
}

#endif // WITH_CUDA_TESTS
