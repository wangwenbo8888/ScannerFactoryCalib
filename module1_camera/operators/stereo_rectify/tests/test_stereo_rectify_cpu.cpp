#include <gtest/gtest.h>
#include "stereo_rectify_cpu.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

using namespace calib;

namespace {

StereoRectifyCpuParams makeSyntheticParams(double alpha = 0.0, int flags = cv::CALIB_ZERO_DISPARITY) {
    StereoRectifyCpuParams p;

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;

    p.cameraMatrixL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    p.cameraMatrixR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    p.distCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    p.distCoeffsR = cv::Mat::zeros(1, 5, CV_64F);
    p.imageSize = cv::Size(2048, 1536);

    p.R = cv::Mat::eye(3, 3, CV_64F);
    p.T = (cv::Mat_<double>(3, 1) << 100.0, 0, 0);

    p.alpha = alpha;
    p.flags = flags;

    return p;
}

} // anonymous namespace

// ============= Params Validation Tests =============

TEST(StereoRectifyCpuParamsTest, ValidateRejectsEmptyCameraMatrixL) {
    auto p = makeSyntheticParams();
    p.cameraMatrixL = cv::Mat();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsTest, ValidateRejectsEmptyR) {
    auto p = makeSyntheticParams();
    p.R = cv::Mat();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsTest, ValidateRejectsEmptyT) {
    auto p = makeSyntheticParams();
    p.T = cv::Mat();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsTest, ValidateRejectsInvalidImageSize) {
    auto p = makeSyntheticParams();
    p.imageSize = cv::Size(0, 1536);
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsTest, ValidateRejectsAlphaOutOfRange) {
    auto p = makeSyntheticParams();
    p.alpha = -0.5;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsTest, ValidateRejectsInvalidFlags) {
    auto p = makeSyntheticParams();
    p.flags = 999;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsTest, ValidParamsPassValidation) {
    auto p = makeSyntheticParams();
    EXPECT_NO_THROW(p.validate());
}

// ============= Params Serialization Tests =============

TEST(StereoRectifyCpuParamsSerializationTest, JsonRoundTrip) {
    auto p = makeSyntheticParams();
    auto j = p.toJson();
    auto p2 = StereoRectifyCpuParams::fromJson(j);

    EXPECT_EQ(p.imageSize.width, p2.imageSize.width);
    EXPECT_EQ(p.imageSize.height, p2.imageSize.height);
    EXPECT_DOUBLE_EQ(p.alpha, p2.alpha);
    EXPECT_EQ(p.flags, p2.flags);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(p.cameraMatrixL.at<double>(r, c),
                             p2.cameraMatrixL.at<double>(r, c));
            EXPECT_DOUBLE_EQ(p.R.at<double>(r, c),
                             p2.R.at<double>(r, c));
        }
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(p.T.at<double>(i, 0), p2.T.at<double>(i, 0));
    }
}

TEST(StereoRectifyCpuParamsSerializationTest, FromJsonMissingFieldThrows) {
    nlohmann::json j;
    EXPECT_THROW(StereoRectifyCpuParams::fromJson(j), std::invalid_argument);
}

TEST(StereoRectifyCpuParamsSerializationTest, FromJsonUnknownFieldsIgnored) {
    auto p = makeSyntheticParams();
    auto j = p.toJson();
    j["unknownField123"] = 42;
    auto p2 = StereoRectifyCpuParams::fromJson(j);
    EXPECT_EQ(p.imageSize.width, p2.imageSize.width);
}

// ============= Rectify Tests =============

TEST(StereoRectifyCpuTest, RectifySuccess) {
    auto p = makeSyntheticParams();
    StereoRectifyCpu rect(p);

    auto result = rect.Execute();

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.R1.empty());
    EXPECT_FALSE(result.R2.empty());
    EXPECT_FALSE(result.P1.empty());
    EXPECT_FALSE(result.P2.empty());
    EXPECT_FALSE(result.Q.empty());
    EXPECT_GT(result.validRoiLeft.area(), 0);
    EXPECT_GT(result.validRoiRight.area(), 0);
}

TEST(StereoRectifyCpuTest, RectifyProducesCorrectSizes) {
    auto p = makeSyntheticParams();
    StereoRectifyCpu rect(p);

    auto result = rect.Execute();
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.R1.rows, 3);
    EXPECT_EQ(result.R1.cols, 3);
    EXPECT_EQ(result.R2.rows, 3);
    EXPECT_EQ(result.R2.cols, 3);
    EXPECT_EQ(result.P1.rows, 3);
    EXPECT_EQ(result.P1.cols, 4);
    EXPECT_EQ(result.P2.rows, 3);
    EXPECT_EQ(result.P2.cols, 4);
    EXPECT_EQ(result.Q.rows, 4);
    EXPECT_EQ(result.Q.cols, 4);
}

TEST(StereoRectifyCpuTest, RectifyWithAlphaOne) {
    auto p = makeSyntheticParams(1.0);
    StereoRectifyCpu rect(p);

    auto result = rect.Execute();

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.R1.empty());
    EXPECT_FALSE(result.Q.empty());
}

TEST(StereoRectifyCpuTest, RectifyWithZeroDisparity) {
    auto p = makeSyntheticParams(0.0, cv::CALIB_ZERO_DISPARITY);
    StereoRectifyCpu rect(p);

    auto result = rect.Execute();

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.Q.empty());
}

// ============= Constructor / setParams Tests =============

TEST(StereoRectifyCpuTest, ConstructorInvalidThrows) {
    auto p = makeSyntheticParams();
    p.R = cv::Mat();
    EXPECT_THROW(StereoRectifyCpu rect(p), std::invalid_argument);
}

TEST(StereoRectifyCpuTest, SetParamsValid) {
    auto p = makeSyntheticParams();
    StereoRectifyCpu rect(p);

    auto p2 = makeSyntheticParams();
    p2.alpha = 0.5;
    EXPECT_NO_THROW(rect.SetParams(p2));
    EXPECT_DOUBLE_EQ(rect.GetParams().alpha, 0.5);
}

TEST(StereoRectifyCpuTest, SetParamsInvalidThrows) {
    auto p = makeSyntheticParams();
    StereoRectifyCpu rect(p);

    auto p2 = makeSyntheticParams();
    p2.cameraMatrixL = cv::Mat();
    EXPECT_THROW(rect.SetParams(p2), std::invalid_argument);
    EXPECT_DOUBLE_EQ(rect.GetParams().alpha, p.alpha);
}

// ============= QualityFlag toString Tests =============

TEST(StereoRectifyCpuTest, QualityFlagToString) {
    EXPECT_STREQ(qualityFlagToString(QualityFlag::Normal), "Normal");
    EXPECT_STREQ(qualityFlagToString(QualityFlag::Warning), "Warning");
}

// ============= StereoRectifyCpuResult JSON Serialization Tests =============

TEST(StereoRectifyCpuResultJsonTest, RoundTrip) {
    StereoRectifyCpuResult original;
    original.success = true;
    original.message = "test";
    original.qualityFlag = QualityFlag::Normal;
    original.R1 = cv::Mat::eye(3, 3, CV_64F);
    original.R2 = cv::Mat::eye(3, 3, CV_64F);
    original.P1 = cv::Mat::eye(3, 4, CV_64F);
    original.P2 = cv::Mat::eye(3, 4, CV_64F);
    original.Q = cv::Mat::eye(4, 4, CV_64F);
    original.validRoiLeft = cv::Rect(0, 0, 2048, 1536);
    original.validRoiRight = cv::Rect(1, 1, 2046, 1534);

    auto j = original.toJson();
    auto loaded = StereoRectifyCpuResult::fromJson(j);

    EXPECT_EQ(loaded.success, true);
    EXPECT_EQ(loaded.message, "test");
    EXPECT_EQ(loaded.qualityFlag, QualityFlag::Normal);
    EXPECT_DOUBLE_EQ(loaded.R1.at<double>(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(loaded.P1.at<double>(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(loaded.Q.at<double>(0, 0), 1.0);
    EXPECT_EQ(loaded.validRoiLeft.x, 0);
    EXPECT_EQ(loaded.validRoiLeft.width, 2048);
    EXPECT_EQ(loaded.validRoiRight.x, 1);
    EXPECT_EQ(loaded.validRoiRight.width, 2046);
}

TEST(StereoRectifyCpuResultJsonTest, RoundTripFromRectify) {
    auto p = makeSyntheticParams();
    StereoRectifyCpu rect(p);
    auto result = rect.Execute();
    ASSERT_TRUE(result.success);

    auto j = result.toJson();
    auto loaded = StereoRectifyCpuResult::fromJson(j);

    EXPECT_EQ(loaded.success, result.success);
    EXPECT_EQ(loaded.validRoiLeft.x, result.validRoiLeft.x);
    EXPECT_EQ(loaded.validRoiLeft.y, result.validRoiLeft.y);
    EXPECT_EQ(loaded.validRoiLeft.width, result.validRoiLeft.width);
    EXPECT_EQ(loaded.validRoiLeft.height, result.validRoiLeft.height);
    EXPECT_EQ(loaded.validRoiRight.x, result.validRoiRight.x);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(loaded.R1.at<double>(r, c), result.R1.at<double>(r, c), 1e-10);
            EXPECT_NEAR(loaded.R2.at<double>(r, c), result.R2.at<double>(r, c), 1e-10);
        }
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(loaded.P1.at<double>(r, c), result.P1.at<double>(r, c), 1e-10);
            EXPECT_NEAR(loaded.P2.at<double>(r, c), result.P2.at<double>(r, c), 1e-10);
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(loaded.Q.at<double>(r, c), result.Q.at<double>(r, c), 1e-10);
        }
    }
}
