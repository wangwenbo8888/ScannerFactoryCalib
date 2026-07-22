#include <gtest/gtest.h>
#include "extrinsic_calib_cpu.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <random>

using namespace calib;

namespace {

ExtrinsicCalibCpuParams makeSyntheticParams(int numViews = 10, int numPoints = 88,
                                              bool calibrateMono = false) {
    ExtrinsicCalibCpuParams p;
    p.imageSize = cv::Size(2048, 1536);
    p.flags = 0;
    p.calibrateMono = calibrateMono;
    p.patternSize = cv::Size(11, 8);
    p.squareSize = 15.0f;
    p.maxReprojError = 1.0;
    p.minViewCount = 8;
    p.rotateRightImage180 = false;
    p.maxEpipolarError = 0.05;

    for (int i = 0; i < numPoints; ++i) {
        int row = i / 11;
        int col = i % 11;
        p.objectPoints.emplace_back(col * 15.0f, row * 15.0f, 0.0f);
    }

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    double baseline = 100.0;

    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    cv::Mat R_stereo = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T_stereo = (cv::Mat_<double>(3, 1) << baseline, 0, 0);

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);

    for (int v = 0; v < numViews; ++v) {
        double tx = 150.0 + (v % 5) * 50.0;
        double ty = 100.0 + (v % 3) * 50.0;
        double tz = 300.0 + v * 10.0;
        double angle = 0.02 * (v - numViews / 2);

        cv::Mat rvec = (cv::Mat_<double>(3, 1) << angle, angle * 0.5, 0);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << tx, ty, tz);

        std::vector<cv::Point2f> leftPts, rightPts;
        for (const auto& objPt : p.objectPoints) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(std::vector<cv::Point3f>{objPt}, rvec, tvec, KL, distL, projected);
            leftPts.emplace_back(static_cast<float>(projected[0].x + noise(rng)),
                                  static_cast<float>(projected[0].y + noise(rng)));

            cv::Mat R_left;
            cv::Rodrigues(rvec, R_left);
            cv::Mat R_right = R_stereo * R_left;
            cv::Mat t_right = R_stereo * tvec + T_stereo;
            cv::Mat rvec_right;
            cv::Rodrigues(R_right, rvec_right);

            cv::projectPoints(std::vector<cv::Point3f>{objPt}, rvec_right, t_right, KR, distR, projected);
            rightPts.emplace_back(static_cast<float>(projected[0].x + noise(rng)),
                                   static_cast<float>(projected[0].y + noise(rng)));
        }
        p.leftPointsPerView.push_back(std::move(leftPts));
        p.rightPointsPerView.push_back(std::move(rightPts));
    }

    return p;
}

} // anonymous namespace

// ============= Params Validation Tests =============

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsEmptyLeftPoints) {
    auto p = makeSyntheticParams();
    p.leftPointsPerView.clear();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsEmptyRightPoints) {
    auto p = makeSyntheticParams();
    p.rightPointsPerView.clear();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsViewCountMismatch) {
    auto p = makeSyntheticParams();
    p.rightPointsPerView.push_back(p.rightPointsPerView.back());
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsInsufficientViews) {
    auto p = makeSyntheticParams(3);
    p.minViewCount = 8;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsEmptySubView) {
    auto p = makeSyntheticParams();
    p.leftPointsPerView[3].clear();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsPointCountMismatch) {
    auto p = makeSyntheticParams();
    p.leftPointsPerView[0].push_back({0, 0});
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsInvalidImageSize) {
    auto p = makeSyntheticParams();
    p.imageSize = cv::Size(0, 1536);
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsMinViewCountBelow3) {
    auto p = makeSyntheticParams();
    p.minViewCount = 1;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidateRejectsMonoWithoutPatternSize) {
    auto p = makeSyntheticParams();
    p.calibrateMono = true;
    p.patternSize = cv::Size(0, 0);
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsTest, ValidParamsPassValidation) {
    auto p = makeSyntheticParams();
    EXPECT_NO_THROW(p.validate());
}

// ============= Params Serialization Tests =============

TEST(ExtrinsicCalibCpuParamsSerializationTest, JsonRoundTrip) {
    auto p = makeSyntheticParams();
    auto j = p.toJson();
    auto p2 = ExtrinsicCalibCpuParams::fromJson(j);
    EXPECT_EQ(p.leftPointsPerView.size(), p2.leftPointsPerView.size());
    EXPECT_EQ(p.rightPointsPerView.size(), p2.rightPointsPerView.size());
    EXPECT_EQ(p.objectPoints.size(), p2.objectPoints.size());
    EXPECT_EQ(p.imageSize.width, p2.imageSize.width);
    EXPECT_EQ(p.imageSize.height, p2.imageSize.height);
    EXPECT_EQ(p.flags, p2.flags);
    EXPECT_EQ(p.calibrateMono, p2.calibrateMono);
    EXPECT_DOUBLE_EQ(p.maxReprojError, p2.maxReprojError);
    EXPECT_EQ(p.minViewCount, p2.minViewCount);
    EXPECT_EQ(p.rotateRightImage180, p2.rotateRightImage180);
    EXPECT_DOUBLE_EQ(p.maxEpipolarError, p2.maxEpipolarError);
}

TEST(ExtrinsicCalibCpuParamsSerializationTest, FromJsonMissingFieldThrows) {
    nlohmann::json j;
    EXPECT_THROW(ExtrinsicCalibCpuParams::fromJson(j), std::invalid_argument);
}

TEST(ExtrinsicCalibCpuParamsSerializationTest, FromJsonUnknownFieldsIgnored) {
    auto p = makeSyntheticParams();
    auto j = p.toJson();
    j["unknownField123"] = 42;
    auto p2 = ExtrinsicCalibCpuParams::fromJson(j);
    EXPECT_EQ(p.leftPointsPerView.size(), p2.leftPointsPerView.size());
}

// ============= Stereo Calibration Tests (Flow 1) =============

TEST(ExtrinsicCalibCpuTest, CalibrateStereoSuccess) {
    auto p = makeSyntheticParams(10);
    ExtrinsicCalibCpu cal(p);

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    auto result = cal.Execute(KL, distL, KR, distR);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.R.empty());
    EXPECT_FALSE(result.T.empty());
    EXPECT_GT(result.stereoReprojError, 0.0);
    EXPECT_FALSE(result.F.empty());
    EXPECT_GT(result.epipolarErrorMean, 0.0);
    EXPECT_EQ(result.perViewEpipolarErrors.size(), 10u);
}

TEST(ExtrinsicCalibCpuTest, CalibrateStereoMinimalViews) {
    auto p = makeSyntheticParams(8);
    p.minViewCount = 8;
    ExtrinsicCalibCpu cal(p);

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    auto result = cal.Execute(KL, distL, KR, distR);
    EXPECT_TRUE(result.success);
}

// ============= Epipolar Error Tests =============

TEST(ExtrinsicCalibCpuTest, EpipolarErrorComputedCorrectly) {
    auto p = makeSyntheticParams(10);
    ExtrinsicCalibCpu cal(p);

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    auto result = cal.Execute(KL, distL, KR, distR);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.epipolarErrorMean, 0.0);
    EXPECT_GE(result.epipolarErrorStd, 0.0);
    EXPECT_EQ(result.perViewEpipolarErrors.size(), p.leftPointsPerView.size());

    for (const auto& err : result.perViewEpipolarErrors) {
        EXPECT_GT(err, 0.0);
    }
}

TEST(ExtrinsicCalibCpuTest, EpipolarErrorQualityDegraded) {
    auto p = makeSyntheticParams(10);
    p.maxEpipolarError = 1e-6;
    ExtrinsicCalibCpu cal(p);

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    auto result = cal.Execute(KL, distL, KR, distR);
    ASSERT_TRUE(result.success);
    EXPECT_NE(result.qualityFlag, QualityFlag::Normal);
}

// ============= Full Calibration Tests (Flow 2) =============

TEST(ExtrinsicCalibCpuTest, CalibrateFullSuccess) {
    auto p = makeSyntheticParams(10, 88, true);
    ExtrinsicCalibCpu cal(p);

    auto result = cal.Execute();

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.monoLeft.success);
    EXPECT_TRUE(result.monoRight.success);
    EXPECT_TRUE(result.stereo.success);
    EXPECT_FALSE(result.stereo.R.empty());
    EXPECT_GT(result.monoLeft.reprojError, 0.0);
    EXPECT_GT(result.monoRight.reprojError, 0.0);
    EXPECT_GT(result.stereo.epipolarErrorMean, 0.0);
}

TEST(ExtrinsicCalibCpuTest, CalibrateFullMonoFailureShortCircuit) {
    auto p = makeSyntheticParams(10, 4, true);
    p.minViewCount = 8;
    p.patternSize = cv::Size(2, 2);
    p.objectPoints = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}
    };
    p.leftPointsPerView.clear();
    p.rightPointsPerView.clear();

    for (int v = 0; v < 10; ++v) {
        p.leftPointsPerView.push_back({{100, 100}, {200, 100}, {100, 200}, {200, 200}});
        p.rightPointsPerView.push_back({{150, 100}, {250, 100}, {150, 200}, {250, 200}});
    }

    ExtrinsicCalibCpu cal(p);
    auto result = cal.Execute();

    EXPECT_TRUE(result.stereo.success);
}

// ============= setParams Tests =============

TEST(ExtrinsicCalibCpuTest, SetParamsValid) {
    auto p = makeSyntheticParams();
    ExtrinsicCalibCpu cal(p);

    auto p2 = makeSyntheticParams(12);
    EXPECT_NO_THROW(cal.SetParams(p2));
    EXPECT_EQ(cal.GetParams().leftPointsPerView.size(), 12u);
}

TEST(ExtrinsicCalibCpuTest, SetParamsInvalidThrows) {
    auto p = makeSyntheticParams();
    ExtrinsicCalibCpu cal(p);

    auto p2 = makeSyntheticParams();
    p2.leftPointsPerView.clear();
    EXPECT_THROW(cal.SetParams(p2), std::invalid_argument);
    EXPECT_EQ(cal.GetParams().leftPointsPerView.size(), 10u);
}

// ============= Constructor Tests =============

TEST(ExtrinsicCalibCpuTest, ConstructorInvalidThrows) {
    auto p = makeSyntheticParams();
    p.leftPointsPerView.clear();
    EXPECT_THROW(ExtrinsicCalibCpu cal(p), std::invalid_argument);
}

// ============= Quality Tests =============

TEST(ExtrinsicCalibCpuTest, CalibrateFullAggregationSuccess) {
    auto p = makeSyntheticParams(10, 88, true);
    p.maxEpipolarError = 1.0;
    ExtrinsicCalibCpu cal(p);
    auto result = cal.Execute();

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.qualityFlag, QualityFlag::Normal);
    EXPECT_FALSE(result.message.empty());
}

TEST(ExtrinsicCalibCpuTest, RotateRightImage180LogsInfo) {
    auto p = makeSyntheticParams();
    p.rotateRightImage180 = true;
    ExtrinsicCalibCpu cal(p);

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    auto result = cal.Execute(KL, distL, KR, distR);
    EXPECT_TRUE(result.success);
}

// ============= QualityFlag toString Tests =============

TEST(ExtrinsicCalibCpuTest, QualityFlagToString) {
    EXPECT_STREQ(qualityFlagToString(QualityFlag::Normal), "Normal");
    EXPECT_STREQ(qualityFlagToString(QualityFlag::Degraded), "Degraded");
    EXPECT_STREQ(qualityFlagToString(QualityFlag::Warning), "Warning");
}

// ============= ExtrinsicCalibCpuResult JSON Serialization Tests =============

TEST(ExtrinsicCalibCpuResultJsonTest, RoundTrip) {
    ExtrinsicCalibCpuResult original;
    original.success = true;
    original.message = "test message";
    original.qualityFlag = QualityFlag::Normal;
    original.R = cv::Mat::eye(3, 3, CV_64F);
    original.T = (cv::Mat_<double>(3, 1) << 100.0, 0.0, 0.0);
    original.E = cv::Mat::eye(3, 3, CV_64F);
    original.F = cv::Mat::eye(3, 3, CV_64F);
    original.stereoReprojError = 0.5;
    original.epipolarErrorMean = 0.3;
    original.epipolarErrorStd = 0.1;
    original.perViewErrors = {0.1, 0.2, 0.3};
    original.perViewEpipolarErrors = {0.01, 0.02, 0.03};
    original.cameraMatrixL = (cv::Mat_<double>(3, 3) << 2500, 0, 1024, 0, 2500, 768, 0, 0, 1);
    original.distCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    original.cameraMatrixR = (cv::Mat_<double>(3, 3) << 2500, 0, 1024, 0, 2500, 768, 0, 0, 1);
    original.distCoeffsR = cv::Mat::zeros(1, 5, CV_64F);

    auto j = original.toJson();
    auto loaded = ExtrinsicCalibCpuResult::fromJson(j);

    EXPECT_EQ(loaded.success, true);
    EXPECT_EQ(loaded.message, "test message");
    EXPECT_EQ(loaded.qualityFlag, QualityFlag::Normal);
    EXPECT_DOUBLE_EQ(loaded.stereoReprojError, 0.5);
    EXPECT_DOUBLE_EQ(loaded.epipolarErrorMean, 0.3);
    EXPECT_DOUBLE_EQ(loaded.epipolarErrorStd, 0.1);
    EXPECT_EQ(loaded.perViewErrors.size(), 3u);
    EXPECT_EQ(loaded.perViewEpipolarErrors.size(), 3u);
    EXPECT_DOUBLE_EQ(loaded.R.at<double>(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(loaded.T.at<double>(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(loaded.cameraMatrixL.at<double>(0, 0), 2500.0);
    EXPECT_DOUBLE_EQ(loaded.distCoeffsL.at<double>(0, 0), 0.0);
}

TEST(ExtrinsicCalibCpuResultJsonTest, RoundTripEmptyIntrinsics) {
    ExtrinsicCalibCpuResult original;
    original.success = true;
    original.message = "";
    original.qualityFlag = QualityFlag::Degraded;
    original.R = cv::Mat::eye(3, 3, CV_64F);
    original.T = (cv::Mat_<double>(3, 1) << 50.0, 0.0, 0.0);
    original.E = cv::Mat::eye(3, 3, CV_64F);
    original.F = cv::Mat::eye(3, 3, CV_64F);
    original.stereoReprojError = 1.5;
    original.epipolarErrorMean = 0.0;
    original.epipolarErrorStd = 0.0;
    original.perViewErrors = {};
    original.perViewEpipolarErrors = {};

    auto j = original.toJson();
    EXPECT_FALSE(j.contains("camera_matrix_l"));

    auto loaded = ExtrinsicCalibCpuResult::fromJson(j);
    EXPECT_EQ(loaded.qualityFlag, QualityFlag::Degraded);
    EXPECT_TRUE(loaded.cameraMatrixL.empty());
    EXPECT_TRUE(loaded.perViewErrors.empty());
}

TEST(ExtrinsicCalibCpuResultJsonTest, RoundTripFromCalibration) {
    auto p = makeSyntheticParams(10);
    ExtrinsicCalibCpu cal(p);

    double fx = 2500.0, fy = 2500.0;
    double cx = 1024.0, cy = 768.0;
    cv::Mat KL = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat KR = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distR = cv::Mat::zeros(1, 5, CV_64F);

    auto result = cal.Execute(KL, distL, KR, distR);
    ASSERT_TRUE(result.success);

    auto j = result.toJson();
    auto loaded = ExtrinsicCalibCpuResult::fromJson(j);

    EXPECT_EQ(loaded.success, result.success);
    EXPECT_DOUBLE_EQ(loaded.stereoReprojError, result.stereoReprojError);
    EXPECT_DOUBLE_EQ(loaded.epipolarErrorMean, result.epipolarErrorMean);
    EXPECT_EQ(loaded.perViewEpipolarErrors.size(), result.perViewEpipolarErrors.size());

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(loaded.R.at<double>(r, c), result.R.at<double>(r, c), 1e-10);
        }
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(loaded.T.at<double>(i, 0), result.T.at<double>(i, 0), 1e-10);
    }
    EXPECT_NEAR(loaded.cameraMatrixL.at<double>(0, 0), fx, 1e-10);
}
