/**
 * @file test_virtual_camera_pose_cuda.cpp
 * @brief 激光器虚拟相机光心和初步外参CUDA算子单元测试
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "virtual_camera_pose_cuda.h"

using namespace calib;


class VirtualCameraPoseTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.deviceId = 0;
        params_.ransacThreshold = 1.0;
        params_.ransacConfidence = 0.99;
        params_.ransacMaxIterations = 1000;
        params_.minLinesForSolve = 3;
        params_.minPointsPerLine = 3;
        poser_.reset(new VirtualCameraPoseCuda(params_));
    }

    void TearDown() override {
        poser_.reset();
    }

    VirtualCameraPoseResult runProcess(
        const std::vector<cv::Vec3f>& endpoints,
        const std::vector<int>& lineIds,
        const cv::Matx33d& K = cv::Matx33d::eye(),
        const cv::Matx33d& R = cv::Matx33d::eye())
    {
        cv::Mat h_pts(1, static_cast<int>(endpoints.size()), CV_32FC3,
                       const_cast<cv::Vec3f*>(endpoints.data()));
        cv::Mat h_lids(1, static_cast<int>(lineIds.size()), CV_32SC1,
                        const_cast<int*>(lineIds.data()));

        cv::cuda::GpuMat d_pts, d_lids;
        if (!h_pts.empty()) d_pts.upload(h_pts);
        if (!h_lids.empty()) d_lids.upload(h_lids);

        return poser_->Execute(d_pts, d_lids, K, R);
    }

    std::unique_ptr<VirtualCameraPoseCuda> poser_;
    VirtualCameraPoseParams params_;
};

// ============================================================================
// CPU-only tests (always compiled)
// ============================================================================

TEST_F(VirtualCameraPoseTest, ParamsValidation) {
    VirtualCameraPoseParams p;

    p.deviceId = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.deviceId = 0;
    p.ransacThreshold = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.ransacThreshold = -1.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.ransacThreshold = 1.0;
    p.ransacConfidence = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.ransacConfidence = 1.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.ransacConfidence = 0.99;
    p.ransacMaxIterations = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.ransacMaxIterations = 1000;
    p.minLinesForSolve = 1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minLinesForSolve = 3;
    p.minPointsPerLine = 1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minPointsPerLine = 3;
    EXPECT_NO_THROW(p.validate());
}

TEST_F(VirtualCameraPoseTest, ParamsJson) {
    VirtualCameraPoseParams p;
    p.deviceId = 1;
    p.ransacThreshold = 2.5;
    p.ransacConfidence = 0.95;
    p.ransacMaxIterations = 500;
    p.minLinesForSolve = 4;
    p.minPointsPerLine = 5;
    p.enableTiming = true;

    auto j = p.toJson();
    auto p2 = VirtualCameraPoseParams::fromJson(j);

    EXPECT_EQ(p2.deviceId, 1);
    EXPECT_DOUBLE_EQ(p2.ransacThreshold, 2.5);
    EXPECT_DOUBLE_EQ(p2.ransacConfidence, 0.95);
    EXPECT_EQ(p2.ransacMaxIterations, 500);
    EXPECT_EQ(p2.minLinesForSolve, 4);
    EXPECT_EQ(p2.minPointsPerLine, 5);
    EXPECT_TRUE(p2.enableTiming);
}

// ============================================================================
// GPU tests
// ============================================================================

#ifdef WITH_CUDA_TESTS

TEST_F(VirtualCameraPoseTest, EmptyInput) {
    std::vector<cv::Vec3f> pts;
    std::vector<int> lids;
    auto result = runProcess(pts, lids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.totalEndpoints, 0);
}

TEST_F(VirtualCameraPoseTest, InvalidInputType) {
    cv::cuda::GpuMat d_bad(1, 4, CV_32FC2);
    cv::cuda::GpuMat d_lids(1, 4, CV_32SC1);
    cv::Matx33d K = cv::Matx33d::eye();
    auto result = poser_->Execute(d_bad, d_lids, K, K);
    EXPECT_FALSE(result.success);
}

TEST_F(VirtualCameraPoseTest, CountMismatch) {
    cv::Mat h_pts = (cv::Mat_<cv::Vec3f>(1, 4) <<
        cv::Vec3f(0,0,0), cv::Vec3f(1,0,0), cv::Vec3f(0,1,0), cv::Vec3f(0,0,1));
    cv::Mat h_lids = (cv::Mat_<int>(1, 3) << 0, 0, 0);
    cv::cuda::GpuMat d_pts(h_pts);
    cv::cuda::GpuMat d_lids(h_lids);
    cv::Matx33d K = cv::Matx33d::eye();
    auto result = poser_->Execute(d_pts, d_lids, K, K);
    EXPECT_FALSE(result.success);
}

TEST_F(VirtualCameraPoseTest, SyntheticExact) {
    Eigen::Vector3d center(100.0, 200.0, 50.0);

    std::vector<Eigen::Vector3d> directions = {
        Eigen::Vector3d(1.0, 0.0, 0.0).normalized(),
        Eigen::Vector3d(0.0, 1.0, 0.0).normalized(),
        Eigen::Vector3d(0.0, 0.0, 1.0).normalized()
    };

    std::vector<cv::Vec3f> endpoints;
    std::vector<int> lineIds;

    for (int lid = 0; lid < 3; ++lid) {
        Eigen::Vector3d dir = directions[lid];
        for (int j = 0; j < 5; ++j) {
            double t = -2.0 + j * 1.0;
            Eigen::Vector3d p = center + t * dir;
            endpoints.push_back(cv::Vec3f(
                static_cast<float>(p.x()),
                static_cast<float>(p.y()),
                static_cast<float>(p.z())));
            lineIds.push_back(lid);
        }
    }

    cv::Matx33d K(1000, 0, 640, 0, 1000, 360, 0, 0, 1);
    cv::Matx33d R = cv::Matx33d::eye();

    auto result = runProcess(endpoints, lineIds, K, R);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, 3);

    EXPECT_NEAR(result.virtualT(0), 100.0, 0.01);
    EXPECT_NEAR(result.virtualT(1), 200.0, 0.01);
    EXPECT_NEAR(result.virtualT(2), 50.0, 0.01);

    EXPECT_EQ(result.virtualK, K);
    EXPECT_EQ(result.virtualR, R);

    EXPECT_NEAR(result.avgDistToCenter, 0.0, 0.01);
}

TEST_F(VirtualCameraPoseTest, SyntheticWithNoise) {
    Eigen::Vector3d center(50.0, -30.0, 100.0);

    std::vector<Eigen::Vector3d> directions = {
        Eigen::Vector3d(1.0, 1.0, 0.0).normalized(),
        Eigen::Vector3d(0.0, 1.0, 1.0).normalized(),
        Eigen::Vector3d(1.0, 0.0, 1.0).normalized(),
        Eigen::Vector3d(1.0, -1.0, 0.5).normalized()
    };

    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    std::vector<cv::Vec3f> endpoints;
    std::vector<int> lineIds;

    for (int lid = 0; lid < 4; ++lid) {
        Eigen::Vector3d dir = directions[lid];
        for (int j = 0; j < 10; ++j) {
            double t = -5.0 + j * 1.0;
            Eigen::Vector3d p = center + t * dir;
            endpoints.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    auto result = runProcess(endpoints, lineIds);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, 4);

    EXPECT_NEAR(result.virtualT(0), 50.0, 1.0);
    EXPECT_NEAR(result.virtualT(1), -30.0, 1.0);
    EXPECT_NEAR(result.virtualT(2), 100.0, 1.0);
}

TEST_F(VirtualCameraPoseTest, SyntheticWithOutliers) {
    Eigen::Vector3d center(0.0, 0.0, 200.0);

    std::vector<Eigen::Vector3d> directions = {
        Eigen::Vector3d(1.0, 0.0, 0.0).normalized(),
        Eigen::Vector3d(0.0, 1.0, 0.0).normalized(),
        Eigen::Vector3d(1.0, 1.0, 0.0).normalized()
    };

    std::mt19937 rng(456);
    std::normal_distribution<float> noise(0.0f, 0.05f);

    std::vector<cv::Vec3f> endpoints;
    std::vector<int> lineIds;

    for (int lid = 0; lid < 3; ++lid) {
        Eigen::Vector3d dir = directions[lid];
        for (int j = 0; j < 8; ++j) {
            double t = -4.0 + j * 1.0;
            Eigen::Vector3d p = center + t * dir;
            endpoints.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
        for (int j = 0; j < 2; ++j) {
            endpoints.push_back(cv::Vec3f(
                static_cast<float>(center.x()) + 50.0f + noise(rng),
                static_cast<float>(center.y()) + 50.0f + noise(rng),
                static_cast<float>(center.z()) + 50.0f + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    params_.ransacThreshold = 1.0;
    poser_.reset(new VirtualCameraPoseCuda(params_));

    auto result = runProcess(endpoints, lineIds);
    ASSERT_TRUE(result.success) << result.message;

    EXPECT_NEAR(result.virtualT(0), 0.0, 2.0);
    EXPECT_NEAR(result.virtualT(1), 0.0, 2.0);
    EXPECT_NEAR(result.virtualT(2), 200.0, 2.0);

    int totalPerLine = 10;
    for (int cnt : result.lineInlierCounts) {
        EXPECT_LT(cnt, totalPerLine);
        EXPECT_GE(cnt, 3);
    }
}

TEST_F(VirtualCameraPoseTest, KnownVirtualCamera) {
    double fx = 2000.0, fy = 2000.0, cx = 1024.0, cy = 768.0;
    cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d expectedT(100.0, -50.0, 300.0);

    std::mt19937 rng(789);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    std::vector<cv::Vec3f> endpoints;
    std::vector<int> lineIds;

    int numLines = 8;
    for (int lid = 0; lid < numLines; ++lid) {
        double theta = lid * M_PI / numLines;
        double phi = lid * M_PI / (2.0 * numLines);
        Eigen::Vector3d dir(std::cos(theta) * std::cos(phi),
                            std::sin(theta) * std::cos(phi),
                            std::sin(phi));
        dir.normalize();

        Eigen::Vector3d ctr(expectedT(0), expectedT(1), expectedT(2));
        for (int j = 0; j < 6; ++j) {
            double t = -3.0 + j * 1.2;
            Eigen::Vector3d p = ctr + t * dir;
            endpoints.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    auto result = runProcess(endpoints, lineIds, K, R);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);

    EXPECT_NEAR(result.virtualT(0), expectedT(0), 1.0);
    EXPECT_NEAR(result.virtualT(1), expectedT(1), 1.0);
    EXPECT_NEAR(result.virtualT(2), expectedT(2), 1.0);

    EXPECT_EQ(result.virtualK(0, 0), fx);
    EXPECT_EQ(result.virtualK(1, 1), fy);
    EXPECT_EQ(result.virtualK(0, 2), cx);
    EXPECT_EQ(result.virtualK(1, 2), cy);
}

TEST_F(VirtualCameraPoseTest, InsufficientLines) {
    std::vector<cv::Vec3f> pts = {
        {0,0,0}, {1,0,0}, {2,0,0}, {3,0,0},
        {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}
    };
    std::vector<int> lids = {0, 0, 0, 0, 1, 1, 1, 1};

    auto result = runProcess(pts, lids);
    EXPECT_FALSE(result.success);
}

TEST_F(VirtualCameraPoseTest, MultipleLines10Plus) {
    Eigen::Vector3d center(25.0, 75.0, 150.0);

    std::mt19937 rng(321);
    std::normal_distribution<float> noise(0.0f, 0.3f);

    std::vector<cv::Vec3f> endpoints;
    std::vector<int> lineIds;

    int numLines = 15;
    for (int lid = 0; lid < numLines; ++lid) {
        double angle = lid * 2.0 * M_PI / numLines;
        Eigen::Vector3d dir(std::cos(angle), std::sin(angle), 0.3 * lid);
        dir.normalize();

        for (int j = 0; j < 5; ++j) {
            double t = -2.0 + j;
            Eigen::Vector3d p = center + t * dir;
            endpoints.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    auto result = runProcess(endpoints, lineIds);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);
    EXPECT_NEAR(result.virtualT(0), 25.0, 2.0);
    EXPECT_NEAR(result.virtualT(1), 75.0, 2.0);
    EXPECT_NEAR(result.virtualT(2), 150.0, 2.0);
}

#endif // WITH_CUDA_TESTS
