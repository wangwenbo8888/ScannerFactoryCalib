#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "pose_optimize_cuda.h"

using namespace calib;


class PoseOptimizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.deviceId = 0;
        params_.convergenceThreshold = 1e-6;
        params_.maxIterations = 50;
        params_.minLinesForOptimize = 3;
        params_.minPointsPerLine = 10;
        poser_.reset(new PoseOptimizeCuda(params_));
    }

    void TearDown() override {
        poser_.reset();
    }

    PoseOptimizeResult runProcess(
        const std::vector<cv::Vec3f>& points,
        const std::vector<int>& lineIds,
        const cv::Matx33d& K = cv::Matx33d::eye(),
        const cv::Matx33d& R = cv::Matx33d::eye(),
        const cv::Vec3d& initialT = cv::Vec3d(0, 0, 0))
    {
        cv::Mat h_pts(1, static_cast<int>(points.size()), CV_32FC3,
                       const_cast<cv::Vec3f*>(points.data()));
        cv::Mat h_lids(1, static_cast<int>(lineIds.size()), CV_32SC1,
                        const_cast<int*>(lineIds.data()));

        cv::cuda::GpuMat d_pts, d_lids;
        if (!h_pts.empty()) d_pts.upload(h_pts);
        if (!h_lids.empty()) d_lids.upload(h_lids);

        return poser_->Execute(d_pts, d_lids, K, R, initialT);
    }

    std::unique_ptr<PoseOptimizeCuda> poser_;
    PoseOptimizeParams params_;
};

TEST_F(PoseOptimizeTest, ParamsValidation) {
    PoseOptimizeParams p;

    p.deviceId = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.deviceId = 0;
    p.convergenceThreshold = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.convergenceThreshold = -1e-6;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.convergenceThreshold = 1e-6;
    p.maxIterations = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.maxIterations = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.maxIterations = 50;
    p.minLinesForOptimize = 1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minLinesForOptimize = 2;
    p.minPointsPerLine = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minPointsPerLine = 3;
    EXPECT_NO_THROW(p.validate());
}

TEST_F(PoseOptimizeTest, ParamsJson) {
    PoseOptimizeParams p;
    p.deviceId = 1;
    p.maxIterations = 100;
    p.convergenceThreshold = 1e-7;
    p.minLinesForOptimize = 4;
    p.minPointsPerLine = 5;
    p.enableTiming = true;

    auto j = p.toJson();
    auto p2 = PoseOptimizeParams::fromJson(j);

    EXPECT_EQ(p2.deviceId, 1);
    EXPECT_EQ(p2.maxIterations, 100);
    EXPECT_DOUBLE_EQ(p2.convergenceThreshold, 1e-7);
    EXPECT_EQ(p2.minLinesForOptimize, 4);
    EXPECT_EQ(p2.minPointsPerLine, 5);
    EXPECT_TRUE(p2.enableTiming);
}

#ifdef WITH_CUDA_TESTS

TEST_F(PoseOptimizeTest, EmptyInput) {
    std::vector<cv::Vec3f> pts;
    std::vector<int> lids;
    auto result = runProcess(pts, lids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 0);
}

TEST_F(PoseOptimizeTest, InvalidInputType) {
    cv::cuda::GpuMat d_bad(1, 4, CV_32FC2);
    cv::cuda::GpuMat d_lids(1, 4, CV_32SC1);
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T(0, 0, 0);
    auto result = poser_->Execute(d_bad, d_lids, K, R, T);
    EXPECT_FALSE(result.success);
}

TEST_F(PoseOptimizeTest, CountMismatch) {
    cv::Mat h_pts = (cv::Mat_<cv::Vec3f>(1, 4) <<
        cv::Vec3f(0,0,0), cv::Vec3f(1,0,0), cv::Vec3f(0,1,0), cv::Vec3f(0,0,1));
    cv::Mat h_lids = (cv::Mat_<int>(1, 3) << 0, 0, 0);
    cv::cuda::GpuMat d_pts(h_pts);
    cv::cuda::GpuMat d_lids(h_lids);
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T(0, 0, 0);
    auto result = poser_->Execute(d_pts, d_lids, K, R, T);
    EXPECT_FALSE(result.success);
}

TEST_F(PoseOptimizeTest, SyntheticExact) {
    double fx = 2000.0, fy = 2000.0, cx = 1024.0, cy = 768.0;
    cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Matx33d R_mat = cv::Matx33d::eye();
    cv::Vec3d T_true(100.0, -50.0, 300.0);

    int numLines = 5;
    std::vector<cv::Vec3f> points;
    std::vector<int> lineIds;

    for (int lid = 0; lid < numLines; ++lid) {
        double theta = lid * M_PI / numLines;
        Eigen::Vector3d ray_dir(std::cos(theta), std::sin(theta), 0.3);
        ray_dir.normalize();

        for (int j = 0; j < 15; ++j) {
            double t = 200.0 + j * 20.0;
            Eigen::Vector3d p_left = (Eigen::Vector3d(T_true(0), T_true(1), T_true(2))
                + ray_dir * t
                + Eigen::Vector3d(0, 0, j * 0.01));
            points.push_back(cv::Vec3f(static_cast<float>(p_left.x()),
                                       static_cast<float>(p_left.y()),
                                       static_cast<float>(p_left.z())));
            lineIds.push_back(lid);
        }
    }

    cv::Vec3d T_init(102.0, -51.0, 303.0);
    auto result = runProcess(points, lineIds, K, R_mat, T_init);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);
    EXPECT_EQ(result.virtualK, K);
    EXPECT_EQ(result.virtualR, R_mat);
    EXPECT_GT(result.lineCurves.size(), 0u);
    EXPECT_LT(result.totalReprojectionError, result.initialReprojectionError);
}

TEST_F(PoseOptimizeTest, SyntheticWithNoise) {
    double fx = 2000.0, fy = 2000.0, cx = 1024.0, cy = 768.0;
    cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Matx33d R_mat = cv::Matx33d::eye();

    int numLines = 5;
    std::vector<cv::Vec3f> points;
    std::vector<int> lineIds;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    for (int lid = 0; lid < numLines; ++lid) {
        Eigen::Vector3d normal(0.1 * lid, 1.0, 0.2 * lid);
        normal.normalize();
        double d = 500.0 + 100.0 * lid;

        Eigen::Vector3d u(1, 0, 0);
        if (fabs(normal.dot(u)) > 0.9) u = Eigen::Vector3d(0, 1, 0);
        u = (u - normal.dot(u) * normal).normalized();
        Eigen::Vector3d v = normal.cross(u).normalized();

        Eigen::Vector3d p0 = d * normal;

        for (int j = 0; j < 15; ++j) {
            double alpha = -7.0 + j * 1.0;
            double beta = (lid % 2 == 0 ? 1.0 : -1.0) * (j % 3 - 1) * 0.5;
            Eigen::Vector3d p = p0 + alpha * u + beta * v;
            points.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    cv::Vec3d T_init(105.0, -45.0, 310.0);
    auto result = runProcess(points, lineIds, K, R_mat, T_init);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);
    EXPECT_LT(result.totalReprojectionError, result.initialReprojectionError);
}

TEST_F(PoseOptimizeTest, InsufficientLines) {
    std::vector<cv::Vec3f> pts;
    std::vector<int> lids;
    for (int i = 0; i < 10; ++i) {
        pts.push_back(cv::Vec3f(static_cast<float>(i), 0.0f, 0.0f));
        lids.push_back(0);
    }
    auto result = runProcess(pts, lids);
    EXPECT_FALSE(result.success);
}

TEST_F(PoseOptimizeTest, TranslationOnlyOptimization) {
    double angle = M_PI / 6.0;
    cv::Matx33d R(std::cos(angle), -std::sin(angle), 0,
                  std::sin(angle),  std::cos(angle), 0,
                  0, 0, 1);
    cv::Matx33d K(2000, 0, 1024, 0, 2000, 768, 0, 0, 1);
    cv::Vec3d T_init(10.0, 20.0, 300.0);

    int numLines = 4;
    std::vector<cv::Vec3f> points;
    std::vector<int> lineIds;

    for (int lid = 0; lid < numLines; ++lid) {
        Eigen::Vector3d normal(0.1 * lid, 1.0, 0.3 * lid);
        normal.normalize();
        double d = 400.0 + 80.0 * lid;

        Eigen::Vector3d u(1, 0, 0);
        if (fabs(normal.dot(u)) > 0.9) u = Eigen::Vector3d(0, 1, 0);
        u = (u - normal.dot(u) * normal).normalized();
        Eigen::Vector3d v = normal.cross(u).normalized();

        Eigen::Vector3d p0 = d * normal;

        for (int j = 0; j < 12; ++j) {
            double alpha = -6.0 + j * 1.0;
            double beta = (j % 3 - 1) * 0.4;
            Eigen::Vector3d p = p0 + alpha * u + beta * v;
            points.push_back(cv::Vec3f(static_cast<float>(p.x()),
                                       static_cast<float>(p.y()),
                                       static_cast<float>(p.z())));
            lineIds.push_back(lid);
        }
    }

    auto result = runProcess(points, lineIds, K, R, T_init);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(result.virtualR(i, j), R(i, j), 1e-12);
        }
    }
}

TEST_F(PoseOptimizeTest, FixedROptimization) {
    double angle = M_PI / 6.0;
    cv::Matx33d R(std::cos(angle), -std::sin(angle), 0,
                  std::sin(angle),  std::cos(angle), 0,
                  0, 0, 1);
    cv::Matx33d K(2000, 0, 1024, 0, 2000, 768, 0, 0, 1);

    int numLines = 5;
    std::vector<cv::Vec3f> points;
    std::vector<int> lineIds;

    std::mt19937 rng(789);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    for (int lid = 0; lid < numLines; ++lid) {
        Eigen::Vector3d normal(0.1 * lid, 1.0, 0.2 * lid);
        normal.normalize();
        double d = 500.0 + 100.0 * lid;

        Eigen::Vector3d u(1, 0, 0);
        if (fabs(normal.dot(u)) > 0.9) u = Eigen::Vector3d(0, 1, 0);
        u = (u - normal.dot(u) * normal).normalized();
        Eigen::Vector3d v = normal.cross(u).normalized();

        Eigen::Vector3d p0 = d * normal;

        for (int j = 0; j < 12; ++j) {
            double alpha = -6.0 + j * 1.0;
            double beta = (lid % 2 == 0 ? 1.0 : -1.0) * (j % 3 - 1) * 0.5;
            Eigen::Vector3d p = p0 + alpha * u + beta * v;
            points.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    cv::Vec3d T_init(85.0, -25.0, 260.0);
    auto result = runProcess(points, lineIds, K, R, T_init);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);
    EXPECT_LT(result.totalReprojectionError, result.initialReprojectionError);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(result.virtualR(i, j), R(i, j), 1e-12);
        }
    }
}

TEST_F(PoseOptimizeTest, MultipleLines10Plus) {
    double fx = 2000.0, fy = 2000.0, cx = 1024.0, cy = 768.0;
    cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Matx33d R_mat = cv::Matx33d::eye();

    int numLines = 15;
    std::vector<cv::Vec3f> points;
    std::vector<int> lineIds;

    std::mt19937 rng(321);
    std::normal_distribution<float> noise(0.0f, 0.3f);

    for (int lid = 0; lid < numLines; ++lid) {
        Eigen::Vector3d normal(std::cos(lid * 0.4), std::sin(lid * 0.4), 0.5);
        normal.normalize();
        double d = 300.0 + 50.0 * lid;

        Eigen::Vector3d u(1, 0, 0);
        if (fabs(normal.dot(u)) > 0.9) u = Eigen::Vector3d(0, 1, 0);
        u = (u - normal.dot(u) * normal).normalized();
        Eigen::Vector3d v = normal.cross(u).normalized();

        Eigen::Vector3d p0 = d * normal;

        for (int j = 0; j < 10; ++j) {
            double alpha = -5.0 + j * 1.0;
            double beta = (j % 3 - 1) * 0.3;
            Eigen::Vector3d p = p0 + alpha * u + beta * v;
            points.push_back(cv::Vec3f(
                static_cast<float>(p.x()) + noise(rng),
                static_cast<float>(p.y()) + noise(rng),
                static_cast<float>(p.z()) + noise(rng)));
            lineIds.push_back(lid);
        }
    }

    cv::Vec3d T_init(55.0, 20.0, 210.0);
    auto result = runProcess(points, lineIds, K, R_mat, T_init);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.numLines, numLines);
    EXPECT_EQ(static_cast<int>(result.lineCurves.size()), numLines);
    EXPECT_LT(result.totalReprojectionError, result.initialReprojectionError);
}

#endif // WITH_CUDA_TESTS
