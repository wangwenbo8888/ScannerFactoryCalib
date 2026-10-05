#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <random>
#include "plane_map_cuda.h"
#include "common/calib_types.h"

using namespace calib;


TEST(PlaneMapParamsTest, DefaultValidation) {
    PlaneMapParams p;
    EXPECT_NO_THROW(p.validate());
    EXPECT_EQ(p.deviceId, 0);
    EXPECT_EQ(p.method, PlaneMapMethod::Projective);
    EXPECT_FLOAT_EQ(p.gridStep, 0.5f);
}

TEST(PlaneMapParamsTest, InvalidDeviceId) {
    PlaneMapParams p;
    p.deviceId = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PlaneMapParamsTest, InvalidGridStep) {
    PlaneMapParams p;
    p.gridStep = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.gridStep = -0.5f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PlaneMapParamsTest, InvalidDepthRange) {
    PlaneMapParams p;
    p.depthMin = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.depthMin = 100.0f;
    p.depthMax = 50.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PlaneMapParamsTest, InvalidDepthSamples) {
    PlaneMapParams p;
    p.depthSamples = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PlaneMapParamsTest, InvalidEpipolarStep) {
    PlaneMapParams p;
    p.epipolarStep = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PlaneMapParamsTest, JsonRoundTrip) {
    PlaneMapParams p;
    p.deviceId = 1;
    p.method = PlaneMapMethod::FundamentalMatrix;
    p.gridStep = 0.25f;
    p.depthMin = 200.0f;
    p.depthMax = 8000.0f;
    p.depthSamples = 400;
    p.epipolarStep = 0.25f;
    p.enableTiming = true;

    auto j = p.toJson();
    auto p2 = PlaneMapParams::fromJson(j);

    EXPECT_EQ(p2.deviceId, 1);
    EXPECT_EQ(p2.method, PlaneMapMethod::FundamentalMatrix);
    EXPECT_FLOAT_EQ(p2.gridStep, 0.25f);
    EXPECT_FLOAT_EQ(p2.depthMin, 200.0f);
    EXPECT_FLOAT_EQ(p2.depthMax, 8000.0f);
    EXPECT_EQ(p2.depthSamples, 400);
    EXPECT_FLOAT_EQ(p2.epipolarStep, 0.25f);
    EXPECT_TRUE(p2.enableTiming);
}

#ifdef WITH_CUDA_TESTS

class PlaneMapCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.deviceId = 0;
        params_.gridStep = 0.5f;
        params_.depthMin = 50.0f;
        params_.depthMax = 3000.0f;
        params_.depthSamples = 300;
        params_.epipolarStep = 0.5f;

        double fx = 2000.0, fy = 2000.0, cx = 1024.0, cy = 768.0;
        virtualK_ = cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
        virtualR_ = cv::Matx33d::eye();
        virtualT_ = cv::Vec3d(100.0, -20.0, 50.0);

        calib_.imageSize = cv::Size(2048, 1536);
        double baseline = 120.0;
        cv::Matx33d R1 = cv::Matx33d::eye();
        cv::Matx33d R2 = cv::Matx33d::eye();
        cv::Matx34d P1m(fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0);
        cv::Matx34d P2m(fx, 0, cx, -fx * baseline, 0, fy, cy, 0, 0, 0, 1, 0);
        calib_.R1 = cv::Mat(R1);
        calib_.R2 = cv::Mat(R2);
        calib_.P1 = cv::Mat(P1m);
        calib_.P2 = cv::Mat(P2m);
    }

    void TearDown() override { mapper_.reset(); }

    std::vector<cv::Vec3f> generateVirtualPixels(int numLines, int ptsPerLine) {
        std::vector<cv::Vec3f> pixels;
        double fx = virtualK_(0, 0), fy = virtualK_(1, 1);
        double cx = virtualK_(0, 2), cy = virtualK_(1, 2);

        for (int lid = 0; lid < numLines; ++lid) {
            double depth = 800.0 + 200.0 * lid;
            double v_center = cy + (lid - numLines / 2.0) * 60.0;

            for (int j = 0; j < ptsPerLine; ++j) {
                double u = 100.0 + j * ((calib_.imageSize.width - 200.0) / std::max(ptsPerLine - 1, 1));
                double v = v_center + ((j % 5) - 2) * 2.0;

                if (v < 0 || v >= calib_.imageSize.height) continue;

                double x_cam = (u - cx) / fx * depth;
                double y_cam = (v - cy) / fy * depth;
                double z_cam = depth;

                Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_map(virtualR_.val);
                Eigen::Vector3d p_cam(x_cam, y_cam, z_cam);
                Eigen::Vector3d p_world = R_map.inverse() * (p_cam -
                    Eigen::Vector3d(virtualT_(0), virtualT_(1), virtualT_(2)));

                double px = p_world.x();
                double py = p_world.y();
                double pz = p_world.z();

                double u_v = fx * (R_map(0,0)*px + R_map(0,1)*py + R_map(0,2)*pz + virtualT_(0))
                               / (R_map(2,0)*px + R_map(2,1)*py + R_map(2,2)*pz + virtualT_(2)) + cx;
                double v_v = fy * (R_map(1,0)*px + R_map(1,1)*py + R_map(1,2)*pz + virtualT_(1))
                               / (R_map(2,0)*px + R_map(2,1)*py + R_map(2,2)*pz + virtualT_(2)) + cy;

                (void)u_v; (void)v_v;

                pixels.push_back(cv::Vec3f(
                    static_cast<float>(u),
                    static_cast<float>(v),
                    static_cast<float>(lid)));
            }
        }
        return pixels;
    }

    PlaneMapResult runProcess(PlaneMapMethod method,
                              const std::vector<cv::Vec3f>& pixels) {
        params_.method = method;
        mapper_.reset(new PlaneMapCuda(params_));

        cv::Mat h_px(1, static_cast<int>(pixels.size()), CV_32FC3,
                     const_cast<cv::Vec3f*>(pixels.data()));
        cv::cuda::GpuMat d_px(h_px);

        return mapper_->Execute(d_px, virtualK_, virtualR_, virtualT_, calib_);
    }

    std::unique_ptr<PlaneMapCuda> mapper_;
    PlaneMapParams params_;
    cv::Matx33d virtualK_, virtualR_;
    cv::Vec3d virtualT_;
    calib::StereoCalibration calib_;
};

TEST_F(PlaneMapCudaTest, EmptyInput) {
    std::vector<cv::Vec3f> empty;
    params_.method = PlaneMapMethod::Projective;
    mapper_.reset(new PlaneMapCuda(params_));
    cv::cuda::GpuMat d_empty;
    auto result = mapper_->Execute(d_empty, virtualK_, virtualR_, virtualT_, calib_);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.totalPairs, 0);
}

TEST_F(PlaneMapCudaTest, InvalidInputType) {
    params_.method = PlaneMapMethod::Projective;
    mapper_.reset(new PlaneMapCuda(params_));
    cv::cuda::GpuMat d_bad(1, 10, CV_32FC2);
    auto result = mapper_->Execute(d_bad, virtualK_, virtualR_, virtualT_, calib_);
    EXPECT_FALSE(result.success);
}

TEST_F(PlaneMapCudaTest, SyntheticProjective) {
    auto pixels = generateVirtualPixels(5, 20);
    ASSERT_GT(pixels.size(), 0u);

    auto result = runProcess(PlaneMapMethod::Projective, pixels);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GT(result.totalPairs, 0);
    EXPECT_GE(static_cast<int>(result.lineStats.size()), 1);
}

TEST_F(PlaneMapCudaTest, SyntheticFundamental) {
    auto pixels = generateVirtualPixels(5, 20);
    ASSERT_GT(pixels.size(), 0u);

    auto result = runProcess(PlaneMapMethod::FundamentalMatrix, pixels);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GT(result.totalPairs, 0);
    EXPECT_GE(static_cast<int>(result.lineStats.size()), 1);
}

TEST_F(PlaneMapCudaTest, StereoRectifyCheck) {
    auto pixels = generateVirtualPixels(3, 15);
    ASSERT_GT(pixels.size(), 0u);

    auto result = runProcess(PlaneMapMethod::Projective, pixels);
    ASSERT_TRUE(result.success) << result.message;
    ASSERT_GT(result.totalPairs, 0);

    ASSERT_NE(result.d_left_to_right, nullptr);
    cv::Mat h_ltr;
    result.d_left_to_right->download(h_ltr);
    h_ltr = h_ltr.reshape(4);

    int M = h_ltr.rows * h_ltr.cols;
    for (int i = 0; i < M; ++i) {
        cv::Vec4f v = h_ltr.at<cv::Vec4f>(i);
        float uL = v[0], vL = v[1], uR = v[2];
        EXPECT_GE(uL, -0.5f);
        EXPECT_LE(uL, calib_.imageSize.width + 0.5f);
        EXPECT_GE(vL, -0.5f);
        EXPECT_LE(vL, calib_.imageSize.height + 0.5f);
        EXPECT_GE(uR, -0.5f);
        EXPECT_LE(uR, calib_.imageSize.width + 0.5f);
    }
}

TEST_F(PlaneMapCudaTest, MultipleLines10Plus) {
    auto pixels = generateVirtualPixels(15, 10);
    ASSERT_GT(pixels.size(), 0u);

    auto result = runProcess(PlaneMapMethod::Projective, pixels);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GT(result.totalPairs, 100);
    EXPECT_GE(static_cast<int>(result.lineStats.size()), 5);

    for (const auto& st : result.lineStats) {
        EXPECT_GT(st.numPairs, 0);
    }
}

TEST_F(PlaneMapCudaTest, GridStepConsistency) {
    auto pixels = generateVirtualPixels(3, 10);
    ASSERT_GT(pixels.size(), 0u);

    auto result1 = runProcess(PlaneMapMethod::Projective, pixels);
    ASSERT_TRUE(result1.success) << result1.message;

    params_.gridStep = 1.0f;
    auto result2 = runProcess(PlaneMapMethod::Projective, pixels);
    ASSERT_TRUE(result2.success) << result2.message;

    EXPECT_GE(result1.totalPairs, result2.totalPairs);
}

TEST_F(PlaneMapCudaTest, LineIdPreserved) {
    auto pixels = generateVirtualPixels(5, 12);
    ASSERT_GT(pixels.size(), 0u);

    auto result = runProcess(PlaneMapMethod::Projective, pixels);
    ASSERT_TRUE(result.success) << result.message;

    ASSERT_NE(result.d_left_to_right, nullptr);
    cv::Mat h_ltr;
    result.d_left_to_right->download(h_ltr);
    h_ltr = h_ltr.reshape(4);

    std::set<int> lineIds;
    int M = h_ltr.rows * h_ltr.cols;
    for (int i = 0; i < M; ++i) {
        cv::Vec4f v = h_ltr.at<cv::Vec4f>(i);
        int lid = static_cast<int>(v[3]);
        EXPECT_GE(lid, 0);
        EXPECT_LT(lid, 5);
        lineIds.insert(lid);
    }
    EXPECT_GE(static_cast<int>(lineIds.size()), 2);
}

#endif // WITH_CUDA_TESTS
