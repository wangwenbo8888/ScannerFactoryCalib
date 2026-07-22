/**
 * @file test_laser_reconstruct_cuda.cpp
 * @brief 激光线三维重建CUDA算子单元测试
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "laser_reconstruct_cuda.h"

using namespace calib;


static cv::Mat makeTestQ(double cx, double cy, double f, double baseline_inv) {
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64FC1);
    Q.at<double>(0, 0) = 1.0;  Q.at<double>(0, 3) = -cx;
    Q.at<double>(1, 1) = 1.0;  Q.at<double>(1, 3) = -cy;
    Q.at<double>(2, 3) = f;
    Q.at<double>(3, 2) = baseline_inv;
    Q.at<double>(3, 3) = 0.0;
    return Q;
}

class LaserReconstructTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.minDepth = 0.0f;
        params_.maxDepth = 10000.0f;
        recon_.reset(new LaserReconstructCuda(params_));
        Q_ = makeTestQ(400.0, 300.0, 800.0, 0.1);
    }

    void TearDown() override {
        recon_.reset();
    }

    LaserReconstructResult runReconstruct(
        const std::vector<cv::Point2f>& left_pts,
        const std::vector<cv::Point2f>& right_pts,
        const std::vector<int>& fids,
        const cv::Mat& Q)
    {
        cv::Mat h_left(1, static_cast<int>(left_pts.size()), CV_32FC2,
                       const_cast<cv::Point2f*>(left_pts.data()));
        cv::Mat h_right(1, static_cast<int>(right_pts.size()), CV_32FC2,
                        const_cast<cv::Point2f*>(right_pts.data()));
        cv::Mat h_fids(1, static_cast<int>(fids.size()), CV_32SC1,
                       const_cast<int*>(fids.data()));

        cv::cuda::GpuMat d_left, d_right, d_fids;
        if (!h_left.empty()) d_left.upload(h_left);
        if (!h_right.empty()) d_right.upload(h_right);
        if (!h_fids.empty()) d_fids.upload(h_fids);

        return recon_->Execute(d_left, d_right, d_fids, Q);
    }

    LaserReconstructResult runReconstruct(
        const std::vector<cv::Point2f>& left_pts,
        const std::vector<cv::Point2f>& right_pts,
        const std::vector<int>& fids)
    {
        return runReconstruct(left_pts, right_pts, fids, Q_);
    }

    std::vector<cv::Vec3f> getPoints3d(const LaserReconstructResult& r) {
        std::vector<cv::Vec3f> out;
        if (!r.success || !r.d_points3d || r.d_points3d->empty()) {
            return out;
        }
        cv::Mat h_out;
        r.d_points3d->download(h_out);
        h_out = h_out.reshape(3);
        out.assign(h_out.begin<cv::Vec3f>(), h_out.end<cv::Vec3f>());
        return out;
    }

    std::vector<int> getValidFids(const LaserReconstructResult& r) {
        std::vector<int> out;
        if (!r.success || !r.d_valid_line_ids || r.d_valid_line_ids->empty()) {
            return out;
        }
        cv::Mat h_out;
        r.d_valid_line_ids->download(h_out);
        h_out = h_out.reshape(1);
        out.assign(h_out.begin<int>(), h_out.end<int>());
        return out;
    }

    std::unique_ptr<LaserReconstructCuda> recon_;
    LaserReconstructParams params_;
    cv::Mat Q_;
};

#ifdef WITH_CUDA_TESTS

TEST_F(LaserReconstructTest, EmptyInput) {
    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<int> fids;
    auto result = runReconstruct(left_pts, right_pts, fids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.validCount, 0);
}

TEST_F(LaserReconstructTest, SinglePairReconstruct) {
    // cx=400, cy=300, f=800, baseline_inv=0.1
    // Left (450,310), Right (430,310) -> d=20
    // W = 0.1*20 = 2.0, X = 50/2=25, Y = 10/2=5, Z = 800/2=400
    auto result = runReconstruct(
        {{450.0f, 310.0f}},
        {{430.0f, 310.0f}},
        {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.validCount, 1);

    auto pts = getPoints3d(result);
    ASSERT_EQ(pts.size(), 1u);
    EXPECT_NEAR(pts[0][0], 25.0f, 0.01f);
    EXPECT_NEAR(pts[0][1], 5.0f, 0.01f);
    EXPECT_NEAR(pts[0][2], 400.0f, 0.01f);
}

TEST_F(LaserReconstructTest, ZeroDisparity) {
    // left.x == right.x -> d=0 -> W = 0.1*0 = 0 -> filtered
    auto result = runReconstruct(
        {{450.0f, 310.0f}},
        {{450.0f, 310.0f}},
        {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.validCount, 0);
}

TEST_F(LaserReconstructTest, NegativeDepth) {
    // Points with very large disparity produce Z values outside the depth range
    // We test with Q matrix where negative depth would be produced
    // With our Q: baseline_inv=0.1, d>0 always gives positive W and Z
    // To get negative Z, we need negative baseline_inv
    cv::Mat Q_neg = makeTestQ(400.0, 300.0, 800.0, -0.1);
    auto result = runReconstruct(
        {{450.0f, 310.0f}},
        {{430.0f, 310.0f}},
        {0},
        Q_neg);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.validCount, 0);
}

TEST_F(LaserReconstructTest, DepthRangeFilter) {
    // Z = f / (baseline_inv * d) = 800 / (0.1 * d) = 8000 / d
    // d=20 -> Z=400, d=40 -> Z=200, d=10 -> Z=800
    // Set minDepth=300, maxDepth=500 -> only Z=400 passes
    LaserReconstructParams rp;
    rp.minDepth = 300.0f;
    rp.maxDepth = 500.0f;
    recon_.reset(new LaserReconstructCuda(rp));

    auto result = runReconstruct(
        {{450.0f, 310.0f}, {450.0f, 310.0f}, {450.0f, 310.0f}},
        {{430.0f, 310.0f}, {410.0f, 310.0f}, {440.0f, 310.0f}},
        {0, 0, 0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.validCount, 1);

    auto pts = getPoints3d(result);
    ASSERT_EQ(pts.size(), 1u);
    EXPECT_NEAR(pts[0][2], 400.0f, 0.01f);
}

TEST_F(LaserReconstructTest, MultiLineReconstruct) {
    // 3 line_ids, 2 points each -> 6 total, all valid
    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<int> fids;
    for (int fid = 0; fid < 3; ++fid) {
        left_pts.push_back({450.0f, 310.0f});
        right_pts.push_back({430.0f, 310.0f});
        fids.push_back(fid);
        left_pts.push_back({460.0f, 320.0f});
        right_pts.push_back({440.0f, 320.0f});
        fids.push_back(fid);
    }

    auto result = runReconstruct(left_pts, right_pts, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.validCount, 6);

    auto out_fids = getValidFids(result);
    ASSERT_EQ(out_fids.size(), 6u);
    int counts[3] = {0, 0, 0};
    for (int fid : out_fids) {
        ASSERT_GE(fid, 0);
        ASSERT_LE(fid, 2);
        counts[fid]++;
    }
    EXPECT_EQ(counts[0], 2);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 2);
}

TEST_F(LaserReconstructTest, LargeScale100K) {
    const int N = 100000;
    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<int> fids;
    left_pts.reserve(N);
    right_pts.reserve(N);
    fids.reserve(N);

    for (int i = 0; i < N; ++i) {
        float dx = 20.0f + static_cast<float>(i % 100) * 0.1f;
        float y = 100.0f + static_cast<float>(i % 500);
        left_pts.push_back({400.0f + dx, y});
        right_pts.push_back({400.0f, y});
        fids.push_back(i % 10);
    }

    auto result = runReconstruct(left_pts, right_pts, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.validCount, N);

    auto pts = getPoints3d(result);
    ASSERT_EQ(static_cast<int>(pts.size()), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_FALSE(std::isnan(pts[i][0]));
        EXPECT_FALSE(std::isnan(pts[i][1]));
        EXPECT_FALSE(std::isnan(pts[i][2]));
        EXPECT_GT(pts[i][2], 0.0f);
    }
}

TEST_F(LaserReconstructTest, BufferReuse) {
    auto result1 = runReconstruct(
        {{450.0f, 310.0f}},
        {{430.0f, 310.0f}},
        {0});
    ASSERT_TRUE(result1.success);
    EXPECT_EQ(result1.validCount, 1);

    auto result2 = runReconstruct(
        {{460.0f, 320.0f}, {470.0f, 330.0f}},
        {{440.0f, 320.0f}, {450.0f, 330.0f}},
        {0, 1});
    ASSERT_TRUE(result2.success);
    EXPECT_EQ(result2.validCount, 2);

    auto pts1 = getPoints3d(result1);
    ASSERT_EQ(pts1.size(), 1u);
    EXPECT_NEAR(pts1[0][0], 25.0f, 0.01f);

    auto pts2 = getPoints3d(result2);
    ASSERT_EQ(pts2.size(), 2u);
}

TEST_F(LaserReconstructTest, InvalidQMatrix) {
    cv::Mat Q_bad_size = cv::Mat::zeros(3, 4, CV_64FC1);
    auto result = runReconstruct(
        {{450.0f, 310.0f}},
        {{430.0f, 310.0f}},
        {0},
        Q_bad_size);
    EXPECT_FALSE(result.success);

    cv::Mat Q_bad_type = cv::Mat::zeros(4, 4, CV_32FC1);
    result = runReconstruct(
        {{450.0f, 310.0f}},
        {{430.0f, 310.0f}},
        {0},
        Q_bad_type);
    EXPECT_FALSE(result.success);
}

#endif // WITH_CUDA_TESTS

TEST_F(LaserReconstructTest, ParamsValidation) {
    LaserReconstructParams params;
    params.minDepth = -1.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.minDepth = 0.0f;
    params.maxDepth = 0.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.maxDepth = -1.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.maxDepth = 100.0f;
    params.minDepth = 200.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.minDepth = 0.0f;
    params.maxDepth = 10000.0f;
    params.deviceId = -1;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST_F(LaserReconstructTest, ParamsJson) {
    LaserReconstructParams params;
    params.minDepth = 100.0f;
    params.maxDepth = 5000.0f;
    params.deviceId = 1;

    auto j = params.toJson();
    auto params2 = LaserReconstructParams::fromJson(j);

    EXPECT_FLOAT_EQ(params2.minDepth, 100.0f);
    EXPECT_FLOAT_EQ(params2.maxDepth, 5000.0f);
    EXPECT_EQ(params2.deviceId, 1);
}
