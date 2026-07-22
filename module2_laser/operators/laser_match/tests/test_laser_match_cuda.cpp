/**
 * @file test_laser_match_cuda.cpp
 * @brief 婵€鍏夌嚎鍖归厤CUDA绠楀瓙鍗曞厓娴嬭瘯
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "laser_match_cuda.h"

using namespace calib;

class LaserMatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        LaserMatchParams params;
        params.epipolar_row_step = 0.5f;
        params.min_disparity = 0.0f;
        params.max_disparity = 500.0f;
        matcher_.reset(new LaserMatchCuda(params));
    }

    void TearDown() override {
        matcher_.reset();
    }

    std::unique_ptr<LaserMatchCuda> matcher_;
    cv::cuda::Stream stream_;

    LaserMatchResult runMatch(
        const std::vector<cv::Point2f>& left_pts,
        const std::vector<int>& left_fids,
        const std::vector<cv::Point2f>& right_pts,
        const std::vector<int>& right_fids)
    {
        cv::Mat h_left(1, static_cast<int>(left_pts.size()), CV_32FC2, const_cast<cv::Point2f*>(left_pts.data()));
        cv::Mat h_left_fids(1, static_cast<int>(left_fids.size()), CV_32SC1, const_cast<int*>(left_fids.data()));
        cv::Mat h_right(1, static_cast<int>(right_pts.size()), CV_32FC2, const_cast<cv::Point2f*>(right_pts.data()));
        cv::Mat h_right_fids(1, static_cast<int>(right_fids.size()), CV_32SC1, const_cast<int*>(right_fids.data()));

        cv::cuda::GpuMat d_left, d_left_fids, d_right, d_right_fids;
        if (!h_left.empty()) d_left.upload(h_left);
        if (!h_left_fids.empty()) d_left_fids.upload(h_left_fids);
        if (!h_right.empty()) d_right.upload(h_right);
        if (!h_right_fids.empty()) d_right_fids.upload(h_right_fids);

        auto result = matcher_->Execute(d_left, d_left_fids, d_right, d_right_fids, stream_);
        stream_.waitForCompletion();
        return result;
    }

    std::vector<cv::Point2f> getMatchedLeft(const LaserMatchResult& r) {
        std::vector<cv::Point2f> out;
        if (!r.success || !r.d_matched_left || r.d_matched_left->empty()) {
            return out;
        }
        cv::Mat h_out;
        r.d_matched_left->download(h_out);
        h_out = h_out.reshape(2);
        out.assign(h_out.begin<cv::Point2f>(), h_out.end<cv::Point2f>());
        return out;
    }

    std::vector<cv::Point2f> getMatchedRight(const LaserMatchResult& r) {
        std::vector<cv::Point2f> out;
        if (!r.success || !r.d_matched_right || r.d_matched_right->empty()) {
            return out;
        }
        cv::Mat h_out;
        r.d_matched_right->download(h_out);
        h_out = h_out.reshape(2);
        out.assign(h_out.begin<cv::Point2f>(), h_out.end<cv::Point2f>());
        return out;
    }

    std::vector<int> getMatchedFids(const LaserMatchResult& r) {
        std::vector<int> out;
        if (!r.success || !r.d_matched_line_ids || r.d_matched_line_ids->empty()) {
            return out;
        }
        cv::Mat h_out;
        r.d_matched_line_ids->download(h_out);
        h_out = h_out.reshape(1);
        out.assign(h_out.begin<int>(), h_out.end<int>());
        return out;
    }
};

TEST_F(LaserMatchTest, EmptyInput) {
    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<int> left_fids, right_fids;
    auto result = runMatch(left_pts, left_fids, right_pts, right_fids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 0);
}

TEST_F(LaserMatchTest, SinglePairExactMatch) {
    auto result = runMatch(
        {{200.0f, 1.5f}}, {0},
        {{100.0f, 1.5f}}, {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 1);

    auto left = getMatchedLeft(result);
    auto right = getMatchedRight(result);
    auto fids = getMatchedFids(result);
    ASSERT_EQ(left.size(), 1u);
    ASSERT_EQ(right.size(), 1u);
    ASSERT_EQ(fids.size(), 1u);
    EXPECT_FLOAT_EQ(left[0].x, 200.0f);
    EXPECT_FLOAT_EQ(left[0].y, 1.5f);
    EXPECT_FLOAT_EQ(right[0].x, 100.0f);
    EXPECT_FLOAT_EQ(right[0].y, 1.5f);
    EXPECT_EQ(fids[0], 0);
}

TEST_F(LaserMatchTest, FrameIdMismatch) {
    auto result = runMatch(
        {{200.0f, 1.5f}}, {0},
        {{100.0f, 1.5f}}, {1});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 0);
}

TEST_F(LaserMatchTest, RowMismatch) {
    auto result = runMatch(
        {{200.0f, 1.5f}}, {0},
        {{100.0f, 2.5f}}, {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 0);
}

TEST_F(LaserMatchTest, DisparityOutOfRange) {
    auto result = runMatch(
        {{900.0f, 1.5f}}, {0},
        {{100.0f, 1.5f}}, {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 0);
}

TEST_F(LaserMatchTest, DisparityBelowMin) {
    auto result = runMatch(
        {{50.0f, 1.5f}}, {0},
        {{100.0f, 1.5f}}, {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 0);
}

TEST_F(LaserMatchTest, MultiLineMultiRow) {
    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<int> left_fids, right_fids;

    for (int fid = 0; fid < 3; ++fid) {
        for (float y : {1.0f, 2.0f, 3.0f}) {
            left_pts.push_back({200.0f, y});
            left_fids.push_back(fid);
            right_pts.push_back({100.0f, y});
            right_fids.push_back(fid);
        }
    }

    auto result = runMatch(left_pts, left_fids, right_pts, right_fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 9);

    auto fids = getMatchedFids(result);
    ASSERT_EQ(fids.size(), 9u);
    for (int fid : fids) {
        EXPECT_GE(fid, 0);
        EXPECT_LE(fid, 2);
    }
}

TEST_F(LaserMatchTest, BufferReuse) {
    std::vector<cv::Point2f> left_pts = {{200.0f, 1.5f}};
    std::vector<int> left_fids = {0};
    std::vector<cv::Point2f> right_pts = {{100.0f, 1.5f}};
    std::vector<int> right_fids = {0};

    auto result1 = runMatch(left_pts, left_fids, right_pts, right_fids);
    ASSERT_TRUE(result1.success);
    EXPECT_EQ(result1.matchCount, 1);

    auto result2 = runMatch(left_pts, left_fids, right_pts, right_fids);
    ASSERT_TRUE(result2.success);
    EXPECT_EQ(result2.matchCount, 1);
}

TEST_F(LaserMatchTest, LargeScale) {
    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<int> left_fids, right_fids;

    for (int i = 0; i < 10000; ++i) {
        float y = static_cast<float>(i % 100) * 0.5f;
        int fid = i % 5;
        left_pts.push_back({200.0f + static_cast<float>(i % 3), y});
        left_fids.push_back(fid);
        right_pts.push_back({100.0f + static_cast<float>(i % 3), y});
        right_fids.push_back(fid);
    }

    auto result = runMatch(left_pts, left_fids, right_pts, right_fids);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.matchCount, 0);

    auto left = getMatchedLeft(result);
    auto right = getMatchedRight(result);
    auto fids = getMatchedFids(result);
    EXPECT_EQ(static_cast<int>(left.size()), result.matchCount);
    EXPECT_EQ(static_cast<int>(right.size()), result.matchCount);
    EXPECT_EQ(static_cast<int>(fids.size()), result.matchCount);

    for (int i = 0; i < result.matchCount; ++i) {
        EXPECT_GE(fids[i], 0);
        EXPECT_LT(fids[i], 5);
        float disp = left[i].x - right[i].x;
        EXPECT_GE(disp, 0.0f);
        EXPECT_LE(disp, 500.0f);
    }
}

TEST_F(LaserMatchTest, ParamsValidation) {
    LaserMatchParams params;
    params.epipolar_row_step = -1.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.epipolar_row_step = 0.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.epipolar_row_step = 0.5f;
    params.min_disparity = -1.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.min_disparity = 0.0f;
    params.max_disparity = 0.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.max_disparity = -1.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.max_disparity = 500.0f;
    params.deviceId = -1;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST_F(LaserMatchTest, ParamsJson) {
    LaserMatchParams params;
    params.epipolar_row_step = 0.25f;
    params.min_disparity = 10.0f;
    params.max_disparity = 300.0f;
    params.deviceId = 1;

    auto j = params.toJson();
    auto params2 = LaserMatchParams::fromJson(j);

    EXPECT_FLOAT_EQ(params2.epipolar_row_step, 0.25f);
    EXPECT_FLOAT_EQ(params2.min_disparity, 10.0f);
    EXPECT_FLOAT_EQ(params2.max_disparity, 300.0f);
    EXPECT_EQ(params2.deviceId, 1);
}

TEST_F(LaserMatchTest, SetParamsAndProcess) {
    LaserMatchParams newParams;
    newParams.epipolar_row_step = 0.5f;
    newParams.min_disparity = 0.0f;
    newParams.max_disparity = 50.0f;
    matcher_->SetParams(newParams);

    auto result = runMatch(
        {{200.0f, 1.5f}}, {0},
        {{100.0f, 1.5f}}, {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 0);
}

TEST_F(LaserMatchTest, MixedMatchAndMismatch) {
    std::vector<cv::Point2f> left_pts = {
        {200.0f, 1.5f},
        {200.0f, 2.5f},
        {200.0f, 3.5f}
    };
    std::vector<int> left_fids = {0, 0, 0};

    std::vector<cv::Point2f> right_pts = {
        {100.0f, 1.5f},
        {100.0f, 2.5f},
        {100.0f, 1.5f},
        {100.0f, 3.5f}
    };
    std::vector<int> right_fids = {0, 0, 1, 0};

    auto result = runMatch(left_pts, left_fids, right_pts, right_fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.matchCount, 3);

    auto fids = getMatchedFids(result);
    ASSERT_EQ(fids.size(), 3u);
    for (int fid : fids) {
        EXPECT_EQ(fid, 0);
    }
}
