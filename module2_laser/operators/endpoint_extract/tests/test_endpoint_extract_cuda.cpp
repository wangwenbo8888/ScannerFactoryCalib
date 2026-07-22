/**
 * @file test_endpoint_extract_cuda.cpp
 * @brief 激光线3D端点提取CUDA算子单元测试
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "endpoint_extract_cuda.h"

using namespace calib;


class EndpointExtractTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.deviceId = 0;
        params_.maxExpectedLines = 4096;
        extractor_.reset(new EndpointExtractCuda(params_));
    }

    void TearDown() override {
        extractor_.reset();
    }

    EndpointExtractResult runExtract(
        const std::vector<cv::Vec3f>& points,
        const std::vector<int>& fids)
    {
        cv::Mat h_pts(1, static_cast<int>(points.size()), CV_32FC3,
                       const_cast<cv::Vec3f*>(points.data()));
        cv::Mat h_fids(1, static_cast<int>(fids.size()), CV_32SC1,
                        const_cast<int*>(fids.data()));

        cv::cuda::GpuMat d_pts, d_fids;
        if (!h_pts.empty()) d_pts.upload(h_pts);
        if (!h_fids.empty()) d_fids.upload(h_fids);

        return extractor_->Execute(d_pts, d_fids);
    }

    std::vector<cv::Vec3f> getEndpoints(const EndpointExtractResult& r) {
        std::vector<cv::Vec3f> out;
        if (!r.success || !r.d_endpoints || r.d_endpoints->empty()) return out;
        cv::Mat h;
        r.d_endpoints->download(h);
        h = h.reshape(3);
        out.assign(h.begin<cv::Vec3f>(), h.end<cv::Vec3f>());
        return out;
    }

    std::vector<int> getEndpointIds(const EndpointExtractResult& r) {
        std::vector<int> out;
        if (!r.success || !r.d_endpoint_ids || r.d_endpoint_ids->empty()) return out;
        cv::Mat h;
        r.d_endpoint_ids->download(h);
        h = h.reshape(1);
        out.assign(h.begin<int>(), h.end<int>());
        return out;
    }

    std::vector<int> getLineIds(const EndpointExtractResult& r) {
        std::vector<int> out;
        if (!r.success || !r.d_line_ids || r.d_line_ids->empty()) return out;
        cv::Mat h;
        r.d_line_ids->download(h);
        h = h.reshape(1);
        out.assign(h.begin<int>(), h.end<int>());
        return out;
    }

    std::unique_ptr<EndpointExtractCuda> extractor_;
    EndpointExtractParams params_;
};

#ifdef WITH_CUDA_TESTS

TEST_F(EndpointExtractTest, EmptyInput) {
    std::vector<cv::Vec3f> points;
    std::vector<int> fids;
    auto result = runExtract(points, fids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.numEndpoints, 0);
    EXPECT_EQ(result.numLines, 0);
}

TEST_F(EndpointExtractTest, SingleLineTwoPoints) {
    auto result = runExtract(
        {{0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f}},
        {0, 0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 1);
    EXPECT_EQ(result.numEndpoints, 2);

    auto eps = getEndpoints(result);
    ASSERT_EQ(eps.size(), 2u);

    float minX = std::min(eps[0][0], eps[1][0]);
    float maxX = std::max(eps[0][0], eps[1][0]);
    EXPECT_NEAR(minX, 0.0f, 0.01f);
    EXPECT_NEAR(maxX, 10.0f, 0.01f);

    auto eids = getEndpointIds(result);
    ASSERT_EQ(eids.size(), 2u);
    EXPECT_EQ(eids[0], 0);
    EXPECT_EQ(eids[1], 1);
}

TEST_F(EndpointExtractTest, SingleLineThreePoints) {
    auto result = runExtract(
        {{0.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f}},
        {0, 0, 0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 1);
    EXPECT_EQ(result.numEndpoints, 2);

    auto eps = getEndpoints(result);
    ASSERT_EQ(eps.size(), 2u);

    float minX = std::min(eps[0][0], eps[1][0]);
    float maxX = std::max(eps[0][0], eps[1][0]);
    EXPECT_NEAR(minX, 0.0f, 0.01f);
    EXPECT_NEAR(maxX, 10.0f, 0.01f);
}

TEST_F(EndpointExtractTest, SinglePoint) {
    auto result = runExtract(
        {{5.0f, 3.0f, 1.0f}},
        {0});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 1);
    EXPECT_EQ(result.numEndpoints, 2);

    auto eps = getEndpoints(result);
    ASSERT_EQ(eps.size(), 2u);
    EXPECT_NEAR(eps[0][0], 5.0f, 0.01f);
    EXPECT_NEAR(eps[0][1], 3.0f, 0.01f);
    EXPECT_NEAR(eps[0][2], 1.0f, 0.01f);
    EXPECT_NEAR(eps[1][0], 5.0f, 0.01f);
    EXPECT_NEAR(eps[1][1], 3.0f, 0.01f);
    EXPECT_NEAR(eps[1][2], 1.0f, 0.01f);
}

TEST_F(EndpointExtractTest, MultipleLines) {
    auto result = runExtract(
        {{0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f},
         {0.0f, 10.0f, 0.0f}, {0.0f, 20.0f, 0.0f},
         {0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, 20.0f}},
        {0, 0, 1, 1, 2, 2});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 3);
    EXPECT_EQ(result.numEndpoints, 6);

    auto lids = getLineIds(result);
    ASSERT_EQ(lids.size(), 3u);
    EXPECT_EQ(lids[0], 0);
    EXPECT_EQ(lids[1], 1);
    EXPECT_EQ(lids[2], 2);

    auto eids = getEndpointIds(result);
    ASSERT_EQ(eids.size(), 6u);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(eids[i], i);
    }
}

TEST_F(EndpointExtractTest, SparseFrameIds) {
    auto result = runExtract(
        {{0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f},
         {0.0f, 10.0f, 0.0f}, {0.0f, 20.0f, 0.0f}},
        {0, 0, 5, 5});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 2);
    EXPECT_EQ(result.numEndpoints, 4);

    auto lids = getLineIds(result);
    ASSERT_EQ(lids.size(), 2u);
    EXPECT_EQ(lids[0], 0);
    EXPECT_EQ(lids[1], 5);
}

TEST_F(EndpointExtractTest, DiagonalLine) {
    std::vector<cv::Vec3f> points;
    std::vector<int> fids;
    for (int i = 0; i <= 10; ++i) {
        float t = static_cast<float>(i);
        points.push_back({t, t, t});
        fids.push_back(0);
    }

    auto result = runExtract(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, 1);
    EXPECT_EQ(result.numEndpoints, 2);

    auto eps = getEndpoints(result);
    ASSERT_EQ(eps.size(), 2u);

    float minX = std::min(eps[0][0], eps[1][0]);
    float maxX = std::max(eps[0][0], eps[1][0]);
    EXPECT_NEAR(minX, 0.0f, 0.1f);
    EXPECT_NEAR(maxX, 10.0f, 0.1f);
}

TEST_F(EndpointExtractTest, LargeScale10K) {
    const int N = 10000;
    const int numLines = 50;
    std::vector<cv::Vec3f> points;
    std::vector<int> fids;
    points.reserve(N);
    fids.reserve(N);

    for (int i = 0; i < N; ++i) {
        int lid = i % numLines;
        float t = static_cast<float>(i / numLines);
        points.push_back({t + lid * 100.0f, t * 0.5f, t * 0.3f});
        fids.push_back(lid);
    }

    auto result = runExtract(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numLines, numLines);
    EXPECT_EQ(result.numEndpoints, numLines * 2);
    EXPECT_EQ(result.totalInput, N);
}

TEST_F(EndpointExtractTest, BufferReuse) {
    auto result1 = runExtract(
        {{0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f}},
        {0, 0});
    ASSERT_TRUE(result1.success);
    EXPECT_EQ(result1.numLines, 1);

    auto result2 = runExtract(
        {{0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f},
         {0.0f, 10.0f, 0.0f}, {0.0f, 20.0f, 0.0f}},
        {0, 0, 1, 1});
    ASSERT_TRUE(result2.success);
    EXPECT_EQ(result2.numLines, 2);

    auto eps2 = getEndpoints(result2);
    ASSERT_EQ(eps2.size(), 4u);
}

TEST_F(EndpointExtractTest, InvalidInputType) {
    cv::cuda::GpuMat d_bad(1, 2, CV_32FC2);
    cv::cuda::GpuMat d_fids(1, 2, CV_32SC1);

    auto result = extractor_->Execute(d_bad, d_fids);
    EXPECT_FALSE(result.success);

    cv::cuda::GpuMat d_pts(1, 2, CV_32FC3);
    cv::cuda::GpuMat d_bad_fids(1, 2, CV_32FC1);

    result = extractor_->Execute(d_pts, d_bad_fids);
    EXPECT_FALSE(result.success);
}

TEST_F(EndpointExtractTest, CountMismatch) {
    cv::Mat h_pts = (cv::Mat_<cv::Vec3f>(1, 2) << cv::Vec3f(0,0,0), cv::Vec3f(1,1,1));
    cv::Mat h_fids = (cv::Mat_<int>(1, 3) << 0, 0, 0);

    cv::cuda::GpuMat d_pts(h_pts);
    cv::cuda::GpuMat d_fids(h_fids);

    auto result = extractor_->Execute(d_pts, d_fids);
    EXPECT_FALSE(result.success);
}

#endif // WITH_CUDA_TESTS

TEST_F(EndpointExtractTest, ParamsValidation) {
    EndpointExtractParams params;
    params.deviceId = -1;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.deviceId = 0;
    params.maxExpectedLines = 0;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.maxExpectedLines = -1;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.maxExpectedLines = 100;
    EXPECT_NO_THROW(params.validate());
}

TEST_F(EndpointExtractTest, ParamsJson) {
    EndpointExtractParams params;
    params.deviceId = 1;
    params.maxExpectedLines = 2048;

    auto j = params.toJson();
    auto params2 = EndpointExtractParams::fromJson(j);

    EXPECT_EQ(params2.deviceId, 1);
    EXPECT_EQ(params2.maxExpectedLines, 2048);
}
