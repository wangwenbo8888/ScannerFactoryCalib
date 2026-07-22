/**
 * @file test_epipolar_interp_cuda.cpp
 * @brief 婵€鍏変腑蹇冪偣鏋佺嚎鎻掑€糃UDA绠楀瓙鍗曞厓娴嬭瘯
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "epipolar_interp_cuda.h"

using namespace calib;

class EpipolarInterpTest : public ::testing::Test {
protected:
    void SetUp() override {
        EpipolarInterpParams params;
        params.epipolar_row_step = 0.5f;
        params.max_x_diff = 1.0f;
        params.max_y_span = 2.0f;
        interp_.reset(new EpipolarInterpCuda(params));
    }

    void TearDown() override {
        interp_.reset();
    }

    std::unique_ptr<EpipolarInterpCuda> interp_;
    cv::cuda::Stream stream_;

    EpipolarInterpResult runInterp(
        const std::vector<cv::Point2f>& points,
        const std::vector<int>& line_ids)
    {
        cv::Mat h_points(1, static_cast<int>(points.size()), CV_32FC2, const_cast<cv::Point2f*>(points.data()));
        cv::Mat h_fids(1, static_cast<int>(line_ids.size()), CV_32SC1, const_cast<int*>(line_ids.data()));

        cv::cuda::GpuMat d_points, d_fids;
        if (!h_points.empty()) d_points.upload(h_points);
        if (!h_fids.empty()) d_fids.upload(h_fids);

        auto result = interp_->Execute(d_points, d_fids, stream_);
        stream_.waitForCompletion();
        return result;
    }

    std::vector<cv::Point2f> getResultPoints(const EpipolarInterpResult& result) {
        std::vector<cv::Point2f> out;
        if (!result.success || !result.d_interpPoints || result.d_interpPoints->empty()) {
            return out;
        }
        cv::Mat h_out;
        result.d_interpPoints->download(h_out);
        h_out = h_out.reshape(2);
        out.assign(h_out.begin<cv::Point2f>(), h_out.end<cv::Point2f>());
        return out;
    }
};

TEST_F(EpipolarInterpTest, EmptyInput) {
    std::vector<cv::Point2f> points;
    std::vector<int> fids;
    auto result = runInterp(points, fids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, SinglePoint) {
    auto result = runInterp({{100.0f, 200.0f}}, {0});
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, BasicInterpolation_Step05) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 1);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);

    EXPECT_NEAR(out[0].y, 1.5f, 1e-4f);
    float t = (1.5f - 1.2f) / (1.8f - 1.2f);
    float expected_x = 100.0f + t * (100.3f - 100.0f);
    EXPECT_NEAR(out[0].x, expected_x, 1e-3f);
}

TEST_F(EpipolarInterpTest, CrossFrameRejection) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 1};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, LargeXDiffRejection) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {102.0f, 1.8f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, LargeYSpanRejection) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 3.5f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, TargetOnMinBoundary) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.5f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, MultipleValidPairs) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f},
        {100.5f, 2.3f},
        {100.7f, 2.9f}
    };
    std::vector<int> fids = {0, 0, 0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 3);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 3u);

    EXPECT_NEAR(out[0].y, 1.5f, 1e-4f);
    EXPECT_NEAR(out[1].y, 2.0f, 1e-4f);
    EXPECT_NEAR(out[2].y, 2.5f, 1e-4f);
}

TEST_F(EpipolarInterpTest, Step1_IntegerRows) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 1.0f;
    params.max_x_diff = 1.0f;
    params.max_y_span = 2.0f;
    interp_.reset(new EpipolarInterpCuda(params));

    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 2.5f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 1);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 2.0f, 1e-4f);
}

TEST_F(EpipolarInterpTest, Step025_FineGrid) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 0.25f;
    params.max_x_diff = 1.0f;
    params.max_y_span = 2.0f;
    interp_.reset(new EpipolarInterpCuda(params));

    std::vector<cv::Point2f> points = {
        {100.0f, 1.1f},
        {100.3f, 1.6f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 1);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 1.25f, 1e-4f);
}

TEST_F(EpipolarInterpTest, NegativeCoordinates) {
    std::vector<cv::Point2f> points = {
        {-100.0f, -5.8f},
        {-99.7f, -5.2f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 1);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, -5.5f, 1e-4f);
}

TEST_F(EpipolarInterpTest, UnsortedYPair) {
    std::vector<cv::Point2f> points = {
        {100.3f, 1.8f},
        {100.0f, 1.2f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 1);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 1.5f, 1e-4f);
}

TEST_F(EpipolarInterpTest, BufferReuse) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 0};

    auto result1 = runInterp(points, fids);
    ASSERT_TRUE(result1.success);
    EXPECT_EQ(result1.interpCount, 1);

    auto result2 = runInterp(points, fids);
    ASSERT_TRUE(result2.success);
    EXPECT_EQ(result2.interpCount, 1);
}

TEST_F(EpipolarInterpTest, SameYCoordinates) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.5f},
        {100.3f, 1.5f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, LargeScale) {
    std::vector<cv::Point2f> points;
    std::vector<int> fids;
    for (int i = 0; i < 10000; ++i) {
        float y = static_cast<float>(i) * 0.3f;
        points.push_back({100.0f + static_cast<float>(i % 5) * 0.1f, y});
        fids.push_back(0);
    }

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.interpCount, 0);

    auto out = getResultPoints(result);
    EXPECT_EQ(static_cast<int>(out.size()), result.interpCount);

    for (const auto& pt : out) {
        float remainder = fmodf(pt.y, 0.5f);
        remainder = (remainder < 0) ? remainder + 0.5f : remainder;
        EXPECT_NEAR(remainder, 0.0f, 1e-4f);
    }
}

TEST_F(EpipolarInterpTest, ParamsValidation) {
    EpipolarInterpParams params;
    params.epipolar_row_step = -1.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.epipolar_row_step = 0.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.epipolar_row_step = 0.5f;
    params.max_x_diff = 0.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.max_x_diff = 1.0f;
    params.max_y_span = 0.0f;
    EXPECT_THROW(params.validate(), std::invalid_argument);

    params.max_y_span = 2.0f;
    params.deviceId = -1;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST_F(EpipolarInterpTest, ParamsJson) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 0.25f;
    params.max_x_diff = 1.5f;
    params.max_y_span = 3.0f;
    params.deviceId = 1;

    auto j = params.toJson();
    auto params2 = EpipolarInterpParams::fromJson(j);

    EXPECT_FLOAT_EQ(params2.epipolar_row_step, 0.25f);
    EXPECT_FLOAT_EQ(params2.max_x_diff, 1.5f);
    EXPECT_FLOAT_EQ(params2.max_y_span, 3.0f);
    EXPECT_EQ(params2.deviceId, 1);
}

TEST_F(EpipolarInterpTest, SetParamsAndProcess) {
    EpipolarInterpParams newParams;
    newParams.epipolar_row_step = 1.0f;
    newParams.max_x_diff = 1.0f;
    newParams.max_y_span = 2.0f;
    interp_->SetParams(newParams);

    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 2.5f}
    };
    std::vector<int> fids = {0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 1);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 2.0f, 1e-4f);
}

TEST_F(EpipolarInterpTest, ThreePointsOneFrame) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.1f, 1.4f},
        {100.2f, 1.9f}
    };
    std::vector<int> fids = {0, 0, 0};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);

    auto out = getResultPoints(result);
    for (const auto& pt : out) {
        float remainder = fmodf(fabsf(pt.y), 0.5f);
        EXPECT_NEAR(remainder, 0.0f, 1e-4f);
    }
}

// ============================================================================
// lineIdCheck parameter tests (scanning mode support)
// ============================================================================

TEST_F(EpipolarInterpTest, LineIdCheckFalse_InterpolatesAcrossDifferentLineIds) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 0.5f;
    params.max_x_diff = 1.0f;
    params.max_y_span = 2.0f;
    params.lineIdCheck = false;
    interp_.reset(new EpipolarInterpCuda(params));

    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 1};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.interpCount, 0);

    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 1.5f, 1e-4f);
}

TEST_F(EpipolarInterpTest, LineIdCheckTrue_BlocksDifferentLineIds) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 0.5f;
    params.max_x_diff = 1.0f;
    params.max_y_span = 2.0f;
    params.lineIdCheck = true;
    interp_.reset(new EpipolarInterpCuda(params));

    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 1};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, LineIdCheck_DefaultIsTrue) {
    EpipolarInterpParams params;
    EXPECT_TRUE(params.lineIdCheck);
}

TEST_F(EpipolarInterpTest, SameLineId_InterpolatesUnderBothModes) {
    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {100.3f, 1.8f}
    };
    std::vector<int> fids = {0, 0};

    EpipolarInterpParams paramsTrue;
    paramsTrue.lineIdCheck = true;
    interp_.reset(new EpipolarInterpCuda(paramsTrue));
    auto resultTrue = runInterp(points, fids);
    ASSERT_TRUE(resultTrue.success);
    EXPECT_EQ(resultTrue.interpCount, 1);

    EpipolarInterpParams paramsFalse;
    paramsFalse.lineIdCheck = false;
    interp_.reset(new EpipolarInterpCuda(paramsFalse));
    auto resultFalse = runInterp(points, fids);
    ASSERT_TRUE(resultFalse.success);
    EXPECT_EQ(resultFalse.interpCount, 1);
}

TEST_F(EpipolarInterpTest, LineIdCheckFalse_GeometricConstraintsStillApply) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 0.5f;
    params.max_x_diff = 1.0f;
    params.max_y_span = 2.0f;
    params.lineIdCheck = false;
    interp_.reset(new EpipolarInterpCuda(params));

    std::vector<cv::Point2f> points = {
        {100.0f, 1.2f},
        {102.0f, 1.8f}
    };
    std::vector<int> fids = {0, 1};

    auto result = runInterp(points, fids);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, ParamsJson_LineIdCheck) {
    EpipolarInterpParams params;
    params.lineIdCheck = false;

    auto j = params.toJson();
    ASSERT_TRUE(j.contains("lineIdCheck"));
    EXPECT_FALSE(j.at("lineIdCheck").get<bool>());

    auto params2 = EpipolarInterpParams::fromJson(j);
    EXPECT_FALSE(params2.lineIdCheck);

    nlohmann::json j2;
    j2["lineIdCheck"] = true;
    auto params3 = EpipolarInterpParams::fromJson(j2);
    EXPECT_TRUE(params3.lineIdCheck);
}
