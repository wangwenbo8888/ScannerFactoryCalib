/**
 * @file test_epipolar_interp_cuda.cpp
 * @brief 激光中心点极线插值CUDA算子单元测试（v2 极线驱动重采样版）
 *
 * 覆盖: 空输入 / 精确命中直接采用 / 双窗最近点插值 / 缺上或缺下放弃 /
 *       X 差超限放弃 / 多候选择近 / 原始点不透传 / 多线号隔离 / 窗口边界
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "epipolar_interp_cuda.h"

#include <algorithm>

using namespace calib;

class EpipolarInterpTest : public ::testing::Test {
protected:
    void SetUp() override {
        EpipolarInterpParams params;
        params.epipolar_row_step = 0.7f;
        params.max_x_diff = 1.0f;
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

    std::vector<int> getResultFids(const EpipolarInterpResult& result) {
        std::vector<int> out;
        if (!result.success || !result.d_interp_line_ids || result.d_interp_line_ids->empty()) {
            return out;
        }
        cv::Mat h_out;
        result.d_interp_line_ids->download(h_out);
        h_out = h_out.reshape(1);
        out.assign(h_out.begin<int>(), h_out.end<int>());
        return out;
    }
};

// ---------- 空输入 ----------
TEST_F(EpipolarInterpTest, EmptyInput) {
    std::vector<cv::Point2f> points;
    std::vector<int> line_ids;
    auto result = runInterp(points, line_ids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);
}

TEST_F(EpipolarInterpTest, SinglePointNoOutput) {
    auto result = runInterp({{10.0f, 5.0f}}, {1});
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.interpCount, 0);   // 单点无上下对, 且不透传原始点
}

// ---------- 精确命中: 点恰在极线上直接采用 ----------
TEST_F(EpipolarInterpTest, ExactHitAdopted) {
    // step=0.7, y=2.8 = 4*0.7 恰在极线上
    // 上下各放一个干扰点, 命中点 x=50
    std::vector<cv::Point2f> pts = {
        {49.2f, 2.1f},   // 上窗
        {50.0f, 2.8f},   // 精确命中
        {50.8f, 3.5f}    // 下窗
    };
    auto result = runInterp(pts, {1, 1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    // y=2.1..3.5 覆盖极线 2.1(3*0.7), 2.8(4*0.7), 3.5(5*0.7)
    ASSERT_EQ(out.size(), 3u);
    // 极线 2.8 应精确命中 x=50
    bool found = false;
    for (auto& p : out)
        if (std::abs(p.y - 2.8f) < 1e-3f) {
            EXPECT_NEAR(p.x, 50.0f, 1e-3f);
            found = true;
        }
    EXPECT_TRUE(found);
}

// ---------- 双窗插值: 连线与极线交点 ----------
TEST_F(EpipolarInterpTest, TwoWindowIntersection) {
    // 极线 y=2.1 (3*0.7); 上点 (30, 2.0), 下点 (31, 2.4)
    // 交点: t=(2.1-2.0)/0.4=0.25, x=30+0.25*1=30.25
    std::vector<cv::Point2f> pts = {
        {30.0f, 2.0f},
        {31.0f, 2.4f}
    };
    auto result = runInterp(pts, {1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 2.1f, 1e-3f);
    EXPECT_NEAR(out[0].x, 30.25f, 1e-2f);
}

// ---------- 缺上点: 放弃 ----------
TEST_F(EpipolarInterpTest, MissingUpperAbandoned) {
    // 极线 2.8: 上窗(2.1~2.8)无点, 只有下点 (40, 3.0)
    std::vector<cv::Point2f> pts = {
        {40.0f, 3.0f}
    };
    auto result = runInterp(pts, {1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    // 极线 2.8 与 3.5: 2.8 缺上; 3.5 上窗(2.8,3.5)有点3.0但缺下 -> 均无产出
    EXPECT_EQ(out.size(), 0u);
}

// ---------- 缺下点: 放弃 ----------
TEST_F(EpipolarInterpTest, MissingLowerAbandoned) {
    std::vector<cv::Point2f> pts = {
        {40.0f, 2.0f}
    };
    auto result = runInterp(pts, {1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    EXPECT_EQ(out.size(), 0u);
}

// ---------- X 差超限: 放弃 ----------
TEST_F(EpipolarInterpTest, XDiffExceededAbandoned) {
    // 极线 2.1: 上 (30, 2.0), 下 (32, 2.4) |dx|=2 > 1
    std::vector<cv::Point2f> pts = {
        {30.0f, 2.0f},
        {32.0f, 2.4f}
    };
    auto result = runInterp(pts, {1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    EXPECT_EQ(out.size(), 0u);
}

// ---------- 多候选择近 ----------
TEST_F(EpipolarInterpTest, MultipleCandidatesNearestWins) {
    // 极线 2.8: 上窗两点 y=2.0(x=45) 与 y=2.75(x=50) -> 取 2.75 (距 0.05)
    //           下窗两点 y=3.4(x=55) 与 y=2.85(x=52) -> 取 2.85 (距 0.05)
    // 近候选对 X 差 |50-52|=2 会超限 -> 用 50/50.5: 差 0.5 合法
    //   上近 (50.0, 2.75), 下近 (50.5, 2.85), 远候选故意 X 偏离大
    //   交点 x = 50 + (2.8-2.75)/(2.85-2.75)*(50.5-50) = 50 + 0.5*0.5 = 50.25
    // 极线 2.1: 上窗无点(2.0 在窗? 距 0.1 是的) -> 上=2.0(x=45), 下=2.75(x=50) 差 5 超限 -> 无产出
    // 极线 3.5: 上=3.4(距 0.1, x=55), 下窗无 -> 无产出
    std::vector<cv::Point2f> pts = {
        {45.0f, 2.0f},    // 上窗远候选
        {50.0f, 2.75f},   // 上窗近候选
        {50.5f, 2.85f},   // 下窗近候选
        {55.0f, 3.4f}     // 下窗远候选
    };
    auto result = runInterp(pts, {1, 1, 1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].y, 2.8f, 1e-3f);
    EXPECT_NEAR(out[0].x, 50.25f, 1e-2f);
}

// ---------- 原始点不透传 ----------
TEST_F(EpipolarInterpTest, RawPointsNotPassedThrough) {
    // 点 y=2.05, 2.45 均不在极线上(2.1, 2.8, 3.5...)
    // 输出应只有极线点 y≡k*0.7, 不含 2.05/2.45
    std::vector<cv::Point2f> pts = {
        {10.0f, 2.05f},
        {10.5f, 2.45f}
    };
    auto result = runInterp(pts, {1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);   // 仅极线 2.1
    EXPECT_NEAR(out[0].y, 2.1f, 1e-3f);
    for (auto& p : out) {
        float k = p.y / 0.7f;
        EXPECT_NEAR(k, std::round(k), 1e-2f);   // 每点严格在栅格上
    }
}

// ---------- 多线号隔离 ----------
TEST_F(EpipolarInterpTest, LineIdIsolation) {
    // 线1: 上(20,2.0) 下(20.5,2.4); 线2: 上(80,2.0) 下(80.5,2.4)
    // 极线 2.1 各自插值, 不跨线混合
    std::vector<cv::Point2f> pts = {
        {20.0f, 2.0f},
        {20.5f, 2.4f},
        {80.0f, 2.0f},
        {80.5f, 2.4f}
    };
    auto result = runInterp(pts, {1, 1, 2, 2});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    auto fids = getResultFids(result);
    ASSERT_EQ(out.size(), 2u);
    // 按 x 排序后核对
    std::vector<std::pair<float, int>> px;
    for (size_t i = 0; i < out.size(); ++i) px.emplace_back(out[i].x, fids[i]);
    std::sort(px.begin(), px.end());
    EXPECT_NEAR(px[0].first, 20.125f, 1e-2f);   // t=0.25 -> 20+0.25*0.5
    EXPECT_EQ(px[0].second, 1);
    EXPECT_NEAR(px[1].first, 80.125f, 1e-2f);
    EXPECT_EQ(px[1].second, 2);
}

// ---------- 窗口边界: 恰在窗口边缘的点可用 ----------
TEST_F(EpipolarInterpTest, WindowBoundaryInclusive) {
    // 极线 2.8, 上点 y=2.1 (恰为 y_k-0.7), 下点 y=3.5 (恰为 y_k+0.7)
    // 闭窗口: 均有效; 交点 x = 60 + (2.8-2.1)/(3.5-2.1)*(61-60) = 60.5
    std::vector<cv::Point2f> pts = {
        {60.0f, 2.1f},
        {61.0f, 3.5f}
    };
    auto result = runInterp(pts, {1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    // 极线 2.1(命中上点), 2.8(插值), 3.5(命中下点)
    ASSERT_EQ(out.size(), 3u);
    bool found28 = false;
    for (auto& p : out)
        if (std::abs(p.y - 2.8f) < 1e-3f) {
            EXPECT_NEAR(p.x, 60.5f, 1e-2f);
            found28 = true;
        }
    EXPECT_TRUE(found28);
}

// ---------- window_offset 自定义 ----------
TEST_F(EpipolarInterpTest, CustomWindowOffset) {
    EpipolarInterpParams params;
    params.epipolar_row_step = 0.7f;
    params.window_offset = 0.35f;   // 收窄窗口
    interp_.reset(new EpipolarInterpCuda(params));

    // 极线 2.8: 上点 y=2.1 距 0.7 > 0.35 出窗; 下点 y=3.15 距 0.35 恰在窗内
    // 极线 2.1: 上点精确命中(2.1=3*0.7) -> 仍产出(命中不受窗口限制)
    // 极线 3.5: 上 3.15 距 0.35 恰在窗, 缺下 -> 无产出
    std::vector<cv::Point2f> pts = {
        {10.0f, 2.1f},
        {10.5f, 3.15f}
    };
    auto result = runInterp(pts, {1, 1});
    ASSERT_TRUE(result.success);
    auto out = getResultPoints(result);
    ASSERT_EQ(out.size(), 1u);   // 仅极线 2.1 精确命中
    EXPECT_NEAR(out[0].y, 2.1f, 1e-3f);
    EXPECT_NEAR(out[0].x, 10.0f, 1e-3f);
}

// ---------- 参数校验 ----------
TEST(EpipolarInterpParamsTest, ValidateWindowOffset) {
    EpipolarInterpParams p;
    p.window_offset = -2.0f;   // 非 -1 的负数
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(EpipolarInterpParamsTest, EffectiveWindowAuto) {
    EpipolarInterpParams p;
    p.epipolar_row_step = 0.7f;
    EXPECT_FLOAT_EQ(p.effectiveWindow(), 0.7f);   // -1 -> 取 step
    p.window_offset = 1.4f;
    EXPECT_FLOAT_EQ(p.effectiveWindow(), 1.4f);
}
