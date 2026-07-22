#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

#include "chessboard_corner.h"

using namespace fc;

namespace {
// 合成一张带已知棋盘角点的灰度图
cv::Mat makeSyntheticChessboard(int winW, int winH, int squaresX, int squaresY,
                                int squarePx, int margin) {
    int W = margin * 2 + squaresX * squarePx;
    int H = margin * 2 + squaresY * squarePx;
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < squaresY; ++r)
        for (int c = 0; c < squaresX; ++c) {
            if ((r + c) % 2 == 0) {
                cv::Rect roi(margin + c * squarePx, margin + r * squarePx, squarePx, squarePx);
                img(roi) = 0;
            }
        }
    return img;
}
} // namespace

TEST(ChessboardCorner, ExtractsAllCorners) {
    int squaresX = 12, squaresY = 9;     // 内角点 = 11x8
    int squarePx = 30, margin = 40;
    cv::Mat gray = makeSyntheticChessboard(0, 0, squaresX, squaresY, squarePx, margin);

    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX - 1, squaresY - 1);  // 11x8
    ChessboardCornerResult r;
    ASSERT_TRUE(extractChessboardCorners(gray, p, r));
    EXPECT_TRUE(r.found);
    EXPECT_EQ(r.corners.size(), static_cast<size_t>(p.patternSize.area()));
}

TEST(ChessboardCorner, FailsOnBlankImage) {
    cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(255));
    ChessboardCornerParams p;
    ChessboardCornerResult r;
    EXPECT_FALSE(extractChessboardCorners(blank, p, r));
    EXPECT_FALSE(r.found);
}

TEST(ChessboardCorner, SubpixBetterThanPixel) {
    int squaresX = 8, squaresY = 6;
    int squarePx = 40, margin = 30;
    cv::Mat gray = makeSyntheticChessboard(0, 0, squaresX, squaresY, squarePx, margin);
    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX - 1, squaresY - 1);
    ChessboardCornerResult r;
    ASSERT_TRUE(extractChessboardCorners(gray, p, r));
    // OpenCV cornerSubPix 用像素中心约定：cv::Rect 整数坐标绘制的棋盘，其 4 格交汇
    // 内角点的亚像素坐标为 (margin + squarePx - 0.5, ...)（相对板外角 margin 的 -0.5 偏移）。
    const float expectedX = static_cast<float>(margin + squarePx) - 0.5f;
    const float expectedY = static_cast<float>(margin + squarePx) - 0.5f;
    // SB 的起始角不保证左上先出 → 在集合里找 min(x+y) 的点（即左上内角点）
    auto best = std::min_element(r.corners.begin(), r.corners.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) { return (a.x + a.y) < (b.x + b.y); });
    ASSERT_NE(best, r.corners.end());
    EXPECT_NEAR(best->x, expectedX, 0.5);
    EXPECT_NEAR(best->y, expectedY, 0.5);
}

// ---- normalizeLRCornerOrder 单测（确定性，不依赖 SB）----

namespace {
// 构造一个已知走向的角点集：网格 cols×rows，从 (x0,y0) 起，步长 step
ChessboardCornerResult makeGridResult(int cols, int rows, float x0, float y0,
                                      float stepX, float stepY) {
    ChessboardCornerResult r;
    r.found = true;
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
            r.corners.emplace_back(x0 + col * stepX, y0 + row * stepY);
    return r;
}
} // namespace

TEST(NormalizeLRCornerOrder, SameDirectionNoChange) {
    // L 和 R 同向（都从左上往右下扫描）
    auto L = makeGridResult(4, 3, 10.f, 10.f, 5.f, 5.f);
    auto R = makeGridResult(4, 3, 20.f, 10.f, 5.f, 5.f);  // 基线偏移，同向
    auto R0 = R.corners[0];   // 记录原第一个
    EXPECT_TRUE(normalizeLRCornerOrder(L, R));
    // 同向 → 不应翻转，R[0] 不变
    EXPECT_EQ(R.corners[0].x, R0.x);
    EXPECT_EQ(R.corners[0].y, R0.y);
}

TEST(NormalizeLRCornerOrder, OppositeDirectionGetsReversed) {
    // L 从左上往右下，R 从右下往左上（逆序）
    auto L = makeGridResult(4, 3, 10.f, 10.f, 5.f, 5.f);
    auto R = makeGridResult(4, 3, 20.f, 10.f, 5.f, 5.f);
    std::reverse(R.corners.begin(), R.corners.end());   // 造一个反向 R
    auto Rlast_before = R.corners[0];   // 反向后第一个 = 原最后那个
    EXPECT_TRUE(normalizeLRCornerOrder(L, R));
    // 反向 → 应被翻转回来：现在 R[0] 应接近原 L[0] 方向（右上角那个的对面）
    // 校验：翻转后 R[0] 是 R 里 min(x+y) 的点（左上）
    auto best = std::min_element(R.corners.begin(), R.corners.end(),
        [](const cv::Point2f& a, const cv::Point2f& b){ return (a.x+a.y)<(b.x+b.y); });
    EXPECT_NEAR(R.corners[0].x, best->x, 1e-6);
    EXPECT_NEAR(R.corners[0].y, best->y, 1e-6);
}

TEST(NormalizeLRCornerOrder, RejectsSizeMismatch) {
    auto L = makeGridResult(4, 3, 10.f, 10.f, 5.f, 5.f);
    auto R = makeGridResult(3, 3, 20.f, 10.f, 5.f, 5.f);  // 不同大小
    EXPECT_FALSE(normalizeLRCornerOrder(L, R));
}

TEST(NormalizeLRCornerOrder, RejectsNotFound) {
    auto L = makeGridResult(4, 3, 10.f, 10.f, 5.f, 5.f);
    ChessboardCornerResult R;  // found=false 默认
    EXPECT_FALSE(normalizeLRCornerOrder(L, R));
}
