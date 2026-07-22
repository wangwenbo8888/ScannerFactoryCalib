// Task 3.4: module1 端到端测试
// 合成已知内参的棋盘图组，跑完整标定链 extractChessboardCorners → normalizeLRCornerOrder
// → IntrinsicCalibCPU，验证 reproj 误差收敛 + 有效帧数 + 内参可恢复。
//
// 合成方法（与 plan 替代方案一致，保证 K 物理自洽）：
//   1. 渲染平面黑白交替棋盘（板空间，方格足够大）
//   2. 选定 ground-truth K（无畸变 D=0）和每姿态的 (rvec, tvec)
//   3. 用 cv::projectPoints 把棋盘 4 外角投到图像 → 得到 dstQuad
//   4. cv::warpPerspective 把平面棋盘按 dstQuad warp 成图像
// 由于所有 dstQuad 都来自同一个 K 的投影，Zhang 同伦约束自洽，calibrateCamera
// 应能极低 reproj 恢复出 K。主断言 reproj_error_mean < 1.0 px（流水线一致性证据）。

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <cmath>
#include <vector>

#include "chessboard_corner.h"
#include "intrinsic_calib_cpu.h"

using namespace fc;
using namespace calib;

namespace {

// 渲染一张平面黑白交替棋盘（无畸变、无透视，板空间）
cv::Mat makeFlatBoard(int squaresX, int squaresY, int squarePx) {
    int W = squaresX * squarePx, H = squaresY * squarePx;
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < squaresY; ++r)
        for (int c = 0; c < squaresX; ++c)
            if ((r + c) % 2 == 0) {
                cv::Rect roi(c * squarePx, r * squarePx, squarePx, squarePx);
                img(roi) = 0;
            }
    return img;
}

// 把平面棋盘按四点对应 warp 到图像里（透视一致）
cv::Mat renderPose(const cv::Mat& flatBoard, const cv::Size& imSize,
                   const std::vector<cv::Point2f>& dstQuad) {
    std::vector<cv::Point2f> srcQuad = {
        {0.f, 0.f},
        {(float)flatBoard.cols, 0.f},
        {(float)flatBoard.cols, (float)flatBoard.rows},
        {0.f, (float)flatBoard.rows}
    };
    cv::Mat H = cv::getPerspectiveTransform(srcQuad, dstQuad);
    cv::Mat out(imSize, CV_8UC1, cv::Scalar(128));  // 中灰背景
    cv::warpPerspective(flatBoard, out, H, imSize,
                        cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    return out;
}

// 给定 (yaw, pitch, Z, baselineX) 与 K/D，把棋盘 4 个外角投到图像，得到 dstQuad。
// 棋盘板心默认对齐到光轴（无旋转时落在主点附近）。
std::vector<cv::Point2f> poseToDstQuad(double yawRad, double pitchRad, double Z,
                                       float boardW, float boardH,
                                       double baselineX,
                                       const cv::Mat& K, const cv::Mat& D) {
    // OpenCV 相机系：x→右, y→下, z→前。
    // pitch 绕 x 转（板上下翻），yaw 绕 y 转（板左右转）。
    cv::Vec3d rvec(pitchRad, yawRad, 0.0);
    // 板心 (boardW/2, boardH/2, 0) 想落在主点附近：t = (-boardW/2, -boardH/2, Z)
    cv::Vec3d tvec(-boardW / 2.0 + baselineX, -boardH / 2.0, Z);

    std::vector<cv::Point3f> boardOuter = {
        {0.f, 0.f, 0.f},
        {boardW, 0.f, 0.f},
        {boardW, boardH, 0.f},
        {0.f, boardH, 0.f}
    };
    std::vector<cv::Point2f> dst;
    cv::projectPoints(boardOuter, rvec, tvec, K, D, dst);
    return dst;
}

// 检查 dstQuad 是否落在图像内（带边距）且面积非退化
bool quadIsValid(const std::vector<cv::Point2f>& q, const cv::Size& imSize,
                 int margin = 20, double minArea = 20000.0) {
    for (const auto& p : q) {
        if (p.x < margin || p.x > imSize.width - margin) return false;
        if (p.y < margin || p.y > imSize.height - margin) return false;
    }
    return cv::contourArea(q) > minArea;
}

} // namespace

TEST(CameraCalibE2E, PipelineProducesLowReprojError) {
    cv::Size imSize(1280, 960);
    int squaresX = 9, squaresY = 7;                 // 内角点 8x6
    int squarePx = 100;
    cv::Size patSize(squaresX - 1, squaresY - 1);   // 8x6
    float boardW = (float)(squaresX * squarePx);    // 900
    float boardH = (float)(squaresY * squarePx);    // 700

    // Ground-truth 内参（无畸变）
    cv::Mat K_gt = (cv::Mat_<double>(3, 3) <<
        1200.0, 0.0,    640.0,
        0.0,    1200.0, 480.0,
        0.0,    0.0,    1.0);
    cv::Mat D_gt = cv::Mat::zeros(5, 1, CV_64F);

    cv::Mat flat = makeFlatBoard(squaresX, squaresY, squarePx);

    // 姿态候选：不同 yaw/pitch/Z 组合，保证多样性 + 大部分能被 SB 检到
    struct Pose { double yaw; double pitch; double Z; };
    const double deg = CV_PI / 180.0;
    std::vector<Pose> poses = {
        {  0 * deg,  0 * deg, 1800},
        { 15 * deg,  0 * deg, 1800},
        {-15 * deg,  0 * deg, 1800},
        {  0 * deg, 15 * deg, 1800},
        {  0 * deg,-15 * deg, 1800},
        { 22 * deg,  0 * deg, 2000},
        {-22 * deg,  0 * deg, 2000},
        {  0 * deg, 22 * deg, 2000},
        {  0 * deg,-22 * deg, 2000},
        { 18 * deg, 12 * deg, 1900},
        {-18 * deg,-12 * deg, 1900},
        { 12 * deg, 18 * deg, 1950},
        {-12 * deg,-18 * deg, 1950},
        { 25 * deg,  0 * deg, 2200},
        {-25 * deg,  0 * deg, 2200},
        {  0 * deg, 25 * deg, 2200},
        {  8 * deg,  8 * deg, 1700},
        {-8 * deg, -8 * deg, 1700},
    };
    const double baselineX = 100.0;  // L→R 的 x 基线（板坐标系下平移）

    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    int goodFrames = 0;
    int target = 12;
    for (const auto& p : poses) {
        if (goodFrames >= target) break;

        auto dstL = poseToDstQuad(p.yaw, p.pitch, p.Z, boardW, boardH, 0.0, K_gt, D_gt);
        auto dstR = poseToDstQuad(p.yaw, p.pitch, p.Z, boardW, boardH, baselineX, K_gt, D_gt);
        if (!quadIsValid(dstL, imSize) || !quadIsValid(dstR, imSize)) continue;

        cv::Mat imgL = renderPose(flat, imSize, dstL);
        cv::Mat imgR = renderPose(flat, imSize, dstR);

        ChessboardCornerParams cp; cp.patternSize = patSize;
        ChessboardCornerResult rl, rr;
        if (!extractChessboardCorners(imgL, cp, rl) ||
            !extractChessboardCorners(imgR, cp, rr)) {
            continue;  // SB 在某些角度下可能检测不到，跳过
        }
        if (!normalizeLRCornerOrder(rl, rr)) continue;
        lpts.push_back(rl.corners);
        rpts.push_back(rr.corners);
        ++goodFrames;
    }
    ASSERT_GE(goodFrames, 10) << "too few frames where SB detected the board";

    // 跑完整内参标定
    IntrinsicCalibParams ip;
    ip.chessboard_width = patSize.width;
    ip.chessboard_height = patSize.height;
    ip.square_size_mm = squarePx;   // 用像素当 mm（合成图无真实物理尺寸，只测一致性）
    ip.image_width = imSize.width;
    ip.image_height = imSize.height;
    ip.use_calibrateCameraRO = false;
    ip.reproj_error_threshold = 1.0;
    IntrinsicCalibCPU intrin(ip);
    IntrinsicCalibResult res;
    ASSERT_TRUE(intrin.Execute(lpts, rpts, res));
    ASSERT_TRUE(res.success);

    // 主断言：reproj 误差低（证明整条流水线一致收敛）
    EXPECT_LT(res.reproj_error_mean, 1.0)
        << "reproj_mean=" << res.reproj_error_mean;
    EXPECT_GE(res.valid_frames_count, 10);
    EXPECT_TRUE(res.left.isValid() && res.right.isValid());

    // 次断言：恢复出的 fx/fy 应接近 ground-truth（合成图物理自洽，应能恢复）
    ASSERT_EQ(res.left.camera_matrix.rows, 3);
    ASSERT_EQ(res.left.camera_matrix.cols, 3);
    double fx_L = res.left.camera_matrix.at<double>(0, 0);
    double fy_L = res.left.camera_matrix.at<double>(1, 1);
    double fx_R = res.right.camera_matrix.at<double>(0, 0);
    double fy_R = res.right.camera_matrix.at<double>(1, 1);
    // 5% 容差：合成投影+SB 亚像素会引入小量噪声
    EXPECT_NEAR(fx_L, 1200.0, 60.0) << "fx_L=" << fx_L;
    EXPECT_NEAR(fy_L, 1200.0, 60.0) << "fy_L=" << fy_L;
    EXPECT_NEAR(fx_R, 1200.0, 60.0) << "fx_R=" << fx_R;
    EXPECT_NEAR(fy_R, 1200.0, 60.0) << "fy_R=" << fy_R;
}
