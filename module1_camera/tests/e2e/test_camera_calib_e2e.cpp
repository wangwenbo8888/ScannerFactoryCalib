// Task 3.4: module1 端到端测试（参数外置版）
// 合成已知内参的棋盘图组，跑完整标定链 extractChessboardCorners -> normalizeLRCornerOrder
// -> IntrinsicCalibCPU，验证 reproj 误差收敛 + 有效帧数 + 内参可恢复。
//
// 全部设置参数来自同目录 config.json（唯一参数源）：图像尺寸/棋盘格/K_gt/姿态候选表/
// 基线/帧数目标/算子开关/断言门限。config.json 随本测试入 git，缺失或缺键即工程损坏，
// GTEST_FAIL（非 SKIP）。定位用候选路径列表（同 test_camera_chain_analysis 惯例）。
//
// 合成方法（与 plan 替代方案一致，保证 K 物理自洽）：
//   1. 渲染平面黑白交替棋盘（板空间，方格足够大）
//   2. 选定 ground-truth K（无畸变 D=0）和每姿态的 (rvec, tvec)
//   3. 用 cv::projectPoints 把棋盘 4 外角投到图像 -> 得到 dstQuad
//   4. cv::warpPerspective 把平面棋盘按 dstQuad warp 成图像
// 由于所有 dstQuad 都来自同一个 K 的投影，Zhang 同伦约束自洽，calibrateCamera
// 应能极低 reproj 恢复出 K。主断言 reproj_error_mean < gates.reprojMeanMax（流水线一致性证据）。

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chessboard_corner.h"
#include "intrinsic_calib_cpu.h"

using namespace fc;
using namespace calib;
namespace fs = std::filesystem;

namespace {

// ---- config.json 参数载体（字段一一对应 JSON 键）----
struct E2EConfig {
    cv::Size imSize;
    int squaresX, squaresY, squarePx;
    double fx, fy, cx, cy;          // ground-truth 内参
    double baselineX;
    struct Pose { double yawDeg, pitchDeg, z; };
    std::vector<Pose> poses;
    int targetGood, minDetected;
    bool useCalibrateCameraRO;
    double reprojErrorThreshold;
    double reprojMeanMax, fxTolerance;
    int validFramesMin;
};

// 缺键/空值 -> 抛异常带键名（TEST 体内转 GTEST_FAIL）
template <typename T>
T requireKey(const nlohmann::json& j, const char* key) {
    if (!j.contains(key) || j.at(key).is_null())
        throw std::runtime_error(std::string("config.json missing key: ") + key);
    return j.at(key).get<T>();
}

E2EConfig loadConfig() {
    const fs::path candidates[] = {
        fs::path("module1_camera/tests/e2e/config.json"),                             // 仓库根=factory_calib 自身（独立仓库布局, repo 根直跑）
        fs::path("../../module1_camera/tests/e2e/config.json"),                       // ctest cwd=<repo>/build_*/module1_camera（独立仓库布局）
        fs::path("factory_calib/module1_camera/tests/e2e/config.json"),               // repo 根直跑（factory_calib 为子目录的工作区布局）
        fs::path("../../factory_calib/module1_camera/tests/e2e/config.json"),          // ctest cwd=build_fc1_rel/module1_camera（工作区布局）
        fs::path("../../../factory_calib/module1_camera/tests/e2e/config.json"),      // 更深构建目录
        fs::path("E:/JEAMMWARE2601001/factory_calib/module1_camera/tests/e2e/config.json"),  // 绝对兜底
    };
    fs::path found;
    for (const auto& p : candidates)
        if (fs::exists(p)) { found = p; break; }
    if (found.empty())
        throw std::runtime_error(
            "config.json not found in any candidate path "
            "(repo root / build_fc1_rel/module1_camera / absolute fallback)");

    std::ifstream f(found);
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("config.json parse error: ") + e.what());
    }

    E2EConfig c{};
    const auto image = requireKey<nlohmann::json>(j, "image");
    c.imSize = {requireKey<int>(image, "width"), requireKey<int>(image, "height")};
    const auto chess = requireKey<nlohmann::json>(j, "chessboard");
    c.squaresX = requireKey<int>(chess, "squaresX");
    c.squaresY = requireKey<int>(chess, "squaresY");
    c.squarePx = requireKey<int>(chess, "squarePx");
    const auto gt = requireKey<nlohmann::json>(j, "groundTruth");
    c.fx = requireKey<double>(gt, "fx");
    c.fy = requireKey<double>(gt, "fy");
    c.cx = requireKey<double>(gt, "cx");
    c.cy = requireKey<double>(gt, "cy");
    c.baselineX = requireKey<double>(j, "baselineX");
    const auto frames = requireKey<nlohmann::json>(j, "frames");
    c.targetGood = requireKey<int>(frames, "targetGood");
    c.minDetected = requireKey<int>(frames, "minDetected");
    const auto intr = requireKey<nlohmann::json>(j, "intrinsic");
    c.useCalibrateCameraRO = requireKey<bool>(intr, "useCalibrateCameraRO");
    c.reprojErrorThreshold = requireKey<double>(intr, "reprojErrorThreshold");
    const auto gates = requireKey<nlohmann::json>(j, "gates");
    c.reprojMeanMax = requireKey<double>(gates, "reprojMeanMax");
    c.validFramesMin = requireKey<int>(gates, "validFramesMin");
    c.fxTolerance = requireKey<double>(gates, "fxTolerance");
    for (const auto& p : requireKey<std::vector<nlohmann::json>>(j, "poses"))
        c.poses.push_back({requireKey<double>(p, "yawDeg"),
                           requireKey<double>(p, "pitchDeg"),
                           requireKey<double>(p, "z")});
    return c;
}

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
    // OpenCV 相机系：x->右, y->下, z->前。
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
    // 参数唯一源：同目录 config.json（缺失/缺键 -> FAIL）
    E2EConfig cfg;
    try {
        cfg = loadConfig();
    } catch (const std::exception& e) {
        GTEST_FAIL() << "config.json load failure: " << e.what();
        return;
    }

    cv::Size patSize(cfg.squaresX - 1, cfg.squaresY - 1);  // 内角点数 = 方格数-1
    float boardW = (float)(cfg.squaresX * cfg.squarePx);
    float boardH = (float)(cfg.squaresY * cfg.squarePx);

    // Ground-truth 内参（无畸变）
    cv::Mat K_gt = (cv::Mat_<double>(3, 3) <<
        cfg.fx, 0.0,  cfg.cx,
        0.0,   cfg.fy, cfg.cy,
        0.0,   0.0,   1.0);
    cv::Mat D_gt = cv::Mat::zeros(5, 1, CV_64F);

    cv::Mat flat = makeFlatBoard(cfg.squaresX, cfg.squaresY, cfg.squarePx);

    const double deg = CV_PI / 180.0;
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    int goodFrames = 0;
    for (const auto& p : cfg.poses) {
        if (goodFrames >= cfg.targetGood) break;

        auto dstL = poseToDstQuad(p.yawDeg * deg, p.pitchDeg * deg, p.z,
                                  boardW, boardH, 0.0, K_gt, D_gt);
        auto dstR = poseToDstQuad(p.yawDeg * deg, p.pitchDeg * deg, p.z,
                                  boardW, boardH, cfg.baselineX, K_gt, D_gt);
        if (!quadIsValid(dstL, cfg.imSize) || !quadIsValid(dstR, cfg.imSize)) continue;

        cv::Mat imgL = renderPose(flat, cfg.imSize, dstL);
        cv::Mat imgR = renderPose(flat, cfg.imSize, dstR);

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
    ASSERT_GE(goodFrames, cfg.minDetected) << "too few frames where SB detected the board";

    // 跑完整内参标定
    IntrinsicCalibParams ip;
    ip.chessboard_width = patSize.width;
    ip.chessboard_height = patSize.height;
    ip.square_size_mm = cfg.squarePx;   // 用像素当 mm（合成图无真实物理尺寸，只测一致性）
    ip.image_width = cfg.imSize.width;
    ip.image_height = cfg.imSize.height;
    ip.use_calibrateCameraRO = cfg.useCalibrateCameraRO;
    ip.reproj_error_threshold = cfg.reprojErrorThreshold;
    IntrinsicCalibCPU intrin(ip);
    IntrinsicCalibResult res;
    ASSERT_TRUE(intrin.Execute(lpts, rpts, res));
    ASSERT_TRUE(res.success);

    // 主断言：reproj 误差低（证明整条流水线一致收敛）
    EXPECT_LT(res.reproj_error_mean, cfg.reprojMeanMax)
        << "reproj_mean=" << res.reproj_error_mean;
    EXPECT_GE(res.valid_frames_count, cfg.validFramesMin);
    EXPECT_TRUE(res.left.isValid() && res.right.isValid());

    // 次断言：恢复出的 fx/fy 应接近 ground-truth（合成图物理自洽，应能恢复）
    ASSERT_EQ(res.left.camera_matrix.rows, 3);
    ASSERT_EQ(res.left.camera_matrix.cols, 3);
    double fx_L = res.left.camera_matrix.at<double>(0, 0);
    double fy_L = res.left.camera_matrix.at<double>(1, 1);
    double fx_R = res.right.camera_matrix.at<double>(0, 0);
    double fy_R = res.right.camera_matrix.at<double>(1, 1);
    // fxTolerance 容差：合成投影+SB 亚像素会引入小量噪声
    EXPECT_NEAR(fx_L, cfg.fx, cfg.fxTolerance) << "fx_L=" << fx_L;
    EXPECT_NEAR(fy_L, cfg.fy, cfg.fxTolerance) << "fy_L=" << fy_L;
    EXPECT_NEAR(fx_R, cfg.fx, cfg.fxTolerance) << "fx_R=" << fx_R;
    EXPECT_NEAR(fy_R, cfg.fy, cfg.fxTolerance) << "fy_R=" << fy_R;
}
