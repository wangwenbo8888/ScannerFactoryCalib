#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "intrinsic_calib_cpu.h"

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace calib;

namespace {

std::vector<cv::Point3f> makeObjectPoints(int w, int h, double sz) {
    std::vector<cv::Point3f> obj;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            obj.emplace_back(static_cast<float>(j * sz), static_cast<float>(i * sz), 0.0f);
    return obj;
}

std::vector<std::vector<cv::Point2f>> projectSyntheticPoints(
    int num_frames,
    const std::vector<cv::Point3f>& obj,
    const cv::Mat& K, const cv::Mat& D,
    int img_w, int img_h)
{
    std::vector<std::vector<cv::Point2f>> all_pts;
    for (int i = 0; i < num_frames; ++i) {
        double ax = 0.3 * std::sin(i * 0.5);
        double ay = 0.3 * std::cos(i * 0.7);
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << ax, ay, 0.0);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 500.0);

        std::vector<cv::Point2f> pts;
        cv::projectPoints(obj, rvec, tvec, K, D, pts);

        bool all_inside = true;
        for (const auto& p : pts) {
            if (p.x < 0 || p.x >= img_w || p.y < 0 || p.y >= img_h) {
                all_inside = false;
                break;
            }
        }
        if (all_inside) {
            all_pts.push_back(std::move(pts));
        }
    }
    return all_pts;
}

IntrinsicCalibParams makeTestParams() {
    IntrinsicCalibParams p;
    p.chessboard_width = 5;
    p.chessboard_height = 4;
    p.square_size_mm = 20.0;
    p.image_width = 640;
    p.image_height = 480;
    p.use_calibrateCameraRO = false;
    p.calib_flags = 0;
    return p;
}

} // anonymous namespace

TEST(IntrinsicCalibParamsTest, DefaultValues) {
    IntrinsicCalibParams p;
    EXPECT_EQ(p.chessboard_width, 11);
    EXPECT_EQ(p.chessboard_height, 8);
    EXPECT_DOUBLE_EQ(p.square_size_mm, 15.0);
    EXPECT_EQ(p.image_width, 2048);
    EXPECT_EQ(p.image_height, 1536);
    EXPECT_TRUE(p.use_calibrateCameraRO);
}

TEST(IntrinsicCalibParamsTest, HelperMethods) {
    IntrinsicCalibParams p;
    EXPECT_EQ(p.imageSize(), cv::Size(2048, 1536));
    EXPECT_EQ(p.totalCorners(), 88);
}

TEST(IntrinsicCalibParamsTest, FromJsonFileNotFound) {
    IntrinsicCalibParams p = IntrinsicCalibParams::fromJson("nonexistent_file.json");
    EXPECT_EQ(p.chessboard_width, 11);
}

TEST(IntrinsicCalibParamsTest, FromJsonValidFile) {
    std::string path = "test_params_temp.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"chessboard_width": 7, "chessboard_height": 5, "square_size_mm": 25.0})";
    }

    IntrinsicCalibParams p = IntrinsicCalibParams::fromJson(path);
    EXPECT_EQ(p.chessboard_width, 7);
    EXPECT_EQ(p.chessboard_height, 5);
    EXPECT_DOUBLE_EQ(p.square_size_mm, 25.0);

    std::filesystem::remove(path);
}

TEST(MonocularCalibResultTest, DefaultInvalid) {
    MonocularCalibResult r;
    EXPECT_FALSE(r.isValid());
    EXPECT_TRUE(r.camera_matrix.empty());
}

TEST(MonocularCalibResultTest, ClearResetsState) {
    MonocularCalibResult r;
    r.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    r.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    r.rms_error = 0.5;
    EXPECT_TRUE(r.isValid());

    r.clear();
    EXPECT_FALSE(r.isValid());
    EXPECT_DOUBLE_EQ(r.rms_error, 0.0);
    EXPECT_TRUE(r.per_view_errors.empty());
}

TEST(IntrinsicCalibResultTest, DefaultInvalid) {
    IntrinsicCalibResult r;
    EXPECT_FALSE(r.isValid());
}

TEST(IntrinsicCalibCPUTest, EmptyPointsReturnsFalse) {
    IntrinsicCalibParams params = makeTestParams();
    IntrinsicCalibCPU calib(params);

    std::vector<std::vector<cv::Point2f>> empty;
    IntrinsicCalibResult result;
    EXPECT_FALSE(calib.Execute(empty, empty, result));
}

TEST(IntrinsicCalibCPUTest, MismatchedFrameCount) {
    IntrinsicCalibParams params = makeTestParams();
    IntrinsicCalibCPU calib(params);

    std::vector<std::vector<cv::Point2f>> left(3);
    std::vector<std::vector<cv::Point2f>> right(5);
    IntrinsicCalibResult result;
    EXPECT_FALSE(calib.Execute(left, right, result));
}

TEST(IntrinsicCalibCPUTest, SingleCameraEmptyPoints) {
    IntrinsicCalibParams params = makeTestParams();
    IntrinsicCalibCPU calib(params);

    std::vector<std::vector<cv::Point2f>> empty;
    MonocularCalibResult result;
    EXPECT_FALSE(calib.Execute(empty, result));
}

TEST(IntrinsicCalibCPUTest, ParamsAccessor) {
    IntrinsicCalibParams params = makeTestParams();
    IntrinsicCalibCPU calib(params);
    EXPECT_EQ(calib.GetParams().chessboard_width, 5);
    EXPECT_EQ(calib.GetParams().chessboard_height, 4);
}

TEST(IntrinsicCalibCPUTest, SingleCalibrationWithProjectedPoints) {
    IntrinsicCalibParams params = makeTestParams();

    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);

    auto obj = makeObjectPoints(5, 4, 20.0);
    auto pts = projectSyntheticPoints(15, obj, K, D, 640, 480);

    ASSERT_GE(pts.size(), 3u);

    IntrinsicCalibCPU calib(params);
    MonocularCalibResult result;
    ASSERT_TRUE(calib.Execute(pts, result));

    EXPECT_TRUE(result.isValid());
    EXPECT_GT(result.rms_error, 0.0);
    EXPECT_LT(result.rms_error, 1.0);
    EXPECT_EQ(result.valid_frame_count, static_cast<int>(pts.size()));
    EXPECT_EQ(static_cast<int>(result.per_view_errors.size()), result.valid_frame_count);

    double fx_diff = std::abs(result.camera_matrix.at<double>(0, 0) - 800.0);
    double fy_diff = std::abs(result.camera_matrix.at<double>(1, 1) - 800.0);
    EXPECT_LT(fx_diff, 50.0);
    EXPECT_LT(fy_diff, 50.0);
}

TEST(IntrinsicCalibCPUTest, StereoCalibrationWithProjectedPoints) {
    IntrinsicCalibParams params = makeTestParams();

    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);

    auto obj = makeObjectPoints(5, 4, 20.0);
    auto left_pts = projectSyntheticPoints(15, obj, K, D, 640, 480);
    auto right_pts = projectSyntheticPoints(15, obj, K, D, 640, 480);

    size_t n = std::min(left_pts.size(), right_pts.size());
    ASSERT_GE(n, 3u);
    left_pts.resize(n);
    right_pts.resize(n);

    IntrinsicCalibCPU calib(params);
    IntrinsicCalibResult result;
    ASSERT_TRUE(calib.Execute(left_pts, right_pts, result));

    EXPECT_TRUE(result.isValid());
    EXPECT_EQ(result.total_frames_input, static_cast<int>(n));
    EXPECT_GT(result.reproj_error_mean, 0.0);
    EXPECT_LT(result.reproj_error_mean, 1.0);
}

TEST(IntrinsicCalibCPUTest, WrongPointCountFramesSkipped) {
    IntrinsicCalibParams params = makeTestParams();
    int expected = params.totalCorners();

    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);
    auto obj = makeObjectPoints(5, 4, 20.0);
    auto good_pts = projectSyntheticPoints(5, obj, K, D, 640, 480);
    ASSERT_GE(good_pts.size(), 2u);

    std::vector<std::vector<cv::Point2f>> mixed;
    mixed.push_back(good_pts[0]);
    mixed.push_back({cv::Point2f(1, 2), cv::Point2f(3, 4)});
    mixed.push_back(good_pts[1]);

    IntrinsicCalibCPU calib(params);
    MonocularCalibResult result;
    ASSERT_TRUE(calib.Execute(mixed, result));

    EXPECT_EQ(result.valid_frame_count, 2);
}

TEST(SaveLoadTest, SaveAndLoadResult) {
    IntrinsicCalibResult original;
    original.left.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    original.left.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    original.left.rms_error = 0.35;
    original.left.valid_frame_count = 10;
    original.left.per_view_errors = {0.1, 0.2, 0.3, 0.4, 0.5};

    original.right.camera_matrix = (cv::Mat_<double>(3, 3) << 1050, 0, 330, 0, 1050, 250, 0, 0, 1);
    original.right.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    original.right.rms_error = 0.42;
    original.right.valid_frame_count = 10;

    original.reproj_error_mean = 0.38;
    original.reproj_error_std = 0.02;
    original.valid_frames_count = 10;
    original.total_frames_input = 12;

    std::string filepath = "test_save_load_result.yaml";
    EXPECT_TRUE(IntrinsicCalibCPU::SaveResult(filepath, original));

    IntrinsicCalibResult loaded;
    EXPECT_TRUE(IntrinsicCalibCPU::LoadResult(filepath, loaded));

    EXPECT_TRUE(loaded.isValid());
    EXPECT_DOUBLE_EQ(loaded.left.rms_error, 0.35);
    EXPECT_DOUBLE_EQ(loaded.right.rms_error, 0.42);
    EXPECT_DOUBLE_EQ(loaded.reproj_error_mean, 0.38);
    EXPECT_EQ(loaded.left.per_view_errors.size(), 5u);
    EXPECT_NEAR(loaded.left.per_view_errors[2], 0.3, 1e-6);
    EXPECT_EQ(loaded.left.camera_matrix.at<double>(0, 0), 1000);
    EXPECT_EQ(loaded.right.camera_matrix.at<double>(0, 0), 1050);

    std::filesystem::remove(filepath);
}

TEST(SaveLoadTest, LoadNonexistentFile) {
    IntrinsicCalibResult result;
    EXPECT_FALSE(IntrinsicCalibCPU::LoadResult("nonexistent.yaml", result));
}

TEST(SaveLoadTest, SaveLoadMonoResult) {
    MonocularCalibResult original;
    original.camera_matrix = (cv::Mat_<double>(3, 3) << 800, 0, 400, 0, 800, 300, 0, 0, 1);
    original.dist_coeffs = (cv::Mat_<double>(5, 1) << -0.1, 0.05, 0.001, 0.002, 0.0);
    original.rms_error = 0.25;
    original.valid_frame_count = 8;

    std::string filepath = "test_mono_save.yaml";
    EXPECT_TRUE(IntrinsicCalibCPU::SaveMonoResult(filepath, "cam0", original));

    MonocularCalibResult loaded;
    EXPECT_TRUE(IntrinsicCalibCPU::LoadMonoResult(filepath, "cam0", loaded));

    EXPECT_TRUE(loaded.isValid());
    EXPECT_DOUBLE_EQ(loaded.rms_error, 0.25);
    EXPECT_EQ(loaded.camera_matrix.at<double>(0, 0), 800);
    EXPECT_NEAR(loaded.dist_coeffs.at<double>(0), -0.1, 1e-10);

    std::filesystem::remove(filepath);
}

TEST(MoveSemanticsTest, MoveConstructor) {
    IntrinsicCalibParams params = makeTestParams();
    IntrinsicCalibCPU calib1(params);
    EXPECT_EQ(calib1.GetParams().chessboard_width, 5);

    IntrinsicCalibCPU calib2(std::move(calib1));
    EXPECT_EQ(calib2.GetParams().chessboard_width, 5);
}

// ============= JSON Serialization Tests =============

TEST(MonocularCalibResultJsonTest, RoundTrip) {
    MonocularCalibResult original;
    original.camera_matrix = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    original.dist_coeffs = (cv::Mat_<double>(5, 1) << -0.1, 0.05, 0.001, 0.002, 0.0);
    original.rms_error = 0.25;
    original.valid_frame_count = 8;
    original.per_view_errors = {0.1, 0.2, 0.3};

    auto j = original.toJson();
    auto loaded = MonocularCalibResult::fromJson(j);

    EXPECT_TRUE(loaded.isValid());
    EXPECT_DOUBLE_EQ(loaded.rms_error, 0.25);
    EXPECT_EQ(loaded.valid_frame_count, 8);
    EXPECT_EQ(loaded.per_view_errors.size(), 3u);
    EXPECT_DOUBLE_EQ(loaded.camera_matrix.at<double>(0, 0), 800.0);
    EXPECT_NEAR(loaded.dist_coeffs.at<double>(0), -0.1, 1e-10);
}

TEST(IntrinsicCalibResultJsonTest, RoundTrip) {
    IntrinsicCalibResult original;
    original.left.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    original.left.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    original.left.rms_error = 0.35;
    original.left.valid_frame_count = 10;
    original.left.per_view_errors = {0.1, 0.2, 0.3, 0.4, 0.5};

    original.right.camera_matrix = (cv::Mat_<double>(3, 3) << 1050, 0, 330, 0, 1050, 250, 0, 0, 1);
    original.right.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    original.right.rms_error = 0.42;
    original.right.valid_frame_count = 10;

    original.reproj_error_mean = 0.38;
    original.reproj_error_std = 0.02;
    original.valid_frames_count = 10;
    original.total_frames_input = 12;

    auto j = original.toJson();
    auto loaded = IntrinsicCalibResult::fromJson(j);

    EXPECT_TRUE(loaded.isValid());
    EXPECT_DOUBLE_EQ(loaded.left.rms_error, 0.35);
    EXPECT_DOUBLE_EQ(loaded.right.rms_error, 0.42);
    EXPECT_DOUBLE_EQ(loaded.reproj_error_mean, 0.38);
    EXPECT_DOUBLE_EQ(loaded.reproj_error_std, 0.02);
    EXPECT_EQ(loaded.valid_frames_count, 10);
    EXPECT_EQ(loaded.total_frames_input, 12);
    EXPECT_EQ(loaded.left.per_view_errors.size(), 5u);
    EXPECT_DOUBLE_EQ(loaded.left.camera_matrix.at<double>(0, 0), 1000.0);
    EXPECT_DOUBLE_EQ(loaded.right.camera_matrix.at<double>(0, 0), 1050.0);
}

TEST(SaveLoadJsonTest, SaveAndLoadResultJson) {
    IntrinsicCalibResult original;
    original.left.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    original.left.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    original.left.rms_error = 0.35;
    original.left.valid_frame_count = 10;
    original.left.per_view_errors = {0.1, 0.2, 0.3};

    original.right.camera_matrix = (cv::Mat_<double>(3, 3) << 1050, 0, 330, 0, 1050, 250, 0, 0, 1);
    original.right.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    original.right.rms_error = 0.42;
    original.right.valid_frame_count = 10;

    original.reproj_error_mean = 0.38;
    original.reproj_error_std = 0.02;
    original.valid_frames_count = 10;
    original.total_frames_input = 12;

    std::string filepath = "test_save_load_result.json";
    EXPECT_TRUE(IntrinsicCalibCPU::SaveResultJson(filepath, original));

    IntrinsicCalibResult loaded;
    EXPECT_TRUE(IntrinsicCalibCPU::LoadResultJson(filepath, loaded));

    EXPECT_TRUE(loaded.isValid());
    EXPECT_DOUBLE_EQ(loaded.left.rms_error, 0.35);
    EXPECT_DOUBLE_EQ(loaded.right.rms_error, 0.42);
    EXPECT_DOUBLE_EQ(loaded.reproj_error_mean, 0.38);
    EXPECT_DOUBLE_EQ(loaded.left.camera_matrix.at<double>(0, 0), 1000.0);
    EXPECT_DOUBLE_EQ(loaded.right.camera_matrix.at<double>(0, 0), 1050.0);

    std::filesystem::remove(filepath);
}

TEST(SaveLoadJsonTest, LoadNonexistentJsonFile) {
    IntrinsicCalibResult result;
    EXPECT_FALSE(IntrinsicCalibCPU::LoadResultJson("nonexistent.json", result));
}
