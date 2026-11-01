// test_load_laser_input.cpp — loadLaserInput 布局扫描测试（2026-10-12）
//
// 覆盖: left/+right/ 同名顺序帧自动回退 / 缺右对跳过 / pose_* 优先 / 无布局失败。
// 与 test_handoff 同款合成 handoff, 但温度字段对齐 LaserCalibConfig 编译默认
// (cte 23.6e-6 / refTemp 25.0 / ±15 / 0.5) —— 否则 validateHandoffConsistency 拒载。

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "calib_io.h"

#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace fc;

namespace {

std::string makeHandoffJsonMatchingDefaults() {
    json j;
    j["schema"]        = "factory_calib.camera_calib.v1";
    j["imageSize"]     = {128, 128};
    j["referenceTemp"] = 25.0;      // = LaserCalibConfig 默认
    j["cte"]           = 23.6e-6;
    j["tempRangeMin"]  = -15.0;
    j["tempRangeMax"]  = 15.0;
    j["tempStep"]      = 0.5;

    auto I3 = json::array({{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}});
    auto D1x5 = json::array({0.0,0.0,0.0,0.0,0.0});
    auto mono = [&]() {
        json m;
        m["camera_matrix"]     = I3;
        m["dist_coeffs"]       = json::array({D1x5});
        m["rms_error"]         = 0.0;
        m["valid_frame_count"] = 4;
        m["per_view_errors"]   = std::vector<double>{};
        m["rvecs"]             = std::vector<std::vector<double>>{};
        m["tvecs"]             = std::vector<std::vector<double>>{};
        return m;
    };
    json intr;
    intr["left"]  = mono();
    intr["right"] = mono();
    intr["reproj_error_mean"]   = 0.0;
    intr["reproj_error_std"]    = 0.0;
    intr["valid_frames_count"]  = 4;
    intr["total_frames_input"]  = 4;
    j["intrinsic"] = intr;

    json ext;
    ext["success"]     = true;
    ext["message"]     = "";
    ext["qualityFlag"] = 0;
    ext["R"]           = I3;
    ext["T"]           = json::array({json::array({100.0}),
                                      json::array({0.0}),
                                      json::array({0.0})});
    j["extrinsic"] = ext;

    json rec;
    rec["R1"] = I3;
    rec["R2"] = I3;
    rec["P1"] = json::array({{1000.0,0.0,64.0,0.0},
                             {0.0,1000.0,64.0,0.0},
                             {0.0,0.0,1.0,0.0}});
    rec["P2"] = rec["P1"];
    rec["Q"]  = json::array({{1.0,0.0,0.0,-64.0},
                             {0.0,1.0,0.0,-64.0},
                             {0.0,0.0,0.0,1000.0},
                             {0.0,0.0,0.01,0.0}});
    j["rectify"] = rec;

    return j.dump(2);
}

fs::path makeDataset(const std::string& tag) {
    fs::path root = fs::temp_directory_path() / ("fc_load_input_" + tag);
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    std::ofstream(root / "camera_calib.json")
        << makeHandoffJsonMatchingDefaults();
    return root;
}

void writeGray(const fs::path& p) {
    cv::Mat img(8, 8, CV_8UC1, cv::Scalar(30));
    ASSERT_TRUE(cv::imwrite(p.string(), img));
}

} // namespace

// 1) 顺序帧回退: 3 对 → 3 poses × 1 tube, poseDirs = "left/xx.png"
TEST(LoadLaserInput, SequentialPairingFallback) {
    auto root = makeDataset("seq");
    fs::create_directories(root / "left");
    fs::create_directories(root / "right");
    writeGray(root / "left" / "001.png");
    writeGray(root / "left" / "002.png");
    writeGray(root / "left" / "003.png");
    writeGray(root / "right" / "001.png");
    writeGray(root / "right" / "002.png");
    writeGray(root / "right" / "003.png");

    auto in = loadLaserInput(root.string());
    ASSERT_TRUE(in.has_value());
    ASSERT_EQ(in->poseDirs.size(), 3u);
    ASSERT_EQ(in->poseFrames.size(), 3u);
    EXPECT_EQ(in->poseDirs[0], "left/001.png");
    EXPECT_EQ(in->poseDirs[2], "left/003.png");
    for (const auto& tubes : in->poseFrames) {
        ASSERT_EQ(tubes.size(), 1u);
        EXPECT_FALSE(tubes[0].leftGray.empty());
        EXPECT_FALSE(tubes[0].rightGray.empty());
        EXPECT_EQ(tubes[0].leftGray.type(), CV_8UC1);
    }
}

// 2) 缺右对: warn + skip → 剩 2 poses
TEST(LoadLaserInput, MissingRightPairSkipped) {
    auto root = makeDataset("miss");
    fs::create_directories(root / "left");
    fs::create_directories(root / "right");
    writeGray(root / "left" / "001.png");
    writeGray(root / "left" / "002.png");
    writeGray(root / "left" / "003.png");
    writeGray(root / "right" / "001.png");
    writeGray(root / "right" / "003.png");

    auto in = loadLaserInput(root.string());
    ASSERT_TRUE(in.has_value());
    EXPECT_EQ(in->poseDirs.size(), 2u);
}

// 3) pose_* 与 left/ 并存 → pose_* 优先（回归保护）
TEST(LoadLaserInput, PoseLayoutTakesPriority) {
    auto root = makeDataset("prio");
    fs::create_directories(root / "pose_00");
    fs::create_directories(root / "left");
    fs::create_directories(root / "right");
    writeGray(root / "pose_00" / "L_tube0.png");
    writeGray(root / "pose_00" / "R_tube0.png");
    writeGray(root / "left" / "001.png");
    writeGray(root / "right" / "001.png");

    auto in = loadLaserInput(root.string());
    ASSERT_TRUE(in.has_value());
    ASSERT_EQ(in->poseDirs.size(), 1u);
    EXPECT_EQ(in->poseDirs[0], "pose_00");
    EXPECT_EQ(in->poseFrames[0].size(), 1u);
}

// 4) 两种布局皆无 → nullopt
TEST(LoadLaserInput, NoLayoutFails) {
    auto root = makeDataset("none");
    auto in = loadLaserInput(root.string());
    EXPECT_FALSE(in.has_value());
}
