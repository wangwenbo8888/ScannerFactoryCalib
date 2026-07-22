// test_handoff.cpp — 模块1 → 模块2 交接 schema 测试 (Task 7.1)
//
// 直接测 fc::loadCameraCalibHandoff / fc::validateHandoffConsistency,
// 不调 laser_calib.exe (与 test_laser_calib_e2e 互补: 那个跑端到端, 这个测解析逻辑)

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include "calib_io.h"

#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace fc;

namespace {

// 与 test_laser_calib_e2e 同款最小合法 handoff JSON
std::string makeValidHandoffJson() {
    json j;
    j["schema"]        = "factory_calib.camera_calib.v1";
    j["imageSize"]     = {128, 128};
    j["referenceTemp"] = 22.5;
    j["cte"]           = 23.6e-6;
    j["tempRangeMin"]  = -10.0;
    j["tempRangeMax"]  = 10.0;
    j["tempStep"]      = 0.2;

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
    ext["success"]               = true;
    ext["message"]               = "";
    ext["qualityFlag"]           = 0;
    ext["R"]                     = I3;
    ext["T"]                     = json::array({json::array({100.0}),
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

// 辅助：写 JSON 到临时文件，返回路径
fs::path writeTmp(const std::string& name, const std::string& content) {
    fs::path p = fs::temp_directory_path() / name;
    std::ofstream(p) << content;
    return p;
}

} // namespace

// ============================================================================
// TEST 1: 合法 schema 加载成功 + 字段正确
// ============================================================================
TEST(Handoff, AcceptsValidCameraCalibSchema) {
    auto p = writeTmp("handoff_valid.json", makeValidHandoffJson());
    auto h = loadCameraCalibHandoff(p.string());
    ASSERT_TRUE(h.has_value());

    EXPECT_EQ(h->schema, "factory_calib.camera_calib.v1");
    EXPECT_EQ(h->imageSize,  cv::Size(128, 128));
    EXPECT_DOUBLE_EQ(h->referenceTemp, 22.5);
    EXPECT_DOUBLE_EQ(h->cte, 23.6e-6);
    EXPECT_DOUBLE_EQ(h->tempRangeMin, -10.0);
    EXPECT_DOUBLE_EQ(h->tempRangeMax, 10.0);
    EXPECT_DOUBLE_EQ(h->tempStep, 0.2);

    // 必需矩阵非空
    ASSERT_FALSE(h->cameraMatrixL.empty());
    ASSERT_FALSE(h->cameraMatrixR.empty());
    ASSERT_FALSE(h->R.empty());
    ASSERT_FALSE(h->T.empty());
    ASSERT_FALSE(h->R1.empty());
    ASSERT_FALSE(h->R2.empty());
    ASSERT_FALSE(h->P1.empty());
    ASSERT_FALSE(h->P2.empty());
    ASSERT_FALSE(h->Q.empty());

    // K_L 是 3×3 单位矩阵
    EXPECT_EQ(h->cameraMatrixL.size(), cv::Size(3, 3));
    EXPECT_DOUBLE_EQ(h->cameraMatrixL.at<double>(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(h->cameraMatrixL.at<double>(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(h->cameraMatrixL.at<double>(2, 2), 1.0);

    // T = [100, 0, 0]^T
    EXPECT_EQ(h->T.size(), cv::Size(1, 3));
    EXPECT_DOUBLE_EQ(h->T.at<double>(0, 0), 100.0);

    // Q 是 4×4
    EXPECT_EQ(h->Q.size(), cv::Size(4, 4));
}

// ============================================================================
// TEST 2: 缺 Q 字段 → 拒绝 (Q 是 4-8 不可缺的)
// ============================================================================
TEST(Handoff, RejectsMissingRequiredField) {
    json j = json::parse(makeValidHandoffJson());
    j["rectify"].erase("Q");  // 删 Q
    auto p = writeTmp("handoff_no_q.json", j.dump(2));

    auto h = loadCameraCalibHandoff(p.string());
    EXPECT_FALSE(h.has_value()) << "should reject handoff missing Q";
}

// ============================================================================
// TEST 3: 缺 camera_matrix_l → 但 intrinsic.left.camera_matrix 还在, 应该走 fallback 成功
//         这个测试确认 intrinsic 节点是首选路径
// ============================================================================
TEST(Handoff, PrefersIntrinsicNodeOverExtrinsic) {
    json j = json::parse(makeValidHandoffJson());
    // 故意改 intrinsic.left.camera_matrix 为 2*I (可识别)
    j["intrinsic"]["left"]["camera_matrix"] =
        json::array({{2.0,0.0,0.0},{0.0,2.0,0.0},{0.0,0.0,2.0}});
    auto p = writeTmp("handoff_intrinsic_pref.json", j.dump(2));

    auto h = loadCameraCalibHandoff(p.string());
    ASSERT_TRUE(h.has_value());
    // 应该读到 2*I 而不是 fallback 到 extrinsic.camera_matrix_l
    EXPECT_DOUBLE_EQ(h->cameraMatrixL.at<double>(0, 0), 2.0);
}

// ============================================================================
// TEST 4: tempStep 不一致 → validateHandoffConsistency 返回 false
// ============================================================================
TEST(Handoff, RejectsInconsistentTempRange) {
    LaserCalibConfig cfg;
    cfg.tempStep     = 0.5;     // 与 handoff(0.2) 不一致
    cfg.cte          = 23.6e-6;
    cfg.tempRangeMin = -10.0;
    cfg.tempRangeMax = 10.0;

    CameraCalibHandoff h;
    h.cte          = 23.6e-6;
    h.tempStep     = 0.2;
    h.tempRangeMin = -10.0;
    h.tempRangeMax = 10.0;

    std::string why;
    EXPECT_FALSE(validateHandoffConsistency(cfg, h, why));
    EXPECT_FALSE(why.empty());
    // 错误信息应提及 tempStep
    EXPECT_NE(why.find("tempStep"), std::string::npos);
}

// ============================================================================
// TEST 5: 一致的 cfg + handoff → 返回 true
// ============================================================================
TEST(Handoff, AcceptsConsistentCfgAndHandoff) {
    LaserCalibConfig cfg;
    cfg.cte          = 23.6e-6;
    cfg.tempStep     = 0.2;
    cfg.tempRangeMin = -10.0;
    cfg.tempRangeMax = 10.0;

    CameraCalibHandoff h;
    h.cte          = 23.6e-6;
    h.tempStep     = 0.2;
    h.tempRangeMin = -10.0;
    h.tempRangeMax = 10.0;

    std::string why;
    EXPECT_TRUE(validateHandoffConsistency(cfg, h, why));
}

// ============================================================================
// TEST 6: 文件不存在 → 返回 nullopt (不抛异常)
// ============================================================================
TEST(Handoff, MissingFileReturnsNullopt) {
    auto h = loadCameraCalibHandoff("definitely_does_not_exist_xyz.json");
    EXPECT_FALSE(h.has_value());
}

// ============================================================================
// TEST 7: JSON 解析错误 → 返回 nullopt (不抛异常)
// ============================================================================
TEST(Handoff, MalformedJsonReturnsNullopt) {
    auto p = writeTmp("handoff_malformed.json", "{not valid json");
    auto h = loadCameraCalibHandoff(p.string());
    EXPECT_FALSE(h.has_value());
}
