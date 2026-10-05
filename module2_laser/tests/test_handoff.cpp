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

// ============================================================================
// TEST 8: stereoRectifyTempTable 解析 → haveRectifyTempTable=true, 档数/排序/矩阵
//         (含一档畸形 → 弃档不弃表; 两档乱序 → 升序排序)
// ============================================================================
TEST(Handoff, ParsesRectifyTempTable) {
    json j = json::parse(makeValidHandoffJson());

    auto I3 = json::array({{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}});
    auto P  = json::array({{1000.0,0.0,64.0,0.0},
                           {0.0,1000.0,64.0,0.0},
                           {0.0,0.0,1.0,0.0}});
    auto Q4 = json::array({{1.0,0.0,0.0,-64.0},
                           {0.0,1.0,0.0,-64.0},
                           {0.0,0.0,0.0,1000.0},
                           {0.0,0.0,0.01,0.0}});
    auto roi = json{{"x", 0}, {"y", 0}, {"w", 128}, {"h", 128}};

    auto entry = [&](double t) {
        json e;
        e["temperature"] = t;
        e["deltaT"]      = t - 22.5;
        e["R1"] = I3; e["R2"] = I3; e["P1"] = P; e["P2"] = P; e["Q"] = Q4;
        e["validRoiLeft"]  = roi;
        e["validRoiRight"] = roi;
        return e;
    };

    json bad;  // 缺 R2/P1/P2/Q → 应被弃档
    bad["temperature"] = 26.0;
    bad["R1"] = I3;

    j["stereoRectifyTempTable"] = json{
        {"success", true}, {"message", ""}, {"qualityFlag", 0},
        {"referenceTemp", 22.5}, {"cte", 23.6e-6}, {"tableSize", 2},
        {"table", json::array({entry(25.5), bad, entry(25.0)})}};

    auto p = writeTmp("handoff_ttable.json", j.dump(2));
    auto h = loadCameraCalibHandoff(p.string());
    ASSERT_TRUE(h.has_value());

    EXPECT_TRUE(h->haveRectifyTempTable);
    ASSERT_EQ(h->rectifyTempTable.table.size(), 2u);  // 畸形档被弃
    EXPECT_TRUE(h->rectifyTempTable.success);
    EXPECT_EQ(h->rectifyTempTable.tableSize, 2);
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.referenceTemp, 22.5);

    // 升序排序: 首档 25.0（JSON 中 25.5 在前）
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.table[0].temperature, 25.0);
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.table[1].temperature, 25.5);

    // 首档矩阵形状与内容
    EXPECT_EQ(h->rectifyTempTable.table[0].R1.size(), cv::Size(3, 3));
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.table[0].R1.at<double>(0, 0), 1.0);
    EXPECT_EQ(h->rectifyTempTable.table[0].Q.size(), cv::Size(4, 4));
    EXPECT_EQ(h->rectifyTempTable.table[0].validRoiLeft, cv::Rect(0, 0, 128, 128));
}

// ============================================================================
// TEST 9: 无 stereoRectifyTempTable 节 → have=false, load 仍成功（容忍老输出）
// ============================================================================
TEST(Handoff, ToleratesMissingRectifyTempTable) {
    auto p = writeTmp("handoff_no_ttable.json", makeValidHandoffJson());
    auto h = loadCameraCalibHandoff(p.string());
    ASSERT_TRUE(h.has_value());
    EXPECT_FALSE(h->haveRectifyTempTable);
    EXPECT_TRUE(h->rectifyTempTable.table.empty());
}

// ============================================================================
// TEST 10: 0.2 步距网格浮点表示误差 → 参考温档就地校正为精确值
//          夹具: 一档 temperature=22.500000000000004 (0.2 二进制不可精确表示,
//          模块1 网格生成累计误差; C++/JSON 十进制解析同落 22.5+1ulp, 与 22.5
//          位级不等但 |Δ|=3.55e-15<1e-6), 一档 25.5; 表节点 referenceTemp=22.5
//          → 解析后首档位级精确等于 22.5 (EXPECT_DOUBLE_EQ), 25.5 档不受影响
// ============================================================================
TEST(Handoff, CorrectsFloatGridReferenceTier) {
    json j = json::parse(makeValidHandoffJson());

    auto I3 = json::array({{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}});
    auto P  = json::array({{1000.0,0.0,64.0,0.0},
                           {0.0,1000.0,64.0,0.0},
                           {0.0,0.0,1.0,0.0}});
    auto Q4 = json::array({{1.0,0.0,0.0,-64.0},
                           {0.0,1.0,0.0,-64.0},
                           {0.0,0.0,0.0,1000.0},
                           {0.0,0.0,0.01,0.0}});
    auto roi = json{{"x", 0}, {"y", 0}, {"w", 128}, {"h", 128}};

    auto entry = [&](double t) {
        json e;
        e["temperature"] = t;
        e["deltaT"]      = t - 22.5;
        e["R1"] = I3; e["R2"] = I3; e["P1"] = P; e["P2"] = P; e["Q"] = Q4;
        e["validRoiLeft"]  = roi;
        e["validRoiRight"] = roi;
        return e;
    };

    j["stereoRectifyTempTable"] = json{
        {"success", true}, {"qualityFlag", 0},
        {"referenceTemp", 22.5}, {"cte", 23.6e-6},
        {"table", json::array({entry(22.500000000000004), entry(25.5)})}};

    auto p = writeTmp("handoff_floatgrid.json", j.dump(2));
    auto h = loadCameraCalibHandoff(p.string());
    ASSERT_TRUE(h.has_value());
    ASSERT_TRUE(h->haveRectifyTempTable);
    ASSERT_EQ(h->rectifyTempTable.table.size(), 2u);

    // 参考温档被校正为精确 22.5 (EXPECT_DOUBLE_EQ 位级比较), 补偿参数原样保留
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.table[0].temperature, 22.5);
    EXPECT_EQ(h->rectifyTempTable.table[0].R1.size(), cv::Size(3, 3));
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.table[0].R1.at<double>(0, 0), 1.0);
    // 非参考档 25.5 距 22.5 为 3.0 >> 1e-6, 不在校正窗口, 保持不变
    EXPECT_DOUBLE_EQ(h->rectifyTempTable.table[1].temperature, 25.5);
}
