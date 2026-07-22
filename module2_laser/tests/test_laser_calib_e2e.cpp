// test_laser_calib_e2e.cpp — 模块2 端到端冒烟测试
//
// 主工程无完整 4-1~4-13 串联 fixture → 降级为冒烟测试（Task 6.3 Step 2）
// 策略：构造最小合法 handoff + config + 全黑小图, 调 laser_calib.exe,
//      校验: 不崩、输出 JSON 存在、schema 正确、build 字段 = 6.2-e
// 不验证精度（全黑图没有激光线，绝大多数帧会 skip，haveVirtualPose=false 退 1）

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int kW = 128;
constexpr int kH = 128;

// 构造最小合法 camera_calib.json（与模块1 buildCameraCalibJson 输出 schema 对齐）
// 内参用单位矩阵, 畸变全 0, R=I, T=[100,0,0], Q 标准三角化矩阵
std::string makeHandoffJson() {
    json j;
    j["schema"]        = "factory_calib.camera_calib.v1";
    j["imageSize"]     = {kW, kH};
    j["referenceTemp"] = 22.5;
    j["cte"]           = 23.6e-6;
    j["tempRangeMin"]  = -10.0;
    j["tempRangeMax"]  = 10.0;
    j["tempStep"]      = 0.2;

    auto I3 = json::array({{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}});
    auto D1x5 = json::array({0.0,0.0,0.0,0.0,0.0});

    // intrinsic.left/right（MonocularCalibResult::toJson 字段）
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
    intr["left"]                = mono();
    intr["right"]               = mono();
    intr["reproj_error_mean"]   = 0.0;
    intr["reproj_error_std"]    = 0.0;
    intr["valid_frames_count"]  = 4;
    intr["total_frames_input"]  = 4;
    j["intrinsic"] = intr;

    // extrinsic（ExtrinsicCalibCpuResult::toJson 字段）
    json ext;
    ext["success"]               = true;
    ext["message"]               = "";
    ext["qualityFlag"]           = 0;
    ext["R"]                     = I3;
    ext["T"]                     = json::array({json::array({100.0}),
                                                json::array({0.0}),
                                                json::array({0.0})});
    ext["E"]                     = I3;
    ext["F"]                     = I3;
    ext["stereoReprojError"]     = 0.0;
    ext["epipolarErrorMean"]     = 0.0;
    ext["epipolarErrorStd"]      = 0.0;
    ext["perViewErrors"]         = std::vector<double>{};
    ext["perViewEpipolarErrors"] = std::vector<double>{};
    j["extrinsic"] = ext;

    // rectify（StereoRectifyCpuResult::toJson 字段）
    json rec;
    rec["success"]      = true;
    rec["message"]      = "";
    rec["qualityFlag"]  = 0;
    rec["R1"]           = I3;
    rec["R2"]           = I3;
    rec["P1"]           = json::array({{1000.0,0.0,kW/2.0,0.0},
                                       {0.0,1000.0,kH/2.0,0.0},
                                       {0.0,0.0,1.0,0.0}});
    rec["P2"]           = rec["P1"];
    rec["Q"]            = json::array({{1.0,0.0,0.0,-kW/2.0},
                                       {0.0,1.0,0.0,-kH/2.0},
                                       {0.0,0.0,0.0,1000.0},
                                       {0.0,0.0,1.0/100.0,0.0}});
    rec["validRoiLeft"]  = {{"x",0},{"y",0},{"w",kW},{"h",kH}};
    rec["validRoiRight"] = {{"x",0},{"y",0},{"w",kW},{"h",kH}};
    j["rectify"] = rec;

    return j.dump(2);
}

std::string makeConfigJson() {
    json j;
    j["deviceId"] = 0;
    j["plane_map"] = {{"gridStep", 1.0f},
                      {"depthMin", 50.0f},
                      {"depthMax", 500.0f},
                      {"depthSamples", 10},
                      {"epipolarStep", 1.0f}};
    j["lineIds"] = std::vector<int>{0, 1};  // 显式提供, 避免依赖反推
    j["temperature"] = {{"cte", 23.6e-6},
                        {"referenceTemp", 22.5},
                        {"tempRangeMin", -10.0},
                        {"tempRangeMax", 10.0},
                        {"tempStep", 0.2}};
    j["rectify"] = {{"alpha", 0.0}, {"flags", 1}};
    return j.dump(2);
}

// review I5: lineIds 缺省 (config 不提供) 全黑图也无 4-11 lineCurves
// 期望: 进入 4-13 时 effectiveLineIds 仍空 → 跳过 4-13 (而非崩溃)
//       输出 JSON status=partial, 不抛异常
std::string makeConfigJsonNoLineIds() {
    json j;
    j["deviceId"] = 0;
    j["plane_map"] = {{"gridStep", 1.0f},
                      {"depthMin", 50.0f},
                      {"depthMax", 500.0f},
                      {"depthSamples", 10},
                      {"epipolarStep", 1.0f}};
    // 故意不提供 lineIds
    j["temperature"] = {{"cte", 23.6e-6},
                        {"referenceTemp", 22.5},
                        {"tempRangeMin", -10.0},
                        {"tempRangeMax", 10.0},
                        {"tempStep", 0.2}};
    j["rectify"] = {{"alpha", 0.0}, {"flags", 1}};
    return j.dump(2);
}

void writeFile(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p);
    ASSERT_TRUE(ofs.is_open()) << "cannot write " << p.string();
    ofs << content;
}

void writeBlackPng(const fs::path& p) {
    cv::Mat black(kH, kW, CV_8UC1, cv::Scalar(0));
    ASSERT_TRUE(cv::imwrite(p.string(), black)) << "cannot write " << p.string();
}

} // namespace

// ============================================================================
// TEST 1: 完整 smoke —— 调 laser_calib.exe, 校验不崩 + schema
// ============================================================================
TEST(LaserCalibE2E, SmokeDoesNotCrash) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr) << "LASER_CALIB_EXE env must be set (CMakeLists sets it)";
    ASSERT_TRUE(fs::exists(exe)) << "exe not found: " << exe;

    // 临时数据目录
    fs::path root = fs::temp_directory_path() / "laser_smoke_e2e";
    fs::remove_all(root);
    fs::create_directories(root);

    writeFile(root / "config.json",         makeConfigJson());
    writeFile(root / "camera_calib.json",   makeHandoffJson());

    fs::path pose = root / "pose_00";
    fs::create_directories(pose);
    writeBlackPng(pose / "L_tube0.png");
    writeBlackPng(pose / "R_tube0.png");

    fs::path outJson = root / "out.json";
    std::string cmd = std::string(exe) + " " + root.string() + " " + outJson.string();
    SCOPED_TRACE("cmd: " + cmd);

    int rc = std::system(cmd.c_str());

    // 冒烟: exit 0 或 1 都接受 (全黑图 → 多数 frame skip → haveVirtualPose=false → exit 1)
    // 关键是不崩 (rc 不应该是 -1 / 0xC0000005 等)
    EXPECT_GE(rc, 0);
    EXPECT_LE(rc, 1);

    // 输出 JSON 存在 + schema 正确
    EXPECT_TRUE(fs::exists(outJson)) << "output json missing";
    if (fs::exists(outJson)) {
        std::ifstream ifs(outJson);
        ASSERT_TRUE(ifs.is_open());
        json j;
        ifs >> j;
        EXPECT_EQ(j.value("schema", ""), "factory_calib.laser_calib.v1");
        EXPECT_EQ(j.value("build", ""),  "6.2-e");
        EXPECT_EQ(j.value("posesProcessed", -1), 1);
        // 全黑图无激光线 → 累积点应为 0, virtualPose=false
        EXPECT_FALSE(j.value("haveVirtualPose", true));
    }
}

// ============================================================================
// TEST 2: 缺 handoff 文件 → 应当 graceful exit 1, 不崩
// ============================================================================
TEST(LaserCalibE2E, MissingHandoffGracefulExit) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr);

    fs::path root = fs::temp_directory_path() / "laser_smoke_nohandoff";
    fs::remove_all(root);
    fs::create_directories(root);
    writeFile(root / "config.json", makeConfigJson());
    // 故意不写 camera_calib.json

    fs::path outJson = root / "out.json";
    std::string cmd = std::string(exe) + " " + root.string() + " " + outJson.string();
    int rc = std::system(cmd.c_str());

    EXPECT_EQ(rc, 1);              // loadLaserInput 失败 → return 1
    EXPECT_FALSE(fs::exists(outJson)); // 不应该写输出
}

// ============================================================================
// TEST 3: review C1/I5 防回归 — lineIds 缺省 + 全黑图, 不应崩溃
//   之前 bug: lineIds 空 → PlaneMapTempTable 构造抛 invalid_argument → terminate
//   修复后期望: effectiveLineIds 空 → 跳过 4-13 → status=partial, exit=1, 不崩
// ============================================================================
TEST(LaserCalibE2E, LineIdsEmptyDoesNotCrash) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr) << "LASER_CALIB_EXE env must be set";
    ASSERT_TRUE(fs::exists(exe)) << "exe not found: " << exe;

    fs::path root = fs::temp_directory_path() / "laser_smoke_no_lineids";
    fs::remove_all(root);
    fs::create_directories(root);

    writeFile(root / "config.json",       makeConfigJsonNoLineIds());  // 不提供 lineIds
    writeFile(root / "camera_calib.json", makeHandoffJson());

    fs::path pose = root / "pose_00";
    fs::create_directories(pose);
    writeBlackPng(pose / "L_tube0.png");
    writeBlackPng(pose / "R_tube0.png");

    fs::path outJson = root / "out.json";
    std::string cmd = std::string(exe) + " " + root.string() + " " + outJson.string();
    SCOPED_TRACE("cmd: " + cmd);

    int rc = std::system(cmd.c_str());

    // 关键断言: 不崩溃 (rc 0 或 1, 而非 0xC0000005 等)
    EXPECT_GE(rc, 0);
    EXPECT_LE(rc, 1);

    // 输出文件存在 + status=partial (全黑图无 4-11 lineCurves → 跳过 4-13)
    EXPECT_TRUE(fs::exists(outJson));
    if (fs::exists(outJson)) {
        std::ifstream ifs(outJson);
        ASSERT_TRUE(ifs.is_open());
        json j;
        ifs >> j;
        EXPECT_EQ(j.value("schema", ""), "factory_calib.laser_calib.v1");
        EXPECT_EQ(j.value("status", ""), "partial");  // review I1: status 字段
        EXPECT_FALSE(j.value("havePlaneTable", true));
    }
}

// ============================================================================
// TEST 4: review I3 防回归 — imageSize 缺失, loadCameraCalibHandoff 应拒绝
// ============================================================================
TEST(LaserCalibE2E, RejectsHandoffWithoutImageSize) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr);

    fs::path root = fs::temp_directory_path() / "laser_smoke_no_imagesize";
    fs::remove_all(root);
    fs::create_directories(root);
    writeFile(root / "config.json", makeConfigJson());

    // handoff 故意删 imageSize
    json hj = json::parse(makeHandoffJson());
    hj.erase("imageSize");
    writeFile(root / "camera_calib.json", hj.dump(2));

    fs::path pose = root / "pose_00";
    fs::create_directories(pose);
    writeBlackPng(pose / "L_tube0.png");
    writeBlackPng(pose / "R_tube0.png");

    fs::path outJson = root / "out.json";
    std::string cmd = std::string(exe) + " " + root.string() + " " + outJson.string();
    int rc = std::system(cmd.c_str());

    EXPECT_EQ(rc, 1);  // loadCameraCalibHandoff 返回 nullopt
    EXPECT_FALSE(fs::exists(outJson));
}
