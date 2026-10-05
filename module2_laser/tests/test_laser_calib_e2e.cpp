// test_laser_calib_e2e.cpp 鈥?妯″潡2 绔埌绔啋鐑熸祴璇?
//
// 涓诲伐绋嬫棤瀹屾暣 4-1~4-14 涓茶仈 fixture 鈫?闄嶇骇涓哄啋鐑熸祴璇曪紙Task 6.3 Step 2锛?
// 绛栫暐锛氭瀯閫犳渶灏忓悎娉?handoff + config + 鍏ㄩ粦灏忓浘, 璋?laser_calib.exe,
//      鏍￠獙: 涓嶅穿銆佽緭鍑?JSON 瀛樺湪銆乻chema 姝ｇ‘銆乥uild 瀛楁 = 6.3-cmtt
// 涓嶉獙璇佺簿搴︼紙鍏ㄩ粦鍥炬病鏈夋縺鍏夌嚎锛岀粷澶у鏁板抚浼?skip锛宧aveVirtualPose=false 閫€ 1锛?

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

// 鏋勯€犳渶灏忓悎娉?camera_calib.json锛堜笌妯″潡1 buildCameraCalibJson 杈撳嚭 schema 瀵归綈锛?
// 鍐呭弬鐢ㄥ崟浣嶇煩闃? 鐣稿彉鍏?0, R=I, T=[100,0,0], Q 鏍囧噯涓夎鍖栫煩闃?
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

    // intrinsic.left/right锛圡onocularCalibResult::toJson 瀛楁锛?
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

    // extrinsic锛圗xtrinsicCalibCpuResult::toJson 瀛楁锛?
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

    // rectify锛圫tereoRectifyCpuResult::toJson 瀛楁锛?
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
    j["lineIds"] = std::vector<int>{0, 1};  // 鏄惧紡鎻愪緵, 閬垮厤渚濊禆鍙嶆帹
    j["temperature"] = {{"cte", 23.6e-6},
                        {"referenceTemp", 22.5},
                        {"tempRangeMin", -10.0},
                        {"tempRangeMax", 10.0},
                        {"tempStep", 0.2}};
    j["rectify"] = {{"alpha", 0.0}, {"flags", 1}};
    return j.dump(2);
}

// review I5: lineIds 缂虹渷 (config 涓嶆彁渚? 鍏ㄩ粦鍥句篃鏃?4-11 lineCurves)
// 鏈熸湜: 杩涘叆 4-14 鏃?effectiveLineIds 浠嶇┖ 鈫?璺宠繃 4-14 (鑰岄潪宕╂簝)
//       杈撳嚭 JSON status=partial, 涓嶆姏寮傚父
std::string makeConfigJsonNoLineIds() {
    json j;
    j["deviceId"] = 0;
    j["plane_map"] = {{"gridStep", 1.0f},
                      {"depthMin", 50.0f},
                      {"depthMax", 500.0f},
                      {"depthSamples", 10},
                      {"epipolarStep", 1.0f}};
    // 鏁呮剰涓嶆彁渚?lineIds
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
// TEST 1: 瀹屾暣 smoke 鈥斺€?璋?laser_calib.exe, 鏍￠獙涓嶅穿 + schema
// ============================================================================
TEST(LaserCalibE2E, SmokeDoesNotCrash) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr) << "LASER_CALIB_EXE env must be set (CMakeLists sets it)";
    ASSERT_TRUE(fs::exists(exe)) << "exe not found: " << exe;

    // 涓存椂鏁版嵁鐩綍
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

    // 鍐掔儫: exit 0 鎴?1 閮芥帴鍙?(鍏ㄩ粦鍥?鈫?澶氭暟 frame skip 鈫?haveVirtualPose=false 鈫?exit 1)
    // 鍏抽敭鏄笉宕?(rc 涓嶅簲璇ユ槸 -1 / 0xC0000005 绛?
    EXPECT_GE(rc, 0);
    EXPECT_LE(rc, 1);

    // 杈撳嚭 JSON 瀛樺湪 + schema 姝ｇ‘
    EXPECT_TRUE(fs::exists(outJson)) << "output json missing";
    if (fs::exists(outJson)) {
        std::ifstream ifs(outJson);
        ASSERT_TRUE(ifs.is_open());
        json j;
        ifs >> j;
        EXPECT_EQ(j.value("schema", ""), "factory_calib.laser_calib.v2");
        EXPECT_EQ(j.value("build", ""), "6.3-cmtt");
        EXPECT_EQ(j.value("posesProcessed", -1), 1);
        // 鍏ㄩ粦鍥炬棤婵€鍏夌嚎 鈫?绱Н鐐瑰簲涓?0, virtualPose=false
        EXPECT_FALSE(j.value("haveVirtualPose", true));
    }
}

// ============================================================================
// TEST 2: 缂?handoff 鏂囦欢 鈫?搴斿綋 graceful exit 1, 涓嶅穿
// ============================================================================
TEST(LaserCalibE2E, MissingHandoffGracefulExit) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr);

    fs::path root = fs::temp_directory_path() / "laser_smoke_nohandoff";
    fs::remove_all(root);
    fs::create_directories(root);
    writeFile(root / "config.json", makeConfigJson());
    // 鏁呮剰涓嶅啓 camera_calib.json

    fs::path outJson = root / "out.json";
    std::string cmd = std::string(exe) + " " + root.string() + " " + outJson.string();
    int rc = std::system(cmd.c_str());

    EXPECT_EQ(rc, 1);              // loadLaserInput 澶辫触 鈫?return 1
    EXPECT_FALSE(fs::exists(outJson)); // 涓嶅簲璇ュ啓杈撳嚭
}

// ============================================================================
// TEST 3: review C1/I5 闃插洖褰?鈥?lineIds 缂虹渷 + 鍏ㄩ粦鍥? 涓嶅簲宕╂簝
//   涔嬪墠 bug: lineIds 绌?鈫?PlaneMapTempTable 鏋勯€犳姏 invalid_argument 鈫?terminate
//   淇鍚庢湡鏈? effectiveLineIds 绌?鈫?璺宠繃 4-14 鈫?status=partial, exit=1, 涓嶅穿
// ============================================================================
TEST(LaserCalibE2E, LineIdsEmptyDoesNotCrash) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr) << "LASER_CALIB_EXE env must be set";
    ASSERT_TRUE(fs::exists(exe)) << "exe not found: " << exe;

    fs::path root = fs::temp_directory_path() / "laser_smoke_no_lineids";
    fs::remove_all(root);
    fs::create_directories(root);

    writeFile(root / "config.json",       makeConfigJsonNoLineIds());  // 涓嶆彁渚?lineIds
    writeFile(root / "camera_calib.json", makeHandoffJson());

    fs::path pose = root / "pose_00";
    fs::create_directories(pose);
    writeBlackPng(pose / "L_tube0.png");
    writeBlackPng(pose / "R_tube0.png");

    fs::path outJson = root / "out.json";
    std::string cmd = std::string(exe) + " " + root.string() + " " + outJson.string();
    SCOPED_TRACE("cmd: " + cmd);

    int rc = std::system(cmd.c_str());

    // 鍏抽敭鏂█: 涓嶅穿婧?(rc 0 鎴?1, 鑰岄潪 0xC0000005 绛?
    EXPECT_GE(rc, 0);
    EXPECT_LE(rc, 1);

    // 杈撳嚭鏂囦欢瀛樺湪 + status=partial (鍏ㄩ粦鍥炬棤 4-11 lineCurves 鈫?璺宠繃 4-14)
    EXPECT_TRUE(fs::exists(outJson));
    if (fs::exists(outJson)) {
        std::ifstream ifs(outJson);
        ASSERT_TRUE(ifs.is_open());
        json j;
        ifs >> j;
        EXPECT_EQ(j.value("schema", ""), "factory_calib.laser_calib.v2");
        EXPECT_EQ(j.value("status", ""), "partial");  // review I1: status 瀛楁
        EXPECT_FALSE(j.value("haveCurveMapTable", true));
    }
}

// ============================================================================
// TEST 4: review I3 闃插洖褰?鈥?imageSize 缂哄け, loadCameraCalibHandoff 搴旀嫆缁?
// ============================================================================
TEST(LaserCalibE2E, RejectsHandoffWithoutImageSize) {
    const char* exe = std::getenv("LASER_CALIB_EXE");
    ASSERT_NE(exe, nullptr);

    fs::path root = fs::temp_directory_path() / "laser_smoke_no_imagesize";
    fs::remove_all(root);
    fs::create_directories(root);
    writeFile(root / "config.json", makeConfigJson());

    // handoff 鏁呮剰鍒?imageSize
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

    EXPECT_EQ(rc, 1);  // loadCameraCalibHandoff 杩斿洖 nullopt
    EXPECT_FALSE(fs::exists(outJson));
}
