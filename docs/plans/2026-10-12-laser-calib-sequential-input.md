# laser_calib left/right 顺序帧输入支持 — 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `loadLaserInput` 在无 `pose_*` 子目录时自动回退扫描 `<dir>/left/` + `<dir>/right/` 同名帧配对（每对 = 1 pose × 1 tube），使 `test_data/laser261006-1668/` 可直接输入。

**Architecture:** 只改 fc2_io 装载器（calib_io.cpp）单函数分支；pose_* 路径逐字不动；tools/算子/config 键零触碰。测试新增 `tests/test_load_laser_input.cpp`（CMake glob 自动注册）。

**Tech Stack:** C++17 / std::filesystem / OpenCV imgcodecs (imread/imwrite) / GoogleTest / CMake VS2022 Release（build_fc2_rel）。

**设计文档:** `factory_calib/docs/plans/2026-10-12-laser-calib-sequential-input-design.md`

**红线:** factory_calib 为保护区——不自动 commit（需人工指令）；不触碰算子/主库参数；输出绝不写入数据集目录。

---

### Task 1: 写失败测试 tests/test_load_laser_input.cpp

**Files:**
- Create: `factory_calib/module2_laser/tests/test_load_laser_input.cpp`

**Step 1: 写测试文件**（完整代码）

```cpp
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
```

**Step 2: 构建（新目标经 glob 自动收录，需重新跑 cmake configure）**

```powershell
cmake -S factory_calib -B build_fc2_rel -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Release `
      -DFC_BUILD_MODULE1=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc2_rel --config Release --target test_load_laser_input
```

注意：configure 会重新 glob；务必确认其余参数与既有 cache 一致（幂等）。

**Step 3: 跑测试确认红**

```powershell
.\build_fc2_rel\module2_laser\Release\test_load_laser_input.exe
```

预期：**SequentialPairingFallback / MissingRightPairSkipped FAIL**（现状无回退，nullopt）；PoseLayoutTakesPriority / NoLayoutFails PASS（现有行为）。= 2 红 2 绿。

---

### Task 2: 实现 calib_io.cpp 回退分支

**Files:**
- Modify: `factory_calib/module2_laser/calib_io.cpp:510-513`（`poseDirs.empty()` 错误分支）

**Step 1: 替换该分支为**（pose 循环体与最终空检查逐字不动）

```cpp
    if (poseDirs.empty()) {
        // 2026-10-12: pose_* 缺席时回退 left/+right/ 同名顺序帧布局（原始采集形态,
        // 如 test_data/laser261006-1668）。每对同名帧 = 1 pose × 1 tube 帧,
        // 对齐 DUIBI1005-1 人工转换惯例; pose_* 与 left/ 并存时 pose_* 优先。
        const fs::path ldir = fs::path(dir) / "left";
        const fs::path rdir = fs::path(dir) / "right";
        if (!fs::is_directory(ldir) || !fs::is_directory(rdir)) {
            spdlog::error("no pose_* subdirectories or left/+right/ layout in {}", dir);
            return std::nullopt;
        }

        std::vector<fs::path> lFiles;
        for (auto& e : fs::directory_iterator(ldir)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext == ".png" || ext == ".jpg") lFiles.push_back(e.path());
        }
        std::sort(lFiles.begin(), lFiles.end());

        for (auto& lf : lFiles) {
            const fs::path rf = rdir / lf.filename();
            if (!fs::exists(rf)) {
                spdlog::warn("skip {}: no right pair {}",
                             lf.filename().string(), rf.string());
                continue;
            }
            cv::Mat l = cv::imread(lf.string(), cv::IMREAD_GRAYSCALE);
            cv::Mat r = cv::imread(rf.string(), cv::IMREAD_GRAYSCALE);
            if (l.empty() || r.empty()) {
                spdlog::warn("skip {}: read failed", lf.filename().string());
                continue;
            }
            in.poseDirs.push_back("left/" + lf.filename().string());
            in.poseFrames.push_back({{l, r}});
        }
    }
```

（其后既有 `for (auto& pd : poseDirs)` 循环与 `in.poseFrames.empty()` 终检保持原样——left/right 存在但全跳过时由终检兜底报错。）

**Step 2: 增量构建 + 跑测试**

```powershell
cmake --build build_fc2_rel --config Release --target test_load_laser_input
.\build_fc2_rel\module2_laser\Release\test_load_laser_input.exe
```

预期：**4/4 PASS**。

---

### Task 3: 注释与用法文本同步

**Files:**
- Modify: `factory_calib/module2_laser/calib_io.h:180-186`（布局注释块）
- Modify: `factory_calib/module2_laser/laser_calib_cli.cpp:153`（usage 文本）

calib_io.h 布局注释追加一行：`//   备选布局（2026-10-12 自动回退, pose_* 缺席时生效）: <dir>/left/*.png + <dir>/right/*.png 同名配对, 每对 = 1 pose × 1 tube`

laser_calib_cli.cpp 用法文本追加：`或 left/ + right/ 同名顺序帧（无 pose_* 时自动回退）`。

---

### Task 4: 回归验证

**Step 1:** 重建受影响目标

```powershell
cmake --build build_fc2_rel --config Release --target laser_calib test_handoff test_laser_calib_e2e
```

**Step 2:** 跑新测试 + handoff 回归 + e2e 冒烟

```powershell
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
.\build_fc2_rel\module2_laser\Release\test_load_laser_input.exe
.\build_fc2_rel\module2_laser\Release\test_handoff.exe
.\build_fc2_rel\module2_laser\Release\test_laser_calib_e2e.exe
```

预期：全部 PASS（e2e 冒烟走 pose 合成布局，不受回退影响）。

---

### Task 5: 实跑 laser261006-1668（人工确认输出位置后执行）

```powershell
# 输出目录 = 独立新目录（防 sidecar 覆盖数据集内 729MB 参考 curve_map_temp_table.bin）
New-Item -ItemType Directory -Force factory_calib\data_out\laser261006-1668_rerun
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
.\build_fc2_rel\module2_laser\Release\laser_calib.exe `
    factory_calib\module2_laser\test_data\laser261006-1668 `
    factory_calib\data_out\laser261006-1668_rerun\laser_calib.json
```

预期日志：`loaded 29 poses (29 tubes total)`；对拍历史基线口径：posesProcessed=29 / framesOk=23 / framesSkipped=6 / verdict=PASS（CUDA CMTT 引擎全链约 3.5min，sidecar ~3.3GB 磁盘）。

---

### Task 6: 文档增补（factory_calib/AGENTS.md）

按本目录惯例在 AGENTS.md 头部增补段加一行 2026-10-12 记录：loadLaserInput 自动回退 left/right 顺序帧布局（laser261006-1668 直输；pose_* 优先零影响；新增 tests/test_load_laser_input.cpp 4 cases）。**不 commit**（保护区，等人工指令）。
