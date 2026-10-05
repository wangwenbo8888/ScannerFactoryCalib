# tests/e2e — L3 合成端到端测试自包含单元

> 本文件夹是 `module1_camera` 的合成图端到端测试（L3）自包含单元：测试源码＋参数＋说明同驻一处。
> 上级访问规则见 `factory_calib/AGENTS.md`（AI 只读保护区，人工明确指令除外）。

## 1. 文件清单

| 文件 | 职责 |
|---|---|
| `test_camera_calib_e2e.cpp` | L3 端到端测试：合成已知内参的棋盘图组 → `extractChessboardCorners` → `normalizeLRCornerOrder` → `IntrinsicCalibCPU`，按 `gates` 门限断言收敛与内参可恢复 |
| `config.json` | **测试参数唯一源**（见 §2 字段表）；随 git 入库，始终存在 |
| `AGENTS.md` | 本说明文档 |

## 2. config.json 字段说明

| 键 | 类型 | 含义 | 默认值 |
|---|---|---|---|
| `image.width` / `image.height` | int | 合成图像尺寸（px） | 1280 / 960 |
| `chessboard.squaresX` / `squaresY` | int | 方格数（内角点数＝方格数−1，代码内推导） | 9 / 7 |
| `chessboard.squarePx` | int | 单方格像素边长；同时充当 `square_size_mm`（合成图无物理尺寸，只测一致性） | 100 |
| `groundTruth.fx` / `fy` / `cx` / `cy` | double | ground-truth 内参（无畸变，D=0 在代码内） | 1200 / 1200 / 640 / 480 |
| `baselineX` | double | L→R 基线 x 平移（板坐标系） | 100.0 |
| `poses[]` | array | 姿态候选表（按序消费至凑满 `targetGood`）：`yawDeg`/`pitchDeg` 角度、`z` 距离 | 18 项 |
| `frames.targetGood` | int | 凑满即停的合格帧目标数 | 12 |
| `frames.minDetected` | int | 合格帧数下限（ASSERT 门禁） | 10 |
| `intrinsic.useCalibrateCameraRO` | bool | `IntrinsicCalibParams::use_calibrateCameraRO` | false |
| `intrinsic.reprojErrorThreshold` | double | `IntrinsicCalibParams::reproj_error_threshold` | 1.0 |
| `gates.reprojMeanMax` | double | 主断言：重投影均值上限（px） | 1.0 |
| `gates.validFramesMin` | int | 主断言：有效帧数下限 | 10 |
| `gates.fxTolerance` | double | 次断言：fx/fy 与 groundTruth 容差（px） | 60.0 |

## 3. 参数源语义（重要）

- **JSON 唯一源**：测试代码不含参数默认值（仅 `D_gt`＝全零、`patSize` 推导、角度 rad 换算属代码逻辑）。
- **缺失即失败**：`config.json` 找不到 / JSON 语法错 / 缺键 / 类型错 → `GTEST_FAIL` 带键名（**非 SKIP**——config 随测试入 git，缺席即工程损坏）。
- **定位策略**：候选路径列表逐个探测（同 `test_camera_chain_analysis` 惯例）：
  1. `factory_calib/module1_camera/tests/e2e/config.json`（repo 根直跑）
  2. `../../factory_calib/module1_camera/tests/e2e/config.json`（ctest cwd＝`build_fc1_rel/module1_camera`）
  3. `../../../factory_calib/module1_camera/tests/e2e/config.json`（更深构建目录）
  4. `E:/JEAMMWARE2601001/factory_calib/module1_camera/tests/e2e/config.json`（绝对兜底）

## 4. 构建联动（CMake）

- 本文件夹测试靠 `module1_camera/CMakeLists.txt` 的 `file(GLOB_RECURSE tests/*.cpp)` ＋ 分类正则 `tests[\\/].*test_[^/\\]*\.cpp$` 捕获（**任意深度子目录**均可，2026-10-04 随本文件夹建立放宽）。
- 新增源文件须以 `test_` 前缀命名才会注册为测试可执行；非 `test_` 前缀的 `.cpp` 会被静默跳过（不编译）。
- glob 在 configure 期求值：增删文件后需重新 configure（改 CMakeLists.txt 会自动触发）。

## 5. 修改守则

1. **调参数只改 `config.json`，不改测试代码**（改姿态表/门限/图像尺寸等均如此）。
2. 改键名须三处同步：`config.json` ＋ `loadConfig()`/`E2EConfig` ＋ 本文档 §2。
3. 本文件夹属 `factory_calib/` 保护区：AI 改动需人工明确指令（上级 AGENTS.md 第一部分）。
