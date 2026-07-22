# factory_calib — 厂家标定工程（自包含）

两个独立可执行：
- `module1_camera/camera_calib.exe` — 相机内/外参 + 立体矫正 + 温度补偿表（CPU）
- `module2_laser/laser_calib.exe` — 激光虚拟相机标定 + 温度补偿表（CUDA）

详细设计见 `../docs/plans/2026-07-18-factory-calib-design.md`，实现计划见 `../docs/plans/2026-07-18-factory-calib-impl.md`。

## 前置依赖

| 依赖 | 版本 | 路径（默认） | 备注 |
|---|---|---|---|
| Visual Studio | 2022 / MSVC v14.44 | `C:/Program Files/...` | 工具集 v143 |
| OpenCV | 4.13.0（含 CUDA 模块）| `C:/opencv-cuda-4.13.0` | Release 带全部组件（含 imgcodecs） |
| OpenCV Debug（可选）| 4.13.0 | `C:/opencv-cuda-4.13.0-debug/x64/vc17/lib` | **不含 imgcodecs**，仅供算子自测 |
| CUDA Toolkit | 12.4+（pin v12.6 实测 v12.4.99） | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6` | 仅模块2 需要；sm_75/86/87 |
| Eigen | 3.4.1 | `C:/devlibs/eigen-3.4.1-install` | |
| spdlog / nlohmann_json / GoogleTest | 自动 FetchContent | — | GitHub 被封需 `-DJMW_GH_MIRROR=https://ghfast.top/` |

## 构建命令

### 模块1（无 CUDA，任何机器可构建）

```powershell
cmake -S factory_calib -B build_fc -DFC_BUILD_MODULE2=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc --config Release
ctest --test-dir build_fc -C Release --output-on-failure
```

Debug 配置同主工程规范（独立 build 目录，`-DCMAKE_CONFIGURATION_TYPES=Debug`）。

### 模块2（CUDA + Release OpenCV）

```powershell
cmake -S factory_calib -B build_fc2_rel -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Release `
      -DFC_BUILD_MODULE1=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc2_rel --config Release
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc2_rel -C Release --output-on-failure
```

Debug（仅算子自测，CLI 部分 Release-only）：

```powershell
cmake -S factory_calib -B build_fc2 -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Debug `
      -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib `
      -DFC_BUILD_MODULE1=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc2 --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc2 -C Debug --output-on-failure
```

### 全量（双模块）

```powershell
cmake -S factory_calib -B build_fc_all_rel -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Release `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc_all_rel --config Release
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc_all_rel -C Release --output-on-failure
```

## 输入/输出目录约定

### 模块1 输入（`data_in/camera/`）

```
camera/
├── config.json         棋盘格 + 算子参数（参考 §5.1）
├── temps.txt           参考温度（单行 "ref_temp 22.5"）
├── left/
│   ├── 001.png  002.png ...   棋盘格图（CV_8UC1 或转灰度）
└── right/
    └── 001.png  002.png ...
```

### 模块1 输出（`data_out/camera_calib.json`）

字段：`schema / imageSize / referenceTemp / cte / tempRange{Min/Max/Step}`
+ `intrinsic` (left/right K/D + rvecs/tvecs + rms)
+ `extrinsic` (R/T/E/F + 可选 K_L/R)
+ `rectify` (R1/R2/P1/P2/Q + validRoi{L/R})
+ 4 张温度表（intrinsicTempTableL/R, extrinsicTempTable, stereoRectifyTempTable）

### 模块2 输入（`data_in/laser/`）

```
laser/
├── config.json              激光链参数（gridStep/depthMin/Max/epipolarStep/lineIds…）
├── camera_calib.json        ← 拷贝/软链自 data_out/camera_calib.json
├── pose_00/
│   ├── L_tube0.png  R_tube0.png
│   ├── L_tube1.png  R_tube1.png
│   └── temp.txt
├── pose_01/  ...
└── pose_24/
```

### 模块2 输出（`data_out/laser_calib.json`）

字段：`schema / build / posesProcessed / framesOk / accumulatedPoints3D`
+ `virtualK/R/T`（4-11 优化后）
+ `laserExtrinsicTempTable`（virtual→L/R 每温度，5-3 输出）
+ `planeMapTempTable`（每温度映射，4-13 输出）

## 运行示例

```powershell
# 模块1：相机标定
.\build_fc_all_rel\module1_camera\Release\camera_calib.exe `
    data_in\camera data_out\camera_calib.json

# 模块2：激光标定（需把模块1 输出软链/拷贝到 data_in\laser\camera_calib.json）
Copy-Item data_out\camera_calib.json data_in\laser\camera_calib.json
.\build_fc_all_rel\module2_laser\Release\laser_calib.exe `
    data_in\laser data_out\laser_calib.json
```

`laser_calib.exe` 退出码：0 = 完整流水线成功；1 = 部分失败（任何算子 `success=false`、或 `haveVirtualPose=false`、或无累积 3D 点）。stderr 通过 spdlog 打印详细诊断。

## 已知限制

1. **模块2 CLI 仅 Release 可构建**：Debug 自定义 OpenCV 缺 `imgcodecs`（`cv::imread` 所需），CMakeLists 自动跳过 `fc2_io`/`laser_calib` 构建；算子自测（fc2_ops + tests）Debug 可跑。
2. **模块2 端到端精度需真实样本数据**：当前 Task 6.3 只覆盖冒烟（合成全黑图）。真实激光线图 + ground-truth virtual pose 的精度验证需用户提供产线样本。
3. **nvcc 版本飘移**：顶层 CMakeLists pin v12.6 但实测 `v12.4.99` 被选中（CMake cache 无 FORCE），但能编译 + 测试绿，按经验 #4 接受。
4. **CRT 部署**：`fc_deploy_crt()` 自动把 `MSVCP140.dll` / `vcruntime140*.dll` 拷到 exe 同目录；OpenCV DLL 仍需 PATH 设置。
5. **GitHub 封锁**：所有 configure 命令必须带 `-DJMW_GH_MIRROR=https://ghfast.top/`。

## 测试覆盖

| 层级 | 测试文件 | 数量 | 状态 |
|---|---|---|---|
| L1 算子回归（模块1）| `operators/*/tests/test_*_cpu.cpp` | 6 | Release 绿 |
| L1 算子回归（模块2）| `operators/*/tests/test_*_cuda*.cpp` | 16 | Debug/Release 双绿 |
| L2 棋盘格（模块1）| `tests/test_chessboard_corner.cpp` | 5 | Release 绿 |
| L3 端到端（模块1）| `tests/test_camera_calib_e2e.cpp` | 1 | Release 绿 |
| L3 端到端（模块2）| `tests/test_laser_calib_e2e.cpp` | 2 | Release 绿（冒烟）|
| L4 交接（模块2）| `tests/test_handoff.cpp` | 7 | Release 绿 |
