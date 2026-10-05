# module1_camera 五文件简记

> 2026-09-01 由人工指令生成：简洁记录 `module1_camera/` 下 5 个非算子源文件内容（算子本体在 `operators/`，此处不涉及）。
> 2026-09-04 对照代码更新：温度窗统一 ±15℃/0.5 步距（61 档，对齐模块2 CMTT）；stereoRectifyTempTable 新增下游消费方。

## calib_io.h / calib_io.cpp — 配置与 IO（命名空间 `fc`）

**职责**：模块1 的 JSON 配置读取、图像对装载、结果汇总写出。

- `CameraCalibConfig`：全部标定参数（带默认值）＋ `fromJson()` 解析 `config.json`
  - 棋盘格：11×8 内角点、方格 2.0mm；图像 2048×1536
  - 内参：`useCalibrateCameraRO=true`、重投影阈值 0.012
  - 温度：cte=23.6e-6、参考温度 22.5、范围 ±15℃、步长 0.5（**2026-09-02 统一为 61 档**，与模块2 CMTT 温网对齐；补偿算子 Params 缺省同步改）
  - 矫正：alpha=0、`CALIB_ZERO_DISPARITY`
  - 标定板热膨胀系数 5.0e-6、板温 21.0（喂 intrinsic 的 temperature_coeff）
  - 解析失败仅 warn 并用默认值；字段逐个 `contains` 判空，缺省不覆盖
- `FramePair`（left/right 灰度图）/ `CameraInput`（config＋帧对列表）
- `loadCameraInput(dir)`：
  1. 读 `config.json` → 读 `temps.txt` 的 `ref_temp` 行覆盖参考温度（可选）
  2. 遍历 `left/` 下 png/jpg 排序，配对 `right/` 同名文件（缺右图/读失败跳帧并 warn）
  3. `imread` 灰度载入；0 帧返回 `nullopt`
- `buildCameraCalibJson(...)`：复用各算子 `toJson()`，汇总 schema `factory_calib.camera_calib.v1` ＋ imageSize/referenceTemp/cte/temp 三键 ＋ intrinsic/extrinsic/rectify ＋ 4 张温度表（intrinsicTempTableL/R、extrinsicTempTable、stereoRectifyTempTable）。**stereoRectifyTempTable 整节由模块2 handoff 容错解析消费（CMTT 4-14 源①）**——模块1 输出须含该节且参考温锚点与模块2 config 一致
- `writeJson()`：缩进 2 写出，失败返回 false

## camera_calib_cli.cpp — CLI 入口（camera_calib.exe）

**用法**：`camera_calib <input_dir> [output_json]`（缺省输出 `camera_calib.json`；参数不足返回 2）。

主流程（任一步失败打印错误并 return 1）：

1. **角点提取**：逐帧 `extractChessboardCorners`（L/R 各一次）＋ `normalizeLRCornerOrder`；失败帧跳过；有效帧 <4 报错退出
2. **内参**：`IntrinsicCalibCPU`（参数映射自 config；含板热膨胀 temperature_coeff），打印 reproj_mean
3. **外参**：`ExtrinsicCalibCpu::Execute(KL,DL,KR,DR)` 重载直接传内参结果（Params 不持 K/D）；物点网格行主序＋板热膨胀（`actualSize = squareSize × (1 + plateTempCoeff × (plateTemp−20))`）；`maxReprojError = 阈值×100`、`minViewCount=4`
4. **立体矫正**：`StereoRectifyCpu`（K/D/R/T 来自前两步结果＋config alpha/flags）
5. **三张温度表**：
   - 内参表 L/R：`IntrinsicCompensateCPU`（从 K 取 fx/fy/cx/cy 组 `CameraIntrinsics`）
   - 外参表：`ExtrinsicCompensateCPU`（R 9 元素＋T 3 元素展开）
   - 矫正表：`StereoRectifyTempTableCpu`
6. **写出**：`buildCameraCalibJson`＋`writeJson`，成功 return 0

## chessboard_corner.h / chessboard_corner.cpp — 棋盘格角点检测

- `extractChessboardCorners(gray, params, result)`：
  - 非 8U 图先转换；`findChessboardCornersSB`（EXHAUSTIVE＋NORMALIZE_IMAGE）粗提
  - 数量须 == patternSize.area()，否则 found=false 返回 false
  - `cornerSubPix` 亚像素精化（11×11 窗、准则 EPS+MAX_ITER/30/1e-4）
  - 诊断量 `meanSubpixDelta`：精化前后平均位移（px）
- `normalizeLRCornerOrder(left, right)`：L/R 编号一致性
  - 判据：两图首→末角点向量夹角 >90°（cos<0）时把右图逆序，保证 `corners[i]` 对应同一物理角点
  - 未找到/数量不等/零向量 → 返回 false
