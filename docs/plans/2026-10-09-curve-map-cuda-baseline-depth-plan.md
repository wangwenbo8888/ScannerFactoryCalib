# curve_map_cuda 基线景深参数化＋黄金测试 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> （用户指令：直接连续执行到运行通过，不再询问；不提交 git）

**Goal:** 为 CurveMapCuda 增加基线景深窗 API（SetBaselineDepthWindow/Get），新建
tests/baseline_depth/ 黄金测试（合成 A1–A5＋真数据），构建并运行至全绿。

**Architecture:** 概念上基线景深=现行 depthMin/Max（设计文档 §2 已证），故 API 是
语义别名写同一真相源；测试经 CurveMapTable 公有解码接口逐行流式验证条目深度。

**Tech Stack:** CUDA(nvcc -fmad=false) / gtest / nlohmann_json / CMake。

**权威规格:** `docs/plans/2026-10-09-curve-map-cuda-baseline-depth-design.md`（§3 API、
§4 测试矩阵、§5 论证、红线 1–4）——本计划只列任务序，规格以设计文档为准。

---

### Task 1: curve_map_cuda API（只加不改）

**Files:**
- Modify: `module2_laser/operators/curve_map_cuda/curve_map_cuda.h`（+`<utility>`、两声明＋基线景深定义注释）
- Modify: `module2_laser/operators/curve_map_cuda/curve_map_cuda.cu`（文件尾追加两实现；isfinite 前置＋临时对象先验后交）

**Step 1:** 写实现（isfinite→tmp 赋 float→tmp.validate()→提交；Get 从 float 扩回 double）。
**Step 2:** 编译 `cmake --build build_fc2_rel --config Release --target fc2_curve_map_cuda` 通过。

### Task 2: CMake 测试注册 elseif

**Files:**
- Modify: `module2_laser/CMakeLists.txt`（foreach 内、跳过分支**之前**插入
  `elseif(f MATCHES "tests[\\\\/][^/\\\\]+[\\\\/]test_[^/\\\\]*\\.cpp$")` → M2_TESTS）

**Step 1:** 插入 elseif。**Step 2:** configure 零报错、新 target 出现。

### Task 3: 新测试文件（A1–A5 合成组）

**Files:**
- Create: `module2_laser/tests/baseline_depth/test_curve_map_cuda_baseline_depth.cpp`

合成场景（红线 1–4 全部自证）：640×480、f=500、pp=(320,240)、B=60、virtualT=(30,10,12)、
rowStep=0.5；两条直线 conic：F1=u−0.3v−50、F2=u+0.2v−500；窗 (d1,d2)=(120,400)。
镜像 sweep（复刻 kernel 公式，仅去 Z 窗过滤）自证：扫深跨骑窗两端、W·2≤4095、
nRows·nLines≤65535、零窗 (1, zAchMin−1) 前置 zAchMin>2。

断言按设计 §4.2 表格逐行实现（δ 精确式现算；EnumerateRow −1 区分；空表用
entryCount()==0 勿用 valid()；A4 条目级对拍不 memcmp；A5 回显按 float 口径）。

**Step 1:** 写测试。**Step 2:** 构建 target。**Step 3:** 运行（PATH 含
`C:\opencv-cuda-4.13.0\x64\vc17\bin`），A1–A5 全绿；失败则迭代至绿。

### Task 4: 真数据用例（并入 Task 3 文件）

golden 式根探测（env `FC_BASELINE_DEPTH_ROOT` → CWD 上溯 → E:/JEAMMWARE2601001/factory_calib）：
handoff=`module2_laser/test_data/laser261006-1668/camera_calib.json`、
laser=`data_out/laser261006-1668_depth100-500/laser_calib.json`；缺席 GTEST_SKIP。
复刻 toCurveMapInput＋hasPjcF 诊断；窗 (100,500)；断言 25 线全有、逐线 Z 界（δ 精确式
由 JSON f/B 现算）、rowStep==0.5f、窗口头回读、entryCount∈[15M,25M]；逐行流式。
可选 `FC_BASELINE_DEPTH_SIDECAR=1`：CmttReader 参考档 entryCount 对拍。

**Step 1:** 实现。**Step 2:** 构建运行全绿（含真数据用例实跑，本机数据在位）。

### Task 5: 回归与文档

**Step 1:** 既有套件抽样回归：`test_curve_map`、`test_cmtt_cuda`、`test_golden_ref_tier` 单跑绿。
**Step 2:** 文档卡 `docs/operator_docs/curve_map_cuda/curve_map_cuda-极线切曲线映射表CUDA.md`
补记新 API＋基线景深定义。**Step 3:** 汇报（不提交 git）。
