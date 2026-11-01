# opt_bench — 相机标定优化实验场

## 定位

在不影响主线的前提下，对 `camera_calib` 的流程与算子做优化实验：
**本文件夹内的代码不改动 operators/、CLI、现有测试一个字**；实验结论达标后由人工决定是否并入主线。

合入提案（证据包+分批方案+放行清单）：`../../docs/plans/2026-10-13-module1-camera-opt-bench-adoption.md`

CMake 零改动自动收录：`tests/opt_bench/test_*.cpp` 会被模块1 的递归 glob 注册为独立测试可执行。
注意：本目录下**非 test_ 开头的 .cpp 不会被编译**（路由规则会丢弃），共享代码一律 header-only。

## 数据与基线

- 基准数据集：`../../test_data/camera261006-1688`（143 对左右图）
- 数据集内 `camera_calib.json` 是现役流程的历史输出，基准测试用它做框架自检
- `marked_gates*.csv` 是人工标注的坏帧研究记录，坏帧剔除实验用它做参照
- 数据集缺席时测试 SKIP 不报错（可用环境变量 `FC_OPT_BENCH_DATA` 指定其他目录）

## 验收标准（分实验类型）

| 类型 | 例子 | 标准 |
|---|---|---|
| 流程类（不改数值） | 并行角点提取 | 输出与基线**逐点/逐字段一致** |
| 算子类（改数值） | 两阶段检测、坏帧剔除 | 允许数值差异，但精度指标**不得劣化**、有效帧数不得减少 |

## 现有内容

| 文件 | 内容 |
|---|---|
| `bench_utils.h` | header-only 共享件：计时器/六步流水线复刻/指标计算/报告落盘 |
| `test_calib_baseline.cpp` | 基准：复刻 CLI 六步，分步+逐帧计时，与历史基线对拍自检 |
| `test_opt_parallel_corner.cpp` | 实验S1：并行角点提取 A/B（流程类，逐点一致验收） |
| `test_opt_sb_two_stage.cpp` | 实验S2：SB 检测两阶段（快检优先、失败回退穷举；算子类验收） |
| `test_opt_frame_reject.cpp` | 实验A1：按重投影误差剔坏帧重标定（算子类验收） |
| `test_diag_intrinsic_threads.cpp` | 诊断：内参 OpenCV 内部线程并行度（3+3 次 ABBA 取中位） |
| `test_opt_intrinsic_lu.cpp` | 实验E1：CALIB_USE_LU 换解方程器（数值容差验收：K≤1e-8/畸变≤1e-7/rms≤1e-6/下游≤1e-3） |
| `test_opt_intrinsic_parallel.cpp` | 实验E2：左右相机并行标定（LU 口径，全局线程≤12，跨臂跨次逐字节验收） |
| `test_opt_corner_10s.cpp` | 实验C10s：角点提取 78s→10s 四臂实验（12w+setNumThreads(1)+半分辨率快路径+位移守卫回退；门禁=有效集相等/逐目分层对拍/下游全指标；实测 7.8s） |

## 产物

全部落在 `out/`（已 gitignore）：

- `baseline_snapshot.json` — 基线快照（参数+指标+分步耗时+完整输出 JSON）
- `timing_baseline_steps.csv` / `timing_baseline_frames.csv` — 步骤表/逐帧角点耗时表
- `exp_*_report.json` — 各实验 A/B 对拍报告

## 如何加新实验

1. 新建 `test_opt_<名称>.cpp`，A/B 两臂同数据进程内对拍
2. 流程类改动用 `PipelineOptions` 的开关（如 `cornerThreads`）；算子类改动传自定义 `CornerFn`
3. 按类型断言验收标准，报告写 `out/exp_<名称>_report.json`
4. 达标 ≠ 上线：并入主线需人工发起正式 review + 全量回归
