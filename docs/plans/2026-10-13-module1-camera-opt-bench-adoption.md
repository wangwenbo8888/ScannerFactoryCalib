# module1_camera 优化合入提案（opt_bench 证据包）

> 2026-10-13 · 依据 `module1_camera/tests/opt_bench/` 实验场全部已过实验 · 供人工决策分批合入主线
>
> 红线：本文档只汇总证据与方案，不含任何已执行的主线改动。

## 1. 目的与范围

相机标定全流程实测 10.6 分钟（数据集 `test_data/camera261006-1688`，143 对图/129 有效帧），
其中内参标定占 85%。本提案申请将三项**已通过实验验收**的优化分批并入主线，
预期全流程压至 **约 40 秒**（含并行角点）。

## 2. 实验场方法（可信度前提）

- 实验全部位于 `tests/opt_bench/`，主线（operators/CLI/CMake）零改动
- 基准测试复刻 CLI 六步流程，与数据集内历史 `camera_calib.json` 对拍自检通过
  （rmsL 0.05748 / rmsR 0.06548 / epi 0.3242 / 129 帧，全部吻合）
- 验收双轨制：流程类改动要求逐字节/逐点一致；算子类改动要求数值容差内不劣化
- 计时规范（审核修正后）：3+3 次 ABBA 换序取中位；本机同臂极差可达 ±20%，
  解读小于 1.5x 的计时差异需谨慎
- 多代理审核：方案已经过数值正确性 / 工程落地 / 实验设计三路独立审核，
  两项方案被驳回或降级（见 §5）

## 3. 分步基准（优化前）

| 步骤 | 耗时 | 占比 |
|---|---|---|
| 读图 | 4.9 s | 1% |
| 角点提取 | 78 s | 13% |
| **内参标定** | **515 s** | **85%** |
| 外参标定 | 40~49 s | 7% |
| 矫正/温度表/写盘 | <1 s | ≈0 |

根源（OpenCV 4.13 源码级核实）：`calibrateCameraRO`/`stereoCalibrate` 每轮 LM 迭代
对约 780×780 阻尼正规方程做**稠密 SVD**（OpenCV 自带实现慢），且默认终止条件恒跑满轮数；
OpenCV 内部多线程对此几乎无效（加固诊断实测仅约 1.2 倍贡献）。

## 4. 三项待合入优化（按建议顺序）

### 4.1 批次一：内参 LU 解方程器（收益最大·改动最小·先做）

- **改动**：主库 `module1_camera/camera_calib_params.json` 的 `intrinsic_flags`: 0 → **131072**
  （= `CALIB_USE_LU`，1<<17；该键为整值替换，当前值 0，直接写 131072 即可）
- **证据**（`out/exp_intrinsic_lu_report.json`，3+3 次 ABBA 取中位）：
  - 578 s → **6.3 s，提速 92.4 倍**（SVD 臂 526/578/660 s，LU 臂 6.2/6.3/6.9 s）
  - 数值门全部通过且富余 6 个数量级：K 相对差 7.0e-14（门 1e-8）、
    畸变系数差 1.3e-4 倍容差（门 1e-7 相对）、rms 相对差 6.6e-15/7.8e-15（门 1e-6）、
    下游极线相对差 1.2e-12、行对齐相对差 0（门 1e-3）
- **⚠ 配置覆盖排查**（present-key 逐叶覆盖，含此键的数据集会把主库 LU 静默覆盖回 SVD）：
  - `module1_camera/test_data/camera/config.json:8`（`intrinsic_flags: 0`）→ 需同步改 131072
  - `module1_camera/tests/e2e/data_out/config.json:8`（同上）→ 需同步改
  - **基准数据集（2026-10-13 人工裁定）＝ `test_data/camera261006-1688`**：无 config.json/temps.txt，
    直接吃主库参数，改主库即生效、零覆盖风险
  - `data_in/CESHI260831`：人工裁定**不在合入范围**，前置确认项解除
  - exe 旁主库副本随 POST_BUILD `copy_if_different` 重建时自动同步（改完源码树需重编或手拷）
- **回滚**：键值改回 0，即刻恢复 SVD 行为
- **机制说明**：LU 带部分主元；JtJ 阻尼后良态。审核附条件：LU 无 SVD 的小奇异值截断、
  `CvLevMarq::step()` 忽略 solve 返回值——已由上述 6 个数量级富余的数值门覆盖实证

### 4.2 批次二：并行角点提取（S1，流程类）【2026-10-14 已合入 ✅】

- **落地形态（保守版）**：CLI 角点循环 12 worker 帧级并行＋提取期间 setNumThreads(1)（存还原）；
  配置键 `corner.workers`（主库 12，置 1=回滚串行）；半分辨率快检测（实验 7.8s）因 B 类扩集
  （救回 11 个基线漏检帧→极线劣化 8%）未合入，留档 `test_opt_corner_10s`
- **验收**：camera261006-1688 复跑 115.8s→79.9s（角点 78→29s）；
  输出与批次一金标 SHA256 逐字节一致；交付回归 10/10 绿

- **改动**：`camera_calib_cli.cpp` 角点提取循环改为多线程（每帧独立：左/右提取+编号归一），
  按帧号归位回顺序结构；帧数 143、线程 8；建议日志加 L/R 帧号前缀（防交错难读），
  并确认产线脚本不依赖 stdout 行序（spdlog 走 stderr 不受影响）
- **证据**（`out/exp_parallel_corner_report.json`）：
  - 74.9 s → 27.7 s，**提速 2.70 倍**（8 线程）
  - **逐点零差异**：pointMismatch=0、patternMismatch=0、maxPointDelta=0
    （每帧独立计算+按序归位，理论上逐字节一致，已实证）
- **回滚**：线程数参数化（如 `cornerThreads` 配置键），置 1 即回退串行
- **注意**：OpenCV 内部线程池与本改动正交（帧级并行）；不建议同批并入其他改动，
  独立批次独立回归

### 4.3 批次三：外参 LU 解方程器（需动算子内部·背同步债务·缓行）【2026-10-14 已合入 ✅】

- **落地形态**：factory/09 两侧算子同步一行（stereoCalibrate flags +CALIB_USE_LU，零差异保持）
- **验收**：复跑 79.9s→42.2s（外参段 0.78s）；delta 与 E6 预言逐项吻合（rms 逐位一致、R/T 差万分之三容差）；
  factory 交付回归 10/10＋主工程分范围 12/12 绿（test_calib_camera 消费链含）
- **遗留**：主工程全量回归被 mask_sep_cli 既有编译错阻断（与本批无关），待该工具修复后补跑

- **改动**：`module1_camera/operators/extrinsic_calib/extrinsic_calib_cpu.cpp:558`
  `cv::CALIB_FIX_INTRINSIC` → `cv::CALIB_FIX_INTRINSIC | cv::CALIB_USE_LU`（一行）
- **证据**（`out/exp_extrinsic_lu_report.json`，3+3 次 ABBA，直调 OpenCV 复刻算子调用）：
  - 49.0 s → **0.72 s，提速 67.7 倍**（SVD 45.2/51.2/49.0 s，LU 0.71/0.72/0.73 s）
  - 数值门全过：R 差 3.5e-4 倍容差、T 差 1.9e-4 倍容差、rms 相对差 3.6e-13、
    极线相对差 2.1e-13（门均为实质等值）
- **同步债务**（审核已核实当前状态）：09 侧镜像
  `modules/09_operatorlib/calibration/camera/extrinsic_calib/extrinsic_calib_cpu.cpp`
  与 factory 侧**当前逐字节零差异**（fc 已核对）；改 factory 侧后必须同改 09 侧
- **主工程回归范围**：09 侧算子被 07 消费（`calib_workflow.cpp:63`、
  `CameraChain.cpp:129`），同步后需主工程全量 ctest（100 项口径）
- **回滚**：两侧行级改回（保留 diff 即可逆）

## 5. 已验证暂缓项与已否决项

| 项 | 状态 | 说明 |
|---|---|---|
| A1 坏帧剔除 | 已验证·**暂缓** | 自动剔 6 帧与人工标注 6/6 重合，全指标改善（rms 0.0575→0.0539、epi 0.3242→0.3223、row 0.0335→0.0331）。属"改变输出值"类，建议等速度类合入跑稳后单独立项（报告 `out/exp_frame_reject_report.json`） |
| S2 两阶段检测 | 已否决 | 精度无损但慢 26%（0.74x），数据否决 |
| E2 内参左右并行 | 已验证·不采纳 | 1.48 倍且逐字节一致，但 LU 后仅省约 3 秒，无价值 |
| E3/E4 迭代轮数/两阶段初值 | 未做·无必要 | LU 后内参仅 6 秒 |
| E5 视图削减 | 未做·降级 | 提速动机消失；若未来为精度再做，须按审核补 hold-out 外部验证 |

## 6. 统一放行流程（每批次）

1. 冻结该批次改动前的 SVD/串行**黄金基线**（现有 `out/baseline_snapshot.json` 即基线快照）
2. 合入改动 → 真实数据集复跑 `camera_calib.exe camera261006-1688`
3. 产出 **delta 报告**：新旧 JSON 逐项容差对拍（K/D/rms/R/T/F/四温度表），不要求字节一致
4. 人工 code review（保护区先例：`14760f1`）
5. 模块1 Release 全量 ctest 回归（批次三另加主工程 100 项）
6. 计时复测确认收益（注意本机 ±20% 噪声，以中位为准）
7. 产线灰度 + 回滚预案就位（批次一：键改回 0；批次二：线程置 1；批次三：两侧行回退）

## 7. 预期收益（分批累计）

| 阶段 | 全流程耗时 | 依据 |
|---|---|---|
| 现状 | 10.6 min | 基准实测 |
| +批次一（内参 LU） | ~1.6 min | 515→6.3 s |
| +批次二（并行角点） | ~50 s | 78→28 s |
| +批次三（外参 LU） | **~40 s** | 40→0.7 s |

## 8. 风险与遗留

1. **LU 静默失败路径**：`CvLevMarq::step()` 不检查 solve 返回值（OpenCV 行为）；
   实证数值门富余充分，但换数据集/换 OpenCV 版本时须重跑 E1/E6 对拍
2. **本机计时噪声 ±20%**：合入后性能复测一律 3+3 ABBA 取中位
3. ~~`data_in/CESHI260831` 配置人工确认~~（2026-10-13 已解除：人工裁定不在合入范围，
   基准数据集＝camera261006-1688，无覆盖风险）
4. **analyze_camera_calib 对拍工具**：为指标对拍制非字节 diff，微差进一致性指示器；
   其自带回归基准数值可能需随新基线更新（见该工具 AGENTS §维护约定）
5. 模块2/06 交接不受影响：schema 与参考温锚点不变，仅数值微差（1e-13 级）

## 9. 附录：证据文件索引（`module1_camera/tests/opt_bench/out/`）

| 文件 | 内容 |
|---|---|
| `baseline_snapshot.json` | 优化前基线快照（指标+分步耗时+完整输出） |
| `timing_baseline_steps/frames.csv` | 分步/逐帧耗时表 |
| `exp_intrinsic_lu_report.json` | E1 内参 LU（92.4x，数值门全过） |
| `exp_extrinsic_lu_report.json` | E6 外参 LU（67.7x，数值门全过） |
| `exp_parallel_corner_report.json` | S1 并行角点（2.70x，逐点零差异） |
| `exp_intrinsic_parallel_report.json` | E2 并行内参（1.48x，不采纳） |
| `exp_frame_reject_report.json` | A1 坏帧剔除（改善，暂缓） |
| `exp_sb_two_stage_report.json` | S2 两阶段（0.74x，否决） |
| `exp_diag_intrinsic_threads_report.json` | 线程诊断（内部弱并行 1.2x） |
