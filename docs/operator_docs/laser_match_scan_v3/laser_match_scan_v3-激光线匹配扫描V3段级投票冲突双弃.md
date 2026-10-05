# 激光线匹配扫描 V3（段级投票 + 冲突双弃）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为 factory_calib/module2_laser/operators/laser_match_scan_v3
> 金标准（pose_06 算子实测，**基线 A**＝上游 interp_dual 无单侧投影兜底，2026-09-01 前）：matched=14004 / correct=14004 / wrong=0（v2 GPU 实测 13244 匹配、含 2564 错配；CPU 模拟口径 v2pairs=13243）
> **基线 B**（上游 interp_dual 启用单侧点垂直投影兜底，2026-09-01 起）：matched=24866 / tracks=64 / dropped=11 / lost=3610；金标准正式断言仍全过（真值已知点 19772/19772 零错配），诊断计数 correct=16162 / wrong=8704 系长段跨真值 lid 边界的归属混计（判读注意见 G/K 段）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-7-V3`（扫描期调用） |
| 中文名称 | 激光线匹配扫描 V3（段级投票 + 冲突双弃） |
| 英文目录名 | `laser_match_scan_v3` |
| 运行平台 | CUDA（全 GPU 管线，CUB DeviceSelect/DeviceRadixSort） |
| 所属流程 | 扫描流程（epipolar_interp_dual 之后、laser_reconstruct 之前） |
| 精度档次 | `③`（浮点类；依赖映射表精度 + match_window） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机插值点集（矫正坐标系） | `GpuMat`（CV_32FC2） |
| **输入②** | 左相机**连通段 ID**（前端 epipolar_interp_dual Scan 模式产出段 ID） | `GpuMat`（CV_32SC1），预期为段 ID；本算子不校验取值（含 0 也建段） |
| **输入③** | 右相机插值点集 + 右段 ID（同①②格式） | 同上（右段 ID 仅用于行 CSR 排序散射，不参与段判决；输出段 ID 取自左段） |
| **输入④** | 温度补偿映射表 | `SetTempTable(shared_ptr<const LaserPlaneMapTempTable>)` |
| **输入⑤** | 当前温度 | `SetCurrentTemperature(double)` |
| **输出** | 紧凑配对数组（L/R 点 + **查表 lid + 段 ID**）+ 段诊断 + 各点状态 | `LaserMatchScanResultV3` |

## C. 算法

**核心思想**：输入点按段组织，发卡（每 lid 取最近 sweep 条目 ±match_window 内有右点即生效并记录 hitRx），段内投票选出每段的 winner lid；若同一极线行出现两段持相同 winner，则两段**全部丢弃**（不猜谁对谁错）；保留段直接取 winner 卡片的 (hitRx, lid) 作为配对输出。

**核心流程**（5 阶段 GPU 管线，中间一次 host 同步构建段表）：

```
Step 1-4  右侧排序/CSR：GenKeys → RadixSort → Scatter → 行 CSR
          （左侧不排序，所有操作在原始索引空间）
Step 5    kernelIssueCardsV3:
          每左点查表 row 行桶内二分 → 展开 ±vL_tolerance 收集
          → 每 lid 取最近条目（单卡制，同 lid 就近更新）
          → 预测 uR ±match_window 行内右点二分搜索
          → 卡片 (lid, hitRx) 写入 d_card_lid_/d_card_hitRx_
Step 6    [host] 段表构建：
          D2H 左段 ID → host sort/unique → segCount 段
          → orig→segIdx 映射表 U2H
Step 7    kernelSegVoteV3 → kernelSegWinnerV3 → kernelConflictMarkV3:
          (段,lid) 票数直方图 → 每段 argmax winner
          → (行,winner-lid) 桶 atomicCAS：首段写入，异段标记双方 drop
Step 8    kernelGatherV3:
          保留段每点找 lid==winner 的卡 → (lx,hitRx,lid,trackId) + flag
Step 9    CUB DeviceSelect::Flagged ×4 → 紧凑配对数组
Step 10   [host] D2H 计数 + 段诊断统计
```

**关键函数**：

| 函数 | 用途 |
|------|------|
| `kernelIssueCardsV3` | 就近单卡发卡+HIT 验证（核心 kernel） |
| `kernelSegVoteV3` | 段内 (段,lid) 票数直方图 |
| `kernelSegWinnerV3` | 段级 winner 选举 |
| `kernelConflictMarkV3` | 行冲突检测 + 双弃标记（atomicCAS） |
| `kernelGatherV3` | 保留段 winner 卡片收集 |
| `cub::DeviceSelect::Flagged` | 四路配对数组紧凑化 |

## D. 依赖

```
epipolar_interp_dual(L) ─┐   ← 段 ID 必须有效（scan_link_dx>0 才有段 ID）
                         ├→ 本算子 → laser_reconstruct
epipolar_interp_dual(R) ─┘
curve_map → plane_map_temp_table → SetTempTable()
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 与 v1/v2 共存 | 独立算子类 `LaserMatchScanCudaV3`，不依赖 v2 代码 |
| 表处理 | `uploadMapTable`（含 byrow CSR 构建）复用 v2 逻辑 |
| 排序 kernel | `kernelGenKeysIdxV3`/`kernelScatterSortedV3`/`kernelBuildCSRV3` 逻辑同源 v1/v2（本算子文件内独立拷贝，非链接级共享） |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `e1.d_interpPoints`（CV_32FC2） | 值传递（ROI 视图） | 左插值点集 |
| `e1.d_interp_line_ids`（CV_32SC1） | 值传递（ROI 视图） | **段 ID**（scan_link_dx>0 时为行间链接段 ID；=0 时为簇 ID——同段行间不重号但不跨断口合并，段更碎）。2026-09-01 起上游簇内插值新增**单侧点垂直投影兜底**（仅上/下窗一侧有点时取该侧最近点投影到 y_k，行边缘簇不再丢弃）→ 断口减少、同线段合并变长（pose_06：142 段→64 段，插值点 +22~26%） |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `d_matched_left` | laser_reconstruct | `Execute()` 第1参 | `shared_ptr<GpuMat>` ROI 视图 | 必用 |
| `d_matched_right` | laser_reconstruct | `Execute()` 第2参 | 同上 | 必用 |
| `d_matched_line_ids` | laser_reconstruct | `Execute()` 第3参 | 同上 | 必用 |
| `d_matched_track_ids` | 后处理（点云融合等） | — | 同上 | 可选 |
| `success` / `message` | Pipeline 调度 | — | 值 | 必用 |
| `trackTotal` / `trackDropped` / `excludedLeftCount` | 诊断/监控 | — | 值 | 可选 |

**上游段 ID 开关速查**（epipolar_interp_dual 参数，源码位于 `module2_laser/operators/epipolar_interp_dual/`，定义在 `epipolar_interp_dual_cuda.h`）：

| 参数 | 类型 | 默认 | 建议 | 说明 |
|------|------|------|------|------|
| `scan_link_dx` | float | 3.0f | >0（保持默认） | 行间簇链接横向阈值；**=0 时段退化为簇 ID（断口不合并），本算子精度承诺（wrong=0）不适用** |
| `scan_link_gap_rows` | int | 10 | 保持默认 | 断口容忍行数 |

**行号语义承接**：上游 epipolar_interp_dual 按 `epipolar_row_step`（0.7）栅格插值，本算子行桶化与查表行量化必须同参。

## E. 架构

```
laser_match_scan_v3/
├── laser_match_scan_v3_cuda.h          # 公开头文件（纯 C++）
├── laser_match_scan_v3_cuda_pimpl.h    # CUDA 桥接头
├── laser_match_scan_v3_cuda.cpp        # 桥接实现
├── laser_match_scan_v3_cuda_impl.cu    # CUDA 实现（V3 kernel 组）
└── tests/                               # （算子内单测预留，尚未编写；链路级验证见 I 段）
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserMatchScanCudaV3` |
| 核心方法 | `Execute(d_left_points, d_left_line_ids, d_right_points, d_right_line_ids, stream)` |
| 参数结构体 | `LaserMatchScanParamsV3` |
| 结果结构体 | `LaserMatchScanResultV3` |
| 日志标签 | `"07-LaserMatchScanCudaV3"` |
| 关键常量 | `V3_MAX_CARDS=32`（每点卡上限）/ `V3_MAX_LIDS=32`（lid 桶 stride）|

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | `opencv_core`、`opencv_cudaarithm` |
| CUDA Toolkit | 12.6 | sm_75/86/87；CUB（Toolkit 内置） |
| nlohmann_json | ≥ 3.11 | 参数序列化（`toJson()`/`fromJson()`，key 同 F 段参数名） |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试（FetchContent）——算子内 tests/ 预留尚未编写；链路级 gtest（test_pose06_lid_vote / test_pose06_chain_v3_viz）经模块顶层 tests/ 覆盖 |
| C++ 标准 | C++17 | 禁止 C++20 |

**编译特性**：`BUILD_CUDA` 条件编译宏；MSVC `/bigobj`；GLOB 自动注册（`M2_OP_CPP/M2_OP_CU` 通配 operators/*.cpp/*.cu）。

## F. 参数

**LaserMatchScanParamsV3**（代码内直接设置，或 `LaserMatchScanParamsV3::fromJson(j)` 从 JSON 构造，key 名同参数名，无预设配置）：

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `epipolar_row_step` | float | 0.5f | > 0 | 极线行距（**必须与前端建表侧一致**，本工程用 0.7） |
| `match_threshold` | float | 1.0f | > 0 | 验卡右点搜索窗（±px；本工程用 **2.0**） |
| `vL_tolerance` | float | 0.01f | ≥ 0 | 查表 uL 容差（本工程用 **0.7**，= 表网格步距） |
| `max_right_per_row` | int | 1024 | > 0 | 语义保留，v3 不消费（右侧 CSR 无上限） |
| `deviceId` | int | 0 | ≥ 0 | CUDA 设备号 |

**与 v1/v2 的参数语义差异**：

| 参数 | v1/v2 语义 | v3 语义 |
|------|-----------|---------|
| `match_threshold` | 匹配唯一性判定窗（totalHits==1 才配） | **验卡右点搜索窗**（卡片 HIT 判定） |
| `vL_tolerance` | 查表条目收集容差 | 发卡容差（就近单卡：每 lid 取最近条目） |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | 基线 A：wrong=0（零错配）、correct=14004；基线 B：金标准正式断言零错配保持（真值已知点 19772/19772） |
| 单帧性能 | 基线 A：~4.8ms（左 31883 点、右 31247 点、142 段）；基线 B：~2.6-3.2ms（左 40217 点、右 38051 点、64 段）——GPU events 实测 |
| 显存占用 | ~40 MB（左右点集 + 卡表 + 段管道 + 紧凑配对） |
| 数据规模 | 每点最多 32 卡（V3_MAX_CARDS）→ 单 lid 单卡制下最多 25 卡（25 lid） |
| 段数规模 | segCount = 段 ID 去重数（基线 A 142 / 基线 B 64）；票直方图 segCount×32 int |
| 行数上限 | 8192（`MAX_EPIPOLAR_ROWS`，同 v2） |
| 线程安全 | **非线程安全**；Debug 模式 `atomic<bool>` 检测并发调用 |
| 实例隔离 | 每实例独占 GPU 资源；多实例各自独立 |

**判读注意（基线 B）**：段合并变长后，段内部分点的真值 lid 与 winner lid 不一致（诊断口径 wrong=8704）；lid 仅用于线号标注/着色与 track 输出，**不参与重建几何**（重建仅用点对 + Q）。金标准正式断言按真值已知点计数仍零错配通过。

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 默认值（构造时初始化） | 所有路径均保持 Normal——**本算子当前不产出 Degraded/Warning**（异常经 `success=false` + `message` 文字报告，不设 flag） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数非法 | `validate()` 抛 `std::invalid_argument` |
| 温度未设 | `result.success = false` + "Temperature not set" |
| 输入为空 | `result.success = true` + "No points to match" |
| CUDA kernel 失败 | `result.success = false` + kernel 名 + cudaGetErrorString |
| 紧凑化缓冲不足 | `result.success = false` + "select temp alloc failed" |

**段诊断字段**（`LaserMatchScanResultV3` 独有）：

| 字段 | 含义 | 排查指引（pose_06 基线） |
|------|------|----------------------|
| `trackTotal` | 帧内总段数 | 基线 A 142 / 基线 B 64（单侧投影兜底致段合并）；异常升高 → 查前端 scan_link_dx 是否=0（簇模式）或断口参数过紧；异常降低 → 查上游是否误合并 |
| `trackDropped` | 冲突双弃段数 | 基线 A 30/142 / 基线 B 11/64；占比突增 → 查同行双段同号（激光遮挡/断裂/伪影） |
| `excludedLeftCount` | 无卡左点数（**查到表但所有卡片 HIT 失败**的点；不含行桶无条目[status=-1]与段被弃的点） | 基线 A 7107/31883（22%）；基线 B 未复测；占比突增 → 查 match_window 是否过窄或右图漏检加剧 |

## H. 风险

| 严重程度 | 风险描述 | 影响 | 对策 |
|:--------:|---------|------|------|
| 🔴 高 | **段 ID 语义依赖前端**：若前端 `epipolar_interp_dual.scan_link_dx=0`，输出为簇 ID（断口不合并），段更碎 → 段票基数下降、冲突检测灵敏度变化 | wrong=0 精度承诺不适用 | 前端保持 scan_link_dx>0（默认 3.0） |
| 🟡 中 | 单卡制下每点每 lid 仅取最近条目：若表 sweep 有局部陡峭区（uL 步距不均匀），最近条目可能非最优 | hitRx 偏差 <1px，实测不影响 correct | 接受（精度余量内） |
| 🟡 中 | **上游单侧投影兜底改变段粒度**（2026-09-01 起）：行边缘簇不再丢弃 → 同线段合并变长（142→64 段），长段内混入真值 lid 不同的点（诊断 wrong 计数非零） | lid 归属标注混杂；几何不受影响（lid 不参与重建坐标） | 金标准正式断言仍零错配；如需细粒度段可收紧上游 scan_x_gap/scan_link_dx 或回退该兜底分支 |
| 🟡 中 | 冲突双弃为保守策略：同一行两段持同号即双方全弃 → **正确段也被丢弃**（CPU 模拟口径弃 33 段/correct=13602；**基线 A 算子实测弃 30 段/correct=14004**，因算子无最小段门槛多救回小段） | 换取 wrong=0 的确定性 | 接受；如需找回对家可扩展视差带重选（见 I 段后续） |
| 🟡 中 | host 段表构建在 Step 6 有一次 D2H 同步（~50µs），非纯 GPU 流水线 | 单帧预算内可忽略 | 后续可 GPU 化（排序 unique kernel） |
| 🟢 低 | 段 ID 帧间不稳定：每帧独立编号，不做时序连续 | 后续点云融合不受影响（track_id 不跨帧引用） | 如需时序连续可加质心 IoU 匹配稳定化 |
| 🟢 低 | `V3_MAX_CARDS=32` 截断：>25 条激光线时截断溢出卡 | 本工程 25 线（每 lid 单卡），余量 **7** | 线数扩展时同步扩 V3_MAX_CARDS |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ✅ 新开发（已完成） |
| **现有模块** | 无前代代码可复用（v3 kernel 组全新；表处理/排序三件从 v1/v2 同源拷贝） |
| **复用方式** | N/A（新算子） |
| **金标准** | 基线 A：matched=14004 / correct=14004 / **wrong=0** / lost=5768；基线 B（上游单侧投影兜底后）：matched=24866 / lost=3610 / tracks=64 / dropped=11，正式断言零错配保持（19772/19772） |
| **对拍** | `module2_laser/tests/test_pose06_lid_vote.cpp` 内置 V3 端到端验证段（自动评分） |
| **验证工具** | `module2_laser/tools/fcchain_v3.cpp`（全链 mask→CCL→steger→undistort→interp_dual→match_v3→recon，含耗时+图片+PLY）；`module2_laser/tests/test_pose06_chain_v3_viz.cpp`（L6 链路逐步骤可视化＋宽松门禁 matched/cloudPts≥8000，产物落 data_out/pose06_chainv3/） |
| **版本选型** | **默认推荐 v3**（零错配 + 段级输出）；v1/v2 参考卡见 `docs/operator_docs/laser_match_scan/`——v2 为逐点唯一命中制（13244 匹配/2564 错），无段概念，仅在需要与旧产线对拍时使用 |
| **后续** | 独立单测（tests/）、Warmup 契约完善、OperatorInfo 版本文案、kBandReelect 策略扩展（找回双弃对家） |

---

## 附：最小调用样例

```cpp
#include "laser_match_scan_v3_cuda.h"
using namespace calib;

// 1. 构造（参数可从 JSON 构造：LaserMatchScanParamsV3::fromJson(j["matcher"])）
LaserMatchScanParamsV3 p;
p.epipolar_row_step = 0.7f;    // 必须与前端 interp_dual / 建表侧一致
p.vL_tolerance      = 0.7f;    // 发卡容差 = 表网格步距
p.match_threshold   = 2.0f;    // 验卡右点搜索窗
LaserMatchScanCudaV3 matcher(p);

// 2. 注入表 + 温度（顺序不可反；未设温度 Execute 返回 success=false "Temperature not set"）
matcher.SetTempTable(tempTable);              // shared_ptr<const LaserPlaneMapTempTable>
matcher.SetCurrentTemperature(referenceTemp); // double

// 3. 执行（e1/e2 为 epipolar_interp_dual 左右输出；d_interp_line_ids 携带段 ID）
auto mr = matcher.Execute(*e1.d_interpPoints, *e1.d_interp_line_ids,
                          *e2.d_interpPoints, *e2.d_interp_line_ids, stream);
stream.waitForCompletion();
if (!mr.success || mr.matchedCount == 0) { /* 查 mr.message */ }

// 4. 消费（ROI 视图，有效期至下一次 Execute/Destroy/SetParams）
// mr.d_matched_left / d_matched_right / d_matched_line_ids / d_matched_track_ids
// 诊断: mr.trackTotal(基线A 142/基线B 64) / mr.trackDropped(30/11) / mr.excludedLeftCount(基线A 7107)
```

> **参考卡结束**
