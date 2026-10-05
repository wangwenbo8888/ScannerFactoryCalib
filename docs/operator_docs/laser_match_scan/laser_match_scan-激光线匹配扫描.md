# 激光线匹配扫描（温度补偿查表匹配）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为主工程 09 现版（factory_calib 无此算子副本，扫描期消费 factory_calib 产出的温度表）
> 本卡当前重点：记录 0.5/0.7 步距分叉待办（见 F 节 + H 节 + I 节续作）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-7`（扫描期调用） |
| 中文名称 | 激光线匹配扫描（温度补偿查表匹配） |
| 英文目录名 | `laser_match_scan` |
| 运行平台 | CUDA（全 GPU 管线，CUB + Thrust） |
| 所属流程 | 扫描流程（epipolar_interp 之后、laser_reconstruct 之前） |
| 精度档次 | `③`（亚像素/浮点类；依赖映射表精度 + match_threshold） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机激光中心点集（极线插值后） | `GpuMat`（CV_32FC2）+ 线 ID（CV_32SC1） |
| **输入②** | 右相机激光中心点集（同上） | `GpuMat`（CV_32FC2）+ 线 ID（CV_32SC1） |
| **输入③** | 温度补偿映射表（factory_calib 生成；现链=4-14 CMTT sidecar 按温选档 blob，旧链=4-13 JSON） | `SetTempTable()` 注入 或 `LoadTempTable(jsonPath)` |
| **输入④** | 当前温度值 | `SetCurrentTemperature(double)` |
| **输出** | 匹配对（L/R 点 + **line_id 为输出**）+ 各点状态 + 计数 | `LaserMatchScanResult` |

## C. 算法

**核心思想**：查表预测每个左点在右相机的期望 u（`uR_expected`），右相机同极线行内阈值搜唯一匹配；匹配同时**产出**标定 line_id。

**核心流程**（8 步 GPU 管线）：

```
Step 1a/1b/1c  排序键 (rowKey<<32|x位模式) → CUB RadixSort → 重排
               行键 = roundf(y / epipolar_row_step)      ← ⚠ 与建表侧步距必须一致（见 H）
Step 2  CSR 行索引（每极线行一个桶）
Step 4  查表预测 kernelLookupTableScan:
        row = roundf(yL / epipolar_row_step) → 行桶内按 xL 二分最近条目
        → uR_expected + line_id（表条目 .z 分量；超容差 NAN + -1）
Step 5  逐行匹配 kernelMatch: 行内共享内存 + 二分 + 唯一性判定 + 占用规则
Step 7/8 提取匹配坐标 + 统计排除
```

**关键函数**：`cub::DeviceRadixSort::SortPairs`、`kernelLookupTableScan`（全局按行搜索）、`kernelMatch`、`uploadMapTable`（表 → 行桶 CSR，建表侧行量化 `round(yL/step)` :383）。

## D. 依赖

```
epipolar_interp(L) ─┐
                     ├→ 本算子 → laser_reconstruct
epipolar_interp(R) ─┘
表源（factory_calib）: 4-14 curve_map_temp_table sidecar 选档 blob（fcscan_e2e/fcchain_v3 经
                       selectCmttTier → CurveMapTable::Load 内存段注入；旧 4-13 plane_map_temp_table
                       JSON → LoadTempTable 已随 4-13 退役，仅 09 侧旧链兼容）
```

共享：`common/calib_types.h` / `calib_logging.h` / `json_utils.h`。

## D2. 衔接

| 方向 | 内容 | 说明 |
|------|------|------|
| 上游→本算子 | 极线插值点集 | **行号语义承接**：上游 4-6 按 0.7 栅格插值，本算子行桶化必须同参 |
| 表注入 | `SetTempTable(shared_ptr<const LaserPlaneMapTempTable>)`（§3.6 首选）/ JSON 重载 | 表内条目 y 已按**建表侧步距**（0.7）分桶 |
| 本算子→下游 | d_matched_left/right/line_ids → laser_reconstruct | 必用 |

## E. 架构

```
laser_match_scan/（modules/09_operatorlib/scanning/laser/）
├── laser_match_scan_cuda.h / _pimpl.h / .cpp
├── laser_match_scan_cuda_impl.cu      # 8 步管线
└── tests/test_laser_match_scan_cuda.cpp
```

核心类 `LaserMatchScanCuda`（pImpl）；`SetTempTable/SetCurrentTemperature/Execute`；日志标签 `"07-LaserMatchScanCuda"`；缓冲 `MAX_EPIPOLAR_ROWS=8192`。

## J. 环境

OpenCV+CUDA 4.13（core/cudaarithm）、CUDA 12.6（sm_75/87）、CUB/Thrust（Toolkit 内置）、nlohmann_json ≥3.11、spdlog、gtest；C++17 禁 C++20；`BUILD_CUDA` 条件编译；MSVC /bigobj。

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| match_threshold | float | 1.0 | >0 | u 方向匹配阈值（px） |
| **epipolar_row_step** | float | **0.5** | >0 | ⚠ **硬编码默认值与建表侧 0.7 分叉**（见 H） |
| max_right_per_row | int | 1024 | >0 | 每行右点共享内存上限 |
| vL_tolerance | float | 0.01 | ≥0 | 查表 u 最近邻容差（命名历史遗留，实为 u 向） |
| deviceId | int | 0 | ≥0 | GPU |

预设：`default`(1.0px) / `strict`(0.5px) / `loose`(2.0px)。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 行桶化 | `round(y/step)` 三处（:383 建表 / :460 排序键 / :586 查表）——**必须与建表侧同 step** |
| 行数上限 | MAX_EPIPOLAR_ROWS=8192（1510px/0.7≈2157 行，充裕）；kernelMatch grid 未 clamp（≥4096px 大图越界，文档已知风险） |
| 线程安全 | 非线程安全；实例隔离（L/R 同实例） |
| 前置条件 | LoadTempTable + SetCurrentTemperature 必须先于 Execute |

## K. 质量

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 管线完成（匹配数 0 也算 Normal）；质量看 matchedCount/excluded 计数 |

错误模式：参数非法抛 invalid_argument；表未载/温未设 → success=false 或抛；类型/数量不符 → success=false；GPU 错误 → success=false+message。

## H. 风险

| 严重程度 | 风险描述 | 影响 | 对策 |
|:--------:|---------|------|------|
| 🔴 高 | **【本次记录的待办】`epipolar_row_step` 默认 0.5 与建表侧 0.7 分叉** | 行号错位（y=1.0 → 建表行 1 / 查表行 2）→ 查错行、匹配失败/错配；当前 factory_calib 产出的表按 0.7 建桶，直接消费必错 | **待办（暂不改代码）**：F4 修复——从标定 JSON `pjc.epipolarRowStep`（factory_calib 已落盘 0.7）继承，删除默认值分叉；修复前禁止用默认参数消费 0.7 表 | 
| 🟡 中 | kernelMatch grid 未 clamp 到 8192 | ≥4096px 大图越界读 | 扩 CSR 或 clamp |
| 🟡 中 | 负坐标 clamp 到 0 碰撞 | 坐标系偏移场景排序退化 | 上游保证非负 |
| 🟡 中 | 每行单线程串行匹配 | 密集行瓶颈 | 已知性能项 |
| 🟡 中 | vL_tolerance 命名误导（实为 u 向） | 维护易错 | F4 时一并改名 u_lookup_tolerance |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ⚠ 可用但受步距分叉制约——**消费 factory_calib 0.7 表前必须先做 F4** |
| **现有模块** | factory 副本：`module2_laser/operators/retired/laser_match_scan/`（2026-09-04 按人工指令归档，**fcscan_e2e 仍在用**——retired/ 仍编译、include 自动解析）；09 侧：`modules/09_operatorlib/scanning/laser/laser_match_scan/` |
| **续作（待办）** | ① **F4 步距统一**：`epipolar_row_step` 从标定 JSON 继承（0.7 单点下发），同步改名 vL_tolerance；② F3 曲线带建表落地后回归合成链路（line_id 正确率 100% 验收）；③ 09 侧与 factory_calib 其余分歧同步一并裁决 |

---

> **参考卡结束**
