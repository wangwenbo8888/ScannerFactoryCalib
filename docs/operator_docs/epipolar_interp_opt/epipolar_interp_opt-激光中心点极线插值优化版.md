# 激光中心点极线插值（优化版）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为 factory_calib 模块2 优化副本（自 `epipolar_interp` v2 极线驱动重采样版复制优化，原算子零改动并存）
> 定位：**性能评估/对照实现**——输出与原版 bit 级一致（left_skew 44 帧-side 全量验证），尚未接入 `laser_calib_cli` 主链；替换决策见 I 节

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-6b`（4-6 性能优化分叉，原版保留为基准；编号后缀惯例同 4b/12b） |
| 中文名称 | 激光中心点极线插值（极线驱动重采样·优化版） |
| 英文目录名 | `epipolar_interp_opt` |
| 运行平台 | CUDA（CUB 管线；插值 kernel 与原版逐字一致） |
| 所属流程 | 模块2 激光标定 4-6 优化对照版（undistort_points 之后、laser_match 之前）；当前仅由 `fcstepdump --interp-bench` 调用 |
| 精度档次 | `③`（亚像素/浮点类；插值 kernel 与原版逐字一致） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 矫正后激光中心点集 | `const cv::cuda::GpuMat&`（CV_32FC2，1×N） |
| **输入②** | 点对应线号 | `const cv::cuda::GpuMat&`（CV_32SC1，1×N，与①同长） |
| **输出①** | 极线栅格插值点（仅结果点，原始点不透传） | `shared_ptr<GpuMat>`（CV_32FC2） |
| **输出②** | 插值点线号 | `shared_ptr<GpuMat>`（CV_32SC1） |
| **输出③** | 插值点计数 | `int interpCount` |

## C. 算法

**核心思想**与原版一致：对每条极线栅格行 `y_k = k·epipolar_row_step`，在上下搜索窗内各取最近同线点，连线与极线交点作为插值点；点恰在极线上时直接采用。**优化只改执行方式，不改算法语义**：

```
Step 1  排序: optBuildSortKeys 打包键 (fid<<32|y位序) → optFillIndices 填充索引
              → CUB DeviceRadixSort::SortPairs（替代原版 thrust::sort_by_key）
              → optGatherSorted 重排
Step 2  GPU 端线号统计（替代原版 host 全量下载+扫描，原版共 ~430KB、2 次同步往返）:
        optBoundaryFlags 边界标记 → CUB DeviceScan::ExclusiveSum 前缀和
        → optScatterLineTable 建升序去重线号表+区间表（含 end 项/num_lines）
        → optInitMeta 初始化归约初值 → optYRange 原子归约 y 极值
        → 唯一一次 12B pinned 回传 {num_lines, umin, umax}
Step 3  插值: optEpipolarInterp —— 与原版 kernelEpipolarInterp 逐字一致
Step 4  压缩: CUB DeviceSelect::Flagged ×2（temp 容量缓存，替代每次 size 查询）
```

**与原版差异一览**：

| 项 | 原版 epipolar_interp | 本优化版 |
|----|---------------------|---------|
| 排序 | thrust::sort_by_key | CUB 基数排序（双缓冲 ping-pong） |
| 线号统计 | 下载全部 fids+全部点（~12B/点，合计 ~430KB）到 host 扫描，线表 GpuMat 每次新建+回传 | GPU 端三 kernel 建表，仅 12B pinned 回传，线表常驻显存 |
| 中途同步 | 2 次（两次 D2H 后） | 1 次（12B 回传后） |
| CUB temp / 缓冲 | 每次查询 / create 按帧尺寸 realloc | 容量 grow-only 缓存 |
| 插值 kernel / 输出契约 | — | **逐字一致**（含 clone 与 interpCount 语义） |

**关键函数**：`cub::DeviceRadixSort::SortPairs`、`cub::DeviceScan::ExclusiveSum`、`cub::DeviceSelect::Flagged`、`atomicMin/atomicMax`（y 极值，y≥0 位序单调假设与原版排序键同源）。

## D. 依赖

```
undistort_points_cuda(4-5) ──→ 本算子 ──→ laser_match(4-7)
                                        （当前实际消费方: tools/interp_bench.cu 对照基准）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `EpipolarInterpParams` / `EpipolarInterpResult` | 直接复用原版头文件类型（不重复定义，保证契约同型） |
| 插值 kernel | 与原版逐字一致（optEpipolarInterp 仅改符号名） |
| `common/` | calib_types.h / calib_logging.h / scanner_api.h / version.h / calib_warmup_config.h |

## D2. 衔接

**上游→本算子**：与原版同位替换，签名 `Execute(GpuMat, GpuMat, Stream)` 完全一致。

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `result.d_interpPoints` | laser_match(4-7) | `Execute(GpuMat, GpuMat, GpuMat, GpuMat, Stream)` | `shared_ptr`（解引用） | 必用 |
| `result.d_interp_line_ids` | laser_match(4-7) | 同上 | `shared_ptr`（解引用） | 必用 |
| `result.success` / `interpCount` | Pipeline 调度 / 日志 | — | 值 | 必用 / 可选 |

## E. 架构

```
epipolar_interp_opt/（factory_calib/module2_laser/operators/）
├── epipolar_interp_opt_cuda.h         # 公开头文件（纯 C++；类 EpipolarInterpOptCuda）
├── epipolar_interp_opt_cuda_pimpl.h   # CUDA 桥接头（Impl 结构：双缓冲/temp/容量字段）
├── epipolar_interp_opt_cuda.cpp       # 桥接实现（输入校验同原版）
└── epipolar_interp_opt_cuda_impl.cu   # CUDA 实现（8 个 opt 前缀 kernel + 4 步管线）
```

核心 API：类 `EpipolarInterpOptCuda`（pImpl，禁拷贝）；`Execute(GpuMat, GpuMat[, Stream])` / `Warmup(int)` / `SetParams` / `Destroy`；参数结构体 `EpipolarInterpParams` / 结果结构体 `EpipolarInterpResult`（均复用原版）；日志标签 `"06-EpipolarInterpOptCuda"`。kernel 全部 `opt` 前缀避免与原版符号冲突；CMake 由 `file(GLOB_RECURSE)` 自动编入 `fc2_ops`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | core / cuda 模块 |
| CUDA Toolkit | 12.6 | sm_75/86/87；CUB 随 Toolkit 内置 |
| spdlog | ≥ 1.15 | 日志 |
| nlohmann_json | ≥ 3.11 | 参数序列化（经 Params toJson） |
| GoogleTest | ≥ 1.14 | 间接覆盖（经原版测试） |
| C++ 标准 | C++17 | 禁止 C++20 |

**编译特性**：`BUILD_CUDA` 条件编译；MSVC nvcc `/bigobj`；无 thrust 依赖（原版有）。

## F. 参数

复用 `EpipolarInterpParams`（结构与默认值同原版，无新增参数）：

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| epipolar_row_step | float | 0.7 | >0 | 极线栅格行距（px） |
| window_offset | float | -1.0 | >0 或 -1 | 搜索窗半径，-1=取行距 |
| max_x_diff | float | 1.0 | >0 | 上下选点 X 差上限 |
| deviceId | int | 0 | ≥0 | GPU |
| lineIdCheck | bool | true | — | ⚠ **未生效**（原版 v2 同样未引用，见 H 风险④） |

预设：无（复用原版 EpipolarInterpParams，无预设档）。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | 与原版输出 bit 级一致：44/44 帧-side 逐点差异 0（容差 1e-6，count/fid 全等） |
| 单次性能 | 均值 **0.52ms**（原版 0.94ms）；单帧 0.43-0.71ms（原版 0.72-1.49ms）——left_skew 22 pose×L/R×20 迭代实测 |
| 加速比 | 总体均值 **~1.81×**（单帧波动 1.05-3.43×；另一次独立运行均值比 1.79×，timing.csv 为后次运行产物） |
| 数据规模 | ~3.65 万点 / 25 线 / ~1624 极线行每帧（L ~1672 / R ~1577），2048×1536 图 |
| 显存占用 | ~3 MB/实例（排序双缓冲+线号表+插值/CUB 缓冲 grow-only；点数上限受容量自适应） |
| 同步点 | 每调用 2 次（12B pinned 回传 + count 回传）；原版 Release 3 次（Debug 另有 1 次断言同步） |
| 线程安全 | **非线程安全**（Debug 模式 `atomic<bool>` 检测并发；与原版同） |
| 实例隔离 | L/R 各持独立实例（与原版同惯例） |
| 剩余瓶颈 | 见 H 风险⑥（count 回传同步 + 结果 clone，~0.3ms 余量） |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常处理 | 管线成功（含空输入早退返回空结果） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数非法 | 构造/`SetParams` 抛 `std::invalid_argument`（validate） |
| 无 CUDA 设备 / pinned 分配失败 | 构造抛 `std::runtime_error` |
| 输入为空 / 点数<2 | `success=true` + 空结果（早退） |
| kernel/CUB/temp 失败 | `success=false` + message（逐级日志） |

## H. 风险

| # | 严重程度 | 风险描述 | 影响 |
|:-:|:--------:|---------|------|
| ① | 🟡 中 | **y<0 假设**：排序键与 y 极值原子归约均按 y≥0 位序单调处理（继承原版简化；矫正坐标若越出图像上边界为负则次序错乱） | 与原版同病：异常输入下双双退化，非优化引入 |
| ② | 🟡 中 | **双副本已三体分歧**：本优化版 / factory_calib 原版 v2 / 主工程 modules/09 v1（kernelMarkAndCompute 相邻对法）。替换或同步需人工决策 | 边界管理成本；AGENTS.md「逐字拷贝」约定已破（原版层面先行） |
| ③ | 🟡 中 | 排序平局次序为实现细节：CUB 基数排序与 thrust 排序对 (fid,y) 全同键点的相对次序可能不同（left_skew 实测零差异，理论未证等价） | 极端重复 y 数据下输出点选取可能分歧 |
| ④ | 🟢 低 | `lineIdCheck` 参数声明未生效（原版 v2 同） | 若扫描链路依赖该语义为隐患 |
| ⑤ | 🟢 低 | 结果 clone 走默认流 ×2、count 回传同步在前 | 输出契约层面的性能余量 |
| ⑥ | 🟢 低 | 未接入 `laser_calib_cli`/主链，无独立单测（正确性由 interp_bench 对照验证守护） | 转正前需补测试与接入决策 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 🟢 可直接用（评估态）——同输入输出与原版 bit 级一致，总体均值加速 ~1.8× |
| **现有模块** | `factory_calib/module2_laser/operators/retired/epipolar_interp_opt/`（4 文件，无独立 tests/；2026-09-04 归档——factory 内无 exe 消费，仅 --interp-bench 评估路径） |
| **复用方式** | 自 epipolar_interp v2 复制优化，原版零改动并存；复用原版 Params/Result 类型 |
| **验证方式** | `fcstepdump --interp-bench data_in\left_skew data_out\interp_bench 20`：逐点比对（容差 1e-6）+ 绿/红叠加对比图（`*_6interp_orig_green_vs_opt_red.png`）+ timing.csv |
| **转正待办** | (1) 人工决策替换或与原版并存；(2) 三体分歧同步方案；(3) 补独立单测；(4) 二期：池化输出+延迟计数（预计再降至 ~0.3ms） |

---

> **参考卡结束**
