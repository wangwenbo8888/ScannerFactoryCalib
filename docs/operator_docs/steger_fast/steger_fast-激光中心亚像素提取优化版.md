# 激光中心亚像素提取·优化版（steger_fast）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为 `module2_laser/operators/steger_fast/`——原版 `steger/`（4-4）的**性能优化分叉副本**，同契约同精度，导数与点坐标与原版**逐位一致**。优化过程与实测见 §L。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-4b`（4-4 性能优化分叉；原版 4-4 保留为基准，未删除未改动） |
| 中文名称 | 激光中心亚像素提取（优化版） |
| 英文目录名 | `steger_fast`（文件名 `steger_fast.h/.cpp/_pimpl.h/_impl.cu`，无 `_cuda` 后缀） |
| 运行平台 | CUDA（自写 kernel，无 thrust/无 OpenCV CUDA 算法依赖） |
| 所属流程 | 激光标定链 4-4 候选替代；**尚未接入** laser_calib_cli / fcstepdump / fcscan_e2e（链路仍调原版） |
| 精度档次 | `③`（亚像素/浮点类，理论 0.01~0.1px；同原版） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 灰度图（2048×1536） | `const cv::cuda::GpuMat&`（CV_8UC1） |
| **输入②** | 编号掩膜（ByLabel）/ 二值掩膜（Flat） | `const cv::cuda::GpuMat&`（CV_32SC1 / CV_8UC1） |
| **输出①** | 分组后的亚像素中心点（CPU） | `std::map<int, std::vector<cv::Point2f>> centerPoints` |
| **输出②** | 中心点 GPU 数组（按 label 分组连续存放） | `shared_ptr<GpuMat>`（1×N，CV_32FC2） |
| **输出③** | 每点线号 GPU 数组（与输出②对齐） | `shared_ptr<GpuMat>`（1×N，CV_32SC1） |

## C. 算法

```
Step 1+2a  RowFan3Kernel      uchar 直读，单次算出行级 g/gx/gxx 三个中间图
                              （原版 5 次行卷积：g 与 g' 各被重复计算一次；本版读 1 遍写 3 遍）
Step 2b    ColFan5Kernel      单次算出 Ix/Iy/Ixx/Iyy/Ixy 五个导数（原版 5 次列卷积）
                              映射：Ix←(gx,g) Iy←(g,gx) Ixx←(gxx,g) Iyy←(g,gxx) Ixy←(gx,gx)
                              renormalize（边界 w_sum 除法）语义与原版逐核一致
Step 3+4   HessianEigenAndTaylorKernel<MaskIs8U>
                              与原版逐位相同（Hessian 特征方向 + |t|<0.5 Taylor 修正 + 阈值过滤）；
                              Flat 模式模板化 8U 掩膜（原版需先转 CV_32SC1 再进 kernel）
Step 5     计数直写分组       LabelHistKernel（label 直方图）→ CPU 前缀和 → LabelScatterKernel
                              （base[label]+atomicAdd(cursor[label]) 直写，按 label 分组连续输出）
Step 6     单次 D2H + D2D     h_pts 单次下载建 CPU map；d_coords/d_ids D2D 拷入 result 自持有 GpuMat
```

**数值等价性**：导数累加保持原版 k 升序与 renormalize 语义，uchar→float 转换无损 → 导数与点坐标**逐位一致**；label 内点序原版（atomicAdd 时序）本就非确定，scatter 后顺序不同但坐标值不变。

**关键实现**：全部自写 kernel；无 `cv::cuda::filter`、无 thrust；kernel 系数（≤31 tap）常驻 device。

## D. 依赖

```
4-1 mask_extract (d_grayImage) ──┐
4-3 laser_label (labeledMask) ───┴→ steger_fast → 4-5 undistort_cuda
                                               (d_centerPoints + d_line_ids)
```

**共享/复用**：`StegerParams`/`StegerResult`/`GroupMode` 三类型唯一驻点为 `common/steger_types.h`（2026-09-04 自原版头抽出，定义逐字未动）——本头**不再 include 原版头**，steger_fast 可脱离 retired/steger 独立编译；原版头同样改引公共头，对拍测试类型同源无重定义。与主工程 `modules/09_operatorlib/core/laser/steger/` 的逐字拷贝边界新增两处分叉（抽公共类型头＋本算子），同步时需对齐（同步债务）。

## D2. 衔接

| 方向 | 内容 | 说明 |
|------|------|------|
| 上游→本算子 | GpuMat 灰度＋掩膜（ByLabel CV_32SC1 / Flat CV_8UC1） | 与原版同签名（4 个 Execute 重载） |
| 本算子→下游 | `d_centerPoints` + `d_line_ids` → undistort_cuda `Execute(d_points, d_line_ids, stream)` | 必用（GPU 数组按 label 分组连续，与原版排序后布局等价） |
| | `centerPoints`（CPU map） | 测试/调试消费；CLI 主链用 GPU 侧 |

**生命周期**：输出 GpuMat 为 result 自持有（每帧新分配 ~0.4/~0.2MB），**跨帧持有安全**（无 ping-pong 复用别名问题）；点序与原版一样非确定，下游不得依赖 label 内顺序。

## E. 架构

```
steger_fast/
├── steger_fast.h            # 公开头（复用原版 Params/Result/GroupMode）
├── steger_fast_pimpl.h      # Impl 结构（FastPoint、缓冲/事件/pinned 成员）
├── steger_fast.cpp          # 桥接实现（输入校验与错误消息同原版）
├── steger_fast_impl.cu      # CUDA 实现（5 kernel + Impl 方法）
└── tests/
    └── test_steger_fast.cpp # 9 cases：等价性×6 + 基准×2 + 逐帧可视化×1
```

| 项目 | 名称 |
|------|------|
| 核心类 | `calib::StegerExtractorFast`（pImpl） |
| 核心方法 | `Execute(gray, labeledMask, stream)` / `Execute(gray, mask, stream, GroupMode)`（另有无 stream 重载） |
| 其他 | `Warmup(rows, cols)`／`Warmup(WarmupConfig)`／`SetParams`／`GetParams`／`Destroy` |
| 参数/结果 | 复用 `StegerParams` / `StegerResult` |
| 日志标签 | `"10-StegerExtractorFast"`（BENCH 行格式 `[BENCH_FAST] … Row3/Col5/Hessian/Group/Out`） |

## J. 环境

OpenCV 4.13（core / core-cuda 头，无算法模块依赖）；CUDA 12.x（sm_75/86/87，实测 v12.4 编译）；spdlog；nlohmann_json（经原版头传递）；gtest；C++17；`BUILD_CUDA`；MSVC `/bigobj`。**无需 thrust**（原版用，本版已去除）。

## F. 参数

复用 `StegerParams`（字段同原版，无新增）：

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| sigma | float | 1.5 | [0.5,10] | 高斯 σ（left_skew 链路实测 1.5，kernelSize 自动 11） |
| kernelSize | int | 0 | 0/3/5/7/9 | 0=auto（ceil(3σ)×2+1，钳位 [3,31]） |
| lowThreshold | float | 2.0 | ≥0 | 特征值下限 |
| highThreshold | float | 0.0 | ≥0 | 0=关 |
| maxLabels | int | 256 | [1,4096] | 语义校验用；内部分组 bin 固定 4096 |
| deviceId | int | 0 | ≥0 | 设备选择 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 单帧性能 | 2048×1536 真实样本：**中位 1.5~2.2ms**（22 帧均值 2.20ms，范围 1.98-2.78；同机同进程原版均值 5.58ms，**2.3~2.9×**）；小图（200×100）~0.36ms |
| 分步中位 | Row3=0.40 / Col5=0.73 / Hessian=0.11 / Group=0.49 / Out=0.27ms（70 帧实测） |
| 显存占用 | ~214MB/实例（3 行级 temp 37.7 + 5 导数 62.9 + points/sorted 75.5 + coords/ids 37.8MB）；较原版 ~126MB **多 ~1.7×**；L/R 双实例 ~430MB；另 pinned 主存 37.7MB/实例 |
| 数据规模 | 图像 2048×1536，点容量 rows×cols（实际 4~5 万点/帧） |
| 线程安全 | 非线程安全；Debug 模式 `atomic<bool>` 并发检测（同原版） |
| 实例隔离 | 左右相机各持独立实例 |
| 同步点 | 每 Execute 2 次中途 stream 同步（count 回读、直方图回读）＋末尾事件同步（同原版）——流水线重叠时需知悉 |

## K. 质量

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 提取成功 | 且无退化条件 |
| Degraded | 部分线点数过少 | 任一 label 点数 < 10 |
| Warning | 0 点或噪声过多 | point_count==0（success=true，"No subpixel points extracted"）或 >50% 像素成点 |

错误处理同原版：参数非法抛 `std::invalid_argument`；空图/类型错/尺寸不匹配 `success=false + message`（消息文本与原版逐字一致）；CUDA 错误 `success=false`。

## H. 风险

| 严重程度 | 描述 | 影响 |
|:--------:|------|------|
| 🟡 中 | **未接入链路**：laser_calib_cli（4-4 处）/ fcstepdump / fcscan_e2e 仍实例化原版 `StegerExtractorCUDA` | 切换属链路变更，需人工指令＋全量回归后方可替换 |
| 🟡 中 | label ≥ 4096 截断进 4095 桶（原版 std::map 无上限） | 实际线号 ≤25，不可达；防御性截断，行为差异已记录 |
| 🟡 中 | 与主工程 09 的逐字拷贝边界新增分叉 | 同步策略：原版仍逐字对齐；fast 为主工程侧独立优化候选 |
| 🟢 低 | 显存较原版 +~88MB/实例（分组缓冲 coords/ids/sorted） | 16GB 卡无压力，多实例部署需计数 |
| 🟢 低 | 输出 GpuMat 每帧 2 次小 cudaMalloc（自持有，保生命周期安全） | ~10µs 级；未池化（安全优先于极致） |
| 🟢 低 | Group 步含 2 次中途强制同步（count/hist 回读） | 单流离线无影响；多流重叠收益受限 |

## I. 状态

| 项目 | 说明 |
|------|------|
| 判定 | ✅ **4-4 产线现役（2026-09-04 切换）**：与原版逐位等价（全 22 帧全精度对拍 n=0；容量越界修复后两链全流程逐字节一致——accum/rms/verdict/sha/projectorT 全同）；切换后验证 PASS/rms 1.1053/与等价基线逐字节一致；全量 35/35 绿（基准用例需冷机，热漂移 3.0ms 阈值附近） |
| 现有模块 | factory_calib `module2_laser/operators/steger_fast/`（类型源 `common/steger_types.h`，不依赖原版目录——解耦保留有效） |
| 回归 | 本算子 9 cases 绿（等价性 6 + 基准 2 + 可视化 1）；2026-09-04 全量 35/35 绿；两次切换失败证据 `data_out/e2e_fast/`（sf1/sf2 双跑＝确定性劣化；revert_check 回滚复现基线） |
| 待办 | 若三评：需先在**全部 22 帧**逐位对拍定位系统差帧（现仅 3 帧对拍过）；dual·Labeled 同理；根因或与 scan 链语义的窗口/聚类默认值在标定形态下的行为差有关 |

## L. 优化记录（2026-08-28）

**动机**：4-4 原版实测 2048×1536 中位 5.25ms（fcstepdump 18 帧，CUDA 事件分步：S2 高斯卷积 3.01ms/57% 为热点，带宽效率仅 ~18%；S6 D2H→H2D 往返 1.33ms；S5 thrust stable_sort 0.54ms；另有每帧 6+ 次 cudaMalloc/Free 与默认流操作）。目标：**Total ≤3ms**（下达指标 1-3ms）。

**原版 → fast 改动清单**：

| # | 原版 | fast | 性质 |
|---|------|------|------|
| 1 | 5 次行卷积（g/g' 各重复 2 次）+ 独立 8U→32F kernel | RowFan3：uchar 直读，单 kernel 出 3 个行级中间图 | 位级等价 |
| 2 | 5 次列卷积 | ColFan5：单 kernel 出 5 导数（列方向 k 升序累加保持） | 位级等价 |
| 3 | thrust stable_sort（12B 结构体比较排序） | 直方图＋CPU 前缀和＋计数直写 scatter（O(N)） | 值等价（label 内顺序不同，本就非确定） |
| 4 | D2H→cv::Mat→H2D 往返组装 GPU 输出 | 单次 D2H（CPU map）+ D2D（GPU 输出），同源不复返 | 值等价 |
| 5 | 每帧 cudaMalloc/Free ×6+（thrust 临时）、事件创建销毁 ×7、默认流操作 ×5 | Warmup/按需预分配全部复用、事件常驻、全部操作挂调用方 stream | 无风险 |
| 6 | Flat 模式先 binaryToIntLabel 转换再进 kernel | Hessian kernel 模板化直读 8U 掩膜（省 12.6MB 写读） | 位级等价 |

**实测**（Quadro RTX 5000，Release，同进程同输入对照）：

| 场景（2048×1536） | 原版 | fast | 加速 |
|------|------|------|------|
| 合成 25 线（ByLabel）中位 | 5.53ms | **2.10ms** | 2.64× |
| 真实样本（Flat）中位 | 3.84ms | **1.52ms** | 2.53× |
| 真实 22 帧逐帧均值 | 5.58ms | **2.20ms**（全部 1.98-2.78） | 2.54× |

**验证**：①6 组等价性测试（ByLabel 单/多线、Flat、全背景、无 Warmup、SetParams 后）点坐标逐位一致；②22 帧逐帧内联等价断言全过；③模块2 ctest 21/21 绿。

**过程 bug**：初版 scatter 的 `d_cursor_` 误用 label 计数直方图作初值（slot 从 count 起跳，点全部写偏、输出区为 0），被等价性测试立即捕获——改 `cudaMemsetAsync` 清零后全绿。**教训：分组直写的 cursor 初值必须是 0，base 已由前缀和提供段起点。**

**工具与产物**：`test_steger_fast.exe`（--gtest_filter 可单跑；`VisualizeRealFrames` 输出 22 帧 orig/fast 叠加图＋timing.csv 至 `%TEMP%\opencode\steger_fast_vis\`）。

**残余空间**（未做，收益递减）：Col5 共享内存 tile 化（0.73→~0.4ms 预期）；label 直方图并入 Hessian kernel（省一次同步 ~0.1ms）；全融合单 kernel（导数不落盘，理论 ~0.5ms，复杂度高）。当前 1-3ms 目标已达成。
