# 激光连通域分析算子（region_analyze_cuda）

> 依据《算子提炼模板.md》v1.0 十二段落编写，对象为 factory_calib 副本（`module2_laser/operators/ccl/`）。与主工程 09 副本（`modules/09_operatorlib/core/vision/ccl/`）**已分叉**：factory_calib 侧后加 topX 二次筛选，2026-08-28 完成 topX GPU 化性能优化，含 §L 优化记录。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-2`（4-2 前端环节；公开头文件旧编号"标定流程1❺/流程2❻"） |
| 中文名称 | 激光连通域分析 |
| 英文目录名 | `ccl`（源文件名带 `_cuda` 后缀：`region_analyze_cuda.h` 等） |
| 运行平台 | CUDA（全 GPU 管线，仅 topX 阈值决策在 CPU） |
| 所属流程 | 激光标定流水线 4-2；fcscan_e2e 扫描端到端；fcstepdump 工具链 |
| 精度档次 | 按模板语义记 `③`（像素级：标签+包围盒，无亚像素量）。注：公开头文件自declare"档次②（整像素/几何类）"，与模板②（亚像素 ~0.05px）定义冲突，口径待统一 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入** | 清洗后激光二值掩膜（2048×1536，来自 4-1 mask_extract） | `shared_ptr<GpuMat>`（CV_8UC1） |
| **输出①** | 重编号连通域掩膜（1-indexed 稠密编号） | `shared_ptr<GpuMat>`（CV_32SC1） |
| **输出②** | 各连通域包围盒列表（label/x/y/w/h） | `std::vector<ComponentStats>`（CPU） |
| **输出③** | 域计数 + 质量标记 | `int componentCount` + `QualityFlag` |

## C. 算法

```
Step 1  cv::cuda::connectedComponents (BKE, conn=8)        全 GPU，默认流
Step 2  initStatsKernel    清零统计数组（面积×1 + bbox×4 + remap + 计数）
Step 3  computeStatsKernel 逐像素 5 原子累加（面积 + min/max_x/y）
Step 4  buildRemapKernel   面积过滤（minArea/maxArea）+ atomicAdd 稠密重编号
Step 5  relabelKernel      全图按 remap 重写标签 → d_relabeled_
Step 6  compactStatsKernel 有效域 bbox 压缩到紧凑数组
Step 7  cudaDeviceSynchronize + pinned D2H（域计数 + bbox 列表）
Step 7b topX（可选，topXCount>0 且首遍域数超限）:
        collectValidAreasKernel GPU 收集有效域面积（~几十-百个 int，2026-08-28 新增）
        → 小传输 D2H → CPU nth_element 求第 X 大面积阈值
        → 提高下界重跑 Step 4/5/6；面积==阈值并列全保留（结果域数可 ≥ X）
```

**关键函数**：`cv::cuda::connectedComponents`（BKE 算法；**输出标签不保证连续**，见 §H）。

## D. 依赖

```
上游: 4-1 mask_extract (d_cleanedMask) → 本算子 → 下游: 4-3 laser_label (d_labeledMask)
```

**共享/复用关系**：主工程 `modules/09_operatorlib/core/vision/ccl/` 为历史基线副本；**factory_calib 侧已分叉**（新增 topX 及其 GPU 化），主工程副本无 topX 功能。后续同步须整体采用 factory_calib 版。

## D2. 衔接

| 方向 | 传递方式 | 说明 |
|------|---------|------|
| 上游→本算子 | `Execute(shared_ptr<GpuMat>, Stream&)` | 必用显式 Stream 重载；无 Stream 重载仅测试/调试 |
| 本算子→4-3 | `result.d_labeledMask` → laser_label `Execute(GpuMat&, stream)` | 必用；laser_label 按扫描方向切线的几何位置二次重编号（默认=中心列最小 Y；斜线形态 scanDirection=1 为横切中心行），不依赖本算子编号顺序 |
| 本算子→CLI/工具 | `result.components` / `componentCount` | 可选（统计/诊断用） |

## E. 架构

```
ccl/
├── region_analyze_cuda.h            # 公开头文件（纯 C++，不含 CUDA 类型）
├── region_analyze_cuda_pimpl.h      # CUDA 桥接头文件（struct Impl 完整声明）
├── region_analyze_cuda.cpp          # 桥接实现（校验/转发/生命周期）
├── region_analyze_cuda_impl.cu      # CUDA 实现（kernels + Impl::Execute）
└── tests/test_region_analyze_cuda.cpp  # 58 用例
```

| 项目 | 名称 |
|------|------|
| 核心类 | `RegionAnalyzerCUDA`（pImpl → `Impl`） |
| 核心方法 | `Execute(const shared_ptr<GpuMat>&, cv::cuda::Stream&)` |
| 生命周期 | `Warmup(rows, cols)` 预分配（不调则首次 Execute 自预热）；`SetParams()`（不得与 Execute 并发）；`Destroy()` 析构前必须调用（幂等） |
| 参数/结果结构体 | `RegionAnalyzerParams` / `RegionAnalysisResult` |
| kernels | `initStatsKernel` / `computeStatsKernel` / `buildRemapKernel` / `relabelKernel` / `compactStatsKernel` / `collectValidAreasKernel` |
| 日志标签 | `"07-RegionAnalyzerCUDA"` |
| 内置计时 | cudaEvent 7 区间（GPU 步骤分解）+ **WALL 墙钟**（整函数，含 topX/copyTo） |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | `opencv_cudaimgproc`（connectedComponents）、`opencv_core` |
| CUDA Toolkit | 12.6 | sm_75（RTX 5000）/ sm_86 / sm_87 |
| spdlog / nlohmann_json / GoogleTest | FetchContent | 日志 / 参数序列化 / 单测 |
| C++ 标准 | C++17 | — |

编译：`BUILD_CUDA=1` 宏条件编译；CPU-only 模式下 Execute/Warmup/SetParams/GetParams 抛 `runtime_error`（构造仅告警+参数校验，Destroy 为 no-op）。

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `minArea` | int | 100 | ≥ 0 | 最小域面积（产线配 0 全保留） |
| `maxArea` | int | 100000 | > minArea | 最大域面积 |
| `topXCount` | int | 0 | ≥ 0 | 只保留面积最大前 X 域（0=不启用；产线/工具配 27） |
| `deviceId` | int | 0 | < 设备数 | GPU 设备号 |

产线配置：`t=50/e=1/d=19/post13 + ccl minArea=0/topX27`（laser_calib_cli）。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 单帧性能 | 优化后墙钟 p50 **2.9ms**（2048×1536，RTX 5000，含 topX）；GPU 步骤 ~1.4ms；帧级 L+R ≈ 5.8ms |
| 显存占用 | ~75MB/实例（统计数组 6×12.6MB，因 BKE 稀疏**无法缩小**，见 §H） |
| 数据规模 | 图像 2048×1536；原始域实测 29~83 个/帧 |
| 线程安全 | **非线程安全**；Debug 模式 `atomic<bool>` 检测并发调用 |
| 实例隔离 | L/R 相机各持独立实例（laser_calib_cli 串行调用） |
| 标签空间 | `scan_range = 像素总数 + 1`（BKE 稀疏，见 §H） |

## K. 质量

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 过滤后域数 1~200 |
| `Warning` | 警告（仍 success=true） | 过滤后域数 = 0（空掩膜，或全部域被 minArea/maxArea 过滤掉） |
| `Degraded` | 降级（仍 success=true） | 域数 > 200 |

| 错误类型 | 处理方式 |
|---------|---------|
| 参数非法（minArea<0 等） | 构造/SetParams 抛 `std::invalid_argument` |
| 输入空 / 非 CV_8UC1 / 空指针 | `success=false` + message |
| CUDA OOM / 异常 | catch 后 `success=false` |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🔴 高 | **BKE 输出标签极度稀疏**：真实掩膜 max_label 实测 80 万~313 万（≈像素总数），即使实际只有几十个域。统计数组必须按像素总数分配；任何"用 max_label 缩小扫描范围"的优化**已被实测证伪**（2026-08-28 尝试后回退，源码注释留档） | 显存 ~75MB/实例下不来；initStats 等步骤全量扫描不可避免 |
| 🟡 中 | kernels 与 CCL 均走**默认流**，忽略调用方 Stream（OpenCV CCL API 无 stream 参数）；每帧 `cudaDeviceSynchronize` 一次、topX 触发时再追加一次（重跑后） | L/R 无法并发；无法与流水线重叠 |
| 🟡 中 | `buildRemapKernel` atomicAdd 编号顺序**逐次运行不确定**（纯外观抖动，下游按几何重排已消除） | 回归验证**不可用标签 id 序列**做基准（用 3label/4steger 输出或 bbox 集合） |
| 🟡 中 | 域数 > COMPACT_MAX(65536) 时 `components` 尾部零填充，且超出容量的域在重编号掩膜中被置 0（buildRemap 的 compact_max 守卫；沿袭旧版行为） | 病态图下组件列表与掩膜均不完整（componentCount 计数仍正确） |
| 🟢 低 | 计时口径：GPU 步骤事件区间**不含** topX/copyTo，看 TOTAL 会低估；以 WALL 行为准 | 可观测性（2026-08-28 已加 WALL 修正） |
| 🟢 低 | 帧间空闲（如写 PNG）导致 GPU 降频，尾帧全步骤 ×3~5 膨胀 | 环境效应，连续运行无此问题 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ✅ 可直接用（factory_calib 优化版，commit b620d91） |
| **与 09 关系** | 已分叉：主工程副本为旧版（无 topX）。扫描链（fcscan_e2e topXCount=27）需要 top-27 过滤，未来同步应整体采用 factory_calib 版并跑主工程 95/95 测试 |
| **后续优化空间** | 稠密重编号可再省 ~0.5ms（消除 initStats/buildRemap/compact/collect 四步的全量扫描 ≈0.45ms 实测），但 BKE 稀疏使其实现复杂（并发哈希/CAS 认领），收益风险比差，暂缓 |

## L. 优化记录（2026-08-28）

**背景**：三轮实测审计发现算子自带计时（cudaEvent 7 区间）只报告 ~1.3ms，但日志时间戳反推背靠背墙钟 **12.7ms/次**——topX 二次筛选路径位于计时事件区**之外**，隐藏 ~10ms：12.6MB 面积表 **pageable** D2H（std::vector 非 pinned）+ CPU 扫 315 万标签 + vector 零初始化 + 三 kernel 全量重跑 + 一次额外全设备同步（重跑后）。产线配置（topXCount=27，原始域 29~83）下 **42/44 帧触发**。另发现三个生产入口（laser_calib_cli / fcstepdump / fcscan_e2e）均未调 Warmup，每帧进重分配分支。

| # | 改动 | 性质 | 实测收益 |
|---|------|------|---------|
| 1 | `collectValidAreasKernel`：有效域面积 GPU 压缩到紧凑缓冲（实测 ~30-100 个 int）小传输 D2H，替代整表 pageable D2H + CPU 全量扫描 | **位级等效**（面积多重集不变 → nth_element 阈值不变） | 消灭隐藏 ~10ms；墙钟 p50 13→2.9ms（**4.5×**） |
| 2 | Execute 自预热：首次分配后置位 `warmed_up_`，同尺寸不再进重分配分支 | 输出等价（日志频次/分支行为变化，功能输出不变） | 消除每帧 WARN + 分支开销 |
| 3 | 计时新增 WALL 墙钟口径（含 topX/copyTo） | 可观测性 | 修正"日志只报约一半"的历史口径问题 |
| 4 | ~~max_label 自适应上界~~（尝试后回退：BKE 标签稀疏证伪，max_label≈像素总数） | 无效 | 源码注释留档防重蹈 |

**优化前后对比**（44 帧真实数据，RTX 5000）：

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 单次 Execute 墙钟 p50 / avg | 13.0 / 12.68 ms | **2.89 / 4.19** ms |
| GPU 七步耗时 | 1.32 ms | 1.37 ms（按设计不变） |
| 帧级 L+R | ~26 ms | ~5.8 ms |
| 显存 | ~75MB/实例 | 不变（BKE 稀疏限制） |

**验证**：改动 1 为位级等效——44 帧真实数据下游 `3label`/`4steger`/`1mask` 输出与旧版**逐位一致**（3 类输出 × 44 帧 = 132 个文件 MD5 逐一对比）；CCL 单测 58/58 绿（新增 `ManyComponentsStress` 116964 域压力 + `TopXKeepsTiedAreas` 并列语义回归）；模块2全量 ctest 20/20 绿。commit `b620d91`。

**复现方法**：`fcstepdump.exe data_in\left_skew data_out\stepdump`（看 `[07-RegionAnalyzerCUDA]` 的 `WALL(含topX+copyTo)` 与 `topX filter` 日志行）；背靠背墙钟用相邻两次 Execute 的日志时间戳差。注意：fcstepdump 每帧写 PNG 会造成 GPU 空闲降频尾帧（环境效应），连续运行/产线无此问题。
