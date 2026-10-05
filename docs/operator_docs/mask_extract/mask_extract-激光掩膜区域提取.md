# 激光掩膜区域提取（mask_extract_cuda）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为 factory_calib 副本（`module2_laser/operators/mask_extract/`，自主工程 09 逐字拷贝＋2026-08-28 性能优化分叉，见 §L 优化记录）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-1`（4-1；前端入口） |
| 中文名称 | 激光掩膜区域提取 |
| 英文目录名 | `mask_extract`（factory_calib 与主工程 09 同名；文件名带 `_cuda` 后缀，如 `mask_extract_cuda.h`） |
| 运行平台 | CUDA（GPU 形态学，输入 CPU Mat 内部上传） |
| 所属流程 | 激光标定链第一步；fcscan_e2e 扫描链、fcstepdump 调试链共用 |
| 精度档次 | `③`（像素级二值掩膜） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 单帧灰度图（2048×1536，CV_8UC1） | `const cv::Mat&`（算子内部 H2D 上传） |
| **输出①** | 灰度图 GPU 副本（供 4-4 steger 复用，免二次上传） | `shared_ptr<GpuMat>`（CV_8UC1） |
| **输出②** | 膨胀后激光掩膜（postErode 前） | `shared_ptr<GpuMat>`（CV_8UC1） |
| **输出③** | 最终清理掩膜（threshold→腐蚀→膨胀→收边） | `shared_ptr<GpuMat>`（CV_8UC1） |

## C. 算法

```
Step 1  H2D 上传 (d_inputBuffer.upload)
Step 2  cv::cuda::threshold (THRESH_BINARY)         ← params.threshold
Step 3  首腐蚀 (椭圆核 erodeSize)                    ← erodeSize<=1 恒等旁路(2026-08-28)
Step 4  膨胀 (laserDilateSize)
        morphApprox=0: 2D 椭圆核（精确）
        morphApprox=1: 横/竖两趟线核（矩形和近似, 实测约 3~5× 快）
Step 4b 收边腐蚀 (postErodeSize, 可选; 核选择同 Step 4)
Step 5  copyTo → d_cleanedMask（面积过滤职责在 4-2）
输出    ping-pong 结果池×2 流式 copyTo（免每次 clone 的 cudaMalloc+默认流屏障）
```

**关键函数**：`cv::cuda::threshold`、`cv::cuda::createMorphologyFilter`（MORPH_ERODE/DILATE；线核用 MORPH_RECT (k,1)/(1,k)）、`cv::cuda::createContinuous`。

## D. 依赖

```
（无上游, 流水线入口）→ mask_extract → 4-2 region_analyze (d_cleanedMask)
                          └──────────→ 4-4 steger (d_grayImage)
```

**共享/复用**：与主工程 `modules/09_operatorlib/core/vision/mask_extract/` 为逐字拷贝关系（本次优化已分叉，同步时需注意 §L）；与 4-2/4-4 无头文件耦合。

## D2. 衔接

| 方向 | 内容 | 说明 |
|------|------|------|
| 上游→本算子 | CPU 灰度图值传入 | 算子为 GPU 链入口，接口无 GpuMat 重载（上传无法外移，含 ~0.45ms H2D） |
| 本算子→下游 | `d_cleanedMask` → region_analyze `Execute(GpuMat&, stream)` | 必用 |
| | `d_grayImage` → steger `Execute(gray, mask, stream, GroupMode)` | 必用 |
| | `d_laserMask`（膨胀中间结果） | 当前 factory_calib 链路未消费，契约保留 |

**结果生命周期**（2026-08-28 起）：返回 GpuMat 为 ping-pong 池视图，**至少到再执行两次 Execute 前有效**；跨帧长期持有需自行 clone。

## E. 架构

```
mask_extract/
├── mask_extract_cuda.h            # 公开头（纯 C++, Params/Result/算子类）
├── mask_extract_cuda_pimpl.h      # CUDA 桥接（Impl: 滤波器/缓冲/结果池）
├── mask_extract_cuda.cpp          # 桥接实现
├── mask_extract_cuda_impl.cu      # CUDA 实现（rebuildFilters/allocateBuffers/executePipeline/Execute/Warmup）
└── tests/
    ├── test_mask_extract_cuda.cpp          # 18 cases
    └── test_mask_extract_cuda_precision.cpp # 6 cases
```

| 项目 | 名称 |
|------|------|
| 核心类 | `calib::MaskExtractCUDA`（pImpl） |
| 核心方法 | `Execute(const cv::Mat&, Stream&)`／`Execute(const cv::Mat&)`（仅自测） |
| 其他 | `Warmup(rows, cols)`／`Warmup(const WarmupConfig&)`／`SetParams`／`GetParams`／`Destroy` |
| 参数/结果 | `MaskExtractParams` / `MaskExtractResult` |
| 日志标签 | `"06-MaskExtractCUDA"` |

## J. 环境

OpenCV+CUDA 4.13（core / cudaarithm / cudaimgproc / cudafilters / imgproc[CPU 核生成]）；CUDA 12.6（sm_75/86/87）；nlohmann_json ≥3.11；spdlog；gtest；C++17 禁 C++20；`BUILD_CUDA`；MSVC `/bigobj`。

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| threshold | int | 80 | [0,255] | 二值化阈值（left_skew 实测 50） |
| erodeSize | int | 5 | 奇数≥1 | 首腐蚀核；**≤1 时恒等跳过**（2026-08-28） |
| laserDilateSize | int | 3 | 奇数≥1 | 膨胀核（left_skew 实测 19） |
| postErodeSize | int | 0 | 奇数或 0 | 收边腐蚀核；0=关（实测 13；典型 e3+d19+p13） |
| **morphApprox** | int | **0** | 0/1 | **2026-08-28 新增**：1=大核形态学横竖两趟线核近似（矩形和）；0=精确 2D 椭圆核 |

left_skew 实测组配：`threshold=50, erodeSize=1, laserDilateSize=19, postErodeSize=13`（fcscan_e2e / fcstepdump 硬编码同值；fcscan 另支持 `FC_MASK_APPROX=1` 环境变量切换 morphApprox）。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 单帧性能 | 空载 Execute：精确 ~7.3ms／近似 ~2.4ms（2048×1536，CUDA 事件）；流水线负载态 mask 21.6／8.9 ms/帧（负载态约为空载 ×1.9，降频所致） |
| 显存占用 | 精确路径 12 张 CV_8UC1 ≈37MB/实例（6 工作＋池 6）；近似路径 13 张 ≈40MB；L/R 双实例 ~75-80MB |
| 线程安全 | 非线程安全；Debug 模式 `atomic<bool> inProcess_` 并发检测 |
| 实例隔离 | 左右相机各持独立实例（maskL/maskR） |
| Warmup | 生产使用前必须 `Warmup(1536,2048)`，否则首帧内部分配＋WARN 日志 |
| 结果池 | ping-pong ×2；调用方不得跨两帧以上持有输出 |

## K. 质量

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 提取成功 | 流水线正常完成 |
| Degraded | OpenCV/标准异常 | catch (cv::Exception / std::exception)，message 带详情 |
| Warning | — | **本算子从不置位**（契约三值枚举中未使用） |

错误处理：参数非法抛 `std::invalid_argument`（validate）；失败路径 `success=false + message`（无空输入专判，空图由 OpenCV 抛出转 Degraded）。

## H. 风险

| 严重程度 | 描述 | 影响 |
|:--------:|------|------|
| 🟡 中 | morphApprox=1 为矩形和近似：掩膜 2.5~3.5% 像素与椭圆核不同（对角向 ±~3px） | 下游点数 −1.45%（22 姿态实测），深度分布不变；默认关闭，启用需过 A/B 门 |
| 🟡 中 | 与主工程 09 的「逐字拷贝」边界已因 §L 优化分叉 | 后续同步需按 §L 清单对齐，否则两边行为漂移 |
| 🟢 低 | 接口只收 CPU Mat，H2D 上传（~0.45ms）在算子内、无法与流水线重叠 | 改 GpuMat 入参属接口变更，未做 |
| 🟢 低 | 阈值固定非自适应 | 亮度变化场景需调参（现由 config/调用方指定） |

## I. 状态

| 项目 | 说明 |
|------|------|
| 判定 | ✅ 可直接用（2026-08-28 性能优化后基线） |
| 现有模块 | factory_calib `module2_laser/operators/mask_extract/`（本次会话已优化＋验证） |
| 回归 | 模块2 全量 ctest 20/20 绿（2026-08-28，优化后）；含本算子 24 cases |

## L. 优化记录（2026-08-28）

**动机**：fcscan_e2e 全链计时显示 mask 占 28.08ms/帧（36%，第一大项）；分步基准（`fcstepdump --mask-bench`，44 图×20 次，CUDA 事件）定位：膨胀 19×19 占 4.07ms（54%）＋收边腐蚀 13×13 占 2.02ms（27%）＋ clone×3 占 0.78ms（每次 clone 隐含临时 cudaMalloc＋默认流屏障）。

| # | 改动 | 性质 | 实测收益 |
|---|------|------|---------|
| 1 | erodeSize≤1 恒等旁路（不建核不启动 kernel） | 位级等价 | −0.12ms/图 |
| 2 | 结果 ping-pong 池（Warmup 预分配×2 组）＋流式 copyTo 替代 clone()（免临时 cudaMalloc＋默认流屏障） | 位级等价 | 流水线 mask 28.08→21.56ms/帧（**−6.5ms，主要收益在此**）；空载 Execute 仅 7.54→7.35（clone 的 cudaMalloc 代价空载下占比小，负载态默认流屏障被显著放大） |
| 3 | Warmup/Execute 重分配统一 `allocateBuffers()`，补齐 d_postMask 缺口（原首帧隐式分配 WARN） | 无风险 | 消首帧分配告警 |
| 4 | `morphApprox` 参数：线核两趟近似（默认关） | **近似**（矩形和≠椭圆和） | 纯形态学 6.80→1.59ms（4.3×）；含 threshold 的分步序列 6.90→1.69ms（4.1×）；整算子 7.35→2.38ms（3.1×，空载） |

**整算子/流水线实测**：

| 配置 | Execute(空载) | mask/帧(流水线) | 整帧(流水线) | 22 姿态匹配点 |
|------|--------------|----------------|--------------|---------------|
| 优化前 | 7.54ms | 28.08ms | 78.43ms | 397,658 |
| 优化 1-3（精确路径） | 7.35ms（空载基本持平，收益见 mask/帧列） | 21.56ms | 68.77ms | **397,658（逐位一致）** |
| ＋morphApprox=1 | **2.38ms** | **8.93ms** | **61.41ms** | 391,888（−1.45%） |

**验证**：①优化 1-3 为**构造性等价**（1×1 腐蚀恒等旁路＋拷贝替代 clone 不改位），实证：全链 22 姿态匹配数与旧版完全相同（397,658）；②近似路径 22 姿态 A/B：掩膜差 2.5~3.5%px → 点数 −1.45%，点云深度分布不变（p50 455.1 vs 453.9mm）；③ctest 20/20 绿。

**工具与产物**：`fcstepdump --mask-bench <in_dir> <out_dir> [iters]`（分步计时＋base/approx/diff 掩膜 PNG×132）；`data_out/mask_bench/`、`data_out/scan_e2e_ab_{base,approx}/`。基准注意：空载逐 isolated 步测值与流水线负载态差约 1.9×（降频），以流水线复测为准。
