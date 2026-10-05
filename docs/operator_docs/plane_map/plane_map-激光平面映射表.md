# 激光平面映射表生成（含坐标系约定审查记录）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为 factory_calib 现版
> **本文档核心价值**：记录 2026-08-26 R1 坐标系审查定案的接口约定与修复（方案 Y）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-12` |
| 中文名称 | 激光平面映射表生成 |
| 英文目录名 | `plane_map` |
| 运行平台 | CUDA (Thrust) + CPU (Eigen) |
| 所属流程 | 激光标定流程（PJC 之后；原 4-13 plane_map_temp_table 链路**已于 2026-09-03 从 CLI 退役**，由 4-14 curve_map_temp_table 替代——算子本体仍编译，fcstepdump `--pm-ablation` 消融保留） |
| 精度档次 | `③`（亚像素/浮点类） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | d_virtual_pixels（u/v/lineId） | `GpuMat` CV_32FC3 |
| **输入②** | virtualK, **virtualR, virtualT（⚠ 原始左相机系约定，见 C/H）** | Matx33d / Vec3d |
| **输入③** | calib::StereoCalibration（R1/R2/P1/P2/imageSize） | 结构体 |
| **输出** | d_left_to_right（uL/vL/uR/lid）+ d_right_u + lineStats | CV_32FC4 / CV_32FC1 |

## C. 算法

**核心流程**：

```
Step 1: VirtualPixelGenerator 生成虚拟像素网格（全幅 gridStep 均匀; F3 待改曲线带采样）
Step 2: [Projective 法] kernelProjective:
        射线 P(t) = virtualT + t·(virtualRᵀ·K⁻¹·uv), t ∈ [depthMin,depthMax] 采样
        uL = P1·(R1·P(t))/z   ← ⚠ R1 在此: 原始左系 → 矫正左系
        uR = P2·(R2·P(t))/z;  FOV 内保留 → gridStep 量化
Step 3: [FundamentalMatrix 法] F_VL/F_VR 极线扫描（同坐标系约定）
Step 4: kernelCompactCandidates → thrust 排序去重（键 lid,vL,uL）→ kernelComputeStats
```

**⚠ 坐标系接口约定（2026-08-26 --r1-audit 实验定案）**：

| 约定 | 内容 |
|---|---|
| kernel 期望 | **virtualT/virtualR 表达于原始左相机系**（kernel 内自乘 R1 完成 原始→矫正） |
| 上游 PJC 输出 | tRect/R=I 在**矫正左系**（4-8 Q 重投影点云同系） |
| **正确喂法（方案 Y）** | `virtualT = R1ᵀ·tRect, virtualR = R1`——恒等式 R1·(R1ᵀ·(tRect+t·K⁻¹·uv))=tRect+t·K⁻¹·uv 保证 kernel 投影=矫正系直投 |
| 直喂 tRect 的后果 | 系统错位 R1（本机 ≈10.4° yaw），**最大 488.2px**（--r1-audit 实测）；pairs 不敏感（全幅网格仍大量在 FOV），**表条目位置静默错误** |

**关键函数**：kernelProjective / kernelFundamental / computeF_VL / computeF_VR / gpuSortUnique（thrust::stable_sort+unique）。

## D. 依赖

```
PJC（projectorT——经方案Y翻译）→ 本算子 → plane_map_temp_table(4-13 内部实例化调用；4-13 已退役，见 A/I)
VirtualPixelGenerator（内部子模块）
```

## D2. 衔接

| 方向 | 内容 | 说明 |
|------|------|------|
| 上游→本算子 | virtualK/R/T | **必须方案 Y 翻译**（CLI 已实现 2026-08-26）；单测旧约定 R1=I 时两种喂法等价——**单测无法捕捉该错误**（历史遗漏根因） |
| 本算子→下游 | d_left_to_right/d_right_u | 4-13 每温度档调用（历史；4-13 已退役，见 A/I） |

## E. 架构

```
plane_map/
├── plane_map_cuda.{h,cpp} / _pimpl.h / _impl.cu
├── virtual_pixel_gen.{h,cu}          # 全幅网格（F3 待改曲线带）
└── tests/test_plane_map_cuda.cpp     # ⚠ 夹具 R1=R2=I——从未覆盖真实 R1
```

核心类 `PlaneMapCuda`(pImpl)+`VirtualPixelGenerator`；日志标签 `"12-PlaneMapCuda"`；LineMapStats（lineId/numPairs/uV 范围）。

## J. 环境

OpenCV+CUDA 4.13（cudaimgproc/cudaarithm）、CUDA 12.6、Thrust、Eigen≥3.3；`--extended-lambda`。

## F. 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| method | PlaneMapMethod | Projective(0) | Projective=射线采样 / FundamentalMatrix=极线扫描 |
| gridStep | float | 0.5（left_skew cfg=16） | 量化步长 |
| depthMin/Max | float | 100/5000 | 射线深度范围 mm |
| depthSamples | int | 200（cfg=100） | 深度采样数 |
| epipolarStep | float | 0.5（cfg=0.7） | F 法极线扫描步距 |
| deviceId | int | 0 | — |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 虚拟像素数 | >0 |
| 坐标系 | **输入原始左系（方案Y 翻译责任在调用方）**；输出像素坐标（矫正像面） |
| 显存 | Projective 法 candidates = N×depthSamples，大分辨率 OOM 风险 |
| 线程安全 | 非线程安全 |

## K. 质量

Normal 单一标记；参数非法抛 invalid_argument；空输入/类型不符/CUDA 错误 → success=false。
**⚠ 成功≠正确**：本算子对坐标系错误不报错（488px 错位的表仍 success=true）——正确性靠调用方翻译＋B1 正向闭环验收（待做）。

## H. 风险

| 严重程度 | 风险描述 | 影响 | 对策 |
|:--------:|---------|------|------|
| 🔴→✅已修 | **CLI 直喂矫正系 tRect（488px 错位）**——2026-08-26 定案并于 CLI 实现方案 Y 翻译（`--r1-audit` 假设A=488.2px/假设B=0px；`--pm-ablation` 两喂法表实测不同） | 修复前 4-13 全部 101 档表**不可用于扫描** | 已修（CLI 3d 段 virtualR=R1/virtualT=R1ᵀ·tRect）；**旧标定产物作废，须用修复后 CLI 重生成** |
| 🟡 中 | **单测夹具 R1=R2=I**，坐标系错误免疫 | 无法回归防护 | 待办：补真实 R1 单测（喂 R1ᵀ·t 期望=矫正系直投） |
| 🟡 中 | 全幅网格 × 每线一遍 → lineId 语义退化（同位置多线重复条目） | 表冗余、扫描查表 line_id 无几何含义 | F3 曲线带建表一并解决 |
| 🟡 中 | pairs 计数对坐标系错误不敏感（消融实测差仅 5.5%） | 不能用 pairs 当验收指标 | 待办：B1 正向闭环（点→表→反解→比对） |
| 🟢 低 | Projective OOM 风险（大图×depthSamples） | 显存 | 已知文档项 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ✅ 算子数学正确可直接使用；**调用方必须方案 Y 翻译**（CLI 已实现）。**CLI 主链已退役（2026-09-03 CMTT Task 3C）**：4-13 plane_map_temp_table 由 4-14 curve_map_temp_table（61 档 sidecar，曲线真源+重标）替代，本算子仅存于 fcstepdump `--pm-ablation` 消融/对照路径 |
| **现有模块** | `factory_calib/module2_laser/operators/retired/plane_map/`（2026-09-04 移入 retired/ 归档；仍编译进 fc2_ops） |
| **修复记录** | 2026-08-26：`--r1-audit`（假设对比定案）＋ `--pm-ablation`（表级消融）＋ CLI 方案 Y（3 行）＋全量重跑（status=ok, 101 档） |
| **续作（待办）** | ①~~5-3 外参表坐标系审查~~（已被 4-14 链路取代——CMTT 经 TempParamInterpolator 消费 5-3 表，派生配方与 fcscan CurveMapInput 组装一致）②单测补真实 R1 用例（若保留消融路径）③~~F3 曲线带建表~~ **已落地为 4-14 CMTT** ④~~旧温度表产物重生成~~ **改由 4-14 sidecar 出产物** |

---

> **参考卡结束**
