# steger_fast dedup v2 移植回 factory_calib — 设计文档

日期：2026-10-13｜类型：主工程算子升级后的同步拷贝（AGENTS.md 例外清单场景）

## 背景与目标

主工程 `modules/09_operatorlib/core/laser/steger_fast/` 在产线镜像（2026-10-07 三防线同步）之后继续演进，新增了 **dedup v2（平顶双点去重，位移式）**：过曝平顶使同一条亮带在同一列产出两个亚像素中心点，dedup 用"网格排序 + 邻域 PCA 几何切向 + 位移门配对（定长邻接行）+ pull/shortcut 标签传播 + (簇键,rrid) 排序段归约"把同簇双点按质心/法向投影中位合并。由 `StegerParams::dedupMode` 控制（0=Off 默认、1=Centroid；2 号位保留），`StegerResult::mergedCount` 记账。

现状比对（SHA256 实测）：
- `steger_fast.h` / `steger_fast.cpp`：两侧逐字节一致（无需动）
- `steger_types.h`：09 侧 +dedupMode/mergedCount/validate/JSON（+7 行）
- `steger_fast_pimpl.h`：09 侧 +DedupPt 与 ~20 个 dedup 缓冲成员
- `steger_fast_impl.cu`：5 个核心内核逐行一致；09 侧 +11 个 Dedup 内核（约 +760 行）+ `src_points` 间接指针 + bench 日志 Dedup 列（dedupMode=0 时 `src_points==d_points_`，行为逐位不变）
- 测试：fc 侧 15 例（等价性×6＋基准×2＋可视化×1＋宽度门禁×6）；09 侧 18 例（冒烟×6＋基准×3＋可视化×1＋dedup×9）——**fc 侧无 dedup 测试；09 侧无宽度门禁测试**

目标：把 dedup v2 移植回 factory_calib；`dedupMode=0`（默认）下产线行为**逐位不变**（红线）。

## 方案（已与用户确认）

1. **整文件覆盖 3 文件**（覆盖后两侧逐字节一致，维持"逐字拷贝边界"）：
   - `core/common/steger_types.h` → `operators/common/steger_types.h`
   - `core/laser/steger_fast/steger_fast_pimpl.h` → `operators/steger_fast/steger_fast_pimpl.h`
   - `core/laser/steger_fast/steger_fast_impl.cu` → `operators/steger_fast/steger_fast_impl.cu`
   - 依赖自包含：include 仅 `common/calib_types.h`、`common/calib_logging.h`、`<cub/cub.cuh>`（CUDA 12 自带）等两侧同名项；fc2_ops GLOB_RECURSE 自动收录，CMake 零改动
2. **CLI 接线**（仿 `steger.maxLineWidth` 先例，三处）：
   - `calib_io.h`：`LaserOpParams` 增 `int stegerDedupMode = 0;`
   - `calib_io.cpp`：`applyLaserParamsJson` steger 组增 `dedupMode` present-key 读取
   - `laser_calib_cli.cpp`：`stegerParams.dedupMode = ops.stegerDedupMode;`
   - 主库 `laser_calib_params.json` steger 组增 `"dedupMode": 0`（显式 off，产线默认行为不变）
3. **测试双向对齐**：
   - fc 侧 `tests/test_steger_fast.cpp` 合入 09 侧 9 个 Dedup 用例＋helper（DrawFlatTopColumn/FlattenSorted/ExpectIdenticalPointSets/SliceX/MakeFlatTopImage），保留现有 15 例；fc 侧 `DrawGaussianLine` 已有（实现略异但满足用例构图），`Median` 已有
   - 09 侧反向补齐 fc 的宽度门禁 6 例（StegerFastWidthGate 套件）＋helper（DrawLineSeg/MakeThinPlusWideImage）
4. **验证（先红后绿）**：合测试→构建（红：dedupMode 不存在）→覆盖源码→构建（绿）→CLI 接线重建→fc2 全量 ctest（Release）→laser261006-1668 真实数据复跑对拍基线逐位一致（dedupMode=0）→主工程构建＋09 测试绿
5. `factory_calib/AGENTS.md` 补日期条目（惯例）

## 风险与对策

- CUB 在 fc2_ops 构建下首次使用 → 编译失败则补 CMake（预期不需要：CUDA 12 自带 cub 头）
- 真实数据复跑不逐位一致 → 说明默认路径被动过，必须查明（09 侧 Off 路径仅 `src_points` 间接化，理论逐位一致）
- dedup 键打包守卫：图幅 ≤4096 且像素 <4M，超限整帧报错（非静默）；1688×1400≈2.36M 像素在域内
