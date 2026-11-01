# 极线切曲线映射表·CUDA 版（curve_map_cuda）

> 依据《算子提炼模板.md》v1.0 编写（精简 8 段）；对象为 `module2_laser/operators/curve_map_cuda/`——CPU 版 `curve_map/`（12b）的 **GPU 求交分叉**，2026-10-08 由 tests 试验目录转正为 CMTT 主引擎的计算核。

## A. 标识

| 字段 | 内容 |
|------|------|
| 算子编号 | `激光标定-12b-CUDA`（4-14 每档调用一次） |
| 运行平台 | CUDA（自写 kernel＋thrust sort；打包/序列化留 CPU 复刻 E.2 格式） |
| 所属流程 | 仅被 `cmtt_cuda`（12d-CUDA）消费；CLI/工具不直调 |
| 精度档次 | `③`；与 CPU 版结构等价＋参考档逐条目一致（实测），**不承诺逐字节相同** |

## B/C. 数据流与算法

输入/输出/参数类型**直接复用** `curve_map.h` 的 `CurveMapInput/Result/Params`（零新类型）。

```
[CPU] uL 采样序列(逐位复刻 CPU 累加循环) + 25 曲线系数上传
[GPU] intersectKernel: 一线程一(行×线×uL) —— 二次方程求交(双精度, 逐式复刻
      intersectLineConic) → 像面/深度窗/uR 过滤 → 原子池收集{key, uLq,uRq,dInt}
      键 = bucket(行×线)<<48 | uLq<<36 | (dInt+4096)  （全序）
[GPU] thrust::sort_by_key  → 行×线分组 + 组内 (uL,d) 升序（等价 CPU 的 sort+unique）
[CPU] 线性扫描 → 差分位打包 [ΔuL:6|Δd:10zigzag]+转义/锚点 → E.2 序列化(52B头+
      行索引+行块) —— 与 CPU 版字节布局同构, CurveMapTable::Load 可直接解析
```

分批策略：128 行/批 × 上限池（防越界），批间 device 端拼接，末尾单次排序。

## D. 依赖

```
curve_map.h(类型) / common/calib_logging.h / OpenCV core / thrust / cudart
上游: PJC curves + InterpParams(f/pp/B/virtualT/roi)
下游: cmtt_cuda 编排 → CmttBuilder sidecar
```

## E. 架构与构建（关键差异）

`curve_map_cuda.h/.cu`；核心类 `calib::CurveMapCuda`（`Generate(const CurveMapInput&)`）；日志标签 `XC-CurveMapCuda`。**专用库 `fc2_curve_map_cuda`**：nvcc `-fmad=false`（抑制 FMA 缩并对齐 MSVC CPU 舍入，等价性关键）；从 fc2_ops 通配**排除**（防波及其余算子既有旗标/数值基线），经 `target_link_libraries(fc2_ops PUBLIC fc2_curve_map_cuda)` 向全下游符号透传。09 侧暂无对应物（07-E 提速移植为独立债务）。

## F. 参数

同 `CurveMapParams`（rowStep/depthMin/depthMax/uSampleStep/maxGap/threads——threads 不消费，档级并行由编排器负责）。无新增参数字段。

**基线景深窗接口（2026-10-09 增，语义别名）**：`SetBaselineDepthWindow(d1,d2)` / `GetBaselineDepthWindow()`——写入/回读 `params_.depthMin/depthMax` 同一真相源。**基线景深＝矫正系 z 坐标**（平面1=两光轴所在平面 y=0；景深面⊥平面1 且∥基线 ⟺ z=const；到基线的投影距离=z）——kernel 的 `Z=(qc·tz−f·tx)/(q_u−uL)` 即该量，与 depthMin/Max 过滤口径同一（推导见 `docs/plans/2026-10-09-curve-map-cuda-baseline-depth-design.md` §2）。校验：值有限（NaN/Inf 拒绝——validate 比较式对 NaN 静默放行）且 0<d1<d2（按窄化后 float 值）；违例抛 `std::invalid_argument` 且参数保持原值（临时对象先验后交）。**产线接线（同日）**：`cmtt_cuda.cpp` 已改为经本接口设置基线景深窗（不再直写 depthMin/Max），行为逐位不变（laser_calib 复跑输出 FC 字节级一致实证）。

## F2. 黄金测试（2026-10-09 增）

`tests/baseline_depth/test_curve_map_cuda_baseline_depth.cpp`（tests 首个一级子文件夹测试，CMake 注册经新增 elseif 收录）：
- **合成组 A1–A5**：窗内一致性（全条目 Z′∈[d1−δ₁,d2+δ₂]，δ 精确式现算）/边界剔除＋空窗（0 条目→Warning 且产物合法）/非法窗口拒跑（NaN/Inf/倒置/非正，异常后参数不被半改）/CPU↔CUDA 同窗逐条目精确一致（直线 conic 构造性等价）/别名等价（SetBaselineDepthWindow ≡ SetParams 直写，逐字节一致）。
- **真数据组**（golden 式根探测，缺席 SKIP）：laser261006-1668 窗 100–500 单档生成，25 线全有条目＋逐线 Z 界＋rowStep=0.5 冻结回显＋entryCount∈[15M,25M]；可选 `FC_BASELINE_DEPTH_SIDECAR=1` 与 sidecar 参考档 entryCount 对拍。实测 6/6 绿（含真数据）。

## J/L. 环境与关键实测（2026-10-08 转正验证）

CUDA 12.x；-fmad=false。**等价性**：合成单档 685,919==685,919 精确一致；**真实数据参考档 27,538,991 条目与 CPU 版/CPU 交付 sidecar 逐条目一致**。**性能**：1.82s/档（真实全尺寸 1757 行×2325 格×25 线）vs CPU 档级并行 ~25s。开发期四 bug 存档：gridDim.y 画蛇添足（任务 ~128 倍重复执行）、v 漏加 rowMin（0.33% 漂移）、rowStep 0.7/0.5 口径误设（条目差半）、JSON 手工解析键名错（改走生产解析器）。
