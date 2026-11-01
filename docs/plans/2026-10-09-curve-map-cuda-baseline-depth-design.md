# curve_map_cuda 基线景深参数化与黄金测试 — 设计文档

> 日期：2026-10-09
> 状态：五轮主审＋四代理（几何数学/CUDA 实现/构建/测试设计）并行复审定稿
> 范围：`factory_calib/module2_laser/operators/curve_map_cuda`（只加不改）＋ 新测试目录
> `factory_calib/module2_laser/tests/baseline_depth/`
> 红线：不影响原代码——`curve_map.h`、CPU 版 curve_map、cmtt_cuda、laser_calib CLI、全部既有测试零改动

---

## 1. 背景与动机

产线设备在**固定景深**内工作（当前数据集 laser261006-1668 定稿深度窗 100–500mm）。
对 `curve_map_temp_table.bin` 参考档（22.5°C，19,157,130 条目）的逐线深度审计证实：

- 全部条目解码深度 ∈ [100.0, 500.4]mm（0.4 为量化展宽，非越界）；
- 但各线近端覆盖不对称：1–14 号线最近仅到 103.6–166.5mm，15–25 号线才全覆盖到 100mm。

用户需求：把"景深"概念**显式化**为"基线景深"，以可设置的景深 1/景深 2 两参数接入
curve_map_cuda，只对该范围内的激光线条目进行计算，并以新测试固化该口径。

## 2. 概念定稿：基线景深 = 矫正系 z 坐标（与现行实现等价）

### 2.1 几何定义

- **平面 1**：立体矫正后左右相机两光轴所在平面 = 矫正系 **y=0（XZ 平面）**；
- **景深面**：⊥ 平面 1 且 ∥ 基线（X 轴）⟺ **z = const 平面**；
- **基线景深**：景深面到基线（z=0）的投影距离 = **z 值**。

### 2.2 现行实现等价性论证

`curve_map_cuda.cu:127` 求交内核：

```
Z = (qc·tz − f·tx) / (q_u − uL)      // qc = q_u − cx
```

推导：左矫正系下，左射线点 P = Z·((uL−cx)/f, (v−cy)/f, 1)；同一 P 在虚拟相机
（R=I、K=f/主点与左目共享）下的投影 q 满足 (q_u−cx)/f = (P_x−tx)/(Z−tz)。
消元即得上式 → **Z 就是交点 P 的 z 坐标 = 基线景深**。

结论：现行 `depthMin/depthMax` 过滤（`curve_map_cuda.cu:128`）的口径**已经就是**
基线景深。本设计是**显式命名＋接口化＋测试固化**，数值行为零变化。

（旁证：极线束几何自洽——q(Z) 轨迹直线过 (uL,v)（Z=∞）与极点 E（Z=0），
"过 (uL,v) 与 E 的直线求交"构造正确。）

## 3. 代码修改（curve_map_cuda，只加不改）

### 3.1 新增 API

`curve_map_cuda.h` 公有段新增：

```cpp
/// 基线景深窗（语义别名）：基线景深 = 矫正系 z 坐标（见文件头注释定义）。
/// 概念上与 CurveMapParams.depthMin/depthMax 同一量——写入同一真相源。
/// 校验：0 < d1 < d2 且两者有限（NaN/Inf 拒绝——旧 validate() 比较式对 NaN 全 false
/// 会静默放行，本接口前置 isfinite 守卫堵住）。
void SetBaselineDepthWindow(double d1, double d2);
std::pair<double, double> GetBaselineDepthWindow() const;
```

（实现注记：`curve_map_cuda.h` 需补 `#include <utility>`——std::pair 来源，
多代理审核发现；Get 返回值从 float 字段扩回 double，Set→Get 往返仅当值为
float 可精确表示时逐位还原，A5 断言按 float 精度比对。）

### 3.2 实现要点

1. **先窄化后校验**：`params_.depthMin = float(d1)` 赋值后校验 **float 值**
   （防极端 double 窄化越界）；校验失败抛 `std::invalid_argument`，
   参数保持半改状态前抛出（构造临时 CurveMapParams 验证通过才提交——
   比 SetParams 旧习"先赋值后校验"更干净，新 API 新卫生）。
2. **isfinite 前置**：`std::isfinite(d1) && std::isfinite(d2)` 不满足直接抛。
3. `curve_map_cuda.cu` 仅追加两个 host 函数；kernel 零改动。
4. 头注释补"基线景深"几何定义段（§2.1 原文）。

### 3.3 影响面清单（零改动项）

- `curve_map.h` / CPU 版 `curve_map.cpp`——不动（附带规避 09 侧同步债务：
  CPU curve_map 与 modules/09_operatorlib 逐字镜像，动共享头会触发两边同步义务）；
- `cmtt_cuda`（CLI 4-14 主引擎消费方）——不感知（经 `CurveMapParams` 原路径行为不变）；
- laser_calib CLI / 全部既有测试——零改动；
- 默认构造路径逐位不变（新方法不调用即无影响）。

## 4. 新测试：`tests/baseline_depth/test_curve_map_cuda_baseline_depth.cpp`

### 4.1 CMake 注册（module2_laser/CMakeLists.txt）

现状：`file(GLOB_RECURSE tests/*.cpp)` 会收子文件夹文件，但注册正则
`tests[\\/]test_[^/\\]*\.cpp$` 只认 `tests/` 平铺——子文件夹测试被**静默跳过**。
变更：foreach 内**跳过分支之前**插入 elseif：

```cmake
elseif(f MATCHES "tests[\\\\/][^/\\\\]+[\\\\/]test_[^/\\\\]*\\.cpp$")
    list(APPEND M2_TESTS ${f})
```

- 实证：`tests/` 现无任何子文件夹（仅平铺 test_*.cpp + .gitkeep）→ 新正则只捕获新文件，零波及；
- 插入位置必须在前——放跳过分支之后即成死码；
- operator 侧测试（`operators/*/tests/test_*.cpp`）命中首分支，路径不变。

链接经既有 foreach：fc2_ops（PUBLIC 透传 fc2_curve_map_cuda）+ GTest + nlohmann；
无 imgcodecs 依赖 → Debug 也可构建。

### 4.2 合成组（自包含，CUDA 必需）

**场景构造约定**（四条红线，缺一不可）：

1. **直线型 conic**（A=B=C=0 的退化二次曲线）：CPU `intersectLineConic` 与
   CUDA kernel 的 `|a|<1e-18` 一次方程分支逐式一致（实证）；
2. **uStep=0.5（2 的幂）**：`invRowStep=2.0` 精确 **且** `u += 0.5` 二进制精确步进
   （整数+0.5 在 <2⁵¹ 域内 IEEE 精确，零累积误差）——量化与采样两条路径零舍入漂移；
   uLq 随样本严格 +1；
3. **场景自证**：直线 conic 的 Z(uL) 存在闭式解（两 2D 直线求交 → q → §2.2 公式）。
   测试先解析算出全采样扫深 [Zsweep_min, Zsweep_max]，**断言其跨骑窗口两端**
   （Zsweep_min < d1 < d2 < Zsweep_max）后才跑 GPU——场景有效性由测试自证，
   杜绝边界断言空转；
4. **makeKey 位域域约束自证**（多代理审核新增）：kernel 排序键为
   `bucket(≤16b)<<48 | uLq(12b)<<36 | (dInt+4096,13b)`——测试场景须显式断言
   ① `W·invRowStep ≤ 4095`（uLq 不溢出 12bit；同时 dInt≤uLq 支配 13bit 不别名）
   ② `nRows·nLines ≤ 65535`（bucket 不回卷——真实 ROI 1757×25=43925 ✓，
   但这是既有代码不设防前提，合成场景必须自查）。

| 用例 | 内容 | 断言 |
|---|---|---|
| A1 窗内一致性 | 窗 [d1,d2] 生成 → `CurveMapTable::Load`＋`EnumerateRow` 逐行流式解码（**−1（cap 不足）必须与 0（空行）区分**：ASSERT_NE(n,−1)；勿 materialize 全量条目） | 每条目 Z′=f·B/(d·rowStep 量化格) ∈ **[d1−δ₁, d2+δ₂]**，δ 用**精确式**：近端 δ₁=d1²·s/(2fB+s·d1) 级、远端 δ₂=d2²·s/(2fB−s·d2)（s=量化格；线性式在远端差 2.9×10⁻⁴mm 非严格上界，多代理勘正）——**δ 一律由 f/B 现算，不硬编码**；fB 按 kernel 操作数序 `f * baseline` 复刻；另断言 `t.depthMin()==d1 && t.depthMax()==d2`（窗口写进序列化头的回读） |
| A2 边界与空窗 | 解析扫深对照：窗外子区间条目被剔除、条目数随窗收紧单调不增；窗设扫深之外 | 0 条目 + `qualityFlag==Warning` + `success==true`；且 Load 成功、`entryCount()==0`（**勿用 valid()**——空表语义未定）；空窗产物仍是合法表 |
| A3 非法窗口 | (d1≥d2) / (d1≤0) / (NaN) / (Inf) | 抛 `std::invalid_argument`；**每例抛后 `GetBaselineDepthWindow()` 仍返回先前值**（"参数不被半改"的唯一可观测判据） |
| A4 CPU↔CUDA 同窗对拍 | 同输入同窗，`CurveMapGenerator`（CPU）vs `CurveMapCuda` | entryCount＋逐条目 (row,lid,uL,uR) 精确一致（条目级比较判别力 ≥ memcmp 且诊断更精准；两侧序列化为同款 E.2 CPU 复刻，相同条目流⇒字节必同，**不另加 memcmp** 防未来头部字段演进脆断）。**构造性论证**（非经验）：直线 conic ⇒ 每(行,uL)单根 ⇒ uLq 严格递增 ⇒ (uLq,dInt) 碰撞不可能 ⇒ CPU 去重分支恒空操作（见 §6 已知分歧） |
| A5 别名等价 | `SetBaselineDepthWindow(100,500)` vs `SetParams` 直写 depthMin/Max（其余参数同） | 两次 Generate 的 tableBytes **memcmp 逐字节一致**；`result.effectiveParams.depthMin/Max == 设置值`＋`GetBaselineDepthWindow()` 回读一致（**按 float 精度比对**——params 字段为 float，非可表示值须 `float(d)` 口径） |

### 4.3 真数据组（自动探测，缺数据 GTEST_SKIP）

- **门控改为仓内惯例**（多代理勘正）：golden 式**数据在位自动跑＋缺席 SKIP**
  （同 test_golden_ref_tier / test_cmtt_cuda RealData 用例），不再 opt-in——
  单档 Generate＋流式解码量级远小于先例默认读 3.4GB sidecar 的用例。
  根探测：env `FC_BASELINE_DEPTH_ROOT` → CWD 上溯若干级 → 缺省：
  - handoff：`test_data/laser261006-1668/camera_calib.json`
  - 曲线：`data_out/laser261006-1668_depth100-500/laser_calib.json`
- 输入组装**逐字段复刻** `test_golden_ref_tier.cpp` 的 `toCurveMapInput` 配方
  （f=P1(0,0)、pp=(P1(0,2),P1(1,2))、B=\|P2(0,3)\|/f、roi=validRoiLeft、
  imageSize 顶层、virtualT=pjc.projectorT、rowStep=pjc.epipolarRowStep——
  **注意缺省 0.7 陷阱，本数据集口径 0.5**），
  **连同 hasPjcF 交叉诊断断言一起抄**（pjc.f==P1(0,0) 等 3 项，handoff/JSON
  不一致时原因前置可见）。
- 流程：`SetBaselineDepthWindow(100,500)` → 单档 Generate → **逐行流式解码**
  （每行解码→更新 per-line 统计→丢弃，勿 materialize 全量 19M 条目）。
- 断言（把 2026-10-09 的 19,157,130 条审计固化为 gtest）：
  - 25 线全部有条目；
  - 每线 Zmin ≥ 100 − δ₁、Zmax ≤ 500 + δ₂（δ 按 §5.2 精确式**由 JSON 实际
    f/B 现算**，勿硬编码常数）；
  - `EXPECT_FLOAT_EQ(t.rowStep(), 0.5f)`（冻结网格回显，先例 test_cmtt_cuda 同款，
    兼防 0.7 缺省渗入）＋ `t.depthMin()==100 && t.depthMax()==500`（窗口头回读）；
  - entryCount 落在宽松界 [15M, 25M]（同架构确定性下与 CMTT 参考档 19,157,130
    一致，但不钉死精确值以免跨 GPU 架构脆断；架构差异最多翻窗边界 ±1 格量化，
    不会整线清零）。
- 可选交叉核对：`FC_BASELINE_DEPTH_SIDECAR=1` 时经 CmttReader 读 sidecar
  参考档 entryCount 与生成值比对（保留 opt-in——读 2.5GB 文件属超昂贵操作，
  符合 FC_CMTT_CUDA_FULL 先例定位）。

## 5. 关键技术论证

### 5.1 解码一致性（d ≡ dInt）

打包字流编码 `dInt = llround(d_px·invRowStep)`（**量化格＝epipolarRowStep**
——invRowStepD，.cu:131-133；非 uSampleStep，二者默认相等但语义源不同，
uSampleStep 单独设置时量化格仍随 rowStep，多代理勘正措辞）；解码侧
`uR := uLq − lround(curD)`，故测试侧 `d = uL − uR ≡ dInt` **精确成立**。注意
独立量化的 `uRq = llround(uRpx·invRowStep)` 可能与 `uLq − dInt` 差 ±1 格——
但那不是落盘值，测试取解码值即无口径裂缝。

### 5.2 双向量化界

接受判据在**真值** Z∈[d1,d2]（kernel 双精度过滤）；dInt = llround(d_px/s)
（s=量化格）→ |e| ≤ s/2 → Z′ = Z/(1+e/d_px)，**两侧都可能越界**：

- 远端（**精确式**，多代理勘正）：Z′ ≤ d2/(1 − d2·s/(2fB))（线性近似
  d2+d2²·s/(2fB) 在 Z=500 参数域差 2.9×10⁻⁴mm，非严格上界）；
- 近端（线性式即保守严格）：Z′ ≥ d1 − d1²·s/(2fB)。

实测校准（laser261006-1668 真数据，fB=164123.855，s=0.5）：
δ(500)=0.3808（实测 Zmax 500.375 ✓）、δ(100)=0.0152（实测 Zmin 100.0 ✓）。
**测试断言一律现算，不硬编码。**

### 5.3 batchCap 无截断

`batchCap = rows×lines×nU×2` = 理论最大双根数 → 收集永不截断 →
窗收紧单调性断言无截断噪声。

## 6. 已知分歧与既有隐患（记录，不修）

**CPU 去重 vs CUDA 不去重**：CPU `curve_map.cpp:173-176` 对条目去重，判据是
float d **逐位相等**（(uL,d) 精确相等——比"lround 同格"更窄：两根 d 不等但
lround 同格时 CPU 也落两条，两侧不分歧；CUDA `thrust::sort_by_key` 只排序
不去重。真正 entryCount 分歧仅发生在 float d 逐位相等时——方向恒 CUDA≥CPU，
触发面极窄；Lookup 返回重复候选，良性）。真数据 2754 万条未发生
（test_cmtt_cuda 全绿是经验背书，非构造保证）。

**为何不修**：修了会改变与已交付 sidecar 的行为，违反"不影响原代码"红线。
**未来若需严格等价**：CUDA 侧排序后线性扫描去重 (row,line,uLq,dInt) 一遍即可
（一行修法，留给后续同步决策）。本设计 A4 用直线 conic 从构造上绕开该分歧。

**makeKey 位域不设防**（既有隐患，多代理审核发现）：排序键
`bucket(≤16b)<<48 | uLq(12b)<<36 | (dInt+4096,13b)` 无运行时断言——
① `W·invRowStep > 4095` 时 uLq 溢出污染 bucket 位域；② `nRows·nLines > 65535`
时 bucket 回卷别名（真实 ROI 1757 行×25 线=43925 ✓ 安全，但全幅 2800×25=70000
会越界）。本设计以场景构造红线 4（§4.2）在测试侧自证规避，不修既有代码。

**kernel 异常路径泄漏**（既有行为）：循环内 thrust::copy 抛异常经外层 catch
返回时不释放 5 个裸指针（kernel error 路径与正常路径均配平，仅此路径泄漏）。
合法参数测试不触发，记录备查。

## 7. 错误处理

| 场景 | 行为 |
|---|---|
| 非法窗口（d1≥d2 / d1≤0 / NaN / Inf） | `std::invalid_argument`，参数不被半改 |
| 全部条目出窗 | success=true + qualityFlag=Warning + 合法空表 |
| CUDA kernel 错误 | 既有错误路径（success=false + message） |
| 真数据/门控缺失 | GTEST_SKIP（仓库惯例） |

## 8. 验证与回归

1. 新 exe `test_curve_map_cuda_baseline_depth` 单跑绿（合成组常跑；真数据组 env 门控）；
2. 既有套件零影响回归：`test_curve_map` / `test_cmtt_cuda` / `test_curve_map_temp_table`
   / `test_golden_ref_tier`（源零改动，仅新增）；
3. build_fc2_rel Release 增量编译（fc2_curve_map_cuda 重编 → fc2_ops/测试重链接）；
4. 文档卡同步：`docs/operator_docs/curve_map_cuda/` 补记新 API 与基线景深定义。

## 9. 交付物清单

| 文件 | 动作 |
|---|---|
| `operators/curve_map_cuda/curve_map_cuda.h/.cu` | 只加不改（2 方法＋注释） |
| `module2_laser/CMakeLists.txt` | foreach 插入 1 个 elseif（位置：跳过分支之前） |
| `tests/baseline_depth/test_curve_map_cuda_baseline_depth.cpp` | 新建 |
| `docs/operator_docs/curve_map_cuda/`（参考卡） | 补记 |
| `docs/plans/2026-10-09-curve-map-cuda-baseline-depth-design.md` | 本文档 |

## 10. 审计轨迹（五轮主审＋四代理并行复审）

1. 一轮：Z=基线景深推导、δ 界、CMake 静默跳过实证、双 JSON 需求；
2. 二轮：`test_golden_ref_tier` 先例（toCurveMapInput 配方/直线 conic/env 探测）、
   解码 d≡dInt 闭环、batchCap 无截断；
3. 三轮：uStep=0.5 双重精确、fB 操作数序、先窄化后校验、sidecar 交叉核对默认关、
   elseif 插入位置显式化、09 侧同步债务规避；
4. 四轮：**CPU/CUDA 去重不对称**（唯一结构性发现）→ A4 构造性论证修正；
5. 五轮：场景自证化、isfinite 守卫、effectiveParams 回显断言、空窗合法表断言、文档卡同步；
6. 四代理并行复审（几何数学/CUDA 实现/构建/测试设计）：**全部"可否定稿"，零否定项**。
   并入修正：远端 δ 界改精确式（线性式差 2.9×10⁻⁴mm 非严格上界）、量化格措辞
   更正为 epipolarRowStep、makeKey 位域域约束升级为场景构造红线 4（uLq<4096/
   bucket≤65535）、去重判据精确化为 float d 逐位相等、bucket 不设防与 kernel
   异常路径泄漏记入 §6、EnumerateRow −1/0 区分、真数据门控改自动探测＋SKIP、
   补 Get 回读/异常后状态不变/窗口头回读/rowStep 冻结回显四断言、逐行流式解码、
   补 `#include <utility>`。
