# 极线切曲线映射温度表（curve_map_temp_table / CMTT）

> 依据《算子提炼模板.md》v1.0 十二段落编写；设计文档 `docs/plans/2026-09-01-curve-map-temp-table-design.md`（两阶段契约＋硬规则 #1~#8）
> 2026-09-04 对照代码成卡：一期（4-14 接入 CLI，`575323b`）→ 二期（工具 sidecar 选档注入＋--cmtt 诊断，`2efa0aa`）→ 档级并行（`17996dd`）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-12d`（日志标签 `12d-CurveMapTempTable`；伴生 `12b-TempParamInterpolator` / `12c-CmttContainer`） |
| 中文名称 | 极线切曲线映射温度表生成器（61 档 sidecar） |
| 英文目录名 | `curve_map_temp_table` |
| 运行平台 | CPU（OpenMP 档级并行，嵌套禁用；无 CUDA） |
| 所属流程 | 激光标定 4-14（5-3 与 PJC 成功之后；**2026-09-03 替换退役的 4-13 plane_map_temp_table**，CMTT Task 3C） |
| 精度档次 | `③`（解析重标＋插值；参考档与金标准逐位同路径） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | PJC 发射曲线（源③） | `vector<ImplicitCurve>` |
| **输入②** | 模块1 立体矫正温度表（源①，handoff `stereoRectifyTempTable` 整表容错解析） | `StereoRectifyTempTableResult` |
| **输入③** | 5-3 激光外参补偿表（源②，取 virtual→left 组作 virtualT 源） | `LaserExtrinsicCompensateCPUResult` |
| **输入④** | imageSize | `cv::Size` |
| **输出①** | `CurveMapTempTableResult`：tiers 逐档诊断（温度/ok/flags/entryCount/rowCountWithData）＋ clampedRatio ＋ gridHash ＋ **sidecarBytes**（容器整字节流） | move-only 结构体 |
| **输出②** | sidecar 落盘文件 `curve_map_temp_table.bin`（CLI 负责）＋ JSON `curveMapTempTable` meta 10 键 | 文件/JSON |
| **查询** | `CmttReader::load(sidecarBytes)` → `tierBytes(referenceIndex())` 取参考档 blob（Task 5 金标准对拍消费契约） | `CmttReader` |

## C. 算法

**派生配方**（与 fcscan_e2e CurveMapInput 组装逐字一致）：f=P1(0,0)；pp=(P1(0,2), P1(1,2))；B=|P2(0,3)|/f；virtualT=v2l 补偿表 T。

**生成流程**：

```
Step 1  TempParamInterpolator 构造（两源表校验/升序/参考节点定位）
        └ 参考温一致性违约（params vs 两源表）→ invalid_argument 穿透
Step 2  参考档先行（硬规则 #4）: atReference() 零插值＋原始曲线不重标
        └ CurveMapGenerator 生成 → 失败即整体 FAIL（硬规则 #3, sidecarBytes 空）
Step 3  冻结基准: 参考档 blob 头 52B 直解（rowStep/roi/rowCount/rowMin/depth,
        不做全量 Load）→ CmttFrozenGrid ＋ FNV-1a gridHash（逐字段）
Step 4  温度网格逐档（含端点 61 档, temp = ref + k·step 逐位同算式）
        ├ 档级 omp 并行（dynamic, tierThreads 可配; 每档独立生成器实例＋
        │ 独占下标写; 嵌套禁用 → 内层行级自动串行 → 线程数无关确定性 #8）
        ├ 插值 at(temp): f/pp/B/virtualT 线性插值; 越界端点档直取＋clamped
        ├ 曲线重标（Task 0 像素坐标契约）: q′ = s·q + m
        │   s = f_T/f_ref, m = pp_T − s·pp_ref → rescaleCurve 隐式系数变换
        │   （a′=a b′=b c′=c; d′/e′ 含 m 一次项; f′ 含 m²——m=0 退化纯缩放）
        ├ roi 恒取冻结基准（硬规则 #1, 任何温度含 clamp 永不插值）
        └ 冻结网格断言 cmttTierGridMatches（六字段 vs blob 头; #1 防线）
            违例 → 该档降级 missing（bit1, blob 空）
Step 5  CmttBuilder 拼装 sidecar（blob move-through——档载荷只驻 sidecarBytes
        一份, tiers 不保留 blob）
Step 6  质量统计: Normal（无 missing 无 clamped）/ Warning; message 汇总
```

**容器格式（cmtt_container）**：96B 超级头（magic "CMTT"/version 1/tierCount≤1024/tempBaseX10=**参考温**×10 定点/tempStepX10/rowStep/depth 窗/冻结 ROI/rowMin/rowCount/gridHash；**头内无 sha 字段**——整文件 sha256 只存标定 JSON，不自指）＋ tierCount×32B 索引（temperatureX10/offset/length/crc32/entryCount/flags）＋各档 CMPU blob。读取红线：load 校验 magic/version/tierCount/索引边界先于一切；`tierBytes` 拷出→逐档 crc32 懒校验通过才返回（未触碰档不付费）；`nearestTier` 等距 tie-break 偏低温侧（≤）。

**按温选档（common/cmtt_tier_select.h，factory 工具层助手非 09 组件）**：档温锚定——`tierTemp(i) = tempBase + (i − i_ref)·tempStep`（参考档居中，先扫 isReference）；选最大 档温≤目标 的非 missing 档（偏低温 tie-break）；目标以下无非 missing 档 → 升温侧最近（missing 就近跳过先低温后高温）；越上界 → 末个非 missing 档（clamp 注记）；无参考档（理论不可达）→ 拒选。

## D. 依赖

```
模块1 stereoRectifyTempTable(源①) ─┐
5-3 laser_extrinsic_compensate(源②) ─┼→ TempParamInterpolator → 本生成器 → CmttBuilder → sidecar
PJC emissionCurves(源③) ───────────┘        ↑ curve_map(CurveMapGenerator 逐档实例)
下游: fcscan_e2e / fcchain_v3（selectCmttTier 选档, CurveMapTable::Load 内存段重载）
      fcstepdump --cmtt-gen（独立生成）/ --cmtt（容器头转储诊断）
```

**伴生件**：`temp_param_interpolator`（12b，两源表→逐温插值）、`cmtt_container`（12c，Builder/Reader/crc32）；模块2 `operators/stereo_rectify_temp_table/`（自模块1 拷入的类型载体——handoff 解析与 CMTT 输入共用 `StereoRectifyTempTableResult`）。

## D2. 衔接

**上游→本算子**（CLI 4-14 编排）：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| handoff.rectifyTempTable | 整表 | 缺节（模块1 老输出）→ 4-14 跳过不出产物 |
| 5-3 laserExtrinTable | 结果结构体 | 5-3 失败 → 跳过 |
| PJC emissionCurves | 曲线数组 | 空（PJC success 但无线）→ 跳过（07 先例降级语义） |
| config | rowStep=epipolarStep（**禁硬编码红线 #6**）；depthMin/Max（全档统一 #2）；cmttThreads |

**本算子→下游**：

| 输出 | 传递给 | 说明 |
|------|--------|------|
| sidecarBytes | CLI 落盘 `{out 父目录}/curve_map_temp_table.bin` | 落盘前 CmttReader 复核（任何失败不留文件）；写失败/hash 重读失败 best-effort 删半文件 |
| meta 10 键 | JSON `curveMapTempTable` 节（键名照 07 CalibSerialize） | success/message/path/tempBase/tempStep/tierCount/tierCountOk/clampedRatio/gridHash/**sha256**（对落盘文件流式重读计算——hash 覆盖文件字节而非内存 bytes） |
| sidecar 文件 | fcscan_e2e / fcchain_v3 第 4 参 `[temp_c]` 按温选档 | 缺省 tempBase 参考温；sidecar 不可用回落 curve_map 现场生成 |

## E. 架构

```
curve_map_temp_table/
├── curve_map_temp_table.{h,cpp}   # 生成器＋rescaleCurve＋cmttTierGridMatches（BlobHeaderV1 52B 复刻）
├── cmtt_container.{h,cpp}         # CmttBuilder/CmttReader/cmttCrc32（96B 头＋32B 索引 pack(1)）
├── temp_param_interpolator.{h,cpp}# 两源表→InterpParams（at/atReference）
└── tests/
    ├── test_curve_map_temp_table.cpp       # 生成器单元（594 行）
    ├── test_temp_param_interpolator.cpp    # 插值器单元（315 行）
    ├── test_cmtt_container.cpp             # 容器读写/CRC/边界（261 行）
    ├── test_golden_ref_tier.cpp            # 参考档金标准对拍（396 行）
    └── test_acceptance.cpp                 # 端到端验收（680 行）
common/cmtt_tier_select.h          # 工具层按温选档助手（header-only）
```

核心类：`CurveMapTempTableGenerator`（无状态，Generate 一次性）/ `TempParamInterpolator`（构造期校验，查询纯 const）/ `CmttBuilder`·`CmttReader`（Reader 为**非拥有视图**——调用方须保证 data 在生命周期内驻留）。

## J. 环境

OpenCV core / nlohmann_json / OpenMP（档级并行；无 OpenMP 构建退化纯串行）/ C++17；无 CUDA、无 Eigen。

## F. 参数（CurveMapTempTableGenParams）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| referenceTemp | double | 0.0 | 参考温度（**须与两源表 referenceTemp 一致**，违例抛） |
| tempHalfRange | float | 15.0 | ±15°C，须为 tempStep 整数倍（整除性校验，否则抛） |
| tempStep | float | 0.5 | 档距（61 档温网） |
| rowStep | float | 0.0 | **红线 #6：调用方从标定 JSON 继承（CLI=cfg.epipolarStep），无猜默认** |
| depthMin/depthMax | float | 100/700 | 深度窗，全档统一（硬规则 #2；CLI 取 cfg.depthMin/Max） |
| tierThreads | int | 0 | 档级并行：0=自动 min(硬件线程−2, 16)；>0=精确值；≤−1 视为 1（CLI 透传 config `cmttThreads`） |

## G. 约束（设计硬规则全录）

| 规则 | 内容 |
|------|------|
| #1 | roi 恒取参考档 validRoiLeft 冻结——任何温度（含越界 clamp）永不插值；冻结网格断言六字段比对（blob 头 52B 直解，不做 61 档 × 31MB 全量 Load） |
| #2 | 深度窗全档统一（禁逐档漂移） |
| #3 | **参考档 FAIL 整体不出产物**（sidecarBytes 空、message 透传） |
| #4 | atReference() 零插值直取参考节点——独立于 at() 的代码路径（浮点路径不同即不可能与金标准逐位同） |
| #5 | 温窗 ±15 覆盖对齐源表；查询越界 clamp 端点档直取 |
| #6 | rowStep/深度窗由调用方注入禁硬编码 |
| #8 | 线程数无关确定性：档级 omp 嵌套禁用（MSVC OMP 2.0）→ 内层行级自动单线程，行计算序与纯串行一致（**勿开 OMP_NESTED**）；单元 1/4/8 与冻结载体 1/8/16 档 sha256 对拍验证 |
| — | 内存契约：档 blob 生成期 move-through 至 sidecarBytes，全程只一份 |
| — | 温网整除性（halfRange/step 非整数倍 → 端点漂移，构造抛 invalid_argument） |

## K. 质量

**QualityFlag**：Normal（61 档全 ok 且零 clamped）/ Warning（有 missing 或 clamped 档）。missing 档语义：生成失败/冻结网格违例/档内异常（omp 区内异常绝不逃逸，降级 bit1）。

**验收**：单元（生成器/插值器/容器边界/CRC）＋ test_golden_ref_tier（参考档 vs fcstepdump `--curve-map` 金标准逐位对拍）＋ test_acceptance 端到端；真数据验收（二期 `2efa0aa`）：**61 档/参考档对拍 +0.012%＋锚定修复**。CLI 侧另有 CmttReader 复核＋文件级 sha256 端到端契约。

**错误处理**：参数非法/两源表坏（空表、P1·P2 缺失或非 CV_64F、f≤0、参考节点缺失、两表参考温不一致）/参考温违约 → `std::invalid_argument` 穿透；档级失败宽容降级 missing（不废整体）。

## H. 风险

| 程度 | 风险描述 | 对策（已落地） |
|:--:|---|---|
| 🟡 | **blob 头 52B 按契约复刻**（curve_map 序列化头为文件私有，无法 include）——布局漂移（如 rowMin↔dataRows 同尺寸重排）会静默错读 | parseBlobHeader 自洽防线：\|rowMin − floor(roi.y/rowStep+0.5)\| ≤ 1（金标准实例 286/0.7→408.57→408）；根治=curve_map 侧人工加 rowMin()/depth 访问器（遗留项） |
| 🟡 | 0.2 步距旧温表浮点累计误差（12.5+50×0.2=22.500000000000004）→ 参考温节点精确匹配失败 | handoff 解析期表示层校正（阈值 1e-6，就地改 temperature 不触补偿参数） |
| 🟡 | 主点非严格同比例缩放（温度效应以 f 为主但 pp 独立插值） | 重标取一般式 q′=s·q+m（m=pp_T−s·pp_ref；m=0 自动退化纯缩放） |
| 🟢 | 档级并行确定性依赖嵌套禁用 | #8 约束文档化＋三档 sha256 对拍；勿开 OMP_NESTED |
| 🟢 | CmttReader 非拥有视图悬垂 | 契约：data 须在 Reader 生命周期内驻留（整块内存/文件映射） |
| 🟢 | 温窗覆盖不足的旧源表 | validateHandoffConsistency 仅 warn，运行期 clamp 端点档兜底（偏差 B 语义） |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ✅ 可直接使用（factory_calib 现版 4-14 主链） |
| **现有模块** | `factory_calib/module2_laser/operators/curve_map_temp_table/` ＋ `operators/common/cmtt_tier_select.h` |
| **落地史** | 一期 `575323b`（4-14 接入 CLI，退役 4-13）→ 二期 `2efa0aa`（fcscan/fcchain_v3 选档注入＋fcstepdump --cmtt/--cmtt-gen）→ 档级并行 `17996dd`（真规模 t16=65s vs 串行 263s；线程数无关确定性三档对拍） |
| **实测** | left_skew 真数据：61 档生成/参考档对拍 +0.012%（锚定修复后）；消费链 fcscan_e2e/fcchain_v3 sidecar 选档回退现场生成 |
| **主工程 09 侧** | 09 已有同源组件（TempParamInterpolator/CmttContainer/CurveMapTempTableGenerator）；factory 侧 cmtt_tier_select.h 为工具层私有——**同步边界待人工裁决** |
| **续作** | GPU 查询 kernel（curve_map_lookup.cu 后置）；09 侧同步对齐；温度插值档密策略（0.5 → 更细）如需 |

---

> **参考卡结束**
