# 极线切曲线映射表（curve_map）

> 依据《算子提炼模板.md》v1.0 十二段落；设计文档 `docs/plans/2026-08-26-curve-map-design.md`（五轮审查+实施三轮纠错定稿）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-12b` |
| 中文名称 | 极线切曲线映射表 |
| 英文目录名 | `curve_map` |
| 运行平台 | CPU 生成（OpenMP 行级并行）；GPU 查询（curve_map_lookup.cu 后置） |
| 所属流程 | 激光标定（PJC 之后、温度表打包之前；与 12 plane_map 并列的新选项，旧算子保留为对照） |
| 精度档次 | `③`（解析求交；解码 d ≡ lround(真值)，误差有界 ±0.5 格） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 25 条发射曲线 coeffs | `vector<ImplicitCurve>`（←laser_calib.json pjc 节） |
| **输入②** | f/主点（←P1）、基线 B=\|P2(0,3)\|/f、virtualT=tRect | double/Point2d/Vec3d（**矫正系直连**，R≡I 不入参） |
| **输入③** | imageSize＋ROI（←validRoiLeft 实读） | Size/Rect |
| **输入④** | epipolarRowStep（JSON 继承，禁硬编码）＋深度窗 | float |
| **输出①** | 压缩表字节流（文件头+绝对行索引+线分组位打包） | `vector<uint8_t>`（left_skew 实测 31.4MB） |
| **输出②** | 覆盖统计 perLine×perRow（uLmin/max/maxGap） | `vector<CurveMapCover>` |
| **查询** | Lookup(row, uL) → 候选 {(lid,uR)}（每线 ≤2 双根） | `CurveMapTable` |

## C. 算法

```
逐行(OpenMP 并行) × 逐 uL 格(0.7 步进) × 25 曲线:
① 极线 = 过 (uL,v) 与极点 E=(f·tx/tz+cx, f·ty/tz+cy)≈(5570,1710) 的直线（线束）
   闭式 q(Z) = p + [tz·(p−c) − f·t_xy]/(Z−tz)（加号, 插桩数值验证）
② 直线×隐式二次曲线 → 二次方程闭式解（≤2 根, 退化 a≈0 退一次, Δ<0 无解）
③ 交点 → Z=(qc·tz−f·tx)/(qc−(uL−cx)) → 深度窗过滤 → uR = uL − f·B/Z（视差恒正）
④ 量化 0.7 栅格; (行,lid,uL,uR) 入表
```

**关键实现教训（H 级）**：Δd 差分必须取"取整绝对值之差"（lround(e.d)−prevDint）——浮点差取整会以 +0.44 格/条目持续累计漂移（实测漂至 14 格）；取整差使解码 d 恒等于真值取整。

## D. 依赖

```
PJC(曲线+tRect, 矫正系直连零翻译) → curve_map → 温度表打包 → laser_match_scan(查询协议待适配)
```

**不依赖**：R1/R2（红线：禁入本算子）、VirtualPixelGenerator、CUDA（生成纯 CPU）。

## D2. 衔接

| 方向 | 说明 |
|------|------|
| 上游 | 全部从 laser_calib.json 直读；ROI←validRoiLeft |
| 下游 | rowStep 写入表头供 match_scan 继承（F4 内建）；多候选(≤2/线)判决协议与 F4 同批适配 |

## E. 架构

```
curve_map/
├── curve_map.h/.cpp        # 生成器+压缩表（Lookup/Save/Load）
├── curve_map_table.h       # （并入 .h, 预留拆分）
├── curve_map_lookup.cu     # GPU 查询（后置）
└── tests/test_curve_map.cpp  # 8 用例
```

表格式：文件头(magic/version/rowStep/窗/ROI/行数)＋uint32 绝对行索引＋行内线分组{lid,count,anchor}＋16bit 条目 [ΔuL:6\|Δd:10 zigzag]；ΔuL≥63 或每 32 条强制转义（3 word：esc+绝对uL+绝对d，双绝对兼作二分轴标）。

核心类：`CurveMapGenerator`（无状态）/`CurveMapTable`；日志标签 `"12b-CurveMap"`。

## J. 环境

Eigen/OpenCV core/nlohmann_json/OpenMP(可选)；无 CUDA；C++17。
**注**：模块2 自 2026-09-03 起才真正链接 OpenMP（`dedec91`，写法对齐 09 侧）——此前行级 `#pragma omp` 被静默忽略（61 档生成串行）；生效后线程数无关确定性经三档 sha256 对拍，`vcomp140.dll` 由 `fc_deploy_crt` best-effort 部署。

## F. 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| epipolarRowStep | 0.7 | JSON 继承红线 |
| depthMin/Max | 100/700 | 收录窗 mm |
| uSampleStep | −1(=rowStep) | 行内 u 步 |
| maxGap / threads | 8 / 0 | 空洞阈值/并行 |

## G. 约束

矫正系直连禁 R1/R2；曲线唯一真源；位宽（u≤2926 格/视差≤2443/行≤2194 → uint16）；生成耗时目标 ≤10s（实测 ~4s）；查询单帧 20-50μs 级（CSR 行定位＋组内顺扫）。

## K. 质量

Normal（无空洞）/Degraded（gap≤maxGap）/Warning（超阈值或零条目）。
**B1 闭环验收**（唯一判据，success≠正确）：left_skew 实数据 **15552 例 0 失配 100% PASS**（fcstepdump --curve-map 内建）；合成场景 8/8 单测绿（几何闭环/符号红线/往返/双根上限/深度窗不变式）。

## H. 风险（实施期实际踩坑全录）

| 程度 | 坑 | 对策（已落地） |
|:--:|---|---|
| 🔴 | Δd 浮点差取整累计漂移（+0.44/条目 → 14 格） | 取整绝对值之差编码（C 节教训） |
| 🔴 | 极线闭式符号写反（−N vs +N，设计文档曾错） | 插桩数值验证定案 +N；文档已纠 |
| 🟡 | 转义需双绝对（uL+d）——单绝对时断档 Δd 越界 | 3-word 转义格式 |
| 🟡 | B1 行采样域混淆（像素 y vs 栅格行号） | 采样域=行号区间 [floor(y/s), ceil((y+h)/s)] |
| 🟡 | 真实曲线系数 1e-7 量级，\|F\| 扫描式真值无鉴别力 | B1 真值必须用精确二次求交 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ✅ 可直接使用（factory_calib 现版，2026-08-27 实跑验收；2026-09-03 起行级 OpenMP 真正生效） |
| **实测** | left_skew：25 曲线/1395 ROI 行/**1535 万条/31.4MB**（预测区间内）/worstGap=1 无空洞/Normal/B1 100% |
| **对比旧 plane_map** | 曲线真源✓/线号几何事实✓/矫正系直连✓/表密度高一量级 |
| **续作** | GPU 查询 kernel；~~温度档策略（增量 vs 按需重算）~~ **已落地：curve_map_temp_table（4-14, 61 档 sidecar 重标生成，见 `docs/operator_docs/curve_map_temp_table/`）**；match_scan 多候选协议适配（与 F4 同批）；09 侧同步 |

---

> **参考卡结束**
