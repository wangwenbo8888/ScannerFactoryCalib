# 2026-11-14/15 双模块标定速度优化战役（全记录）

> 会话模式：三轮多代理分析→实施→验证→提交，全程自动化。
> 范围：`factory_calib` 模块1（camera_calib.exe）＋模块2（laser_calib.exe）。
> 分支 `1002` 本地提交（未推送）。设计细节见各代理报告（会话记录），本文为权威落账。

## 0. 最终成绩

| 程序 | 优化前 | 优化后 | 加速比 | 精度 |
|---|---:|---:|---:|---|
| 激光标定 | 128.5s | **10.2s** | **÷12.6** | verdict=PASS, rms 0.0853px, A1-A7 全过（与历史主导盆地一致） |
| 相机标定 | 49.0s | **22.4s** | **÷2.2** | 输出与优化前金标 **SHA256 逐字节一致** |
| 整机合计 | ~177.5s | ~32.6s | ÷5.4 | — |

最终每步耗时/精度实测（2026-11-15）：

**激光**（laser261006-1668, WALL=10.2s）：启动 0.18 → PNG 并行解码 0.13 → 帧循环 0.69 → stage-1 0.90 → 盆地探针 0.43 → stage-2 主解 1.43 → F7 验收 2.23 → 5-3 ~0 → CMTT 3.70 → 写盘 0.20。
projectorT=(39.8679, 8.1292, 10.6062)mm；A1=0.0014/A2=0.0038/A3=0；A5=0.104mm(阈1.0)/A6=0.232mm(阈2.0)/A7=0.115mm(阈0.5)；cond=3.08e12；solveFailures=0。

**相机**（camera261006-1688, WALL=22.4s）：启动+枚举 2.09 → 角点提取 18.10 → 内参(并行) 1.32 → 指标 0.56 → 外参+矫正 ~0.01 → 温表+写盘 0.19。
left rms=0.0575px / right rms=0.0655px / reproj_mean=0.0463px / stereo RMS=0.2468px / 极线=0.3242px（历史口径）/ SHA256=E6EB3369…（=金标）。

## 1. 提交清单（15 笔，本地分支 1002）

| # | commit | 轮次 | 内容 | 效果 | 验证口径 |
|---|---|---|---|---|---|
| 1 | `4b10491` | R1-R3 | f7Threads 装载层归一 bug 修复＋stage-1/1.5 并行探针＋PJC jac OpenMP | 128.5→42.3s | 逐字节 |
| 2 | `20c6579` | R4 | CMTT 参考档复用＋条件数移出循环＋IRLS 收敛判据 | →41.0s | 逐字节 |
| 3 | `03d0d70` | R5b | curve_map_cuda 打包 reserve＋bulk append | CMTT 4050→3415ms | 逐字节 |
| 4 | `4199feb` | R6b | F7 denoise 预计算缓存＋CMTT 双执行流 | F7 −2.1s | 三方逐字节 |
| 5 | `005af43` | A1 | **TermCriteria 修复**：calibrateCameraRO 默认 EPS=DBL_EPSILON 永不可达→恒满 30 轮；改 1e-10 | 相机 49→36.9s | 精度差 1e-13 |
| 6 | `407e0cf` | A1四' | 左右目内参 std::thread 并行（E2 翻案） | 36.9→34.0s | SHA256 逐字节 |
| 7 | `7f07cbe` | A3 | relDrop30 数据项口径＋IRLS 权稳定检查前移（双重死代码修复） | stage-2 12.4→6.9s | 数值门 PASS |
| 8 | `2e99155` | A2 | **F7 任务禁 Phase B**（28 任务各全价重跑右目项≈75%）＋MultiLineResult.phaseAT 同口径参考 | F7 20.1→5.0s | 数值门 PASS |
| 9 | `04e15b7` | B1 | 单线解 relDrop 同款修复＋探针共享 denoise 缓存 | 21.8→20.1s | PASS |
| 10 | `14ce654` | B3 | **解析雅可比**（默认开）：t 列+C 列逐点解析装配替代 9 FEE 差分 | 20.1→17.8s | rms relDiff 7.5e-9 |
| 11 | `7230969` | B2四 | 流式读图（imread 融进角点 worker，内存 675→57MB） | 相机 34→27.6s | 逐字节（除时间戳） |
| 12 | `a634995` | C3 修复 | opt_bench loadDataset 钉 streamIO=false（测试装载缺口） | fc1 回归恢复 10/10 | 回归 |
| 13 | `19a538d` | C2 | 按线并行残差评估（computeMultiResidualsPar，jacThreads 旋钮复用） | 16.8→14.5s | 双跑逐字节 |
| 14 | `704d333` | C1 | PNG 并行预读＋BCrypt 硬件 sha256（失败回退标量） | →14.5s | BCrypt 摘要交叉验证 |
| 15 | （实验） | — | 角点标志消融＋arm3 半分辨率复测（只测未改产品） | 分析数据 | 见 §5 |

## 2. 技术要点（按发现价值排序）

### 2.1 隐藏在 OpenCV 默认参数里的"永动机"
- **A1**：`calibrateCameraRO` 公开 criteria 参数默认 `TermCriteria(COUNT+EPS, 30, DBL_EPSILON)`——2.2e-16 在双精度迭代中不可达 ⇒ 恒跑满 30 轮。改 1e-10 后 LM 8-15 轮自然收敛。内参 12→3.9s，精度差 1e-13 级。**这是相机侧最大单点。**
- **A3**：PJC 主循环 relDrop30 平台判据比较的是含正则 cost——λReg 每迭代 ×0.95 衰减使正则项持续缩小，数据零改善时仍产出 >5% 假下降（"喂饱"）。判据语义恢复为数据项口径后主循环 100→~50 迭代。单线解（stage-1）同病同修（B1）。
- **A3/B1**：IRLS 权稳定检查是**双重死代码**——检查在轮体末尾（maxRounds=2 下必跑满）且比较对象 robustW 已被上一轮置空 → break 永不触发。前移到轮首＋保留上轮权重后可整轮跳过。

### 2.2 F7 的结构性浪费
- **A2**：F7 的 28 个漂移任务按值拷贝 mlIn（enableRightTerm=true）→ **每个任务全价重跑 Phase B**（右目项 uR 行成本 ×2.5-3.5，占短解 ~75%）。对齐 stage-1.5 盆地探针先例（"uR 对盆地选择无贡献"）禁用；新增 `MultiLineResult.phaseAT`（Phase B 前的 t）作漂移参考，消除 ‖t_PB−t_PA‖ 常量偏置。F7 20.1→5.0s。
- **R6b**：denoise 逐姿态独立 → (line,pose) 预计算缓存，剔线/剔姿态子集按原序拼接即原输出（纯缓存）。B1 又把缓存前移共享给探针（原每次探针白算全量降噪）。

### 2.3 解析雅可比（B3 交叉验证制）
- A2/A3 两代理独立推导，**B3 审核抓出两处会致错的 bug**：A3 的 t 列像素导数整体反号（照抄发散）；A2 的 C 列简化在 Huber/Cauchy 相漏 α 核导数项。合并修正后实施。
- 范围：accumBlockHg 块路径 t 列(3)+C 列(6×L) 一次逐点解析 pass，替代 3 次全量差分＋150 次线内差分（9 FEE→1 pass）。**Phase B（uR 行含二次求根）自动回差分**。开关 `pjc.analyticJacobians`（主库=true，回滚键）。
- 验证：4 个 FD↔解析单测（1 步 L2/鲁棒/IRLS/全程收敛）＋真数据门 rms relDiff **7.5e-9**、t 差 1e-6mm、A5-A7 一致。
- 公式存档（B3 报告修正版，符号已按 P−t 约定核对）：
  - α=√wt+dd·Ws·w·w_rob'/(2√wt)；γ=dd·Ws·w_rob/(2√wt)（仅 t_z 列，w clamp 区取 0）
  - t 列：u_x=−f/Zp, u_z=+(u−cx)/Zp, v_y=−f/Zp, v_z=+(v−cy)/Zp
  - C 列：∂dd/∂C_k=(φ_k−dd·G_k/den)/den，φ=[up²,up·vp,vp²,up,vp,1]，G=[2up·Fu, vp·Fu+up·Fv, 2vp·Fv, Fu, Fv, 0]

### 2.4 评估/IO 并行化
- **C2**：解析雅可比后试探评估占主解 wall 80-90% → computeMultiResidualsPar 按线并行（行域不相交→r 逐位一致；costS 线间合并浮点结合律变化→数值门）。复用 jacThreads 旋钮（F7 任务 jacThreads=1 自动豁免串行）。附带收益：探针/阶梯重试/Phase B 全量重算自动加速。
- **C1**：58 张 PNG 串行 imread（~1.5-2s）→ 8 线程领任务解码（0.13s）；imread=文件字节纯函数→逐字节一致。sidecar 447MB sha256 标量实现 ~0.4GB/s（1.0-1.3s）→ BCrypt(CNG) 硬件路径（~0.25s），摘要与 PowerShell Get-FileHash 交叉验证一致，provider 失败回退标量。
- **B2 批次四**：相机流式读图——imread 融进角点 worker 解码即弃（峰值内存 675→57MB），读失败压缩重编号与装载期丢弃语义同构。主库键 `corner.streamIO=true`（回滚=false）。
- **盆地备注**：C2 的 costS 合并序 1e-15 扰动使求解器在两个近平局盆地间翻转，落回**历史主导盆地**（R1-R6 时代 t≈(39.87,8.13,10.605)，rms 0.0853 优于 A3 时代的 0.0864）；A5/A6/A7 逐位不变（漂移测量盆地协变）。PJC 盆地敏感性是既有特性（AGENTS #9）。

### 2.5 验证方法学（双轨制沿用）
- **逐字节轨**（SHA256 对拍）：R 系列、内参并行、流式读图、PNG 并行、C2 双跑确定性、BCrypt。
- **数值门轨**（终止时机/雅可比/合并序变更）：解析雅可比（rms 7.5e-9）、F7 禁 Phase B、relDrop、C2（盆地翻转→按盆地敏感性既有特性论，PASS+A5-A7 逐位不变+双跑一致）。
- **回归**：fc1 交付 10/10；fc2 37/38（1 项 left_skew 数据缺席为战役前既有）；PJC 算子 29/29（含 4 个新增解析雅可比对拍用例）。
- **C3 全面审计**曾判 NO-GO（fc1 9/10：streamIO 翻转后 opt_bench 装载缺口）→ `a634995` 修复恢复 10/10。

## 3. 三轮代理分析记录（谁发现了什么）

| 轮 | 代理 | 关键产出 |
|---|---|---|
| R 系列（前置） | R1-R6b | f7Threads bug、CMTT 双流、denoise 缓存、λDecay=0.80 否决（cond 恶化 1000×） |
| 第一轮 | A1 相机 | TermCriteria 病根；E2 翻案论证；半分辨率复活条件 |
| | A2 激光 F7 | Phase B 隐藏成本 75%；解析雅可比 C 列推导；任务间共享否决 |
| | A3 主解 | relDrop 喂饱机制定量推导；IRLS 死代码；α/γ 全式；snap-to-floor 变体 |
| 第二轮 | B1 | 单线解同病；探针冗余降噪；P0-2 复用 |
| | B2 | 流式读图字节一致论证；批次六三层门禁设计（G1 守卫/G2 MAD/G3 复检）＋有效集等价构造性证明 |
| | B3 | **解析雅可比数学审核：抓 A3 反号＋A2 漏项**；实施规格；验证门 |
| 第三轮 | C1 | PNG 串行/sha256 标量两瓶颈；CUDA 初始化时机 |
| | C2 | 试探评估占比 80-90%；按线并行方案；F7 超订阅豁免论证 |
| | C3 | 放行审计（抓 streamIO 测试缺口）；双跑确定性取证；NO-GO→修复→GO |

## 4. 角点提取可行性分析结论（2026-11-15 实测，未改产品）

- **标志消融**：去 EXHAUSTIVE 丢 41 帧（88/143）；去 NORMALIZE 多救 11 帧（140/143，集合变化）；且各自仅省 7-12%（噪声内）——**标志调优证死**。
- **arm3 半分辨率复测**（12-worker+5px 守卫+失败回退）：角点段 25.2→**12.2s**；角点差 max 0.0003px；下游 K 1.8e-7/rms 1.3e-5/极线 1.7e-6。生产版加 G2/G3 门禁＋第二遍内参后净 ~14-15s，全流程 →~19s。
- **阻塞**：项目算子类门 K≤1e-8/rms≤1e-6 与角点来源变化的物理极限（~1e-7/1e-5）冲突，**需人工裁定 B' 类口径**后方可实施（设计完备：B2 报告 §2.3，回滚键 `corner.halfRes=false`）。
- CUDA 移植 SB 稠密段（→~3-5s）为后备大工程（周级＋等价性难证）。

## 5. 回滚手册

| 优化 | 回滚方式 |
|---|---|
| 解析雅可比 | 数据集 config `"pjc":{"analyticJacobians":false}`（或主库同键） |
| C2 并行评估 | `"pjc":{"jacThreads":1}`（同时回退 jac 并行） |
| F7 禁 Phase B | 代码回滚 `2e99155`（无运行时开关——语义变更） |
| relDrop/IRLS | 代码回滚 `7f07cbe`/`04e15b7` |
| 流式读图 | 主库 `corner.streamIO=false` |
| 内参并行 | 代码回滚 `407e0cf` |
| TermCriteria | 代码回滚 `005af43` |
| PNG 并行/BCrypt | 代码回滚 `704d333` |
| denoise 缓存/CMTT 双流 | `4199feb`（cmttThreads=1 回单流） |

## 6. 遗留与债务（只记录）

1. **09 侧同步债务扩大**：PJC 系改动（relDrop/IRLS/phaseAT/denoise 缓存/解析雅可比/并行评估）未镜像 `modules/09_operatorlib`，镜像需人工安排。
2. **批次六（相机半分辨率）**：设计＋实测证据齐备，等验收口径人工裁定。
3. R5a 发现的既有问题：jacThreads=0 孤立场景跑间非确定性（管线内未复现）；CESHI261003-1/laser1003-2 退化集同二进制跑间差。
4. 根目录 `AGENTS.md`"factory_calib AI 只读"表述过时（子目录 AGENTS 2026-10-14 已废止），建议人工更新。
5. 激光下一瓶颈=CMTT 3.7s（GPU 生成天花板＋sidecar 落盘 447MB）；相机下一瓶颈=角点 18-25s（批次六）。

## 7. 本机计时噪声备注

单次 wall 波动 ±20%（激光 10.2↔16.8s、相机 22.4↔30.7s 均出现过）。重要结论均以阶段日志时间戳差＋多轮复跑交叉验证，勿以单次 wall 比较下结论。

## 8. 装机基线换代——调查结论与 SOP（2026-11-15 执行"第三项"的落账）

### 8.1 候选基线核验（本战役产物 `data_out/laser261006-1668_c1c2/`）

| 项 | 值 | 与产线终态口径对照 |
|---|---|---|
| verdict / framesOk / accum | PASS / 23 / 226,271（dedup-on） | ✓ |
| rms / projectorT | 0.085256 / (39.8679, 8.1292, 10.6062) | 历史主导盆地 |
| sidecar | **185.0 MB**，sha256 `e6c953e7…`，与 JSON 配对 ✓ | 条目 89,805,206（装机版 87,387,013，+2.8% ↔ 尺寸 +2.7% 自洽） |
| 容器口径 | 10 档 4°C（14.5~50.5°C）/rowStep 0.67/**深度窗 [200,500]**/gridHash `0x26fe285a…`（与装机版逐字段一致）/ROI 同/clamped=4/crc 全 OK | ✓ 0.67 格＋[200,500]＋10 档终态定稿 |
| 打包密度 | 2.16 B/entry（与装机版完全一致） | ✓ |

### 8.2 阻塞链（为何 root 装机位现在不能换）

- **装机位现状**：`config\laser_calib.json`＝2026-08-30 的 6.2-e-pjc 三代前古董（rms 0.4479，内联 planeMapTempTable），`config\calibration.json` 同期——版本化在 git。
- **app 消费链**（实证）：`AppContext` → `CalibrationRepository::load("config/calibration.json")` → 工厂档适配＋**同目录 laser_calib.json 合并**（`mergeFactoryLaser`，CalibrationRepository.cpp:148-161）。
- **阻塞点**：`mergeFactoryLaser` 只认**旧版内联表**键（`tempTable`/`planeMapTempTable` 的 `table` 数组）；6.3-cmtt v2 档案的 `curveMapTempTable`（meta+sidecar）它**不读**→ 换上后合并结果为空 → readyForScan 如实报"激光档表缺"→ 扫描链断供（比古董更糟）。
- **解锁前置**＝主工程 v3 接线战役（挂账项二）：06/07 接 CMTT sidecar 装载（09 侧 `LoadTempTableFromCmtt` 现成）。**顺序必须是：先接线、后换代。**
- 09 TEST ×2 `date_in\laser-mark1688-1006` 的 2026-10-13 基线对（dedup-off, rms 0.08728, sha `b6e2694f…`）**未被 git 跟踪**且是 09 侧在飞优化实验（fullgpu/speed_optimize，归档至 2026-11-14）的输入——换代需该战役方确认，勿单方面覆写。

### 8.3 换代 SOP（v3 接线落地后执行）

```powershell
# 1. 用本战役二进制重出正式产物（确定性：双跑 SHA 一致）
laser_calib.exe test_data\laser261006-1668 <staging>\laser_calib.json
# 2. 成对校验（JSON meta.sha256 == Get-FileHash(bin)；fcstepdump --cmtt 全档 crc OK）
# 3. 装机位成对换代（先 git mv 备份旧对）
#    config\laser_calib.json + config\curve_map_temp_table.bin（sidecar 放 JSON 同目录）
#    ⚠ config\calibration.json（相机档）若也要换代须与 camera_calib 产物同批
# 4. 09 TEST date_in 两处随 owning 战役同步换代
# 5. 回归：主工程 06/07 消费链测试＋fc 全量
# 6. git commit 成对换代
```

### 8.4 ⚠ 风险提示（需人工知悉）

**装机位仍停在 8/30 三代前基线**（6.2-e-pjc / rms 0.4479 / 旧内联表）——在 v3 接线＋换代完成前，主工程扫描链消费的是古董标定。若近期有装机/交付需求，请优先排期 v3 接线战役。
