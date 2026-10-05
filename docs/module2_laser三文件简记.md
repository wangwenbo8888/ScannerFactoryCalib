# module2_laser 三文件简记

> 2026-09-01 由人工指令生成：简洁记录 `module2_laser/` 下 3 个非算子源文件内容（算子本体在 `operators/`，此处不涉及）。
> 2026-09-04 对照代码更新：温度窗统一 61 档、handoff 温表解析、CLI 6.3-cmtt（4-14 CMTT 替换 4-13、F7 并发）。

## calib_io.h / calib_io.cpp — 配置与 IO（命名空间 `fc`）

**职责**：模块2 的 config 解析、模块1 交接文件（handoff）解析校验、pose 数据目录扫描、JSON 写出。

- `LaserCalibConfig`：`laser/config.json` 解析结果（字段对齐各算子 Params）
  - plane_map：gridStep=0.5 / depthMin=100 / depthMax=5000 / depthSamples=200 / epipolarStep=0.5
  - deviceId=0；lineIds 缺省空（运行期由实际出现的线号决定）
  - 温度五字段缺省与 handoff 一致（**2026-09-02 统一：±15℃ / 0.5 步距 = 61 档**，对齐 CMTT），config 可覆盖
  - rectifyAlpha=0 / rectifyFlags=1；`laser_label.scanDirection`（0=竖切中心列水平多线，1=横切中心行斜线）＋ centerRowOffset
  - `f7Threads`（顶层键）：F7 漂移解并发线程数；缺省 0=自动 min(逻辑核−2, 8)；1=串行回退；键值 ≤0 解析时归一为 1
  - `cmttThreads`（顶层键）：CMTT 生成器档级并行线程数；缺省 0=自动 min(硬件线程−2, 16)；>0=精确值；≤−1 由生成器视为 1（任意取值产物逐字节一致）
  - `fromJson` 兼容嵌套 `{"plane_map":{...}}`（优先）与顶层平铺两种写法；文件缺失仅 warn 用默认值
- `CameraCalibHandoff`：解析模块1 输出的 `camera_calib.json`
  - K/D（L/R）、R/T、R1/R2/P1/P2/Q、imageSize ＋ 顶层温度五字段＋schema
  - `rectifyTempTable`：**模块1 `stereoRectifyTempTable` 整表（CMTT 4-14 源①）**，容错解析——档内键缺失/形状不符弃该档继续（R1/R2 3×3、P1/P2 3×4、Q 4×4 形状契约）；节缺失/全弃 → `haveRectifyTempTable=false` 仅 warn 不报错（容忍模块1 老输出）；有效则升序排序＋参考温表示层校正（0.2 步距网格浮点累计误差，阈值 1e-6，不触补偿参数）
  - `loadCameraCalibHandoff()`：JSON 解析异常捕获；intrinsic 缺时可从 extrinsic.camera_matrix_l/r 兜底；**必需矩阵（K_L/R、R、T、R1/R2/P1/P2/Q）或 imageSize 缺失/非法 → nullopt**（review I3：imageSize 缺会让 4-14 curve_map_temp_table 构造抛异常）
- `validateHandoffConsistency()`：
  - cte / tempRangeMin/Max / tempStep / referenceTemp 五项必须与 config 一致（1e-9 容差；review I2：referenceTemp 锚定所有温度表）
  - **rectifyTempTable.referenceTemp 三方一致性**（config ↔ handoff ↔ 温表锚点）不一致 → false
  - 温窗覆盖：表首末档应覆盖 [ref−15, ref+15]；不足不 fail，运行期 clamped 兜底（对齐主工程偏差 B 语义）——仅 warn
- `PoseFrame`（L/R 灰度图）/ `LaserInput`（config＋handoff＋poseDirs＋poseFrames[pose][tube]）
- `loadLaserInput(dir)`：
  1. 读 config.json → 读 handoff → 一致性校验（不过则 nullopt）
  2. 扫 `pose_*` 子目录按名排序（可复现）；无 pose 目录 → nullopt
  3. 每个 pose 内匹配 `L_tube*.png`（接受 L/l 前缀、png/jpg）配对同名 `R_tube*`（缺右图/读失败跳帧 warn）；空 pose 跳过
  4. 全空 → nullopt；否则 info 打印 pose 数与 tube 总数
- `writeJson()`：缩进 2 写出（与模块1 相同）

## laser_calib_cli.cpp — CLI 入口（laser_calib.exe，build 6.3-cmtt）

**用法**：`laser_calib <input_dir> [output_json]`（缺省 `laser_calib.json`；参数不足返回 2）。main 顶层 try/catch 把算子异常转为 error+exit 1（防 0xC0000005 崩溃，review C1/I3 防御）。内嵌自包含 SHA-256（自 07 LaserChain 逐字拷贝；4-14 sidecar 文件级哈希用）。

主流程：

1. **加载输入＋一致性校验**（loadLaserInput＋validateHandoffConsistency）
2. **构造算子**（L/R 各一独立实例，单 `cv::cuda::Stream`）：
   - 4-1 MaskExtractCUDA ×2：t=50/e=1/dilate=19/postErode=13（2026-08-25/26 六轮扫参定稿：22/22 姿态满 25 线、~2.1 万点/帧、PJC rms 0.44px）
   - 4-2 RegionAnalyzerCUDA ×2：minArea=0 / topX=27
   - 4-3 LaserLabelerCUDA ×2：scanDirection/centerRowOffset 来自 config、tol=10
   - 4-4 StegerExtractorFast ×2：**2026-09-04 起产线**（与原版逐位等价——容量越界修复后两链全流程逐字节一致；行/列扇出 kernel＋计数直写分组提速）；默认 sigma/threshold
   - 4-5 UndistortPointsCuda ×2：K/D/R1(2)/P1(2) 来自 handoff；**P 只取 3×3（清零 P(0,3)）**——否则右路点被视差基准项平移 ~-17 万像素
   - 4-6 EpipolarInterpDualCuda ×2：**2026-09-04 起产线**，`mode=Labeled`（按 line_id 同线插值、线号透传，与原版 lineIdCheck=true 逐位等价）
   - 4-7 LaserMatchCuda（单实例吃 L+R）：`max_disparity = |f·Tx|/depthMin×1.05` 从 Q 推导（**2026-09-04 双修复**：①哈希→排序＋二分确定性重写（字典序代表点，调度/顺序双无关）；②allocateBuffers 容量越界修复（缓冲统一按 maxCount——原策略单侧超配即越界踩显存产出"幽灵线号"点，原哈希实现同病）。修复后原版链与 fast/dual 链逐字节一致；09 侧同步债务见 AGENTS #9）
   - 4-8 LaserReconstructCuda（单实例）：depthMin/Max 来自 config，Q 按调用传入
   - PJC `ProjectorJointCalib`（替代旧 4-9/4-10/4-11 端点链路；默认参数——useBlockJtJ=true 块稀疏 GN）；stereoK=P1 左上 3×3、stereoR=R
   - 5-3 LaserExtrinsicCompensateCPU（温度参数）
3. **主循环 pose × tube 跑 4-1→4-8**：任一步 success=false 或中间产物 null → warn＋跳帧（framesSkip）；成功则 download 3D 点＋线号，按姿态累积 `PosePointSet`；循环后若设 `FC_PLY_DUMP=<path.ply>` 可导出累积三维点云（左相机矫正系，含 pose_idx/line_id 标量，2026-09-04 增）；3b 丢弃空姿态组
4. **3c PJC 多线联合**（模型 t(3)＋每线 6 参发射曲线=9DOF，R=I、K=f+主点固定）：
   - 重组：按姿态分组 → 按线分组（每线跨姿态）；初值 tMech=R1·(80,3,3)
   - 阶段1 逐线共识 median（取 rms<1 且 cond<1e15 的线，≥3 线取中值）
   - 阶段1.5 z-lift 盆地探测（共识点 vs z+60 短跑 15 迭代，低 rms 胜）
   - 阶段2 完整多线联合＋自动阶梯（L0 默认 → L1/L2 huberDelta 0.7/0.5 收核），A1-A3 验收（maxLineRms≤0.5px / maxP95≤1.5px / outlierRatio≤0.02）
   - **F7 深度验收（2026-09-03 起并发化）**：A5 剔线（隔 3 抽 9 线，≤1mm）/ A6 剔姿态（隔 2 抽 11 姿态，≤2mm）/ A7 basin 散布（±3mm 8 邻点重启，≤0.5mm）共 28 个短程重优化任务入单池，LPT 动态自调度（A7 长任务先入队）；`f7Threads` 配置（0=自动硬顶 8 线程 / 1=纯串行回退）；确定性＝每任务独立算子实例＋按值拷贝输入＋max 可交换聚合；护栏＝任务体整体 try/catch（异常只计该任务失败）＋**失败哨兵 (1e9,0,0) 计数进 `solveFailures`（任一 drift 解失败强制 DEGRADED，堵假 PASS）**；verdict=PASS|DEGRADED
   - 产出 finalVirtualK/R/T（矫正系语义）＋ pjcJson（projectorT/25 曲线/epipolarRowStep/acceptance/perLine 诊断）＋ emissionCurves（提升供 3d 4-14 消费）
5. **3d 温度表**（依赖 PJC 成功）：
   - 5-3：v2l=R=I,T=projectorT；v2r 链式复合 `R_v2r=stereoR·R_v2l`、`T_v2r=stereoR·T_v2l+T_stereo` → laserExtrinsicTempTable
   - **4-14 CurveMapTempTableGenerator（2026-09-03 替换退役的 4-13 PlaneMapTempTable，CMTT Task 3C）**：三源＝模块1 rectifyTempTable（handoff 整表）＋ 5-3 laserExtrinTable ＋ PJC emissionCurves；61 档（±15℃/0.5）；rowStep=cfg.epipolarStep（禁硬编码红线）、depth 窗全档统一、tierThreads=cfg.cmttThreads。流程：生成 → CmttReader 复核（不过不留文件）→ 落盘 `{out 父目录}/curve_map_temp_table.bin` → 对落盘文件流式 sha256 → meta 10 键（success/message/path/tempBase/tempStep/tierCount/tierCountOk/clampedRatio/gridHash/sha256，键名照 07 CalibSerialize）写 JSON `curveMapTempTable` 节。红线：**参考档 FAIL 整体不出产物**；写失败/hash 重读失败 best-effort 删半文件、haveCurveMapTable=false。前置缺失（无 rectifyTempTable / 曲线空 / 无 5-3 表）→ error log 跳过不出产物（07 先例语义）
6. **显式 Destroy 全部算子**（规范要求）
7. **写 laser_calib.json**：schema `factory_calib.laser_calib.v2` / build `6.3-cmtt` / status `ok|partial` ＋ posesProcessed / framesOk/Skipped / accumulatedPoints3D / haveVirtualPose·LaserExtrin·**CurveMapTable** ＋ virtualK/R/T ＋ pjc ＋ laserExtrinsicTempTable ＋ **curveMapTempTable**（meta；sidecar 本体不进 JSON）
   - **退出码**：0=完整成功；1=partial（三 have 任一 false，含写失败/异常/入参外）；单帧跳帧不影响
