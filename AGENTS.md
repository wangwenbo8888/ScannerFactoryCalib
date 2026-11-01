# factory_calib — 厂家标定独立子工程

本文档由原 `AGENTS.md`（AI 访问规则）与原 `README.md`（工程说明）于 2026-08-25 合并而成；2026-08-26 增补：curve_map / laser_match_scan 算子、fcscan_e2e 工具、left_skew 斜线形态支持、输出 schema 更新。2026-09-01 增补：laser_match_scan_v3 算子（段级投票+冲突双弃）、epipolar_interp_dual 段 ID 输出、fcchain_v3 全链工具、test_pose06_lid_vote 方案验证测试。同日多代理复审校正：吸收 2026-08-28 CCL topX GPU 化（commit `b620d91`）；补全 operators/ 清单（steger_fast · epipolar_interp_dual/_opt · laser_match_scan_v2 · common）与 tools/ 基准（--mask-bench / --interp-bench）；补记 2026-08-29/30 新增 5 个诊断/可视化测试；修正 2026-07-18 悬空引用、模块1 Debug 可构建性、退出码语义与 JSON 键名记法；目录图算子与测试清单改为逐行展开＋职责简介。2026-09-04 增补：CMTT（curve_map_temp_table 61 档 sidecar，4-14 替换退役 4-13，一期 `575323b`/二期 `2efa0aa`/档级并行 `17996dd`）、模块2 拷入 stereo_rectify_temp_table（handoff 源①）、温度表统一 ±15℃/0.5 步距 61 档（`87c8f7f`）、PJC 三次演进（线内雅可比/块稀疏 GN/相位计时）＋F7 并发（`240ff2a`）＋假 PASS 修复（`dedec91`）、模块2 OpenMP 链接（`dedec91`）、tools sidecar 选档注入、输出 schema v2、test_pose06_chain_v3_viz、docs/ 参考卡同步（新增 curve_map_temp_table 卡）；同日退役/闲置算子分两批移入 `operators/retired/`（batch1：endpoint_extract·virtual_camera_pose·pose_optimize·plane_map·plane_map_temp_table；batch2 按人工指令：epipolar_interp·epipolar_interp_opt·laser_match_scan·laser_match_scan_v2——其中 epipolar_interp（标定链 4-6）与 laser_match_scan（fcscan_e2e）仍在用，语义见目录图 ⚠ 注记；git mv 保史，CMake GLOB_RECURSE 自动收录，构建零改动）；同日 CLI 4-6 由 EpipolarInterpCuda 切换 EpipolarInterpDualCuda（mode=Labeled，复刻 opt 内核与原版 bit 级一致）——原版 Execute 消费方仅剩 fcscan_e2e/fcstepdump，头文件因 dual include Result 类型仍不可删；同日 steger 解耦归档：GroupMode/StegerParams/StegerResult 抽至 common/steger_types.h（定义逐字未动，09 侧同步债务）→ steger_fast 自此不依赖原版目录；**同日 4-4 切 steger_fast＋4-6 切 dual·Labeled 后经 left_skew 22 姿态真实数据全流程验证被证伪**（rms 0.45→1.16/2.57、verdict DEGRADED、两次运行自身不一致），**当日回滚**（回滚后 PASS/rms 0.4479 恢复，证据 data_out/e2e_fast/）——合成等价性单测不足以背书真实数据切换；同日根因定位至 laser_match 并行哈希＋**allocateBuffers 容量越界**（原哈希实现同病，越界踩显存产出"幽灵线号"点＝一切环境依赖之谜）并**双修复根治**（排序＋二进字典序代表点；缓冲按 maxCount）——修复后原版链与 fast/dual 链**全流程逐字节一致**（accum 464143/rms 1.1053/PASS/sha 00c90b6d…），fast/dual 无罪；**同日产线切换加速链**：4-4→StegerExtractorFast、4-6→EpipolarInterpDualCuda(Labeled)，切换后验证与等价基线逐字节一致、全量 35/35 绿。同日文档收口：L6 行补全「首轮切换 34/34→真实数据证伪回滚→根因修复→复切 35/35 现役」终态；测试可执行计数 42→43（约 558 cases，含当日新增 frontend_ab_compare）。同日 test_pose06_chain_v3_viz 表源切换（人工指令）：curve_map 现场生成→`data_in/left_skew/curve_map_temp_table.bin`（2.36GB CMTT sidecar）selectCmttTier 按温选档（参考温 22.5 档，rowStep 取自表内 rowStep()，温度注入＝档温）——数据自包含，去 data_out 标定 JSON 依赖；实测 PASS：matched=cloudPts 24245、tracks 64 弃 17、TOTAL 37.1ms（新基线 sidecar 表拟合优于旧 on-the-fly 单档基线 14004，门禁 ≥8000 仍宽裕）。2026-10-07/08 增补（laser261006-1668 事故修复＋三道防线＋CMTT CUDA 转正）：①过曝粘连编号错位根因闭环（高光融合→split 凑数→编号错位毒化 PJC，A7 盆地发散实证）；②laser_label 双门禁 `expectedLineCount`(25)/`maxRunWidth`(30) 主库激活、CLI 4-3 帧级跳过；③mask_extract 高光分离路径 `highlightErodeSize/highlightDilateSize`（0=off 兼容，数据集层启用）；④steger_fast 逐点宽度门禁 `maxLineWidth`（0=off 与原版逐位一致；平滑后口径 w=sqrt(g/|λ|)，数据集定稿 6.0）——三防线叠加后该数据集 verdict=PASS（23 帧/43.4 万点/A7=0.03mm）；⑤CMTT CUDA 转正：`operators/curve_map_cuda`（单档 GPU 求交，专用库 fc2_curve_map_cuda 带 -fmad=false，从 fc2_ops 通配排除）＋`operators/cmtt_cuda`（61 档编排主引擎，CLI 4-14 默认；`cmtt.cudaEnabled=false` 切回 CPU 备用 curve_map_temp_table）；61 档 24.8min→113s，真实参考档与 CPU 版逐条目一致（27,538,991==27,538,991），全链 31.4min→3.5min；⑥fcstepdump mask 参数改走三层合并＋新增 `_1binary` 二值化可视化（4-1 空掩膜检查前落盘）；⑦数据集定稿 `test_data/laser261006-1668/config.json`（120/3/9＋门禁 6.0；**postErodeSize/highlightErodeSize 显式置 0 不可省**——编译默认 post=13 会净腐蚀灭线，2026-10-07 固化事故）；rowStep 口径勘正：CLI 用 `cfg.epipolarStep`（主库 plane_map 组=0.5）非算子默认 0.7。测试可执行 43→44（约 600 cases）。2026-10-09 增补（curve_map_cuda 基线景深参数化＋tests/baseline_depth 黄金测试，设计/实施文档 `docs/plans/2026-10-09-curve-map-cuda-baseline-depth-*.md`）：①**基线景深概念定稿**＝矫正系 z 坐标（平面1=两光轴所在平面 y=0；景深面⊥平面1 且∥基线⟺z=const；到基线投影距离=z）——与现行 depthMin/Max 过滤口径同一（kernel `Z=(qc·tz−f·tx)/(q_u−uL)` 推导见设计文档 §2），无数值行为变化；②`CurveMapCuda` 只加不改新增 `SetBaselineDepthWindow(d1,d2)/GetBaselineDepthWindow()`（isfinite 前置堵 NaN 静默放行＋先窄化 float 后校验＋临时对象先验后交，违例抛 invalid_argument 且参数原值不动）——CLI/cmtt_cuda/共享头 curve_map.h 零改动（附带规避 09 侧同步债务）；③**tests/ 首个一级子文件夹测试** `tests/baseline_depth/test_curve_map_cuda_baseline_depth.cpp`（6 用例全绿：A1 窗内一致（全条目 Z′∈[d1−δ₁,d2+δ₂] δ 精确式现算）/A2 边界＋空窗（0 条目→Warning 且产物合法）/A3 非法窗口拒跑＋异常后参数不被半改/A4 CPU↔CUDA 同窗逐条目精确一致（直线 conic 构造性等价论证）/A5 别名逐字节等价/真数据 laser261006-1668 窗 100–500（25 线全有＋逐线 Z 界＋rowStep=0.5 冻结回显；golden 式根探测缺席 SKIP；`FC_BASELINE_DEPTH_SIDECAR=1` 可选与交付 sidecar 参考档 entryCount 对拍）；④CMake 测试注册新增 elseif 收录 `tests/<子文件夹>/test_*.cpp`（既有平铺/operator 侧测试路由不变，实证零波及）；⑤**已知分歧与既有隐患记录不修**（设计文档 §6）：CPU (uL,d) 逐位相等去重 vs CUDA sort 不去重（触发面极窄，直线 conic 构造绕开）；makeKey 位域不设防（uLq 12bit→W·invRowStep≤4095、bucket 16bit→nRows·nLines≤65535，现 ROI 43925 安全）；kernel 异常路径 thrust::copy 抛时不释放 5 裸指针；⑥同日激光标定复跑（定稿参数）验证 PASS 与基线逐项一致（23 帧/434,228 点/rms 0.0854/A7=0.03mm）＝新 API 零影响实证；深度窗 100–500 变体实测 verdict 仍 PASS（18 poses/229,350 点/A7=0.10mm，产物 `data_out/laser261006-1668_depth100-500/`）。⑦**产线接线（同日追加，人工指令）**：`cmtt_cuda.cpp` 构造处由直写 `cmp.depthMin/Max` 改为 `gen.SetBaselineDepthWindow(params_.depthMin/Max)`（语义显式化"4-14 按基线景深窗逐档生成"，行为逐位不变）——实证：laser_calib 定稿复跑输出 JSON 与接线前 **FC 字节级一致**（含 sidecar sha256→3.34GB 温表亦逐字节同）；test_cmtt_cuda 3 PASS＋1 env SKIP（真数据参考档 vs 交付 sidecar 对拍过）。测试可执行 44→45（约 606 cases）。2026-10-13 增补（格距统一 0.67·矫正缩放比口径，人工指令）：①主库 `plane_map.epipolarStep` 0.5→**0.67**（＝round2(P1焦距 1231.1388/原始左目 fx 1839.6028)＝0.66924→0.67，本机型矫正缩放比——0.67 格≈1.046 原像素＝原生采样密度，0.5 为 1.34× 过采样；键位预算 uLq 2519/4095、bucket 32775/65535 ✓；换机型按同公式重算）；②方案取舍：运行时派生方案（哨兵+resolve+预算校验）经三路代理审核后弃用——单机型现实下写死更简更稳（审核抓出的 bucket ROI 域错误等 5 必修项全为派生机器付税，详见 docs/plans 无存档、会话记录）；③产物 0.67 再生成：`data_out/laser261006-1668`（sidecar 2.67GB/参考档 21,951,874 条，原 3.34GB）＋`_depth100-500` 变体（1.81GB/14,827,900），PJC/accum 逐值不变（434,228/229,350；cfg.epipolarStep 消费点 cli:855/1159 grep 实证不进标定链）；④测试断言同批：baseline_depth :504→0.67f、:533-534→9M~20M（PJC 双盆地摆幅 ±40% 兼容：现盆 0.0846/旧盆 0.0854，本机三次复跑稳定落现盆）；test_cmtt_cuda :323/:362/:388→0.67f；ctest 2/2 绿；⑤主工程：scan_chain_replay 配置 0.5→0.67×2＋新表落地→gtest **14/14 绿**（matched 2.1~2.4 万/帧、rowResid=0、确定性逐位一致）；装机基线 `config/laser_calib.json`（6.3-cmtt/PASS）＋`curve_map_temp_table.bin` 成对换代（原为 6.2-e-pjc 三代前古董）；app/CMakeLists POST_BUILD 补 sidecar 播种（if-not-exist 同 json 语义）；⑥**steger_dedup_replay 0.67 迁移实测失败已回退**：其实验门 P5/P6b/P7 行计数常数＋分簇参数系按 0.7 表/旧盆联合调定（关实验门 laser3D 中位 15930 vs 原 27661；开实验门 10030、段 banned 28/53）→ 回退 0.7 冻结配对 **16/16 绿**继续服役；~~迁移待办：需专门复调参会话~~（**同日晚间已收口**：真因定位＝深度窗不匹配而非格子——换窄窗 [200,500] 10 档 0.67 表（testdate 表源＝cmtt_4deg_10tiers_dw200_500 产物，用户人工放入）后配置对齐 0.67，全量 14 帧 gtest **16/16 绿且无需调任何工作点参数**：laser3D 中位 26,011（阈值 22,000 直接过，≈0.7 时代 27,661 的 94%）、P7 banned=0、去重对齐 opMerged=cpuMerged=25,319 逐位同、rowResid=0——dedup 完成 0.67 迁移，此前"需复调参"结论系深度窗误导，撤回）；⑦06/07 断供决策：A（新产物保留旧 planeMapTempTable 键）**否决**——旧键系退役算法产物且现装机基线该键本就空载无人消费；B（主工程接 CMTT 装载）归入主工程 v3 接线战役（09 侧 LoadTempTableFromCmtt 现成）。⑧分析结论存档：CMTT 相邻档差异（键配对口径）＝左图点构造冻结不动、右图每 +0.5°C 右移 0.0265px（视差 −9.3e-5/°C，温度项占 ±6px 查表窗 <6%）——0.5°C 档距余量过大、2°C 更合理（未改）。同日第二批（非对称档位支持，人工指令）：①GenParams 新增 `tiersBelow/tiersAbove`（双正时覆盖 tempHalfRange 对称路径；validate 含成对校验；toJson/fromJson/CLI/装载器全链接线，键＝`cmtt.tempStep/tiersBelow/tiersAbove`，`cmtt.tempStep` 原 CLI 硬编码 0.5 外置）；②TDD：test_cmtt_cuda 新增 `SyntheticAsymmetricTiers`（下2/上7/步1.0→10 档/温度序列/参考档下标2/CPU↔CUDA 逐档对拍/sidecar 结构）先红后绿；③laser261006-1668 增 10 档实验（数据集 config `cmtt:{4.0/2/7}`）：产物落 `test_data/laser261006-1668/cmtt_4deg_10tiers/`——10 档 14.5~50.5°C（4°C 步距）、verdict=PASS、accum 434,228 逐值不变、每档 ~2195 万条、sidecar 447MB（61 档的 17%）、4-14 生成 12.3s；⚠ 上方 4 档（38.5~50.5°C）超两源表覆盖 ±15°C → 落 clamped 边缘档（clampedRatio 0.4 触发 warn，条目=37.5°C 边缘复制）——**若需真实外推须先扩模块1/5-3 源表温区**。同日第三批（终态统一，人工指令）：产线终态定稿＝**0.67 格＋深度窗 [200,500]＋10 档（4.0°C/下2上7，14.5~50.5°C）**——①主库 cmtt 组定稿 step=4.0/tiersBelow=2/tiersAbove=7（所有数据集重跑默认出 10 档）；②装机基线 config\laser_calib.json＋sidecar 换 10 档 180MB 版（tierCount=10/PASS/accum 229,350）；③scan_chain_replay 换 10 档窄窗表→全量 gtest **14/14 绿且阈值零重钉**（laser3D 中位 23,398＞阈 16,000；窗收窄点数 −9% 为 [100,5000]→[200,500] 砍远尾预测所致，与 dedup 同型）。至此全链（主库→标定产物→测试基线→装机基线→双回放）口径一致；唯一遗留＝主工程 v3 接线接 CMTT 装载（B 方案，等主工程战役）。2026-10-12 增补（laser_calib 顺序帧输入支持，人工指令）：`loadLaserInput` 在 `pose_*` 子目录缺席时**自动回退**扫 `<dir>/left/`+`<dir>/right/` 同名帧配对（.png/.jpg，按名排序；缺右对/读失败 warn+skip；每对同名帧＝1 pose×1 tube，对齐 DUIBI1005-1 人工转换惯例；pose_* 与 left/ 并存时 pose_* 优先，既有数据集零影响）；新增 `tests/test_load_laser_input.cpp` 4 cases（顺序帧回退/缺右对跳过/pose 优先/无布局失败，TDD 先红后绿）＋calib_io.h/CLI 用法文本同步；实跑 laser261006-1668 直输验证：`loaded 29 poses (29 tubes)`，29/23 ok/6 skipped/accum 434,228/verdict=PASS——与 2026-10-09 基线逐项一致（产物 `data_out/laser261006-1668_rerun/`，sidecar 5.12GB；数据集目录未动）。设计/实施文档 `docs/plans/2026-10-12-laser-calib-sequential-input(-design).md`。2026-10-13 增补（模块1 opt_bench 优化实验场＋批次一内参 LU 合入，全程人工指令）：①`module1_camera/tests/opt_bench/` 新建优化实验场（模块1 测试可执行 8→17）：基准/逐实验进程内 A/B 对拍，验收双轨制（流程类逐字节一致·算子类数值门 K≤1e-8/畸变≤1e-7/rms≤1e-6/下游≤1e-3）；共享件强制 header-only（tests/ 下非 test_ 前缀 .cpp 会被 CMake 路由静默丢弃）；基准数据集人工裁定＝`test_data/camera261006-1688`（无 config.json 直吃主库参数），`data_in/CESHI260831` 人工裁定不在合入范围。②基准实测：内参标定 515s 占全流程 85%（根因＝每轮 LM 对 ~780×780 阻尼正规方程做稠密 SVD＋默认终止条件恒满 30 轮；OpenCV 内部多线程仅 1.2x，3+3 ABBA 换序取中位口径）、角点 78s、外参 40s。③实验结论：E1 内参 `CALIB_USE_LU` **92.4x**（578→6.3s，K 相对差 7e-14）；E6 外参同款 **67.7x**（49→0.72s，直调 stereoCalibrate 原型，R/T 差 3e-4 倍容差）——待批次三（算子一行 `extrinsic_calib_cpu.cpp:558` ＋09 侧同步，两文件当前零差异已核实）；S1 并行角点 **2.70x** 逐点零差异——待批次二（CLI 装配层）；A1 坏帧剔除已验证暂缓（自动剔 6 帧＝人工 marked_gates 6/6 重合，全指标改善）；S2 两阶段检测否决（0.74x 反慢）；E2 并行内参不采纳（LU 后仅省 3s）；E3/E4/E5 免做（多代理三路审核驳回 E4：好初值不减轮数）。审核另揭两坑：present-key 覆盖陷阱（含 `intrinsic_flags` 键的数据集 config 会静默盖回主库）与本机计时噪声 ±20%（计时一律 3+3 ABBA 取中位）。④**批次一已合入**：主库＋`test_data/camera/config.json:8`＋`tests/e2e/data_out/config.json:8` 三处 `intrinsic_flags` 0→131072；模块1 全量回归 **17/17 绿**；camera261006-1688 真实复跑 ~630s→**115.8s**（5.4x，剩余大头角点 78s＋外参 SVD 40s 待批次二/三）；新旧输出 delta 全过门且富余 ≥4 个数量级（epiMean 逐位一致、K 差 1e-14 级、rms 差 1e-15 级、畸变差万分之一容差、R/T 差 3e-13）；回滚＝键改回 0 即恢复 SVD。证据链：`tests/opt_bench/out/batch1_adoption_report.json`＋合入提案 `docs/plans/2026-10-13-module1-camera-opt-bench-adoption.md`（含批次二/三改动点与统一放行清单）。2026-10-14 增补（角点加速实验＋批次二合入，全程人工指令）：①角点 78s→10s 攻关：3 设计代理（调度/算法/IO）＋3 审核代理＋2 对抗审核代理三轮回合制——关键发现：OpenCV 有**嵌套并行闸**（parallel.cpp flagNestedParallelFor：任一线程占用并行区期间其他线程 parallel_for 串行内联），此为帧级并行仅 2.70x 真因；纯调度（12w＋setNumThreads(1)）天花板 25s。②`test_opt_corner_10s` 四臂实验全绿：臂三（12w＋半分辨率快检测＋5px 位移守卫＋失败回退）实测 **7.8s**、角点差 ≤0.0003px、下游全指标 1e-5 级过门；但发现**B 类扩集**——半分辨率检测救回 11 个基线漏检帧（129→140）致极线误差劣化 8%——实验内以基线名单强制压制（⊆ 断言）通过，正式程序无名单可用故**快检测方案不合入**，留档实验场。③**批次二已合入（保守并行版）**：CLI 角点循环改 12 worker 帧级并行＋提取期间 setNumThreads(1)（存还原），新增主库组 `corner:{workers:12}`（置 1=回滚）＋`CameraOpParams.cornerWorkers`＋applyCameraParamsJson 解析；camera261006-1688 真实复跑 115.8s→**79.9s**（角点段 78→29s），输出与批次一金标 **SHA256 逐字节一致**；交付回归 10/10 绿（108s；全量 18/18 绿于同库早前跑）。④过程事故两条留档：PowerShell Set-Content 默认 ANSI 编码写坏 exe 旁主库中文注释→JSON 解析失败静默回退编译默认（SVD 慢模式，803s 废数据）——**改主库 json 须 UTF-8 写入或从源码树拷贝**；一次工具假成功（CLI 编辑未落盘）＋陈旧 calib_io.obj 致并行静默未生效（日志告警严格递增=串行铁证，删除 obj 强制重编根治）——**合入验证须查"并行乱序证据"而非只看耗时**。⑤测试分档：交付回归＝10 项（~2 分钟，`ctest -E "^test_opt_|^test_diag_intrinsic_threads"`）；实验全量＝18 项（~50 分钟，按需跑）。⑥**批次三已合入**（同日，人工指令）：外参 stereoCalibrate flags 加 CALIB_USE_LU——factory 侧 `extrinsic_calib_cpu.cpp:558` ＋09 侧镜像同改（fc 复核零差异保持）；camera261006-1688 复跑 79.9s→**42.2s**（外参段 0.78s），delta 与 E6 预言逐项吻合（rmsL/R 逐位一致、stereoReproj 差 3.6e-13、R/T 差万分之三容差）；factory 交付回归 10/10 绿＋主工程分范围 12/12 绿（含 07 test_calib_camera 189.65s——ExtrinsicCalibCpu 消费链）；⚠主工程全量构建曾因 **mask_sep_cli.cpp 既有 WIC 编译错**受阻（f00cef0 带入、从未成功构建、与本批无关）——当会话已修复（四处：IPropertyBag2 笔误/NOMINMAX/core-cuda.hpp/calib 前缀），主工程全量回归 **101/101 绿**（2026-10-14）＝批次三放行闸门完整闭环。三批累计：CLI 全流程 10.6min→**42.2s**。2026-10-13 第四批增补（steger_fast dedup v2 移植回厂，人工指令）：①背景＝09 侧产线镜像（三防线）后继续演进，新增 **dedup v2 平顶双点去重（位移式：网格排序＋邻域 PCA 几何切向＋位移门配对定长邻接行＋pull/shortcut 标签传播＋(簇键,rrid) 排序段归约）**，fc 侧此前缺席（steger_types.h/pimpl.h/impl.cu 三件分叉）。②**整文件覆盖三源码**（operators/common/steger_types.h＋operators/steger_fast/steger_fast_pimpl.h＋steger_fast_impl.cu）→ 覆盖后与 09 侧**逐字节一致**（steger_fast.h/.cpp 本就一致；09 侧 2026-09-04 记的"类型头布局同步债务"随镜像已清）；CUB 随 CUDA 12 自带，fc2_ops **零 CMake 改动**。契约：StegerParams 增 `dedupMode`（0=Off 默认/1=Centroid/2 保留）、StegerResult 增 `mergedCount`；**dedupMode=0 时 Off 路径仅 src_points 间接化，行为逐位一致（红线）**。③CLI 接线仿 maxLineWidth 先例：calib_io.h/cpp 增 `steger.dedupMode`（present-key）、laser_calib_cli.cpp 4-4 注入＋构造日志带出、主库 steger 组增 `dedupMode: 0`（产线默认行为不变，数据集层可启用）。④测试双向对齐（**测试文件两侧为合编、非逐字同源**）：fc 侧合入 09 侧 9 个 Dedup 用例＋helper（DrawFlatTopColumn/FlattenSorted/ExpectIdenticalPointSets/SliceX/MakeFlatTopImage）；BenchDedup 前移至 BenchFullResSynthetic 之后（镜像 09 文件顺序——防长图 IO 用例后 GPU 状态拖累中位计时，实测全量跑假红 3.15ms vs 单跑 2.58ms）；DedupThinLine 构图对齐 09 单行标签（fc 侧 DrawGaussianLine 涂满 dy∈[-2,2] 带宽标签，线端伪双点 merged=2 属构图差异非算子差异）；**VisualizeRealFrames 性能闸改相对门 speedup≥1.8×**（原绝对门 mean≤3ms 仅 ~10% 余量，整机后台负载时 orig/fast 同步慢 2-3× 假红——实测 orig 12-17ms vs 健康 5.7ms；健康 speedup 2.2-2.3×，fast 退化时比值→1 仍可抓）；09 侧反向补齐宽度门禁 6 例（StegerFastWidthGate 套件＋DrawLineSeg/MakeThinPlusWideImage）。⑤验证：fc2 ctest **39/39 绿**（test_steger_fast 25 例＝15 原有＋10 合入）；laser261006-1668 真实复跑（dedupMode=0，产物 `data_out/laser261006-1668_dedupport/`）与装机基线 `config/laser_calib.json` **全字段对拍一致**——PJC finalSampsonRms 逐位同（0.08728393830048103）、basinProbe 同、projectorT/emissionCurves/laserExtrinsicTempTable 节哈希同、CMTT 10 档 sidecar **sha256 逐字节同**（b6e2694f…）；主工程 Debug test_steger_fast 22 过＋2 perf gate 按口径 skip。设计文档 `docs/plans/2026-10-13-steger-dedup-v2-port-design.md`。2026-10-14 增补（steger.dedupMode 默认开启，人工指令）：①主库 `laser_calib_params.json` steger 组 dedupMode **0→1**——与 maxLineWidth 6.0 组成默认双防线（门禁杀"宽"、去重杀"窄但成双"）。②laser261006-1668 复跑（产物 `data_out/laser261006-1668_dedupon/`）：framesOk 23 不降、accum 229,350→226,271（−1.3%）、逐帧合并 1.2k–3.4k（4–10%）、**PJC rms 0.08728→0.08548（−2.1%）**、A5 0.09→0.08/A6 0.15→0.14/A7 0.09 持平、verdict PASS。③left_skew 影子 A/B（`data_in/left_skew_ab/` junction 影子集＋steger.dedupMode 数据集层覆盖；产物 `data_out/left_skew_ab_{off,on}/`）：**OFF 臂即 DEGRADED（A7≈6.97mm）＝既有状况**——2026-10-12 主库对齐 laser261006 定稿后 left_skew 从未按新默认复跑（宽度门禁 6.0/mask 默认对该老数据集不适配，与 dedup 无关；如需复用该数据集须数据集层调参）；dedup 增量效应＝accum −1.8%、合并 42,995、A5 0.85→0.40/A6 1.08→0.54 **减半改善**、A7 同差——无害偏正面。④fc2 ctest **39/39 绿**。⑤遗留：装机基线 `config/laser_calib.json`＋sidecar 仍为 dedup-off 口径（rms 0.08728）——下次产线标定自然出新基线，或人工确认后以 `data_out/laser261006-1668_dedupon/` 产物成对换代。 2026-11-01 增补：261101 更新包全量合入新工程 D:\factory_calib（git 沿用本仓库 1002 分支）——1002 的 GUI 产线文件字节级恢复、laser_calib_cli 库化（runLaserCalibRaw/FC2_BUILDING_LIB）、SB 兜底恢复（58e7991）、基线对拍口径升级（相机全字段容差 1e-9/激光字段级＋sidecar SHA 精确）、三用例环境解耦，全量 ctest 59/59 绿；详见 开发记录.md 2026-11-01 节与 verify_baseline/基线档案.md §5。

两个独立可执行：

- `module1_camera/camera_calib.exe` — 相机内/外参 + 立体矫正 + 温度补偿表（CPU）
- `module2_laser/laser_calib.exe` — 激光虚拟相机标定 + 温度补偿表（CUDA）

历史设计/实施文档（2026-07-18 系列）已于 2026-08-05 归档清理（commit `db73b7b`）；现行权威说明为本文件工程说明部分与 `../docs/流水线/03-厂家标定流水线.md`。

> **2026-10-14 起：本目录对 AI 开放自由访问**（原「AI 只读保护区」整套访问规则经人工指令废止，随之移除）。产线交付物属性不变：精度敏感参数（温度系数/阈值）、与 `../modules/09_operatorlib/` 的「逐字拷贝」同步边界、以及「前端变更须真实数据全流程验证」红线仍按工程惯例执行。

---

# 工程说明

## 工程目录地图

```
factory_calib/
├── CMakeLists.txt             # 自包含构建（FC_BUILD_MODULE1/MODULE2 开关）
├── cmake/                     #   构建辅助（factory_calib_common.cmake：fc_deploy_crt 等）
│
├── module1_camera/            # ── 模块1：相机标定 exe（CPU，无 CUDA）──
│   ├── CMakeLists.txt         #   模块构建（fc1_ops/fc1_chess/fc1_io＋camera_calib＋测试 glob 注册）
│   ├── camera_calib_cli.cpp   #   CLI 入口
│   ├── calib_io.{h,cpp}       #   JSON/图像读写
│   ├── chessboard_corner.*    #   棋盘格角点检测
│   ├── operators/             #   算子库：
│   │   ├── intrinsic_calib/   #     双目内参标定
│   │   ├── extrinsic_calib/   #     双目外参标定
│   │   ├── stereo_rectify/    #     立体矫正
│   │   ├── stereo_rectify_temp_table/  # 温度补偿立体矫正参数表
│   │   ├── intrinsic_compensate/  #   内参温度补偿
│   │   ├── extrinsic_compensate/  #   外参温度补偿
│   │   └── common/            #     共享类型头（非算子）
│   ├── tests/                 #   棋盘格 L2 + 端到端 L3 测试 + e2e/ 自包含单元
│   │   └── opt_bench/         #   优化实验场（2026-10-13，9 测试：基准/S1 并行角点/
│   │                           #   E1 内参 LU/E6 外参 LU/E2/A1 剔帧/S2/线程诊断；
│   │                           #   验收双轨制，README 内详；批次一 LU 已据此合入）
│
├── module2_laser/             # ── 模块2：激光标定 exe（CUDA，CLI 仅 Release）──
│   ├── CMakeLists.txt         #   模块构建（Debug 无 imgcodecs 时自动跳过 CLI＋tools；
│   │                          #   tests/ 一级子文件夹 test_*.cpp 经 elseif 收录——
│   │                          #   tests/baseline_depth/ 为首例，2026-10-09）
│   ├── laser_calib_cli.cpp    #   CLI 入口
│   ├── calib_io.{h,cpp}       #   JSON/图像读写
│   ├── tools/                 #   辅助工具（Release 同批构建）：
│   │                          #     fcstepdump — 逐帧 3D 重建＋全步骤可视化，含子命令
│   │                          #       --curve-map / --r1-audit / --pm-ablation / --analyze-ply
│   │                          #       --pjc-cmos / --pjc-per-line / --pjc-multiline / --pjc-from-ply
│   │                          #       --mask-bench / --interp-bench（基准；实现源 mask_bench.cpp /
│   │                          #       interp_bench.cu 编入 fcstepdump 目标）
│   │                          #       --cmtt（sidecar 容器头转储诊断）/ --cmtt-gen（标定 JSON
│   │                          #       独立生成 sidecar，可附 tierThreads）
│   │                          #     fcscan_e2e — 扫描式端到端：curve_map 表 → laser_match_scan
│   │                          #       匹配 → laser_reconstruct → PLY 点云；表源优先 CMTT sidecar
│   │                          #       按温选档（selectCmttTier，可选第 4 参 temp_c），缺 sidecar
│   │                          #       回落现场生成
│   │                          #     fcchain_v3 — V3 全链工具：mask→CCL→steger_fast→undistort
│   │                          #       →interp_dual(段ID)→laser_match_scan_v3→reconstruct
│   │                          #       输出每步耗时+图片+三维点云（data_out/chainv3/）；
│   │                          #       表源同 fcscan（sidecar 选档优先，第 4 参 temp_c 可选）
│   ├── operators/             #   算子库（15 算子目录＋common＋retired/）：
│   │                          #   ── 前端视觉 ──
│   │   ├── mask_extract/      #     激光掩膜区域提取 [CUDA]（2026-10-07 起含高光分离
│   │                          #     可选路径 highlightErodeSize/highlightDilateSize,
│   │                          #     0=off 走 legacy 单路; 厚度判据剔过曝斑, 数据集层启用）
│   │   ├── ccl/               #     连通域分析（topX 面积收集 GPU 化）
│   │   ├── steger_fast/       #     快速版 Steger——**4-4 产线现役**（2026-09-04 切换;
│   │   │                       #     与原版逐位等价, 容量越界修复后两链全流程逐字节
│   │   │                       #     一致; 全量 35/35 绿）; 类型源 common/steger_types.h;
│   │   │                       #     2026-10-07 增逐点宽度门禁 maxLineWidth(0=off 逐位
│   │   │                       #     一致; 平滑后口径 w=sqrt(g/|λ|); 过宽点=过曝噪点剔除);
│   │   │                       #     2026-10-13 移植 09 侧 dedup v2 平顶双点去重
│   │   │                       #     (dedupMode 0=Off 默认逐位一致/1=Centroid; 三源码
│   │   │                       #     与 09 侧逐字节对齐; CLI 键 steger.dedupMode)
│   │   ├── undistort_cuda/    #     去畸变矫正 [CUDA]
│   │                          #   ── 激光链 ──
│   │   ├── laser_label/       #     激光线编号（scanDirection 0/1 两形态; 2026-10-07 起
│   │                          #     双门禁 expectedLineCount/maxRunWidth——线数不符或
│   │                          #     切割游程展布超阈→frameRejected, CLI 4-3 整帧跳过）
│   │   ├── laser_match/       #     激光线匹配（标定版）
│   │   ├── epipolar_interp_dual/  # 极线插值双目版：Scan 模式段 ID 输出（V3 消费）
│   │   ├── laser_reconstruct/ #     激光线三维重建
│   │                          #   ── 虚拟相机 ──
│   │   ├── projector_joint_calib/ # PJC 联合标定（现用：光心＋25 条发射曲线）
│   │                          #   ── 映射/补偿 ──
│   │   ├── curve_map/         #     极线切曲线映射表（CPU，12b 打包）
│   │   │                       #     2026-09-03 起模块2 链接 OpenMP（对齐 09 侧），curve_map 行级并行生效（线程数无关确定性经三档 sha256 对拍）
│   │   ├── curve_map_temp_table/ # CMTT：curve_map 温度表 61 档 sidecar 生成器
│   │   │                       #     （**4-14 备用引擎**——2026-10-08 CUDA 转正后退居热备, cmtt.cudaEnabled=false
#     切回; 插值器 temp_param_interpolator＋
│   │   │                       #     容器 cmtt_container＋冻结网格断言；2026-09-04 起
│   │   │                       #     档级并行 GenParams.tierThreads/config cmttThreads
│   │   │                       #     可配；线程数无关确定性经单元 1/4/8 与冻结载体
│   │   │                          #     1/8/16 档 sha256 对拍）
│   │   ├── curve_map_cuda/    #     curve_map 单档 GPU 生成器（2026-10-08 转正: 求交+
│   │                          #     thrust 排序上 GPU, 打包/序列化留 CPU 复刻 E.2;
│   │                          #     专用库 fc2_curve_map_cuda 带 -fmad=false 保 CPU
│   │                          #     等价性, 从 fc2_ops 通配排除防波及其余算子旗标）
│   │   ├── cmtt_cuda/         #     CMTT CUDA 版: 61 档编排主引擎（2026-10-08 转正,
│   │                          #     CLI 4-14 默认; 复用 CPU 版插值/重标/冻结断言/
│   │                          #     CmttBuilder 公开 API; 61 档 24.8min→113s, 全链
│   │                          #     31.4min→3.5min; tests 含真实全尺度对拍——参考档
│   │                          #     27,538,991 条与 CPU 版逐条目一致）
│   │   ├── stereo_rectify_temp_table/ # 模块1 同名算子拷入（handoff 整表解析＋
│   │   │                       #     CMTT 4-14 源① 类型载体，2026-09-03）
│   │   ├── laser_extrinsic_compensate/ # 激光外参温度补偿
│   │   ├── extrinsic_compensate/ #    外参温度补偿
│   │                          #   ── 扫描消费（tools 用）──
│   │   ├── laser_match_scan_v3/  # v3：段级投票＋冲突双弃，输出含段 ID [CUDA]
│   │   ├── common/            #     共享类型头（非算子；含 steger_types.h——
│   │   │                       #     GroupMode/StegerParams/StegerResult 唯一驻点，
│   │   │                       #     2026-09-04 自原版头抽出）
│   │   └── retired/           #     退役/归档算子（2026-09-04 git mv 移入；仍编译进
│   │                          #     fc2_ops——CMake GLOB_RECURSE＋逐头文件目录
│   │                          #     include 自动收录，构建零改动）：
│   │       ├── endpoint_extract/     # 旧 4-9 端点链（PJC 替代）
│   │       ├── virtual_camera_pose/  # 旧 4-10 光心求解（PJC 替代）
│   │       ├── pose_optimize/        # 旧 4-11 外参优化（PJC 替代）
│   │       ├── plane_map/            # 旧 4-12（fcstepdump --pm-ablation 消费）
│   │       ├── plane_map_temp_table/ # 旧 4-13（CMTT 4-14 替代）
│   │       ├── epipolar_interp/      # ⚠ 归档但**仍被引用**：头文件被 dual include
│   │       │                         #   （EpipolarInterpResult 类型源）；Execute 消费方
│   │       │                         #   ＝fcscan_e2e＋fcstepdump（CLI 4-6 已切 dual·Labeled,
│   │       │                         #   2026-09-04 产线切换, 两链逐字节等价验证后）
│   │       ├── epipolar_interp_opt/  # 极线插值优化版（生产链无消费；仅 fcstepdump
│   │       │                         #   --interp-bench 评估路径引用）
│   │       ├── laser_match_scan/     # ⚠ 按人工指令归档但**仍在用**：fcscan_e2e
│   │       │                         #   扫描链现役匹配算子（v1）
│   │       ├── laser_match_scan_v2/  # v2（仅测试作 V3 对照基线）
│   │       └── steger/              # ⚠ 归档但**仍在用**：fcscan_e2e/fcstepdump/
│   │                                #   interp_bench＋对拍测试（CLI 4-4 已切 steger_fast,
│   │                                #   2026-09-04 产线切换; 头文件为 common/steger_types.h
│   │                                #   类型源之一, 同日抽取解耦后本目录可独立缺席）
│   └── tests/                 #   顶层测试（12 件；依赖 left_skew 数据者缺数据时 SKIP；
│                              #   另 tests/baseline_depth/＝首个一级子文件夹测试
│                              #   test_curve_map_cuda_baseline_depth（6 cases，2026-10-09，
│                              #   curve_map_cuda 基线景深黄金测试：真数据在位自动跑/
│                              #   缺席 SKIP，FC_BASELINE_DEPTH_SIDECAR=1 可选 sidecar 对拍）；
│                              #   算子级测试另在各 operators/*/tests/，含 2026-10-08
│                              #   cmtt_cuda/tests/test_cmtt_cuda——CMTT 双引擎对拍）：
│       ├── test_laser_calib_e2e.cpp        # L3 冒烟：最小合法 handoff＋全黑小图调
│       │                                   #   laser_calib.exe，校验不崩/JSON 存在/schema
│       │                                   #   与 build 字段（不验证精度）
│       ├── test_handoff.cpp                # L4 交接：loadCameraCalibHandoff＋一致性校验
│       │                                   #   解析逻辑单测（不调 exe，与 e2e 互补）；
│       │                                   #   2026-09-03 增 stereoRectifyTempTable 整表
│       │                                   #   容错解析/三方一致性 3 cases
│       ├── test_frontend_ab_compare.cpp    # 诊断对拍（2026-09-04）：pose_12/pose_13
│       │                                   #   四臂（orig/fast steger × orig/dual interp）
│       │                                   #   ×3 重复逐步计数＋量化点集比对——定位
│       │                                   #   已知限制 #9 的非确定性（单帧隔离=等价）
│       ├── test_pose06_lid_vote.cpp        # L5 方案验证：pose_06 段级投票 V3 金标准
│       │                                   #   对拍＋消融 TV2~TV8c＋全套审计（见测试覆盖）
│       ├── test_pose06_chain_v3_viz.cpp    # pose_06 V3 全链逐步骤可视化（无 CCL）：
│       │                                   #   mask→steger_fast(Flat)→undistort→interp_dual
│       │                                   #   (Scan 段ID)→match_v3→recon；门禁 ≥8000 点；
│       │                                   #   表源＝data_in/left_skew/curve_map_temp_table.bin
│       │                                   #   sidecar 按温选档（2026-09-04 起免现场生成，
│       │                                   #   数据自包含，无 data_out 标定 JSON 依赖）；
│       │                                   #   产物落 data_out/pose06_chainv3/
│       ├── test_calib_pipeline_viz.cpp     # 标定链逐步骤可视化（前 2 帧，原版算子）：
│       │                                   #   mask→ccl→label→steger→undistort→interp→
│       │                                   #   match_v1→recon；产物落 data_out/calib_viz/
│       ├── test_scan_pipeline_fast.cpp     # 扫描链全量计时（无线号链）：mask→steger_fast
│       │                                   #   →undistort→interp_dual(扫描模式)→match_v2→
│       │                                   #   recon；PLY＋timing.csv 落 data_out/scane2e_fast/
│       │                                   #   （与 fcscan 有号基线的差异仅报告不判失败）
│       ├── test_scan_pipeline_viz.cpp      # 扫描链逐步骤可视化（前 2 帧，同上链路）：
│       │                                   #   各步 PNG＋recon.ply＋summary 落
│       │                                   #   data_out/scane2e_viz/
│       ├── test_laser_match_scan_v2.cpp    # v1/v2 对比基准：结果逐位一致＋CUDA event
│       │                                   #   计时（v2 中位目标 ≤3ms，v1 基线 ≈7.5ms）
│       └── test_scan_vs_calib_analysis.cpp # 对拍分析（pose_06/07）：标定有号链为金标准
│                                           #   vs 扫描链匹配质量＋表预测精度/区间排除审计，
│                                           #   仅分析不改算子；产物落 data_out/analysis/
│
├── data_in/                   # 输入样本：camera/（棋盘格）· left_skew/（pose_xx 真实样本，pose_06 起）
│                              #   另存运行时采集目录 1/、20260831_171343/（gitignore）
├── data_out/                  # 输出：camera_calib.json / laser_calib.json ＋ 调试检查目录
│                              #   analysis/（方案验证输出）/ chainv3/（V3 全链工具输出）
├── docs/                      # 文档
│   ├── module1_camera五文件简记.md / module2_laser三文件简记.md  # 非算子源文件简记
│   └── operator_docs/         #   算子参考卡（按《算子提炼模板》12 段落；每算子一子目录）：
│       └── curve_map / curve_map_temp_table / laser_match_scan / laser_match_scan_v3
│           / mask_extract / plane_map / projector_joint_calib / steger_fast
│           / epipolar_interp_opt / ccl
├── .gitignore                 # build_fc*/ ＋ data_in/* ＋ data_out/* 忽略规则
├── build_fc1_rel/             # 模块1 Release 构建产物（gitignore）
├── build_fc2_rel/             # 模块2 Release 构建产物（gitignore）
└── AGENTS.md                  # 本文件（工程说明＋变更史；2026-10-14 起 AI 自由访问）
```

## 前置依赖

| 依赖 | 版本 | 路径（默认） | 备注 |
|---|---|---|---|
| Visual Studio | 2022 / MSVC v14.44 | `C:/Program Files/...` | 工具集 v143 |
| OpenCV | 4.13.0（含 CUDA 模块）| `C:/opencv-cuda-4.13.0` | Release 带全部组件（含 imgcodecs） |
| OpenCV Debug（可选）| 4.13.0 | `C:/opencv-cuda-4.13.0-debug/x64/vc17/lib` | **不含 imgcodecs**，仅供算子自测 |
| CUDA Toolkit | 12.4+（pin v12.6 实测 v12.4.99） | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6` | 仅模块2 需要；sm_75/86/87 |
| Eigen | 3.4.1 | `C:/devlibs/eigen-3.4.1-install` | |
| spdlog / nlohmann_json / GoogleTest | 自动 FetchContent | — | GitHub 被封需 `-DJMW_GH_MIRROR=https://ghfast.top/` |

## 构建命令

### 模块1（无 CUDA，任何机器可构建）

```powershell
cmake -S factory_calib -B build_fc -DFC_BUILD_MODULE2=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc --config Release
ctest --test-dir build_fc -C Release --output-on-failure
```

Debug 注意：模块1 CLI 同样依赖 imgcodecs（`find_package` 列为 REQUIRED、无模块2 式跳过逻辑），对 Debug OpenCV 会在 configure 期直接失败——模块1 实际仅 Release OpenCV 可构建（见已知限制 #1）。

### 模块2（CUDA + Release OpenCV）

```powershell
cmake -S factory_calib -B build_fc2_rel -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Release `
      -DFC_BUILD_MODULE1=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc2_rel --config Release
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc2_rel -C Release --output-on-failure
```

Debug（仅算子自测，CLI 部分 Release-only）：

```powershell
cmake -S factory_calib -B build_fc2 -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Debug `
      -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib `
      -DFC_BUILD_MODULE1=OFF `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc2 --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc2 -C Debug --output-on-failure
```

### 全量（双模块）

```powershell
cmake -S factory_calib -B build_fc_all_rel -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CONFIGURATION_TYPES=Release `
      -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc_all_rel --config Release
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc_all_rel -C Release --output-on-failure
```

## 输入/输出目录约定

### 模块1 输入（`data_in/camera/`）

```
camera/
├── config.json         棋盘格 + 算子参数
├── temps.txt           参考温度（可选；单行 "ref_temp 22.5"，缺失时回退 config 的 referenceTemp）
├── left/
│   ├── 001.png  002.png ...   棋盘格图（CV_8UC1 或转灰度）
└── right/
    └── 001.png  002.png ...
```

**主控参数库（2026-10-04）**：`module1_camera/camera_calib_params.json`＝camera_calib CLI 全量可调参数主库（上述 config 全部键＋新增 `plateTempCoeff`/`plateTemp` 平铺键（此前不可配）＋算子组 `extrinsic`{maxReprojFactor, minViewCount}/`frames`{minValidFrames}）。四层优先级：编译内置 ← 主库 ← 数据集 config.json（逐叶覆盖）← temps.txt（ref_temp 最高）。主库定位：exe 旁（POST_BUILD 拷贝）→ cwd → 源码树；全缺回退内置同值默认（warn）。注意：主库基线**已对齐本机产线基准**（CESHI260831 config：15mm 方格/alpha 1.0；其余＝编译默认）——无 config 的数据集按此基准跑；改主库无需重编译（改源码树主库后需同步 exe 旁副本或触发重建）。

### 模块1 输出（`data_out/camera_calib.json`）

字段：`schema / imageSize / referenceTemp / cte / tempRangeMin·Max·Step`（三个顶层扁平键；**2026-09-02 起 ±15℃/0.5 步距=61 档**）
+ `intrinsic` (left/right K/D + rvecs/tvecs + rms)
+ `extrinsic` (R/T/E/F + 可选 camera_matrix_l/r ＋ dist_coeffs_l/r)
+ `rectify` (R1/R2/P1/P2/Q + validRoi{L/R})
+ 4 张温度表（intrinsicTempTableL/R, extrinsicTempTable, stereoRectifyTempTable）

**`stereoRectifyTempTable` 整节由模块2 handoff 容错解析消费（CMTT 4-14 源①）**：缺节/全弃仅降级（4-14 跳过不出产物）不报错；参考温锚点须与模块2 config 一致（三方一致性校验）。

### 模块2 输入（`data_in/laser/`——名义布局，当前磁盘无此目录）

```
laser/
├── config.json              激光链参数（gridStep/depthMin/Max/epipolarStep/lineIds…
│                            laser_label.scanDirection：0=竖切中心列(默认，水平多线)
│                            1=横切中心行(斜线形态)；laser_label.centerRowOffset
│                            f7Threads：F7 漂移解并发（0=自动 min(核−2,8)/1=串行/≤0 归 1）
│                            cmttThreads：CMTT 档级并行（0=自动 min(硬件线程−2,16)）
├── camera_calib.json        ← 拷贝/软链自 data_out/camera_calib.json
│                            （含 stereoRectifyTempTable 整节——4-14 源①）
├── pose_00/
│   ├── L_tube0.png  R_tube0.png
│   ├── L_tube1.png  R_tube1.png
│   └── temp.txt
├── pose_01/  ...
└── pose_24/
```

温度五字段缺省 ±15℃/0.5 步距（61 档，2026-09-02 统一对齐 CMTT）。2026-10-12 起支持备选布局：`left/`+`right/` 同名顺序帧（pose_* 缺席时自动回退，每对同名帧＝1 pose×1 tube——laser261006-1668 原始采集形态可直输）。真实样本集为 `data_in/left_skew/`（单斜线形态，config 已设 `scanDirection=1`；22 个 pose（pose_06 起），每 pose 含 L/R_tube0.png＋temp.txt）。**第二数据集 `test_data/laser261006-1668/`（2026-10-07 定稿）**：1688×1400 亮采集集体（left/right 顺序帧 143~171＋handoff＋定稿 config.json——mask 120/3/9＋steger 门禁 6.0；⚠ 显式零值键不可省，见已知限制 #10）；产线参数扫参结论：threshold=召回旋钮（120 救回暗线帧）、maxLineWidth=精度旋钮（治过曝噪点与光晕裙边偏置）、erodeSize≥5 灭线红线；rowStep 实际生效口径＝`cfg.epipolarStep`（主库 plane_map 组，2026-10-13 起＝0.67 矫正缩放比，非算子默认 0.7）。

**主控参数库（2026-10-04）**：`module2_laser/laser_calib_params.json`＝laser_calib CLI 全量可调参数主库（上述 config 全部键＋新增算子组 `laser_label.realLineTolerance`/`mask`/`ccl`/`steger`/`match`/`pjc`(19 键＋`acceptance` 六阈；**a1/a2 阈值口径＝perLine 归一化坐标残差，非像素**——与 finalSampsonRms 像素口径差 ~百倍，2026-10-04 勘正标注)/`cmtt.tempHalfRange`/`cmtt.clampedWarnRatio`）。三层优先级：编译内置 ← 主库 ← 数据集 config.json（逐叶覆盖，present-key）。主库定位：exe 旁（POST_BUILD 拷贝）→ cwd → 源码树；全缺回退编译内置同值默认（warn）。红线：主库基线＝编译默认值（非某数据集值）；`lineIds` 主库省略（空＝累积点云反推）；CMTT `rowStep/depthMin/depthMax` 随标定走不外置。改主库无需重编译。同日验收增强：产物新增顶层 `verdict` 键（透传 PJC 验收 PASS/DEGRADED，堵 status=ok 静默放行降级品）＋`pjc.basinProbe`（盆地探针决策落盘）＋clamped 档占比超 `clampedWarnRatio` 告警。2026-10-07/08 增键：`laser_label.expectedLineCount`(25)/`maxRunWidth`(30)（编号错位双门禁，帧级）；`steger.maxLineWidth`(0=off；数据集层按需启用，laser261006-1668 定稿 6.0)；`cmtt.cudaEnabled`(true=CUDA 主引擎/false=CPU 备用)；`mask.highlightErodeSize/highlightDilateSize` 键已支持（主库不写入=off，数据集层启用）。

### 模块2 输出（`data_out/laser_calib.json`）

字段：`schema(factory_calib.laser_calib.v2) / build(6.3-cmtt) / status(ok|partial)`
+ `posesProcessed / framesOk / framesSkipped / accumulatedPoints3D`
+ `haveVirtualPose / haveLaserExtrin / haveCurveMapTable`
+ `virtualK/R/T`（PJC 联合标定后）＋ `pjc`（projectorT / 25 条发射曲线 / epipolarRowStep / acceptance / perLine 诊断）
+ `laserExtrinsicTempTable`（virtual→L/R 每温度，5-3 输出）
+ `curveMapTempTable`（meta 10 键：success/message/path/tempBase/tempStep/tierCount/tierCountOk/clampedRatio/gridHash/sha256；4-14 输出）

**4-14 CMTT sidecar 落盘 `{输出 JSON 同目录}/curve_map_temp_table.bin`**（96B 头＋逐档 32B 索引＋61 档 CMPU blob；落盘前 CmttReader 复核，sha256 对落盘文件流式计算）。参考卡见 `docs/operator_docs/curve_map_temp_table/`。旧 4-13 planeMapTempTable 已退役（2026-09-03）；`curve_map` 单档表仍可由 tools（fcstepdump `--curve-map`）现场生成（sidecar 不可用时的回落路径）。

## 运行示例

```powershell
# 模块1：相机标定
.\build_fc_all_rel\module1_camera\Release\camera_calib.exe `
    data_in\camera data_out\camera_calib.json

# 模块2：激光标定（需把模块1 输出软链/拷贝到 data_in\laser\camera_calib.json）
Copy-Item data_out\camera_calib.json data_in\laser\camera_calib.json
.\build_fc_all_rel\module2_laser\Release\laser_calib.exe `
    data_in\laser data_out\laser_calib.json

# 工具：逐帧调试转储 / 扫描式端到端（Release 同批构建）
.\build_fc2_rel\module2_laser\Release\fcstepdump.exe data_in\left_skew data_out\stepdump
.\build_fc2_rel\module2_laser\Release\fcstepdump.exe --curve-map data_in\left_skew data_out\curve_map_table.bin
.\build_fc2_rel\module2_laser\Release\fcscan_e2e.exe data_out\laser_calib.json data_in\left_skew data_out\scane2e
.\build_fc2_rel\module2_laser\Release\fcscan_e2e.exe data_out\laser_calib.json data_in\left_skew data_out\scane2e_t30 30.0

# 工具：CMTT sidecar 诊断 / 独立生成（tierThreads 可选）
.\build_fc2_rel\module2_laser\Release\fcstepdump.exe --cmtt curve_map_temp_table.bin
.\build_fc2_rel\module2_laser\Release\fcstepdump.exe --cmtt-gen data_out\laser_calib.json data_out\curve_map_temp_table.bin 16

# 工具：V3 全链（每步耗时+图片+PLY 点云；第 4 参可选目标温度，sidecar 按温选档）
.\build_fc2_rel\module2_laser\Release\fcchain_v3.exe `
    data_in\left_skew data_out\left_skew_calib.json data_out\chainv3 22.5

# 测试：V3 方案验证（金标准对拍 + 全套诊断输出）
.\build_fc2_rel\module2_laser\Release\test_pose06_lid_vote.exe
```

`laser_calib.exe` 退出码：0 = 完整流水线成功；1 = 部分失败——`haveVirtualPose` / `haveLaserExtrin` / `haveCurveMapTable` 任一为 false（含无累积 3D 点、PJC/5-3/4-14 失败等诱因）；输出写出失败、未捕获异常、入参错误亦返回 1。单帧算子 `success=false` 仅跳帧计入 `framesSkipped`，不必然置 1。stderr 通过 spdlog 打印详细诊断。

## 已知限制

1. **模块1/模块2 CLI 及 tools 均仅 Release OpenCV 可构建**：Debug 自建 OpenCV 缺 `imgcodecs`（`cv::imread` 所需）。模块2 CMakeLists 探测到缺失时自动跳过 `fc2_io`/`laser_calib` ＋ 三个 tools exe 的构建（算子自测 fc2_ops＋tests Debug 可跑）；模块1 无跳过逻辑，Debug OpenCV 下 configure 直接失败。
2. **模块2 端到端精度需真实样本数据**：当前 Task 6.3 只覆盖冒烟（合成全黑图）。真实激光线图 + ground-truth virtual pose 的精度验证需用户提供产线样本。
3. **nvcc 版本飘移**：顶层 CMakeLists pin v12.6 但实测 `v12.4.99` 被选中（CMake cache 无 FORCE），但能编译 + 测试绿，按经验 #4 接受。
4. **CRT 部署**：`fc_deploy_crt()` 自动把 `MSVCP140.dll` / `vcruntime140*.dll` 拷到 exe 同目录；2026-09-03 起追加 OpenMP 运行时 `vcomp140.dll`（redist 各版本取最大，best-effort 探测，找不到仅告警不阻断）；OpenCV DLL 仍需 PATH 设置。
5. **GitHub 封锁**：所有 configure 命令必须带 `-DJMW_GH_MIRROR=https://ghfast.top/`。
6. **虚拟相机链路已切换 PJC（2026-08-25）**：CLI 的 4-9/4-10/4-11（endpoint_extract / virtual_camera_pose / pose_optimize）已由 `projector_joint_calib`（自 `modules/09_operatorlib/.../projector_joint_calib/` 拷入，CPU/Eigen/手写 LM）替代——模型 t(3)+发射曲线 C(6)，R=I、K=f+主点 固定（见 `docs/流水线/客户端标定流水线.md` v2.2 阶段4）。旧三算子源码已移入 `operators/retired/`（2026-09-04 归档；仍编译进库但 CLI 不再调用）。lineIds 反推改自累积点云统计。left_skew 单斜线形态已支持（2026-08-26 起）：`laser_label` 经 config `laser_label.scanDirection=1`（横切中心行）编号，不再在 4-3 跳过。PJC 已在单测＋真实点云 harness（fcstepdump `--pjc-*` 子命令）验证：正常求解/收敛/退化诊断（条件数、anomalyRmsThreshold）均符合设计。**2026-09-02/03 加速演进**：线内雅可比＋块稀疏 Gauss-Newton 累加（useBlockJtJ 默认开，弃 M×P 稠密 J≈557MB）＋相位计时常开化（enableTiming，fcstepdump 接线）＋F7 漂移解 28 任务并发（f7Threads 可配，硬顶 8 线程）＋F7 失败计数进验收（任一 drift 解失败强制 DEGRADED，堵假 PASS）——全链 ×29（96min→3min20s），载体轨迹字节对拍同一。
7. **扫描链段级匹配 V3 已落地（2026-09-01）**：`epipolar_interp_dual` Scan 模式新增**段 ID 输出**（`scan_link_dx>0` 时行间簇链接合并为段，pointer doubling 求根；`=0` 输出簇 ID）；`laser_match_scan_v3` 消费段 ID 实现**段级投票+冲突双弃**（就近单卡发卡 → 段 winner 选举 → 同行两段同号双弃 → winner 卡收集 → CUB 紧凑化）。pose_06 实测 14004 匹配/**零错配**（v2 逐点唯一命中制 13244 匹配/2564 错配）。V3 **不复用 v2 匹配 kernel**（票制复核证明 v2 唯一命中制下双弃只杀 28% 错配）。冲突双弃保守策略会丢弃对家正确段（实测弃 30/142 段），换取 wrong=0 确定性；后续可扩展视差带重选找回对家。参考卡：`docs/operator_docs/laser_match_scan_v3/`。
8. **4-13 已由 4-14 CMTT 替代（2026-09-03）**：CLI 温度表产出从 plane_map_temp_table（plane_map 链）切换为 `curve_map_temp_table`（CMTT：curve_map 曲线真源＋逐档重标，61 档 ±15℃/0.5 sidecar `curve_map_temp_table.bin`＋JSON meta）。三源＝模块1 stereoRectifyTempTable（handoff 整表，**模块1 老输出缺该节时 4-14 跳过**）＋5-3 laserExtrinsicTempTable＋PJC emissionCurves。红线：参考档 FAIL 整体不出产物；落盘前 CmttReader 复核＋文件级 sha256。旧 plane_map/plane_map_temp_table 已移入 `operators/retired/`（仍编译；fcstepdump `--pm-ablation` 消融保留）。工具侧 fcscan_e2e/fcchain_v3 支持 sidecar 按温选档（第 4 参 temp_c，缺省参考温；缺 sidecar 回落现场生成）。参考卡：`docs/operator_docs/curve_map_temp_table/`。
9. **前端运行间非确定——已根治（2026-09-04 五轮定位＋两处修复，全程留档）**：现象为 full run 累积点抖动（±33~+252）、sidecar sha 逐跑不同、换 fast/dual 算子后稳定劣化。**最终真凶（第五轮点云导出定位）＝laser_match `allocateBuffers` 容量策略越界**：缓冲按单侧 left/right 分配但 early-return 只看 maxCount——单侧超过历史分配尺寸而 maxCount 未增长时 kernel 写越界踩相邻显存（实测产出"幽灵线号"点：11.4 亿级 fid、真实坐标，仅部分环境显现；**原哈希实现同病**，历史 matched 抖动部分源于此；踩法随堆布局而变＝此前所有"环境依赖"之谜）。**修复①**（重写哈希→排序＋二分，字典序代表点，调度/顺序双无关）；**修复②**（缓冲统一按 maxCount 分配，越界消除）。**修复后闭环验证**：幽灵清零（match/recon 值域恒 [1,25]，matched=reconCnt）；**原版链与 fast/dual 链全流程逐字节一致**（accum 464143/rms 1.1053/PASS/sha 00c90b6d… 全同，projectorT 零差）——fast/dual 四轮排除法＋本结论**无罪**，此前"劣化"全是越界在不同布局下的不同表现。注意：新基线点集不再含幽灵点，PJC 落盆与旧基线不同（rms 1.105 vs 旧 0.4479，均 PASS——盆地敏感性仍是既有特性）。**PJC 盆地敏感性教训保留：任何前端变更后以真实数据全流程验证为准**。⚠ laser_match 与 09 侧逐字拷贝分歧（结构性重写＋容量修复），**同步 09 侧需移植**。CLI 附 `FC_PLY_DUMP=<path>` 环境变量可导出累积三维点云（含 pose_idx/line_id 标量，诊断用）。
10. **过曝粘连编号错位——三道防线叠加根治（2026-10-07，laser261006-1668 事故）**：根因链＝图像高光过曝→相邻激光线在掩膜粘连（实测切割游程 38~132px，正常 12~22px）→4-3 split 凑回 25 线但编号按位置顺序错位→错号点毒化 PJC（Sampson 38px/A7 盆地 1480mm）。三防线：①laser_label 双门禁（expectedLineCount=25 线数门＋maxRunWidth=30px 游程展布门，split 前测原始游程，任一目触发 CLI 4-3 整帧跳过）；②mask 高光分离可选路径（hlErode 吃细线留粗斑→回扩当减法模板，厚度判据）；③steger 逐点宽度门禁（w=sqrt(g/|λ|) 平滑后口径，过宽点=偏移噪点直接剔除——**点级防线同时治愈小闭运算链的 A7 盆地发散**，230/3/7 链 A7 5.10→0.21 实证）。该数据集定稿组合＝mask 120/3/9＋maxLineWidth 6.0（23 帧/43.4 万点/PASS，A7=0.03mm）。⚠ 09 侧三算子（laser_label/mask_extract/steger_fast）已同步镜像；**数据集 config 显式零值键（postErodeSize/highlightErodeSize=0）不可省**——编译默认 post=13 顶上会净腐蚀灭线（2026-10-07 固化事故，1/29 帧当场暴露）。
11. **CMTT 双引擎：CUDA 主＋CPU 备（2026-10-08 转正）**：`operators/cmtt_cuda`（编排）＋`operators/curve_map_cuda`（单档 GPU 求交，-fmad=false 专用库 fc2_curve_map_cuda）为 4-14 默认引擎，61 档 24.8min→113s、全链 31.4min→3.5min；`cmtt.cudaEnabled=false` 一键切回 CPU（curve_map_temp_table 完整保留）。**等价性边界**：两版产物结构一致、真实参考档逐条目一致（27,538,991==27,538,991），但**不承诺 61 档逐字节相同**（浮点边界）——CPU 版线程数无关确定性红线不受影响，CUDA 版保证同架构逐次可复现。测试 `test_cmtt_cuda`（3 用例＋env 门控全量计时）。09 侧 CMTT 仍为 CPU——07-E 若要提速需另行移植（同 laser_match 债务模式）。

## 测试覆盖

| 层级 | 测试文件 | 数量 | 状态 |
|---|---|---|---|
| L1 算子回归（模块1）| `operators/*/tests/test_*_cpu.cpp` | 6 文件 / 116 cases | Release 绿 |
| L1 算子回归（模块2）| `operators/*/tests/test_*.cpp` | 26 文件 / 约 435 cases | 26/26 基线双绿；2026-08-26 扩充（+curve_map +projector_joint_calib，laser_label/epipolar_interp/ccl 用例重写扩充）、2026-08-28 再扩（+steger_fast；ccl 随 `b620d91` GPU 化重写）、2026-09-03/04 又扩（+curve_map_temp_table 5 件、+stereo_rectify_temp_table）、2026-10-07/08 三防线扩（laser_label +8 门禁、mask_extract +4 高光分离、steger_fast +6 宽度门禁）后以人工全量回归为准 |
| L2 棋盘格（模块1）| `tests/test_chessboard_corner.cpp` | 7 | Release 绿 |
| L3 端到端（模块1）| `tests/test_camera_calib_e2e.cpp` | 1 | Release 绿 |
| L3 端到端（模块2）| `tests/test_laser_calib_e2e.cpp` | 4 | Release 绿（冒烟）|
| L4 交接（模块2）| `tests/test_handoff.cpp` | 10 | Release 绿（含 stereoRectifyTempTable 整表解析 3 cases，2026-09-03） |
| L5 方案验证（模块2）| `tests/test_pose06_lid_vote.cpp` | 1 | Release 绿（left_skew 数据不存在时 SKIP）；含 V3 算子端到端对拍（金标准 14004/0）＋全套消融实验（TV2 门控/TV4 裸投票/TV5 认领否决/TV6 第二名/TV7 排名迭代/TV8c 视差带）＋审计（forensic 尸检/左右一致性/逐线画像/逐行簇计数/CSV 导出） |
| L6 诊断/可视化（模块2）| `test_scan_pipeline_fast.cpp` · `test_scan_pipeline_viz.cpp` · `test_calib_pipeline_viz.cpp` · `test_laser_match_scan_v2.cpp` · `test_scan_vs_calib_analysis.cpp` · `test_pose06_chain_v3_viz.cpp` | 6 文件 / 各 1 case | 2026-08-29/30 新增 5 件＋2026-09-01 增 pose06_chain_v3_viz；2026-09-04 retired/ 归档＋首轮 4-4/4-6 切换时点全量 34/34 绿→真实数据验证证伪**当日回滚**→laser_match 越界根因修复（两链逐字节一致）后**复切 fast/dual**，全量 35/35 绿（现役） |
| L7 诊断对拍（模块2）| `tests/test_frontend_ab_compare.cpp` | 1 文件 / 3 cases | 2026-09-04 新增（选项 C 定位手段）：①pose_12/13 四臂 ×3 逐步计数＋量化点集比对；②shuffle 探针（证 4-6/4-7 输入点序不敏感）；③Sequential22Frames CLI 复刻＋逐级量化哈希（定位 4-7 非确定＋修复验证 4 进程全同）；单跑绿 |
| L8 CMTT 双引擎（模块2）| `operators/cmtt_cuda/tests/test_cmtt_cuda.cpp` | 1 文件 / 4 cases（1 env 门控） | 2026-10-08 新增：①合成单档 CPU/CUDA 精确一致（685,919==685,919）；②合成 CMTT 9 档 vs 生产 CPU 生成器逐档一致；③真实数据参考档 vs 已交付 sidecar 全尺度对拍（27,538,991==27,538,991）；④FC_CMTT_CUDA_FULL=1 全量 61 档计时（95s/61 档）；单跑绿 |
| L9 基线景深黄金（模块2）| `tests/baseline_depth/test_curve_map_cuda_baseline_depth.cpp` | 1 文件 / 6 cases | 2026-10-09 新增（curve_map_cuda 基线景深参数化配套）：合成 A1–A5（窗内一致 δ 精确式/边界＋空窗/非法窗口拒跑/CPU↔CUDA 同窗逐条目精确一致/别名逐字节等价）＋真数据 laser261006-1668 窗 100–500（25 线 Z 界＋rowStep 冻结回显；golden 式根探测缺席 SKIP；`FC_BASELINE_DEPTH_SIDECAR=1` 可选与交付 sidecar 参考档 entryCount 对拍）；单跑 6/6 绿 |

---

# 开发规则（自 1002 产线分支 AGENTS.md 合并保留，2026-11-01 移植时并入）

# 寮€鍙戣鍒?
## 鏍稿績鍘熷垯锛氭湁鐜版垚鐨勫氨鐓ф惉锛屼笉鑷綔鑱槑

**褰撳瓨鍦ㄥ凡楠岃瘉鍙伐浣滅殑瀹炵幇鏃讹紝鍘熸牱鐓ф惉锛屼笉瑕佸彂鏄庢浛浠ｆ柟妗堛€?*

### 鑳屾櫙

`E:\workfold\20260509intergrate\LeadScanK2\series` 鏄壂鎻忎华鐩告満鐨?*宸查獙璇佺ǔ瀹氬疄鐜?*锛堥噰闆嗐€佸仠姝€佸啀閲囬泦鍧囨甯稿嚭鍥撅級銆俧actory_calib 鍦ㄧЩ妞嶅畠鐨勫姛鑳芥椂锛屽繀椤诲師鏍风収鎼叾浠ｇ爜缁撴瀯鍜岃皟鐢ㄦ柟寮忥紝涓嶈"鍙傝€冪潃鏀?銆?
### 鍏蜂綋瑕佹眰

1. **鐓ф惉锛屼笉瑕?鍙傝€?**锛氱収鎼?= 閫愬瓧澶嶅埗鍏抽敭浠ｇ爜 + 淇濇寔鐩稿悓鐨勮皟鐢ㄩ『搴忓拰瀵硅薄鐢熷懡鍛ㄦ湡銆?鍙傝€冪潃鏀?= 鍙€熼壌鎬濊矾銆佽嚜宸遍噸鏂扮粍缁囦唬鐮?鈥斺€?鍚庤€呭凡琚娆¤瘉鏄庝細寮曞叆闅愯斀 bug锛屾氮璐规椂闂淬€?
2. **Galaxy 鐩告満閲囬泦**锛堝凡韪╁潙锛屽姟蹇呯収鎼?LeadScanK2 `CameraControl.cpp`锛夛細
   - `start_ScannerAcquisition`锛氭瘡娆￠兘 `GetRemoteFeatureControl` + `OpenStream`锛堜簩鑰呰法 stop 涓嶅彲澶嶇敤锛夛紝`RegisterCaptureCallback` + `StartGrab`锛岄厤缃?`Line2` 瑙﹀彂锛?*涓ゅ彴鐩告満閮介厤瀹屽悗鍐嶅垎鍒?`AcquisitionStart`**銆?   - `close_ScannerAcquisition`锛歚AcquisitionStop` 鈫?`StopGrab` 鈫?`UnregisterCaptureCallback` 鈫?delete handler 鈫?`Close` stream銆?
3. **棰勮鏄剧ず**锛堝凡韪╁潙锛屽姟蹇呯収鎼?LeadScanK2 `LEADSCANSeries.cpp`锛夛細
   - 鍥炶皟绾跨▼璋?`setLeftImage`/`setRightImage`锛?*宸﹀彸鎸?frameID 閰嶅**鍚?`cv::resize(0.25)` 鈫?`convertMattoQImage`锛坄Format_Indexed8` + 閫愯 memcpy 鍒?scanLine + 256 鐏板害棰滆壊琛級鈫?`emit updateImages(QPixmap::fromImage(...).toImage(), QPixmap::fromImage(...).toImage())`銆?   - `connect(updateImages, onUpdateImages, Qt::QueuedConnection)`銆?   - `onUpdateImages`锛歚label->setPixmap(QPixmap::fromImage(img))`锛岀函 `QLabel`銆?   - **涓嶈**鐢細鍗曞紶鐙珛 emit銆乣PreviewWidget` 瀛愮被銆乣convertToFormat`銆佽嚜宸卞彂鏄庣殑 metatype 璺ㄧ嚎绋嬫姇閫掋€?
4. **鏀瑰姩鍓嶅厛璇荤幇鎴愬疄鐜?*锛氬姩鎵嬪墠鍏堣 LeadScanK2 瀵瑰簲鍑芥暟鐨勫畬鏁翠唬鐮侊紝纭姣忎竴琛岋紝鍐嶇収鎼€備笉瑕佸嚟璁板繂銆?
### 宸茶俯杩囩殑鍧戯紙涓嶈鍐嶈俯锛?
- **scanner 鍚姩鎸夐挳閲屽厛 `stop`(N11) 鍐?`start`(N10) 杩炵画鍙戦€?鈫?MCU 鍋跺彂涓嶅惎鍔?*锛堢涓€娆＄ǔ瀹氥€佸悗缁伓鍙戜笉鍑哄浘/涓嶅嚭甯х殑鏍瑰洜锛夈€傜収鎼?LeadScanK2锛歚on_pushButton_Start_Scanner_Clicked` 鐩存帴鍙?N10锛?*涓嶅厛 stop**銆傚仠鍐嶅惎鐢辩敤鎴峰垎鍒偣銆屽仠姝€嶃€屽惎鍔ㄣ€嶏紙绉掔骇闂撮殧锛夛紝涓嶅湪涓€娆＄偣鍑婚噷杩炲彂 N11+N10銆?- stream 璺?stop/start 澶嶇敤 鈫?绗簩娆￠噰闆嗕笉鍑哄抚
- QImage 璺ㄧ嚎绋嬬敤 `convertToFormat` / 鍗曞紶 emit 鈫?鍋跺彂涓嶆樉绀?- `PreviewWidget` 瀛愮被 + 鑷畾涔?styleSheet 鈫?娓叉煋鍋跺彂澶辨晥
- AcquisitionStart 鏃舵満锛堝乏鍏堝彸鍚庯級鈫?瑙﹀彂 race
- `CameraFrame::frameIndex` 鐢ㄩ粯璁?0 鑰岄潪鐪熷疄 `GetFrameID()` 鈫?setLeftImage 閰嶅閿欒锛堝垵濮?m_iRightId=0==m_iLeftId=0锛夛紝resize 绌?Mat 宕╂簝

杩欎簺 LeadScanK2 閮藉凡姝ｇ‘澶勭悊锛岀収鎼嵆鍙伩鍏嶃€?

