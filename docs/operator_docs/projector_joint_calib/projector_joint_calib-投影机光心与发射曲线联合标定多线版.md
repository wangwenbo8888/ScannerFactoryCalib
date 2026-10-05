# 投影机光心与发射曲线联合标定（多线版）

> 依据《算子提炼模板.md》v1.0 十二段落编写；对象为 factory_calib 现版 `ExecuteMultiLine`（2026-08-26 v2；2026-09-02/03 加速演进：线内雅可比＋块稀疏 GN 默认开＋相位计时）
> 主工程 09 侧同名算子仍为单线版——同步待裁决（见 H/I 节）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | `激光标定-14` |
| 中文名称 | 投影机光心与发射曲线联合标定（多线） |
| 英文目录名 | `projector_joint_calib` |
| 运行平台 | CPU（Eigen 手写 LM；可选 Ceres AutoDiff 单线版） |
| 所属流程 | 激光标定流程 3c 步（laser_reconstruct 4-8 聚合之后、4-14/5-3 之前） |
| 精度档次 | `③`（亚像素/浮点类；left_skew 实测 rms 0.45px、t 漂移 ≤0.1mm） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | lines：每条激光线的多姿态点集（lines[lineIdx][poseIdx]，按线分组） | `vector<vector<PosePointSet>>`（points3d CV_32FC3 + lineIds） |
| **输入②** | f：焦距（像素，← stereoK=P1 3×3 派生） | `double` |
| **输入③** | principalPoint：主点 (cx, cy) | `cv::Point2d` |
| **输入④** | initialT：光心初值（矫正系！＝R1·(80,3,3)，非机械原值） | `cv::Vec3d` |
| **输出** | MultiLineResult：projectorT（共享光心）+ emissionCurves[25]（逐线曲线）+ perLine/perPose 诊断 + cond/rms | 见 D2 |

## C. 算法

**物理模型**：投影机＝针孔虚拟相机（R=I、K=f+主点固定）；25 条激光在虚拟 CMOS 上是**固定位置固定形状的连续解析曲线**（图案常数，不随姿态/温度变）；空间点必在「光心→曲线点」射线上。

**核心流程**：

```
Step 0  逐线×逐姿态降噪（denoisePoses）
        ├ SVD 平面拟合: |离面| < 0.6mm 内点 → 投影到平面
        ├ 平面曲线降噪: 3 阶多项式 β(α) + 3σ 剔除重拟合（横向噪声）
        └ 带出逐姿态 σ_s（内点 RMS 离面厚度）与逐姿态点数
Step 1  帧定权: W_s = (σ_ref/σ_s)² clamp [0.2,5]（平面厚=重建差 → 降权）
Step 2  逐线归一化: u'=(u-μ)/σ（独立 NormParams/线，消 JᵀJ 病态）
        曲线初始化 C={0,0,0,0,1,0}（F'=v' 直线, ‖C‖=1）
Step 3  联合 LM (3+6L DOF: 共享 t + 每线 6 参曲线)
        ├ 残差: r=√(W_s·Zp/f·w_rob)·F/‖∇F‖  (Sampson 距离)
        ├ 核退火: 前 10 iter 纯 L2 → Huber(δ=1px) → 残差中位数<cauchyToL2Thresh×0.5=0.15 后 Cauchy(c=max(0.2, 2×medR))
        ├ 正则: λ‖[A,B,C]‖² 退火 λ₀=1→1e-6（先调 t 后放开弯曲）
        ├ 雅可比: 数值前向差分 eps=1e-7; LDLT 解正规方程
        │   └ B·块稀疏 GN 累加（useBlockJtJ=true 默认, 主循环与 IRLS 两处共用）:
        │     t 列 3 次全量差分（按线切片消费）; C_l 列线内差分（computeMultiResidualsLine
        │     仅算 l 线——循环体逐字取自全量版, 算术逐位一致）; 按线 J_lᵀ·J_l 累加 H/g
        │     （9×9 块散布 {t,t}/{t,c_l}/{c_l,c_l}）, 不再分配 M×P 稠密 J（455k×153≈557MB
        │     → 每解数十 MB）; false 回退稠密 JᵀJ（回退分支字节保真）
Step 4  IRLS ≤2 轮: 用残差刷新外部权 w=1/(1+(r/r̃)²) → 再优化 30 iter
Step 5  诊断: perLine(rms/p95/离群率/帧权) + perPose(带符号偏差)
Step 6  输出: t + 25 条去归一化曲线 + cond/rms + qualityFlag
        （enableTiming=true 时附相位耗时汇总日志: denoise/norm/jac/JtJ/trials/misc
         ＋ iter/eval 计数; 默认 false, 数值路径零变化——fcstepdump --pjc-* 已接线）
```

**两阶段初始化（调用方职责，fcstepdump/cli 已内置）**：联合代价面**非凸**（实测 3 盆地）——先逐线单解取中位数共识（rms<1 且 cond<1e15 的线），再 z+60 盆地探测（单线 t_z 系统性偏负，联合最优在其上方），15 iter 短跑低 rms 者胜出作完整优化初值。

**关键函数**：

| 函数 | 用途 |
|------|------|
| `Eigen::SelfAdjointEigenSolver` | 平面拟合协方差分解 + cond 退化检测 |
| `Eigen::bdcSvd` | Step 0 曲线降噪最小二乘 |
| `Eigen::LDLT` | LM 正规方程 (JᵀJ+λI)δ=−Jᵀr |
| `computeMultiResidualsLine()` | 线内残差（A·线内雅可比基础；仅算 li 线, 与全量版逐位一致） |
| `denormalizeCurve()` | 归一化曲线 → 原坐标（矩阵相似变换 T⁻ᵀM_pT⁻¹） |

## D. 依赖

**上下游算子**：

```
laser_reconstruct(4-8) ──编排层按线聚合──→ 本算子 ──projectorT──→ 5-3 laser_extrinsic_compensate
                                                  ├→ 4-12 plane_map（仅 fcstepdump --pm-ablation 消融）
                                                  └→ 4-14 curve_map_temp_table（emissionCurves 三源之一, 2026-09-03 起）
emissionCurves ──→ 4-14 生成 61 档 sidecar ＋落盘 laser_calib.json pjc 节
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `denoisePoses()` | **仅多线版使用**（单线版 Execute 保留原内联降噪、契约不动——主工程同步时是合并候选） |
| `common/calib_types.h` 等 | QualityFlag/日志/版本共享类型 |
| 两阶段初始化 | fcstepdump 与 laser_calib_cli 各持一份（~80 行, 待提取公共） |

## D2. 衔接

**上游→本算子**（编排层负责）：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| 4-8 逐帧点云 | CPU 聚合按线分组（pointCloud→perLine buckets） | **必须按线分组**——混线输入=崩坏（rms 31px） |
| 3-4 stereo_rectify | 值传递 f/主点 ← P1 3×3 | 点云在矫正系 |
| 机械装配 | 值传递 initialT＝R1·(80,3,3) | **注意 R1 旋转**（实测 10.4° yaw） |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `projectorT` | 5-3（v2l.T）/4-12（virtualT, --pm-ablation） | 参数结构体字段 | 值 | 必用 |
| `emissionCurves[].coeffs` | 4-14 curve_map_temp_table（重标生成 61 档）＋ laser_calib.json `pjc` 节 | Generate / writeJson | 值 | 必用（2026-09-03 起 F3 已落地） |
| `perLine/perPose` | JSON 诊断/验收框架 | writeJson | 值 | 可选 |
| `qualityFlag/message` | CLI status 判定 | 值 | 必用 |

## E. 架构

**文件结构**：

```
projector_joint_calib/
├── projector_joint_calib.h      # 公开接口（Params/Input/Result/MultiLine*/算子类）
├── projector_joint_calib.cpp    # Execute(单线,原契约) + ExecuteMultiLine(多线) + denoisePoses 公共件
└── tests/
    └── test_projector_joint_calib.cpp   # 21 用例（单线 16 + 多线 5）
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `ProjectorJointCalib`（无状态、无 pImpl——CPU 算子） |
| 核心方法 | `Execute(const ProjectorJointCalibInput&)` 单线（旧契约保留）/ `ExecuteMultiLine(const MultiLineInput&)` 多线 |
| 参数结构体 | `ProjectorJointCalibParams`（含 F6 鲁棒开关） |
| 结果结构体 | `MultiLineResult`（move-only；内嵌 LineDiag/PoseDiag） |
| 曲线结构体 | `ImplicitCurve`（coeffs[6]/discriminant/pointCount） |
| 日志标签 | `"14-ProjectorJointCalib"` |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Eigen | ≥ 3.4 | 全部线性代数（无 OpenCV CUDA） |
| OpenCV | ≥ 4.x | 仅 core（Vec3d/Point2d 数据结构） |
| nlohmann_json | — | Params 序列化 |
| Ceres（可选） | 2.2 | 仅单线版 useCeres=true（BUILD_CERES；多线版无 Ceres 路径） |
| MSVC | — | /bigobj |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| maxIterations | int | 100 | >0 | LM 最大迭代 |
| convergenceThreshold | double | 1e-8 | >0 | 步长收敛 |
| minPoses | int | 5 | ≥3 | 每线最少有效姿态 |
| minPointsPerPose | int | 50 | ≥10 | 每姿态最少点 |
| planeFitInlierThresh | double | 0.6 | >0 | 平面内点阈值 mm（3σ@σ=0.2） |
| curveDegree | int | 3 | 2/3 | 降噪多项式阶（3 阶精度优 14×） |
| lambda0 / lambdaDecay | double | 1.0 / 0.95 | — | 二次项正则退火 |
| cauchyToL2Thresh | double | 0.3 | >0 | Cauchy 切换：残差中位数 < 此值×0.5 时切换 |
| anomalyRmsThreshold | double | 0.15 | >0 | 异常拟合告警阈值 px |
| **useBlockJtJ** | bool | true | — | B·块稀疏 GN：按线 J_lᵀJ_l 累加 H/g（弃 M×P 稠密 J）；false=稠密 JᵀJ 回退（字节保真） |
| **enableTiming** | bool | false | — | 相位计时常开骨架（数值中性, 仅多一条汇总日志；fcstepdump 已接线） |
| **robustEnabled** | bool | true | — | F6 核退火总开关 |
| **huberDelta0** | double | 1.0 | >0 | Huber δ 初值 px（自动阶梯收紧至 0.7/0.5） |
| **irlsMaxRounds** | int | 2 | ≥0 | IRLS 轮数上限 |
| **poseWeightEnabled** | bool | true | — | 帧定权 W_s 开关 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 几何假设 | R=I（绝对轴线平行）；K=f+主点固定 |
| 输入分组 | **必须按线分组**——25 条线混喂单曲线＝模型违背（实测 rms 31px/t 飞至 −1869mm） |
| 优化自由度 | 3+6L（L=25 → 153 DOF） |
| 曲线约束 | ‖C‖=1 尺度归一；discriminant B²−4AC≤0 碗状 |
| 吸引盆 | 联合代价面非凸（实测 3 盆地）；**必须两阶段初始化**（机械初值直跑落 rms 4.42 劣盆 vs 共识+z-lift 0.44） |
| 求解耗时 | 加速前 ~6-7min/次全量 455k 点（100 iter+IRLS）；2026-09-02/03 演进（线内雅可比＋块稀疏 GN＋F7 并发）后全链 ×29（96min→3min20s, plans 附录实录）；等价性：载体轨迹字节对拍同一，纯效应 dT 1.5e-7mm 属条件数放大噪声 |
| 实测精度 | left_skew 455k 点：rms 0.448px、剔线漂移 0.086mm、basin 散布 0.009mm |
| 线程安全 | 非线程安全；多实例并行各自独占（F7 28 解并发即按此约束设计：每任务独立实例＋按值拷贝输入） |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | improvementRatio ≤0.5 且 cond<1e10 且 rms<anomalyRmsThreshold |
| Degraded | 降级 | improvementRatio 0.5~0.9 |
| Warning | 警告 | improvementRatio>0.9 或 cond>1e10（t_z 不可信）或 rms≥0.15（伪极小值） |

**外部验收（CLI acceptance 节，参考精度 v1）**：A1 逐线 rms≤0.5px / A2 p95≤1.5 / A3 离群≤2% / A5 剔线漂移≤1mm / A6 剔姿态≤2mm / A7 basin 散布≤0.5mm → PASS/DEGRADED/FAIL；L0-L2 自动阶梯（核收紧）attempts 留痕。**FAIL 不出标定产物**。2026-09-03 假 PASS 修复：drift 解失败哨兵 (1e9,0,0) 曾被静默跳过（失败越多统计越低越易假 PASS）——现逐段计数＋逐条 warn＋`solveFailures` 进 verdict（**任一 drift 解失败强制 DEGRADED**）；F7 28 任务单池 LPT 动态自调度（f7Threads 可配, 硬顶 8 线程, 1=串行回退）。

**错误处理模式**：参数非法抛 `std::invalid_argument`；空输入/f≤0/无线可用（各线 <minPoses）→ success=false；std 异常捕获转 message。

## H. 风险

| 严重程度 | 风险描述 | 影响 | 对策/实测 |
|:--------:|---------|------|---------|
| 🔴 高 | **混线输入**（未按线分组直接喂 Execute/ExecuteMultiLine 单线语义） | rms 31px、t 无效 | 编排层必须按 lineIds 分桶；多线用 ExecuteMultiLine |
| 🔴 高 | 初值落劣质盆地（机械初值直跑） | rms 差 10 倍（4.42 vs 0.44） | 两阶段初始化为调用方必做步骤 |
| 🟡 中 | initialT 用原始相机系值（未乘 R1） | ~15mm z 偏差 | 点云在矫正系——喂 R1·(80,3,3) |
| 🟡 中 | t_z 弱观测（单线 cond~1e11） | 单线 z 散布 −13~−135mm | 多线联合+验收框架锁住；换设备复验 A7 |
| 🟡 中 | perLine rms 为归一化坐标口径（÷σ_u≈440px） | 阈值解读易混淆 | A1 阈值 0.5 按归一化口径标定；像素口径换算待统一 |
| 🟢 低 | 数值雅可比 1e-7 | LM 鲁棒可忽略 | Ceres 交叉验证一致（单线版） |
| 🟢 低 | IRLS/核退火超参固定 | 极端数据可能过/欠收缩 | 自动阶梯 L1/L2 兜底；attempts 可回滚 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | ✅ 可直接使用（factory_calib 现版） |
| **现有模块** | `factory_calib/module2_laser/operators/projector_joint_calib/`（ExecuteMultiLine + 鲁棒优化 + 验收框架，2026-08-26；2026-09-02/03 加速：线内雅可比 `596b855` / 块稀疏 GN `ad6c86b` / 相位计时 `57de7d7` / F7 并发＋假 PASS 修复 `240ff2a`·`dedec91`） |
| **实测** | left_skew 22 姿态 455k 点 acceptance **PASS**（tRect=(40.53,8.47,11.39)mm、rms 0.448px、25 曲线落盘）；加速后载体轨迹同一（字节对拍） |
| **主工程 09 侧** | 仍为单线旧版（无多线/无鲁棒/无验收/无块稀疏）——**同步待裁决**（连同 postErodeSize/topXCount/minArea 删除/realLineTolerance/useBlockJtJ 共六处分歧） |
| **后续** | ~~F3 曲线带建表~~ **已落地为 4-14 curve_map_temp_table**（61 档 sidecar, 见 `docs/operator_docs/curve_map_temp_table/`）→ F4 扫描侧 0.7 对齐（fcscan/fcchain_v3 sidecar 选档已接）→ B1 正向闭环验收 |

---

> **参考卡结束**
