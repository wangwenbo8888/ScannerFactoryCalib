# 曲线映射温度表·CUDA 版（cmtt_cuda）

> 依据《算子提炼模板.md》v1.0 编写（精简 8 段）；对象为 `module2_laser/operators/cmtt_cuda/`——CMTT 61 档编排的 **CUDA 主引擎**（2026-10-08 转正），CPU 版 `curve_map_temp_table/`（12d）退居热备。

## A. 标识

| 字段 | 内容 |
|------|------|
| 算子编号 | `激光标定-12d-CUDA`（4-14 现役默认引擎） |
| 运行平台 | 编排为 CPU（档级串行），计算委托 `curve_map_cuda`（GPU） |
| 所属流程 | laser_calib_cli 4-14：三源装配 → 61 档生成 → sidecar 落盘＋sha256 |
| 切换 | 主库/数据集 `cmtt.cudaEnabled`：true=CUDA 主（默认）/false=CPU 备 |

## B. 数据流

```
输入（与 CPU 版同契约）: PJC emissionCurves + stereoRectifyTempTable(源①)
                        + laserExtrinsicTempTable(5-3) + imageSize
输出: CurveMapTempTableResult（tiers/sidecarBytes/clampedRatio/gridHash——类型复用）
```

## C. 算法（复刻 CPU 编排，仅计算核换 GPU）

```
参考档(零插值+原始曲线, curve_map_cuda 生成) → blob 头直解冻结基准(rowStep/roi/
rowCount/rowMin/depth 六字段) → FNV gridHash → 温度网格逐档:
  at(temp) 插值 → rescaleCurve 重标(s=f'/f_ref, m=pp'−s·pp_ref) → curve_map_cuda
  生成 → cmttTierGridMatches 冻结断言 → 过则入档, 不过降级 missing
→ CmttBuilder 拼装 sidecar（96B 头+61×32B 索引+blob）
```

与 CPU 版差异仅两处：档级**串行**（GPU 单档内部已并行，无 omp 档级池）；tierThreads 参数不消费。`TempParamInterpolator`/`rescaleCurve`/`cmttTierGridMatches`/`CmttBuilder` 四件**直接 include CPU 版**（公开 API），本卡不重复其描述。

## D/E. 依赖与架构

```
curve_map_cuda(计算核) + curve_map_temp_table.h(共用件) + cmtt_container
cmtt_cuda.h/.cpp + tests/test_cmtt_cuda.cpp（4 cases, 1 env 门控）
核心类 calib::CmttCudaGenerator（Generate 同签名同返回类型）
日志标签 XD-CmttCuda；构建: cmtt_cuda.cpp 入 fc2_ops 通配, .cu 在 fc2_curve_map_cuda
```

## J/L. 环境与关键实测（2026-10-08）

**性能**：61 档真实数据 **113 秒**（CPU 版 24.8 分钟，档级 t16 并行）；CLI 全链 31.4→3.5 分钟（EXIT=0/verdict PASS 逐指标相同）。**等价性边界**：sidecar 结构一致、参考档逐条目一致（27,538,991==27,538,991，真实全尺度对拍入测试）；其余 60 档浮点边界允许极小条目差（实测同量级 27,538,132~27,539,589 波动内）——**故 CUDA 产物 sha256 不承诺与 CPU 版相同**；CPU 版线程数无关确定性红线不受影响，CUDA 同架构逐次可复现。测试：①合成单档精确一致②合成 9 档 vs 生产 CPU 生成器③真实参考档 vs 交付 sidecar④FC_CMTT_CUDA_FULL=1 全量计时。09 侧 CMTT 仍 CPU——07-E 提速需另行移植（同 laser_match 债务模式）。
