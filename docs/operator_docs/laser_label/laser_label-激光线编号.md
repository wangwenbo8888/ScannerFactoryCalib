# 激光线编号（laser_label）

> 依据《算子提炼模板.md》v1.0 编写；对象为 `module2_laser/operators/laser_label/`。2026-10-07 增**双门禁**（编号错位事故修复），本卡为该算子首张参考卡。

## A. 标识

| 字段 | 内容 |
|------|------|
| 算子编号 | `激光标定-9`（4-3） |
| 运行平台 | CUDA（OpenCV CUDA GpuMat + 自写 kernel + thrust 排序） |
| 所属流程 | 激光标定链 4-3：CCL 标记掩膜 → 按扫描线交点位置顺序重编号 1..N |
| 精度档次 | `②`（整像素/几何类） |

## B. 数据流

| 方向 | 数据 | 类型 |
|------|------|------|
| 输入① | 连通域标记掩膜（来自 4-2） | `GpuMat` CV_32SC1 |
| 输出① | 重编号掩膜 | `shared_ptr<GpuMat>` CV_32SC1 |
| 输出② | 有效线数（=componentCount=**lineCount**） | `int` |
| 输出③ | **frameRejected/rejectReason**（2026-10-07 门禁诊断） | `bool`/`string` |

## C. 算法

```
Step 1   CCL 输入直接消费（maxLabels 校验）
Step 2   扫描线: 竖切中心列(scanDirection=0)/横切中心行(=1, 斜线形态)
Step 3.5 游程分解 → ★门2 maxRunWidth(split 前测原始游程展布=末段末-首段起,含间隙)
         → splitMergedRuns: 同标签≥2 游程段按间隙中点拆分(仅横切)
Step 5/6 扫描线取每标签 minX(minY) → 按交点坐标排序
Step 6.5 realLine 等距甄别(tol=10px, 局部等距判据+端点传播) → 编号 1..N
         ★门1 expectedLineCount: 编号后线数≠期望 → frameRejected
Step 7/8 映射表重标 → 输出
```

**编号本质**：交点**位置顺序**分配（无绝对锚定）——任何中部缺线/粘连导致后段整体错位，此为门禁存在的根本理由。

## D. 依赖与衔接

```
4-2 region_analyze (d_labeledMask) → laser_label → 4-4 steger(ByLabel 分组)
上游另吃 4-1 的灰度无关；CLI 侧: 任一目 frameRejected → 整帧跳过(L+R 同弃)
```

## E. 架构

`laser_label_cuda.h/.cpp/_pimpl.h/_impl.cu + tests/test_laser_label_cuda.cpp`（44 cases：36 既有＋8 门禁）；核心类 `calib::LaserLabelerCUDA`；日志标签 `09-LaserLabelerCUDA`。09 侧 `calibration/laser_calib/laser_label/` 逐字镜像（fc 校验一致）。

## F. 参数

| 参数 | 默认(算子/主库) | 说明 |
|------|------|------|
| scanDirection | 0 / 数据集(斜线=1) | 0=竖切中心列, 1=横切中心行 |
| centerRowOffset / centerColOffset | 0 | 扫描线偏移 |
| splitMergedRuns | true | 间隙中点拆分粘连游程 |
| realLineTolerance | 0 / 主库 10 | 等距甄别容差(px, 0=off) |
| **expectedLineCount** | 0(off) / **主库 25** | **门1**：编号后线数≠期望→frameRejected。25 线设备中部缺 1 条→后段错号→必须整帧弃 |
| **maxRunWidth** | 0(off) / **主库 30** | **门2**：切割游程展布>阈→判粘连(过曝融合)→frameRejected。实测正常 12~22px/粘连 38~132px，30 落空档；**split 前测量**（防拆分凑回 25 线漏检——147-149 帧实证） |
| maxLabels / deviceId | 256 / 0 | 常规 |

## L. 关键实测（laser261006-1668 事故，2026-10-07）

根因链：高光过曝→相邻线粘连→split 凑数但编号错位→错号点毒化 PJC（Sampson 37.98px、A5/A6/A7≈1480mm）。双门禁上线路 11/29 脏帧全中（count 门 7 帧＋runWidth 门独杀 147-149 三帧），干净 18 帧结果与人工挑帧**逐位一致**；旧 LASER 基线回归 PASS 无误伤（另揪出 1 帧漏网 61px 粘连仍 PASS）。门禁语义：执行 success 不变，拒绝走 `frameRejected`＋`rejectReason`（如 `runWidth=132>maxRunWidth(30) label=5; lineCount=23!=expectedLineCount(25)`）。
