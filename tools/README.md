# factory_calib tools — 标定辅助工具集

均为 python 脚本（免编译，`python tools/<名>.py` 直接跑；工作目录 factory_calib 根）。

## 标定采集与勘探

| 工具 | 用法 | 用途 |
|---|---|---|
| `calib_shot_monitor.py [数据集目录]` | 监听式 | 拍一张判一张（过曝/粘连/线数）——命令行版判定（GUI 面板的脚本形态，采集时无 GUI 可用） |
| `calib_pose_classify.py <数据集> [--expect 25]` | 采集后 | **pose 质量体检**：好帧/坏帧分类（黑帧/过曝档/碎噪声/组件异常＋坏因＋双眼指标），混档帧一眼清 |
| `calib_pjc_grid.py <数据集> [--z "…"] [--xy x,y]` | 标定时 | **PJC initialT 网格勘探**：不同 z 初值全量跑，找最深盆地（自由收敛漂移时用——z+11 深盆地 rms 6.7→5.0 实证） |

## 内置工具（C++，随 Release 构建）

`fcstepdump`（逐帧转储/盆地探针/PLY 分析等子命令）、`fcscan_e2e`（扫描端到端）、
`fcchain_v3`（V3 全链）——用法见各 `--help` 与 AGENTS 工程说明。

## 使用纪律

- 数据集目录须含 `config.json`＋`camera_calib.json`（handoff）＋`pose_*`
- 采集操作规范见主仓 `docs/用户文档/标定数据采集指南.md`（环境/调参/铁律）
- GUI 关窗自动清 data_in 图像——**标定完成前勿关 GUI**；历史数据在 `data_bak/<时间戳>/`
- 删除 pose 目录须人工确认（工程红线）
