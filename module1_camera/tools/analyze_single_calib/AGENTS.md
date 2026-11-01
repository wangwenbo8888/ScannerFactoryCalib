# AGENTS.md — analyze_single_calib 单次标定分析工具

> 单文件 Node 脚本：对**单次** camera_calib.exe 产物（JSON＋逐帧/逐角点 CSV）做体检式分析。零依赖、不进 CMake、不重算标定。设计与实施档案：`docs/plans/2026-10-07-单次标定分析工具analyze_single_calib-design.md`（主仓）。

## 一句话用法

```powershell
node analyze_single_calib.mjs <dataset_dir> [--out 目录] [--reproj-th px] [--row-align-th px] [--stereo-th px] [--stereo-rms-th px] [--concession-rounds N]
```

## 输入要求

| 输入 | 必需 | 用途 |
|---|---|---|
| camera_calib.json | ✓ | 参数本体/门禁上下文 |
| camera_calib_perframe.csv | ✓ | 帧级全量（含失败帧行） |
| camera_calib_percorner.csv | ✓ | 角点级残差/双域极线/半径 |
| left/*.png | 可选 | image_size 审计读 PNG IHDR |
| config.json / frames_meta.csv | 可选 | config 规范化哈希 / 温曝相关 |

三件套缺一 → 退出码 2＋原因清单，不出半份结论。

## 输出（默认 `<dataset_dir>/analysis_single/`）

| 文件 | 内容 |
|---|---|
| report.md | 结论卡（灯号/仲裁/门禁表/建议动作码/熔断）＋五节折叠（审计/分布离群/归因/指纹/口径附录） |
| summary.json | 机器契约 schema `factory_calib.single_calib_summary.v1`（semver additive-only，批量工具的口粮） |
| charts/*.svg | 逐帧条形（双阈值线）/半径分桶双系列/残差矢量场 |

## 必读口径注记

1. **row_align 为主极线指标**（矫正域）；epi_raw 仅对照不设门禁——原始像素×F 口径含 ~0.3px 伪影基线（2026-10-07 实证，完美自洽标定量 0.318px）
2. **G2 阈值**：默认 0.05（2026-10-07 统一门禁 G1 0.05 / G2 0.05 / G3 0.07；2026-10-14 增 G4 立体 RMS 0.30）；`--row-align-th` 可覆盖；噪声地板 σ_floor=1.4826×MAD(row_align_dy) 仍计算并输出、仅作审计参考线（3σ）；PASS 双条件 P50≤T 且 P90≤2T
3. **G4 立体 RMS**（2026-10-14 增，v1.1.0）：读 `extrinsic.stereoReprojError`（cv::stereoCalibrate 返回值，含 R/T 拟合与平方放大口径——与 G3 逐帧 stereo_reproj 不同源、更贴扫描实际）；默认 0.30、`--stereo-rms-th` 可覆盖；字段缺席时 G4 跳过不影响灯号
4. 离群帧报**原始文件名**（perframe CSV 的 left_file）；view# 仅辅助
5. 指纹 R1-R8 为单批化口径 `v2-single`（最小 N 守卫；全态证据输出）；**共性问题定罪归批量层工具**，本工具只出证据
6. verdict 分级 PASS/WARN/FAIL＋到阈距离；判定仲裁优先级 审计＞门禁＞解稳＞哨兵
7. summary 内嵌数值为批量直需最小集，完整数据以 CSV/JSON 引用为准（单一真源）

## 退出码

0 正常；1 report/summary 写失败（终端表先印）；2 输入不完整/无法判定。

## 与周边工具分工

- `analyze_camera_calib`（旧）：跨批对比＋legacy 单批——保留不动，其单批节在本工具验收后标 deprecated
- 未来批量工具：消费本工具 summary.json（九类读数＋位姿分箱键）；跨批三件套（投影差异/Δ 首列/view 重现）由批量层实现

## 维护约定

- 只改 `analyze_single_calib.mjs` 一个文件；无注释（仓库风格）；无测试框架，验收靠真实数据回归（基准见设计文档 §11：camera261006-1688 七项对拍）
- 指标口径变更 bump estimator 版本（向后兼容 additive-only）；指纹规则调整 bump `fingerprint_logic_version`
- 统计核（pearson/ibeta/logLogSlope）与旧工具逐位同源，改动需双向对拍
