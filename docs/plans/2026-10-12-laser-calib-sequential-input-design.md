# 设计：laser_calib 输入装载器支持 left/right 顺序帧布局（自动回退）

- 日期：2026-10-12
- 范围：`factory_calib/module2_laser/`（人工指令授权改动）
- 动机：数据集 `test_data/laser261006-1668/`（1688×1400，left/right 各 29 帧顺序帧 143~171 + handoff + 定稿 config）不满足 `loadLaserInput` 的 `pose_*/L_tube*.png` 目录约定，直接运行报 "no pose_* subdirectories" 退出。此前惯例（DUIBI1005-1）是人工把 left/right 转成 pose_NNN/L_tube0.png+R_tube0.png。本设计把该转换内化到装载器，原始采集目录可直接输入。

## 1. 行为定义

1. **pose_* 优先，行为不变**：`loadLaserInput`（calib_io.cpp）先扫 `pose_*` 子目录（现有逻辑逐字不动）。left_skew / DUIBI1005-1 / CESHI 等既有数据集（含 pose_* 与 left/ 并存的 DUIBI）零影响。
2. **自动回退**：`pose_*` 为空时，扫 `<dir>/left/` 下图像文件（`.png`/`.jpg`，按文件名排序保证可复现），对每个文件找 `<dir>/right/` 同名配对：
   - 缺右对 → warn + skip（对齐现有 L_tube 无 R 对的行为）；
   - 读图失败（空图）→ warn + skip；
   - **每对同名帧 = 1 个 pose × 1 个 tube 帧**（复刻 DUIBI 转换惯例；laser261006-1668 → 29 poses，可与历史基线 23 ok / 6 skipped 对拍）。
3. **诊断名**：顺序帧布局的 `poseDirs[i]` = `left/<文件名>`（如 `left/143.png`），日志/PLF pose_idx 可追溯。
4. **两布局皆无** → 报错退出（错误信息同时说明两种布局约定）。
5. **不做的事**：不加 config 键；不改算子；不改 tools（fcstepdump/fcscan_e2e/fcchain_v3 各有扫描器，不在本次范围）。

## 2. 代码位置

| 文件 | 改动 |
|---|---|
| `calib_io.cpp` `loadLaserInput` | pose_* 为空分支后加 left/right 回退（约 +35 行） |
| `calib_io.h` | 布局注释补 left/right 说明 |
| `laser_calib_cli.cpp` 用法文本 | 补一句 left/right 布局说明 |

## 3. 测试（TDD）

新增 `module2_laser/tests/test_load_laser_input.cpp`（CMake glob 自动注册，Release 链接 fc2_io）：

- **SequentialPairing**：临时目录合成 3 对 left/right 微图 + 最小合法 handoff（复用 test_handoff 的合成手法）→ 断言 3 poses × 1 tube、poseDirs 命名 `left/xx.png`；
- **MissingRightSkipped**：left 3 张、right 2 张 → skip 后 2 poses；
- **PoseLayoutPriority**：pose_* 与 left/ 并存 → 断言走 pose_*（回归保护）；
- **NoLayoutFails**：两布局皆无 → nullopt。

## 4. 验证链

1. Release 增量构建（build_fc2_rel）；
2. 新单测 4 cases 绿；
3. `test_handoff` 回归绿（装载器共用代码路径）；
4. 实跑 `laser261006-1668`（输出目录独立于数据集目录，防止 sidecar 同名覆盖 729MB 参考 `curve_map_temp_table.bin`），对拍历史基线口径（29 poses / 23 ok / 6 skipped）。

## 5. 风险与边界

- 回退仅在 pose_* **完全为空**时触发，不改变任何现有数据集行为；
- 顺序帧布局无 per-pose temp.txt，参考温以 config/handoff 为准（laser261006-1668 已一致 22.5）；
- Debug 构建（无 imgcodecs）下 fc2_io/tests 装载链本就整体跳过，本改动无 Debug 侧影响。
