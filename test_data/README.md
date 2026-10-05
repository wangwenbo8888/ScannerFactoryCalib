# test_data — 测试资产统一目录（2026-10-05 规范化）

> 本目录集中存放**测试用例使用的固化输入数据与对比（金标准）结果**。
> 完整指纹（SHA256）、基线数值与环境判读规则见 `verify_baseline/基线档案.md`。

```
test_data/
├── camera/
│   ├── input/                              # 相机标定固定输入（CESHI260831，49 对棋盘＋config.json）
│   └── gold/
│       ├── camera_calib.F430.json          # 本机基线（F: OpenCV，49/49 视角）← test_camera_baseline 对照物
│       └── camera_calib.C413.gold44.json   # 历史金标准（C: OpenCV 时代，44 视角；本机不可复现，仅存档）
├── laser/
│   ├── input/                              # 激光标定固定输入（CESHI260831，22 pose＋config＋handoff）
│   └── gold/
│       ├── laser_calib.json                # 激光金标准 ← test_laser_baseline 对照物
│       └── curve_map_temp_table.bin        # CMTT sidecar 金标准（2.2GB，SHA=00C90B6D…）
└── camera261003/
    ├── config.json                         # 真实链分析测试参数（test_camera_chain_analysis）
    └── images/                             # 52 对 L{n}/R{n}.bmp（源侧原名 `1 (k).bmp`，迁入时归一重命名；
                                            #   2026-10-05 起链测试实跑，首次 PASS）
```

## 入库策略（.gitignore）

- **不入 git**（留盘）：`*.png` / `*.bmp` / `*.bin`（图像与 2.2GB sidecar——SHA 指纹登记在基线档案）
- **入 git**：金标准 JSON、config、handoff、本 README——「对比结果」小体积部分全部版本化

## 消费方

| 测试/脚本 | 用的资产 |
|---|---|
| `test_camera_baseline`（ctest, label=baseline） | camera/input ＋ camera/gold/F430 |
| `test_laser_baseline`（ctest, label=baseline） | laser/input ＋ laser/gold/* |
| `test_camera_chain_analysis` | camera261003/images（缺则 SKIP）＋ config.json |
| `verify_baseline/verify_baseline.ps1`（手动/ctest 同一脚本） | 同上基线两项 |

## 约定

1. 新增测试资产一律放本目录（`<主题>/input`＋`<主题>/gold` 分层），不再散入模块子目录
2. 金标准一经固化不得手改；重新生成都必须走全流程并更新 `verify_baseline/基线档案.md` 的 SHA 记录
3. e2e 合成测试（`module1_camera/tests/e2e/`）参数与源码同目录的自包含约定不变，不属于本目录管辖
