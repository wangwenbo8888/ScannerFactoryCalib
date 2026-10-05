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
├── laser1003-2/
│   └── calib_input/                        # ★ CLI 在线标定就绪（CESHI261003-1 批次，11 pose）：
│                                            #   pose_01..11/{L,R}_tube0.png＋temp.txt(25.0)
│                                            #   ＋handoff（camera261003 重标定产物）＋config
│                                            #   实测跑通（41s）：accum=222575、PJC projectorT 与
│                                            #   22-pose 金标准差 <0.2mm、CMTT 61/61；
│                                            #   verdict=DEGRADED（11 pose 少于建议 15-25，A5/A6/A7
│                                            #   漂移门禁超）——产物已出，产线采用前建议补采姿态
└── camera261003/
    ├── config.json                         # 真实链分析测试参数（test_camera_chain_analysis）
    ├── images/                             # 52 对 L{n}/R{n}.bmp（⚠ 2026-10-05 已做左右互换修正：
    │                                       #   原迁移方向 T_x=+131/P2[0][3] 为正，矫正视差全负致
    │                                       #   激光匹配 0；互换后与工作基线同号，链测试 32s PASS）
    └── calib_input/                        # ★ CLI 在线标定就绪数据集（camera_calib.exe 直接消费）：
                                            #   left/01..52.png + right/01..52.png（bmp→png＋同名配对，
                                            #   左右已互换）＋config.json（11×8/15mm 装机口径）；
                                            #   实测 52 装载、51 有效，T_x=−131.28、P2[0][3]=−168901
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
