# -*- coding: utf-8 -*-
"""calib_pose_classify.py — 标定 pose 质量分类器

用法：
  python calib_pose_classify.py <数据集目录> [--expect 25]

对每个 pose 双眼量化（组件数/过曝/背景），分出好帧/坏帧（含坏因）。
用途：采集后快速体检——混档帧/黑帧/过曝帧一眼清（261108 左斜战役工具，
当时混档帧曾致整批被门禁拒绝）。

判定口径（右斜 PASS 批 14~19 标定）：
  黑帧    : 组件 <10 或 过曝 <1%
  过曝档  : 过曝 >5.5%
  碎噪声  : 组件 >38
  正常    : 组件 18~38 且过曝 1~5.5%（过曝下限取消——线亮由组件数反映）
"""
import argparse
import glob
import os

import cv2
import numpy as np


def metrics(img, mask_thr=120):
    bw = (img > mask_thr).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    bw = cv2.erode(bw, k)   # 与标定 config erodeSize=1 同口径
    n, lab, st, ce = cv2.connectedComponentsWithStats(bw * 255, 8)
    comps = sum(1 for i in range(1, n)
                if st[i][4] >= 500 and max(st[i][3], st[i][2]) >= 50)
    hot = float((img > 240).sum()) / img.size * 100
    bg = float(np.median(img))
    return comps, hot, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--expect", type=int, default=25, help="目标线数（精细 7）")
    a = ap.parse_args(a=None if False else None) if False else None
    args = ap.parse_args()

    good, bad = [], []
    for d in sorted(glob.glob(os.path.join(args.dataset, "pose_*"))):
        name = os.path.basename(d)
        lf, rf = os.path.join(d, "L_tube0.png"), os.path.join(d, "R_tube0.png")
        if not (os.path.exists(lf) and os.path.exists(rf)):
            bad.append((name, "空目录"))
            continue
        lc, lh, lb = metrics(cv2.imread(lf, 0))
        rc, rh, rb = metrics(cv2.imread(rf, 0))
        reasons = []
        if max(lc, rc) < 10 or min(lh, rh) < 1.0:
            reasons.append("黑帧")
        if max(lh, rh) > 5.5:
            reasons.append("过曝档")
        if min(lc, rc) > 38:
            reasons.append("碎噪声过多")
        if not reasons and not (18 <= min(lc, rc) and max(lc, rc) <= 38):
            reasons.append(f"组件异常(L{lc}/R{rc})")
        line = f"L组件{lc}/过曝{lh:.1f}%/bg{lb:.0f}  R组件{rc}/过曝{rh:.1f}%/bg{rb:.0f}"
        if reasons:
            bad.append((name, "、".join(reasons) + f"  [{line}]"))
        else:
            good.append(f"{name}  [{line}]")

    print(f"===== 好帧 {len(good)} =====")
    for g in good:
        print("  " + g)
    print(f"===== 坏帧 {len(bad)}（可删） =====")
    for n, w in bad:
        print(f"  {n}  ← {w}")
    print(f"\n提示：好帧 ≥5 可标定；<5 建议同条件补采。删除坏帧目录需人工确认。")


if __name__ == "__main__":
    main()
