# -*- coding: utf-8 -*-
"""calib_shot_monitor.py — 标定采集参数实时判定器（左斜/斜线形态）

用法：python calib_shot_monitor.py [数据集目录]
  默认监听 data_in/laser/left_skew；每 1s 轮询，发现新拍的 pose 立即分析。
  在 GUI 里每调一档亮度就拍一张测试帧，看本工具的判定，直到「✅ 可以采集」。

判定三项（261108 两批数据标定的窗口）：
  ① 过曝占比   目标 2.8%~5.0%（>6% 粘连假线 / <2.5% 丢暗线——右斜成功批 3.89%）
  ② 游程>30px  目标 0（斜线正常游程 8~22px；>30=粘连）
  ③ 线数估计   目标 25±1（取画面中部 3 个横切行的游程计数中位）
双眼必须全部通过（R 眼是短板——重点看 R）。
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

MASK_THR = 120      # 与标定 config mask.threshold 同口径
EXPECT_LINES = 25   # 左斜/右斜 25 线（精细改 7）
HOT_LO, HOT_HI = 2.8, 5.0
RUN_MAX = 30


def analyze(img):
    h, w = img.shape
    hot = float((img > 240).sum()) / img.size * 100
    bw = (img > MASK_THR).astype(np.uint8)
    # 游程：全图每 16 行采样
    runs, wide_runs = [], 0
    line_counts = []
    for y in range(0, h, 16):
        row = bw[y]
        x = 0
        cnt = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                r = x - x0
                runs.append(r)
                if r > RUN_MAX:
                    wide_runs += 1
                cnt += 1
            else:
                x += 1
        line_counts.append(cnt)
    med_lines = int(np.median(line_counts)) if line_counts else 0
    return hot, wide_runs, med_lines


def verdict(side, hot, wide, lines):
    ok = (HOT_LO <= hot <= HOT_HI) and wide == 0 \
         and abs(lines - EXPECT_LINES) <= 1
    parts = []
    parts.append(("过曝", f"{hot:.2f}%", HOT_LO <= hot <= HOT_HI))
    parts.append(("粘连", f"{wide}条>30px", wide == 0))
    parts.append(("线数", f"{lines}(目标{EXPECT_LINES})", abs(lines - EXPECT_LINES) <= 1))
    s = "  ".join(f"[{'✓' if p[2] else '✗'}]{p[0]}={p[1]}" for p in parts)
    return ok, s


def advice(l, r):
    tips = []
    for side, (hot, wide, lines) in (("L", l), ("R", r)):
        if lines < EXPECT_LINES - 1:
            tips.append(f"{side}眼线数不足({lines})——亮度/激光档【升】半档")
        if lines > EXPECT_LINES + 1:
            tips.append(f"{side}眼多线({lines})——补光/激光档【降】半档")
        if wide > 0:
            tips.append(f"{side}眼有粘连({wide}条)——激光档【降】")
        if hot > HOT_HI:
            tips.append(f"{side}眼过曝{hot:.1f}%——降档")
        if hot < HOT_LO and lines >= EXPECT_LINES - 1:
            tips.append(f"{side}眼偏暗({hot:.1f}%)但线齐——可用")
    return tips or ["双眼达标"]


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "data_in/laser/left_skew"
    print(f"监听 {base}（拍一张判一张；EXPECT_LINES={EXPECT_LINES}，Ctrl+C 退出）")
    seen = set()
    while True:
        pngs = glob.glob(os.path.join(base, "pose_*", "*_tube*.png"))
        fresh = [p for p in pngs if p not in seen]
        for p in sorted(fresh, key=os.path.getmtime):
            seen.add(p)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            side = "L" if os.path.basename(p).startswith("L") else "R"
            hot, wide, lines = analyze(img)
            ok, s = verdict(side, hot, wide, lines)
            tag = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {s} {'✓' if ok else '✗'}")
            # 一对 L/R 到齐才给总结论
            pair = glob.glob(os.path.join(os.path.dirname(p), "*_tube*.png"))
            if len(pair) >= 2 and all(x in seen for x in pair):
                res = {}
                for x in pair:
                    im = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        continue
                    sd = "L" if os.path.basename(x).startswith("L") else "R"
                    res[sd] = analyze(im)
                if len(res) == 2:
                    okL = verdict("L", *res["L"])
                    okR = verdict("R", *res["R"])
                    if okL[0] and okR[0]:
                        print("════════ ✅ 可以采集（双眼全过，按此档采满 pose）════════")
                    else:
                        for t in advice(res["L"], res["R"]):
                            print(f"    → {t}")
        time.sleep(1)


if __name__ == "__main__":
    main()
