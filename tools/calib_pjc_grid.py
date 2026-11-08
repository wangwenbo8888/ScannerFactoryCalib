# -*- coding: utf-8 -*-
"""calib_pjc_grid.py — PJC initialT 网格扫描（盆地勘探）

用法：
  python calib_pjc_grid.py <数据集目录> [--z "-20,-10,0,5,10,15,20,30"] [--xy 31.1,-6.4]

对同一数据集用不同 z 初值（xy 固定）全量跑标定，输出各盆地收敛的 rms/A5/A6。
用途：PJC 自由收敛盆地漂移时（261108 左斜实证：自由跑 z=-1.5、网格锚到 z+11
深盆地 rms 6.7→5.0）确定最深盆地并产出最优 json。

注意：全量跑（非挑帧子集）——门禁与 PJC 前段无关，过门帧由数据决定。
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # factory_calib 根
EXE = os.path.join(BASE, "build_fc_all", "module2_laser", "Release", "laser_calib.exe")
GUI_LIB = os.path.join(BASE, "build_fc_all", "gui_qt", "Release")


def run_once(dataset, out_json, init_t, cfg_preset):
    cfg = dict(cfg_preset)
    if init_t:
        cfg["pjc"] = {"initialT": init_t}
    else:
        cfg.pop("pjc", None)
    json.dump(cfg, open(os.path.join(dataset, "config.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    log = out_json.replace(".json", ".log")
    env = dict(os.environ)
    env["PATH"] = GUI_LIB + ";" + env["PATH"]
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run([EXE, dataset, out_json], stdout=f,
                       stderr=subprocess.STDOUT, env=env, cwd=BASE)
    out = io.open(log, encoding="utf-8", errors="replace").read()
    m = re.search(r"rms [\d.]+->([\d.]+)", out)
    t = re.search(r"projectorT=\(([-\d., ]+)\)", out)
    a5 = re.search(r"A5 lineDrift=([\d.]+)", out)
    a6 = re.search(r"A6 poseDrift=([\d.]+)", out)
    oks = re.findall(r"pose \d+ tube 0: 4-4 L n=\d+", out)
    return dict(rms=float(m.group(1)) if m else None,
                T=t.group(1) if t else "?",
                a5=a5.group(1) if a5 else "?", a6=a6.group(1) if a6 else "?",
                n_ok=len(oks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="数据集目录（须含 config.json + camera_calib.json + pose_*）")
    ap.add_argument("--z", default="-20,-10,0,5,10,15,20,30",
                    help="z 初值列表（逗号分隔；含 free=不加 initialT）")
    ap.add_argument("--xy", default="31.1,-6.4", help="x,y 初值（逗号分隔，取自同机好解）")
    ap.add_argument("--out", default=None, help="输出目录（默认 dataset 同级 data_out）")
    a = ap.parse_args()

    cfg_preset = json.load(open(os.path.join(a.dataset, "config.json"),
                                encoding="utf-8"))
    outdir = a.out or os.path.join(BASE, "data_out")
    os.makedirs(outdir, exist_ok=True)
    xy = [float(v) for v in a.xy.split(",")]

    print(f"{'z_init':>7} | {'rms':>8} | {'A5':>6} | {'A6':>6} | ok | projectorT")
    best = (None, 9e9, None)
    for zs in a.z.split(","):
        zs = zs.strip()
        if zs.lower() == "free":
            tag, init_t = "free", None
        else:
            tag, init_t = f"z{zs}", [xy[0], xy[1], float(zs)]
        out_json = os.path.join(outdir, f"pjcgrid_{os.path.basename(a.dataset)}_{tag}.json")
        r = run_once(a.dataset, out_json, init_t, cfg_preset)
        print(f"{zs:>7} | {r['rms']!s:>8} | {r['a5']:>6} | {r['a6']:>6} | {r['n_ok']:>2} | {r['T']}")
        if r["rms"] and r["rms"] < best[1]:
            best = (zs, r["rms"], out_json)
    print(f"\n最优: z_init={best[0]} rms={best[1]:.4f}\n产物: {best[2]}")


if __name__ == "__main__":
    main()
