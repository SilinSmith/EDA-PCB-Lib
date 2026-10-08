# -*- coding: utf-8 -*-
"""新补充训练集 bbox 约定终审：把候选约定放到「引脚对称性」下比较。

判据：2 脚/多脚器件的引脚点关于**器件本体中心**应当近似对称。
对每个有 pins 的元件，用各候选约定解出框中心，算「引脚质心 - 框中心」的偏移，
偏移越小说明该约定越对。同时给出「有多少元件的引脚质心落在框内」。
"""
import json
from pathlib import Path
from statistics import median

ROOT = Path(r"Y:\workspace\EDA\新补充训练集")
CANDS = {
    "A_xywh_topleft": lambda x, y, w, h: (x + w / 2, y + h / 2),
    "C_cxcywh_topleft": lambda x, y, w, h: (x, y),
    "B_xywh_bottomleft": lambda x, y, w, h: (x + w / 2, 1.0 - y - h / 2),
}
acc = {k: [] for k in CANDS}
ins = {k: 0 for k in CANDS}
tot = 0
js = sorted(ROOT.glob("*/*.json"))
for jp in js:
    d = json.loads(jp.read_text(encoding="utf-8"))
    comps, pins = d.get("components", {}), d.get("pins", {})
    for key, pd in pins.items():
        c = comps.get(key)
        if not c or not c.get("bbox") or len(pd) < 2:
            continue
        x, y, w, h = [float(v) for v in c["bbox"]]
        pts = [(float(v["point"]["x"]), float(v["point"]["y"])) for v in pd.values()
               if isinstance(v, dict) and isinstance(v.get("point"), dict)]
        if len(pts) < 2:
            continue
        px = sum(p[0] for p in pts) / len(pts)
        py = sum(p[1] for p in pts) / len(pts)
        tot += 1
        diag = (w * w + h * h) ** 0.5 or 1e-9
        for k, fn in CANDS.items():
            cx, cy = fn(x, y, w, h)
            acc[k].append((((px - cx) ** 2 + (py - cy) ** 2) ** 0.5) / diag)
            if abs(px - cx) <= w / 2 and abs(py - cy) <= h / 2:
                ins[k] += 1
print(f"元件（有 >=2 引脚）样本数 = {tot}   JSON 文件 = {len(js)}")
print(f"{'候选约定':22s} {'偏移/对角 中位':>14s} {'偏移/对角 均值':>14s} {'引脚质心在框内':>14s}")
for k in CANDS:
    v = acc[k]
    print(f"{k:22s} {median(v):14.4f} {sum(v)/len(v):14.4f} {ins[k]:>8d}/{tot}"
          f"  ({100*ins[k]/tot:.1f}%)")
