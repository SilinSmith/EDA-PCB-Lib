"""同类近邻 GT 框（corner_dev<=20px）到底是「一颗符号被标了两遍」还是「两颗真符号」。

判据不靠肉眼：在**两个框的并集**里数连通的墨迹团。
  1 团  ⇒ 两颗框套的是同一颗符号（标注重复 / 或本该合成一颗）
  >=2 团 ⇒ 图上有两坨墨，是真的两颗符号挤在一起

用法：python work\\tools\\diag\\_gtdup3.py [--split val] [--tol 20]
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(r"Y:\workspace\EDA\trainset_v3")
NAMES = ['amp', 'amp_3pin', 'amp_5pin', 'ant', 'battary', 'bjt', 'bjt_pnp', 'block', 'box',
         'buzzer', 'c', 'circle_header', 'crystal', 'crystal_2pin', 'crystal_3pin',
         'crystal_4pin', 'd', 'dc', 'esd', 'fuse', 'gnd', 'jumper', 'l', 'led', 'm3螺丝',
         'mosfet', 'mosfet_pnp', 'motor', 'net_bidirection', 'net_input', 'net_output',
         'net_short', 'opt', 'other', 'pin', 'r', 'seg', 'switch', 'testpoint', 'transfomer',
         'v', 'bjt_npn', 'mosfet_npn']
SPLIT, TOL, PAD = "val", 20.0, 3
for i, a in enumerate(sys.argv):
    if a == "--split":
        SPLIT = sys.argv[i + 1]
    if a == "--tol":
        TOL = float(sys.argv[i + 1])


def n_blobs(g: np.ndarray) -> int:
    ink = (g < 180).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    return sum(1 for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 3)


lab_dir, img_dir = ROOT / "labels" / SPLIT, ROOT / "images" / SPLIT
per_cls: dict[str, Counter] = defaultdict(Counter)
hist = Counter()
cache: dict[str, np.ndarray] = {}
cache_name = None

for f in sorted(lab_dir.glob("*.txt")):
    img = img_dir / f"{f.stem}.png"
    if not img.exists():
        continue
    with Image.open(img) as im:
        W, H = im.size
        if img.name != cache_name:
            cache = {"g": cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)}
            cache_name = img.name
    g = cache["g"]
    boxes = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *(float(v) for v in p[1:5])
        boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[0] != b[0]:
                continue
            if max(abs(a[1 + k] - b[1 + k]) for k in range(4)) > TOL:
                continue
            x0 = max(0, int(min(a[1], b[1])) - PAD)
            y0 = max(0, int(min(a[2], b[2])) - PAD)
            x1 = min(W, int(max(a[3], b[3])) + PAD)
            y1 = min(H, int(max(a[4], b[4])) + PAD)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            nb = n_blobs(g[y0:y1, x0:x1])
            key = "1团(=同一颗)" if nb <= 1 else ("2团" if nb == 2 else ">=3团")
            hist[key] += 1
            per_cls[NAMES[a[0]]][key] += 1

tot = sum(hist.values())
print(f"[{SPLIT}] 同类近邻框对 {tot}（容差 {TOL}px）")
for k in ("1团(=同一颗)", "2团", ">=3团"):
    print(f"  {k:<12} {hist[k]:>4}  {hist[k]/max(1,tot):6.1%}")
print(f"\n  {'类':<14}{'对':>5}{'1团':>7}{'2团':>7}{'>=3团':>8}   ← 1团占比")
for k, c in sorted(per_cls.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
    t = sum(c.values())
    print(f"  {k:<14}{t:>5}{c['1团(=同一颗)']:>7}{c['2团']:>7}{c['>=3团']:>8}"
          f"   {c['1团(=同一颗)']/max(1,t):6.1%}")
