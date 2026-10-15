# -*- coding: utf-8 -*-
"""§5.111 取证（五）：引脚桩的**颜色剖面** —— 桩（符号色，暗红）与导线（绿）分得开吗？

沿某个引脚的外法线方向打印 BGR 剖面，并给出「暗红（桩）」与「绿（导线）」的连续段，
用来验证「引脚点 = 桩尖（暗红段末端）」这个假设。

用法：python -X utf8 work\\_pincol.py --case 8 --comp U1 [--run gsd9] [--pins pin_1,pin_3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc   # noqa: E402


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get('x'), raw.get('y'))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return float(raw[0]), float(raw[1])


ap = argparse.ArgumentParser()
ap.add_argument("--case", default="8")
ap.add_argument("--comp", default="U1")
ap.add_argument("--pins", default="")
ap.add_argument("--span", type=int, default=40)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cas = cases[args.case]
gt = normalise_doc(json.load(open(str(cas.json_path), encoding="utf-8")))
img = cv2.imread(str(cas.png_path), cv2.IMREAD_COLOR)
H, W = img.shape[:2]
print("图 %dx%d  case %s" % (W, H, args.case))
gb = (gt['components'].get(args.comp) or {}).get('bbox')
print("框 %s" % (gb,))
x0, y0, x1, y1 = [float(v) for v in gb]
gpins = gt['pins'].get(args.comp) or {}
want = set(args.pins.split(",")) if args.pins else None
for pk in sorted(gpins, key=lambda s: (len(str(s)), str(s))):
    if want and pk not in want:
        continue
    p = _pt((gpins[pk] or {}).get('point'))
    if not p:
        continue
    left = p[0] < (x0 + x1) / 2.0
    ey = int(round(H - p[1]))
    print("\n%s  GT=(%.0f,%.0f) doc → 图 (%.0f,%d)  %s侧  框边 x=%.0f"
          % (pk, p[0], p[1], p[0], ey, "左" if left else "右", x0 if left else x1))
    rows = []
    meta = []
    rng = range(int(x0) - 1, int(x0) - 1 - args.span, -1) if left else \
          range(int(x1) + 1, int(x1) + 1 + args.span)
    for x in rng:
        if not (0 <= x < W and 0 <= ey < H):
            continue
        b, g, r = [int(v) for v in img[ey, x]]
        # 暗红=桩：R 明显高、G/B 低；绿=导线：G 高
        kind = "."
        if b < 110 and g < 110 and r > 90 and r - max(g, b) > 25:
            kind = "S"          # 桩（symbol）
        elif g > 90 and g - max(r, b) > 25:
            kind = "W"          # 导线（wire）
        elif r < 110 and g < 110 and b < 110:
            kind = "K"          # 黑
        rows.append((x, kind, "x=%4d %3d,%3d,%3d %s%s"
                     % (x, r, g, b, kind, "  <== GT" if abs(x - p[0]) < 1.5 else "")))
        meta.append(kind)
    # 压缩输出：只打印有颜色变化的段
    prev = None
    for (x, kind, row) in rows:
        if kind != prev:
            print("   " + row)
            prev = kind
        elif "GT" in row:
            print("   " + row)
    # 汇总段
    kinds = [k for (_, k, _) in rows]
    xs = [x for (x, _, _) in rows]
    runs = []
    for k, x in zip(kinds, xs):
        if runs and runs[-1][0] == k:
            runs[-1][2] = x
        else:
            runs.append([k, x, x])
    print("   段（从框边向外）: " + "  ".join("%s[%d..%d]" % (k, a, b) for k, a, b in runs))
