"""沿那条 45 度斜线逐级采样：ink -> thin -> skel，看它在哪一级出现。"""
import math
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline.pskel import WireExtractor, zhang_suen, rdp, snap_polyline  # noqa: E402
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
case = cases["0101"]
bgr = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
ex = WireExtractor()
thin, ink, chroma = ex.foreground(bgr)
skel = zhang_suen(thin, max_iter=ex.max_thin_iter)

DIAG = [((1179.0, 224.0), (980.0, 25.0)), ((19.0, 639.0), (208.0, 828.0))]
for a, b in DIAG:
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = int(L)
    for name, m in (("ink", ink), ("thin", thin), ("skel", skel > 0)):
        h = 0
        for i in range(n + 1):
            t = i / n
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]:
                h += 1
        print(f"  ({a[0]:.0f},{a[1]:.0f})->({b[0]:.0f},{b[1]:.0f}) len {L:6.1f}  "
              f"{name:5s} 覆盖 {h/(n+1):6.1%}")

# 在斜线附近 ±3px 内找骨架像素
for a, b in DIAG:
    x1 = int(min(a[0], b[0])) - 6
    y1 = int(min(a[1], b[1])) - 6
    x2 = int(max(a[0], b[0])) + 6
    y2 = int(max(a[1], b[1])) + 6
    x1, y1 = max(0, x1), max(0, y1)
    sub = (skel[y1:y2, x1:x2] > 0).astype(np.uint8)
    print(f"\n斜线 bbox ({x1},{y1})-({x2},{y2})  骨架像素 {int(sub.sum())}")
    # 打印下半区的 ASCII 缩略（每 4px 一格）
    step = 4
    lines = []
    for r in range(0, sub.shape[0], step):
        row = ""
        for c in range(0, sub.shape[1], step):
            row += "#" if sub[r:r + step, c:c + step].any() else "."
        lines.append(row)
    for ln in lines[:60]:
        print("   ", ln)
