"""查 0101 里那些「莫名其妙的大尺度斜线」。

对每条提取线段算长度与「与最近 0/45/90/135 度的偏差」，
把长的斜线（30°~60°）挑出来，到原图上量它上面的压墨率，
并渲染裁剪图看它到底压着什么。
"""
import json
import math
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "0101"
OUT = os.environ.get("STAGE1_OUT", r"Y:\workspace\EDA\work\stage1_all6")
PREV = r"Y:\workspace\EDA\work\preview"

cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
case = cases[CASE]
bgr = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
H, W = gray.shape
d = json.load(open(os.path.join(OUT, f"{CASE}.stage1.json"), encoding="utf-8"))
segs = d["wire_detail"]["segments"]
print(f"{CASE}  {W}x{H}  线段 {len(segs)}")

ink = (gray < 170).astype(np.uint8)


def dev_deg(a, b):
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
    m = ang % 45.0
    return min(m, 45.0 - m)


rows = []
for s in segs:
    a = (float(s[0][0]), float(s[0][1]))
    b = (float(s[1][0]), float(s[1][1]))
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    rows.append((L, dev_deg(a, b), a, b))

rows.sort(key=lambda r: -r[0])
print(f"\n最长 20 条线段：")
print(f"  {'len':>8s} {'dev°':>6s}  {'端点'}")
for L, dv, a, b in rows[:20]:
    print(f"  {L:8.1f} {dv:6.1f}  ({a[0]:7.1f},{a[1]:7.1f}) -> ({b[0]:7.1f},{b[1]:7.1f})")

# 斜线 = dev 接近 0 且整体角度在 30~60 度
diag = []
for L, dv, a, b in rows:
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
    if 25.0 <= ang <= 65.0 and L >= 50.0:
        diag.append((L, ang, dv, a, b))
print(f"\n长斜线（长度>=50 且整体角度 25~65 度）：{len(diag)} 条")
for L, ang, dv, a, b in diag[:15]:
    # 沿线采样压墨率
    n = max(2, int(L))
    hits = 0
    for i in range(n + 1):
        t = i / n
        x = int(round(a[0] + (b[0] - a[0]) * t))
        y = int(round(a[1] + (b[1] - a[1]) * t))
        if 0 <= y < H and 0 <= x < W and ink[y, x]:
            hits += 1
    print(f"  len {L:7.1f} ang {ang:5.1f} dev {dv:4.1f} 压墨 {hits/(n+1):5.1%}  "
          f"({a[0]:6.1f},{a[1]:6.1f})->({b[0]:6.1f},{b[1]:6.1f})")

# 渲染最长 3 条斜线的裁剪
os.makedirs(PREV, exist_ok=True)
for i, (L, ang, dv, a, b) in enumerate(diag[:3]):
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    hw, hh = int(L * 0.7) + 60, int(L * 0.7) + 60
    x1, y1 = max(0, int(cx - hw)), max(0, int(cy - hh))
    x2, y2 = min(W, int(cx + hw)), min(H, int(cy + hh))
    crop = bgr[y1:y2, x1:x2].copy()
    p1 = (int(a[0] - x1), int(a[1] - y1))
    p2 = (int(b[0] - x1), int(b[1] - y1))
    cv2.line(crop, p1, p2, (0, 0, 255), 3)
    cv2.circle(crop, p1, 8, (0, 255, 0), 2)
    cv2.circle(crop, p2, 8, (0, 255, 0), 2)
    p = os.path.join(PREV, f"diag_{CASE}_{i}.png")
    cv2.imwrite(p, crop)
    print(f"  crop -> {p}  ({x1},{y1})-({x2},{y2})")

# 统计：长线段的长度分布
Ls = np.array([r[0] for r in rows])
print(f"\n线段长度 p50 {np.percentile(Ls,50):.1f} p90 {np.percentile(Ls,90):.1f} "
      f"p99 {np.percentile(Ls,99):.1f} max {Ls.max():.1f}")
print(f"长度>=50 的线段 {int((Ls>=50).sum())} 条，其中斜线 {len(diag)} 条")
