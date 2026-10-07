"""在 chain 里精确定位斜向跳变，并回溯是哪个结点的配对造成的。"""
import math
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline.pskel import WireExtractor  # noqa: E402
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
case = cases["0101"]
bgr = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
ex = WireExtractor()
res = ex.extract(bgr)
thin, ink, chroma = ex.foreground(bgr)


def cov(a, b):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(L))
    h = 0
    for k in range(n + 1):
        t = k / n
        x = int(round(a[0] + (b[0] - a[0]) * t))
        y = int(round(a[1] + (b[1] - a[1]) * t))
        if 0 <= y < thin.shape[0] and 0 <= x < thin.shape[1] and thin[y, x]:
            h += 1
    return h / (n + 1)


print("=== chain 中 L>40 且非轴对齐（25~65 度）的跳变 ===")
tot = 0
for i, chain in enumerate(res.polylines):
    for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L <= 40:
            continue
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        if 25.0 <= ang <= 65.0:
            tot += 1
            if tot <= 20:
                print(f"  chain#{i:4d} idx{k:5d} L {L:7.1f} ang {ang:5.1f} "
                      f"压thin {cov(a,b):6.1%}  ({a[0]:7.1f},{a[1]:7.1f})->({b[0]:7.1f},{b[1]:7.1f})")
                print(f"        前后顶点: {[(round(p[0],1),round(p[1],1)) for p in chain[max(0,k-2):k+4]]}")
print(f"\n合计 {tot} 条")

print("\n=== chain 中 L>40 且压 thin < 20% 的任意跳变 ===")
bad = 0
for i, chain in enumerate(res.polylines):
    for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L <= 40:
            continue
        c = cov(a, b)
        if c < 0.20:
            bad += 1
            if bad <= 12:
                print(f"  chain#{i:4d} idx{k:5d} L {L:7.1f} 压thin {c:6.1%}  "
                      f"({a[0]:7.1f},{a[1]:7.1f})->({b[0]:7.1f},{b[1]:7.1f})")
print(f"\n合计 {bad} 条")
