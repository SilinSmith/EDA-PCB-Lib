"""冒烟：`pskel.cross_core_fill` 到底补了多少像素、交叉处骨架有没有接上。

用法：`python -X utf8 work\\_crosssmoke.py --case 1 --x 842 --y0 160 --y1 190`
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline import pskel  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--x", type=int, default=842)
    ap.add_argument("--y0", type=int, default=160)
    ap.add_argument("--y1", type=int, default=190)
    ap.add_argument("--r", type=float, default=8.0)
    ap.add_argument("--axes", type=int, default=2)
    a = ap.parse_args()
    c = None
    for cc in discover(DEFAULT_DATA_ROOT):
        if str(getattr(cc, "case_id", "") or "") == a.case:
            c = cc
            break
    if c is None:
        print("找不到 case", a.case)
        return
    bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
    os.environ["STAGE1_WIRE_CROSS_FILL"] = "0"
    ex = pskel.WireExtractor(ink_mode="adaptive", thin_w=2.6, use_colour=True)
    thin, ink, _h = ex.foreground(bgr)
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    filled = pskel.cross_core_fill(thin, dt, 2.6, r=a.r, min_axes=a.axes)
    add = int(((filled > 0) & (thin == 0)).sum())
    print("case %s：thin %d px，补上 %d px（%.3f%%），r=%.1f axes=%d"
          % (a.case, int((thin > 0).sum()), add, 100.0 * add / max(1, int((thin > 0).sum())),
             a.r, a.axes))
    sk0 = pskel.zhang_suen((thin * 255).astype(np.uint8)) > 0
    sk1 = pskel.zhang_suen((filled * 255).astype(np.uint8)) > 0
    col = a.x
    seg = slice(a.y0, a.y1 + 1)
    print("列 x=%d 上 y∈[%d,%d]：骨架像素 补前 %d 个、补后 %d 个"
          % (col, a.y0, a.y1, int(sk0[seg, col].sum()), int(sk1[seg, col].sum())))
    print("      " + "".join("%4d" % x for x in range(col - 10, col + 11, 5)))
    for y in range(a.y0, a.y1 + 1):
        r0 = "".join("S" if sk0[y, x] else ("*" if thin[y, x] else ("." if ink[y, x] == 0 else " "))
                     for x in range(col - 10, col + 11))
        r1 = "".join("S" if sk1[y, x] else ("*" if filled[y, x] else ("." if ink[y, x] == 0 else " "))
                     for x in range(col - 10, col + 11))
        print("y=%3d 补前 %s   补后 %s" % (y, r0, r1))


main()
