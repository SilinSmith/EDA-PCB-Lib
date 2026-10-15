"""打印某个窗口的 dt（到墨迹边界的距离）与细笔画掩膜，用来看「交叉芯」有多大。

用法：`python -X utf8 work\\_dtmap.py --case 1 --x0 833 --x1 853 --y0 164 --y1 188`
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from pipeline import pskel  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402

ROOT = r"Y:\workspace\EDA\work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--x0", type=int, default=833)
    ap.add_argument("--x1", type=int, default=853)
    ap.add_argument("--y0", type=int, default=164)
    ap.add_argument("--y1", type=int, default=188)
    ap.add_argument("--dir", default=os.path.join(ROOT, "_fast_in"))
    a = ap.parse_args()
    c = None
    for cc in discover(DEFAULT_DATA_ROOT):
        if str(getattr(cc, "case_id", "") or "") == a.case:
            c = cc
            break
    if c is None:
        print("找不到 case %s" % a.case)
        return
    bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    print("%s  %dx%d" % (os.path.basename(str(c.png_path)), W, H))
    ex = pskel.WireExtractor(ink_mode="adaptive", thin_w=2.6, use_colour=True)
    thin, ink, _h = ex.foreground(bgr)
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    th = thin > 0
    print("列 = x %d..%d；每格 2 字符 = round(dt)，'**' = 细笔画，'..' = 无墨" % (a.x0, a.x1))
    print("      " + "".join("%4d" % x for x in range(a.x0, a.x1 + 1, 5)))
    for y in range(a.y0, a.y1 + 1):
        row = []
        for x in range(a.x0, a.x1 + 1):
            if ink[y, x] == 0:
                row.append("..")
            elif th[y, x]:
                row.append("**")
            else:
                row.append("%2d" % min(99, int(round(float(dt[y, x])))))
        print("y=%3d %s" % (y, "".join(row)))


main()
