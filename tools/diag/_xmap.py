# -*- coding: utf-8 -*-
"""打印窗口内「细笔画 / 骨架 / 一般墨迹」的逐行 x 区间，判断交叉处缺的是墨还是骨架。

用法: python -X utf8 work\\_xmap.py --case 1 --x0 830 --x1 858 --y0 166 --y1 212
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

from pipeline import pskel  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def _ranges(row):
    xs = np.nonzero(row)[0]
    if xs.size == 0:
        return "-"
    out = []
    s = p = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x == p + 1:
            p = x
            continue
        out.append("%d-%d" % (s, p) if p > s else "%d" % s)
        s = p = x
    out.append("%d-%d" % (s, p) if p > s else "%d" % s)
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--x0", type=int, default=830)
    ap.add_argument("--x1", type=int, default=858)
    ap.add_argument("--y0", type=int, default=166)
    ap.add_argument("--y1", type=int, default=212)
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    c = None
    for cc in discover(DEFAULT_DATA_ROOT):
        if str(getattr(cc, "case_id", "") or "") == a.case:
            c = cc
            break
    if c is None:
        print("找不到 case", a.case)
        return 1
    bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
    H = bgr.shape[0]
    ex = pskel.WireExtractor(ink_mode="adaptive", thin_w=2.6, use_colour=True)
    thin, ink, _h = ex.foreground(bgr)
    skel = pskel.zhang_suen((thin * 255).astype(np.uint8)) > 0

    gt = json.loads(Path(str(c.json_path)).read_text(encoding="utf-8"))
    spans = []
    for nm, net in (gt.get("nets") or {}).items():
        for ek, e in (net.get("edges") or {}).items():
            pts = [(float(t["x"]), H - float(t["y"])) for t in e]
            if len(pts) < 2:
                continue
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            if abs(x0 - x1) > abs(y0 - y1):
                continue
            if a.x0 <= min(x0, x1) <= a.x1:
                spans.append((min(y0, y1), max(y0, y1), nm, ek, round((x0 + x1) / 2.0, 1)))
    print("== GT 竖边落在 x∈[%d,%d] ==" % (a.x0, a.x1))
    for y0, y1, nm, ek, xc in sorted(spans):
        print("   y[%7.1f,%7.1f] 长 %5.1f  x≈%.1f  %s.%s" % (y0, y1, y1 - y0, xc, nm, ek))
    print("== 逐行（y_image；x 限定 [%d,%d]）==" % (a.x0, a.x1))
    for y in range(a.y0, a.y1 + 1, a.step):
        t = _ranges(thin[y, a.x0:a.x1 + 1])
        s = _ranges(skel[y, a.x0:a.x1 + 1])
        i = _ranges(ink[y, a.x0:a.x1 + 1])
        mark = "  <== GT 切点" if any(abs(y - yy) <= 0.5 for sp in spans for yy in sp[:2]) else ""
        print("   y=%4d ink %-28s thin %-28s skel %-20s%s" % (y, i, t, s, mark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
