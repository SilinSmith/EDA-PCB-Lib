# -*- coding: utf-8 -*-
"""§5.113 probe: what is at GT's wire end in the IMAGE? (green ink / maroon stub / blank)

Prints a run-length category string along one row or column:
    G = green wire ink, S = maroon (symbol stub / border), K = black (text), W = blank
Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_inkat.py --case 5 --x 443 --y0 350 --y1 480 --axis y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="5")
ap.add_argument("--x", type=int, default=0)
ap.add_argument("--y", type=int, default=0)
ap.add_argument("--x0", type=int, default=None)
ap.add_argument("--x1", type=int, default=None)
ap.add_argument("--y0", type=int, default=None)
ap.add_argument("--y1", type=int, default=None)
ap.add_argument("--axis", default="y", choices=["x", "y"])
ap.add_argument("--noflip", action="store_true", help="pass image-space y instead of document y")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
c = cases[args.case]
p = None
for attr in ("image_path", "img_path", "image", "png_path"):
    v = getattr(c, attr, None)
    if v:
        p = Path(str(v)); break
if p is None:
    for cand in sorted(Path(c.json_path).parent.glob("*.png")) + sorted(Path(c.json_path).parent.glob("*.jpg")):
        p = cand; break
print("image: %s" % p)
img = cv2.imread(str(p))
if img is None:
    sys.exit("cannot read image")
H = img.shape[0]
if not args.noflip:
    # document coords are y-up; image rows are y-down (same convention as the emitted JSON)
    def flat(v):
        return None if v is None else H - v
    args.y = flat(args.y) if args.y else 0
    args.y0, args.y1 = flat(args.y0), flat(args.y1)
    if args.y0 is not None and args.y1 is not None and args.y0 > args.y1:
        args.y0, args.y1 = args.y1, args.y0
    args.axis = 'y'

b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
green = (g > 60) & (g - b > 25) & (g - r > 25)
maroon = (r > 90) & (g < 130) & (b < 130) & (r - np.maximum(g, b) > 25)
black = (r < 90) & (g < 90) & (b < 90)


def cat(y, x):
    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
        if green[y, x]:
            return 'G'
        if maroon[y, x]:
            return 'S'
        if black[y, x]:
            return 'K'
    return 'W'


def mean_bgr(y0, y1, x, axis):
    if axis == 'y':
        seg = img[y0:y1 + 1, x]
    else:
        seg = img[x, y0:y1 + 1]
    if seg.size == 0:
        return (0, 0, 0)
    m = seg.reshape(-1, 3).mean(axis=0)
    return (int(round(m[0])), int(round(m[1])), int(round(m[2])))


if args.axis == 'y':
    y0 = args.y0 if args.y0 is not None else args.y - 40
    y1 = args.y1 if args.y1 is not None else args.y + 40
    s = "".join(cat(y, args.x) for y in range(y0, y1))
    # RLE
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append("%s%d[%d..%d](%s)" % (s[i], j - i, y0 + i, y0 + j - 1,
                                         ",".join(str(v) for v in mean_bgr(y0 + i, y0 + j - 1, args.x, 'y'))))
        i = j
    print("column x=%d, y %d..%d : %s" % (args.x, y0, y1 - 1, " ".join(out)))
else:
    x0 = args.x0 if args.x0 is not None else args.x - 40
    x1 = args.x1 if args.x1 is not None else args.x + 40
    s = "".join(cat(args.y, x) for x in range(x0, x1))
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append("%s%d[%d..%d]" % (s[i], j - i, x0 + i, x0 + j - 1))
        i = j
    print("row y=%d, x %d..%d : %s" % (args.y, x0, x1 - 1, " ".join(out)))
