# -*- coding: utf-8 -*-
"""§5.113 probe: dump GT and Pred edges inside a window, with the ink category at each end.

Ink categories come from the case image (document y is flipped to image rows):
    G = green wire ink | S = maroon (symbol stub / border) | K = black text | W = blank
Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_win.py --case 6 --x 505 --y 300 --r 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="6")
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--x", type=float, required=True)
ap.add_argument("--y", type=float, required=True)
ap.add_argument("--r", type=float, default=200.0)
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
    for cand in sorted(Path(c.json_path).parent.glob("*.png")):
        p = cand; break
img = cv2.imread(str(p))
H = img.shape[0]
b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
G = (g > 90) & (g - np.maximum(r, b) > 25)
S = (b < 110) & (g < 110) & (r > 90) & (r - np.maximum(g, b) > 25)
K = (r < 110) & (g < 110) & (b < 110)


def cat(pt):
    x = int(round(pt[0])); y = int(round(H - pt[1]))
    if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
        return '?'
    if G[y, x]:
        return 'G'
    if S[y, x]:
        return 'S'
    if K[y, x]:
        return 'K'
    return 'W'


def inwin(pt):
    return abs(pt[0] - args.x) <= args.r and abs(pt[1] - args.y) <= args.r


def dump(tag, doc):
    for nid, net in sorted((doc.get('nets') or {}).items()):
        if not isinstance(net, dict):
            continue
        for eid, ev in sorted((net.get('edges') or {}).items()):
            pts = M._edge_points(ev)
            if len(pts) < 2:
                continue
            a, b2 = pts[0], pts[1]
            if inwin(a) or inwin(b2) or (abs(a[0] - b2[0]) < 1 and not (a[0] < args.x - args.r or a[0] > args.x + args.r)) or (abs(a[1] - b2[1]) < 1 and not (a[1] < args.y - args.r or a[1] > args.y + args.r)):
                print("%s %-8s %-8s (%7.1f,%7.1f)->(%7.1f,%7.1f)  L=%6.1f ends %s/%s"
                      % (tag, nid, eid, a[0], a[1], b2[0], b2[1],
                         ((a[0] - b2[0]) ** 2 + (a[1] - b2[1]) ** 2) ** 0.5, cat(a), cat(b2)))


gt = normalise_doc(json.load(open(str(cases[args.case].json_path), encoding="utf-8")))
pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % args.case)),
                      encoding="utf-8"))
print("=== window (%.0f,%.0f) r=%.0f  image H=%d" % (args.x, args.y, args.r, H))
dump("GT", gt)
dump("PR", pred)
