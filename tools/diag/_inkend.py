# -*- coding: utf-8 -*-
"""§5.113 probe: how far does the wire INK extend beyond a given wire endpoint?

For each endpoint we walk outward along the wire direction and measure the last position
where non-background ink is present.  t_outer > 0  => the ink continues past the endpoint;
t_outer < 0  => the endpoint sits beyond the visible ink.

Groups: GT ends of the single-end-off failing edges, all GT ends, our ends.
Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_inkend.py --run work/_fast_out_gsd9 --scan 22
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--scan", type=float, default=22.0)
ap.add_argument("--dmax", type=float, default=30.0)
ap.add_argument("--half", type=float, default=2.0, help="perpendicular half-width counted as ink")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c


def img_of(c):
    for attr in ("image_path", "img_path", "image", "png_path"):
        v = getattr(c, attr, None)
        if v:
            return cv2.imread(str(v))
    for cand in sorted(Path(c.json_path).parent.glob("*.png")):
        return cv2.imread(str(cand))
    return None


def edges_of(doc):
    out = []
    for nid, net in (doc.get('nets') or {}).items():
        for eid, ev in ((net or {}).get('edges') or {}).items():
            p = M._edge_points(ev)
            if len(p) >= 2:
                out.append((str(nid), str(eid), p[0], p[1]))
    return out


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
stats = {'gt_off': [], 'gt_all': [], 'pr_all': [], 'pr_off': []}
for cid in sel:
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    img = img_of(cases[cid])
    if img is None:
        continue
    H = img.shape[0]
    ink = ~((img[:, :, 0] > 180) & (img[:, :, 1] > 180) & (img[:, :, 2] > 180))
    ge, pe = edges_of(gt), edges_of(pred)

    def t_outer(p, u):
        # walk outward (u points away from the wire body); find the farthest ink sample
        last = None
        t = 0.0
        while t <= args.scan:
            x = p[0] + u[0] * t
            y = p[1] + u[1] * t
            iy = int(round(H - y))
            hit = False
            for o in (-args.half, 0.0, args.half):
                ix = int(round(x - u[1] * o))
                iyy = int(round(iy + u[0] * o))
                if 0 <= iyy < ink.shape[0] and 0 <= ix < ink.shape[1] and ink[iyy, ix]:
                    hit = True
            if hit:
                last = t
            elif last is not None and t - last > 4.0:
                break
            t += 1.0
        return -1.0 if last is None else last

    def out_dir(p, other):
        dx, dy = p[0] - other[0], p[1] - other[1]
        L = math.hypot(dx, dy) or 1.0
        return (dx / L, dy / L)

    for nid, eid, a, b in ge:
        stats['gt_all'].append(t_outer(a, out_dir(a, b)))
        stats['gt_all'].append(t_outer(b, out_dir(b, a)))
    for nid, eid, a, b in pe:
        stats['pr_all'].append(t_outer(a, out_dir(a, b)))
        stats['pr_all'].append(t_outer(b, out_dir(b, a)))
    for nid, eid, a, b in ge:
        best = None
        for pnid, peid, c, d in pe:
            dd = M._seg_distance([a, b], [c, d], True)
            if best is None or dd < best[0]:
                best = (dd, c, d)
        if best is None:
            continue
        dd, c, d = best
        if dd <= 5.0 or dd > args.dmax:
            continue
        straight = math.hypot(c[0] - a[0], c[1] - a[1]) + math.hypot(d[0] - b[0], d[1] - b[1])
        rev = math.hypot(d[0] - a[0], d[1] - a[1]) + math.hypot(c[0] - b[0], c[1] - b[1])
        if rev < straight:
            c, d = d, c
        e0 = math.hypot(c[0] - a[0], c[1] - a[1])
        e1 = math.hypot(d[0] - b[0], d[1] - b[1])
        if e0 <= 3.0 and e1 > 3.0:
            stats['gt_off'].append(t_outer(b, out_dir(b, a)))
            stats['pr_off'].append(t_outer(d, out_dir(d, c)))
        elif e1 <= 3.0 and e0 > 3.0:
            stats['gt_off'].append(t_outer(a, out_dir(a, b)))
            stats['pr_off'].append(t_outer(c, out_dir(c, d)))


def rep(name, vals):
    if not vals:
        print("%s: empty" % name); return
    v = sorted(vals)
    n = len(v)
    print("%-8s n=%4d  min %6.1f  p25 %6.1f  median %6.1f  p75 %6.1f  max %6.1f | <=0 (no ink beyond) %d (%.0f%%)"
          % (name, n, v[0], v[n // 4], v[n // 2], v[3 * n // 4], v[-1],
             sum(1 for q in v if q < 0), 100.0 * sum(1 for q in v if q < 0) / n))


print("t_outer = farthest ink sample beyond the endpoint along the wire direction")
for k in ('gt_off', 'pr_off', 'gt_all', 'pr_all'):
    rep(k, stats[k])
