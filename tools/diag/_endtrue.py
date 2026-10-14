# -*- coding: utf-8 -*-
"""§5.113 probe: for GT edges that fail only at ONE end, what is GT's end attached to?

Classifies each such GT end by the nearest explanation:
    PIN   = a GT pin point (<= tol)
    JUNC  = an endpoint of another GT edge (<= tol)
    BOTH  = both nearby
    NONE  = neither
Also reports the same for OUR off-end (nearest PRED pin point / our own other edges).
Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_endtrue.py --run work/_fast_out_gsd9 --tol 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--tol", type=float, default=3.0)
ap.add_argument("--dmax", type=float, default=30.0)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c


def pins_of(doc):
    out = []
    for comp, group in (doc.get('pins') or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, v in group.items():
            p = M._point((v or {}).get('point'))
            if p:
                out.append((str(comp), str(pk), p))
    return out


def edges_of(doc):
    out = []
    for nid, net in (doc.get('nets') or {}).items():
        for eid, ev in ((net or {}).get('edges') or {}).items():
            p = M._edge_points(ev)
            if len(p) >= 2:
                out.append((str(nid), str(eid), p[0], p[1]))
    return out


def bboxes_of(doc):
    out = []
    for cid, comp in (doc.get('components') or {}).items():
        if not isinstance(comp, dict):
            continue
        for key in ('bbox', 'bbox_img', 'box', 'rect'):
            v = comp.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                try:
                    x0, y0, x1, y1 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
                except Exception:
                    continue
                out.append((str(cid), min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                break
    return out


def border_dist(p, bbs):
    best = None
    for cid, x0, y0, x1, y1 in bbs:
        dx = max(x0 - p[0], 0.0, p[0] - x1)
        dy = max(y0 - p[1], 0.0, p[1] - y1)
        d = math.hypot(dx, dy)
        if best is None or d < best[0]:
            best = (d, cid)
    return best


def mind(p, pts, skip=None):
    best = None
    for it in pts:
        q = it[2] if len(it) == 3 else it
        if skip is not None and q is skip:
            continue
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if best is None or d < best[0]:
            best = (d, it)
    return best


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
tot = Counter()
near_pin = []
for cid in sel:
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    gtp, prp = pins_of(gt), pins_of(pred)
    gtb, prb = bboxes_of(gt), bboxes_of(pred)
    ge, pe = edges_of(gt), edges_of(pred)
    gends = [(e[0], e[1], e[2]) for e in ge] + [(e[0], e[1], e[3]) for e in ge]
    for nid, eid, a, b in ge:
        best = None
        for pnid, peid, c, d in pe:
            dd = M._seg_distance([a, b], [c, d], True)
            if best is None or dd < best[0]:
                best = (dd, pnid, peid, c, d)
        if best is None:
            continue
        dd, pnid, peid, c, d = best
        if dd <= 5.0 or dd > args.dmax:
            continue
        straight = math.hypot(c[0] - a[0], c[1] - a[1]) + math.hypot(d[0] - b[0], d[1] - b[1])
        rev = math.hypot(d[0] - a[0], d[1] - a[1]) + math.hypot(c[0] - b[0], c[1] - b[1])
        if rev < straight:
            c, d = d, c
        e0 = math.hypot(c[0] - a[0], c[1] - a[1])
        e1 = math.hypot(d[0] - b[0], d[1] - b[1])
        if e0 <= 3.0 and e1 > 3.0:
            gend, oend = b, d
        elif e1 <= 3.0 and e0 > 3.0:
            gend, oend = a, c
        else:
            tot['not_single'] += 1
            continue
        dp = mind(gend, gtp)
        dj = mind(gend, [e for e in gends if math.hypot(e[2][0] - gend[0], e[2][1] - gend[1]) > 1e-6])
        has_pin = dp is not None and dp[0] <= args.tol
        has_j = dj is not None and dj[0] <= args.tol
        bgt = border_dist(gend, gtb)
        bpr = border_dist(oend, prb)
        has_b = bgt is not None and bgt[0] <= args.tol
        lab = 'PIN+JUNC' if (has_pin and has_j) else ('PIN' if has_pin else
                                                      ('JUNC' if has_j else
                                                       ('BORDER' if has_b else 'NONE')))
        tot[lab] += 1
        tot['single'] += 1
        if bgt is not None:
            tot['bd_gt_%d' % min(9, int(bgt[0]))] += 1
        if bpr is not None:
            tot['bd_pr_%d' % min(9, int(bpr[0]))] += 1
        if dp is not None:
            near_pin.append(round(dp[0], 1))
            tot['pin_d_%d' % min(9, int(dp[0]))] += 1
        op = mind(oend, prp)
        tot['our_end_pin_d_%d' % min(9, int(op[0])) if op else 'our_end_pin_none'] += 1
print("== single-end-off GT edges: %s" % dict(tot))
near_pin.sort()
if near_pin:
    n = len(near_pin)
    print("== distance from GT off-end to nearest GT pin point: median %.1f  p25 %.1f  p75 %.1f  <=3px %d/%d"
          % (near_pin[n // 2], near_pin[n // 4], near_pin[3 * n // 4],
             sum(1 for v in near_pin if v <= 3.0), n))
