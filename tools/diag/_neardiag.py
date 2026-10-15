# -*- coding: utf-8 -*-
"""§5.113 probe: classify why a GT wire edge fails the 5px test (the `gt.near` class).

For every GT edge it finds the closest predicted edge (`_seg_distance`, best reversal) and
classifies the pair: one end aligned + the other short/long, translated, perpendicular, or
a fragment (our edge covers only part of GT's). Prints counts + a few samples.

Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_neardiag.py --run work/_fast_out_gsd9
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
ap.add_argument("--tol", type=float, default=5.0)
ap.add_argument("--samples", type=int, default=6)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()

sel = [s.strip() for s in args.cases.split(",") if s.strip()]
tot = Counter()
deltas = Counter()
shown = 0
for cid in sel:
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    gted = []
    for nid, net in (gt.get('nets') or {}).items():
        for eid, ev in ((net or {}).get('edges') or {}).items():
            p = M._edge_points(ev)
            if len(p) >= 2:
                gted.append((nid, eid, p[0], p[1]))
    preded = []
    for nid, net in (pred.get('nets') or {}).items():
        for eid, ev in ((net or {}).get('edges') or {}).items():
            p = M._edge_points(ev)
            if len(p) >= 2:
                preded.append((nid, eid, p[0], p[1]))
    for nid, eid, a, b in gted:
        best = None
        for pnid, peid, c, d in preded:
            dd = M._seg_distance([a, b], [c, d], True)
            if best is None or dd < best[0]:
                best = (dd, pnid, peid, c, d)
        if best is None or best[0] <= args.tol:
            tot['hit'] += 1
            continue
        dd, pnid, peid, c, d = best
        # orientation
        straight = math.hypot(c[0] - a[0], c[1] - a[1]) + math.hypot(d[0] - b[0], d[1] - b[1])
        rev = math.hypot(d[0] - a[0], d[1] - a[1]) + math.hypot(c[0] - b[0], c[1] - b[1])
        if rev < straight:
            c, d = d, c
        e0 = math.hypot(c[0] - a[0], c[1] - a[1])
        e1 = math.hypot(d[0] - b[0], d[1] - b[1])
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        lp = math.hypot(d[0] - c[0], d[1] - c[1]) or 1.0
        # is our edge mostly parallel?
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        vx, vy = (d[0] - c[0]) / lp, (d[1] - c[1]) / lp
        cros = abs(ux * vy - uy * vx)
        if cros > 0.20:
            kind = 'perp'
        elif max(e0, e1) <= 3.0:
            kind = 'translated'
        elif lp < 0.7 * L:
            kind = 'fragment'
        elif e0 <= 3.0:
            kind = 'end1_off'
        elif e1 <= 3.0:
            kind = 'end0_off'
        else:
            kind = 'both_off'
        tot[kind] += 1
        tot['near'] += 1
        if kind in ('end0_off', 'end1_off', 'fragment'):
            deltas[int(round(max(e0, e1) / 2.0) * 2)] += 1
        if shown < args.samples and kind in ('end0_off', 'end1_off', 'fragment', 'both_off'):
            shown += 1
            print("case %-3s %s %s GT (%.0f,%.0f)->(%.0f,%.0f) L=%.0f | ours (%.0f,%.0f)->(%.0f,%.0f) L=%.0f"
                  " | %s d=%.1f e0=%.1f e1=%.1f cros=%.2f"
                  % (cid, nid, eid, a[0], a[1], b[0], b[1], L, c[0], c[1], d[0], d[1], lp,
                     kind, dd, e0, e1, cros))
print("\n== classification over %d GT edges: %s" % (sum(tot.values()), dict(tot)))
print("== magnitude (px, rounded to 2) of the off end for end_off/fragment: %s"
      % dict(sorted(deltas.items())))
