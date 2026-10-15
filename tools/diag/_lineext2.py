# -*- coding: utf-8 -*-
"""§5.113 probe: wire endpoints are SHORT — extend them to the nearest pin point and rescore.

`work/_linegap.py` on gsd9: net_line mean 0.5800 with gt.hit 384 / gt.near **226**
(geometry right there but max endpoint distance > line_tol=5) / gt.no_ours only 22.
Typical gap 8..16px and often on ONE end only (case 5 GT net_7 (443,363)->(443,447)
vs ours (442,433)->(442,359): bottom end 14px short). Hypothesis: GT draws the wire
all the way to the pin connection point, while our wire stops where the green ink
stops (the dark-red symbol stub is not counted as wire).

Offline mutation (pipeline untouched): for each edge endpoint, look for a pin point
within R whose direction is outward and parallel to the edge (cos >= --cos) and move
the endpoint there; then rescore with `evaluate_case`.
--src pred = our own pin points  |  --src gt = GT pin points (oracle upper bound).

Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_lineext2.py --run work/_fast_out_gsd9 --src pred --R 25
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--src", default="pred", choices=["pred", "gt"])
ap.add_argument("--R", type=float, default=25.0)
ap.add_argument("--cos", type=float, default=0.7)
ap.add_argument("--free-only", action="store_true",
                help="only move ends that are dead ends (no other wire endpoint within 6px)")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()


def pin_points(doc):
    out = []
    for comp, group in (doc.get('pins') or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, v in group.items():
            p = M._point((v or {}).get('point'))
            if p:
                out.append((comp, pk, p))
    return out


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
sum_base = sum_new = 0.0
sco_base = sco_new = 0.0
for cid in sel:
    if cid not in cases:
        print("!! missing case %s" % cid); continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    pts = pin_points(gt if args.src == 'gt' else pred)
    base = M.evaluate_case(gt, pred, cid, cfg)
    mut = copy.deepcopy(pred)
    all_ends = []
    for _, net0 in (pred.get('nets') or {}).items():
        if not isinstance(net0, dict):
            continue
        for _, ev0 in (net0.get('edges') or {}).items():
            pts0 = M._edge_points(ev0)
            if len(pts0) >= 2:
                all_ends.append(pts0[0]); all_ends.append(pts0[1])

    def is_free(pt):
        for q in all_ends:
            if abs(q[0] - pt[0]) < 1e-6 and abs(q[1] - pt[1]) < 1e-6:
                continue
            if math.hypot(q[0] - pt[0], q[1] - pt[1]) <= 6.0:
                return False
        return True

    n_move = 0
    n_skip = 0
    for _, net in (mut.get('nets') or {}).items():
        if not isinstance(net, dict):
            continue
        for _, ev in (net.get('edges') or {}).items():
            if not isinstance(ev, list) or len(ev) < 2:
                continue
            a, b = M._point(ev[0]), M._point(ev[1])
            if not a or not b:
                continue
            newp = []
            for end, other in ((a, b), (b, a)):
                if args.free_only and not is_free(end):
                    newp.append(end); n_skip += 1; continue
                dx, dy = end[0] - other[0], end[1] - other[1]
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    newp.append(end); continue
                ux, uy = dx / L, dy / L
                best = None
                for _, _, p in pts:
                    ex, ey = p[0] - end[0], p[1] - end[1]
                    d = math.hypot(ex, ey)
                    if d > args.R or d < 0.5:
                        continue
                    c = (ex * ux + ey * uy) / d
                    if c < args.cos:
                        continue
                    if best is None or d < best[0]:
                        best = (d, p)
                if best is not None:
                    newp.append(best[1]); n_move += 1
                else:
                    newp.append(end)
            ev[0] = {'x': round(newp[0][0], 2), 'y': round(newp[0][1], 2)}
            ev[1] = {'x': round(newp[1][0], 2), 'y': round(newp[1][1], 2)}
    new = M.evaluate_case(gt, mut, cid, cfg)
    sum_base += base.net_line.f1; sum_new += new.net_line.f1
    sco_base += base.score; sco_new += new.score
    print("case %-3s line %.4f -> %.4f (%+.4f) | total %.2f -> %.2f (%+.2f) | moved %d ends"
          % (cid, base.net_line.f1, new.net_line.f1, new.net_line.f1 - base.net_line.f1,
             base.score, new.score, new.score - base.score, n_move))
n = max(1, len(sel))
print("\n== src=%s R=%.0f cos=%.2f : net_line mean %.4f -> %.4f (%+.4f = %+.4f pts)"
      " | total mean %.4f -> %.4f (%+.4f)"
      % (args.src, args.R, args.cos, sum_base / n, sum_new / n,
         (sum_new - sum_base) / n, (sum_new - sum_base) / n * 0.10,
         sco_base / n, sco_new / n, (sco_new - sco_base) / n))
