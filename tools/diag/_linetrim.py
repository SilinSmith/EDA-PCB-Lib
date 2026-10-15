# -*- coding: utf-8 -*-
"""§5.113 probe: trim wire ends that overshoot a component border back onto the border.

Evidence (`_endtrue.py`): of the 160 GT edges that fail only at one end, our off-end is
within 2px of one of OUR pin points in 98 cases, and >=9px away from any component border
in 149 cases; GT's off-end sits exactly ON a component border in 48 cases (0px) / 1px in 21.
So our wire runs ~12px past the symbol body to our pin point while GT stops at the body.

Mutation: if an edge endpoint is within --pin_tol of our own pin point, belongs to a
component whose bbox border is --min..--max px away, move the endpoint onto that border
(foot of the perpendicular). Then rescore with `evaluate_case`.

Usage (cwd = Y:\\workspace\\EDA\\eda-pcb):
    python -X utf8 ..\\work\\_linetrim.py --run work/_fast_out_gsd9 --min 2 --max 16
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
ap.add_argument("--pin_tol", type=float, default=2.5)
ap.add_argument("--min", type=float, default=2.0)
ap.add_argument("--max", type=float, default=16.0)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()


def comp_bboxes(doc):
    out = {}
    for cid, comp in (doc.get('components') or {}).items():
        if not isinstance(comp, dict):
            continue
        for key in ('bbox', 'bbox_img', 'box', 'rect'):
            v = comp.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                try:
                    x0, y0, x1, y1 = float(v[0]), float(v[1]), float(v[2]), float(v[3])
                except Exception:
                    continue
                out[str(cid)] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                break
    return out


def pin_list(doc):
    out = []
    for comp, group in (doc.get('pins') or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, v in group.items():
            p = M._point((v or {}).get('point'))
            if p:
                out.append((str(comp), p))
    return out


def border_project(p, bb):
    x0, y0, x1, y1 = bb
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    d = math.hypot(dx, dy)
    return d, (min(max(p[0], x0), x1), min(max(p[1], y0), y1))


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
sb = sn = 0.0
for cid in sel:
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    base = M.evaluate_case(gt, pred, cid, cfg)
    bbs = comp_bboxes(pred)
    pins = pin_list(pred)
    mut = copy.deepcopy(pred)
    n_trim = 0
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
            for end in (a, b):
                target = None
                for comp, p in pins:
                    if math.hypot(p[0] - end[0], p[1] - end[1]) <= args.pin_tol:
                        bb = bbs.get(comp)
                        if bb is None:
                            continue
                        d, proj = border_project(end, bb)
                        if args.min <= d <= args.max:
                            target = proj
                            break
                if target is not None:
                    newp.append(target); n_trim += 1
                else:
                    newp.append(end)
            ev[0] = {'x': round(newp[0][0], 2), 'y': round(newp[0][1], 2)}
            ev[1] = {'x': round(newp[1][0], 2), 'y': round(newp[1][1], 2)}
    new = M.evaluate_case(gt, mut, cid, cfg)
    sb += base.net_line.f1; sn += new.net_line.f1
    print("case %-3s line %.4f -> %.4f (%+.4f) | total %.2f -> %.2f (%+.2f) | trimmed %d ends"
          % (cid, base.net_line.f1, new.net_line.f1, new.net_line.f1 - base.net_line.f1,
             base.score, new.score, new.score - base.score, n_trim))
n = max(1, len(sel))
print("\n== trim-to-border [%g..%g]px : net_line mean %.4f -> %.4f (%+.4f = %+.4f pts) | total mean %.4f -> %.4f"
      % (args.min, args.max, sb / n, sn / n, (sn - sb) / n, (sn - sb) / n * 0.10,
         sum(M.evaluate_case(normalise_doc(json.load(open(str(cases[c].json_path), encoding="utf-8"))),
                             json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % c)),
                                            encoding="utf-8")), c, cfg).score
             for c in sel if c in cases) / n, 0.0))
