# -*- coding: utf-8 -*-
"""Pin recovery, take 2 -- measured against ground truth on a fixed sample.

The first attempt (``ppins.candidates_from_skeleton`` with the stroke-mask skeleton) scored
F1 = 0.087: far too many candidates, because the stroke mask is full of text and symbol
fragments and every one of their skeleton endpoints became a pin.

The measured structure that a pin detector must exploit:

* pins are **attached to wires** (97.9% of ground-truth pins appear in some net's hyperGraph),
  and on the sample cases each pin is at most ~10 px from its component bbox border
  (median 0, 63% within 5 px, 84% within 10 px);
* a pin is where a wire *terminates*, i.e. a wire endpoint -- not where a wire crosses;
* a component's pins are spread over several edges and never coincide.

So the detector is: for each component box, take the wire endpoints within ``reach``, cluster
them per edge, keep at most one candidate per cluster (the one furthest out), and optionally
snap onto a regular pitch along the edge.

Every variant is scored with the same one-to-one matcher and the same 5 px tolerance that
``metrics.pin_f1`` uses, so the numbers here are directly comparable to the competition metric.

Usage:  python -m pipeline.validate_pins2 [--n=12] [case_id ...]
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore, ppins, pskel

# ------------------------------------------------------------------ data


def gt_pins(case) -> Dict[str, List[Tuple[float, float, Optional[str]]]]:
    doc = json.load(open(case.gt_json, encoding='utf-8'))
    im = cv2.imread(case.png, cv2.IMREAD_COLOR)
    H = float(im.shape[0])
    out: Dict[str, List] = {}
    for key, grp in (doc.get('pins') or {}).items():
        for pk, info in (grp or {}).items():
            pt = info.get('point') or {}
            if pt.get('x') is None:
                continue
            out.setdefault(key, []).append(
                (float(pt['x']), H - float(pt['y']), info.get('pinname')))
    return out


def gt_boxes(case) -> Dict[str, Tuple[float, float, float, float]]:
    doc = json.load(open(case.gt_json, encoding='utf-8'))
    im = cv2.imread(case.png, cv2.IMREAD_COLOR)
    H = float(im.shape[0])
    return {k: pcore.bbox_json_to_img(v.get('bbox') or [0, 0, 0, 0], H)
            for k, v in (doc.get('components') or {}).items() if v.get('bbox')}


def gt_wire_pts(case) -> List[Tuple[float, float]]:
    """Every vertex of every ground-truth net edge -- the *perfect* wire input."""
    doc = json.load(open(case.gt_json, encoding='utf-8'))
    im = cv2.imread(case.png, cv2.IMREAD_COLOR)
    H = float(im.shape[0])
    pts = []
    for net in (doc.get('nets') or {}).values():
        for e in (net.get('edges') or {}).values():
            for q in e:
                pts.append((float(q['x']), H - float(q['y'])))
    return pts


# ------------------------------------------------------------------ matching

def score(pred: Dict[str, List[Tuple[float, float]]],
          gt: Dict[str, List[Tuple[float, float, Optional[str]]]],
          tol: float = 5.0):
    tp = fp = fn = 0
    errs = []
    for key, gl in gt.items():
        pl = list(pred.get(key, []))
        used = [False] * len(gl)
        for (px, py) in pl:
            best, bd = -1, tol * tol
            for gi, (gx, gy, _n) in enumerate(gl):
                if used[gi]:
                    continue
                d = (gx - px) ** 2 + (gy - py) ** 2
                if d <= bd:
                    bd, best = d, gi
            if best >= 0:
                used[best] = True
                tp += 1
                errs.append(math.sqrt(bd))
            else:
                fp += 1
        fn += used.count(False)
    for key in pred:
        if key not in gt:
            fp += len(pred[key])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, prec=p, rec=r,
                f1=2 * p * r / (p + r) if p + r else 0.0,
                med_err=float(np.median(errs)) if errs else float('nan'))


# ------------------------------------------------------------------ variants

def cands_from_pts(pts: Sequence[Tuple[float, float]],
                   boxes: Dict[str, Tuple[float, float, float, float]],
                   reach: float = 10.0, merge: float = 5.0,
                   cluster: bool = True) -> Dict[str, List[Tuple[float, float]]]:
    """Assign points to the nearest box within *reach*, cluster per edge, keep the far one."""
    out: Dict[str, List[ppins.PinCand]] = {}
    for key, box in boxes.items():
        ex = pcore.expand_box(box, reach)
        for (x, y) in pts:
            if not pcore.point_in_box((x, y), ex):
                continue
            e, d = ppins.nearest_edge((x, y), box)
            out.setdefault(key, []).append(ppins.PinCand(key, x, y, e, dist=d))
    res: Dict[str, List[Tuple[float, float]]] = {}
    for key, cs in out.items():
        if not cluster:
            res[key] = [(c.x, c.y) for c in cs]
            continue
        picked: List[ppins.PinCand] = []
        by_edge: Dict[str, List[ppins.PinCand]] = {}
        for c in cs:
            by_edge.setdefault(c.edge, []).append(c)
        for edge, es in by_edge.items():
            horiz = edge in ('T', 'B')
            es.sort(key=lambda c: (c.x if horiz else c.y))
            cur = [es[0]]
            for c in es[1:]:
                gap = abs((c.x - cur[-1].x) if horiz else (c.y - cur[-1].y))
                if gap <= merge:
                    cur.append(c)
                else:
                    picked.append(max(cur, key=lambda c: c.dist))
                    cur = [c]
            picked.append(max(cur, key=lambda c: c.dist))
        res[key] = [(c.x, c.y) for c in picked]
    return res


def main(argv):
    n = 12
    for a in argv:
        if a.startswith('--n='):
            n = int(a.split('=')[1])
    ids = [a for a in argv if not a.startswith('-')]
    if ids:
        cases = [c for c in (pcore.find_case(i) for i in ids) if c is not None and c.gt_json]
    else:
        cases = [c for c in pcore.discover() if c.split == '200_train_cases' and c.gt_json][:n]
    print('cases: %s' % ', '.join(c.case_id for c in cases))

    ex = pskel.WireExtractor(use_colour=False, close_ink=True)
    variants: Dict[str, List[dict]] = {}

    def add(name, s):
        variants.setdefault(name, []).append(s)

    skel_cache = {}
    for c in cases:
        boxes = gt_boxes(c)
        gt = gt_pins(c)
        # --- ceiling: perfect wires
        add('CEILING gt wire vertices', score(cands_from_pts(gt_wire_pts(c), boxes), gt))
        # --- my skeleton
        if c.case_id not in skel_cache:
            bgr = cv2.imread(c.png, cv2.IMREAD_COLOR)
            skel_cache[c.case_id] = ex.extract(bgr)
        res = skel_cache[c.case_id]
        my_pts = [(x, y) for x, y in res.endpoints]
        for reach in (6.0, 10.0, 16.0):
            add('skel endpoints reach=%g' % reach,
                score(cands_from_pts(my_pts, boxes, reach=reach), gt))
        for reach in (6.0, 10.0, 16.0):
            add('skel endpoints no-cluster reach=%g' % reach,
                score(cands_from_pts(my_pts, boxes, reach=reach, cluster=False), gt))
        # wire-polyline endpoints as extra sources of "wire terminations"
        poly_pts = []
        for p in res.polylines:
            poly_pts.append(p[0]); poly_pts.append(p[-1])
        add('polyline endpoints reach=10', score(cands_from_pts(poly_pts, boxes, reach=10.0), gt))
        seg_pts = []
        for a, b in res.segments:
            seg_pts.append(a); seg_pts.append(b)
        add('segment vertices reach=10', score(cands_from_pts(seg_pts, boxes, reach=10.0), gt))

    print('\n%-36s %6s %6s %6s %6s %6s %6s' % ('variant', 'tp', 'fp', 'fn', 'P', 'R', 'F1'))
    for name, rows in variants.items():
        tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows)
        fn = sum(r['fn'] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print('%-36s %6d %6d %6d %6.3f %6.3f %6.3f' % (name, tp, fp, fn, p, r, f1))
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    with open(os.path.join(C.OUT_ROOT, 'validate_pins2.json'), 'w', encoding='utf-8') as f:
        json.dump(variants, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main([a for a in sys.argv[1:]])
