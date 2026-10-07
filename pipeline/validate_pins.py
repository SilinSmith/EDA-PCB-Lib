# -*- coding: utf-8 -*-
"""Score pin-tip recovery against ground truth, independent of the competition metrics.

For each case we know every pin tip exactly (competition JSON), so this reports the number
that ``ppins.recover_pins`` puts within 5 px -- the same tolerance ``metrics.pin_f1`` uses.

Usage:  python -m pipeline.validate_pins [case_id ...] [--n 20] [--sweep]
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore, ppins, pskel


def gt_pins(case) -> List[Tuple[str, str, float, float, Optional[str]]]:
    with open(case.gt_json, encoding='utf-8') as f:
        doc = json.load(f)
    H = float(doc.get('imageHeight') or 0) or None
    if H is None:
        im = cv2.imread(case.png, cv2.IMREAD_COLOR)
        H = float(im.shape[0])
    out = []
    for key, grp in (doc.get('pins') or {}).items():
        for pk, info in (grp or {}).items():
            pt = info.get('point') or {}
            y = H - float(pt.get('y', 0.0))
            out.append((key, pk, float(pt.get('x', 0.0)), y, info.get('pinname')))
    return out


def gt_boxes(case) -> Dict[str, Tuple[float, float, float, float]]:
    with open(case.gt_json, encoding='utf-8') as f:
        doc = json.load(f)
    im = cv2.imread(case.png, cv2.IMREAD_COLOR)
    H = float(im.shape[0])
    out = {}
    for key, c in (doc.get('components') or {}).items():
        out[key] = pcore.bbox_json_to_img(c.get('bbox') or [0, 0, 0, 0], H)
    return out


def match(pred_pts: List[Tuple[float, float]],
          gt: List[Tuple[float, str, float, float, Optional[str]]],
          tol: float = 5.0):
    """Greedy nearest matching, one-to-one."""
    used = set()
    hits = []
    for (px, py) in pred_pts:
        best, bd = None, 1e18
        for gi, (_key, _pk, gx, gy, _nm) in enumerate(gt):
            if gi in used:
                continue
            d = (gx - px) ** 2 + (gy - py) ** 2
            if d < bd:
                bd, best = d, gi
        if best is not None and bd <= tol * tol:
            used.add(best)
            hits.append((best, math.sqrt(bd)))
    return len(hits), len(pred_pts), len(gt)


def eval_case(case, ex: pskel.WireExtractor, reach=6.0, merge=5.0, tips=True,
              verbose=True, method='all') -> Dict:
    boxes = gt_boxes(case)
    bgr = cv2.imread(case.png, cv2.IMREAD_COLOR)
    res = ex.extract(bgr)
    if method == 'endpoint':
        cands = ppins.candidates_from_skeleton(res, boxes, reach=reach)
        cands = [c for c in cands if c.via == 'endpoint']
        cands = ppins.cluster_candidates(cands, merge=merge)
    elif method == 'crossing':
        cands = ppins.candidates_from_skeleton(res, boxes, reach=reach)
        cands = [c for c in cands if c.via == 'crossing']
        cands = ppins.cluster_candidates(cands, merge=merge)
    else:
        cands = ppins.recover_pins(res, [pcore.Comp(k, '', b) for k, b in boxes.items()],
                                   reach=reach, merge=merge, tips=tips)

    # per-component matching (the competition key is (component, pin_number))
    by_comp: Dict[str, List[Tuple[float, float]]] = {}
    for c in cands:
        by_comp.setdefault(c.comp, []).append((c.x, c.y))
    g = gt_pins(case)
    gby: Dict[str, List] = {}
    for gi, (key, pk, x, y, nm) in enumerate(g):
        gby.setdefault(key, []).append((gi, pk, x, y, nm))

    tp = fp = fn = 0
    dists = []
    for key, pts in by_comp.items():
        gl = [(gi, pk, x, y, nm) for (gi, pk, x, y, nm) in gby.get(key, [])]
        h, np_, ng = match(pts, gl)
        del gl
        tp += h
        fp += np_ - h
        fn += ng - h
    for key, lst in gby.items():
        if key not in by_comp:
            fn += len(lst)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(case=case.case_id, source=case.source, n_pred=sum(len(v) for v in by_comp.values()),
               n_gt=len(g), tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1,
               n_comps=len(boxes))
    if verbose:
        print('  %-22s (%s) gt=%4d pred=%4d tp=%4d P=%.3f R=%.3f F1=%.3f' % (
            case.case_id, case.source, len(g), out['n_pred'], tp, prec, rec, f1))
    _ = dists
    return out


def pick(n_train=20, seed=0) -> List:
    cases = [c for c in pcore.discover() if c.gt_json]
    tr = [c for c in cases if c.split == '200_train_cases']
    other = [c for c in cases if c.split != '200_train_cases']
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(tr))[:n_train]
    return [tr[i] for i in sorted(idx)] + other


def main(argv):
    sweep = '--sweep' in argv
    n = 20
    for a in argv:
        if a.startswith('--n='):
            n = int(a.split('=')[1])
    ids = [a for a in argv if not a.startswith('-')]
    if ids:
        cases = [c for c in (pcore.find_case(i) for i in ids) if c is not None and c.gt_json]
    else:
        cases = pick(n)
    ex = pskel.WireExtractor(use_colour=False, close_ink=True)
    rows = []
    print('== pin recovery: GT boxes, stroke-mask skeleton (colour gate off, close_ink on) ==')
    for c in cases:
        try:
            rows.append(eval_case(c, ex))
        except Exception as e:
            print('  %s FAILED %s' % (c.case_id, e))
    if rows:
        print('\n== mean over %d cases ==' % len(rows))
        for k in ('prec', 'rec', 'f1'):
            print('  %-6s %.4f' % (k, sum(r[k] for r in rows) / len(rows)))
        print('  summed tp=%d fp=%d fn=%d  micro F1=%.4f' % (
            sum(r['tp'] for r in rows), sum(r['fp'] for r in rows), sum(r['fn'] for r in rows),
            _f1(sum(r['tp'] for r in rows), sum(r['fp'] for r in rows), sum(r['fn'] for r in rows))))
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    with open(os.path.join(C.OUT_ROOT, 'validate_pins.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    if not sweep:
        return
    print('\n== method ablation ==')
    for title, kw in (('all (endpoint+crossing, tip walk)', dict(method='all', tips=True)),
                      ('all, no tip walk', dict(method='all', tips=False)),
                      ('endpoints only', dict(method='endpoint')),
                      ('crossings only', dict(method='crossing'))):
        rr = []
        for c in cases:
            try:
                rr.append(eval_case(c, ex, verbose=False, **kw))
            except Exception:
                pass
        tp = sum(r['tp'] for r in rr); fp = sum(r['fp'] for r in rr); fn = sum(r['fn'] for r in rr)
        print('  %-32s P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)' % (
            title, tp / max(1, tp + fp), tp / max(1, tp + fn), _f1(tp, fp, fn), tp, fp, fn))


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


if __name__ == '__main__':
    main([a for a in sys.argv[1:]])
