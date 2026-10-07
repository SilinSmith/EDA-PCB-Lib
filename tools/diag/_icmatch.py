# -*- coding: utf-8 -*-
"""Use the OFFICIAL matcher to see how GT IC boxes are matched.

`_icbox.py` compares component keys literally, but the official
`metrics.assign_components` also pairs *unkeyed* GT components (`pNNN`/`\u2205NNN`)
with predictions by coordinates.  This script therefore reports the truth:

    A1  matched by key          (exact string)
    A2  matched by coordinates  (GT key was unkeyed)
    A3  detected but unmatched  (a prediction overlaps the GT box)
    A4  not detected at all

Usage::

    python work\\tools\\diag\\_icmatch.py --pred work\\_s220_w16_1.5\\runs\\stage2
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser import metrics as M                       # noqa: E402
from schematic_parser.data import IC_TYPES, discover, load_json  # noqa: E402


def _bbox(c):
    b = c.get('bbox')
    if not b:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--show', default='')
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}
    cfg = M.EvalConfig()

    tot = {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0}
    pins = {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0}
    ex = {k: [] for k in tot}

    for cid in sorted(cases):
        p = os.path.join(a.pred, '%s.json' % cid)
        if not os.path.exists(p):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        det = {}
        _f1, km = M.assign_components(gt, pr, cfg, detail=det)
        matched_g = set(km.values())
        gcomp = gt.get('components') or {}
        pcomp = pr.get('components') or {}
        gpins = gt.get('pins') or {}
        gboxes = {k: v for k, v in gcomp.items() if v.get('type') in IC_TYPES}
        if not gboxes:
            continue
        pboxes = {k: v for k, v in pcomp.items() if v.get('type') in IC_TYPES}
        for gk, gv in gboxes.items():
            npin = len(gpins.get(gk) or {})
            if gk in matched_g:
                b = 'A1' if gk in pcomp else 'A2'
                tot[b] += 1
                pins[b] += npin
                continue
            gb = _bbox(gv)
            ov = None
            if gb:
                for pk, pv in pboxes.items():
                    pb = _bbox(pv)
                    if pb and _iou(gb, pb) >= 0.5:
                        ov = pk
                        break
            b = 'A3' if ov else 'A4'
            tot[b] += 1
            pins[b] += npin
            if len(ex[b]) < 4000:
                ex[b].append((cid, gk, ov, (gv.get('Name') or '')[:22]))

    n = sum(tot.values())
    print('pred =', a.pred)
    print('GT box components = %d' % n)
    lab = {'A1': 'matched by KEY',
           'A2': 'matched by COORDINATES (GT key unkeyed)',
           'A3': 'detected box, NOT matched',
           'A4': 'box not detected'}
    for k in ('A1', 'A2', 'A3', 'A4'):
        print('  %s  %-38s %5d  %6.1f%%   pins %6d'
              % (k, lab[k], tot[k], 100.0 * tot[k] / max(1, n), pins[k]))
    for k in ('A3', 'A4'):
        print()
        print('--- %s samples ---' % k)
        for row in ex[k][:a.top]:
            print('   %s  GT=%-12s ours=%-12s Name=%s' % row)

    if a.show:
        cid = a.show
        gt = load_json(cases[cid].json_path)
        pr = load_json(os.path.join(a.pred, '%s.json' % cid))
        det = {}
        _f1, km = M.assign_components(gt, pr, cfg, detail=det)
        print()
        print('=== case %s : keymap (%d pairs) ===' % (cid, len(km)))
        gb = {k: v for k, v in (gt.get('components') or {}).items()
              if v.get('type') in IC_TYPES}
        for gk, gv in gb.items():
            back = [pk for pk, v in km.items() if v == gk]
            print('  GT %-10s bbox=%s Name=%-18s <- %s'
                  % (gk, _bbox(gv), gv.get('Name'), back or 'NONE'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
