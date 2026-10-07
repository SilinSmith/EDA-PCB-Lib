# -*- coding: utf-8 -*-
"""For every GT `box` (IC) component, decide WHY it failed the key match.

Buckets
-------
  A1 key_matched      : our output has a component with *exactly* the GT key.
  A2 box_detected     : no key match, but some predicted component overlaps the
                        GT box strongly (IoU >= --iou or centre distance small).
                        => detection is fine, only the *name* is wrong.
  A3 box_missing      : no key match and no spatially overlapping prediction.
                        => we never found the box at all.

Usage::

    python work\\tools\\diag\\_icbox.py --pred work\\_s220_w16_1.5\\runs\\stage2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

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
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--show', default='', help='dump one case verbatim')
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}

    tot = {'A1': 0, 'A2': 0, 'A3': 0}
    pins = {'A1': 0, 'A2': 0, 'A3': 0}
    per_case = []
    names = []

    for cid in sorted(cases):
        p = os.path.join(a.pred, '%s.json' % cid)
        if not os.path.exists(p):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        gcomp = gt.get('components') or {}
        pcomp = pr.get('components') or {}
        gpins = gt.get('pins') or {}

        gt_boxes = {k: v for k, v in gcomp.items()
                    if (v.get('type') in IC_TYPES)}
        if not gt_boxes:
            continue
        pr_boxes = {k: v for k, v in pcomp.items()
                    if (v.get('type') in IC_TYPES)}
        used = set()
        for gk, gv in gt_boxes.items():
            npin = len(gpins.get(gk) or {})
            if gk in pcomp:
                tot['A1'] += 1
                pins['A1'] += npin
                continue
            gb = _bbox(gv)
            best, bk = 0.0, None
            if gb:
                for pk, pv in pr_boxes.items():
                    if pk in used:
                        continue
                    pb = _bbox(pv)
                    if not pb:
                        continue
                    s = _iou(gb, pb)
                    if s > best:
                        best, bk = s, pk
            if bk is not None and best >= a.iou:
                tot['A2'] += 1
                pins['A2'] += npin
                used.add(bk)
                names.append((cid, gk, bk, round(best, 3),
                              (gv.get('Name') or '')[:24],
                              (pr_boxes[bk].get('Name') or '')[:24]))
            else:
                tot['A3'] += 1
                pins['A3'] += npin
        per_case.append((cid, len(gt_boxes)))

    print('pred =', a.pred)
    print('GT box components total = %d  (over %d cases)'
          % (sum(tot.values()), len(per_case)))
    for k in ('A1', 'A2', 'A3'):
        n = tot[k]
        print('  %-2s %-14s %5d boxes  %6.1f%%   pins %6d'
              % (k,
                 {'A1': 'key_matched', 'A2': 'box_ok/name_bad',
                  'A3': 'box_missing'}[k],
                 n, 100.0 * n / max(1, sum(tot.values())), pins[k]))
    print()
    print('--- A2 samples (GT key | our key | IoU | GT Name | our Name) ---')
    for row in names[:a.top]:
        print('  %s  %-8s <- %-8s iou=%.2f  %-24s %-24s' % row)

    if a.show:
        cid = a.show
        p = os.path.join(a.pred, '%s.json' % cid)
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        gb = {k: v for k, v in (gt.get('components') or {}).items()
              if v.get('type') in IC_TYPES}
        pb = {k: v for k, v in (pr.get('components') or {}).items()
              if v.get('type') in IC_TYPES}
        print()
        print('=== case %s : GT %d boxes / pred %d boxes ===' % (cid, len(gb), len(pb)))
        print('GT:')
        for k, v in gb.items():
            print('   %-10s bbox=%s Name=%s' % (k, _bbox(v), v.get('Name')))
        print('PRED:')
        for k, v in pb.items():
            print('   %-10s bbox=%s Name=%s' % (k, _bbox(v), v.get('Name')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
