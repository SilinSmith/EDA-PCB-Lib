# -*- coding: utf-8 -*-
"""Bucket the A1/A2/A3/A4 IC-box accounting of `_icmatch.py` by SOURCE TYPE.

The dataset filenames carry the provenance of every case
(``0004-other.png`` / ``0113-jlc.png`` / ``0003-Altium Designer.png`` ...).
This script runs the official matcher per case and aggregates the four buckets
below by that provenance, so we can see *which* sources are recoverable.

    A1  matched by key          (exact string)
    A2  matched by coordinates  (GT key was unkeyed)
    A3  detected but unmatched  (a prediction overlaps the GT box)
    A4  not detected at all

Usage::

    python work\\tools\\diag\\_icsource.py --pred work\\_submit_out220r21\\runs\\stage2
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser import metrics as M                        # noqa: E402
from schematic_parser.data import IC_TYPES, discover, load_json  # noqa: E402

SRC_RE = re.compile(r'-(?P<src>[^-]+?)_target', re.I)


def _src(case) -> str:
    base = os.path.basename(case.json_path)
    m = SRC_RE.search(base)
    if m:
        return m.group('src').strip()
    return os.path.basename(case.png_path).rsplit('.', 1)[0].split('-', 1)[-1]


def _bbox(c):
    b = c.get('bbox')
    if not b:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    den = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / den if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--split', default='200_train_cases')
    a = ap.parse_args()

    cases = [c for c in discover() if c.split == a.split]
    cfg = M.EvalConfig()

    agg = defaultdict(lambda: {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,
                               'ncase': 0, 'pins1': 0, 'pins3': 0})
    # a GT key is "recoverable" when GT Name == GT key (the symbol prints the
    # designator as its name too); count that inside each source / bucket
    rec = defaultdict(lambda: {'A3': 0, 'A3_rec': 0, 'A4': 0, 'A4_rec': 0})

    for c in cases:
        p = os.path.join(a.pred, '%s.json' % c.case_id)
        if not os.path.exists(p):
            continue
        gt = load_json(c.json_path)
        pr = load_json(p)
        det = {}
        _f1, km = M.assign_components(gt, pr, cfg, detail=det)
        matched_g = set(km.values())
        gcomp = gt.get('components') or {}
        pcomp = pr.get('components') or {}
        gpins = gt.get('pins') or {}
        gboxes = {k: v for k, v in gcomp.items() if v.get('type') in IC_TYPES}
        pboxes = {k: v for k, v in pcomp.items() if v.get('type') in IC_TYPES}
        s = _src(c)
        d = agg[s]
        r = rec[s]
        d['ncase'] += 1
        for gk, gv in gboxes.items():
            npin = len(gpins.get(gk) or {})
            if gk in matched_g:
                b = 'A1' if gk in pcomp else 'A2'
                d[b] += 1
                if b == 'A1':
                    d['pins1'] += npin
                continue
            gb = _bbox(gv)
            hit = False
            if gb:
                for _pk, pv in pboxes.items():
                    pb = _bbox(pv)
                    if pb and _iou(gb, pb) >= 0.5:
                        hit = True
                        break
            b = 'A3' if hit else 'A4'
            d[b] += 1
            r[b] += 1
            if (gv.get('Name') or '') == gk:
                r[b + '_rec'] += 1
            if b == 'A3':
                d['pins3'] += npin

    print('pred =', a.pred)
    print()
    hdr = ('%-22s %5s %6s | %5s %5s %5s %5s | %8s %8s | A3 rec  A4 rec'
           % ('source', 'cases', 'GTbox', 'A1', 'A2', 'A3', 'A4',
              'pins(A1)', 'pins(A3)'))
    print(hdr)
    print('-' * len(hdr))
    T = defaultdict(int)
    R = defaultdict(int)
    for s in sorted(agg, key=lambda k: -(agg[k]['A3'] + agg[k]['A4'])):
        d = agg[s]
        n = d['A1'] + d['A2'] + d['A3'] + d['A4']
        r = rec[s]
        print('%-22s %5d %6d | %5d %5d %5d %5d | %8d %8d | %5d/%d %5d/%d'
              % (s, d['ncase'], n, d['A1'], d['A2'], d['A3'], d['A4'],
                 d['pins1'], d['pins3'],
                 r['A3_rec'], r['A3'], r['A4_rec'], r['A4']))
        for k in ('A1', 'A2', 'A3', 'A4', 'pins1', 'pins3', 'ncase'):
            T[k] += d[k]
        for k in ('A3', 'A3_rec', 'A4', 'A4_rec'):
            R[k] += r[k]
    n = T['A1'] + T['A2'] + T['A3'] + T['A4']
    print('-' * len(hdr))
    print('%-22s %5d %6d | %5d %5d %5d %5d | %8d %8d | %5d/%d %5d/%d'
          % ('TOTAL', T['ncase'], n, T['A1'], T['A2'], T['A3'], T['A4'],
             T['pins1'], T['pins3'],
             R['A3_rec'], R['A3'], R['A4_rec'], R['A4']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
