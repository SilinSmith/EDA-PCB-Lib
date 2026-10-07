# -*- coding: utf-8 -*-
"""Could geometric *pin stubs* fill the pins that have no wire attached?

`stage1/pins.py:find_pins` only creates a pin where a **wire segment touches** the
box (or, if there is no touch at all, from a type prior).  GT, however, lists
every pin of the symbol -- including unconnected ones.  `stage1/pintrace.py:
find_pins` looks for geometric stubs *outside* the box regardless of wiring, so
it can see them.

This script, for matched IC boxes, compares

    GT pins
    touch points (what stage1/pins.py produces)
    stub points (what stage1/pintrace.find_pins produces, pushed 10 px out so it
                 is on GT's convention -- GT points sit a median 10 px outside)

and reports, within `--tol` px, how many GT pins each source covers.

Usage::

    python work\\tools\\diag\\_stubfill.py --cases 0087,0080,0001 --tol 5
    python work\\tools\\diag\\_stubfill.py --split 200_train_cases --limit 40
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser import metrics as M                        # noqa: E402
from schematic_parser.data import IC_TYPES, discover, load_json  # noqa: E402
from stage1 import pintrace as PT                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=r'work\_submit_out220r21\runs\stage2')
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--cases', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tol', type=float, default=5.0)
    ap.add_argument('--band', type=float, default=16.0)
    ap.add_argument('--push', type=float, default=6.0)
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}
    cfg = M.EvalConfig()
    want = [s.strip() for s in a.cases.split(',') if s.strip()]
    ids = want or sorted(cases)
    if a.limit:
        ids = ids[:a.limit]
    tot = Counter()

    for cid in ids:
        p = os.path.join(a.pred, '%s.json' % cid)
        if not os.path.exists(p):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        bgr = cv2.imread(cases[cid].png_path)
        if bgr is None:
            continue
        H = float(bgr.shape[0])
        ink = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < PT.INK).astype(np.uint8)
        _f1, km = M.assign_components(gt, pr, cfg)
        back = {g: k for k, g in km.items()}
        for gk, gv in (gt.get('components') or {}).items():
            if gv.get('type') not in IC_TYPES:
                continue
            if gk not in back:
                continue
            bb = gv.get('bbox')
            if not bb:
                continue
            x0, y0, x1, y1 = [float(v) for v in bb]
            box = (x0, H - y1, x1, H - y0)
            gpts = []
            for k, v in ((gt.get('pins') or {}).get(gk) or {}).items():
                q = (v or {}).get('point') or {}
                try:
                    gpts.append((float(q['x']), H - float(q['y'])))
                except Exception:
                    pass
            if not gpts:
                continue
            garr = np.asarray(gpts, float)

            ours = []
            for k, v in ((pr.get('pins') or {}).get(back[gk]) or {}).items():
                q = (v or {}).get('point') or {}
                try:
                    ours.append((float(q['x']), H - float(q['y'])))
                except Exception:
                    pass
            oarr = np.asarray(ours, float) if ours else np.zeros((0, 2))

            stubs = []
            for pin in PT.find_pins(ink, [box], band=a.band):
                stubs.append((pin.x + pin.dx * a.push, pin.y + pin.dy * a.push))
            sarr = np.asarray(stubs, float) if stubs else np.zeros((0, 2))

            tot['gt'] += len(gpts)
            tot['ours'] += len(ours)
            tot['stubs'] += len(stubs)
            for arr, tag in ((oarr, 'cov_ours'), (sarr, 'cov_stubs')):
                if arr.shape[0] == 0:
                    continue
                d = np.hypot(arr[None, :, 0] - garr[:, None, 0],
                             arr[None, :, 1] - garr[:, None, 1]).min(axis=1)
                tot[tag] += int((d <= a.tol).sum())
                tot[tag + '_all'] += int(d.size)
            if oarr.shape[0] and sarr.shape[0]:
                d2 = np.hypot(oarr[:, None, 0] - sarr[None, :, 0],
                              oarr[:, None, 1] - sarr[None, :, 1])
                tot['stub_new'] += int((d2.min(axis=0) > a.tol).sum())

    g = max(1, tot['gt'])
    print('cases %d' % len(ids))
    print('  GT pins            %6d' % tot['gt'])
    print('  our touch points   %6d  (%.3f of GT)' % (tot['ours'], tot['ours'] / g))
    print('  stub points        %6d  (%.3f of GT)' % (tot['stubs'], tot['stubs'] / g))
    print('  stubs not near a touch point : %6d' % tot['stub_new'])
    print('  GT pins covered by touch @%.0fpx : %6d  (%.3f)'
          % (a.tol, tot['cov_ours'], tot['cov_ours'] / g))
    print('  GT pins covered by stubs @%.0fpx : %6d  (%.3f)'
          % (a.tol, tot['cov_stubs'], tot['cov_stubs'] / g))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
