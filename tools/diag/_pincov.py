# -*- coding: utf-8 -*-
"""Pin coverage on *matched* ICs: is the loss in the POINT or in the NUMBER?

For every GT IC component the official matcher pairs with one of ours, walk the
GT pins and ask two independent questions:

    point@5/15/30  : is there one of our pin POINTS within that many pixels?
    num_ok         : for those, does our key set contain the right pin NUMBER?

`pin.position_mismatch` uses `cfg.pin_tol` (5.0 px), so `point@5` is the strict
gate the scorer applies.

Usage::

    python work\\tools\\diag\\_pincov.py --pred work\\_submit_out220r21\\runs\\stage2
    python work\\tools\\diag\\_pincov.py --pred ... --split 10GT
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser import metrics as M                        # noqa: E402
from schematic_parser.data import IC_TYPES, discover, load_json  # noqa: E402


def _pt(info, H):
    p = (info or {}).get('point') or {}
    try:
        return float(p['x']), H - float(p['y'])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--tol', type=float, default=5.0)
    ap.add_argument('--bad', type=int, default=12)
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}
    cfg = M.EvalConfig()
    cnt = Counter()
    worst = []

    for cid in sorted(cases):
        p = os.path.join(a.pred, '%s.json' % cid)
        if not os.path.exists(p):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        H = float((pr.get('image') or {}).get('height')
                  or (gt.get('image') or {}).get('height') or 0.0)
        if not H:
            import cv2
            im = cv2.imread(cases[cid].png_path)
            H = float(im.shape[0]) if im is not None else 1.0
        _f1, km = M.assign_components(gt, pr, cfg)
        back = {g: k for k, g in km.items()}
        for gk, gv in (gt.get('components') or {}).items():
            if gv.get('type') not in IC_TYPES:
                continue
            pk = back.get(gk)
            if pk is None:
                cnt['unmatched_ic'] += 1
                cnt['unmatched_pins'] += len((gt.get('pins') or {}).get(gk) or {})
                continue
            cnt['matched_ic'] += 1
            ours = (pr.get('pins') or {}).get(pk) or {}
            o_pts, o_nums = [], set()
            for k, v in ours.items():
                q = _pt(v, H)
                if q:
                    o_pts.append(q)
                n = None
                try:
                    n = M.pin_key_to_number(k)
                except Exception:
                    pass
                if n is not None:
                    o_nums.add(str(n))
            gpn = (gt.get('pins') or {}).get(gk) or {}
            miss_p = miss_n = 0
            for k, v in gpn.items():
                cnt['gt_pin'] += 1
                try:
                    gn = M.pin_key_to_number(k)
                except Exception:
                    gn = None
                gq = _pt(v, H)
                d = None
                if gq and o_pts:
                    arr = np.asarray(o_pts, float)
                    d = float(np.hypot(arr[:, 0] - gq[0], arr[:, 1] - gq[1]).min())
                if d is None:
                    cnt['no_points_at_all'] += 1
                else:
                    if d <= a.tol:
                        cnt['point@tol'] += 1
                    if d <= 15.0:
                        cnt['point@15'] += 1
                    if d <= 30.0:
                        cnt['point@30'] += 1
                if gn is not None and str(gn) in o_nums:
                    cnt['num_ok'] += 1
                    if d is not None and d <= a.tol:
                        cnt['both_ok'] += 1
                else:
                    miss_n += 1
                    if d is not None and d <= a.tol:
                        cnt['point_ok_num_bad'] += 1
                if d is None or d > a.tol:
                    miss_p += 1
            if miss_p or miss_n:
                worst.append((miss_p + miss_n, cid, gk, pk, len(gpn), miss_p, miss_n))

    print('pred =', a.pred)
    print('  matched ICs %d (unmatched %d, carrying %d GT pins)'
          % (cnt['matched_ic'], cnt['unmatched_ic'], cnt['unmatched_pins']))
    tot = cnt['gt_pin']
    print('  GT pins on matched ICs = %d' % tot)
    for k in ('point@tol', 'point@15', 'point@30', 'num_ok', 'both_ok',
              'point_ok_num_bad', 'no_points_at_all'):
        print('  %-18s %6d  %6.1f%%' % (k, cnt[k], 100.0 * cnt[k] / max(1, tot)))
    print()
    print('worst components (miss_pins, case, GT key <- our key, gt pins, missing_point, missing_num):')
    for row in sorted(worst, reverse=True)[:a.bad]:
        print('   ', row)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
