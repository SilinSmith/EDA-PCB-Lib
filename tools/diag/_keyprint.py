# -*- coding: utf-8 -*-
"""Is the GT key of a mis-named IC box *readable anywhere in the image*?

For every GT `box` component whose key did not match our output we collect
every candidate designator string that the CRNN actually read in that case
(`<assigned_dir>/<case>.json` -> ``desig_slots``) and ask two questions:

  near   : is the GT key among the candidates attached to *that* box?
  any    : is the GT key present among the candidates of the case at all?

``near`` is the hard ceiling of a smarter assignment; ``any`` the ceiling of
*any* text-based route.  If both are ~0 the reference designator is simply not
in the raster for these images.

Usage::

    python work\\tools\\diag\\_keyprint.py --assigned work\\_s220_w16_1.5\\runs\\stage2\\_assigned ^
        --pred work\\_s220_w16_1.5\\runs\\stage2 --pad 60
"""
from __future__ import annotations

import argparse
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
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--assigned', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--pad', type=float, default=60.0)
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}

    tot = {'near': 0, 'any': 0, 'none': 0}
    ex = {'any': [], 'none': []}
    n = 0

    for cid in sorted(cases):
        p = os.path.join(a.pred, '%s.json' % cid)
        s = os.path.join(a.assigned, '%s.json' % cid)
        if not (os.path.exists(p) and os.path.exists(s)):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        asg = load_json(s)
        gcomp = gt.get('components') or {}
        pcomp = pr.get('components') or {}
        gt_boxes = {k: v for k, v in gcomp.items() if v.get('type') in IC_TYPES}
        if not gt_boxes:
            continue
        pr_boxes = {k: v for k, v in pcomp.items() if v.get('type') in IC_TYPES}

        # every designator-looking string the CRNN read in this case
        pool = []          # (canon_lower, box_img)
        for cands in (asg.get('desig_slots') or {}).values():
            for c in cands or []:
                t = (c.get('canon') or c.get('text') or '').strip()
                b = c.get('box')
                if t and b:
                    pool.append((t.lower(), [float(v) for v in b]))
        allstr = set(t for t, _ in pool)

        used = set()
        for gk, gv in gt_boxes.items():
            if gk in pcomp:
                continue
            gb = _bbox(gv)                      # NOTE: GT bbox is y-UP
            hit = None
            if gb:
                for pk, pv in pr_boxes.items():
                    if pk in used:
                        continue
                    pb = _bbox(pv)
                    if pb and _iou(gb, pb) >= 0.5:
                        hit, used = pk, used | {pk}
                        break
            if hit is None:
                continue
            n += 1
            tgt = gk.lower()
            if tgt in allstr:
                tot['any'] += 1
                ex['any'].append((cid, gk, hit))
            else:
                tot['none'] += 1
                ex['none'].append((cid, gk, hit))

    print('A2 (detected box, wrong key) inspected = %d' % n)
    print('  GT key among the CRNN candidates of the case : %5d  %6.1f%%'
          % (tot['any'], 100.0 * tot['any'] / max(1, n)))
    print('  GT key absent from every candidate           : %5d  %6.1f%%'
          % (tot['none'], 100.0 * tot['none'] / max(1, n)))
    print()
    print('--- GT key IS somewhere in the case (recoverable by better assignment) ---')
    for row in ex['any'][:a.top]:
        print('   %s  GT=%-10s ours=%-10s' % row)
    print()
    print('--- GT key nowhere in the case (unrecoverable from the raster) ---')
    for row in ex['none'][:a.top]:
        print('   %s  GT=%-10s ours=%-10s' % row)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
