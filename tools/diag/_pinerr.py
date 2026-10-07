# -*- coding: utf-8 -*-
"""Where exactly do our pin POINTS miss the GT pin points?

`_pincov.py` says `point@5px` is only ~17%.  This script asks the follow-up
question: among the GT pins whose *number* we did produce (so we clearly meant
the same physical pin), how far off is the point, and **in which direction**?

Reported per GT pin (only IC components matched by the official matcher):

    dn   distance from the GT point to our nearest point with the SAME number
    d    distance from the GT point to our nearest point of any number
    ox   signed outward error   (positive = our point is farther from the box)
    oy   signed along-edge error

Usage::

    python work\\tools\\diag\\_pinerr.py --pred work\\_submit_out220r21\\runs\\stage2
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser import metrics as M                        # noqa: E402
from schematic_parser.data import (IC_TYPES, discover, load_json,  # noqa: E402
                                   pin_key_to_number)


def _pct(vals, q):
    if not vals:
        return float('nan')
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def _bbox(c):
    b = c.get('bbox') or c.get('bbox_img')
    if not b:
        return None
    x0, y0, x1, y1 = (float(v) for v in b)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def edge_of(bb, x, y):
    """(edge, offset) —— 最近的那条框边 + 沿边坐标 + 朝外距离。"""
    x0, y0, x1, y1 = bb
    cand = [
        (abs(y - y0), 0, x, y0, 0.0, -1.0),
        (abs(x - x1), 1, x1, y, 1.0, 0.0),
        (abs(y - y1), 2, x, y1, 0.0, 1.0),
        (abs(x - x0), 3, x0, y, -1.0, 0.0),
    ]
    d, e, px, py, nx, ny = min(cand, key=lambda t: t[0])
    outward = (x - px) * nx + (y - py) * ny
    along = (x - px) * (-ny) + (y - py) * nx
    return e, px, py, outward, along


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--cases', default='')
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}
    if a.cases:
        want = {s.strip() for s in a.cases.split(',') if s.strip()}
        cases = {k: v for k, v in cases.items() if k in want}
    cfg = M.EvalConfig()

    dn_all, d_all, ox_all, oy_all = [], [], [], []
    n_gpins = n_samenum = n_dn5 = n_d5 = 0
    pair_rows = []

    for cid in sorted(cases):
        p = os.path.join(a.pred, '%s.json' % cid)
        if not os.path.exists(p):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        det = {}
        _f1, km = M.assign_components(gt, pr, cfg, detail=det)
        if not km:
            continue
        back = {}
        for pk, gk in km.items():
            back.setdefault(gk, []).append(pk)
        gpins = gt.get('pins') or {}
        ppins = pr.get('pins') or {}
        pcomp = pr.get('components') or {}
        for gk, pks in back.items():
            gv = (gt.get('components') or {}).get(gk) or {}
            if M.canon_type(gv.get('type')) not in IC_TYPES:
                continue
            gb = _bbox(gv)
            g = gpins.get(gk) or {}
            if not gb or not g:
                continue
            # union of the pins of every prediction mapped onto this GT key
            ournums, ourpts = {}, []
            for pk in pks:
                for k, info in (ppins.get(pk) or {}).items():
                    pt = info.get('point') if isinstance(info, dict) else None
                    if not pt:
                        continue
                    x, y = float(pt['x']), float(pt['y'])
                    ourpts.append((x, y, pin_key_to_number(k)))
                    ournums.setdefault(pin_key_to_number(k), (x, y))
            if not ourpts:
                continue
            e_g, px, py, _o, _a = edge_of(gb, *[float(v) for v in (gb[0], gb[1])])
            for key, gi in g.items():
                pt = (gi or {}).get('point')
                if not pt:
                    continue
                gx, gy = float(pt['x']), float(pt['y'])
                num = pin_key_to_number(key)
                n_gpins += 1
                d_all.append(min(math.hypot(gx - ux, gy - uy) for ux, uy, _ in ourpts))
                if d_all[-1] <= 5.0:
                    n_d5 += 1
                if num in ournums:
                    ux, uy = ournums[num]
                    n_samenum += 1
                    dn = math.hypot(gx - ux, gy - uy)
                    dn_all.append(dn)
                    if dn <= 5.0:
                        n_dn5 += 1
                    _e, _px, _py, out_g, al_g = edge_of(gb, gx, gy)
                    _e2, _px2, _py2, out_u, al_u = edge_of(gb, ux, uy)
                    ox_all.append(out_u - out_g)
                    oy_all.append(al_u - al_g)
                    if len(pair_rows) < 4000:
                        pair_rows.append((cid, gk, num, dn, out_u - out_g, al_u - al_g))

    print('pred =', a.pred)
    print('GT box pins (on matched comps) = %d' % n_gpins)
    print()
    print('nearest point, ANY number      : d<=5 %5d / %5d = %5.1f%%   '
          'p25=%.1f p50=%.1f p75=%.1f'
          % (n_d5, n_gpins, 100.0 * n_d5 / max(1, n_gpins),
             _pct(d_all, .25), _pct(d_all, .5), _pct(d_all, .75)))
    print('nearest point, SAME number     : d<=5 %5d / %5d = %5.1f%%   '
          'p25=%.1f p50=%.1f p75=%.1f'
          % (n_dn5, n_samenum, 100.0 * n_dn5 / max(1, n_samenum),
             _pct(dn_all, .25), _pct(dn_all, .5), _pct(dn_all, .75)))
    print('  we produced the number for   : %d / %d = %.1f%%'
          % (n_samenum, n_gpins, 100.0 * n_samenum / max(1, n_gpins)))
    print()
    print('signed OUTWARD error (our - GT), same-number pairs  n=%d' % len(ox_all))
    print('   p10=%+.1f  p25=%+.1f  p50=%+.1f  p75=%+.1f  p90=%+.1f  mean=%+.2f'
          % (_pct(ox_all, .10), _pct(ox_all, .25), _pct(ox_all, .5),
             _pct(ox_all, .75), _pct(ox_all, .90),
             sum(ox_all) / max(1, len(ox_all))))
    print('signed ALONG-EDGE error (our - GT), same-number pairs')
    print('   p10=%+.1f  p25=%+.1f  p50=%+.1f  p75=%+.1f  p90=%+.1f  mean=%+.2f'
          % (_pct(oy_all, .10), _pct(oy_all, .25), _pct(oy_all, .5),
             _pct(oy_all, .75), _pct(oy_all, .90),
             sum(oy_all) / max(1, len(oy_all))))
    # how many same-number pairs would pass at a shifted outward distance?
    for shift in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        ok = 0
        for _c, _g, _n, dn, ox, oy in pair_rows:
            if math.hypot(ox + shift, oy) <= 5.0:
                ok += 1
        print('   shift outward %+d px -> same-number within 5px: %d / %d = %.1f%%'
              % (shift, ok, len(pair_rows), 100.0 * ok / max(1, len(pair_rows))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
