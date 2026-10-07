# -*- coding: utf-8 -*-
"""Skeleton-extraction validation against the official reference wire skeletons.

The `Dataset300我不知道干嘛用的先放置` cases carry the vendor pipeline's own intermediate
representation, and `edges[E].keyPoints.skeleton.wire` is the **per-pixel skeleton of every
wire net**.  That is an exact ground truth for the wire centreline, so this script measures
`pskel.py` directly:

  mask_recall@r    fraction of reference skeleton pixels covered by my thin mask (dilate r)
  skel_recall@r    fraction of reference skeleton pixels within r px of my skeleton
  skel_prec@r      fraction of my skeleton pixels within r px of the reference skeleton
  seg_recall@r     fraction of reference polyline geometry covered by my segments
  seg_prec@r       fraction of my segment geometry covered by the reference geometry
  end_hit@r        reference endPoints matched by one of my endpoints
  turn_hit@r       reference turnPoints matched by one of my segment vertices

Usage:  python -m pipeline.validate_skel [case_id ...] [--sweep]
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
from pipeline import pcore, pskel


# ------------------------------------------------------------------ reference

def ref_of(case) -> Dict:
    """Reference wire skeleton / polyline / vertices for one case (if present)."""
    if not case.ref_json or not os.path.exists(case.ref_json):
        return {}
    with open(case.ref_json, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    skel: List[Tuple[float, float]] = []
    polys: List[List[Tuple[float, float]]] = []
    ends: List[Tuple[float, float]] = []
    turns: List[Tuple[float, float]] = []
    for e in (doc.get('edges') or {}).values():
        if not isinstance(e, dict):
            continue
        kp = e.get('keyPoints') or {}
        for p in ((kp.get('skeleton') or {}).get('wire')) or []:
            if p and len(p) >= 2:
                skel.append((float(p[0]), float(p[1])))
        for p in ((kp.get('endPoints') or {}).get('wire')) or []:
            if p and len(p) >= 2:
                ends.append((float(p[0]), float(p[1])))
        for p in ((kp.get('turnPoints') or {}).get('wire')) or []:
            if p and len(p) >= 2:
                turns.append((float(p[0]), float(p[1])))
        for path in ((e.get('keyPointPaths') or {}).get('wire')) or []:
            pts = [(float(q[0]), float(q[1])) for q in path if q and len(q) >= 2]
            if len(pts) >= 2:
                polys.append(pts)
    return dict(skel=skel, polys=polys, ends=ends, turns=turns, doc=doc)


# ------------------------------------------------------------------ geometry

class Grid:
    """Uniform-grid nearest-point index.  Cell size is the search radius."""

    def __init__(self, pts: Sequence[Tuple[float, float]], cell: float):
        self.cell = max(float(cell), 1.0)
        self.pts = np.asarray(pts, np.float64) if len(pts) else np.zeros((0, 2))
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        for i, (x, y) in enumerate(self.pts):
            self.grid.setdefault((int(x // self.cell), int(y // self.cell)), []).append(i)

    def near(self, p, r: float) -> bool:
        if len(self.pts) == 0:
            return False
        x, y = float(p[0]), float(p[1])
        c = self.cell
        gx, gy = int(x // c), int(y // c)
        rr = r * r
        n = int(math.ceil(r / c)) or 1
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for i in self.grid.get((gx + dx, gy + dy), ()):  # type: ignore
                    ddx = self.pts[i, 0] - x
                    ddy = self.pts[i, 1] - y
                    if ddx * ddx + ddy * ddy <= rr:
                        return True
        return False


def sample_segments(segs, step: float = 1.0):
    out = []
    for a, b in segs:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / step))
        for i in range(n + 1):
            t = i / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def poly_to_segs(polys):
    segs = []
    for p in polys:
        for a, b in zip(p[:-1], p[1:]):
            segs.append((a, b))
    return segs


def frac_hit(pts, mask) -> float:
    """Fraction of *pts* landing on a non-zero pixel of *mask*."""
    if not pts:
        return float('nan')
    h, w = mask.shape[:2]
    good = 0
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            good += 1
    return good / len(pts)


def frac_near(pts, index: Grid, r: float) -> float:
    if not pts or len(index.pts) == 0:
        return float('nan')
    good = sum(1 for p in pts if index.near(p, r))
    return good / len(pts)


def rasterize(pts, shape, r: int = 0) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < shape[0] and 0 <= xi < shape[1]:
            m[yi, xi] = 255
    if r > 0:
        m = cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    return m


# ------------------------------------------------------------------ per-case eval

def eval_case(case, ex: pskel.WireExtractor, radii=(1.0, 2.0, 3.0), verbose=True,
              ref: Optional[Dict] = None) -> Dict:
    if ref is None:
        ref = ref_of(case)
    if not ref or not ref['skel']:
        return {}
    bgr = cv2.imread(case.png, cv2.IMREAD_COLOR)
    res = ex.extract(bgr)
    shape = bgr.shape

    my_skel = _skel_px(res.skel)
    my_segs = res.segments
    my_samp = sample_segments(my_segs)
    ref_segs = poly_to_segs(ref['polys'])
    ref_samp = sample_segments(ref_segs)

    out = dict(case=case.case_id, source=case.source, has_chroma=res.has_chroma,
               n_ref_skel=len(ref['skel']), n_ref_seg=len(ref['polys']),
               n_my_seg=len(my_segs), n_my_poly=len(res.polylines),
               n_nodes=res.n_nodes, n_branch=res.n_branch_paths,
               n_my_end=len(res.endpoints), n_ref_end=len(ref['ends']),
               n_ref_turn=len(ref['turns']),
               n_ref_skel_px=len(my_skel))
    for r in radii:
        out['mask_recall@%g' % r] = frac_hit(ref['skel'], res.thin)
        out['skel_recall@%g' % r] = frac_near(ref['skel'], Grid(my_skel, r), r)
        out['skel_prec@%g' % r] = frac_near(my_skel, Grid(ref['skel'], r), r)
    for r in radii:
        out['seg_recall@%g' % r] = frac_near(ref_samp, Grid(my_samp, r), r)
        out['seg_prec@%g' % r] = frac_near(my_samp, Grid(ref_samp, r), r)
    out['end_hit@2'] = frac_near(ref['ends'], Grid(res.endpoints, 2.0), 2.0) \
        if ref['ends'] else float('nan')
    verts = []
    for p in res.polylines:
        verts.extend(p[1:-1])
    out['turn_hit@2'] = frac_near(ref['turns'], Grid(verts, 2.0), 2.0) \
        if ref['turns'] else float('nan')
    if verbose:
        print('  '.join(['%s(%s)' % (out['case'], out['source']),
                         'ref_skel=%d' % out['n_ref_skel'],
                         'mine_seg=%d' % out['n_my_seg'],
                         'maskR@2=%.3f' % out['mask_recall@2'],
                         'skelR@2=%.3f' % out['skel_recall@2'],
                         'skelP@2=%.3f' % out['skel_prec@2'],
                         'segR@2=%.3f' % out['seg_recall@2'],
                         'segP@2=%.3f' % out['seg_prec@2'],
                         'end@2=%.2f' % out['end_hit@2'],
                         'turn@2=%.2f' % out['turn_hit@2']]))
    return out


def _skel_px(skel: np.ndarray) -> List[Tuple[float, float]]:
    ys, xs = np.nonzero(skel)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


# ------------------------------------------------------------------ reporting

def summarize(rows: List[Dict], title: str):
    if not rows:
        return
    print('\n== %s (%d cases) ==' % (title, len(rows)))
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    for k in keys:
        vals = [r[k] for r in rows if not math.isnan(r.get(k, float('nan')))]
        if vals:
            print('  %-16s %.4f   (n=%d)' % (k, sum(vals) / len(vals), len(vals)))


def run(cases, ex: pskel.WireExtractor, title: str, radii=(1.0, 2.0, 3.0)) -> List[Dict]:
    rows = []
    for c in cases:
        ref = ref_of(c)
        if not ref or not ref['skel']:
            continue
        r = eval_case(c, ex, radii=radii, ref=ref)
        if r:
            rows.append(r)
    summarize(rows, title)
    return rows


DEFAULT_IDS = ['0059', '0095', '0112', '0140', '0170', '0182', '0253', '0124']


def main(argv):
    sweep = '--sweep' in argv
    ids = [a for a in argv if not a.startswith('-')] or DEFAULT_IDS
    cases = [c for c in (pcore.find_case(i) for i in ids) if c is not None]
    if not cases:
        print('no cases matched'); return
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    rows = run(cases, pskel.WireExtractor(), 'baseline (adaptive + colour + thin1.6)')
    with open(os.path.join(C.OUT_ROOT, 'validate_skel.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    if not sweep:
        return
    # one-factor-at-a-time over the knobs that matter
    cfgs = [
        ('thin_w=1.2', dict(thin_w=1.2)),
        ('thin_w=2.0', dict(thin_w=2.0)),
        ('thin_w=2.5', dict(thin_w=2.5)),
        ('thin_w=3.0', dict(thin_w=3.0)),
        ('thin_w=1e9 (no stroke gate)', dict(thin_w=1e9)),
        ('adap_c=4', dict(adap_c=4)),
        ('adap_c=12', dict(adap_c=12)),
        ('adap_c=20', dict(adap_c=20)),
        ('adap_k=61', dict(adap_k=61)),
        ('close_ink', dict(close_ink=True)),
        ('close_thin', dict(close_thin=True)),
        ('dilate_to=1', dict(dilate_to=1)),
        ('ink=otsu', dict(ink_mode='otsu')),
        ('ink=bgdiff', dict(ink_mode='bgdiff')),
        ('no colour gate', dict(use_colour=False)),
        ('no colour + close_ink', dict(use_colour=False, close_ink=True)),
        ('no colour + thin2.0', dict(use_colour=False, thin_w=2.0)),
        ('no colour + thin1e9 (ink only)', dict(use_colour=False, thin_w=1e9)),
        ('no colour + close_ink + ink=otsu', dict(use_colour=False, close_ink=True, ink_mode='otsu')),
        ('no colour + close_ink + ink=bgdiff', dict(use_colour=False, close_ink=True, ink_mode='bgdiff')),
        ('no colour + close_ink + thin1e9', dict(use_colour=False, close_ink=True, thin_w=1e9)),
        ('no colour + close_ink + thin2.5', dict(use_colour=False, close_ink=True, thin_w=2.5)),
        ('no snap', dict(use_colour=False, close_ink=True, snap=False)),
        ('no snap + rdp1.0', dict(use_colour=False, close_ink=True, snap=False, rdp_eps=1.0)),
        ('no snap + rdp2.5', dict(use_colour=False, close_ink=True, snap=False, rdp_eps=2.5)),
        ('no snap + rdp0.5', dict(use_colour=False, close_ink=True, snap=False, rdp_eps=0.5)),
        ('min_branch_len=1', dict(use_colour=False, close_ink=True, min_branch_len=1.0)),
        ('min_branch_len=6', dict(use_colour=False, close_ink=True, min_branch_len=6.0)),
    ]
    allrows = {'baseline': rows}
    for title, kw in cfgs:
        r = run(cases, pskel.WireExtractor(**kw), title)
        allrows[title] = r
    with open(os.path.join(C.OUT_ROOT, 'validate_skel_sweep.json'), 'w', encoding='utf-8') as f:
        json.dump(allrows, f, ensure_ascii=False, indent=1)
    print('\n== ablated summary (maskR@2 / skelR@2 / skelP@2 / segR@2 / segP@2) ==')
    for title, rr in allrows.items():
        if not rr:
            continue
        def m(k):
            v = [x[k] for x in rr if not math.isnan(x.get(k, float('nan')))]
            return sum(v) / len(v) if v else float('nan')
        print('  %-28s mR %.3f  sR %.3f  sP %.3f  gR %.3f  gP %.3f  nseg %.0f' % (
            title, m('mask_recall@2'), m('skel_recall@2'), m('skel_prec@2'),
            m('seg_recall@2'), m('seg_prec@2'), m('n_my_seg')))


if __name__ == '__main__':
    main([a for a in sys.argv[1:]])
