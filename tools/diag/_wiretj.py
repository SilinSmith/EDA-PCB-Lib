# -*- coding: utf-8 -*-
"""Can a pure-CV *T-junction* count pick the wire colour?

Fifth attempt at replacing the one VLM question ("which colour are the wires?")
with geometry.  Idea: a schematic wire is the only long thin stroke that carries
**T-junctions** (a branch tapping into it).  Frame borders only ever meet at L
corners; component outlines are closed rectangles with no branches.  So: build
the hue mask, remove everything inside the component boxes (dilated), skeletonise,
and count pixels with >= 3 skeleton neighbours.

For every candidate hue bucket the script prints

    px     : pixels of that hue inside the global chroma mask
    junc   : T-junction pixels (>= 3 neighbours)
    j/1k   : junctions per 1000 kept pixels

Usage::

    python work\\tools\\diag\\_wiretj.py --cases 0001,0103,2,3,9 --bins 30
    python work\\tools\\diag\\_wiretj.py --split 10GT
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser.data import IC_TYPES, discover, load_json  # noqa: E402

try:
    from pipeline import pskel                      # noqa: E402
except Exception:                                    # pragma: no cover
    pskel = None


def _bbox_boxes(gt):
    out = []
    for k, v in (gt.get('components') or {}).items():
        b = v.get('bbox')
        if not b:
            continue
        try:
            x0, y0, x1, y1 = (float(t) for t in b)
        except (TypeError, ValueError):
            continue
        out.append([x0, y0, x1, y1])
    return out


def _to_img(box, H):
    """GT bbox is y-UP; return row range (y-down) plus x range."""
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return int(x0), int(H - y1), int(x1), int(H - y0)


def _skeleton(mask):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask > 0)
    except Exception:
        th = (mask > 0).astype(np.uint8) * 255
        return cv2.ximgproc.thinning(th) > 0 if hasattr(cv2, 'ximgproc') else th > 0


def _junctions(sk):
    si = sk.astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D(si, -1, k, borderType=cv2.BORDER_CONSTANT)
    return (si > 0) & (nb >= 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='')
    ap.add_argument('--split', default='10GT')
    ap.add_argument('--bins', type=int, default=30, help='hue bucket width (deg)')
    ap.add_argument('--tol', type=float, default=42.0)
    ap.add_argument('--pad', type=int, default=8)
    ap.add_argument('--minlen', type=int, default=60,
                    help='drop connected components whose bbox is shorter than this')
    ap.add_argument('--top', type=int, default=6)
    a = ap.parse_args()

    want = [s.strip() for s in a.cases.split(',') if s.strip()]
    cases = [c for c in discover()
             if c.json_path and (c.case_id in want if want else c.split == a.split)]

    for c in cases:
        bgr = cv2.imread(c.png_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print('!! cannot read', c.png_path)
            continue
        H, W = bgr.shape[:2]
        gt = load_json(c.json_path)
        chroma = pskel.chroma(bgr)
        ink = (chroma >= 22.0) if pskel else (bgr.max(2).astype(np.int16)
                                              - bgr.min(2).astype(np.int16) >= 22)
        n_ink = int(ink.sum())
        gray = n_ink < 200
        print()
        print('=== %s/%s  %dx%d  ink=%d  %s'
              % (c.split, c.case_id, W, H, n_ink,
                 'GRAYSCALE (early return)' if gray else 'colour'))

        # blank out every component box (dilated) so symbol outlines cannot vote
        keep = np.ones((H, W), bool)
        boxes = []
        for b in _bbox_boxes(gt):
            x0, r0, x1, r1 = _to_img(b, H)
            x0 = max(0, x0 - a.pad); r0 = max(0, r0 - a.pad)
            x1 = min(W, x1 + a.pad); r1 = min(H, r1 + a.pad)
            if x1 <= x0 or r1 <= r0:
                continue
            keep[r0:r1, x0:x1] = False
            boxes.append((x0, r0, x1, r1))
        # frame band: outer 5 %
        fb = max(4, int(min(W, H) * 0.05))
        keep[:fb, :] = False; keep[-fb:, :] = False
        keep[:, :fb] = False; keep[:, -fb:] = False

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.int16)        # OpenCV hue: 0..179
        hd = h * 2.0                             # degrees
        rows = []
        for h0 in range(0, 360, a.bins):
            d = np.abs(((hd - h0 + 180.0) % 360.0) - 180.0)
            m = (ink & (d <= a.tol) & keep).astype(np.uint8)
            n = int(m.sum())
            if n < 40:
                continue
            if a.minlen > 0:
                nl, lab, stats, _c = cv2.connectedComponentsWithStats(m, 8)
                keepcc = np.zeros(nl, bool)
                for i in range(1, nl):
                    if max(stats[i, 2], stats[i, 3]) >= a.minlen:
                        keepcc[i] = True
                m = keepcc[lab].astype(np.uint8)
                n = int(m.sum())
                if n < 40:
                    continue
            sk = _skeleton(m)
            j = _junctions(sk)
            nj = int(j.sum())
            rows.append((h0, n, nj, 1000.0 * nj / n))
        rows.sort(key=lambda r: -r[3])
        for h0, n, nj, r in rows[:a.top]:
            print('   h=%3d  px=%7d  junc=%5d  j/1k=%6.2f' % (h0, n, nj, r))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
