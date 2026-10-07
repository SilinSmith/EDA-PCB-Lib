# -*- coding: utf-8 -*-
"""Text detector diagnostics: what do the detected lines actually look like?

Anchored on the vendor reference cases in ``Dataset300`` because only those carry
per-text-box truth.  This script prints, for a handful of vendor text boxes, the ink
bitmap around them (ASCII) next to the line boxes my detector produced nearby, so a low
score can be attributed to binarisation, glyph filtering or line grouping.

Usage:  python -m pipeline.probe_text2 [case_id ...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import List, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore, ptxt
from pipeline.validate_txt import ior, prf

Box = Tuple[float, float, float, float]


def sauvola(gray: np.ndarray, k: float = 0.2, w: int = 25) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (w, w))
    sq = cv2.boxFilter(g * g, cv2.CV_32F, (w, w))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(g > thr, 0, 255).astype(np.uint8)


def ink_variants(bgr, name):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if name == 'adaptive31':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 8)
    if name == 'adaptive15':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 6)
    if name == 'adaptblur':
        b = cv2.medianBlur(gray, 3)
        return cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 7)
    if name == 'otsu':
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if name == 'sauvola':
        return sauvola(gray)
    raise ValueError(name)


def refine(ink, close_k=0, open_k=0, stroke_max=0.0):
    out = ink
    if close_k:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_k + 1,) * 2, np.uint8))
    if open_k:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_k,) * 2, np.uint8))
    if stroke_max > 0:
        dt = cv2.distanceTransform((out > 0).astype(np.uint8), cv2.DIST_L2, 3)
        out = np.where(dt <= stroke_max, out, 0).astype(np.uint8)
    return out


def ref_items(c):
    doc = json.load(open(c.ref_json, encoding='utf-8'))
    out = []
    for tid, t in (doc.get('texts') or {}).items():
        loc = (t or {}).get('location') or {}
        try:
            box = (float(loc['x1']), float(loc['y1']), float(loc['x2']), float(loc['y2']))
        except Exception:
            continue
        out.append((tid, box, str(t.get('content') or ''), C.role_of(t.get('details'))))
    return out


def cc_stats(ink, rel):
    """Histogram of connected components *inside* the vendor text boxes: are glyphs
    connected as whole characters, or fragmented?"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    x1, y1, x2, y2 = [int(round(v)) for v in rel]
    inside = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        cx, cy = x + w / 2.0, y + h / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            inside.append((w, h, a))
    return inside


def main(argv):
    ids = [a for a in argv if not a.startswith('-')] or ['0059']
    for cid in ids:
        c = pcore.find_case(cid)
        if c is None or not c.ref_json:
            print('%s: no reference file' % cid)
            continue
        bgr = ptxt.imread(c.png)
        print('\n===== %s  image=%s =====' % (c.case_id, bgr.shape))
        items = ref_items(c)
        refs = [it[1] for it in items]
        print('vendor text boxes: %d  heights p5/p50/p95 = %s'
              % (len(refs), np.percentile([b[3] - b[1] for b in refs], [5, 50, 95]).round(1).tolist()))
        for ink_name in ('adaptive31', 'adaptblur', 'sauvola', 'otsu'):
            for ck in (0, 2):
                ink = refine(ink_variants(bgr, ink_name), close_k=ck)
                gl = ptxt.glyph_components(bgr, ink=ink, close_k=0)
                lines = ptxt.group_lines(gl)
                tp, fp, fn, p, r, f1 = prf([l.box for l in lines], refs, fn_=ior)
                print('  %-10s ck=%d ink=%.4f glyphs=%5d lines=%5d tp=%4d fp=%4d fn=%4d P=%.3f R=%.3f F1=%.3f gly/line=%.1f'
                      % (ink_name, ck, (ink > 0).mean(), len(gl), len(lines), tp, fp, fn, p, r, f1,
                         np.mean([l.n_glyphs for l in lines]) if lines else 0))
        # how many raw CCs are there, before any filter?
        for ck in (0, 2, 3, 4):
            ink = refine(ink_variants(bgr, 'adaptblur'), close_k=ck)
            n, lab, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
            hs = stats[1:, 3]
            print('    raw CC (adaptblur ck=%d): n=%d  h p50/p90/p99=%s  h in [6,40]=%d  area>=12=%d'
                  % (ck, n - 1, np.percentile(hs, [50, 90, 99]).round(1).tolist(),
                     int(((hs >= 6) & (hs <= 40)).sum()), int((stats[1:, 4] >= 12).sum())))
        # How fragmented is a character?  Look at the CCs inside each vendor box.
        ink = refine(ink_variants(bgr, 'adaptblur'))
        sizes = Counter()
        for tid, box, content, role in items:
            for w, h, a in cc_stats(ink, box):
                sizes[(h, a)] += 1
        print('  CC sizes inside vendor boxes (h,area) top20:', sizes.most_common(20))
        print('  glyph filters reject: min_area=12, min_h=6, h>46')
        # line boxes near the first few vendor boxes
        lines = ptxt.group_lines(ptxt.glyph_components(bgr, ink=ink, close_k=0))
        for tid, box, content, role in items[:5]:
            near = [l for l in lines
                    if not (l.box[2] < box[0] - 30 or l.box[0] > box[2] + 30
                            or l.box[3] < box[1] - 30 or l.box[1] > box[3] + 30)]
            print('  %s %r %s ref=%s  -> %d lines: %s'
                  % (tid, content, role, [round(v) for v in box], len(near),
                     [[round(v) for v in l.box] + [l.n_glyphs] for l in near[:4]]))
        for tid, box, content, role in items[:3]:
            print('  --- ASCII %s %r %s box=%s' % (tid, content, role, [round(v, 1) for v in box]))
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, x1 - 6); y1 = max(0, y1 - 6)
            x2 = min(ink.shape[1], x2 + 6); y2 = min(ink.shape[0], y2 + 6)
            for row in ink[y1:y2, x1:x2]:
                print('      ' + ''.join('#' if v else '.' for v in row))


if __name__ == '__main__':
    main(sys.argv[1:])
