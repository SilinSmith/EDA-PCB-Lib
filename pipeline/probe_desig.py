# -*- coding: utf-8 -*-
"""Where does a designator sit relative to its component bbox?

The end-to-end plan is "search for text only where text can be": the component boxes (YOLO
or ground truth) pin down where a designator may be, which removes the need for an
image-wide text detector.  This probe quantifies the search window using the vendor's own
OCR pointers (``components[C].searchNameByOCR`` -> ``texts[T].location``), which is the only
reliable attribution -- matching text *content* against the component key is wrong, because
one string such as ``R1`` occurs in many unrelated boxes and the nearest such box is often
hundreds of pixels away.

Usage:  python -m pipeline.probe_desig
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pipeline import config as C
from pipeline import pcore, ptxt


def pointers(case):
    """``['components'][C] -> {'searchName': T_n, 'value': T_n, ...}`` from the vendor file."""
    doc = json.load(open(case.ref_json, encoding='utf-8'))
    texts = {}
    for tid, t in (doc.get('texts') or {}).items():
        loc = (t or {}).get('location') or {}
        try:
            texts[tid] = (float(loc['x1']), float(loc['y1']),
                          float(loc['x2']), float(loc['y2']),
                          str((t or {}).get('content') or ''))
        except Exception:
            pass
    out = {}
    for ck, c in (doc.get('components') or {}).items():
        ptr = {}
        for field, tgt in (('searchNameByOCR', 'searchName'),
                           ('valueByOCR', 'value'),
                           ('nameByOCR', 'name')):
            tid = (c or {}).get(field)
            if tid in texts:
                ptr[tgt] = texts[tid]
        if ptr:
            out[ck] = ptr
    return out, texts


def main(argv):
    cases = [c for c in pcore.discover() if c.gt_json and c.ref_json]
    dxs, dys, gaps = [], [], []
    sides = Counter()
    n_ptr = 0
    n_missing = 0
    p90_gap = []
    for c in cases:
        gt = pcore.load_gt(c)
        comps = gt.get('components') or {}
        ptr, _ = pointers(c)
        for ck, comp in comps.items():
            box = (comp or {}).get('bbox')
            if not box:
                continue
            p = ptr.get(ck) or {}
            if not p:
                n_missing += 1
                continue
            n_ptr += 1
            for tgt, b in p.items():
                if tgt != 'searchName':
                    continue
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                w = max(1.0, box[2] - box[0]); h = max(1.0, box[3] - box[1])
                tcx, tcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                dxs.append((tcx - cx) / w)
                dys.append((tcy - cy) / h)
                sx = 'L' if tcx < box[0] else ('R' if tcx > box[2] else '-')
                sy = 'T' if tcy < box[1] else ('B' if tcy > box[3] else '-')
                sides[sx + sy] += 1
                gx = max(box[0] - b[2], b[0] - box[2], 0.0)
                gy = max(box[1] - b[3], b[1] - box[3], 0.0)
                gaps.append(max(gx, gy))
        # search-window scan: fraction of designators inside a box grown by k px
        for k in (12, 20, 30, 40, 60):
            hit = 0
            tot = 0
            for ck, comp in comps.items():
                box = (comp or {}).get('bbox')
                p = (ptr.get(ck) or {}).get('searchName')
                if not box or p is None:
                    continue
                tot += 1
                tcx, tcy = (p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0
                if (box[0] - k) <= tcx <= (box[2] + k) and (box[1] - k) <= tcy <= (box[3] + k):
                    hit += 1
            if tot:
                p90_gap.append((k, hit / tot))
    print('cases=%d  comps with designator pointer=%d  without=%d' % (len(cases), n_ptr, n_missing))
    print('side counts:', sides.most_common())
    print('dx/w p5/p25/p50/p75/p95 = %s' % np.percentile(dxs, [5, 25, 50, 75, 95]).round(2).tolist())
    print('dy/h p5/p25/p50/p75/p95 = %s' % np.percentile(dys, [5, 25, 50, 75, 95]).round(2).tolist())
    print('designator-centre gap to bbox px p50/p75/p90/p95/p99 = %s max=%.0f'
          % (np.percentile(gaps, [50, 75, 90, 95, 99]).round(1).tolist(), max(gaps)))
    print('inside bbox+kpx:', [(k, round(v, 3)) for k, v in p90_gap])


if __name__ == '__main__':
    main(sys.argv[1:])
