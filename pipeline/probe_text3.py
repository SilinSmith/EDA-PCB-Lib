# -*- coding: utf-8 -*-
"""Is the text failure a *binarisation* problem or a *grouping* problem?

Prints, for every vendor text box of one case, the ink fraction inside it and the fraction
of the box covered by the largest connected component that touches it -- i.e. whether the
box holds a whole connected glyph cluster at all.  Then scores the same case with IoR at a
low threshold (0.1), which separates "found nothing there" from "found something slightly
smaller/offset".

Usage:  python -m pipeline.probe_text3 <case_id> [...]
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2

from pipeline import config as C
from pipeline import pcore, ptxt
from pipeline.probe_text2 import ink_variants, refine, ref_items
from pipeline.validate_txt import ior, prf

GRID = ['adaptblur', 'adaptive31', 'otsu']


def main(argv):
    ids = [a for a in argv if not a.startswith('-')] or ['0059']
    for cid in ids:
        c = pcore.find_case(cid)
        if c is None or not c.ref_json:
            print('%s: no reference' % cid)
            continue
        bgr = ptxt.imread(c.png)
        items = ref_items(c)
        refs = [it[1] for it in items]
        print('\n===== %s %s =====' % (c.case_id, bgr.shape))
        for name in GRID:
            ink = refine(ink_variants(bgr, name), close_k=0)
            n, lab, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
            rows = []
            for tid, box, content, role in items:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                sub = ink[y1:y2, x1:x2]
                fa = float((sub > 0).mean())
                # largest CC whose pixels fall inside the box
                ids_in = np.unique(lab[y1:y2, x1:x2])
                ids_in = ids_in[ids_in > 0]
                # fraction of the box covered by the biggest CC that *mostly* sits in it
                best = 0.0
                for i in ids_in:
                    xs, ys, ws, hs, as_ = stats[i]
                    inter = max(0, min(x2, xs + ws) - max(x1, xs)) * max(0, min(y2, ys + hs) - max(y1, ys))
                    ba = max(1.0, ws * hs)
                    if inter / ba < 0.5:      # mostly outside -> ignore
                        continue
                    boxa = max(1.0, (x2 - x1) * (y2 - y1))
                    best = max(best, inter / boxa)
                rows.append((tid, role, content, fa, best))
            far = np.mean([r[3] for r in rows])
            covr = np.mean([r[4] for r in rows])
            scores = {}
            for thr in (0.1, 0.3, 0.5):
                gl = ptxt.glyph_components(bgr, ink=ink, close_k=0)
                lines = ptxt.group_lines(gl)
                scores[thr] = prf([l.box for l in lines], refs, thr=thr, fn_=ior)[:3]
            print('  %-10s ink=%.4f | mean ink-in-box=%.3f mean largest-CC-cover=%.3f'
                  % (name, (ink > 0).mean(), far, covr))
            for thr in (0.1, 0.3, 0.5):
                tp, fp, fn = scores[thr]
                p = tp / (tp + fp) if tp + fp else 0
                r = tp / (tp + fn) if tp + fn else 0
                print('      IoR@%.1f tp=%3d fp=%3d fn=%3d P=%.3f R=%.3f F1=%.3f'
                      % (thr, tp, fp, fn, p, r, 2 * p * r / (p + r) if p + r else 0))
            for r_ in rows[:8]:
                print('        %-6s %-11s %-8r ink=%.2f cover=%.2f' % r_)


if __name__ == '__main__':
    main(sys.argv[1:])
