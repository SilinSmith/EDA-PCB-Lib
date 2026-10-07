# -*- coding: utf-8 -*-
"""Text detector ablation on the *labelled* synthetic corpus.

``合成原理图demo/texts.jsonl`` gives, for each of the 12 synthetic sheets, every text
line's box, string, semantic role and owning component.  That is a far better validation
set than the real data (which has no per-line boxes) and it covers all four vendor styles.

Reported per style and overall:

* recall / precision / F1 of the text boxes at IoU 0.5;
* how the numbers move with the ink mode (adaptive / otsu / bgdiff), the glyph closing
  kernel and the line-grouping tolerance.

Usage:  python -m pipeline.validate_txt [--sweep]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore, ptxt, pskel

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ior(a: Box, b: Box) -> float:
    """Intersection over reference: forgiving to a detector that merges or overshoots,
    which is the right notion for *binding* (the binder only needs the text located)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    ra = (b[2] - b[0]) * (b[3] - b[1])
    return (ix2 - ix1) * (iy2 - iy1) / ra if ra > 0 else 0.0


def prf(pred: List[Box], gt: List[Box], thr: float = 0.5, fn_=iou):
    used = [False] * len(gt)
    tp = 0
    for p in pred:
        best, bv = -1, thr
        for i, g in enumerate(gt):
            if used[i]:
                continue
            v = fn_(p, g)
            if v >= bv:
                bv, best = v, i
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(pred) - tp
    fn = used.count(False)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, p, r, (2 * p * r / (p + r) if p + r else 0.0)


def synth_cases() -> List[Tuple[str, str, str]]:
    """(case, png_path, style) for every synthetic sheet that has labels."""
    cases = ptxt.synth_texts()
    out = []
    img_root = os.path.join(C.SYNTH_ROOT, 'images')
    if not os.path.isdir(img_root):
        return out
    for name in sorted(os.listdir(img_root)):
        if not name.lower().endswith('.png'):
            continue
        stem = os.path.splitext(name)[0]
        if stem in cases:
            style = '?'
            try:
                doc = json.load(open(os.path.join(C.SYNTH_ROOT, 'gt', stem + '.json'),
                                     encoding='utf-8'))
                style = str(doc.get('style') or doc.get('palette') or '?')
            except Exception:
                pass
            out.append((stem, os.path.join(img_root, name), style))
    return out


def run_one(png: str, labels: List[dict], **kw):
    bgr = ptxt.imread(png)
    if bgr is None:
        return None
    H = float(bgr.shape[0])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if kw.pop('ink_mode', 'adaptive') == 'otsu':
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 8)
    kw.pop('use_otsu', None)
    lines = ptxt.detect(bgr, ink=ink, **kw)
    # texts.jsonl boxes are already in image coordinates (measured: 2478/2478 boxes
    # contain ink as-is, only 381/2478 after a y flip)
    gt = [(tuple(float(v) for v in t['bbox']), t.get('content'), t.get('role'))
          for t in labels if t.get('bbox')]
    return lines, gt


def main(argv):
    sweep = '--sweep' in argv
    cases = synth_cases()
    if not cases:
        print('no synthetic labelled corpus found at %s' % C.SYNTH_ROOT)
        return
    variants = {
        'adaptive close2': dict(ink_mode='adaptive', close_k=2),
        'adaptive close0': dict(ink_mode='adaptive', close_k=0),
        'adaptive close3': dict(ink_mode='adaptive', close_k=3),
        'otsu close2': dict(ink_mode='otsu', close_k=2),
        'adaptive close2 tight': dict(ink_mode='adaptive', close_k=2, gap_ratio=1.0),
        'adaptive close2 loose': dict(ink_mode='adaptive', close_k=2, gap_ratio=1.9),
        'adaptive close2 ytol0.35': dict(ink_mode='adaptive', close_k=2, y_tol_ratio=0.35),
        'adaptive close2 maxh26': dict(ink_mode='adaptive', close_k=2, max_h=26.0),
        'adaptive close2 maxar1.8': dict(ink_mode='adaptive', close_k=2, max_ar=1.8),
    }
    if not sweep:
        variants = {'adaptive close2': variants['adaptive close2']}
    print('synthetic cases: %s' % ', '.join(c[0] for c in cases))
    for vname, kw in variants.items():
        by_style: Dict[str, List] = defaultdict(lambda: [0, 0, 0])
        tot = [0, 0, 0]
        tot_iou = [0, 0, 0]
        print('\n== %s ==' % vname)
        for stem, png, style in cases:
            r = run_one(png, ptxt.synth_texts().get(stem, []), **dict(kw))
            if r is None:
                continue
            lines, gt = r
            gtb = [g[0] for g in gt]
            pb = [l.box for l in lines]
            tp, fp, fn, p, rec, f1 = prf(pb, gtb, fn_=ior)
            tp2, fp2, fn2, p2, r2, f12 = prf(pb, gtb, fn_=iou)
            tot[0] += tp; tot[1] += fp; tot[2] += fn
            tot_iou[0] += tp2; tot_iou[1] += fp2; tot_iou[2] += fn2
            by_style[style][0] += tp; by_style[style][1] += fp; by_style[style][2] += fn
            print('  %-12s %-10s gt=%4d pred=%4d tp=%4d P=%.3f R=%.3f F1=%.3f  | iou50 P=%.3f R=%.3f F1=%.3f'
                  % (stem, style, len(gtb), len(pb), tp, p, rec, f1, p2, r2, f12))
        tp, fp, fn = tot
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print('  IOR50 TOTAL tp=%d fp=%d fn=%d P=%.3f R=%.3f F1=%.3f'
              % (tp, fp, fn, p, r, 2 * p * r / (p + r) if p + r else 0.0))
        tp, fp, fn = tot_iou
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print('  IOU50 TOTAL tp=%d fp=%d fn=%d P=%.3f R=%.3f F1=%.3f'
              % (tp, fp, fn, p, r, 2 * p * r / (p + r) if p + r else 0.0))
        for style, (tp, fp, fn) in sorted(by_style.items()):
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            print('    style %-10s tp=%4d fp=%4d fn=%4d P=%.3f R=%.3f F1=%.3f'
                  % (style, tp, fp, fn, p, r, 2 * p * r / (p + r) if p + r else 0.0))


if __name__ == '__main__':
    main(sys.argv[1:])
