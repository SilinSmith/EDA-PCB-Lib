# -*- coding: utf-8 -*-
"""Probe: what does a wire pixel look like in each source, and how wide is it?

Run:  $env:PYTHONIOENCODING='utf-8'; K:\python\311\python.exe -m pipeline.probe_ink
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding='utf-8')

import collections
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image

from .config import DATA_ROOT, REF_ROOT
from .pcore import discover, find_case, truth_from_doc, load_gt


def ref_skeleton_mask(ref_path, h, w):
    """Rasterise the reference pipeline's own wire skeleton (Dataset300 CVJsonStd)."""
    d = json.load(open(ref_path, encoding='utf-8'))
    m = np.zeros((h, w), np.uint8)
    xs, ys = [], []
    n = 0
    for e in d.get('edges', {}).values():
        for x, y in e.get('keyPoints', {}).get('skeleton', {}).get('wire', []) or []:
            if 0 <= x < w and 0 <= y < h:
                m[y, x] = 255
                xs.append(x); ys.append(y); n += 1
    return m, d


def main():
    # ---- 1. probe the reference cases: colour of wire pixels + stroke width
    print('=' * 96)
    print('A. reference-skeleton pixel colours / stroke width per nowCls')
    print('=' * 96)
    per_cls = collections.defaultdict(lambda: {'colours': collections.Counter(),
                                               'widths': collections.Counter(),
                                               'px': 0})
    for path in sorted(glob.glob(os.path.join(REF_ROOT, '*', '*CVJsonStd.json'))):
        cid = os.path.basename(os.path.dirname(path))
        png = glob.glob(os.path.join(os.path.dirname(path), '*.png'))
        if not png:
            continue
        img = np.array(Image.open(png[0]).convert('RGB'))
        h, w = img.shape[:2]
        d = json.load(open(path, encoding='utf-8'))
        cls = d.get('nowCls')
        if d['canvas']['height'] != h or d['canvas']['width'] != w:
            print('  !! canvas mismatch', cid, d['canvas'], (w, h))
        skel, _ = ref_skeleton_mask(path, h, w)
        ys, xs = np.nonzero(skel)
        if len(xs) == 0:
            continue
        cols = img[ys, xs]
        for c in map(tuple, cols[::7]):
            per_cls[cls]['colours'][c] += 1
        # stroke width: distance transform of the darkest mask
        g = np.array(Image.open(png[0]).convert('L'))
        fg = (g < 0.90 * np.median(g) + 8).astype(np.uint8)
        dt = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
        wid = dt[ys, xs] * 2.0
        for v in wid[::7]:
            per_cls[cls]['widths'][round(float(v) * 2) / 2] += 1
        per_cls[cls]['px'] += len(xs)
        top = collections.Counter(map(tuple, cols)).most_common(2)
        print('  %-8s %-6s %-22s skelpx=%6d top-colour=%s' % (cid, cls, png[0][-40:], len(xs), top))

    for cls, agg in per_cls.items():
        print('\n-- nowCls=%s  skelpx=%d' % (cls, agg['px']))
        print('   top colours on skeleton:', agg['colours'].most_common(6))
        ws = sorted(agg['widths'].items())
        print('   stroke width (2*EDT) hist:', ws[:10])


if __name__ == '__main__':
    main()
