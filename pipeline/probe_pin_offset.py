# -*- coding: utf-8 -*-
"""Where are the pin tips relative to the component bbox?  Measured inside the vendor's own
reference frame, where bbox (``components[C].location``) and pin tips
(``components[C].pins[P].tip``) are guaranteed to share one coordinate system.

Usage:  python -m pipeline.probe_pin_offset
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pipeline import config as C


def main():
    offs = []
    per_edge = defaultdict(list)
    n_comp = 0
    n_pin = 0
    wbox: List[float] = []
    hbox: List[float] = []
    for path in sorted(glob.glob(os.path.join(C.REF_ROOT, '*', '*CVJsonStd.json'))):
        try:
            doc = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        comps = doc.get('components') or {}
        for ck, c in comps.items():
            loc = (c or {}).get('location') or {}
            try:
                x1, y1, x2, y2 = (float(loc['x1']), float(loc['y1']),
                                  float(loc['x2']), float(loc['y2']))
            except Exception:
                continue
            n_comp += 1
            wbox.append(x2 - x1)
            hbox.append(y2 - y1)
            for pk, p in ((c or {}).get('pins') or {}).items():
                tip = (p or {}).get('tip')
                if not tip or len(tip) < 2:
                    continue
                x, y = float(tip[0]), float(tip[1])
                n_pin += 1
                d = {'L': x1 - x, 'R': x - x2, 'T': y1 - y, 'B': y - y2}
                e = min(d, key=lambda k: abs(d[k]))
                offs.append(d[e])
                per_edge[e].append(d[e])
    a = np.array(offs)
    print('reference frame: %d components, %d pin tips' % (n_comp, n_pin))
    print('bbox size: w median %.0f  h median %.0f' % (np.median(wbox), np.median(hbox)))
    print('offset to nearest bbox edge: median %.2f' % np.median(a))
    print('  <0 (tip inside body): %d (%.1f%%)' % ((a < 0).sum(), 100 * (a < 0).mean()))
    print('  |offset| <= 5      : %d (%.1f%%)' % ((np.abs(a) <= 5).sum(), 100 * (np.abs(a) <= 5).mean()))
    print('  |offset| <= 10     : %d (%.1f%%)' % ((np.abs(a) <= 10).sum(), 100 * (np.abs(a) <= 10).mean()))
    print('  |offset| <= 15     : %d (%.1f%%)' % ((np.abs(a) <= 15).sum(), 100 * (np.abs(a) <= 15).mean()))
    print('  |offset| <= 20     : %d (%.1f%%)' % ((np.abs(a) <= 20).sum(), 100 * (np.abs(a) <= 20).mean()))
    print('  |offset| >  20     : %d (%.1f%%)' % ((np.abs(a) > 20).sum(), 100 * (np.abs(a) > 20).mean()))
    for e in 'LRTB':
        v = np.array(per_edge.get(e, []))
        if len(v):
            print('  edge %s n=%5d median %6.2f  p10 %6.2f p90 %6.2f' % (
                e, len(v), np.median(v), np.percentile(v, 10), np.percentile(v, 90)))
    # how many pins does each component have
    cnt = Counter()
    for path in sorted(glob.glob(os.path.join(C.REF_ROOT, '*', '*CVJsonStd.json'))):
        try:
            doc = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        for ck, c in (doc.get('components') or {}).items():
            cnt[len(((c or {}).get('pins') or {}))] += 1
    print('pins per component: %s' % dict(sorted(cnt.items())[:12]))


if __name__ == '__main__':
    main()
