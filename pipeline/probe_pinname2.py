# -*- coding: utf-8 -*-
"""What *are* the pinname strings?  How much of PinF1 is guessable without OCR?

PinF1 requires the ``pinname`` string to match exactly; the previous probe shows all 22691
ground-truth pins carry a non-empty name, so no points are available for free.  This probe
splits the names into (a) tiny alphabets (``1``/``2``/``A``/``K``/``G``/``S``/``D`` -- a
passive or 3-pin device can be guessed from geometry alone) and (b) the rest, and reports
the per-type ceiling for the "guess from geometry" strategy.

Usage:  python -m pipeline.probe_pinname2
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline import pcore

TINY = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'K', 'E', 'G',
        'S', 'D', 'V', 'a', 'b', 'c', 'k', 'e', 'g', 's', 'd', '+'}


def main(argv):
    cases = [c for c in pcore.discover() if c.gt_json]
    per_type = defaultdict(Counter)
    per_type_n = Counter()
    allc = Counter()
    for c in cases:
        doc = pcore.load_gt(c)
        ctype = {k: (v or {}).get('type') for k, v in (doc.get('components') or {}).items()}
        for ck, grp in (doc.get('pins') or {}).items():
            t = ctype.get(ck, '?')
            for pk, info in (grp or {}).items():
                name = (info or {}).get('pinname') or ''
                per_type[t][name] += 1
                per_type_n[t] += 1
                allc[name] += 1
    tot = sum(allc.values())
    tiny = sum(v for k, v in allc.items() if k in TINY)
    print('pins=%d  distinct pinnames=%d' % (tot, len(allc)))
    print('names in TINY set=%d (%.3f)' % (tiny, tiny / tot))
    print('top 30 pinnames:', allc.most_common(30))
    print()
    print('%-16s %7s %7s %7s  top names' % ('type', 'pins', 'tiny', 'frac'))
    for t, _ in per_type_n.most_common(20):
        cc = per_type[t]
        n = per_type_n[t]
        ti = sum(v for k, v in cc.items() if k in TINY)
        print('%-16s %7d %7d %7.3f  %s' % (t, n, ti, ti / n, cc.most_common(6)))


if __name__ == '__main__':
    main(sys.argv[1:])
