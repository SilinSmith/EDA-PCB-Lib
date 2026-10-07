# -*- coding: utf-8 -*-
"""Which pin suffix sits on which side of a component?

The pipeline has to invent pin names for the two- and three-pin parts (it cannot read them:
`box` has 9356 pins that are unreadable, and only 41% of all pin names come from a 26-symbol
alphabet).  Inventing them is only worth it if the rule is right, so this probe measures the
rule from ground truth instead of guessing it: for every component with 2 or 3 pins, each pin
is bucketed by which side of the component box its tip sits on, and the distribution of pin
names per (type, side) is reported.

Usage:  python -m pipeline.probe_pinorder
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

from pipeline import config as C
from pipeline import pcore

SIDES = ('L', 'R', 'T', 'B')


def nearest_side(box, x, y) -> str:
    """Which edge of *box* is the point closest to?"""
    x1, y1, x2, y2 = box
    d = {'L': abs(x - x1), 'R': abs(x - x2), 'T': abs(y - y1), 'B': abs(y - y2)}
    return min(d, key=lambda k: d[k])


def main(argv):
    per_type_side = defaultdict(Counter)          # (type, side) -> pinname counter
    per_type_order = defaultdict(Counter)         # type -> counter of ordered name tuples
    per_src_order = defaultdict(Counter)          # (source, type) -> ordered name tuples
    per_src_side = defaultdict(Counter)           # (source, type, side) -> pinname counter
    n_comp = Counter()
    for c in pcore.discover():
        if not c.gt_json:
            continue
        w, h = Image.open(c.png).size
        t = pcore.truth_from_doc(pcore.load_gt(c), w, h)
        by_comp = defaultdict(list)
        for (key, suffix), pin in t.pins.items():
            by_comp[key].append((suffix, pin))
        for key, plist in by_comp.items():
            ctype = t.comps[key].ctype if key in t.comps else '?'
            if len(plist) not in (2, 3):
                continue
            box = t.comps[key].box if key in t.comps else (0, 0, 0, 0)
            n_comp[ctype] += 1
            for suffix, pin in plist:
                side = nearest_side(box, pin.x, pin.y)
                per_type_side[(ctype, side)][suffix] += 1
                per_src_side[(c.source, ctype, side)][suffix] += 1
            # dominant-axis order: left->right, else top->bottom
            xs = [p.x for _s, p in plist]
            ys = [p.y for _s, p in plist]
            if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
                ordered = [s for s, _p in sorted(plist, key=lambda sp: sp[1].x)]
            else:
                ordered = [s for s, _p in sorted(plist, key=lambda sp: sp[1].y)]
            per_type_order[ctype][tuple(ordered)] += 1
            per_src_order[(c.source, ctype)][tuple(ordered)] += 1

    print('component types with 2-3 pins: %s' % dict(n_comp.most_common()))
    print('\n%-14s %-3s %6s  %s' % ('type', 'side', 'n', 'pin names'))
    for (ctype, side), cnt in sorted(per_type_side.items(),
                                     key=lambda kv: (kv[0][0], SIDES.index(kv[0][1]))):
        tot = sum(cnt.values())
        if tot < 4:
            continue
        top = ', '.join('%s:%d(%.2f)' % (k, v, v / tot) for k, v in cnt.most_common(4))
        print('%-14s %-3s %6d  %s' % (ctype, side, tot, top))

    print('\nordered along the dominant axis (earlier name first):')
    for ctype, cnt in sorted(per_type_order.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(cnt.values())
        top = ', '.join('%s:%d(%.2f)' % ('->'.join(k), v, v / tot) for k, v in cnt.most_common(3))
        print('%-14s %6d  %s' % (ctype, tot, top))

    print('\nby data source (the convention is a vendor property, not a symbol one):')
    for (src, ctype), cnt in sorted(per_src_order.items(),
                                    key=lambda kv: -sum(kv[1].values())):
        tot = sum(cnt.values())
        if tot < 6 or ctype not in ('r', 'c', 'l', 'd', 'led', 'fuse', 'crystal_2pin',
                                    'mosfet', 'bjt'):
            continue
        top = ', '.join('%s:%d(%.2f)' % ('->'.join(k), v, v / tot) for k, v in cnt.most_common(2))
        print('%-18s %-12s %5d  %s' % (src, ctype, tot, top))

    os.makedirs(C.OUT_ROOT, exist_ok=True)
    with open(os.path.join(C.OUT_ROOT, 'probe_pinorder.json'), 'w', encoding='utf-8') as f:
        json.dump({'side': {'%s|%s' % k: v for k, v in per_type_side.items()},
                   'order': {'%s' % k: {'->'.join(kk): vv for kk, vv in v.items()}
                             for k, v in per_type_order.items()},
                   'src_order': {'%s|%s' % k: {'->'.join(kk): vv for kk, vv in v.items()}
                                 for k, v in per_src_order.items()},
                   'src_side': {'%s|%s|%s' % k: v for k, v in per_src_side.items()}}, f,
                  ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main([a for a in sys.argv[1:] if not a.startswith('-')])
