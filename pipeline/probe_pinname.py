# -*- coding: utf-8 -*-
"""How much of PinF1 is reachable *without OCR*?

PinF1 compares ``pinname`` strings exactly, so if the ground truth leaves ``pinname`` empty
for most pins then a prediction that also emits an empty string can win points without any
recognition at all.  This probe counts that, plus how the empty/non-empty split correlates
with component type (`box` ICs vs passives), on the 32 cases that have a ground-truth file
in this repository (10GT + 10_GTcase + the 12 sampled 200_train_cases).

Usage:  python -m pipeline.probe_pinname [case_id ...]
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline import config as C
from pipeline import pcore


def main(argv):
    ids = [a for a in argv if not a.startswith('-')]
    cases = [c for c in (pcore.find_case(i) for i in ids) if c] if ids else \
        [c for c in pcore.discover() if c.gt_json]
    print('cases: %d  -> %s' % (len(cases), ', '.join(c.case_id for c in cases)))
    tot = Counter()
    by_type = defaultdict(Counter)
    for c in cases:
        doc = pcore.load_gt(c)
        comps = doc.get('components') or {}
        ctype = {k: (v or {}).get('type') for k, v in comps.items()}
        for ck, grp in (doc.get('pins') or {}).items():
            for pk, info in (grp or {}).items():
                name = (info or {}).get('pinname')
                empty = (name is None or name == '')
                tot['pins'] += 1
                tot['empty'] += int(bool(empty))
                by_type[ctype.get(ck, '?')]['pins'] += 1
                by_type[ctype.get(ck, '?')]['empty'] += int(bool(empty))
    print('all pins=%d  empty pinname=%d (%.3f)' % (tot['pins'], tot['empty'],
                                                    tot['empty'] / max(1, tot['pins'])))
    rows = sorted(by_type.items(), key=lambda kv: -kv[1]['pins'])
    print('%-16s %8s %8s %7s' % ('type', 'pins', 'empty', 'frac'))
    for t, cc in rows[:24]:
        print('%-16s %8d %8d %7.3f' % (t, cc['pins'], cc['empty'], cc['empty'] / max(1, cc['pins'])))
    # how many *components* have at least one text-bearing pair?
    print('empty-name examples:',
          [k for k, v in list((pcore.load_gt(cases[0]).get('pins') or {}).items())[:1]])


if __name__ == '__main__':
    main(sys.argv[1:])
