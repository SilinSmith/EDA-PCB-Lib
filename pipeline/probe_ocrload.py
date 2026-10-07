# -*- coding: utf-8 -*-
"""How many characters of OCR is ComponentF1 worth?

``component_text_ok`` compares only ``Name`` for IC types (``box``) and only ``value`` for
R/C; every other type passes on text unconditionally.  So the text burden per case is the
set of IC names, and this probe measures its size (characters, distinct strings) on the
cases that have ground truth, to judge whether a template-matching character reader is
enough or a real OCR engine is needed.

Usage:  python -m pipeline.probe_ocrload
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schematic_parser.data import IC_TYPES, VALUE_TYPES
from pipeline import pcore


def main(argv):
    cases = [c for c in pcore.discover() if c.gt_json]
    names = Counter()
    values = Counter()
    per_case_chars = []
    n_ic = 0
    for c in cases:
        doc = pcore.load_gt(c)
        chars = 0
        for ck, comp in (doc.get('components') or {}).items():
            t = (comp or {}).get('type')
            if t in IC_TYPES:
                n_ic += 1
                names[(comp or {}).get('Name')] += 1
                chars += len((comp or {}).get('Name') or '')
            elif t in VALUE_TYPES:
                values[(comp or {}).get('value')] += 1
                chars += len((comp or {}).get('value') or '')
        per_case_chars.append(chars)
    print('cases=%d  IC components=%d' % (len(cases), n_ic))
    print('distinct IC Name strings=%d  top15=%s' % (len(names), names.most_common(15)))
    print('Name length: p50/p90/p95/max = %s / %d'
          % (__import__('numpy').percentile([len(k or '') for k in names],
                                            [50, 90, 95]).round(1).tolist(),
             max(len(k or '') for k in names)))
    print('distinct R/C value strings=%d  top10=%s' % (len(values), values.most_common(10)))
    import numpy as np
    print('chars to read per case (IC Name + R/C value): p50/p90/p99/max = %s / %d'
          % (np.percentile(per_case_chars, [50, 90, 99]).round(1).tolist(),
             max(per_case_chars)))
    # how many IC names are pure digits (i.e. easily guessed)?
    import re
    dig = sum(v for k, v in names.items() if re.fullmatch(r'[0-9.\-]+', k or ''))
    print('IC names that are pure numbers: %d / %d' % (dig, n_ic))


if __name__ == '__main__':
    main(sys.argv[1:])
