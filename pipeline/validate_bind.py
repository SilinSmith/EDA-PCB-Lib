# -*- coding: utf-8 -*-
"""Text detector + role binder, scored on the real corpus.

There is no per-line text annotation in the competition ground truth, but the vendor
reference file (``<id>-<source>_CVJsonStd.json``) has one for the 32 cases in
``Dataset300``: ``texts[T_n] = {content, details, location}`` where ``details`` is the
semantic role.  Its bounding boxes are absolute pixels in the same frame as the image, so
this script measures

* detection: recall / precision of my text lines against the vendor text boxes (IoR 0.5);
* binding : for each vendor box that carries a role, work out which component bbox it
  belongs to *from its geometry alone* and compare with the box the reference pipeline
  attached it to.

Usage:  python -m pipeline.validate_bind [--n=8] [case_id ...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore, ptxt
from pipeline.validate_txt import ior, iou, prf

Box = Tuple[float, float, float, float]


# ------------------------------------------------------------------ reference association


def ref_comp_boxes(case) -> Dict[str, Box]:
    """Component bboxes from the vendor reference file (absolute image pixels)."""
    try:
        doc = json.load(open(case.ref_json, encoding='utf-8'))
    except Exception:
        return {}
    out = {}
    for ck, c in (doc.get('components') or {}).items():
        loc = (c or {}).get('location') or {}
        try:
            out[ck] = (float(loc['x1']), float(loc['y1']), float(loc['x2']), float(loc['y2']))
        except Exception:
            continue
    return out


def ref_pin_assoc(case) -> Dict[str, Tuple[str, str]]:
    """``T_n -> (component, pin)`` for pin-number / pin-name texts (vendor's own pointer)."""
    try:
        doc = json.load(open(case.ref_json, encoding='utf-8'))
    except Exception:
        return {}
    out = {}
    for ck, c in (doc.get('components') or {}).items():
        for pk, p in ((c or {}).get('pins') or {}).items():
            for fld, role in (('pinNumberByOCR', 'pinNumber'), ('pinNameByOCR', 'pinName')):
                tid = (p or {}).get(fld)
                if tid:
                    out[tid] = (ck, pk)
    return out


def ref_comp_text_assoc(case) -> Dict[str, str]:
    """``T_n -> component`` for designator / value texts (vendor's own pointer)."""
    try:
        doc = json.load(open(case.ref_json, encoding='utf-8'))
    except Exception:
        return {}
    out = {}
    for ck, c in (doc.get('components') or {}).items():
        for fld in ('searchNameByOCR', 'searchNameByVLM', 'searchNameByLib'):
            tid = (c or {}).get(fld)
            if tid:
                out.setdefault(tid, ck)
    return out


# ------------------------------------------------------------------ binder

def ref_text_items(case) -> List[Tuple[str, Box, str, str]]:
    """``(tid, box, content, role)`` for every vendor text box (absolute image pixels)."""
    try:
        doc = json.load(open(case.ref_json, encoding='utf-8'))
    except Exception:
        return []
    out = []
    for tid, t in (doc.get('texts') or {}).items():
        loc = (t or {}).get('location') or {}
        try:
            box = (float(loc['x1']), float(loc['y1']), float(loc['x2']), float(loc['y2']))
        except Exception:
            continue
        out.append((tid, box, str((t or {}).get('content') or ''),
                    C.role_of((t or {}).get('details'))))
    return out


def assign_component(box: Box, boxes: Dict[str, Box], max_dist: float = 90.0
                     ) -> Optional[str]:
    """Assign a text box to a component: nearest component centre distance."""
    best, bd = None, None
    for ck, cb in boxes.items():
        d = pcore.point_to_bbox_dist(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), cb)
        if d > max_dist:
            continue
        if bd is None or d < bd:
            bd, best = d, ck
    return best


def main(argv):
    n = 6
    for a in argv:
        if a.startswith('--n='):
            n = int(a.split('=')[1])
    ids = [a for a in argv if not a.startswith('-')]
    if ids:
        cases = [c for c in (pcore.find_case(i) for i in ids) if c and c.ref_json]
    else:
        cases = [c for c in pcore.discover() if c.split == 'Dataset300' and c.ref_json][:n]
    print('cases: %s' % ', '.join(c.case_id for c in cases))

    det_tot = [0, 0, 0]
    bind_stat = Counter()
    bind_by_role: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for c in cases:
        bgr = ptxt.imread(c.png)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 8)
        lines = ptxt.detect(bgr, ink=ink, close_k=0)
        items = ref_text_items(c)
        rb = [it[1] for it in items]
        pb = [l.box for l in lines]
        tp, fp, fn, p, r, f1 = prf(pb, rb, fn_=ior)
        det_tot[0] += tp; det_tot[1] += fp; det_tot[2] += fn
        cboxes = ref_comp_boxes(c)
        comp_of = ref_comp_text_assoc(c)          # T_n -> component
        pin_of = ref_pin_assoc(c)                 # T_n -> (component, pin)
        for (tid, box, content, role) in items:
            if role in ('note', '') or tid not in comp_of and tid not in pin_of:
                continue
            want = comp_of.get(tid) or pin_of.get(tid, (None, None))[0]
            got = assign_component(box, cboxes)
            key = 'ok' if got == want else ('none' if got is None else 'wrong')
            bind_stat[(role, key)] += 1
            slot = bind_by_role[role]
            slot[0] += 1
            slot[1] += 1 if key == 'ok' else 0
            slot[2] += 1 if key == 'wrong' else 0
        print('  %-22s ref=%3d lines=%3d det P=%.3f R=%.3f F1=%.3f'
              % (c.case_id, len(rb), len(pb), p, r, f1))
    tp, fp, fn = det_tot
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    print('DETECTION TOTAL tp=%d fp=%d fn=%d P=%.3f R=%.3f F1=%.3f'
          % (tp, fp, fn, p, r, 2 * p * r / (p + r) if p + r else 0.0))
    print('\nBINDING (nearest-component rule, vendor association as truth)')
    print('  %-12s %6s %6s %6s %6s' % ('role', 'n', 'ok', 'wrong', 'miss'))
    for role in sorted(bind_by_role):
        tot, ok, wrong = bind_by_role[role]
        miss = bind_stat[(role, 'none')]
        print('  %-12s %6d %6d %6d %6d  acc=%.3f'
              % (role, tot, ok, wrong, miss, ok / tot if tot else 0.0))


if __name__ == '__main__':
    main(sys.argv[1:])
