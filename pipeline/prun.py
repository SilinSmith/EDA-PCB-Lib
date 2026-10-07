# -*- coding: utf-8 -*-
"""End-to-end prototype: image -> competition JSON, plus the ablation ladder.

The whole point of this module is that every stage can be swapped for its ground-truth
oracle, so the score of each stage can be attributed.  The ladder (see
``docs/算法路线设计.md``) is:

=====================  ====================================================================
variant                what is oracled
=====================  ====================================================================
``gt-all``             component keys, types, boxes **and** pin tips + pin names + nets
``gt-box+pinrules``    component keys/types/boxes and *wire* geometry; pin names invented
``gt-key+skel``        component keys/types/boxes from GT; pins + nets from the image
``read-key+skel``      component keys read off the image; pins + nets from the image
``read-key+skel-otsu`` same, with Otsu instead of adaptive thresholding for the wire mask
``read-key+skel-nonet``same, but nets dropped (isolates the net stage)
``yolo-key+skel``      boxes from YOLO11m instead of GT (the honest end-to-end number)
=====================  ====================================================================

Everything is scored with ``schematic_parser.metrics.evaluate_case`` -- the same code that
scores the competition submission, weights included -- so the numbers are comparable.

Usage:
    python -m pipeline.prun --n=8                 # 8 sampled real cases
    python -m pipeline.prun 0059 0095 0112 ...    # explicit case ids
    python -m pipeline.prun --n=8 --yolo          # add the YOLO-box variant
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pipeline import config as C
from pipeline import pcore, ppins, pread, pskel, ptxt
from schematic_parser import metrics as M

Box = Tuple[float, float, float, float]

#: ``R1``, ``C87``, ``U10``, ``LED3``, ``SW1``, ``Q1``, ``JP2`` ... but not ``100nF``.
DESIG_RE = re.compile(r'^[A-Za-z]{1,4}\d{1,4}$')
#: ``100nF``, ``10k``, ``4.7uF``, ``1M``, ``0Ω`` ...
VALUE_RE = re.compile(r'^\d+(?:\.\d+)?\s*(?:[pnumkKM]?[FRH]|\u03a9|R|ohm)?$')
SNAPPY = '1A2K3BCDVGEStpbcdkegs+'


# ------------------------------------------------------------------ case inputs


@dataclass
class Inputs:
    case: object
    h: int
    w: int
    bgr: np.ndarray
    ink: np.ndarray
    truth: pcore.Truth
    gt_doc: dict
    skel: Optional[pskel.SkeletonResult] = None


_INK_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
_INPUT_CACHE: Dict[str, Inputs] = {}


def ink_of(bgr: np.ndarray, mode: str) -> np.ndarray:
    """Wire/text ink for one sheet.  ``adaptive`` is the default measured best recipe."""
    if mode == 'otsu':
        return pskel.ink_mask(bgr, mode='otsu')
    if mode == 'bgdiff':
        return pskel.ink_mask(bgr, mode='bgdiff', diff_tol=26)
    return pskel.ink_mask(bgr, mode='adaptive', k=31, c=8)


def inputs_of(case, ink_mode: str = 'adaptive') -> Inputs:
    if case.case_id in _INPUT_CACHE:
        inp = _INPUT_CACHE[case.case_id]
        if ink_mode == 'adaptive':
            return inp
        return replace(inp, ink=_INK_CACHE.setdefault((case.case_id, ink_mode),
                                                      ink_of(inp.bgr, ink_mode)))
    bgr = ptxt.imread(case.png)
    h, w = bgr.shape[:2]
    gt_doc = pcore.load_gt(case)
    inp = Inputs(case=case, h=h, w=w, bgr=bgr, ink=ink_of(bgr, 'adaptive'),
                 truth=pcore.truth_from_doc(gt_doc, w, h), gt_doc=gt_doc)
    _INPUT_CACHE[case.case_id] = inp
    if ink_mode != 'adaptive':
        inp.ink = _INK_CACHE.setdefault((case.case_id, ink_mode), ink_of(bgr, ink_mode))
    return inp


def skeleton_of(inp: Inputs, ink_mode: str) -> pskel.SkeletonResult:
    """Wire skeleton for one sheet.

    Recipe fixed by ``validate_skel`` (see the design doc): the colour gate is a net loss
    (skelR@2 0.466 -> 0.984 when it is dropped), the 0/45/90 snap hurts geometric coverage
    (segR@2 0.784 -> 0.875 without it), and ``close_ink`` wins where it matters here
    (segR@2 0.800 with, 0.740 without).
    """
    key = (inp.case.case_id, ink_mode)
    if key in _SKEL_CACHE:
        return _SKEL_CACHE[key]
    ex = pskel.WireExtractor(use_colour=False, close_ink=True, snap=False, ink_mode=ink_mode)
    res = ex.extract(inp.bgr)
    _SKEL_CACHE[key] = res
    return res


_SKEL_CACHE: Dict[Tuple[str, str], pskel.SkeletonResult] = {}


# ------------------------------------------------------------------ components


def gt_components(inp: Inputs) -> List[pcore.Comp]:
    return [pcore.Comp(key=c.key, ctype=c.ctype, box=c.box, name=c.name, value=c.value)
            for c in inp.truth.comps.values()]


def yolo_components(inp: Inputs, conf: float = 0.25) -> List[pcore.Comp]:
    """Detected boxes + class names from YOLO11m.  Keys are placeholders (the detector
    cannot read a designator) and are filled by the text stage like any other box."""
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(C.YOLO_BEST)
    r = _YOLO.predict(inp.bgr, conf=conf, imgsz=1024, verbose=False)[0]
    out = []
    names = r.names
    for i, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.cpu().numpy()):
        cls = int(r.boxes.cls[i].item())
        out.append(pcore.Comp(key='?%d' % i, ctype=str(names[cls]),
                              box=(float(x1), float(y1), float(x2), float(y2)),
                              score=float(r.boxes.conf[i].item())))
    return out


_YOLO = None


# ------------------------------------------------------------------ text stage


_GALLERY = None


def gallery():
    global _GALLERY
    if _GALLERY is None:
        _GALLERY = pread.load_gallery()
    return _GALLERY


def line_texts(inp: Inputs, reach_pad: float = 40.0, ink_mode: str = 'adaptive',
               margin: float = 0.06, min_score: float = 0.45) -> List[ptxt.TextLine]:
    """Detect text lines near the component boxes and read them.

    Only lines whose centre falls inside a padded component box are read: reading every
    detected line on a 5100x3296 sheet costs minutes for no benefit, because a string that is
    nowhere near a component cannot be bound to one anyway.
    """
    box_src = 'yolo' if _FORCE_YOLO else 'gt'
    key = (inp.case.case_id, ink_mode, box_src)
    if key in _LINE_CACHE:
        return _LINE_CACHE[key]
    g = gallery()
    ink = inp.ink if ink_mode == 'adaptive' else ink_of(inp.bgr, ink_mode)
    boxes = [c.box for c in (yolo_components(inp) if _FORCE_YOLO else gt_components(inp))]
    cand = []
    for ln in ptxt.detect(inp.bgr, ink=ink, close_k=0):
        if not any(pcore.point_in_box((ln.cx, ln.cy), pcore.expand_box(b, reach_pad))
                   for b in boxes):
            continue
        sub = pread.tight_crop(ink, ln.box)
        if sub is None:
            continue
        text, chars, scores = pread.read_line(g, sub, margin=margin, min_score=min_score)
        ln.content = text
        ln.score = float(np.mean(scores)) if scores else 0.0
        ln.n_glyphs = sum(1 for c in chars if c)
        cand.append(ln)
    _LINE_CACHE[key] = cand
    return cand


_LINE_CACHE: Dict[Tuple[str, str, str], List[ptxt.TextLine]] = {}


def _comp_reach(c: pcore.Comp, pad: float = 60.0) -> float:
    """How far from a component box its own text may sit (scales with the part size)."""
    return pad + 2.0 * max(c.w, c.h)


def bind_designators(inp: Inputs, lines: Sequence[ptxt.TextLine],
                     reach: Optional[float] = None) -> Dict[int, int]:
    """Greedy nearest-box assignment of designator-shaped strings to components.

    A string like ``R1`` is claimed by exactly one component: candidates are sorted by the
    distance from the string to the component box and consumed in that order, which is the
    conflict resolution for two parts sitting next to each other.
    """
    comps = yolo_components(inp) if _FORCE_YOLO else gt_components(inp)
    pairs = []
    for li, ln in enumerate(lines):
        if not ln.content or not DESIG_RE.match(ln.content):
            continue
        for ci, c in enumerate(comps):
            d = pcore.point_to_bbox_dist((ln.cx, ln.cy), c.box)
            if d <= (_comp_reach(c) if reach is None else reach):
                pairs.append((d, li, ci))
    pairs.sort()
    got: Dict[int, int] = {}
    used_line, used_comp = set(), set()
    for d, li, ci in pairs:
        if li in used_line or ci in used_comp:
            continue
        used_line.add(li)
        used_comp.add(ci)
        got[ci] = li
    return got


def bind_values(inp: Inputs, lines: Sequence[ptxt.TextLine], taken: set,
                reach: Optional[float] = None) -> Dict[int, int]:
    """Nearest *non-designator* readable string per component (the value / part number)."""
    comps = yolo_components(inp) if _FORCE_YOLO else gt_components(inp)
    pairs = []
    for li, ln in enumerate(lines):
        if li in taken or not ln.content or DESIG_RE.match(ln.content):
            continue
        for ci, c in enumerate(comps):
            d = pcore.point_to_bbox_dist((ln.cx, ln.cy), c.box)
            if d <= (_comp_reach(c) if reach is None else reach):
                pairs.append((d, li, ci))
    pairs.sort()
    got: Dict[int, int] = {}
    used_line, used_comp = set(), set()
    for d, li, ci in pairs:
        if li in used_line or ci in used_comp:
            continue
        used_line.add(li)
        used_comp.add(ci)
        got[ci] = li
    return got


def read_components(inp: Inputs, strict_text: bool = False) -> Tuple[List[pcore.Comp], dict]:
    """Components whose key (and value / part number, where the scorer looks at it) come
    from reading the sheet.  Anonymous power symbols are dropped: their key `∅N` exists
    nowhere in the image, so no reader can ever produce it (43.9% of ground-truth keys).

    ``strict_text`` implements the asymmetry measured in ``metrics.component_f1``: a
    component whose text is missing scores ``fn`` only, while one that is emitted with the
    wrong text scores ``fp + fn``.  So for the two types whose text is compared at all
    (``r``/``c`` value, ``box`` Name) an unreadable text is better left out than guessed.
    """
    comps = yolo_components(inp) if _FORCE_YOLO else gt_components(inp)
    lines = line_texts(inp)
    desig = bind_designators(inp, lines)
    vals = bind_values(inp, lines, taken=set(desig.values()))
    out: List[pcore.Comp] = []
    diag = {'n_lines_read': sum(1 for l in lines if l.content), 'n_designator': len(desig),
            'n_value': len(vals), 'dist': [], 'dropped_text': 0, 'no_desig': 0}
    for ci, c in enumerate(comps):
        if pcore.is_anonymous(c.key):
            continue
        li = desig.get(ci)
        if li is None:
            diag['no_desig'] += 1
            continue
        ln = lines[li]
        diag['dist'].append(float(pcore.point_to_bbox_dist((ln.cx, ln.cy), c.box)))
        name, value = c.name, c.value
        text_ok = True
        vi = vals.get(ci)
        if c.ctype in C.IC_TYPES:
            name = lines[vi].content if vi is not None else None
            text_ok = name is not None
        elif c.ctype in C.VALUE_TYPES:
            value = lines[vi].content if vi is not None else None
            text_ok = value is not None
        if strict_text and not text_ok:
            diag['dropped_text'] += 1
            continue
        out.append(pcore.Comp(key=ln.content, ctype=c.ctype, box=c.box,
                              name=name, value=value))
    return out, diag


# ------------------------------------------------------------------ pin stage


#: Pin-name order along the dominant axis, **measured** by ``probe_pinorder`` over 220 sheets.
#: ``'21'`` means the leftmost/topmost tip is pin 2.  The convention is a property of the
#: symbol library, not of the geometry: for resistors it is 54/46 (a coin flip), for
#: capacitors 79/21, for LEDs 80/20 -- those accuracies are what the pin score inherits.
AXIS_ORDER = {
    'c': '21', 'capacitor': '21', 'r': '21', 'resistor': '21',
    'd': '12', 'led': '21', 'l': '12', 'fuse': '12', 'crystal_2pin': '21',
    'mosfet': '312', 'mosfet_npn': '312', 'mosfet_pnp': '312',
    'bjt': '312', 'bjt_npn': '312', 'bjt_pnp': '312',
}
#: Types whose pin names are *not* addressable by any geometric rule: `box` alone carries
#: 9356 pins whose names are real IC pad numbers (`P_13`, `GND`, `721`).  Emitting a wrong
#: name costs fp+fn while omitting a pin costs fn, so these abstain.
PIN_RULE_TYPES = frozenset({
    'r', 'resistor', 'c', 'capacitor', 'l', 'inductor', 'd', 'led', 'fuse',
    'crystal_2pin', 'mosfet', 'mosfet_npn', 'mosfet_pnp', 'bjt', 'bjt_npn', 'bjt_pnp',
    'pin', 'testpoint', 'battary', 'motor', 'jumper', 'ant', 'crystal',
    'amp_3pin', 'amp_5pin', 'crystal_3pin', 'crystal_4pin', 'seg', 'opt',
})


def rule_suffixes(ctype: str, tips: Sequence[Tuple[float, float]],
                  box: Box) -> List[Optional[str]]:
    """Invent pin suffixes from the tip layout (measurement: ``probe_pinorder``)."""
    tips = list(tips)
    if not tips:
        return []
    if ctype not in PIN_RULE_TYPES:
        return [None] * len(tips)
    if ctype in ('pin', 'testpoint'):
        return ['1'] + [None] * (len(tips) - 1)
    xs = [p[0] for p in tips]
    ys = [p[1] for p in tips]
    horiz = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    order = sorted(range(len(tips)),
                   key=(lambda i: tips[i][0]) if horiz else (lambda i: tips[i][1]))
    names = AXIS_ORDER.get(ctype) or ('12' if len(tips) == 2 else '312')
    out: List[Optional[str]] = [None] * len(tips)
    for pos, i in enumerate(order):
        out[i] = names[pos] if pos < len(names) else None
    return out


def pred_pins(inp: Inputs, comps: Sequence[pcore.Comp], pts: Sequence[Tuple[float, float]],
              reach: float = 10.0, merge: float = 5.0) -> List[pcore.Pin]:
    """Assign wire terminations to component boxes and invent their names."""
    from pipeline.validate_pins2 import cands_from_pts
    boxes = {c.key: c.box for c in comps}
    by_key = {c.key: c for c in comps}
    picked = cands_from_pts(pts, boxes, reach=reach, merge=merge)
    out: List[pcore.Pin] = []
    for key, tips in picked.items():
        comp = by_key[key]
        names = rule_suffixes(comp.ctype, tips, comp.box)
        for tip, suf in zip(tips, names):
            if suf is None:
                continue
            out.append(pcore.Pin(comp=key, suffix=suf, x=tip[0], y=tip[1], pinname=suf))
    return out


def gt_pins_of(inp: Inputs, comps: Sequence[pcore.Comp]) -> List[pcore.Pin]:
    keys = {c.key for c in comps}
    return [replace(p) for p in inp.truth.pins.values() if p.comp in keys]


# ------------------------------------------------------------------ net stage


def group_polylines(polys: Sequence[Sequence[Tuple[float, float]]], agg: float = 3.0
                    ) -> List[List[int]]:
    """Union-find polylines whose endpoints touch (crossings are already resolved by the
    skeleton tracer, so connectivity is plain endpoint adjacency)."""
    n = len(polys)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i, p in enumerate(polys):
        for end, pt in ((0, p[0]), (-1, p[-1])):
            k = (int(pt[0] // agg), int(pt[1] // agg))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (j, e2) in cell.get((k[0] + dx, k[1] + dy), []):
                        q = polys[j][0] if e2 == 0 else polys[j][-1]
                        if (q[0] - pt[0]) ** 2 + (q[1] - pt[1]) ** 2 <= agg * agg:
                            ra, rb = find(i), find(j)
                            if ra != rb:
                                parent[max(ra, rb)] = min(ra, rb)
            cell.setdefault(k, []).append((i, end))
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def pred_nets(inp: Inputs, pins: Sequence[pcore.Pin], skel: pskel.SkeletonResult,
              snap: float = 6.0, rdp_eps: float = 1.6) -> Dict[str, dict]:
    """Wires -> nets: polylines are already per-net fragments, so the net is a connected
    component of polylines; its pin set is the predicted pins that terminate on it.

    Polylines come out of the tracer at *pixel* resolution, so they are simplified with RDP
    first -- the ground-truth edges are straight 2-point segments (33037/33037), and a
    pixel-level chain would emit thousands of 1-px edges and score zero on ``net_line_f1``.
    """
    polys = [pskel.rdp(p, rdp_eps) for p in skel.polylines if len(p) >= 2]
    polys = [p for p in polys if len(p) >= 2]
    groups = group_polylines(polys)
    # spatial index over polyline endpoints -> polyline index
    cell: Dict[Tuple[int, int], List[int]] = {}
    for i, p in enumerate(polys):
        for pt in (p[0], p[-1]):
            cell.setdefault((int(pt[0] // 16), int(pt[1] // 16)), []).append(i)
    member: Dict[int, List[pcore.Pin]] = defaultdict(list)
    for pin in pins:
        best, bd = None, snap * snap
        k = (int(pin.x // 16), int(pin.y // 16))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for i in cell.get((k[0] + dx, k[1] + dy), []):
                    for pt in (polys[i][0], polys[i][-1]):
                        d = (pt[0] - pin.x) ** 2 + (pt[1] - pin.y) ** 2
                        if d <= bd:
                            bd, best = d, i
        if best is not None:
            member[best].append(pin)
    gid_of = {}
    for gi, g in enumerate(groups):
        for i in g:
            gid_of[i] = gi
    nets: Dict[str, dict] = {}
    for gi, g in enumerate(groups):
        members: List[pcore.Pin] = []
        for i in g:
            members.extend(member.get(i, []))
        seen = set()
        uniq = []
        for p in members:
            t = (p.comp, p.suffix)
            if t not in seen:
                seen.add(t)
                uniq.append(p)
        if not uniq:
            continue
        edges = {}
        ei = 0
        for i in g:
            p = polys[i]
            for a, b in zip(p, p[1:]):
                if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 0.25:
                    continue
                ei += 1
                edges['edge_%d' % ei] = [{'x': round(a[0], 2), 'y': round(a[1], 2)},
                                         {'x': round(b[0], 2), 'y': round(b[1], 2)}]
        if not edges:
            continue
        nets['net_%d' % (len(nets) + 1)] = {
            'hyperGraph': '(' + ','.join('%s.%s' % (p.comp, p.suffix)
                                         for p in sorted(uniq, key=lambda q: (q.comp, q.suffix))) + ')',
            'edges': edges}
    return nets


def gt_nets_of(inp: Inputs, comps: Sequence[pcore.Comp], pins: Sequence[pcore.Pin]) -> Dict[str, dict]:
    """Ground-truth wires restricted to the components/pins this prediction kept."""
    keys = {c.key for c in comps}
    pk = {(p.comp, p.suffix) for p in pins}
    out = {}
    i = 0
    for nk, net in inp.truth.nets.items():
        members = [t for t in pcore.parse_hypergraph(net['hyperGraph'])
                   if t[0] in keys and t in pk]
        if not members:
            continue
        i += 1
        edges = {}
        for j, (a, b) in enumerate(net['edges']):
            edges['edge_%d' % (j + 1)] = [{'x': round(a[0], 2), 'y': round(a[1], 2)},
                                          {'x': round(b[0], 2), 'y': round(b[1], 2)}]
        out['net_%d' % i] = {
            'hyperGraph': '(' + ','.join('%s.%s' % t for t in sorted(members)) + ')',
            'edges': edges}
    return out


# ------------------------------------------------------------------ document


def to_doc(inp: Inputs, comps: Sequence[pcore.Comp], pins: Sequence[pcore.Pin],
           nets: Dict[str, dict]) -> dict:
    h = inp.h
    comp_doc = {}
    for c in comps:
        comp_doc[c.key] = {'Name': c.name, 'type': c.ctype,
                           'value': c.value, 'bbox': C.bbox_img_to_json(c.box, h)}
    pin_doc: Dict[str, dict] = {}
    for p in pins:
        x, y = C.img_to_json((p.x, p.y), h)
        pin_doc.setdefault(p.comp, {})['pin_%s' % p.suffix] = {
            'pinname': p.pinname, 'point': {'x': round(x, 2), 'y': round(y, 2)}}
    net_doc = {}
    for k, net in nets.items():
        edges = {}
        for ek, seg in net['edges'].items():
            edges[ek] = [{'x': seg[0]['x'], 'y': round(h - seg[0]['y'], 2)},
                         {'x': seg[1]['x'], 'y': round(h - seg[1]['y'], 2)}]
        net_doc[k] = {'hyperGraph': net['hyperGraph'], 'edges': edges}
    return {'components': comp_doc, 'pins': pin_doc, 'nets': net_doc}


# ------------------------------------------------------------------ variants

_FORCE_YOLO = False


@dataclass
class Variant:
    name: str
    key_mode: str = 'read'        # 'gt' | 'read'
    box_mode: str = 'gt'          # 'gt' | 'yolo'
    pin_mode: str = 'skel'        # 'gt' | 'skel' | 'none'
    net_mode: str = 'skel'        # 'gt' | 'skel' | 'none'
    ink_mode: str = 'adaptive'
    strict_text: bool = False
    key_p: float = 1.0            # designator read correctly with this probability
    key_miss: str = 'wrong'       # 'wrong' (emit an unmatchable key) | 'drop' (abstain)


def run_variant(inp: Inputs, v: Variant) -> Tuple[dict, dict]:
    if v.box_mode == 'yolo':
        comps0 = yolo_components(inp)
    else:
        comps0 = gt_components(inp)
    diag: Dict[str, float] = {}
    if v.key_mode == 'gt':
        comps = [c for c in comps0 if not pcore.is_anonymous(c.key)]
        if v.key_p < 1.0:
            # the OCR payoff curve: a designator read correctly with probability ``key_p``.
            # A misread is *not* a missing component in the real pipeline -- the box is still
            # there and still emits a string -- so the default is to keep it with a key that
            # cannot match (fp + fn instead of a single fn).  ``key_miss='drop'`` models an
            # abstaining reader for comparison.
            rng = random.Random(20240517)
            kept: List[pcore.Comp] = []
            n_bad = 0
            for c in comps:
                if rng.random() < v.key_p:
                    kept.append(c)
                elif v.key_miss == 'wrong':
                    n_bad += 1
                    kept.append(pcore.Comp(key='?%d' % n_bad, ctype=c.ctype, box=c.box,
                                           name=c.name, value=c.value))
            comps = kept
            diag['n_badkey'] = n_bad
    else:
        comps, rdiag = read_components(inp, strict_text=v.strict_text)
        diag.update(rdiag)
    if v.pin_mode == 'gt':
        pins = gt_pins_of(inp, comps)
    elif v.pin_mode == 'none':
        pins = []
    else:
        skel = skeleton_of(inp, v.ink_mode)
        pts = [(x, y) for x, y in skel.endpoints]
        pins = pred_pins(inp, comps, pts, reach=10.0)
    if v.net_mode == 'none':
        nets = {}
    elif v.net_mode == 'gt':
        nets = gt_nets_of(inp, comps, pins)
    else:
        skel = skeleton_of(inp, v.ink_mode)
        nets = pred_nets(inp, pins, skel)
    doc = to_doc(inp, comps, pins, nets)
    diag['n_comp'] = len(comps)
    diag['n_pin'] = len(pins)
    diag['n_net'] = len(nets)
    return doc, diag


LADDER = [
    Variant('gt-all', key_mode='gt', pin_mode='gt', net_mode='gt'),
    Variant('ocr-p100', key_mode='gt', pin_mode='gt', net_mode='gt', key_p=1.0),
    Variant('ocr-p90', key_mode='gt', pin_mode='gt', net_mode='gt', key_p=0.9),
    Variant('ocr-p70', key_mode='gt', pin_mode='gt', net_mode='gt', key_p=0.7),
    Variant('ocr-p70-drop', key_mode='gt', pin_mode='gt', net_mode='gt', key_p=0.7,
            key_miss='drop'),
    Variant('ocr-p50', key_mode='gt', pin_mode='gt', net_mode='gt', key_p=0.5),
    Variant('ocr-p30', key_mode='gt', pin_mode='gt', net_mode='gt', key_p=0.3),
    Variant('gt-box+pinrules', key_mode='gt', pin_mode='skel', net_mode='gt'),
    Variant('gt-key+skel', key_mode='gt', pin_mode='skel', net_mode='skel'),
    Variant('read-key+skel', key_mode='read', pin_mode='skel', net_mode='skel'),
    Variant('read-key+skel-strict', key_mode='read', pin_mode='skel', net_mode='skel',
            strict_text=True),
    Variant('read-key+skel-otsu', key_mode='read', pin_mode='skel', net_mode='skel',
            ink_mode='otsu'),
    Variant('read-key+no-nets', key_mode='read', pin_mode='skel', net_mode='none'),
    Variant('ocr-p90+skel', key_mode='gt', pin_mode='skel', net_mode='skel', key_p=0.9),
    Variant('ocr-p70+skel', key_mode='gt', pin_mode='skel', net_mode='skel', key_p=0.7),
    Variant('ocr-p90+rules+gtnets', key_mode='gt', pin_mode='skel', net_mode='gt',
            key_p=0.9),
    Variant('yolo-key+skel', key_mode='read', box_mode='yolo', pin_mode='skel',
            net_mode='skel'),
]


def main(argv):
    global _FORCE_YOLO
    n = 8
    want_yolo = False
    ids = []
    for a in argv:
        if a.startswith('--n='):
            n = int(a.split('=')[1])
        elif a == '--yolo':
            want_yolo = True
        elif not a.startswith('-'):
            ids.append(a)
    if ids:
        cases = [c for c in (pcore.find_case(i) for i in ids) if c is not None and c.gt_json]
    else:
        cases = [c for c in pcore.discover() if c.split == '200_train_cases' and c.gt_json][:n]
    print('cases (%d): %s' % (len(cases), ', '.join(c.case_id for c in cases)))

    variants = list(LADDER)
    if not want_yolo:
        variants = [v for v in variants if v.box_mode != 'yolo']

    rows: Dict[str, List] = defaultdict(list)
    diags: Dict[str, List[dict]] = defaultdict(list)
    for case in cases:
        _FORCE_YOLO = False
        inp = inputs_of(case)
        for v in variants:
            _FORCE_YOLO = (v.box_mode == 'yolo')
            doc, diag = run_variant(inp, v)
            sc = M.evaluate_case(inp.gt_doc, doc, case.case_id)
            rows[v.name].append(sc)
            diags[v.name].append(diag)
            if v.name == 'read-key+skel':
                d = [x for x in diag.get('dist', [])]
                if d:
                    print('  %-8s designator->box dist  p50=%.1f p90=%.1f max=%.1f  '
                          'comps=%d pins=%d nets=%d'
                          % (case.case_id, float(np.median(d)),
                             float(np.percentile(d, 90)), float(np.max(d)),
                             diag['n_comp'], diag['n_pin'], diag['n_net']))
                else:
                    print('  %-8s no designator bound (%d comps, %d pins)'
                          % (case.case_id, diag['n_comp'], diag['n_pin']))

    print('\n%-22s %7s %7s %7s %7s %7s %7s' % ('variant', 'score', 'compF1', 'pinF1',
                                               'netHF1', 'netLF1', 'n_comp'))
    print('-' * 74)
    for v in variants:
        agg = M.aggregate(rows[v.name])
        nc = float(np.mean([d['n_comp'] for d in diags[v.name]]))
        print('%-22s %7.4f %7.4f %7.4f %7.4f %7.4f %7.1f'
              % (v.name, agg['mean_score'], agg['component_f1'], agg['pin_f1'],
                 agg['net_hypergraph_f1'], agg['net_line_f1'], nc))

    os.makedirs(C.OUT_ROOT, exist_ok=True)
    with open(os.path.join(C.OUT_ROOT, 'prun.json'), 'w', encoding='utf-8') as f:
        json.dump({'cases': [c.case_id for c in cases],
                   'variants': {v.name: M.aggregate(rows[v.name]) for v in variants},
                   'per_case': {v.name: [{'case': s.case_id,
                                          'score': round(s.score, 4),
                                          'component': round(s.component.f1, 4),
                                          'pin': round(s.pin.f1, 4),
                                          'net_hypergraph': round(s.net_hypergraph.f1, 4),
                                          'net_line': round(s.net_line.f1, 4)}
                                         for s in rows[v.name]] for v in variants}},
                  f, ensure_ascii=False, indent=1)
    print('\nsaved -> %s' % os.path.join(C.OUT_ROOT, 'prun.json'))


if __name__ == '__main__':
    main([a for a in sys.argv[1:]])
