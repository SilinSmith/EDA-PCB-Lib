# -*- coding: utf-8 -*-
"""Case loading plus the small geometry helpers every stage of the pipeline needs.

Nothing here touches pixels: this module is the bridge between the competition JSON
(bottom-left origin) and the image space every algorithm works in (top-left origin).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (ANONYMOUS_TYPES, DATA_ROOT, REF_ROOT, bbox_json_to_img,
                     json_to_img)


# ---------------------------------------------------------------- case model

@dataclass
class Comp:
    """One detected/known component, in **image** coordinates."""

    key: str
    ctype: str
    box: Tuple[float, float, float, float]          # x1, y1, x2, y2 (top-left origin)
    name: Optional[str] = None
    value: Optional[str] = None
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class Pin:
    """A pin tip in image coordinates."""

    comp: str
    suffix: str
    x: float
    y: float
    pinname: Optional[str] = None
    conf: float = 1.0


@dataclass
class Case:
    split: str
    case_id: str
    png: str
    gt_json: Optional[str]
    ref_json: Optional[str] = None                # Dataset300 CVJsonStd, when present
    ref_cls: Optional[str] = None                 # vendor's own source class from the ref json

    @property
    def key(self) -> str:
        return '%s/%s' % (self.split, self.case_id)

    @property
    def source(self) -> str:
        for s in ('KiCad', 'Datasheet', 'Altium Designer', 'jlc', 'other'):
            if s.lower().replace(' ', '') in self.case_id.lower().replace(' ', ''):
                return s
        return self.ref_cls or 'unknown'


def discover_ref_ids() -> Dict[str, str]:
    """``case id -> CVJsonStd json path`` for the 32 reference-representation cases."""
    out = {}
    for path in glob.glob(os.path.join(REF_ROOT, '*', '*CVJsonStd.json')):
        cid = os.path.basename(os.path.dirname(path))
        # dir names look like "0059-datasheet": key on the leading id
        out[cid.split('-')[0]] = path
    return out


def _ref_class(path: str) -> Optional[str]:
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
    except Exception:
        return None
    for k in ('nowCls', 'source', 'cls', 'model'):
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return None


_CASE_CACHE: Optional[List[Case]] = None


def discover(data_root: str = DATA_ROOT) -> List[Case]:
    global _CASE_CACHE
    if _CASE_CACHE is not None and data_root == DATA_ROOT:
        return _CASE_CACHE
    refs = discover_ref_ids()
    cases: List[Case] = []
    for split in ('10GT', '10_GTcase', '200_train_cases'):
        sdir = os.path.join(data_root, split)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, name)
            if not os.path.isdir(cdir):
                continue
            pngs = sorted(f for f in os.listdir(cdir) if f.lower().endswith('.png'))
            if not pngs:
                continue
            js = sorted(f for f in os.listdir(cdir) if f.lower().endswith('.json'))
            cases.append(Case(split=split, case_id=name,
                              png=os.path.join(cdir, pngs[0]),
                              gt_json=os.path.join(cdir, js[0]) if js else None,
                              ref_json=refs.get(name),
                              ref_cls=_ref_class(refs[name]) if name in refs else None))
    # cases that exist only in the vendor-representation directory: no ground-truth JSON,
    # but a perfect wire-skeleton reference -- the gold standard for the skeleton stage.
    for cdir in sorted(glob.glob(os.path.join(REF_ROOT, '*'))):
        if not os.path.isdir(cdir):
            continue
        name = os.path.basename(cdir)
        cid = name.split('-')[0]
        if any(c.case_id == cid for c in cases):
            continue
        pngs = sorted(f for f in os.listdir(cdir) if f.lower().endswith('.png'))
        rp = refs.get(cid)
        if not pngs or not rp:
            continue
        cases.append(Case(split='Dataset300', case_id=cid,
                          png=os.path.join(cdir, pngs[0]), gt_json=None,
                          ref_json=rp, ref_cls=_ref_class(rp)))
    if data_root == DATA_ROOT:
        _CASE_CACHE = cases
    return cases


def find_case(case_id: str) -> Optional[Case]:
    for c in discover():
        if c.case_id == case_id:
            return c
    return None


def load_gt(case: Case) -> dict:
    with open(case.gt_json, encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------- gt -> image space

@dataclass
class Truth:
    """Ground truth of one case, converted once into image coordinates."""

    h: float
    w: float
    comps: Dict[str, Comp] = field(default_factory=dict)
    raw_comp: Dict[str, dict] = field(default_factory=dict)
    pins: Dict[Tuple[str, str], Pin] = field(default_factory=dict)
    nets: Dict[str, dict] = field(default_factory=dict)      # key -> {'pins': [...], 'edges': [...]}
    doc: dict = field(default_factory=dict)

    def pin_pts(self):
        return np.array([[p.x, p.y] for p in self.pins.values()], dtype=float)


def truth_from_doc(doc: dict, w: float, h: float) -> Truth:
    t = Truth(h=float(h), w=float(w), doc=doc)
    for key, c in (doc.get('components') or {}).items():
        box = bbox_json_to_img(c.get('bbox') or [0, 0, 0, 0], h)
        t.comps[key] = Comp(key=key, ctype=c.get('type') or '', box=box,
                            name=c.get('Name'), value=c.get('value'))
        t.raw_comp[key] = c
    for key, grp in (doc.get('pins') or {}).items():
        for pk, info in (grp or {}).items():
            suffix = pk[4:] if pk.startswith('pin_') else pk
            pt = info.get('point') or {}
            x, y = json_to_img((pt.get('x', 0.0), pt.get('y', 0.0)), h)
            t.pins[(key, suffix)] = Pin(comp=key, suffix=suffix, x=x, y=y,
                                        pinname=info.get('pinname'))
    for nk, net in (doc.get('nets') or {}).items():
        edges = []
        for e in (net.get('edges') or {}).values():
            try:
                edges.append((json_to_img((e[0]['x'], e[0]['y']), h),
                              json_to_img((e[1]['x'], e[1]['y']), h)))
            except Exception:
                continue
        t.nets[nk] = {'hyperGraph': net.get('hyperGraph', ''), 'edges': edges}
    return t


# ---------------------------------------------------------------- net helpers

def parse_hypergraph(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    s = text.strip()
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    out, seen = [], set()
    for tok in s.split(','):
        tok = tok.strip()
        if '.' not in tok:
            continue
        comp, _, num = tok.rpartition('.')
        ref = (comp.strip(), num.strip())
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


# ---------------------------------------------------------------- misc geometry

def seg_point_dist(p, a, b) -> float:
    """Distance from point *p* to segment *a-b*."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float((p - a) @ ab) / L2
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_to_bbox_dist(p, box) -> float:
    """Unsigned distance from a point to an axis-aligned box (0 inside)."""
    x, y = p
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def point_in_box(p, box, margin: float = 0.0) -> bool:
    x, y = p
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def expand_box(box, m: float):
    return (box[0] - m, box[1] - m, box[2] + m, box[3] + m)


def is_anonymous(key: str) -> bool:
    return key.startswith('\u2205')
