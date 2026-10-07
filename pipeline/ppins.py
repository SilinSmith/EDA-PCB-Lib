# -*- coding: utf-8 -*-
"""Pin-tip recovery: bbox + wire skeleton -> the pin tips the competition JSON wants.

The competition scores a pin as ``(component_key, pin_number)`` with a ``pinname`` and a point
that must be within 5 px of ground truth.  The detector only gives us the component box, and
the corpus is unambiguous about where the pins are:

* the ground-truth component ``bbox`` is the **body only** -- 22691/23062 pin tips sit
  *outside* it (``{'outside': 22691, 'on_border': 371}``), because a pin tip is the far end of
  the pin stub (example: ``∅202 (v) bbox=[1074,286,1084,296]``, ``pin=(1079,276)``);
* a wire is attached to nearly every pin (97.9% of pins appear in some net's hyperGraph);

so the pin tips are recovered from the geometry of the wire skeletons around the box: a wire
that terminates next to a box edge does so *at* the pin stub, and the outer end of that stub
is the tip.  This module implements that as

  1. collect the wire skeleton endpoints and the skeleton/box-edge crossings,
  2. for each component box, take the candidates within ``reach`` px of the box,
  3. snap each candidate to the nearest box edge and lay the pins out on the edge,
  4. walk outward along the attached skeleton to the stub end -> the tip.

Everything in image coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pcore


@dataclass
class PinCand:
    comp: str
    x: float
    y: float
    edge: str                     # 'L' | 'R' | 'T' | 'B'
    score: float = 1.0
    via: str = 'stub'             # how it was found: 'endpoint' | 'crossing'
    dist: float = 0.0


def nearest_edge(p, box) -> Tuple[str, float]:
    """Nearest box edge to *p* (outside the box) and the distance to it."""
    x, y = p
    x1, y1, x2, y2 = box
    d = {'L': abs(x - x1), 'R': abs(x - x2), 'T': abs(y - y1), 'B': abs(y - y2)}
    # if the point is inside, use the closest wall anyway
    e = min(d, key=lambda k: d[k])
    return e, float(d[e])


def edge_point(p, box, edge: str) -> Tuple[float, float]:
    x, y = p
    x1, y1, x2, y2 = box
    if edge == 'L':
        return (x1, min(max(y, y1), y2))
    if edge == 'R':
        return (x2, min(max(y, y1), y2))
    if edge == 'T':
        return (min(max(x, x1), x2), y1)
    return (min(max(x, x1), x2), y2)


def _seg_box_hits(a, b, box, tol: float = 1.5, margin: float = 3.0):
    """Intersections of segment a-b with the four lines of *box*, kept when near the border.

    Vectorised: solves the two parametric line equations per wall instead of marching pixels.
    """
    x1, y1, x2, y2 = box
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    out = []
    # vertical walls x = x1, x = x2
    for xw in (x1, x2):
        if abs(dx) < 1e-9:
            continue
        t = (xw - ax) / dx
        if not (-1e-9 <= t <= 1 + 1e-9):
            continue
        y = ay + t * dy
        if y1 - margin <= y <= y2 + margin:
            out.append((xw, y))
    # horizontal walls y = y1, y = y2
    for yw in (y1, y2):
        if abs(dy) < 1e-9:
            continue
        t = (yw - ay) / dy
        if not (-1e-9 <= t <= 1 + 1e-9):
            continue
        x = ax + t * dx
        if x1 - margin <= x <= x2 + margin:
            out.append((x, yw))
    _ = tol
    return out


def candidates_from_skeleton(res, boxes: Dict[str, Tuple[float, float, float, float]],
                             reach: float = 6.0, cross_tol: float = 1.5,
                             comp_of=None) -> List[PinCand]:
    """Pin candidates near each box, from skeleton endpoints and skeleton/box crossings."""
    out: List[PinCand] = []
    ends = list(res.endpoints)
    if not ends and res.polylines:
        ends = [p[0] for p in res.polylines] + [p[-1] for p in res.polylines]

    # spatial index over the boxes so we never do segments x boxes work
    CELL = 64.0
    grid: Dict[Tuple[int, int], List[str]] = {}
    for key, box in boxes.items():
        ex1, ey1, ex2, ey2 = pcore.expand_box(box, reach)
        for gx in range(int(ex1 // CELL), int(ex2 // CELL) + 1):
            for gy in range(int(ey1 // CELL), int(ey2 // CELL) + 1):
                grid.setdefault((gx, gy), []).append(key)

    def near_keys(x1, y1, x2, y2):
        seen = set()
        for gx in range(int(x1 // CELL), int(x2 // CELL) + 1):
            for gy in range(int(y1 // CELL), int(y2 // CELL) + 1):
                for k in grid.get((gx, gy), ()):
                    seen.add(k)
        return seen

    for (x, y) in ends:
        for key in near_keys(x, y, x, y):
            box = boxes[key]
            if pcore.point_in_box((x, y), box):
                continue
            if not pcore.point_in_box((x, y), pcore.expand_box(box, reach)):
                continue
            e, d = nearest_edge((x, y), box)
            out.append(PinCand(comp=key, x=x, y=y, edge=e, via='endpoint', dist=d))

    for a, b in res.segments:
        for key in near_keys(min(a[0], b[0]), min(a[1], b[1]),
                             max(a[0], b[0]), max(a[1], b[1])):
            box = boxes[key]
            for c in _seg_box_hits(a, b, box, cross_tol):
                e, d = nearest_edge(c, box)
                if d > 2.0:
                    continue
                out.append(PinCand(comp=key, x=c[0], y=c[1], edge=e, via='crossing', dist=d))
    _ = comp_of
    return out


def infer_stub_tips(res, box, base_pt, max_out: float = 26.0,
                    step: float = 2.0) -> Tuple[float, float]:
    """Slide outward from *base_pt* along the wire direction and stop where the wire ends.

    The ground truth pin tip is the far end of the pin stub, i.e. typically 10-20 px away from
    the symbol body along the stub.  We estimate it with the local EDT: the stroke keeps being
    ink for the whole stub, then stops.
    """
    ink = res.ink
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    vx, vy = base_pt[0] - cx, base_pt[1] - cy
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return base_pt
    vx, vy = vx / n, vy / n
    h, w = ink.shape[:2]
    last_ink = 0.0
    k = 1
    while k * step <= max_out:
        x = base_pt[0] + vx * k * step
        y = base_pt[1] + vy * k * step
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            break
        # stop once the outward ray has been out of ink for a whole step
        if ink[yi, xi] == 0:
            xb = base_pt[0] + vx * (k - 1) * step
            yb = base_pt[1] + vy * (k - 1) * step
            if ink[int(round(yb)), int(round(xb))] == 0:
                break
            last_ink = k * step
            break
        last_ink = k * step
        k += 1
    tip = (base_pt[0] + vx * last_ink, base_pt[1] + vy * last_ink)
    return tip


def cluster_candidates(cands: Sequence[PinCand], merge: float = 5.0) -> List[PinCand]:
    """Merge candidates of the same component that are close together (same stub)."""
    by_comp: Dict[str, List[PinCand]] = {}
    for c in cands:
        by_comp.setdefault(c.comp, []).append(c)
    out: List[PinCand] = []
    for key, cs in by_comp.items():
        # greedy: sort by edge then along-edge coordinate, merge neighbours on the same edge
        by_edge: Dict[str, List[PinCand]] = {}
        for c in cs:
            by_edge.setdefault(c.edge, []).append(c)
        for edge, es in by_edge.items():
            horiz = edge in ('T', 'B')
            es = sorted(es, key=lambda c: (c.x if horiz else c.y))
            cur = [es[0]]
            for c in es[1:]:
                prev = cur[-1]
                gap = abs((c.x - prev.x) if horiz else (c.y - prev.y))
                if gap <= merge:
                    cur.append(c)
                else:
                    out.append(_merge(cur))
                    cur = [c]
            out.append(_merge(cur))
    return out


def _merge(cs: Sequence[PinCand]) -> PinCand:
    # prefer the far end of the stub: the candidate with the largest distance from the body
    best = max(cs, key=lambda c: c.dist)
    x = float(np.mean([c.x for c in cs]))
    y = float(np.mean([c.y for c in cs]))
    via = 'endpoint' if any(c.via == 'endpoint' for c in cs) else 'crossing'
    return PinCand(comp=best.comp, x=x, y=y, edge=best.edge, score=best.score,
                   via=via, dist=best.dist)


def recover_pins(res, comps, reach: float = 6.0, merge: float = 5.0,
                 tips: bool = True, max_out: float = 26.0) -> List[PinCand]:
    """Full pin recovery for a whole page."""
    boxes = {c.key: c.box for c in comps}
    cands = candidates_from_skeleton(res, boxes, reach=reach)
    cands = cluster_candidates(cands, merge=merge)
    if not tips:
        return cands
    for c in cands:
        box = boxes.get(c.comp)
        if box is None:
            continue
        c.x, c.y = infer_stub_tips(res, box, (c.x, c.y), max_out=max_out)
    return cands
