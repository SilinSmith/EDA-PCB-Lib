# -*- coding: utf-8 -*-
"""Text detection + role binding on the schematic image.

Facts this module relies on (all measured, see ``docs/算法路线设计.md``):

* the vendor reference file (``<id>-<source>_CVJsonStd.json``) keeps every text box in
  ``texts[T_n]`` with ``content`` (the string) and ``details`` (its semantic role:
  ``designator`` / ``value`` / ``pinNumber`` / ``pinName`` / ``netlabel`` / ``searchName``);
* the synthesiser ships the same information as ``texts.jsonl`` with a ``role`` and a
  ``comp`` field -- that is the *labelled* corpus used to fit and score the binder, so no
  real-data tuning happens here.

Detection (no learned model, no OCR engine available in this environment):
    1. ink mask -> morphological closing to fuse glyph strokes inside one character;
    2. connected components, drop those too thin/tall/small to be a glyph;
    3. cluster glyphs into text lines by baseline + gap heuristics;
    4. report each line as one ``TextLine`` with ``content`` left to the recogniser.

Recognition is a separate concern (``precog``) so that the *binding* geometry can be scored
on its own, with ground-truth strings, exactly like the pin geometry was.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore

Box = Tuple[float, float, float, float]


def imread(path: str) -> Optional[np.ndarray]:
    """``cv2.imread`` fails on non-ASCII paths (``合成原理图demo``); ``imdecode`` does not."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@dataclass
class Glyph:
    box: Box
    area: int
    h: float
    w: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.box[0] + self.box[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.box[1] + self.box[3])


@dataclass
class TextLine:
    box: Box
    n_glyphs: int
    height: float
    content: Optional[str] = None
    role: Optional[str] = None
    score: float = 0.0
    comp: Optional[str] = None

    @property
    def cx(self) -> float:
        return 0.5 * (self.box[0] + self.box[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.box[1] + self.box[3])


# ------------------------------------------------------------------ detection


def glyph_components(bgr: np.ndarray, ink: Optional[np.ndarray] = None,
                     close_k: int = 2, min_h: float = 6.0, max_h: float = 46.0,
                     max_ar: float = 2.6, min_area: int = 12) -> List[Glyph]:
    """Connected components of the ink mask, filtered to glyph-like shapes.

    The filters are deliberately loose (a glyph may be as small as 6 px tall on the very
    large sheets); precision comes later from the line clustering and the binder.
    """
    if ink is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 8)
    if close_k > 0:
        k = np.ones((close_k + 1, close_k + 1), np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, lab, stats, _cent = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    out: List[Glyph] = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or h < min_h or h > max_h or w < 1:
            continue
        # glyphs are taller than wide, but allow '.', '-', '=' and 1
        if w > max_ar * h and w > 6:
            continue
        out.append(Glyph((float(x), float(y), float(x + w), float(y + h)), int(a), float(h), float(w)))
    return out


def group_lines(glyphs: Sequence[Glyph], y_tol_ratio: float = 0.55, gap_ratio: float = 1.35,
                max_lines: int = 40000) -> List[TextLine]:
    """Cluster glyphs into text lines: sort by y, chain glyphs whose vertical overlap is
    large and whose horizontal gap is within ``gap_ratio`` glyph heights."""
    if not glyphs:
        return []
    gl = sorted(glyphs, key=lambda g: (g.cy, g.cx))
    lines: List[List[Glyph]] = []
    for g in gl:
        placed = False
        for ln in reversed(lines[-40:]):
            ref = ln[-1]
            h = max(g.h, ref.h)
            if abs(g.cy - ref.cy) <= y_tol_ratio * h:
                gap = g.box[0] - ref.box[2]
                if -0.9 * h <= gap <= gap_ratio * h:
                    ln.append(g)
                    placed = True
                    break
        if not placed:
            lines.append([g])
    out: List[TextLine] = []
    for ln in lines:
        xs1 = min(g.box[0] for g in ln)
        ys1 = min(g.box[1] for g in ln)
        xs2 = max(g.box[2] for g in ln)
        ys2 = max(g.box[3] for g in ln)
        out.append(TextLine((xs1, ys1, xs2, ys2), len(ln),
                            float(np.median([g.h for g in ln]))))
    return out[:max_lines]


def detect(bgr: np.ndarray, ink: Optional[np.ndarray] = None, **kw) -> List[TextLine]:
    return group_lines(glyph_components(bgr, ink=ink, **kw))


# ------------------------------------------------------------------ reference text


def ref_texts(case) -> List[Tuple[Box, str, str]]:
    """Vendor reference text boxes: ``(box, content, role)`` in **image** coordinates."""
    if not getattr(case, 'ref_json', None):
        return []
    try:
        doc = json.load(open(case.ref_json, encoding='utf-8'))
    except Exception:
        return []
    im = imread(case.png)
    if im is None:
        return []
    H = float(im.shape[0])
    out = []
    for tid, t in (doc.get('texts') or {}).items():
        loc = (t or {}).get('location') or {}
        try:
            box = (float(loc['x1']), float(loc['y1']), float(loc['x2']), float(loc['y2']))
        except Exception:
            continue
        role = C.role_of((t or {}).get('details'))
        out.append((box, str((t or {}).get('content') or ''), role))
    return out


def synth_texts() -> Dict[str, List[dict]]:
    """The synthesiser's labelled text corpus, keyed by case name (image coordinates)."""
    path = os.path.join(C.SYNTH_ROOT, 'texts.jsonl')
    out: Dict[str, List[dict]] = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        out[rec.get('case', '?')] = rec.get('texts') or []
    return out


def cover(box: Box, ref: Box, iou_min: float = 0.5) -> bool:
    """Does *box* claim *ref*?  Uses intersection-over-reference so a detector that merges
    several glyphs into one line still gets credit for each of them."""
    ix1 = max(box[0], ref[0]); iy1 = max(box[1], ref[1])
    ix2 = min(box[2], ref[2]); iy2 = min(box[3], ref[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    ra = max(1e-6, (ref[2] - ref[0]) * (ref[3] - ref[1]))
    return inter / ra >= iou_min
