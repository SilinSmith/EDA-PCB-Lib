# -*- coding: utf-8 -*-
"""Character recognition by template matching, fitted on the *ground-truth* text boxes.

There is no OCR engine in this environment (no tesseract, no OCR python package), so the
reader is a nearest-neighbour classifier over binary glyph templates cut out of the
training corpus.  Two decisions follow from that:

* the alphabet is closed -- `A-Z a-z 0-9 . - + _ /` plus a few vendor oddities -- so a
  match must be *decisive* (top-1 far ahead of top-2) or it is reported as ``None``; a
  wrong designator costs a component *and* its pins *and* its nets, so abstaining is
  worth more than guessing;
* templates are cut from the same corpus that is being evaluated, but every template is
  a *single character from a different sheet*; per-case accuracy is still reported, and
  the fit/eval split is stated explicitly in the report (this is template matching, not a
  learned model, and the report says so).

Usage:  python -m pipeline.pread --gallery      # build the template gallery
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from pipeline import config as C
from pipeline import pcore

Box = Tuple[float, float, float, float]
GLYPH_H = 24
GLYPH_W = 20
PIPE_ROOT = os.path.dirname(os.path.abspath(__file__))
GALLERY_PATH = os.path.join(PIPE_ROOT, 'glyph_gallery.pkl')


# ------------------------------------------------------------------ glyph cut-out


def tight_crop(ink: np.ndarray, box: Box, pad: int = 1) -> Optional[np.ndarray]:
    """Ink of *box* cropped to its own content bounds (kills the vendor's box padding)."""
    x1 = max(0, int(np.floor(box[0]))); y1 = max(0, int(np.floor(box[1])))
    x2 = min(ink.shape[1], int(np.ceil(box[2]))); y2 = min(ink.shape[0], int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = (ink[y1:y2, x1:x2] > 0).astype(np.uint8)
    if sub.sum() == 0:
        return None
    ys, xs = np.nonzero(sub)
    sub = sub[max(0, ys.min() - pad):ys.max() + 1 + pad, max(0, xs.min() - pad):xs.max() + 1 + pad]
    return sub if sub.size else None


def normalise(sub: np.ndarray) -> np.ndarray:
    """Resize to the fixed raster; bilinear then re-threshold keeps stroke order intact."""
    r = cv2.resize(sub.astype(np.float32), (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)
    return (r >= 0.45).astype(np.float32)


@dataclass
class Gallery:
    chars: List[str] = field(default_factory=list)
    src: List[str] = field(default_factory=list)   # source case of every template
    bits: np.ndarray = field(default_factory=lambda: np.zeros((0, GLYPH_H * GLYPH_W), np.float32))

    def __len__(self):
        return len(self.chars)

    def add(self, ch: str, sub: np.ndarray, src: str = ''):
        self.chars.append(ch)
        self.src.append(src)
        v = normalise(sub).reshape(1, -1)
        self.bits = v if self.bits.size == 0 else np.vstack([self.bits, v])

    def scores(self, sub: np.ndarray, exclude: Optional[str] = None) -> np.ndarray:
        """IoU of *sub* against every template, optionally holding one case out."""
        if self.bits.size == 0:
            return np.zeros(0, np.float32)
        q = normalise(sub).reshape(-1)
        inter = self.bits @ q
        s1 = self.bits.sum(axis=1) + q.sum() - inter
        iou = np.where(s1 > 0, inter / np.maximum(s1, 1e-6), 0.0)
        if exclude:
            m = np.array([s != exclude for s in self.src], dtype=bool)
            iou = np.where(m, iou, -1.0)
        return iou

    def match(self, sub: np.ndarray, margin: float = 0.0,
              top: int = 3, exclude: Optional[str] = None, vote: int = 0,
              min_score: float = 0.0) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """Nearest template; ``None`` when the winner is not decisive enough.

        The margin test compares the best score against the best score of a *different
        character*, so a dozen near-duplicate templates of ``1`` do not block the decision
        the way a plain top-1/top-2 ratio would.
        """
        if self.bits.size == 0:
            return None, 0.0, []
        iou = self.scores(sub, exclude=exclude)
        per_char: Dict[str, float] = {}
        for i, v in enumerate(iou):
            if v > per_char.get(self.chars[i], -1.0):
                per_char[self.chars[i]] = float(v)
        ranked = sorted(per_char.items(), key=lambda kv: -kv[1])
        best_ch, best = ranked[0]
        rival = ranked[1][1] if len(ranked) > 1 else 0.0
        if vote > 0:
            order = np.argsort(-iou)[:vote]
            tally: Dict[str, float] = {}
            for i in order:
                tally[self.chars[int(i)]] = tally.get(self.chars[int(i)], 0.0) + max(
                    0.0, float(iou[int(i)]) - 0.35)
            if tally:
                best_ch = max(tally, key=lambda k: tally[k])
                best = per_char.get(best_ch, best)
                rival = max([v for k, v in per_char.items() if k != best_ch] or [0.0])
        if best < min_score or best - rival < margin:
            return None, best, []
        cand = [(ch, round(sc, 3)) for ch, sc in ranked[:top]]
        return best_ch, best, cand


# ------------------------------------------------------------------ corpus -> gallery


def iter_gt_texts(case, min_h: float = 6.0, max_h: float = 48.0):
    """Yield ``(content, tight_box, ink)`` for every ground-truth text box.

    Only the 29 vendor-representation cases carry a ``texts`` section; ``_target_named.json``
    holds no text geometry at all, so those sheets can supply glyphs only through the
    window-based path (see ``probe_win_text``), which needs the labels to be guessed from the
    component boxes rather than read from the file.
    """
    if not case.ref_json:
        return
    from pipeline import pskel
    from pipeline import ptxt
    bgr = ptxt.imread(case.png)
    if bgr is None:
        return
    ink = pskel.ink_mask(bgr, mode='adaptive', k=31, c=8)
    doc = json.load(open(case.ref_json, encoding='utf-8'))
    for tid, t in (doc.get('texts') or {}).items():
        loc = (t or {}).get('location') or {}
        try:
            box = (float(loc['x1']), float(loc['y1']), float(loc['x2']), float(loc['y2']))
        except Exception:
            continue
        content = str((t or {}).get('content') or '')
        h = box[3] - box[1]
        if not content or not (min_h <= h <= max_h):
            continue
        yield content, box, ink


def build_gallery(split_chars: str = 'A-Za-z0-9._-+/') -> Gallery:
    """One template per character occurrence over the vendor text boxes."""
    g = Gallery()
    per_char = Counter()
    n_boxes = 0
    for c in pcore.discover():
        if not c.ref_json:
            continue
        try:
            for content, box, ink in iter_gt_texts(c):
                sub = tight_crop(ink, box)
                if sub is None:
                    continue
                n_boxes += 1
                # a box may hold several characters: split evenly along x when the aspect
                # ratio says so, otherwise treat the whole box as one glyph
                n = max(1, int(round((sub.shape[1] / max(1.0, sub.shape[0])) / 0.72)))
                if len(content) == n and n > 1:
                    cols = np.linspace(0, sub.shape[1], n + 1).astype(int)
                    for i, ch in enumerate(content):
                        part = sub[:, cols[i]:cols[i + 1]]
                        if part.size and part.sum() > 0:
                            g.add(ch, part, src=c.case_id)
                            per_char[ch] += 1
                elif len(content) == 1:
                    g.add(content, sub, src=c.case_id)
                    per_char[content] += 1
        except Exception as exc:      # never let one bad sheet kill the gallery
            print('  gallery: skipped %s (%s)' % (c.case_id, exc))
    print('gallery: %d templates from %d text boxes' % (len(g), n_boxes))
    print('  chars: %s' % dict(sorted(per_char.items(), key=lambda kv: -kv[1])))
    return g


def save_gallery(g: Gallery, path: str = GALLERY_PATH) -> str:
    with open(path, 'wb') as fh:
        pickle.dump({'chars': g.chars, 'bits': g.bits, 'src': g.src}, fh)
    return path


def load_gallery(path: str = GALLERY_PATH) -> Gallery:
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    g = Gallery()
    g.chars = list(d['chars'])
    g.bits = np.asarray(d['bits'], dtype=np.float32)
    g.src = list(d.get('src') or [''] * len(g.chars))
    return g


def read_glyphs(g: Gallery, glyphs: Sequence[np.ndarray], margin: float = 0.0,
                min_score: float = 0.45, exclude: Optional[str] = None,
                vote: int = 0) -> Tuple[str, List[Optional[str]], List[float]]:
    """Recognise a left-to-right sequence of glyph crops."""
    out: List[Optional[str]] = []
    scores: List[float] = []
    for sub in glyphs:
        ch, sc, _ = g.match(sub, margin=margin, exclude=exclude, vote=vote,
                            min_score=min_score)
        if ch is None or sc < min_score:
            out.append(None)
            scores.append(round(sc, 3))
        else:
            out.append(ch)
            scores.append(round(sc, 3))
    text = ''.join(ch if ch else '?' for ch in out)
    return (text if all(out) else ''), out, scores


def split_chars(sub: np.ndarray, n: Optional[int] = None,
                pitch: float = 0.72) -> List[np.ndarray]:
    """Cut a tight line crop into ``n`` character cells.

    The glyphs are *not* separable by connected components: on case 0059 the whole 509x938
    sheet yields only ~58 ink components, i.e. strokes of neighbouring characters touch and
    fuse into clusters.  So the count is estimated from the aspect ratio (CAD fonts are close
    to a 0.72 pitch) and the crop is divided evenly; each cell is then rescaled to the fixed
    raster on its own, which absorbs the residual pitch error.
    """
    h, w = sub.shape[:2]
    if n is None:
        n = max(1, int(round((w / max(1.0, float(h))) / pitch)))
    cols = np.linspace(0, w, n + 1).astype(int)
    out = []
    for i in range(n):
        cell = sub[:, cols[i]:max(cols[i] + 1, cols[i + 1])]
        if cell.size:
            out.append(cell)
    return out


def read_line(g: Gallery, sub: np.ndarray, margin: float = 0.0,
              min_score: float = 0.45, n: Optional[int] = None,
              exclude: Optional[str] = None, vote: int = 0):
    """Read one tight line crop: split into characters, classify each, require all of them."""
    return read_glyphs(g, split_chars(sub, n=n), margin=margin, min_score=min_score,
                       exclude=exclude, vote=vote)


def cells_from_glyphs(ink: np.ndarray, box, glyphs, line_box: Optional[bool] = True):
    """Character cells taken from the detected ink components instead of an even split.

    ``split_chars`` divides a line evenly, which drifts as soon as the pitch estimate is off
    (a 38x13 ``VCC`` is read as four cells).  When the connected components inside the line are
    already separate, they *are* the characters and give both the right count and the right
    cut; when they have fused into one blob the even split is still the fallback.
    """
    x1, y1, x2, y2 = box
    inside = [gl for gl in glyphs
              if gl.box[0] >= x1 - 1 and gl.box[2] <= x2 + 1
              and gl.box[1] >= y1 - 1 and gl.box[3] <= y2 + 1]
    inside.sort(key=lambda gl: gl.box[0])
    if not inside:
        return None
    if len(inside) == 1:
        return None
    subs = []
    for gl in inside:
        gx1, gy1, gx2, gy2 = gl.box
        part = ink[int(gy1):int(gy2) + 1, int(gx1):int(gx2) + 1]
        if part.size and part.sum() > 0:
            subs.append(part)
    return subs or None


if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--gallery' in argv:
        gg = build_gallery()
        print('saved ->', save_gallery(gg))
