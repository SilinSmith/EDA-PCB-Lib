"""Conservative image evidence for the final, merged IC pin candidates.

This module ranks candidates; it never moves or removes an output pin. A short
unconnected pin is valid. OCR boxes are soft evidence, not an erasure mask: they
often include a real pin number or a wire crossing through text.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


_EDGES = ("top", "right", "bottom", "left")
_NORMALS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _text_boxes(texts):
    boxes = []
    for text in texts or ():
        box = text.get("bbox_img") if isinstance(text, dict) else text
        try:
            if box is not None and len(box) == 4:
                vals = tuple(float(v) for v in box)
                if all(math.isfinite(v) for v in vals):
                    boxes.append(vals)
        except (TypeError, ValueError):
            pass
    return boxes


def _frame_edge(ink, box, edge, radius=6):
    """Find a nearby long frame stroke, without trusting nearby character ink.

    A detector box can be a few pixels off the actual frame. Only a long,
    substantially filled side can calibrate its location, and movement is
    bounded. This anchor is for evidence, not the emitted pin coordinate.
    """
    x0, y0, x1, y1 = box
    vertical = edge in (1, 3)
    normal = (x1 if edge == 1 else x0) if vertical else (y1 if edge == 2 else y0)
    lo, hi = (y0, y1) if vertical else (x0, x1)
    trim = min(8.0, max(2.0, (hi - lo) * 0.08))
    lo, hi = int(math.ceil(lo + trim)), int(math.floor(hi - trim))
    h, w = ink.shape
    lo, hi = max(0, lo), min(h - 1 if vertical else w - 1, hi)
    if hi - lo < 8:
        return float(normal), False, 0.0
    candidates = []
    limit = w - 1 if vertical else h - 1
    for coord in range(max(0, int(math.floor(normal - radius))),
                       min(limit, int(math.ceil(normal + radius))) + 1):
        row = ink[lo:hi + 1, coord] if vertical else ink[coord, lo:hi + 1]
        coverage = float(np.mean(row))
        if coverage >= 0.65 and _longest_run(row) >= 0.45 * len(row):
            candidates.append((coord, coverage))
    if not candidates:
        return float(normal), False, 0.0
    # Pick a contiguous frame thickness near the strongest, nearest side. Do
    # not average a second unrelated line or nearby underlined model text.
    best_coord, best_coverage = max(candidates, key=lambda q: (q[1], -abs(q[0] - normal)))
    group = [best_coord]
    eligible = {q[0] for q in candidates if q[1] >= best_coverage - 0.05}
    for direction in (-1, 1):
        coord = best_coord + direction
        while coord in eligible and len(group) < 7:
            group.append(coord)
            coord += direction
    return float(np.mean(group)), True, best_coverage


def _candidate_edge(point, box):
    x, y = point
    x0, y0, x1, y1 = box
    # Distance to the finite side, so corner candidates do not project across
    # the whole rectangle. Ties use the established top/right/bottom/left order.
    anchors = ((min(x1, max(x0, x)), y0), (x1, min(y1, max(y0, y))),
               (min(x1, max(x0, x)), y1), (x0, min(y1, max(y0, y))))
    return min(range(4), key=lambda i: math.hypot(x - anchors[i][0], y - anchors[i][1]))


def _probe_path(ink, anchor, normal, text_mask, max_distance=22):
    """Sample normal ink, tolerating a small scan gap but measuring it separately."""
    h, w = ink.shape
    nx, ny = normal
    tx, ty = -ny, nx
    # Round once: repeatedly rounding .5 coordinates uses ties-to-even and
    # would duplicate every other raster row, hiding a real 2px gap.
    ax, ay = (int(math.floor(v + 0.5)) for v in anchor)
    rows = []
    for distance in range(1, max_distance + 1):
        cx = ax + nx * distance
        cy = ay + ny * distance
        if not (0 <= cx < w and 0 <= cy < h):
            break
        # +/-1px avoids turning nearby letters into a broad search corridor.
        central, lateral = [], []
        for offset in range(-5, 6):
            xx, yy = int(round(cx + tx * offset)), int(round(cy + ty * offset))
            hit = 0 <= xx < w and 0 <= yy < h and bool(ink[yy, xx])
            if abs(offset) <= 1 and hit:
                central.append((xx, yy))
            if hit:
                lateral.append(offset)
        text_hit = any(text_mask[yy, xx] for xx, yy in central)
        rows.append((distance, bool(central), text_hit, len(lateral)))
    hit_rows = [row for row in rows if row[1]]
    first = hit_rows[0][0] if hit_rows else None
    attached, n_hit, gap, longest_gap = 0, 0, 0, 0
    last_hit = 0
    for distance, hit, _text, _width in rows:
        if hit:
            n_hit += 1
            last_hit = distance
            gap = 0
        else:
            gap += 1
            longest_gap = max(longest_gap, gap)
            if gap > 2:
                break
        attached = last_hit
    near = [row for row in rows if row[0] <= 14]
    near_hits = [row for row in near if row[1]]
    text_fraction = sum(row[2] for row in near_hits) / max(1, len(near_hits))
    branching = sum(row[3] >= 6 for row in near_hits) / max(1, len(near_hits))
    return {"first_ink": first, "run": attached, "n_hit": n_hit,
            "gap": longest_gap, "text_fraction": text_fraction,
            "branching": branching, "near_hits": len(near_hits)}


def _word_ink_supported(ink, box, edge):
    """A broad OCR rectangle alone is not proof that a short stroke is text.

    Require several separated glyph-sized runs in the *actual* ink projected
    along this side. A real stub under an oversized OCR box is one run. The
    normal coverage requirement excludes isolated scan speckles without making
    a font-size-specific area threshold. This remains soft evidence for an
    attached line: it is only used by the detached-word rejection branch.
    """
    x0, y0, x1, y1 = box
    h, w = ink.shape
    xa, ya = max(0, int(math.ceil(x0))), max(0, int(math.ceil(y0)))
    xb, yb = min(w, int(math.floor(x1)) + 1), min(h, int(math.floor(y1)) + 1)
    if xb <= xa or yb <= ya:
        return False
    band = ink[ya:yb, xa:xb]
    vertical = edge in (1, 3)
    if vertical:
        band = band.T  # rows normal to the side, columns along the side
    occupied = np.any(band, axis=0)
    changes = np.diff(np.r_[0, occupied.astype(np.int8), 0])
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    normal_span = (x1 - x0) if vertical else (y1 - y0)
    minimum_coverage = max(3.0, 0.5 * normal_span)
    glyph_runs = sum(np.count_nonzero(np.any(band[:, start:end], axis=1))
                     >= minimum_coverage for start, end in zip(starts, ends))
    return bool(glyph_runs >= 3)


def _external_text_band(point, frame_coord, edge, text_boxes, ink):
    """A word-like OCR band outside this side, containing this candidate.

    Return the word's normal span and whether actual multi-glyph ink supports
    the OCR rectangle. A single number or I-shaped glyph, even inside a falsely
    broad OCR box, is deliberately not enough to reject. The normal stroke
    must also be detached and terminate within the word.
    """
    x, y = point
    for x0, y0, x1, y1 in text_boxes:
        if not (x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1):
            continue
        if edge in (1, 3):
            tangent_span, normal_span = y1 - y0, x1 - x0
            outside = x0 > frame_coord if edge == 1 else x1 < frame_coord
        else:
            tangent_span, normal_span = x1 - x0, y1 - y0
            outside = y0 > frame_coord if edge == 2 else y1 < frame_coord
        if outside and tangent_span >= max(16.0, 2.5 * normal_span):
            return normal_span, _word_ink_supported(ink, (x0, y0, x1, y1), edge)
    return None, False


def _one_pin(ink, point, box, frames, text_boxes, text_mask):
    edge = _candidate_edge(point, box)
    normal = _NORMALS[edge]
    vertical = edge in (1, 3)
    frame_coord, calibrated, frame_coverage = frames[edge]
    tangent = point[1] if vertical else point[0]
    paths = []
    # The merged candidate may sit at the centre of several pixels/number ink.
    # Search at most 3px along its own side; never migrate to another pin row.
    for offset in range(-3, 4):
        anchor = ((frame_coord, tangent + offset) if vertical
                  else (tangent + offset, frame_coord))
        path = _probe_path(ink, anchor, normal, text_mask)
        paths.append((anchor, path, offset))
    anchor, path, _ = max(paths, key=lambda q: (q[1]["n_hit"], q[1]["run"],
                                               -q[1]["branching"], -abs(q[2])))
    run, hits = path["run"], path["n_hit"]
    text_like = path["text_fraction"] >= 0.6 and path["branching"] >= 0.3
    near_attached = path["first_ink"] is not None and path["first_ink"] <= 3
    word_span, word_ink = _external_text_band(point, frame_coord, edge, text_boxes, ink)
    detached_word = (word_span is not None and word_ink and path["text_fraction"] >= 0.6
                     and path["first_ink"] is not None and path["first_ink"] >= 3
                     and run <= word_span + path["first_ink"] + 2)
    only_text_support = path["text_fraction"] >= 0.5 and run <= 14
    trust, confidence, reason = "uncertain", 0.45, "insufficient_normal_stroke_evidence"
    # A real short floating stub is enough. No external wire is required.
    # Text overlap alone cannot demote an uninterrupted straight attached stub.
    if calibrated and detached_word:
        trust, confidence, reason = "rejected", 0.1, "detached_word_ink_outside_frame"
    elif (calibrated and near_attached and hits >= 4 and run >= 5
          and not text_like and not only_text_support):
        trust, confidence, reason = "trusted", 0.88, "frame_attached_normal_stroke"
        if path["gap"] > 0:
            confidence, reason = 0.76, "frame_normal_stroke_with_small_gap"
    elif calibrated and text_like and (not near_attached or path["first_ink"] >= 4):
        trust, confidence, reason = "rejected", 0.12, "detached_character_ink_outside_frame"
    elif not calibrated:
        reason = "actual_frame_unconfirmed"
    elif text_like:
        confidence, reason = 0.3, "text_overlap_without_clean_normal_stroke"
    return {"point_img": [round(float(v), 1) for v in point], "edge": _EDGES[edge],
            "trust": trust, "confidence": confidence, "reason": reason,
            "anchor_img": [round(float(v), 1) for v in anchor],
            "frame_confirmed": calibrated,
            "frame_coverage": round(frame_coverage, 3),
            "normal_ink_run": run, "normal_ink_hits": hits,
            "first_ink_distance": path["first_ink"],
            "text_overlap_fraction": round(path["text_fraction"], 3),
            "branching_fraction": round(path["branching"], 3),
            "external_word_band": word_span is not None,
            "external_word_ink_confirmed": word_ink}


def build_pin_evidence(components: Sequence[dict], comp_pins: Dict[str, list],
                       ink: np.ndarray, texts: Sequence = None) -> dict:
    """Return records aligned with final merged pins of box/bigic components."""
    mask = np.asarray(ink, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("pin evidence requires a two-dimensional ink mask")
    boxes = _text_boxes(texts)
    # OCR overlap is sampled repeatedly; one image mask makes each lookup O(1)
    # without erasing any original ink. Rounding preserves inclusive bounds.
    text_mask = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for x0, y0, x1, y1 in boxes:
        xa, ya = max(0, int(math.ceil(x0))), max(0, int(math.ceil(y0)))
        xb, yb = min(w, int(math.floor(x1)) + 1), min(h, int(math.floor(y1)) + 1)
        if xb > xa and yb > ya:
            text_mask[ya:yb, xa:xb] = True
    result = {}
    for component in components:
        ctype = str(component.get("type") or "").lower()
        if "box" not in ctype and "bigic" not in ctype:
            continue
        cid, box = component.get("id"), component.get("bbox_img")
        if cid not in comp_pins or not box or len(box) != 4:
            continue
        box = tuple(float(v) for v in box)
        if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            continue
        frames = [_frame_edge(mask, box, edge) for edge in range(4)]
        result[cid] = [_one_pin(mask, point, box, frames, boxes, text_mask)
                       for point in comp_pins[cid]]
    return result
