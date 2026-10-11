"""Conservative image evidence for the final, merged IC pin candidates.

The evidence builder ranks candidates; it never moves or removes an output pin. A short
unconnected pin is valid. OCR boxes are soft evidence, not an erasure mask: they
often include a real pin number or a wire crossing through text.

The separate recovery function can add measured monochrome stroke rows or
calibrate an existing candidate inside its own stroke band. It retains every
old index and does not infer an electrical endpoint from monochrome ink.
"""
from __future__ import annotations

import math
import copy
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


def _one_pin(ink, point, box, frames, text_boxes, text_mask, *, search_radius=3):
    edge = _candidate_edge(point, box)
    normal = _NORMALS[edge]
    vertical = edge in (1, 3)
    frame_coord, calibrated, frame_coverage = frames[edge]
    tangent = point[1] if vertical else point[0]
    paths = []
    # The merged candidate may sit at the centre of several pixels/number ink.
    # Search at most 3px along its own side; never migrate to another pin row.
    for offset in range(-search_radius, search_radius + 1):
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


def _black_ink(bgr, ink=None):
    """Use original, low-chroma pixels even if a caller's ink includes color."""
    pixels = np.asarray(bgr)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("frame recovery requires the original BGR image")
    if (not np.issubdtype(pixels.dtype, np.number)
            or (np.issubdtype(pixels.dtype, np.inexact) and not np.isfinite(pixels).all())
            or np.any(pixels < 0) or np.any(pixels > 255)):
        raise ValueError("frame recovery requires finite image pixels")
    # Keep large schematic images at their original 8-bit storage. Casting the
    # entire BGR image to float64 costs eight times its memory unnecessarily.
    chroma = pixels.max(axis=2).astype(np.float32) - pixels.min(axis=2)
    gray = pixels[:, :, 0].astype(np.float32) * np.float32(0.114)
    gray += pixels[:, :, 1] * np.float32(0.587)
    gray += pixels[:, :, 2] * np.float32(0.299)
    if ink is None:
        supplied = gray < 170
    else:
        supplied = np.asarray(ink, dtype=bool)
        if supplied.shape != pixels.shape[:2]:
            raise ValueError("frame recovery ink must have the image's (H, W) shape")
    return supplied & (gray < 170) & (chroma < 30), (gray < 230) & (chroma >= 30)


def _text_mask(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for x0, y0, x1, y1 in boxes:
        xa, ya = max(0, int(math.ceil(x0))), max(0, int(math.ceil(y0)))
        xb, yb = min(w, int(math.floor(x1)) + 1), min(h, int(math.floor(y1)) + 1)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def _normal_row(ink, edge, normal_coord, tangents):
    h, w = ink.shape
    vertical = edge in (1, 3)
    limit = w if vertical else h
    if not 0 <= normal_coord < limit:
        return np.zeros(len(tangents), dtype=bool)
    return (ink[tangents, normal_coord] if vertical
            else ink[normal_coord, tangents])


def _frame_stroke_bands(ink, box, edge, frame):
    """Find thin, uninterrupted strokes beginning at a measured frame edge.

    This scans actual raster rows, rather than clustering line detector points.
    A blank row splits bands regardless of their spacing. Requiring six pixels
    *past the outer frame thickness* prevents the frame itself from qualifying.
    Detached glyph shafts and scan gaps are deliberately not recovery evidence.
    """
    coord, confirmed, coverage = frame
    if not confirmed:
        return []
    vertical = edge in (1, 3)
    lo, hi = (box[1], box[3]) if vertical else (box[0], box[2])
    # Corner continuations are not pins. This is geometric trimming, not a
    # learned pitch or an assumed pin grid.
    trim = max(3.0, min(8.0, (hi - lo) * 0.04))
    tangents = np.arange(int(math.ceil(lo + trim)), int(math.floor(hi - trim)) + 1)
    limit = ink.shape[0] if vertical else ink.shape[1]
    tangents = tangents[(tangents >= 0) & (tangents < limit)]
    if len(tangents) < 8:
        return []
    sign = int(_NORMALS[edge][0] if vertical else _NORMALS[edge][1])
    center = int(math.floor(coord + 0.5))
    outer = center
    # The calibrated center can lie inside a thick frame. Only the long side
    # itself (coverage >= .65) may extend this boundary, never an individual pin.
    for depth in range(1, 5):
        next_coord = center + sign * depth
        if np.mean(_normal_row(ink, edge, next_coord, tangents)) < 0.65:
            break
        outer = next_coord
    paths = np.stack([_normal_row(ink, edge, outer + sign * depth, tangents)
                      for depth in range(1, 7)], axis=1)
    attached = np.all(paths, axis=1)
    changes = np.diff(np.r_[False, attached, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    bands = []
    for start, end in zip(starts, ends):
        band = tangents[start:end]
        # A broad paragraph or parallel outside bar is not a thin normal line.
        if not 1 <= len(band) <= 7:
            continue
        tangent = float((band[0] + band[-1]) / 2.0)
        anchor = [coord, tangent] if vertical else [tangent, coord]
        bands.append(dict(edge=_EDGES[edge], edge_index=edge,
                          tangent_band=[int(band[0]), int(band[-1])],
                          tangent_center=tangent, anchor_img=anchor,
                          frame_outer_img=float(outer), frame_coverage=float(coverage),
                          continuous_outward_pixels=6, stroke_width=int(len(band))))
    return bands


def recover_frame_pins(components: Sequence[dict], comp_pins: Dict[str, list],
                       bgr: np.ndarray, *, ink=None, texts=None,
                       skip_sides=None, out_dist=12.0):
    """Return (pins, old-index mapping, audit) from measured black frame strokes.

    ``skip_sides`` maps component ids to side names; those sides are never
    scanned or calibrated. Old points and indices are retained. New points are
    appended, and only a unique old candidate within 3px of the same measured
    stroke can be calibrated. ``out_dist`` is an endpoint *estimate*, not proof
    of a monochrome pin/wire boundary. No text, number, pitch, or GT prior is
    used to create a row. Every mutation has its original pixel measurements.
    """
    if not math.isfinite(float(out_dist)) or float(out_dist) <= 0:
        raise ValueError("frame recovery out_dist must be finite and positive")
    out_dist = float(out_dist)
    mask, colored = _black_ink(bgr, ink)
    boxes = _text_boxes(texts)
    text_mask = _text_mask(mask.shape, boxes)
    output = copy.deepcopy(comp_pins)
    index_map = {cid: {i: i for i in range(len(points))}
                 for cid, points in comp_pins.items()}
    audit = {}
    skip_sides = skip_sides or {}
    h, w = mask.shape
    for component in components:
        ctype = str(component.get("type") or "").lower()
        if ctype not in {"box", "bigic"}:
            continue
        cid, raw_box = component.get("id"), component.get("bbox_img")
        if cid not in output or raw_box is None:
            continue
        try:
            box = tuple(float(v) for v in raw_box)
        except (TypeError, ValueError):
            continue
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] - box[0] < 12 or box[3] - box[1] < 12
                or box[0] < 0 or box[1] < 0 or box[2] >= w or box[3] >= h):
            continue
        skipped = set(skip_sides.get(cid) or ())
        frames = [_frame_edge(mask, box, edge) for edge in range(4)]
        log = dict(added=[], calibrated=[], held=[], skipped_sides=sorted(skipped, key=str),
                   endpoint_estimate_pixels=out_dist, old_count=len(comp_pins[cid]))
        audit[cid] = log
        # The CV fallback proposes any large closed rectangle, including
        # document tables. Normal attached lines alone cannot establish that
        # this rectangle is an IC. Preserve its existing flow, but do not
        # expand a shape-only hypothesis without semantic confirmation.
        if component.get("source") == "cv:bigic":
            log["held"].append(dict(reason="shape_only_component_without_semantic_confirmation",
                                    component_source="cv:bigic"))
            log["new_count"] = len(output[cid])
            continue
        claimed_old = set()
        previous_evidence = []
        for old in comp_pins[cid]:
            try:
                valid = len(old) == 2 and all(math.isfinite(float(v)) for v in old)
                previous_evidence.append(_one_pin(mask, old, box, frames, boxes, text_mask)
                                         if valid else None)
            except (TypeError, ValueError):
                previous_evidence.append(None)
        for edge, name in enumerate(_EDGES):
            if name in skipped or edge in skipped:
                continue
            vertical = edge in (1, 3)
            for band in _frame_stroke_bands(mask, box, edge, frames[edge]):
                ax, ay = band["anchor_img"]
                nx, ny = _NORMALS[edge]
                point = [round(ax + nx * out_dist, 1), round(ay + ny * out_dist, 1)]
                # Unlike a noisy old candidate, a measured row must prove its
                # own stroke. Searching +/-3 here can borrow a neighboring pin
                # and accidentally promote a touching glyph shaft.
                evidence = _one_pin(mask, point, box, frames, boxes, text_mask,
                                    search_radius=0)
                details = dict(band, evidence=evidence, point_img=point,
                               estimated_endpoint=True)
                if evidence["trust"] != "trusted":
                    log["held"].append(dict(details, reason="recovery_stroke_not_trusted"))
                    continue
                near, exact = [], []
                for i, old in enumerate(comp_pins[cid]):
                    try:
                        if (len(old) != 2 or not all(math.isfinite(float(v)) for v in old)
                                or _candidate_edge(old, box) != edge):
                            continue
                        old_tangent = float(old[1] if vertical else old[0])
                        old_normal = float(old[0] if vertical else old[1])
                    except (TypeError, ValueError):
                        continue
                    # A point on the frame or inside the IC is not an outside
                    # endpoint candidate and must not be migrated to a pin.
                    if (old_normal - frames[edge][0]) * (nx if vertical else ny) < 1:
                        continue
                    lo, hi = band["tangent_band"]
                    if lo - 0.6 <= old_tangent <= hi + 0.6:
                        exact.append(i)
                    if (lo - 3 <= old_tangent <= hi + 3
                            and abs(old_normal - point[0 if vertical else 1]) <= 7):
                        near.append(i)
                # Multiple old candidates near one physical row are ambiguous.
                # Keep their identities instead of moving or merging them.
                if len(near) > 1:
                    log["held"].append(dict(details, reason="multiple_old_candidates_on_stroke",
                                            old_indices=near))
                    continue
                if near:
                    index = near[0]
                    old = comp_pins[cid][index]
                    if (previous_evidence[index] is None
                            or previous_evidence[index]["trust"] != "trusted"):
                        log["held"].append(dict(details, reason="old_candidate_not_trusted",
                                                old_indices=[index],
                                                previous_evidence=previous_evidence[index]))
                        continue
                    px, py = (int(math.floor(float(v) + 0.5)) for v in old)
                    # Never alter an existing colored contact, even if a caller
                    # omitted a side from its color ledger.
                    local_color = colored[max(0, py - 2):min(h, py + 3),
                                          max(0, px - 2):min(w, px + 3)]
                    if np.any(local_color) or index in claimed_old:
                        log["held"].append(dict(details, reason="protected_existing_candidate",
                                                old_indices=[index]))
                        continue
                    claimed_old.add(index)
                    if max(abs(float(old[k]) - point[k]) for k in (0, 1)) > 0.05:
                        output[cid][index] = point
                        log["calibrated"].append(dict(details, index=index,
                                                       previous_point_img=list(old)))
                    continue
                if exact:
                    # A spatially valid old point with a different normal
                    # estimate already represents this row; do not duplicate it.
                    log["held"].append(dict(details, reason="row_already_represented",
                                            old_indices=exact))
                    continue
                index = len(output[cid])
                output[cid].append(point)
                log["added"].append(dict(details, index=index))
        log["new_count"] = len(output[cid])
    return output, index_map, audit
