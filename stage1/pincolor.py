"""IC pins from attached red stubs and their real green-wire transitions.

Coordinates are original-image pixels, with a top-left origin. This module
does not read GT, OCR, pin numbers or pin names, and never places a pin at a
fixed distance from its component box. A green no-connect cross is not a wire.
Only an actual long red body edge can anchor the search; unrelated red text
and short coloured strokes outside a detection box are not pin evidence.
"""

from __future__ import annotations


import math

from typing import Dict, Sequence


import numpy as np


_EDGES = ("top", "right", "bottom", "left")

_NORMALS = ((0, -1), (1, 0), (0, 1), (-1, 0))

FRAME_RADIUS = 6

MAX_STUB = 96

MIN_RED_HITS = 4

MIN_GREEN_RUN = 6

MAX_TRANSITION_GAP = 4


def _colour_masks(image):
    """Dominance rather than brightness preserves dark red/green and AA pixels."""
    blue, green, red = np.moveaxis(image.astype(np.int16), -1, 0)
    chroma = np.maximum.reduce((blue, green, red)) - np.minimum.reduce((blue, green, red))
    red_mask = ((red - green >= 18) & (red - blue >= 18)
                & (red >= 45) & (chroma >= 24))
    green_mask = ((green - red >= 18) & (green - blue >= 14)
                  & (green >= 40) & (chroma >= 24))
    return red_mask, green_mask


def _runs(values):
    a = np.asarray(values, dtype=np.int8)
    changes = np.diff(np.r_[0, a, 0])
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _round(value):
    # Round absolute raster coordinates once, avoiding ties-to-even row shifts.
    return int(math.floor(float(value) + 0.5))


def _frame(red, box, edge):
    """Bounded calibration to a substantially filled, long red frame side."""
    x0, y0, x1, y1 = box
    vertical = edge in (1, 3)
    predicted = (x1 if edge == 1 else x0) if vertical else (y1 if edge == 2 else y0)
    lo, hi = (y0, y1) if vertical else (x0, x1)
    trim = min(8.0, max(2.0, (hi - lo) * 0.06))
    h, w = red.shape
    lo = max(0, int(math.ceil(lo + trim)))
    hi = min(h - 1 if vertical else w - 1, int(math.floor(hi - trim)))
    if hi - lo < 8:
        return None
    limit = w - 1 if vertical else h - 1
    candidates = []
    for coord in range(max(0, int(math.floor(predicted - FRAME_RADIUS))),
                       min(limit, int(math.ceil(predicted + FRAME_RADIUS))) + 1):
        stroke = red[lo:hi + 1, coord] if vertical else red[coord, lo:hi + 1]
        coverage = float(np.mean(stroke))
        longest = max((end - start for start, end in _runs(stroke)), default=0)
        if coverage >= 0.60 and longest >= 0.40 * len(stroke):
            candidates.append((coord, coverage))
    if not candidates:
        return None
    best, coverage = max(candidates, key=lambda p: (p[1], -abs(p[0] - predicted)))
    eligible = {coord for coord, value in candidates if value >= coverage - 0.05}
    group = [best]
    for direction in (-1, 1):
        coord = best + direction
        while coord in eligible and len(group) < 7:
            group.append(coord)
            coord += direction
    return float(np.mean(group)), coverage


def _point(frame, tangent, edge, distance=0):
    if edge == 0:
        return tangent, frame - distance
    if edge == 1:
        return frame + distance, tangent
    if edge == 2:
        return tangent, frame + distance
    return frame - distance, tangent


def _pixel(mask, frame, tangent, edge, distance, lateral=0):
    x, y = _point(frame, tangent + lateral, edge, distance)
    ix, iy = _round(x), _round(y)
    h, w = mask.shape
    return bool(0 <= ix < w and 0 <= iy < h and mask[iy, ix])


def _red_path(red, frame, tangent, edge, max_distance):
    """Measure a straight attached red core, never a whole letter/line run's centre."""
    first = last = hits = gaps = current_gap = 0
    for distance in range(1, max_distance + 1):
        if _pixel(red, frame, tangent, edge, distance):
            if not first:
                first = distance
            last = distance
            hits += 1
            current_gap = 0
        else:
            current_gap += 1
            if current_gap > 1:
                break
            gaps += 1
    if first not in (1, 2) or hits < MIN_RED_HITS or last < MIN_RED_HITS:
        return None
    if last == max_distance and _pixel(red, frame, tangent, edge, last + 1):
        return None  # Search bound is not evidence of a real red endpoint.
    return {"first": first, "last": last, "hits": hits, "gaps": gaps,
            "coverage": hits / max(1, last - first + 1)}


def _green_cross(green, frame, tangent, edge, first):
    """Require four diagonal arms, so a bend or short wire is not called NC."""
    for normal_center in range(max(1, first - 2), first + 5):
        for tangent_offset in (-1, 0, 1):
            counts = []
            for normal_sign, tangent_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                count = sum(_pixel(green, frame, tangent + tangent_offset + tangent_sign * radius,
                                   edge, normal_center + normal_sign * radius)
                            for radius in (2, 3, 4, 5))
                counts.append(count)
            if min(counts) >= 2:
                return True
    return False


def _green_path(green, frame, tangent, edge, red_last):
    """A narrow, outward normal green run; a diagonal no-connect X fails it."""
    first = 0
    for distance in range(red_last + 1, red_last + MAX_TRANSITION_GAP + 2):
        if any(_pixel(green, frame, tangent, edge, distance, lateral)
               for lateral in (-1, 0, 1)):
            first = distance
            break
    if not first:
        return {"connected": False, "first": None, "last": None,
                "hits": 0, "span": 0, "marker_near_tip": False}
    last = hits = gap = 0
    for distance in range(first, first + 32):
        if any(_pixel(green, frame, tangent, edge, distance, lateral)
               for lateral in (-1, 0, 1)):
            hits += 1
            last = distance
            gap = 0
        else:
            gap += 1
            if gap > 1:
                break
    span = max(0, last - first + 1)
    # A thick anti-aliased X can contain a 6px horizontal central slice. Check
    # its four diagonal arms before accepting that slice as a wire. A long
    # continuation through the patch is still a wire, not an isolated X.
    marker = span <= 12 and _green_cross(green, frame, tangent, edge, first)
    connected = (not marker and hits >= MIN_GREEN_RUN
                 and hits / max(1, span) >= 0.80)
    # A short green stroke can turn immediately or end at a junction. It is
    # insufficient evidence of either a straight wire or a disconnected pin.
    state = True if connected else False if marker else None
    return {"connected": state, "first": first, "last": last,
            "hits": hits, "span": span, "marker_near_tip": bool(marker)}


def _component_pins(red, green, box, frame_report=None):
    x0, y0, x1, y1 = box
    h, w = red.shape
    records = []
    max_distance = MAX_STUB
    frames = [_frame(red, box, edge) for edge in range(4)]
    confirmed_edges = [_EDGES[i] for i, value in enumerate(frames) if value is not None]
    if frame_report is not None:
        frame_report.update({
            "confirmed_edges": confirmed_edges,
            "all_edges_confirmed": len(confirmed_edges) == 4,
            "sides": {_EDGES[i]: {"confirmed": value is not None,
                                    "coord": round(value[0], 1) if value is not None else None,
                                    "coverage": round(value[1], 3) if value is not None else 0.0}
                      for i, value in enumerate(frames)},
        })
    for edge in range(4):
        calibrated = frames[edge]
        if calibrated is None:
            continue
        frame, frame_coverage = calibrated
        lo, hi = (y0, y1) if edge in (1, 3) else (x0, x1)
        limit = h - 1 if edge in (1, 3) else w - 1
        # The frame's two perpendicular corners are not outward pin strokes.
        lo = max(0, int(math.ceil(lo + 2)))
        hi = min(limit, int(math.floor(hi - 2)))
        paths = []
        for tangent in range(lo, hi + 1):
            path = _red_path(red, frame, tangent, edge, max_distance)
            paths.append(path)
        for start, end in _runs([path is not None for path in paths]):
            # Thick strokes are one pin. A broad coloured area is not a stub.
            if end - start > 7:
                continue
            group = [(lo + index, paths[index]) for index in range(start, end)]
            best_length = max(path["last"] for _, path in group)
            # AA edge rows may be shorter. Locate the longest straight core,
            # rather than shifting toward nearby coloured character pixels.
            central = [(tangent, path) for tangent, path in group
                       if path["last"] >= best_length - 1]
            tangent = float(np.mean([t for t, _ in central]))
            path = max((p for _, p in central), key=lambda p: (p["hits"], p["last"]))
            red_last = path["last"]
            green_path = _green_path(green, frame, tangent, edge, red_last)
            connected = green_path["connected"]
            # Endpoint is measured from colour pixels, not from a bbox offset.
            distance = ((red_last + green_path["first"]) / 2.0
                        if connected is not False else float(red_last))
            point = _point(frame, tangent, edge, distance)
            red_tip = _point(frame, tangent, edge, red_last)
            green_start = (_point(frame, tangent, edge, green_path["first"])
                           if green_path["first"] is not None else None)
            records.append({
                "point_img": [round(float(v), 1) for v in point],
                "edge": _EDGES[edge], "connected": connected,
                "confidence": 0.96 if connected is True else 0.80 if connected is False else 0.65,
                "source": ("red_green_transition" if connected is True
                           else "red_stub_endpoint" if connected is False
                           else "red_green_transition_uncertain"),
                "trust": "trusted",
                "anchor_img": [round(float(v), 1) for v in _point(frame, tangent, edge)],
                "frame_confirmed": True, "frame_coverage": round(frame_coverage, 3),
                "confirmed_edges": confirmed_edges,
                "normal_ink_run": int(red_last), "normal_ink_hits": int(path["hits"]),
                "normal_img": list(_NORMALS[edge]),
                "red_tip_img": [round(float(v), 1) for v in red_tip],
                "green_start_img": ([round(float(v), 1) for v in green_start]
                                    if green_start is not None else None),
                "connection_reason": ("outward_straight_green_wire" if connected is True
                                      else "green_marker_without_wire" if green_path["marker_near_tip"]
                                      else "short_green_stroke_connection_uncertain" if connected is None
                                      else "no_outward_green_wire"),
                "color_evidence": {
                    "red_hits": int(path["hits"]), "red_span": int(red_last),
                    "red_coverage": round(path["coverage"], 3),
                    "stroke_width": int(end - start),
                    "green_hits": int(green_path["hits"]),
                    "green_span": int(green_path["span"]),
                    "transition_gap": (int(green_path["first"] - red_last - 1)
                                       if green_path["first"] is not None else None),
                    "green_marker_near_tip": bool(green_path["marker_near_tip"]),
                },
            })
    return records


def extract_color_pins(image: np.ndarray, components: Sequence[dict], *,
                       frame_report: dict | None = None):
    """Return measured colour pins per box IC, without names/numbers or GT.

    Each record has point_img, edge, connected, confidence, source, anchor_img,
    and explicit red/green image evidence. Unconnected red stubs are retained
    at their actual red endpoints. Empty lists mean no confirmed coloured pin
    evidence; callers should keep their existing non-colour fallback in that
    case. bbox_img must use the same top-left original-image coordinates.
    connected=None denotes nearby green ink that might turn immediately: it
    must not be used to exclude that pin from nets. Optional frame_report is
    filled per component even when no red stubs were found, so callers can
    distinguish unconfirmed sides from a confirmed side with no pins.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return {}
    red, green = _colour_masks(image)
    out = {}
    for component in components or ():
        if str(component.get("type") or "").lower().split(":", 1)[0] != "box":
            continue
        raw = component.get("bbox_img")
        try:
            box = tuple(float(v) for v in raw)
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                continue
            if box[2] - box[0] < 12 or box[3] - box[1] < 12:
                continue
        except (TypeError, ValueError):
            continue
        cid = str(component.get("id"))
        side_report = {} if frame_report is not None else None
        out[cid] = _component_pins(red, green, box, side_report)
        if frame_report is not None:
            frame_report[cid] = side_report
    return out
