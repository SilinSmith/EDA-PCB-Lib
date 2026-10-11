"""Complete horizontal pin text from measured monochrome frame/stroke evidence.

The frame anchor and same-side physical rows define the search cells. OCR text
does not define a crop, and no numbering order or name dictionary is consulted.
The source image stays unchanged. Recognition copies may exclude measured long
wire pixels and their raster fringes; ambiguous glyph/line contact stays held.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .structmatch import trusted_frame_pin_evidence


def _finite(value, size):
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if len(result) == size and all(map(math.isfinite, result)) else None


def mono_observations(doc):
    """Yield only current, calibrated, measured left/right pin observations."""
    for comp in doc.get("components") or []:
        if not isinstance(comp, dict) or comp.get("type") != "box":
            continue
        cid, frame = comp.get("id"), _finite(comp.get("bbox_img"), 4)
        if not isinstance(cid, str) or frame is None or frame[2] <= frame[0] or frame[3] <= frame[1]:
            continue
        pins = (doc.get("pins") or {}).get(cid) or []
        evidence = (doc.get("pin_evidence") or {}).get(cid) or []
        for index, (raw, ev) in enumerate(zip(pins, evidence)):
            point = _finite(raw, 2)
            if (point is None or not isinstance(ev, dict) or ev.get("edge") not in ("left", "right")
                    or not trusted_frame_pin_evidence(ev, frame, point)):
                continue
            anchor = _finite(ev.get("anchor_img"), 2)
            sign = -1 if ev["edge"] == "left" else 1
            if anchor is None or (point[0] - anchor[0]) * sign <= 0:
                continue
            yield {"component_id": cid, "pin_index": index, "edge": ev["edge"],
                   "point_img": point, "anchor_img": anchor, "frame": frame,
                   "evidence": dict(ev)}


def _clip(region, shape):
    height, width = shape
    x0, y0, x1, y1 = region
    box = [max(0, int(math.ceil(x0))), max(0, int(math.ceil(y0))),
           min(width, int(math.floor(x1))), min(height, int(math.floor(y1)))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _groups(on, gap):
    locations = np.flatnonzero(on)
    if not len(locations):
        return []
    spans, start, last = [], int(locations[0]), int(locations[0])
    for location in locations[1:]:
        location = int(location)
        if location - last > gap + 1:
            spans.append((start, last + 1))
            start = location
        last = location
    spans.append((start, last + 1))
    return spans


def _pitch(observation, peers):
    row = observation["anchor_img"][1]
    distances = [abs(p["anchor_img"][1] - row) for p in peers
                 if abs(p["anchor_img"][1] - row) > 1]
    return min(distances) if distances else min(44.0, observation["frame"][3] - observation["frame"][1])


def _stroke_top(mask, observation):
    """Measure the attached stroke near the frame, away from external labels.

    Only a short normal probe between the current point and frame is tested.
    A dense continuous run through this probe must include the anchor row.
    The result sets the search cell. Later cleanup needs independent measured
    line evidence, and never reconstructs a numeral or a missing stroke.
    """
    point, anchor, edge = (observation[k] for k in ("point_img", "anchor_img", "edge"))
    length = abs(point[0] - anchor[0])
    reach = min(9.0, max(4.0, length * .65))
    left, right = ((anchor[0] - reach, anchor[0] - 2) if edge == "left"
                   else (anchor[0] + 2, anchor[0] + reach))
    probe = _clip([left, anchor[1] - 4, right, anchor[1] + 5], mask.shape)
    if probe is None or probe[2] - probe[0] < 2:
        return None
    x0, y0, x1, y1 = probe
    dense = np.mean(mask[y0:y1, x0:x1], axis=1) >= .85
    runs = _groups(dense, 0)
    matches = [(a, b) for a, b in runs if y0 + a - 1 <= anchor[1] <= y0 + b + 1]
    if len(matches) != 1:
        # A port triangle or clock marker can occupy the near-frame probe.
        # Try a straight segment immediately beyond the measured point, then
        # require its CURRENT ink to connect to the current frame-side ink.
        sign = -1 if edge == "left" else 1
        bounds = sorted((point[0] + sign * 3, point[0] + sign * 9))
        other = _clip([bounds[0], point[1] - 4, bounds[1], point[1] + 5], mask.shape)
        if other is None:
            return None
        xa, ya, xb, yb = other
        occupancy = np.mean(mask[ya:yb, xa:xb], axis=1)
        possible = [(ya + a, ya + b) for a, b in _groups(occupancy >= .85, 0)
                    if ya + a - 1 <= point[1] <= ya + b + 1 and b - a <= 6]
        if len(possible) != 1:
            return None
        top, bottom = possible[0]
        connection = _clip([min(xa, anchor[0] - 2), min(top, anchor[1]) - 6,
                            max(xb, anchor[0] + 2), max(bottom, anchor[1]) + 6], mask.shape)
        cx0, cy0, cx1, cy1 = connection
        _, labels = cv2.connectedComponents(mask[cy0:cy1, cx0:cx1].astype(np.uint8), 8)
        seeds = set(np.unique(labels[top-cy0:bottom-cy0, xa-cx0:xb-cx0]).tolist()) - {0}
        frame_x = min(cx1 - 1, max(cx0, int(round(anchor[0]))))
        near_frame = set(np.unique(labels[max(0, int(anchor[1])-3-cy0):min(cy1-cy0, int(anchor[1])+4-cy0),
                                             frame_x-cx0]).tolist()) - {0}
        if not seeds.intersection(near_frame):
            return None
        return {"probe": other, "stroke_top": top, "stroke_bottom": bottom,
                "row_occupancy": occupancy.tolist(),
                "method": "outer_measured_stroke_with_current_frame_connectivity",
                "connection_region": connection}
    a, b = matches[0]
    if b - a > 6:
        return None
    return {"probe": probe, "stroke_top": y0 + a, "stroke_bottom": y0 + b,
            "row_occupancy": np.mean(mask[y0:y1, x0:x1], axis=1).tolist(),
            "method": "near_frame_normal_probe_no_pixel_deletion"}


def _regions(observation, peers, mask):
    anchor, frame, edge = (observation[k] for k in ("anchor_img", "frame", "edge"))
    pitch = _pitch(observation, peers)
    row = anchor[1]
    above = [p["anchor_img"][1] for p in peers if p["anchor_img"][1] < row - 1]
    below = [p["anchor_img"][1] for p in peers if p["anchor_img"][1] > row + 1]
    top = max(frame[1] + 2, row - min(24.0, pitch * .48))
    bottom = min(frame[3] - 2, row + min(24.0, pitch * .48))
    if above:
        top = max(top, (row + max(above)) / 2 + .5)
    if below:
        bottom = min(bottom, (row + min(below)) / 2 - .5)
    inside = ([anchor[0] + 2, top, frame[2] - 2, bottom] if edge == "left"
              else [frame[0] + 2, top, anchor[0] - 2, bottom])
    # Numbers occupy the space ABOVE their stroke, rather than the name's
    # symmetric row cell. A three-digit word can extend well beyond the pin
    # candidate point; measured pitch sets an independent external width.
    stroke = _stroke_top(mask, observation)
    outside = None
    if stroke is not None:
        depth = min(96.0, max(pitch * 2.5, abs(anchor[0] - observation["point_img"][0]) + pitch))
        y0 = row - min(44.0, pitch * .98)
        if above:
            y0 = max(y0, max(above) + 4)
        # Keep the last numeral row. Raster fringes are audited from actual
        # long wire runs below, instead of deleting a fixed pixel from 1/4/6.
        stroke["upper_fringe_guard_pixels"] = 0
        outside = ([anchor[0] - depth, y0, anchor[0] - 2, stroke["stroke_top"]]
                   if edge == "left" else
                   [anchor[0] + 2, y0, anchor[0] + depth, stroke["stroke_top"]])
    return _clip(inside, mask.shape), _clip(outside, mask.shape) if outside else None, pitch, stroke


def _text_row_region(mask, region, anchor, role, pitch):
    """Separate adjacent text rows without dropping detached caps or bars."""
    if region is None:
        return None, {}
    x0, y0, x1, y1 = region
    all_spans = [(y0 + a, y0 + b) for a, b in _groups(mask[y0:y1, x0:x1].any(1), 1)]
    spans = [(a, b) for a, b in all_spans if b - a >= 3]
    if len(spans) <= 1:
        return region, {}
    if role == "pin_name":
        a, b = min(spans, key=lambda span: abs((span[0] + span[1]) / 2 - anchor[1]))
    else:
        a, b = min(spans, key=lambda span: abs(anchor[1] - span[1]))
    selected = [a, b]
    gap = (max(3, min(18, int(pitch * .52))) if role == "pin_number"
           else max(3, min(12, int(pitch * .28))))
    words = _groups(mask[a:b, x0:x1].any(0), gap)
    kept, ambiguous = [], []
    if words:
        # The far side's independent name row can be taller or carry a bar.
        # Only small ink overlapping this actual nearest-side word belongs to
        # this glyph integrity audit.
        lo, hi = min(words, key=lambda word: min(abs(x0 + word[0] - anchor[0]),
                                                abs(x0 + word[1] - anchor[0])))
        for top, bottom in all_spans:
            if (top, bottom) == (a, b) or bottom - top > max(3, .35 * (b - a)):
                continue
            # A disconnected numeral cap, overbar or underscore may share no
            # connected component with its body. Horizontal overlap and actual
            # white-gap distance retain nearby small ink without naming it.
            if not np.any(mask[top:bottom, x0 + lo:x0 + hi]):
                continue
            distance = max(a - bottom, top - b, 0)
            if distance <= max(4, .65 * (b - a)):
                selected[0], selected[1] = min(selected[0], top), max(selected[1], bottom)
                kept.append([top, bottom])
            else:
                ambiguous.append([top, bottom])
    return [x0, max(y0, selected[0] - 1), x1, min(y1, selected[1] + 1)], {
        "selected_ink_row": [a, b], "retained_detached_small_ink_rows": kept,
        "ambiguous_detached_small_ink_rows": ambiguous}


def _known_external_lines(mask, region, observation, pitch, stroke, light_mask=None):
    """Identify measured long wires, rather than treating narrow ink as noise.

    A vertical wire must continue through the complete number window and have
    observed length at least 1.8 physical pitches, extend beyond its complete
    cell, and connect to the current attached pin stroke. Raster fringes must
    show the same long continuation; narrow ink alone is never line evidence.
    """
    known = np.zeros(mask.shape, dtype=bool)
    if region is None:
        return known, []
    x0, y0, x1, y1 = region
    context_top, context_bottom = max(0, int(y0 - 2 * pitch)), min(mask.shape[0], int(y1 + 2 * pitch))
    evidence = []
    light = mask if light_mask is None else light_mask
    probe_x = int((stroke["probe"][0] + stroke["probe"][2] - 1) / 2)
    core_columns = []
    for x in range(max(0, x0 - 1), min(mask.shape[1], x1 + 1)):
        for a, b in _groups(mask[context_top:context_bottom, x], 1):
            a, b = a + context_top, b + context_top
            extended = a <= y0 - .5 * pitch or b >= y1 + .5 * pitch
            # A long isolated "1" is not a wire. The current measured stroke
            # must meet this line, without an inferred electrical connection.
            left, right = min(x, probe_x), max(x, probe_x) + 1
            top, bottom = max(0, stroke["stroke_top"] - 1), min(mask.shape[0], stroke["stroke_bottom"] + 1)
            _, labels = cv2.connectedComponents(mask[top:bottom, left:right].astype(np.uint8), connectivity=8)
            source = set(np.unique(labels[:, x-left]).tolist()) - {0}
            target = set(np.unique(labels[:, probe_x-left]).tolist()) - {0}
            attachment = bool(source.intersection(target))
            if b - a >= max(20, 1.8 * pitch) and a <= y0 and b >= y1 and extended and attachment:
                known[a:b, x] |= mask[a:b, x]
                core_columns.append((x, a, b))
                evidence.append({"axis": "vertical", "column": x, "span": [a, b],
                                 "minimum_length_pixels": max(20, 1.8 * pitch)})
    for x, a, b in core_columns:
        for fringe in range(max(0, x - 1), min(mask.shape[1], x + 2)):
            # Whole-cell continuation keeps a neighboring disconnected glyph
            # cap or serif from being erased as a line's antialiasing fringe.
            runs = _groups(light[a:b, fringe], 1)
            for top, bottom in runs:
                top, bottom = top + a, bottom + a
                if top <= y0 and bottom >= y1 and bottom - top >= max(20, 1.8 * pitch):
                    known[top:bottom, fringe] |= light[top:bottom, fringe]
                    evidence.append({"axis": "vertical_raster_fringe", "column": fringe,
                                     "span": [top, bottom], "measured_core_column": x})
    anchor = observation["anchor_img"]
    xa, xb = min(x0, int(anchor[0])), max(x1, int(anchor[0]) + 1)
    for row in range(max(0, stroke["stroke_top"] - 1), min(mask.shape[0], stroke["stroke_bottom"] + 1)):
        # A pale fringe is eligible only inside the observed stroke band plus
        # one raster row, with a continuous run through the measured probe.
        for a, b in _groups(light[row, xa:xb], 1):
            a, b = a + xa, b + xa
            if a <= probe_x < b and b - a >= max(8, .60 * pitch):
                known[row, a:b] |= light[row, a:b]
                evidence.append({"axis": "horizontal", "row": row, "span": [a, b],
                                 "measured_probe_column": probe_x})
    return known, evidence


def _complete_group(mask, region, observation, role, pitch, known_lines=None, original_mask=None):
    """Select the closest separated full word, retaining every small stroke."""
    if region is None:
        return None
    x0, y0, x1, y1 = region
    sub = mask[y0:y1, x0:x1]
    # Horizontal letter/digit gaps grow with measured row pitch. A blank span
    # separates opposite-edge words, which must never enter one another's crop.
    gap = (max(3, min(18, int(pitch * .52))) if role == "pin_number"
           else max(3, min(12, int(pitch * .28))))
    choices = []
    for a, b in _groups(sub.any(0), gap):
        yy, xx = np.nonzero(sub[:, a:b])
        if len(xx) < 3:
            continue
        box = [x0 + a + int(xx.min()), y0 + int(yy.min()),
               x0 + a + int(xx.max()) + 1, y0 + int(yy.max()) + 1]
        if box[3] - box[1] < 3:
            continue
        # Each retained fragment is actual ink; no area filter drops narrow 1
        # or disconnected tops/underscores from a multi-character word.
        distance = min(abs(box[0] - observation["anchor_img"][0]),
                       abs(box[2] - observation["anchor_img"][0]))
        choices.append((distance, box))
    if not choices:
        return None
    _, glyph = min(choices, key=lambda value: value[0])
    touch = [glyph[0] == x0, glyph[1] == y0, glyph[2] == x1, glyph[3] == y1]
    connected = []
    # A full component reaching the ownership edge remains unresolved only if
    # original ink actually continues across it. Do not guess missing letters.
    original = mask if original_mask is None else original_mask
    for side, touched in enumerate(touch):
        if not touched:
            continue
        if side in (0, 2):
            column = x0 - 1 if side == 0 else x1
            border = x0 if side == 0 else x1 - 1
            if 0 <= column < mask.shape[1]:
                active = np.flatnonzero(mask[glyph[1]:glyph[3], border]) + glyph[1]
                # Connected components use 8-connectivity. A slanted stroke
                # can leave the ownership cell diagonally, so same-row-only
                # comparison incorrectly declares a clipped glyph complete.
                if any(np.any(original[np.clip(active + delta, 0, mask.shape[0] - 1), column])
                       for delta in (-1, 0, 1)):
                    connected.append(side)
        else:
            row = y0 - 1 if side == 1 else y1
            border = y0 if side == 1 else y1 - 1
            if 0 <= row < mask.shape[0]:
                active = np.flatnonzero(mask[border, glyph[0]:glyph[2]]) + glyph[0]
                if any(np.any(original[row, np.clip(active + delta, 0, mask.shape[1] - 1)])
                       for delta in (-1, 0, 1)):
                    connected.append(side)
    contact = False
    if known_lines is not None:
        xa, ya, xb, yb = max(0, glyph[0]-1), max(0, glyph[1]-1), min(mask.shape[1], glyph[2]+1), min(mask.shape[0], glyph[3]+1)
        retained = np.zeros((yb-ya, xb-xa), np.uint8)
        retained[glyph[1]-ya:glyph[3]-ya, glyph[0]-xa:glyph[2]-xa] = mask[glyph[1]:glyph[3], glyph[0]:glyph[2]]
        adjacent = cv2.dilate(retained, np.ones((3, 3), np.uint8)) > 0
        contact = bool(np.any(adjacent & known_lines[ya:yb, xa:xb] & original[ya:yb, xa:xb]))
    # If a numeral meets a measured wire, erasing their connected lower ink is
    # speculative. Raw source pixels, rather than cleaned pixels, audit this.
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask[glyph[1]:glyph[3], glyph[0]:glyph[2]].astype(np.uint8), 8)
    return glyph, {"method": "complete_original_ink_group_inside_physical_row_cell",
                   "integrity_unresolved": bool(connected) or contact,
                   "removed_line_touches_retained_glyph": contact,
                   "connected_ink_beyond_region_sides": connected,
                   "boundary_touch_sides": [i for i, v in enumerate(touch) if v],
                   "glyph_pixels": int(mask[glyph[1]:glyph[3], glyph[0]:glyph[2]].sum()),
                   "glyph_component_count": count - 1,
                   "glyph_component_stats": stats[1:].tolist(),
                   "black_pixels_removed": 0, "word_gap_pixels": gap}


def mono_pin_boxes(doc, image, reads=None):
    """Return independent role candidates and an unchanged grayscale copy.

    Missing, weak or stale evidence cannot open a crop. Only horizontal
    left/right rows are supported; top/bottom orientation stays on its old path.
    Existing reads are accepted for interface compatibility and never affect
    glyph extraction or eligibility.
    """
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        return [], None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chroma = image.max(2).astype(np.int16) - image.min(2).astype(np.int16)
    mask = (gray < 190) & (chroma < 30)
    light_mask = (gray < 245) & (chroma < 30)
    observations, output = list(mono_observations(doc)), []
    for ob in observations:
        peers = [p for p in observations if p["component_id"] == ob["component_id"] and p["edge"] == ob["edge"]]
        inside, outside, pitch, stroke = _regions(ob, peers, mask)
        if stroke is None:
            # Cached geometric evidence cannot open a name lane when its
            # claimed attached stroke is absent from the supplied image.
            continue
        for role, region in (("pin_name", inside), ("pin_number", outside)):
            known, line_evidence = (None, [])
            local_mask = mask
            if role == "pin_number" and region is not None:
                known, line_evidence = _known_external_lines(mask, region, ob, pitch, stroke, light_mask)
                local_mask = mask & ~known
            full_region = list(region) if region is not None else None
            region, row_evidence = _text_row_region(local_mask, region, ob["anchor_img"], role, pitch)
            complete = _complete_group(local_mask, region, ob, role, pitch, known_lines=known, original_mask=mask)
            if complete is None:
                continue
            glyph, integrity = complete
            excluded = []
            if known is not None:
                yy, xx = np.nonzero(known[glyph[1]:glyph[3], glyph[0]:glyph[2]])
                excluded = [[glyph[0] + int(x), glyph[1] + int(y)] for y, x in zip(yy, xx)]
            integrity["known_line_exclusions"] = excluded
            integrity["measured_line_evidence"] = line_evidence
            integrity["black_pixels_removed"] = len(excluded)
            integrity["physical_search_region"] = full_region
            integrity["text_row_selection"] = row_evidence
            if row_evidence.get("ambiguous_detached_small_ink_rows"):
                integrity["integrity_unresolved"] = True
                integrity["unresolved_reason"] = "detached_small_ink_cannot_be_assigned_without_guessing"
            if known is not None and full_region is not None:
                a, b, c, d = full_region
                yy, xx = np.nonzero(known[b:d, a:c])
                integrity["excluded_region_pixel_count"] = len(xx)
                integrity["excluded_region_ink_bbox"] = ([a + int(xx.min()), b + int(yy.min()), a + int(xx.max()) + 1, b + int(yy.max()) + 1] if len(xx) else None)
            if np.any((chroma[glyph[1]:glyph[3], glyph[0]:glyph[2]] >= 50)
                      & (gray[glyph[1]:glyph[3], glyph[0]:glyph[2]] < 230)):
                integrity["integrity_unresolved"] = True
                integrity["unresolved_reason"] = "non_monochrome_ink_inside_original_crop"
            target = {k: ob[k] for k in ("component_id", "pin_index", "edge", "point_img", "anchor_img")}
            flags = ["glyph_integrity_unresolved"] if integrity["integrity_unresolved"] else []
            output.append({"bbox": glyph, "geometry_bbox": list(glyph), "glyph_bbox": list(glyph),
                           "region": region, "role": role, "vertical": False,
                           "pin_target": target, "flags": flags,
                           "cleanup_evidence": dict(integrity, stroke_probe=stroke,
                               pitch_pixels=pitch, physical_evidence=ob["evidence"],
                               same_side_rows=[p["anchor_img"][1] for p in peers],
                               independent_of_ocr_text=True)})
    return output, gray
