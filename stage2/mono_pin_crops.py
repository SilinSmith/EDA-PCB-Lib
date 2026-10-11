"""Complete horizontal pin text from measured monochrome frame/stroke evidence.

The frame anchor and same-side physical rows define the search cells. OCR text
does not define a crop, and no numbering order or name dictionary is consulted.
Unlike colour cleanup, this path never erases black pixels: an external number
cell stops at the observed upper edge of the actual pin stroke.
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
    The result clips the search cell; no pixels, including numeral 1's bars,
    are deleted or reconstructed.
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
        return None
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
        # The near-frame line can have sparse raster fringe one pixel higher
        # farther along the wire. Exclude that conservative fringe row from
        # the cell, without applying a morphology filter to digit pixels.
        stroke["upper_fringe_guard_pixels"] = 1
        outside = ([anchor[0] - depth, y0, anchor[0] - 2, stroke["stroke_top"] - 1]
                   if edge == "left" else
                   [anchor[0] + 2, y0, anchor[0] + depth, stroke["stroke_top"] - 1])
    return _clip(inside, mask.shape), _clip(outside, mask.shape) if outside else None, pitch, stroke


def _complete_group(mask, region, observation, role, pitch):
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
                if any(np.any(mask[np.clip(active + delta, 0, mask.shape[0] - 1), column])
                       for delta in (-1, 0, 1)):
                    connected.append(side)
        else:
            row = y0 - 1 if side == 1 else y1
            border = y0 if side == 1 else y1 - 1
            if 0 <= row < mask.shape[0]:
                active = np.flatnonzero(mask[border, glyph[0]:glyph[2]]) + glyph[0]
                if any(np.any(mask[row, np.clip(active + delta, 0, mask.shape[1] - 1)])
                       for delta in (-1, 0, 1)):
                    connected.append(side)
    # If a numeral meets its wire, separating the connected lower ink would
    # be speculative. Record it and hold rather than erasing a bar from a 1.
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask[glyph[1]:glyph[3], glyph[0]:glyph[2]].astype(np.uint8), 8)
    return glyph, {"method": "complete_original_ink_group_inside_physical_row_cell",
                   "integrity_unresolved": bool(connected),
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
    observations, output = list(mono_observations(doc)), []
    for ob in observations:
        peers = [p for p in observations if p["component_id"] == ob["component_id"] and p["edge"] == ob["edge"]]
        inside, outside, pitch, stroke = _regions(ob, peers, mask)
        if stroke is None:
            # Cached geometric evidence cannot open a name lane when its
            # claimed attached stroke is absent from the supplied image.
            continue
        for role, region in (("pin_name", inside), ("pin_number", outside)):
            complete = _complete_group(mask, region, ob, role, pitch)
            if complete is None:
                continue
            glyph, integrity = complete
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
