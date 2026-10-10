"""Generate independent name/number windows from confirmed colour pin lanes.

The windows use the current measured pin and frame contact, never a recognised
number or an expected pin label.  Recognition and padding are intentionally
left to the caller so the original glyph geometry remains unchanged.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

from .color_crops import color_number_boxes
from .pin_text_crops import _finite, _observations, pin_text_boxes


def _clip(region, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = region
    clipped = [max(0, int(math.ceil(x0))), max(0, int(math.ceil(y0))),
               min(w, int(math.floor(x1))), min(h, int(math.floor(y1)))]
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _intersects(first, second):
    return (min(first[2], second[2]) > max(first[0], second[0])
            and min(first[3], second[3]) > max(first[1], second[1]))


def _side_peers(observation, observations):
    return [p for p in observations
            if p["component_id"] == observation["component_id"]
            and p["edge"] == observation["edge"]]


def _pitch(observation, peers):
    tangent = 0 if observation["edge"] in ("top", "bottom") else 1
    center = observation["anchor_img"][tangent]
    distances = [abs(p["anchor_img"][tangent] - center) for p in peers
                 if abs(p["anchor_img"][tangent] - center) > 1]
    frame = observation["frame"]
    span = frame[2] - frame[0] if tangent == 0 else frame[3] - frame[1]
    return min(distances) if distances else max(6.0, min(span, 44.0))


def _number_region(observation, peers, shape):
    """Keep horizontal digits above their row and rotated digits on their stub.

    A nearest-neighbour pitch limits row depth even if no text detector has
    supplied a box.  The frame contact supplies the inside/outside boundary;
    the detector's approximate component bbox never supplies that boundary.
    """
    point, anchor, edge = (observation[k] for k in ("point_img", "anchor_img", "edge"))
    pitch = _pitch(observation, peers)
    if edge in ("left", "right"):
        # A digit's baseline is usually on or immediately above the pin line.
        # Restrict the upper part by the same-side pitch, rather than sharing
        # a broad detector window between adjacent rows.
        depth = min(44.0, max(4.0, pitch * .82))
        extension = min(3.0, max(1.0, pitch * .10))
        y0, y1 = point[1] - depth, point[1] + min(2.0, pitch * .10)
        if edge == "left":
            region = [point[0] - extension, y0, anchor[0], y1]
        else:
            region = [anchor[0], y0, point[0] + extension, y1]
    else:
        half = min(22.0, pitch * .44)
        # Asymmetric gaps have asymmetric ownership cells.  No neighbouring
        # pin can contribute a character merely because the opposite gap is
        # larger.  Do not impose a minimum half-width across a close neighbour.
        left, right = anchor[0] - half, anchor[0] + half
        for peer in peers:
            px = peer["anchor_img"][0]
            if px < anchor[0] - 1:
                left = max(left, (px + anchor[0]) / 2 + .5)
            elif px > anchor[0] + 1:
                right = min(right, (px + anchor[0]) / 2 - .5)
        # Tangent bounds enclose whole pixel cells.  Normal bounds still use
        # the strict external frame boundary, so rounding cannot sample inside.
        left, right = math.floor(left), math.ceil(right)
        if edge == "top":
            region = [left, point[1] - 2, right, anchor[1] - 1]
        else:
            region = [left, anchor[1] + 1, right, point[1] + 2]
    return _clip(region, shape), pitch


def _glyph_box(mask, region):
    x0, y0, x1, y1 = region
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if len(xs) < 3:
        return None
    box = [x0 + int(xs.min()), y0 + int(ys.min()),
           x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]
    if box[2] - box[0] < 1 or box[3] - box[1] < 3:
        return None
    return box


def _complete_number_region(mask, region, observation, peers, pitch):
    """Recover a touching glyph's top without opening a neighbouring row.

    The initial window follows the pin pitch.  Actual ink at its upper border
    is evidence that a high-baseline digit needs a little more room.  A measured
    preceding pin limits that expansion; recognised strings are never used.
    """
    glyph = _glyph_box(mask, region) if region is not None else None
    if (glyph is not None and observation["edge"] in ("left", "right")
            and glyph[1] <= region[1] + 1):
        row = observation["point_img"][1]
        previous_rows = [p["point_img"][1] for p in peers if p["point_img"][1] < row - 1]
        lower_limit = max(previous_rows) + 2 if previous_rows else 0
        extended = list(region)
        extended[1] = max(0, int(math.ceil(max(row - min(44.0, .97 * pitch), lower_limit))))
        if extended[1] < region[1]:
            region = extended
            glyph = _glyph_box(mask, region)
    return region, glyph


def _complete_vertical_glyph(mask, glyph, region, observation, peers, role, pitch):
    """Complete seeded ink components within an asymmetric ownership cell.

    Rotated labels need not be centred on the pin line: their baseline can sit
    on either side of it.  Expand only components already intersecting this
    glyph.  This restores connected clipped strokes without adding a separate
    neighbouring word or guessing a missing letter from an OCR dictionary.
    """
    initial_touch = glyph[0] <= region[0] or glyph[2] >= region[2]
    evidence = {"initial_bbox": list(glyph), "initial_region": list(region),
                "initial_tangent_boundary_touch": initial_touch,
                "method": "seed_connected_components_with_asymmetric_pin_cells",
                "changed": False, "integrity_unresolved": False}
    if not initial_touch:
        return glyph, region, evidence
    frame, center = observation["frame"], observation["anchor_img"][0]
    # Names are inside the body.  External numbers can protrude around a frame
    # corner, but remain within one local pitch of that body's tangent span.
    inset = 2.0 if role == "pin_name" else -pitch
    left, right = frame[0] + inset, frame[2] - inset
    for peer in peers:
        px = peer["anchor_img"][0]
        if px < center - 1:
            left = max(left, (px + center) / 2 + .5)
        elif px > center + 1:
            right = min(right, (px + center) / 2 - .5)
    ownership = _clip([math.floor(left), region[1], math.ceil(right), region[3]], mask.shape)
    evidence["ownership_region"] = ownership
    if (ownership is None or ownership[0] > glyph[0] or ownership[2] < glyph[2]):
        # An initial detector lane already crossed the ownership cell.  Do not
        # silently trim a word and then pretend that it is a complete sample.
        evidence["integrity_unresolved"] = True
        evidence["unresolved_reason"] = "initial_glyph_exceeds_ownership_cell"
        return glyph, region, evidence
    x0, y0, x1, y1 = ownership
    local = mask[y0:y1, x0:x1].astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
    hits = np.unique(labels[glyph[1] - y0:glyph[3] - y0,
                            glyph[0] - x0:glyph[2] - x0])
    hits = hits[hits != 0]
    selected = np.isin(labels, hits)
    ys, xs = np.nonzero(selected)
    if not len(xs):
        evidence["integrity_unresolved"] = True
        evidence["unresolved_reason"] = "no_seeded_ink"
        return glyph, region, evidence
    completed = [x0 + int(xs.min()), y0 + int(ys.min()),
                 x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]
    evidence["changed"] = completed != glyph
    evidence["completed_bbox"] = completed
    # A component reaching the hard cell edge remains potentially clipped
    # only if actual black ink connects across that edge.  A complete digit
    # can end exactly on a half-pixel ownership boundary with white beyond it.
    # Unknown detached fragments immediately outside the selected word are
    # also evidence of uncertainty; they are never pasted into the sample.
    remaining_touch = completed[0] <= x0 or completed[2] >= x1
    connected_beyond = False
    for side, column in ((0, x0 - 1), (-1, x1)):
        if not 0 <= column < mask.shape[1]:
            continue
        edge_ys = np.flatnonzero(selected[:, side])
        for yy in edge_ys:
            ya, yb = max(0, y0 + int(yy) - 1), min(mask.shape[0], y0 + int(yy) + 2)
            if mask[ya:yb, column].any():
                connected_beyond = True
                break
    detached = []
    for label in range(1, count):
        if label in hits:
            continue
        sx, sy, sw, sh, area = [int(v) for v in stats[label]]
        other = [x0 + sx, y0 + sy, x0 + sx + sw, y0 + sy + sh]
        tangent_gap = max(completed[0] - other[2], other[0] - completed[2], 0)
        outside = other[2] <= completed[0] or other[0] >= completed[2]
        normal_overlap = min(other[3], completed[3]) > max(other[1], completed[1])
        if outside and normal_overlap and tangent_gap <= 2:
            detached.append({"bbox": other, "pixels": area})
    evidence["remaining_tangent_boundary_touch"] = remaining_touch
    evidence["connected_ink_beyond_ownership"] = connected_beyond
    evidence["nearby_unattached_components"] = detached
    evidence["integrity_unresolved"] = connected_beyond or bool(detached)
    if connected_beyond:
        evidence["unresolved_reason"] = "seeded_component_continues_beyond_ownership_boundary"
    elif detached:
        evidence["unresolved_reason"] = "unattached_nearby_ink_has_unknown_owner"
    return completed, ownership, evidence


def _source_evidence(observation, role, glyph, region, reads, peers, colored, mask):
    """Record observable re-read reasons without trusting old OCR as a gate."""
    edge, anchor = observation["edge"], observation["anchor_img"]
    normal = 1 if edge in ("top", "bottom") else 0
    tangent = 1 - normal
    flags, sources = set(), []
    for index, read in enumerate(reads or []):
        if not isinstance(read, dict):
            continue
        box = _finite(read.get("bbox"), 4)
        if box is None or box[2] <= box[0] or box[3] <= box[1] or not _intersects(box, glyph):
            continue
        alternatives = read.get("reads")
        alternatives = alternatives if isinstance(alternatives, dict) else {}
        value = alternatives.get(role)
        text = str(value[0]).strip() if isinstance(value, (list, tuple)) and value else ""
        if role == "pin_number" and not re.fullmatch(r"\d{1,3}", text):
            flags.add("non_numeric_existing_read")
        if box[normal] < anchor[normal] < box[normal + 2]:
            flags.add("mixed_frame_box")
        crosses_frame = box[normal] < anchor[normal] < box[normal + 2]
        row_count = sum(box[tangent] <= p["anchor_img"][tangent] <= box[tangent + 2]
                        for p in peers)
        if row_count > 1:
            flags.add("multiple_pin_rows")
        x0, y0, x1, y1 = _clip(box, colored.shape) or [0, 0, 0, 0]
        ncolor = int(np.count_nonzero(colored[y0:y1, x0:x1]))
        nblack = int(np.count_nonzero(mask[y0:y1, x0:x1]))
        if ncolor:
            flags.add("color_contamination")
        truncates_glyph = not (box[0] <= glyph[0] and box[1] <= glyph[1]
                               and box[2] >= glyph[2] and box[3] >= glyph[3])
        if truncates_glyph:
            flags.add("existing_box_truncates_glyph")
        sources.append({"read_index": index, "bbox": box, "role_text": text,
                        "colored_pixels": ncolor, "black_pixels": nblack,
                        "crosses_frame": crosses_frame, "pin_row_count": row_count,
                        "truncates_glyph": truncates_glyph,
                        "color_contamination_ratio": round(ncolor / max(1, ncolor + nblack), 6)})
    if not sources:
        flags.add("missing_role_box")
    elif not any(s["role_text"] for s in sources):
        flags.add("missing_role_read")
    if any(glyph[i] == region[i] for i in range(4)):
        flags.add("glyph_touches_search_boundary")
    return sorted(flags), sources


def pin_lane_boxes(doc, image, reads=None):
    """Return ``(candidates, cleaned_gray)`` in original image coordinates.

    Only box components with trusted, current colour contacts and confirmed
    frame anchors are eligible.  Name and number candidates are independent
    for all four sides.  Every black component in a number lane is retained:
    a broken narrow ``1`` is evidence, not a disposable small component.

    Existing OCR contributes diagnostic flags only.  The caller may use those
    flags for selective extra recognition and must keep existing candidates.
    """
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        return [], None
    observations = list(_observations(doc, include_horizontal=True))
    # Reuse the established colour/fringe mask without the old OCR-number gate.
    # Passing no reads returns the cleaned view while creating no legacy boxes.
    _, cleaned = color_number_boxes(doc, [], image)
    mask = cleaned < 190
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chroma = image.max(2).astype(np.int16) - image.min(2).astype(np.int16)
    strong_colored = (chroma >= 50) & (gray < 230)
    near_color = cv2.dilate(strong_colored.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    colored = strong_colored | (near_color & (chroma > 25) & (gray >= 100) & (gray < 230))
    name_boxes, _ = pin_text_boxes(doc, image, include_horizontal=True)
    names = {(b["pin_target"]["component_id"], b["pin_target"]["pin_index"]): b
             for b in name_boxes if b["role"] == "pin_name"}
    candidates = []
    for observation in observations:
        peers = _side_peers(observation, observations)
        number_region, pitch = _number_region(observation, peers, image.shape)
        key = (observation["component_id"], observation["pin_index"])
        for role in ("pin_name", "pin_number"):
            if role == "pin_name":
                existing = names.get(key)
                if existing is None:
                    continue
                region = _clip(existing["region"], image.shape)
                glyph = existing["bbox"]
                normal = 1 if observation["edge"] in ("top", "bottom") else 0
                gap = min(abs(observation["anchor_img"][normal] - glyph[normal]),
                          abs(observation["anchor_img"][normal] - glyph[normal + 2]))
                # If this side is unnamed, a word beside the opposite frame
                # side is not a replacement.  Near-frame words can still be
                # longer than half the component width.
                if gap > max(8.0, pitch * .75):
                    continue
            else:
                region, glyph = _complete_number_region(mask, number_region, observation, peers, pitch)
            if region is None or glyph is None:
                continue
            completion = None
            if observation["edge"] in ("top", "bottom"):
                glyph, region, completion = _complete_vertical_glyph(
                    mask, glyph, region, observation, peers, role, pitch)
            target = {k: v for k, v in observation.items() if k != "frame"}
            flags, sources = _source_evidence(observation, role, glyph, region, reads, peers, colored, mask)
            if completion is not None:
                if completion["initial_tangent_boundary_touch"]:
                    flags.append("initial_glyph_touches_search_boundary")
                if completion["changed"]:
                    flags.append("glyph_completed_from_boundary")
                if completion["integrity_unresolved"]:
                    flags.append("glyph_integrity_unresolved")
                flags = sorted(set(flags))
            x0, y0, x1, y1 = glyph
            rx0, ry0, rx1, ry1 = region
            removed = int(np.count_nonzero(colored[ry0:ry1, rx0:rx1]))
            black = int(np.count_nonzero(mask[ry0:ry1, rx0:rx1]))
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask[y0:y1, x0:x1].astype(np.uint8), 8)
            candidates.append({"bbox": list(glyph), "glyph_bbox": list(glyph),
                               "role": role, "vertical": observation["edge"] in ("top", "bottom"),
                               "region": region, "pin_target": target, "flags": flags,
                               "cleanup_evidence": {
                                   "sampling_method": "low_chroma_dark_ink_with_near_color_fringe_removal",
                                   "same_side_pitch": pitch,
                                   "glyph_pixels": int(np.count_nonzero(mask[y0:y1, x0:x1])),
                                   "glyph_component_count": component_count - 1,
                                   "glyph_component_stats": stats[1:].tolist(),
                                   "black_components_removed": 0,
                                   "colored_pixels_removed": removed,
                                   "color_contamination_ratio": round(removed / max(1, removed + black), 6),
                                   "source_boxes": sources,
                                   "glyph_completion": completion,
                                   "independent_of_existing_ocr": True}})
    return candidates, cleaned
