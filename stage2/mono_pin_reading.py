"""Read trusted monochrome pin glyphs through two fixed white-margin views.

These are role-only additional observations. Missing optional OCR leaves the
existing pipeline intact. Agreement is never a dictionary or numbering repair.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

from .mono_pin_crops import _finite, mono_observations, mono_pin_boxes
from .pin_name_ocr import get_pin_name_ocr


ACCEPTANCE_THRESHOLD = .90
PADDINGS = (2, 4)


def _text(value, role):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if role == "pin_number":
        return text if re.fullmatch(r"[0-9]{1,3}", text) and 1 <= int(text) <= 999 else None
    return text if text and any(ch.isalpha() for ch in text) else None


def _agreement(views, role, threshold):
    if len(views) != len(PADDINGS):
        return None
    values = []
    for padding, view in zip(PADDINGS, views):
        text = _text(view.get("text"), role)
        try:
            confidence = float(view.get("confidence", 0))
        except (TypeError, ValueError, OverflowError):
            return None
        if (view.get("padding") != padding or text is None or not math.isfinite(confidence)
                or not threshold <= confidence <= 1):
            return None
        values.append((text, confidence))
    return (values[0][0], min(v[1] for v in values)) if values[0][0] == values[1][0] else None


def _overlap(first, second):
    return (max(0, min(first[2], second[2]) - max(first[0], second[0]))
            * max(0, min(first[3], second[3]) - max(first[1], second[1])))


def _old_number_geometry_reasons(doc, number, candidate, target, bgr):
    """Audit the OLD winning numeral, never infer dirt from PP's failure.

    A held or missing new digit is not evidence that the legacy digit is bad.
    This function needs observed mixed geometry or actual ink continuing
    across the old box within the same new physical word cell.
    """
    old = _finite(number.get("box"), 4) if isinstance(number, dict) else None
    if old is None or old[2] <= old[0] or old[3] <= old[1]:
        return []
    cid, edge, anchor = target["component_id"], target["edge"], target["anchor_img"]
    reasons = []
    if old[0] < anchor[0] < old[2]:
        reasons.append("winning_number_box_crosses_actual_frame")
    peers = [ob for ob in mono_observations(doc)
             if ob["component_id"] == cid and ob["edge"] == edge]
    covered = [ob["pin_index"] for ob in peers if old[1] <= ob["anchor_img"][1] < old[3]]
    if len(covered) > 1:
        reasons.append("winning_number_box_contains_multiple_physical_rows")
    if candidate is None:
        return reasons
    glyph = _finite(candidate.get("bbox"), 4)
    region = _finite(candidate.get("region"), 4)
    if glyph is None or region is None:
        return reasons
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = bgr.max(2).astype(np.int16) - bgr.min(2).astype(np.int16)
    ink = (gray < 190) & (chroma < 30)
    height, width = ink.shape
    # A real horizontal stroke must connect this winning box to the measured
    # frame probe. Merely having a baseline close to the pin is not enough.
    stroke = (candidate.get("cleanup_evidence") or {}).get("stroke_probe")
    if isinstance(stroke, dict):
        top, bottom = stroke.get("stroke_top"), stroke.get("stroke_bottom")
        if isinstance(top, int) and isinstance(bottom, int):
            xa = int(math.ceil(old[2])) - 1 if edge == "left" else int(math.floor(anchor[0])) + 2
            xb = int(math.ceil(anchor[0])) - 2 if edge == "left" else int(math.floor(old[0])) + 1
            lo, hi = max(0, min(xa, xb)), min(width, max(xa, xb) + 1)
            if hi - lo >= 3:
                for row in range(max(top, 0, int(math.ceil(old[1]))),
                                 min(bottom, height, int(math.floor(old[3])))):
                    if np.mean(ink[row, lo:hi]) >= .90:
                        reasons.append("attached_pin_stroke_enters_winning_number_box")
                        break
    if _overlap(old, glyph) <= 0:
        return reasons
    # A crop can lose a leading disconnected 1 without crossing an ink
    # component. That alone does not prove a bad old read when PP is held.
    # Require an actual 8-connected stroke cut by an OLD rectangle edge,
    # bounded by the same new physical word cell and observed word bbox.
    x0, y0, x1, y1 = [int(math.ceil(old[0])), int(math.ceil(old[1])),
                       int(math.floor(old[2])), int(math.floor(old[3]))]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return reasons
    support = [max(region[0], glyph[0]), max(region[1], glyph[1]),
               min(region[2], glyph[2]), min(region[3], glyph[3])]
    cut = False
    for border, outside in ((x0, x0 - 1), (x1 - 1, x1)):
        if not support[0] <= outside < support[2]:
            continue
        ya, yb = max(y0, int(math.ceil(support[1]))), min(y1, int(math.floor(support[3])))
        active = np.flatnonzero(ink[ya:yb, border]) + ya
        for delta in (-1, 0, 1):
            neighboring = active + delta
            neighboring = neighboring[(neighboring >= support[1]) & (neighboring < support[3])]
            cut = cut or bool(len(neighboring) and np.any(ink[neighboring, outside]))
    for border, outside in ((y0, y0 - 1), (y1 - 1, y1)):
        if not support[1] <= outside < support[3]:
            continue
        xa, xb = max(x0, int(math.ceil(support[0]))), min(x1, int(math.floor(support[2])))
        active = np.flatnonzero(ink[border, xa:xb]) + xa
        for delta in (-1, 0, 1):
            neighboring = active + delta
            neighboring = neighboring[(neighboring >= support[0]) & (neighboring < support[2])]
            cut = cut or bool(len(neighboring) and np.any(ink[outside, neighboring]))
    if cut:
        reasons.append("winning_number_box_cuts_observed_connected_glyph")
    return reasons


def _guard_recovered_names(doc, reads, output, candidates, bgr, min_conf=0.0):
    names = [rec for rec in output if rec.get("name_only")]
    if not names:
        return
    # Lazy import avoids introducing an import cycle into the reader modules.
    from .assign import assign
    from .pin_lane_reading import _threshold_reads
    eligible_reads = _threshold_reads(reads, min_conf)
    existing = assign(doc.get("components", []), doc.get("pins", {}), eligible_reads,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    accepted_numbers = [r for r in output if r.get("number_only") and r.get("reads")
                        and r["mono_pin_ocr"].get("accepted") is True]
    rebound = (assign(doc.get("components", []), doc.get("pins", {}), eligible_reads + output,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
               if accepted_numbers else {})
    confirmed = set()
    for observed in accepted_numbers:
        target = observed["pin_target"]
        key = (target["component_id"], target["pin_index"])
        actual = rebound.get("comps", {}).get(key[0], {}).get("pins", {}).get(key[1], {})
        number = actual.get("number") or {}
        text = (observed.get("reads") or {}).get("pin_number")
        # OCR acceptance alone is insufficient: assignment can still reject
        # its crop's frame/row geometry. Only the independently-read ACTUAL
        # winner for this same physical pin can release the partner guard.
        if (actual.get("number_independently_read") is True and text
                and number.get("text") == text[0]
                and _finite(number.get("box"), 4) == _finite(observed.get("bbox"), 4)
                and _finite(observed.get("bbox"), 4) is not None):
            confirmed.add(key)
    digit_boxes = {(c["pin_target"]["component_id"], c["pin_target"]["pin_index"]): c
                   for c in candidates if c["role"] == "pin_number"}
    for rec in names:
        target, meta = rec["pin_target"], rec["mono_pin_ocr"]
        key = (target["component_id"], target["pin_index"])
        old = existing.get("comps", {}).get(key[0], {}).get("pins", {}).get(key[1], {})
        meta["previous_name"], meta["previous_number"] = old.get("name"), old.get("number")
        if not rec.get("reads") or old.get("name") or not old.get("number"):
            continue
        reasons = [] if key in confirmed else _old_number_geometry_reasons(
            doc, old["number"], digit_boxes.get(key), target, bgr)
        meta["partner_guard"] = {"new_number_confirmed": key in confirmed,
                                 "winning_old_number_geometry_reasons": reasons}
        if reasons:
            meta.update(accepted=False, held_reason="unresolved_dirty_partner_number_geometry")
            rec["reads"] = {}


def read_mono_pin_texts(doc, reads, engine, bgr, min_conf=0.0, mode="auto",
                        include_names=True, include_numbers=True):
    """Return role-limited targeted records, including held audit observations.

    ``engine`` is accepted for the pipeline interface. PP sees actual complete
    source pixels directly, with pads 2 and 4, without intermediate gray32
    resampling or cross-crop atlases. Input objects are never modified.
    """
    if mode not in ("off", "auto", "on"):
        raise ValueError("mono pin OCR mode must be off, auto or on")
    min_conf = float(min_conf)
    if not math.isfinite(min_conf):
        raise ValueError("min_conf must be finite")
    if mode == "off":
        return []
    if not include_names and not include_numbers:
        return []
    all_candidates, gray = mono_pin_boxes(doc, bgr, reads)
    candidates = [c for c in all_candidates if
                  (include_names if c["role"] == "pin_name" else include_numbers)]
    if not candidates:
        return []
    provider = get_pin_name_ocr(mode)
    if provider is None:
        return []
    threshold, output = max(ACCEPTANCE_THRESHOLD, min_conf), []
    for candidate in candidates:
        role = candidate["role"]
        x0, y0, x1, y1 = candidate["bbox"]
        glyph = gray[y0:y1, x0:x1].copy()
        for x, y in (candidate.get("cleanup_evidence") or {}).get("known_line_exclusions", []):
            if x0 <= x < x1 and y0 <= y < y1:
                glyph[y - y0, x - x0] = 255
        views, held = [], None
        if "glyph_integrity_unresolved" in candidate["flags"]:
            held = "glyph_integrity_unresolved"
        elif max(glyph.shape) + 2 * max(PADDINGS) > 384:
            held = "glyph_exceeds_verified_recognizer_size"
        else:
            for padding in PADDINGS:
                observed = provider.read_glyph(np.pad(glyph, padding, constant_values=255))
                views.append(dict(observed, padding=padding, rotation=0))
        agreed = _agreement(views, role, threshold)
        if held is None and agreed is None:
            held = "white_margin_views_disagree_or_below_threshold"
        text, confidence = agreed if agreed else (None, 0.0)
        proposed = {role: (text, confidence)} if agreed else {}
        metadata = {"role": role, "backend": "ppocr_v4_direct", "accepted": held is None,
                    "held_reason": held, "acceptance_threshold": threshold,
                    "rotation": 0, "orientation_source": "horizontal_physical_pin_row",
                    "view": "original_complete_glyph", "region": list(candidate["region"]),
                    "flags": list(candidate["flags"]),
                    "cleanup_evidence": candidate["cleanup_evidence"],
                    "selected_views": views, "raw_reads": views,
                    "proposed_read": proposed,
                    "confidence_definition": "minimum of two agreeing PP character-probability means"}
        if views:
            metadata["sampling"] = views[0].get("sampling")
        output.append({"bbox": list(candidate["bbox"]), "geometry_bbox": list(candidate["bbox"]),
                       "glyph_bbox": list(candidate["bbox"]), "pin_target": dict(candidate["pin_target"]),
                       "reads": proposed if held is None else {}, "name_only": role == "pin_name",
                       "number_only": role == "pin_number", "source": "mono_" + role + "_ppocr",
                       "mono_pin_ocr": metadata})
    _guard_recovered_names(doc, reads, output, all_candidates, bgr, min_conf=min_conf)
    return output
