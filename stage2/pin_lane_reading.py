"""Selective pin text recovery from observed geometry, never expected labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

import cv2
import numpy as np

from .assign import assign
from .color_crops import _remove_numeric_specks
from .isolated_ocr import read_isolated_many
from .pin_lane_crops import pin_lane_boxes
from .pin_text_crops import _observations


def _rectangle(value):
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return values if (len(values) == 4 and all(map(math.isfinite, values))
                      and values[2] > values[0] and values[3] > values[1]) else None


def _overlap(a, b):
    a, b = _rectangle(a), _rectangle(b)
    if a is None or b is None:
        return 0.0
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def _recovery_reasons(candidate, existing, duplicates):
    """Only observable failures open a new path; confidence alone is not truth."""
    reasons = []
    target = candidate["pin_target"]
    slot = "name" if candidate["role"] == "pin_name" else "number"
    old = existing.get("comps", {}).get(target["component_id"], {}).get(
        "pins", {}).get(target["pin_index"], {}).get(slot)
    if not old:
        reasons.append("missing_bound_role")
    else:
        glyph, old_box = _rectangle(candidate["bbox"]), _rectangle(old.get("box"))
        if glyph and old_box:
            area = (glyph[2]-glyph[0]) * (glyph[3]-glyph[1])
            coverage = _overlap(glyph, old_box) / max(area, 1)
            if .1 < coverage < .92:
                reasons.append("incomplete_bound_glyph")
        if slot == "number" and duplicates.get((target["component_id"], old.get("text")), 0) > 1:
            reasons.append("duplicate_bound_number")
        # Inspect the source that actually won binding. A dirty discarded
        # OCR alternative does not justify replacing an intact selected word.
        sources = (candidate.get("cleanup_evidence") or {}).get("source_boxes", [])
        for source in sources:
            if _rectangle(source.get("bbox")) != old_box:
                continue
            if source.get("crosses_frame"):
                reasons.append("mixed_bound_frame_box")
            if source.get("pin_row_count", 0) > 1:
                reasons.append("multiple_bound_pin_rows")
            if source.get("color_contamination_ratio", 0) >= .15:
                reasons.append("contaminated_bound_role")
    # A touched NEW search boundary is a completeness warning, not evidence
    # that an already valid OLD read should be replaced by another guess.
    return list(dict.fromkeys(reasons))


def _unsafe_number_source(candidate, old_number, *, target, doc, gray, colored, black):
    if not old_number:
        return False
    old_box = _rectangle(old_number.get("box"))
    if old_box is None:
        return True
    for source in ((candidate or {}).get("cleanup_evidence") or {}).get("source_boxes", []):
        if _rectangle(source.get("bbox")) == old_box:
            return (bool(source.get("crosses_frame"))
                    or source.get("pin_row_count", 0) > 1
                    or source.get("color_contamination_ratio", 0) >= .15)
    # A missing digit glyph is exactly when the companion may be unreliable.
    # Inspect the legacy sample directly instead of requiring a new crop.
    edge, anchor = target["edge"], target["anchor_img"]
    normal = 1 if edge in ("top", "bottom") else 0
    tangent = 1 - normal
    if old_box[normal] < anchor[normal] < old_box[normal+2]:
        return True
    row_count = 0
    for observation in _observations(doc, include_horizontal=True):
        if observation["component_id"] != target["component_id"] or observation["edge"] != edge:
            continue
        observed = observation["anchor_img"]
        row_count += old_box[tangent] <= observed[tangent] <= old_box[tangent+2]
    if row_count > 1:
        return True
    x0, y0, x1, y1 = old_box
    h, w = gray.shape
    xa, ya = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    xb, yb = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    if xb <= xa or yb <= ya:
        return True
    ncolor, nblack = int(colored[ya:yb, xa:xb].sum()), int(black[ya:yb, xa:xb].sum())
    return ncolor / max(1, ncolor+nblack) >= .15


def _threshold_reads(reads, min_conf):
    filtered = []
    for rec in reads:
        alternatives = {}
        for role, value in (rec.get("reads") or {}).items():
            text, confidence = value
            confidence = float(confidence)
            alternatives[role] = (text if math.isfinite(confidence) and confidence >= min_conf else "", confidence)
        filtered.append(dict(rec, reads=alternatives))
    return filtered


def _number(text):
    text = str(text).strip()
    return text if re.fullmatch(r"[0-9]{1,3}", text) and 1 <= int(text) <= 999 else None


def _name(text):
    text = str(text).strip()
    return text if text and any(ch.isalpha() for ch in text) else None


def _direction_votes(reads):
    votes, per_pin = defaultdict(Counter), {}
    for rec in reads:
        target, meta = rec.get("pin_target"), rec.get("pin_text_crop")
        if not isinstance(target, dict) or not isinstance(meta, dict) or not rec.get("name_only"):
            continue
        direction = meta.get("rotation")
        name = rec.get("reads", {}).get("pin_name")
        if direction not in (-90, 90) or not name or not _name(name[0]):
            continue
        confidence = float(name[1])
        if not math.isfinite(confidence) or confidence < .6:
            continue
        key = (target.get("component_id"), target.get("edge"))
        votes[key][direction] += float(name[1])
        per_pin[(target.get("component_id"), target.get("pin_index"))] = direction
    return votes, per_pin


def read_pin_lanes(doc, reads, engine, bgr, *, include_names=True,
                   include_numbers=True, cleanup="light", min_conf=0.0):
    """Append targeted, role-limited reads without altering original candidates.

    Both raw and cleaned inputs use the same isolated sampling convention.
    Symbol removal chooses a view by pixels, rather than its OCR confidence.
    The original pipeline continues to decide final component/pin output.
    """
    if cleanup not in ("light", "raw"):
        raise ValueError("cleanup must be 'light' or 'raw'")
    min_conf = float(min_conf)
    if not math.isfinite(min_conf):
        raise ValueError("min_conf must be finite")
    if bgr is None or getattr(bgr, "ndim", 0) != 3:
        return []
    recognizer = engine.recognizer()
    if recognizer is None:
        return []
    reads = _threshold_reads(reads, min_conf)
    candidates, clean_gray = pin_lane_boxes(doc, bgr, reads=reads)
    if not candidates:
        return []
    existing = assign(doc.get("components", []), doc.get("pins", {}), reads,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    duplicates = Counter((cid, (pin.get("number") or {}).get("text"))
                         for cid, comp in existing.get("comps", {}).items()
                         for pin in comp.get("pins", {}).values()
                         if pin.get("number"))
    selected = []
    for candidate in candidates:
        role = candidate.get("role")
        if (role == "pin_name" and not include_names) or (role == "pin_number" and not include_numbers):
            continue
        reasons = _recovery_reasons(candidate, existing, duplicates)
        if reasons and "glyph_integrity_unresolved" not in (candidate.get("flags") or []):
            selected.append(dict(candidate, recovery_reasons=reasons))
    if not selected:
        return []
    raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = bgr.max(2).astype(np.int16) - bgr.min(2).astype(np.int16)
    known_color = (chroma >= 50) & (raw_gray < 230)
    samples, keys = [], []
    for index, candidate in enumerate(selected):
        box = list(map(int, candidate["bbox"]))
        x0, y0, x1, y1 = box
        # The established small numeric speck rule is applied only to an OCR
        # copy. Re-tighten the sampling geometry after cleaning; fragmented 1
        # components taller/wider than 2px remain intact.
        raw_crop = raw_gray[y0:y1, x0:x1].copy()
        light_crop = clean_gray[y0:y1, x0:x1].copy()
        if candidate["role"] == "pin_number":
            ink, removed = _remove_numeric_specks(light_crop < 190)
            light_crop[removed] = 255
            yy, xx = np.nonzero(ink)
            if len(xx):
                xa, ya, xb, yb = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
                raw_crop, light_crop = raw_crop[ya:yb, xa:xb], light_crop[ya:yb, xa:xb]
                candidate["bbox"] = [x0+xa, y0+ya, x0+xb, y0+yb]
                candidate["isolated_pixels_removed"] = int(removed.sum())
        candidate["colored_glyph_pixels"] = int(np.count_nonzero(known_color[y0:y1, x0:x1]))
        candidate["changed_ink_pixels"] = int(np.count_nonzero(raw_crop != light_crop))
        crop_box = [0, 0, raw_crop.shape[1], raw_crop.shape[0]]
        for rotation in ((90, -90) if candidate.get("vertical") else (0,)):
            for view, gray in (("raw", raw_crop), ("light", light_crop)):
                samples.append({"gray": gray, "bbox": crop_box,
                                "role": candidate["role"], "rotation": rotation,
                                "padding": 3 if candidate["role"] == "pin_name" else 2})
                keys.append((index, rotation, view))
    predictions = read_isolated_many(recognizer, samples)
    alternatives = defaultdict(dict)
    for key, prediction in zip(keys, predictions):
        alternatives[key[0]][(key[1], key[2])] = prediction
    votes, per_pin = _direction_votes(reads)
    chosen, pending_numbers = [], []
    for index, candidate in enumerate(selected):
        target = candidate["pin_target"]
        changed = candidate["changed_ink_pixels"]
        # Preserve grayscale/antialiasing in unpolluted names. Thresholding
        # black letter edges is not the same operation as removing a symbol.
        use_light = (candidate["role"] == "pin_number"
                     or candidate["colored_glyph_pixels"] > 0)
        view = "light" if cleanup == "light" and use_light and changed else "raw"
        variants = alternatives[index]
        key = (target["component_id"], target["pin_index"])
        if not candidate.get("vertical"):
            rotation = 0
        elif candidate["role"] == "pin_name":
            valid = [(variants[(r, view)]["confidence"] if _name(variants[(r, view)]["text"]) else 0.0, r)
                     for r in (90, -90)]
            valid.sort(reverse=True)
            if valid[0][0] - valid[1][0] >= .1:
                rotation = valid[0][1]
            else:
                rotation = per_pin.get(key)
                if rotation is None:
                    local = votes.get((target["component_id"], target["edge"]), {})
                    rotation = max(local, key=local.get) if local else valid[0][1]
            result = variants[(rotation, view)]
            if _name(result["text"]) and result["confidence"] >= .6:
                per_pin[key] = rotation
                votes[(target["component_id"], target["edge"])][rotation] += result["confidence"]
        else:
            pending_numbers.append((index, candidate, variants, view, changed))
            continue
        chosen.append((candidate, variants, view, rotation, changed))
    for index, candidate, variants, view, changed in pending_numbers:
        target = candidate["pin_target"]
        rotation = per_pin.get((target["component_id"], target["pin_index"]))
        if rotation is None:
            local = votes.get((target["component_id"], target["edge"]), {})
            if local:
                rotation = max(local, key=local.get)
        if rotation is None:
            valid = [(r, variants[(r, view)]) for r in (90, -90)
                     if _number(variants[(r, view)]["text"])]
            if not valid:
                continue
            if len(valid) == 2 and valid[0][1]["text"] != valid[1][1]["text"]:
                # A directionless 6/9 remains ambiguous. Numbering order is
                # never used to manufacture the missing orientation.
                continue
            best_direction, result = max(valid, key=lambda v: v[1]["confidence"])
            if result["confidence"] < .85:
                continue
            rotation = best_direction
        chosen.append((candidate, variants, view, rotation, changed))
    output = []
    for candidate, variants, view, rotation, changed in chosen:
        result = variants[(rotation, view)]
        role = candidate["role"]
        text = _number(result["text"]) if role == "pin_number" else _name(result["text"])
        conf = float(result["confidence"])
        threshold = .55 if role == "pin_number" else .5
        target = candidate["pin_target"]
        old_number = existing.get("comps", {}).get(target["component_id"], {}).get(
            "pins", {}).get(target["pin_index"], {}).get("number")
        if role == "pin_number" and candidate.get("vertical") and old_number and text != old_number.get("text"):
            threshold = .85
        if text is None or not math.isfinite(conf) or conf < max(threshold, min_conf):
            continue
        output.append({"bbox": list(candidate["bbox"]),
                       "geometry_bbox": list(candidate["bbox"]),
                       "glyph_bbox": list(candidate["bbox"]),
                       "pin_target": dict(candidate["pin_target"]),
                       "reads": {role: (text, conf)},
                       "name_only": role == "pin_name", "number_only": role == "pin_number",
                       "source": "color_pin_lane_recovery",
                       "pin_lane_crop": {"role": role, "view": view, "rotation": rotation,
                           "recovery_reasons": candidate["recovery_reasons"],
                           "region": candidate.get("region"), "changed_ink_pixels": changed,
                           "colored_glyph_pixels": candidate["colored_glyph_pixels"],
                           "isolated_pixels_removed": candidate.get("isolated_pixels_removed", 0),
                           "cleanup_evidence": candidate.get("cleanup_evidence"),
                           "sampling": result.get("sampling"),
                           "alternatives": [{"rotation": r, "view": v, **value}
                                            for (r, v), value in variants.items()]}})
    # A recovered name can make an old symbol-contaminated false number
    # appear as a NEW final pin. Require a usable number observation for that
    # pair instead of silently activating the unreliable legacy number.
    rebound = assign(doc.get("components", []), doc.get("pins", {}), list(reads) + output,
                     pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    number_candidates = {(c["pin_target"]["component_id"], c["pin_target"]["pin_index"]): c
                         for c in candidates if c["role"] == "pin_number"}
    fresh_numbers = defaultdict(set)
    for rec in output:
        if rec["number_only"]:
            target = rec["pin_target"]
            fresh_numbers[(target["component_id"], target["pin_index"])].add(rec["reads"]["pin_number"][0])
    for rec in output:
        if not rec["name_only"]:
            continue
        target = rec["pin_target"]
        cid, pi = target["component_id"], target["pin_index"]
        old_pin = existing.get("comps", {}).get(cid, {}).get("pins", {}).get(pi, {})
        if old_pin.get("name") or not _unsafe_number_source(
                number_candidates.get((cid, pi)), old_pin.get("number"),
                target=target, doc=doc, gray=raw_gray, colored=known_color, black=clean_gray < 190):
            continue
        new_number = rebound.get("comps", {}).get(cid, {}).get("pins", {}).get(pi, {}).get("number")
        if not new_number or new_number.get("text") not in fresh_numbers[(cid, pi)]:
            rec["pin_lane_crop"]["proposed_read"] = dict(rec["reads"])
            rec["pin_lane_crop"]["held_reason"] = "unresolved_symbol_contaminated_partner_number"
            rec["reads"] = {}
    return output
