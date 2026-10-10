"""Confirm observed colour-pin digits using complete original glyph pixels.

Two fixed white-margin views must agree. No expected label, numbering order,
dictionary correction or highest-confidence view selection is used.
"""
from __future__ import annotations

import math

import numpy as np

from .pin_lane_reading import _number, _threshold_reads
from .assign import assign
from .isolated_ocr import read_isolated_many
from .pin_name_reading import _directions
from .pin_name_ocr import get_pin_name_ocr
from .pin_number_crops import number_repair_boxes


NUMBER_ACCEPTANCE_THRESHOLD = .90
PADDINGS = (2, 4)
SOURCE = "color_pin_number_ppocr"


def _agreement(variants, threshold):
    """Both predefined samples must independently pass the same fixed gate."""
    if len(variants) != len(PADDINGS):
        return None
    values = []
    for padding, result in zip(PADDINGS, variants):
        text, conf = _number(result.get("text")), float(result.get("confidence", 0.0))
        if (result.get("padding") != padding or text is None
                or not math.isfinite(conf) or conf < threshold):
            return None
        values.append((text, conf))
    if values[0][0] != values[1][0]:
        return None
    return values[0][0], min(v[1] for v in values)


def read_pin_numbers(doc, reads, engine, bgr, *, min_conf=0.0):
    """Add qualified digits; missing optional PP runtime keeps the old path.

    ``engine`` is retained for the pipeline's reader interface; this recognizer
    does not use CRNN resampling. Coordinates describe original glyphs and
    contacts. Synthetic margins are recorded only as recognition metadata.
    """
    min_conf = float(min_conf)
    if not math.isfinite(min_conf):
        raise ValueError("min_conf must be finite")
    if bgr is None or getattr(bgr, "ndim", 0) != 3:
        return []
    reads = _threshold_reads(reads, min_conf)
    candidates, cleaned = number_repair_boxes(doc, bgr, reads=reads)
    candidates = [c for c in candidates
                  if "glyph_integrity_unresolved" not in c.get("flags", [])]
    if not candidates:
        return []
    provider = get_pin_name_ocr("auto")
    if provider is None:
        return []
    votes, directions = _directions(doc, reads)
    existing = assign(doc.get("components", []), doc.get("pins", {}), reads,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    recognizer = engine.recognizer() if engine is not None else None
    threshold = max(NUMBER_ACCEPTANCE_THRESHOLD, min_conf)
    output = []
    for candidate in candidates:
        target = candidate["pin_target"]
        key = (target["component_id"], target["pin_index"])
        rotation, origin = 0, "horizontal"
        if candidate.get("vertical"):
            rotation = directions.get(key)
            origin = "observed_same_pin_name_direction"
            if rotation is None:
                side_votes = votes.get((key[0], target["edge"]), {})
                if side_votes:
                    winner = max(side_votes, key=side_votes.get)
                    total = sum(side_votes.values())
                    if side_votes[winner] >= .75 * total:
                        rotation = winner
                        origin = "same_side_name_direction_agreement"
        x0, y0, x1, y1 = map(int, candidate["bbox"])
        glyph = cleaned[y0:y1, x0:x1].copy()
        rotations = (rotation,) if rotation is not None else (90, -90)
        by_direction = {}
        for direction in rotations:
            oriented = np.ascontiguousarray(np.rot90(glyph, direction // 90))
            variants = []
            for padding in PADDINGS:
                padded = np.pad(oriented, padding, constant_values=255)
                result = provider.read_glyph(padded)
                variants.append(dict(result, padding=padding, rotation=direction))
            by_direction[direction] = variants
        held = None
        if rotation is None:
            eligible = [(r, _agreement(v, threshold)) for r, v in by_direction.items()]
            eligible = [(r, value) for r, value in eligible if value is not None]
            if len(eligible) == 2 and eligible[0][1][0] == eligible[1][1][0]:
                rotation = eligible[0][0]
                origin = "both_directions_same_number"
            elif len(eligible) == 1 and eligible[0][1][0] not in ("6", "9"):
                rotation = eligible[0][0]
                origin = "single_accepted_non_6_9_direction"
            else:
                held = "unresolved_number_direction"
        selected = by_direction.get(rotation, next(iter(by_direction.values())))
        agreed = _agreement(selected, threshold)
        if held is None and agreed is None:
            held = "white_margin_views_disagree_or_below_threshold"
        text, conf = agreed if agreed is not None else (None, 0.0)
        old_number = existing.get("comps", {}).get(key[0], {}).get("pins", {}).get(key[1], {}).get("number")
        verification = None
        if held is None and old_number and text == old_number.get("text"):
            # Confirmation does not justify reranking an already bound digit.
            held = "unchanged_bound_number"
        elif held is None and recognizer is not None:
            verification = read_isolated_many(recognizer, [{
                "gray": glyph, "bbox": [0, 0, glyph.shape[1], glyph.shape[0]],
                "role": "pin_number", "rotation": rotation, "padding": 2}])[0]
            clean_number = _number(verification["text"])
            if (clean_number is not None and clean_number != text
                    and verification["confidence"] >= max(.90, min_conf)):
                held = "strong_clean_crnn_disagreement"
        output.append({
            "bbox": list(candidate["bbox"]), "geometry_bbox": list(candidate["bbox"]),
            "glyph_bbox": list(candidate["bbox"]), "pin_target": dict(target),
            "reads": {"pin_number": (text, conf)} if held is None else {},
            "number_only": True, "name_only": False, "source": SOURCE,
            "pin_number_ocr": {
                "backend": "ppocr_v4_direct", "accepted": held is None,
                "held_reason": held, "acceptance_threshold": threshold,
                "rotation": rotation, "orientation_source": origin,
                "view": "light_original_glyph", "flags": list(candidate.get("flags", [])),
                "region": candidate.get("region"),
                "cleanup_evidence": candidate.get("cleanup_evidence"),
                "number_repair": candidate.get("number_crop_repair"),
                "previous_number": old_number, "clean_crnn_verification": verification,
                "sampling": selected[0]["sampling"], "selected_views": selected,
                "alternatives": [v for views in by_direction.values() for v in views],
                "confidence_definition": "minimum of two agreeing PP character-probability means",
            },
        })
    return output
