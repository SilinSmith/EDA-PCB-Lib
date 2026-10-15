"""Read complete rotated pin words in lanes defined by observed color pins.

Geometry always remains in the original image. Recognition padding and 90 degree
rotation are separate, and none of the windows use expected names or numbers.
"""

from __future__ import annotations


import math


import cv2

import numpy as np


from .color_crops import COLOR_SOURCES


# Rotated one-digit crops are often much smaller than complete words. Add
# uncertain new words as alternatives, but do not let a weak rotated number
# displace an already observed numeral just because the new crop is tighter.
MIN_ROTATED_NUMBER_CONF = .85


def _finite(values, length):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        return None
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError):
        return None
    return numbers if all(math.isfinite(v) for v in numbers) else None


def _observations(doc, include_horizontal=False, grey_evidence=False):
    """Yield lane anchors for confirmed pins.

    ``grey_evidence`` additionally accepts the monochrome evidence records
    written by ``stage1/pin_evidence.py`` (``source`` missing or empty): those
    carry the same ``point_img``/``anchor_img``/``edge`` geometry, and their
    ``trust == "trusted"`` means the frame edge really is attached to a normal
    stroke -- the colour *source* is the only thing they lack.  Off by default
    so every existing caller keeps the colour-only behaviour.
    """
    for comp in doc.get("components") or []:
        if not isinstance(comp, dict) or comp.get("type") != "box":
            continue
        frame = _finite(comp.get("bbox_img"), 4)
        cid = comp.get("id")
        if not isinstance(cid, str) or frame is None or frame[2] <= frame[0] or frame[3] <= frame[1]:
            continue
        pins = (doc.get("pins") or {}).get(cid) or []
        evidence = (doc.get("pin_evidence") or {}).get(cid) or []
        for index, (raw, ev) in enumerate(zip(pins, evidence)):
            if not isinstance(ev, dict) or ev.get("edge") not in (("top", "bottom", "left", "right") if include_horizontal else ("top", "bottom")):
                continue
            point = _finite(raw, 2)
            observed = _finite(ev.get("point_img"), 2)
            anchor = _finite(ev.get("anchor_img"), 2)
            source = ev.get("source")
            source_ok = (isinstance(source, str) and source in COLOR_SOURCES) or (
                grey_evidence and (source is None or source == ""))
            if (point is None or observed is None or anchor is None
                    or not source_ok
                    or ev.get("trust") != "trusted"
                    or ev.get("frame_confirmed") is not True
                    or math.dist(point, observed) > .001):
                continue
            # A stale frame/anchor must not open a lane in a different region.
            edge = ev["edge"]
            normal, tangent = (1, 0) if edge in ("top", "bottom") else (0, 1)
            side = {"top": frame[1], "bottom": frame[3], "left": frame[0], "right": frame[2]}[edge]
            tangent_min, tangent_max = (frame[0], frame[2]) if tangent == 0 else (frame[1], frame[3])
            if abs(anchor[normal] - side) > 6 or not tangent_min - 2 <= anchor[tangent] <= tangent_max + 2:
                continue
            if abs(point[tangent] - anchor[tangent]) > 1 or (point[normal] - anchor[normal]) * (1 if edge in ("bottom", "right") else -1) <= 0:
                continue
            yield {"component_id": cid, "pin_index": index, "edge": ev["edge"],
                   "point_img": point, "anchor_img": anchor, "frame": frame}


def _ink(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chroma = image.max(2).astype(np.int16) - image.min(2).astype(np.int16)
    mask = (chroma < 30) & (gray < 190)
    return mask, np.where(mask, gray, 255).astype(np.uint8)


def _runs(on, max_gap):
    positions = np.flatnonzero(on)
    if not len(positions):
        return []
    runs, start, last = [], int(positions[0]), int(positions[0])
    for value in positions[1:]:
        value = int(value)
        if value - last > max_gap + 1:
            runs.append((start, last + 1))
            start = value
        last = value
    runs.append((start, last + 1))
    return runs


def _nearest_glyph(mask, region, anchor, edge, *, number=False, min_ratio=1.10):
    h, w = mask.shape
    x0, y0, x1, y1 = region
    x0, y0 = max(0, int(math.ceil(x0))), max(0, int(math.ceil(y0)))
    x1, y1 = min(w, int(math.floor(x1))), min(h, int(math.floor(y1)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    sub = mask[y0:y1, x0:x1]
    # Gaps between rotated letters are small relative to their glyph width.
    # A long blank region separates an opposite-edge word from this word.
    spans = _runs(sub.any(1), 3 if number else 6)
    choices = []
    for start, end in spans:
        ys, xs = np.nonzero(sub[start:end])
        if len(xs) < 5:
            continue
        box = [x0 + int(xs.min()), y0 + start + int(ys.min()),
               x0 + int(xs.max()) + 1, y0 + start + int(ys.max()) + 1]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 2 or bh < 3:
            continue
        if not number and bh < bw * min_ratio:
            continue
        distance = min(abs(anchor[1] - box[1]), abs(anchor[1] - box[3]))
        choices.append((distance, box))
    return min(choices, key=lambda v: v[0])[1] if choices else None


def pin_text_boxes(doc, image, reads=None, *, include_horizontal: bool = False,
                   grey_evidence: bool = False):
    """Return original-image ink boxes and a cleaned grayscale image.

    By default only vertical pin words are added. ``include_horizontal`` is an
    explicit experiment option. Monochrome pins and evidence which does not
    match the current point always retain the existing path.
    """
    _ = reads  # 契约参数：裁片器一律 (doc, image, reads=None)，本函数不消费 reads
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        return [], None
    mask, cleaned = _ink(image)
    observations = list(_observations(doc, include_horizontal, grey_evidence))
    out = []
    for ob in observations:
        frame, anchor, edge = ob["frame"], ob["anchor_img"], ob["edge"]
        vertical = edge in ("top", "bottom")
        tangent = 0 if vertical else 1
        relevant = [p for p in observations if p["component_id"] == ob["component_id"]
                    and (p["edge"] in ("top", "bottom")) == vertical]
        others = [p for p in relevant if abs(p["anchor_img"][tangent] - anchor[tangent]) > 1]
        distances = [abs(p["anchor_img"][tangent] - anchor[tangent]) for p in others]
        pitch = min(distances) if distances else min(frame[2] - frame[0] if vertical else frame[3] - frame[1], 44)
        half = max(5, min(22, pitch * .46))
        if vertical:
            left, right = max(frame[0] + 2, anchor[0] - half), min(frame[2] - 2, anchor[0] + half)
            if edge == "bottom":
                inside = [left, frame[1] + 2, right, anchor[1] - 2]
            else:
                inside = [left, anchor[1] + 2, right, frame[3] - 2]
            box = _nearest_glyph(mask, inside, anchor, edge)
        else:
            top, bottom = max(frame[1] + 2, anchor[1] - half), min(frame[3] - 2, anchor[1] + half)
            inside = ([anchor[0] + 2, top, frame[2] - 2, bottom] if edge == "left"
                      else [frame[0] + 2, top, anchor[0] - 2, bottom])
            # The word baseline can be a few pixels below the wire. Infer its
            # center from actual dark ink before restricting the adjacent lane;
            # a fixed wire-centered half-pitch can cut a trailing underscore.
            xa, ya, xb, yb = map(int, inside)
            xa, ya = max(0, xa), max(0, ya)
            xb, yb = min(mask.shape[1], xb), min(mask.shape[0], yb)
            if xb > xa and yb > ya:
                row_spans = _runs(mask[ya:yb, xa:xb].any(1), 2)
                row_spans = [(ya + a, ya + b) for a, b in row_spans if 3 <= b - a <= min(40, pitch * .9)]
                if row_spans:
                    ra, rb = min(row_spans, key=lambda span: abs((span[0] + span[1]) / 2 - anchor[1]))
                    center = (ra + rb) / 2
                    inside[1], inside[3] = max(frame[1] + 2, center - half), min(frame[3] - 2, center + half)
            transposed = [inside[1], inside[0], inside[3], inside[2]]
            bb = _nearest_glyph(mask.T, transposed, [anchor[1], anchor[0]], edge, min_ratio=.25)
            box = [bb[1], bb[0], bb[3], bb[2]] if bb is not None else None
        if box is None:
            continue
        meta = {k: v for k, v in ob.items() if k != "frame"}
        out.append({"bbox": box, "role": "pin_name", "pin_target": meta,
                    "region": inside, "vertical": vertical})
        # Read the external digit only where a rotated name establishes a
        # vertical text lane. Keep its exact glyph geometry separate from pad.
        if not vertical:
            continue
        point = ob["point_img"]
        lo, hi = sorted((point[1], anchor[1]))
        outside = [max(0, anchor[0] - min(half, 14)), lo + 2,
                   anchor[0] + min(half, 14), hi - 2]
        digit = _nearest_glyph(mask, outside, anchor, edge, number=True)
        if digit is not None:
            out.append({"bbox": digit, "role": "pin_number", "pin_target": meta,
                        "region": outside, "vertical": True})
    return out, cleaned


def _padded_crop(gray, box, padding=3):
    x0, y0, x1, y1 = map(int, box)
    core = gray[y0:y1, x0:x1]
    return np.pad(core, padding, constant_values=255)


def read_pin_texts(doc, reads, engine, bgr, *, min_conf: float = 0.0,
                   include_horizontal: bool = False, grey_evidence: bool = False):
    """Append role-only reads of whole words, never mutate existing reads.

    Both rotations are actually recognized. A component-level orientation
    vote from its complete name crops chooses the reading direction for digit
    crops; no presumed pin order or dictionary of expected words participates.
    """
    boxes, gray = pin_text_boxes(doc, bgr, include_horizontal=include_horizontal,
                                 grey_evidence=grey_evidence)
    if not boxes:
        return []
    recognizer = engine.recognizer()
    if recognizer is None:
        return []
    crops, crop_keys = [], []
    for index, box in enumerate(boxes):
        crop = _padded_crop(gray, box["bbox"])
        for rotation in ((90, -90) if box["vertical"] else (0,)):
            turned = np.rot90(crop, 1 if rotation == 90 else 3).copy() if rotation else crop
            crops.append(turned)
            crop_keys.append((index, rotation))
    atlas = np.full((sum(c.shape[0] + 4 for c in crops), max(c.shape[1] for c in crops)), 255, np.uint8)
    items, y = [], 0
    for crop, (index, _rotation) in zip(crops, crop_keys):
        height, width = crop.shape
        atlas[y:y + height, :width] = crop
        items.append(([0, y, width, y + height], boxes[index]["role"]))
        y += height + 4
    predictions = recognizer.read_many(atlas, items)
    variants = [[] for _ in boxes]
    for (index, rotation), (text, conf) in zip(crop_keys, predictions):
        variants[index].append({"rotation": rotation, "text": str(text), "conf": float(conf)})
    values = list(zip(boxes, variants))
    scores = {}
    for box, alternatives in values:
        if box["role"] != "pin_name" or not box["vertical"]:
            continue
        cid = box["pin_target"]["component_id"]
        for alt in alternatives:
            valid = any(ch.isalpha() for ch in alt["text"])
            scores[(cid, alt["rotation"])] = scores.get((cid, alt["rotation"]), 0) + (alt["conf"] if valid else 0)
    name_directions = {}
    for box, alternatives in values:
        if box["role"] != "pin_name":
            continue
        target = box["pin_target"]
        cid = target["component_id"]
        ranked = sorted(alternatives, key=lambda a: a["conf"] if any(ch.isalpha() for ch in a["text"]) else 0, reverse=True)
        if not box["vertical"]:
            direction = 0
        elif ranked[0]["conf"] - ranked[1]["conf"] >= .10:
            direction = ranked[0]["rotation"]
        else:
            direction = max((90, -90), key=lambda rot: scores.get((cid, rot), 0))
        name_directions[(cid, target["pin_index"])] = direction
    out = []
    for box, alternatives in values:
        target = box["pin_target"]
        cid = target["component_id"]
        direction = name_directions[(cid, target["pin_index"])]
        selected = next(a for a in alternatives if a["rotation"] == direction)
        role = box["role"]
        if (role == "pin_number"
                and (not math.isfinite(selected["conf"])
                     or selected["conf"] < MIN_ROTATED_NUMBER_CONF)):
            continue
        out.append({"bbox": list(box["bbox"]),
                    "reads": {role: (selected["text"], selected["conf"])},
                    "name_only": role == "pin_name", "number_only": role == "pin_number",
                    "source": "color_vertical_pin_text_crop" if box["vertical"] else "color_horizontal_pin_text_crop", "pin_target": target,
                    "pin_text_crop": {"rotation": direction, "alternatives": alternatives,
                                      "region": box["region"], "padding": 3}})
    return out
