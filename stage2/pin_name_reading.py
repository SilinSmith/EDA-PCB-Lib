"""Observe complete, geometrically targeted colour-pin names with optional OCR.

Only name evidence is added. Existing reads, coordinates and number recognition
stay intact; no expected strings or numbering order enter recognition.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 【接口】统一约定（队友分支 13 个模块，2026-10-16 统一；只改签名不改识别逻辑）
#   读数器   read_xxx(doc, reads, engine, bgr, *, min_conf=0.0, **opts) -> List[dict]
#   裁片器   xxx_boxes(doc, image, reads=None, **opts)                 -> List[dict]
#   返回     纯追加：只返回**新增**候选（{"bbox":..., "reads":{role:(text,conf)}}），
#            绝不修改入参、绝不删除既有读数；关掉开关时下游逐位不变
#   开关     一律由 `stage2/run.py` 的 `PIN_READERS` 注册表读取，模块自身不读 env
#   挂载     workflow 节点 ⑦c/⑦d/⑦e/⑦f/⑦g（`tools/flow/graph.json`）
#   来源     队友分支 `origin/codex/improve-pin-network-parsing`
#   本模块入口：
#     read_pin_names(doc, reads, engine, bgr, *, mode, min_conf)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import Counter, defaultdict
import math

import cv2
import numpy as np
from PIL import Image

from .assign import assign, _target_index
from .isolated_ocr import prepare_isolated_sample
from .pin_lane_crops import pin_lane_boxes
from .pin_lane_reading import _name, _number, _threshold_reads, _unsafe_number_source
from .pin_name_ocr import get_pin_name_ocr


NAME_ACCEPTANCE_THRESHOLD = .90
SOURCE = "color_pin_name_ppocr"


def _current_target(doc, rec, target):
    cid = target.get("component_id") if isinstance(target, dict) else None
    if cid is None:
        return None
    return _target_index(dict(rec, pin_target=target), cid,
                         doc.get("pins", {}).get(cid, []),
                         doc.get("pin_evidence", {}).get(cid))


def _directions(doc, reads):
    """Use observed name orientation, including the prior lane recovery."""
    votes, per_pin = defaultdict(Counter), {}
    for rec in reads:
        target = rec.get("pin_target")
        meta = rec.get("pin_text_crop") or rec.get("pin_lane_crop")
        if not isinstance(target, dict) or not isinstance(meta, dict) or not rec.get("name_only"):
            continue
        if _current_target(doc, rec, target) in (None, -1):
            continue
        rotation = meta.get("rotation")
        value = (rec.get("reads") or {}).get("pin_name")
        if rotation not in (-90, 90) or not value or not _name(value[0]):
            continue
        conf = float(value[1])
        if not math.isfinite(conf) or conf < .6:
            continue
        key = (target.get("component_id"), target.get("pin_index"))
        per_pin[key] = rotation
        votes[(key[0], target.get("edge"))][rotation] += conf
    return votes, per_pin


def _confirmed_numbers(doc, reads, min_conf):
    """Current targeted observations can confirm a partner, never invent it."""
    values = defaultdict(set)
    for rec in reads:
        if not rec.get("number_only"):
            continue
        target = rec.get("pin_target") or rec.get("color_crop")
        if not isinstance(target, dict):
            continue
        cid = target.get("component_id")
        index = _current_target(doc, rec, target)
        if index is None or index == -1:
            continue
        value = (rec.get("reads") or {}).get("pin_number")
        if (value and _number(value[0]) and math.isfinite(float(value[1]))
                and float(value[1]) >= max(.55, min_conf)):
            values[(cid, index)].add(value[0])
    return values


def read_pin_names(doc, reads, engine, bgr, *, mode: str = "auto",
                   min_conf: float = 0.0):
    """Return role-limited observations; weak/held ones have empty ``reads``.

    The optional recognizer receives the same isolated gray32 input used in
    the controlled CRNN/PP comparison, then applies its own 48px normalization.
    Confidence is an acceptance gate, not a score comparable to CRNN's score.
    """
    if mode not in ("off", "auto", "on"):
        raise ValueError("pin name OCR mode must be off, auto or on")
    min_conf = float(min_conf)
    if not math.isfinite(min_conf):
        raise ValueError("min_conf must be finite")
    if mode == "off" or bgr is None or getattr(bgr, "ndim", 0) != 3:
        return []
    reads = _threshold_reads(reads, min_conf)
    candidates, cleaned = pin_lane_boxes(doc, bgr, reads=reads)
    names = [c for c in candidates if c["role"] == "pin_name"
             and "glyph_integrity_unresolved" not in (c.get("flags") or [])]
    if not names:
        return []
    provider = get_pin_name_ocr(mode)
    if provider is None:
        return []
    # Preparation is needed only after the optional runtime is available.
    from textrec.dataset import crop_gray
    raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = bgr.max(2).astype(np.int16) - bgr.min(2).astype(np.int16)
    colored = (chroma >= 50) & (raw < 230)
    votes, directions = _directions(doc, reads)
    lane_views = {}
    for rec in reads:
        target, meta = rec.get("pin_target"), rec.get("pin_lane_crop")
        if (isinstance(target, dict) and isinstance(meta, dict) and meta.get("role") == "pin_name"
                and _current_target(doc, rec, target) not in (None, -1)):
            lane_views[(target.get("component_id"), target.get("pin_index"))] = meta.get("view")
    threshold = max(NAME_ACCEPTANCE_THRESHOLD, min_conf)
    output = []
    for candidate in names:
        target = candidate["pin_target"]
        key = (target["component_id"], target["pin_index"])
        view = lane_views.get(key, "raw")
        if view not in ("raw", "light"):
            view = "raw"
        x0, y0, x1, y1 = map(int, candidate["bbox"])
        gray = (cleaned if view == "light" else raw)[y0:y1, x0:x1].copy()
        rotation, origin = 0, "horizontal"
        if candidate.get("vertical"):
            rotation = directions.get(key)
            origin = "observed_name_direction"
            if rotation is None:
                side_votes = votes.get((key[0], target["edge"]), {})
                if side_votes:
                    rotation = max(side_votes, key=side_votes.get)
                    origin = "same_side_name_vote"
        rotations = (rotation,) if rotation is not None else (90, -90)
        variants = []
        for direction in rotations:
            canvas, bbox, shared = prepare_isolated_sample({
                "gray": gray, "bbox": [0, 0, gray.shape[1], gray.shape[0]],
                "role": "pin_name", "rotation": direction, "padding": 3})
            normalized = np.asarray(crop_gray(Image.fromarray(canvas), bbox, 32, 384))
            result = provider.read_normalized(normalized)
            variants.append(dict(result, rotation=direction, isolated_sampling=shared))
        selected = variants[0]
        held_reason = None
        if rotation is None:
            valid = [r for r in variants if _name(r["text"]) and r["confidence"] >= threshold]
            if len(valid) == 1:
                selected = valid[0]
                origin = "single_accepted_direction"
            elif len(valid) == 2 and valid[0]["text"] == valid[1]["text"]:
                selected = valid[0]
                origin = "both_directions_agree"
            else:
                held_reason = "unresolved_name_direction"
                origin = "unresolved_both_directions"
        text, conf = _name(selected["text"]), float(selected["confidence"])
        if held_reason is None and (not text or not math.isfinite(conf) or conf < threshold):
            held_reason = "below_name_acceptance_threshold" if text else "not_a_name"
        proposed = {"pin_name": (text, conf)} if text else {}
        metadata = {"role": "pin_name", "backend": "ppocr_v4", "view": view,
                    "rotation": selected["rotation"], "orientation_source": origin,
                    "acceptance_threshold": threshold, "accepted": held_reason is None,
                    "sampling": selected["sampling"],
                    "isolated_sampling": selected["isolated_sampling"],
                    "region": candidate.get("region"), "flags": candidate.get("flags", []),
                    "cleanup_evidence": candidate.get("cleanup_evidence"),
                    "alternatives": variants, "proposed_read": proposed}
        if held_reason:
            metadata["held_reason"] = held_reason
        output.append({"bbox": list(candidate["bbox"]), "geometry_bbox": list(candidate["bbox"]),
                       "glyph_bbox": list(candidate["bbox"]), "pin_target": dict(target),
                       "reads": proposed if held_reason is None else {}, "name_only": True,
                       "number_only": False, "source": SOURCE, "pin_name_ocr": metadata})
    # Preserve the existing protection against activating a dirty old number
    # merely by recovering its formerly missing name (e.g. NC paired with 25).
    existing = assign(doc.get("components", []), doc.get("pins", {}), reads,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    for rec in output:
        target = rec["pin_target"]
        old = existing.get("comps", {}).get(target["component_id"], {}).get("pins", {}).get(target["pin_index"], {})
        # Store actual prior observations; do not alter the new confidence.
        rec["pin_name_ocr"]["previous_name"] = old.get("name")
    rebound = assign(doc.get("components", []), doc.get("pins", {}), list(reads) + output,
                     pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    number_candidates = {(c["pin_target"]["component_id"], c["pin_target"]["pin_index"]): c
                         for c in candidates if c["role"] == "pin_number"}
    confirmed = _confirmed_numbers(doc, reads, min_conf)
    for rec in output:
        target = rec["pin_target"]
        cid, index = target["component_id"], target["pin_index"]
        old = existing.get("comps", {}).get(cid, {}).get("pins", {}).get(index, {})
        # Preserve the original name evidence solely for NUMBER arbitration.
        # PP's arithmetic confidence cannot rerank a duplicate number against
        # CRNN's geometric confidence. Its real confidence remains unchanged.
        if not rec["reads"]:
            continue
        if old.get("name") or not _unsafe_number_source(
                number_candidates.get((cid, index)), old.get("number"),
                target=target, doc=doc, gray=raw, colored=colored, black=cleaned < 190):
            continue
        new_number = rebound.get("comps", {}).get(cid, {}).get("pins", {}).get(index, {}).get("number")
        if not new_number or new_number.get("text") not in confirmed[(cid, index)]:
            rec["pin_name_ocr"].update(accepted=False,
                held_reason="unresolved_symbol_contaminated_partner_number")
            rec["reads"] = {}
    return output
