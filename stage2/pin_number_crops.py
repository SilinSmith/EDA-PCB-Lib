"""Conservative geometry repair for the extra pin-number recognition path.

Legacy name/number lanes are unchanged. Repairs use connected dark ink and
current trusted colour contacts only; recognized strings never set a crop.
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
#     number_repair_boxes(doc, image, reads)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import copy
import cv2
import numpy as np

from .pin_lane_crops import pin_lane_boxes
from .pin_text_crops import _observations


def _components(mask, region):
    x0, y0, x1, y1 = region
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask[y0:y1, x0:x1].astype(np.uint8), 8)
    return labels, [
        {"label": index, "bbox": [x0 + int(s[0]), y0 + int(s[1]),
                                    x0 + int(s[0] + s[2]), y0 + int(s[1] + s[3])],
         "pixels": int(s[4]), "width": int(s[2]), "height": int(s[3])}
        for index, s in enumerate(stats[1:], 1)]


def _bbox(mask, region):
    x0, y0, x1, y1 = region
    yy, xx = np.nonzero(mask[y0:y1, x0:x1])
    if not len(xx):
        return None
    return [x0 + int(xx.min()), y0 + int(yy.min()),
            x0 + int(xx.max()) + 1, y0 + int(yy.max()) + 1]


def _junction_residue(mask, region, target, near_colour):
    """Remove tiny detached ink only at a known contact, beside a real body.

    This does not remove small digit fragments elsewhere or a narrow 1. The
    additional colour-contact condition distinguishes junction antialiasing
    from legitimate disconnected strokes in a multi-digit number.
    """
    labels, components = _components(mask, region)
    if not any(c["pixels"] >= 8 and max(c["width"], c["height"]) >= 5
               for c in components):
        return [], np.zeros(labels.shape, dtype=bool)
    point = target["point_img"]
    removed, local = [], np.zeros(labels.shape, dtype=bool)
    x0, y0, _, _ = region
    for component in components:
        if component["pixels"] > 9 or max(component["width"], component["height"]) > 3:
            continue
        bx0, by0, bx1, by1 = component["bbox"]
        if not (bx0 - 1 <= point[0] <= bx1 + 1 and by0 - 1 <= point[1] <= by1 + 1):
            continue
        hit = labels == component["label"]
        colour = near_colour[y0:y0 + labels.shape[0], x0:x0 + labels.shape[1]]
        if not np.any(colour & hit):
            continue
        local |= hit
        removed.append({"bbox": component["bbox"], "pixels": component["pixels"],
                        "reason": "tiny_detached_ink_at_actual_colour_contact"})
    return removed, local


def _vertical_completion(mask, original, target, peers):
    """Restore at most one tangent pixel of an already seeded component.

    A full component must remain nearer this pin than every same-side peer,
    stop at white ink beyond the new cell, and have no adjacent unknown body.
    It cannot introduce a disconnected neighbouring digit or widen normal
    bounds across the actual frame. The 1px allowance corrects pixel rounding
    of a half-pitch cell, rather than opening an arbitrary new search window.
    """
    region = original["region"]
    height, width = mask.shape
    expanded = [max(0, region[0] - 1), region[1],
                min(width, region[2] + 1), region[3]]
    labels, components = _components(mask, expanded)
    ex0, ey0, ex1, ey1 = expanded
    gx0, gy0, gx1, gy1 = original["bbox"]
    # Seeds must lie in the previous ownership cell. New disconnected ink
    # outside that cell never becomes part of this number.
    sx0, sx1 = max(gx0, region[0]), min(gx1, region[2])
    sy0, sy1 = max(gy0, ey0), min(gy1, ey1)
    hits = set(np.unique(labels[sy0-ey0:sy1-ey0, sx0-ex0:sx1-ex0]).tolist()) - {0}
    selected = np.isin(labels, list(hits))
    evidence = {"method": "seeded_complete_ink_with_one_pixel_tangent_allowance",
                "previous_region": list(region), "ownership_region": expanded,
                "max_tangent_extension_pixels": 1, "integrity_unresolved": True}
    yy, xx = np.nonzero(selected)
    if not len(xx):
        evidence["unresolved_reason"] = "no_remaining_seeded_ink"
        return None, expanded, evidence
    completed = [ex0 + int(xx.min()), ey0 + int(yy.min()),
                 ex0 + int(xx.max()) + 1, ey0 + int(yy.max()) + 1]
    evidence["completed_bbox"] = completed
    center = target["anchor_img"][0]
    competing = []
    for component in components:
        if component["label"] not in hits:
            continue
        cx = (component["bbox"][0] + component["bbox"][2] - 1) / 2
        for peer in peers:
            if peer["pin_index"] == target["pin_index"]:
                continue
            px = peer["anchor_img"][0]
            if abs(cx - px) <= abs(cx - center) + .25:
                competing.append({"bbox": component["bbox"],
                                  "peer_pin_index": peer["pin_index"]})
    evidence["competing_peer_components"] = competing
    connected_beyond = False
    for column, side in ((ex0 - 1, 0), (ex1, -1)):
        if not 0 <= column < width:
            continue
        for local_y in np.flatnonzero(selected[:, side]):
            y = ey0 + int(local_y)
            if mask[max(0, y-1):min(height, y+2), column].any():
                connected_beyond = True
                break
    unknown = []
    for component in components:
        if component["label"] in hits:
            continue
        box = component["bbox"]
        gap = max(completed[0] - box[2], box[0] - completed[2], 0)
        overlap = min(completed[3], box[3]) > max(completed[1], box[1])
        if overlap and gap <= 2:
            unknown.append({"bbox": box, "pixels": component["pixels"]})
    evidence["connected_ink_beyond_allowance"] = connected_beyond
    evidence["nearby_unknown_components"] = unknown
    evidence["integrity_unresolved"] = bool(competing or connected_beyond or unknown)
    if competing:
        evidence["unresolved_reason"] = "seed_has_competing_observed_pin_owner"
    elif connected_beyond:
        evidence["unresolved_reason"] = "seed_continues_beyond_one_pixel_allowance"
    elif unknown:
        evidence["unresolved_reason"] = "nearby_unseeded_ink_has_unknown_owner"
    return completed, expanded, evidence


def number_repair_boxes(doc, image, reads=None, *, grey_evidence: bool = False):
    """Return only number candidates and an independent cleaned gray copy.

    Metadata preserves the original geometry/flags as well as every removed
    contact fragment and completion decision. Input documents, OCR records,
    and image pixels are never changed. Unproven clipping stays unresolved.
    ``grey_evidence`` widens eligibility to monochrome trusted evidence.
    """
    candidates, cleaned = pin_lane_boxes(doc, image, reads=reads, grey_evidence=grey_evidence)
    if cleaned is None:
        return [], None
    cleaned = cleaned.copy()
    observations = list(_observations(doc, include_horizontal=True, grey_evidence=grey_evidence))
    observed = {(o["component_id"], o["pin_index"]): o for o in observations}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chroma = image.max(2).astype(np.int16) - image.min(2).astype(np.int16)
    colour = (chroma >= 50) & (gray < 230)
    near_colour = cv2.dilate(colour.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = []
    for base in candidates:
        if base["role"] != "pin_number":
            continue
        candidate = copy.deepcopy(base)
        target = candidate["pin_target"]
        key = (target["component_id"], target["pin_index"])
        observation = observed.get(key)
        if observation is None:
            continue
        region = candidate["region"]
        mask = cleaned < 190
        removed, local = _junction_residue(mask, region, target, near_colour)
        if removed:
            x0, y0, x1, y1 = region
            cleaned[y0:y1, x0:x1][local] = 255
            mask = cleaned < 190
        repair = {"previous_bbox": list(base["bbox"]),
                  "previous_flags": list(base["flags"]),
                  "previous_completion": copy.deepcopy(base["cleanup_evidence"].get("glyph_completion")),
                  "contact_residue_removed": removed,
                  "independent_of_ocr_text": True, "changed": bool(removed)}
        flags = set(candidate["flags"])
        # Re-tighten only the already selected glyph. The broader ownership
        # cell can contain a detached peer fragment deliberately left out by
        # the legacy seeded completion, so its full ink union is not a crop.
        glyph = _bbox(mask, candidate["bbox"])
        if glyph is None:
            continue
        # Only previously incomplete vertical digits receive the 1px cell
        # repair. Ordinary correct candidates retain their exact geometry.
        if candidate.get("vertical") and "glyph_integrity_unresolved" in flags:
            peers = [o for o in observations if o["component_id"] == target["component_id"]
                     and o["edge"] == target["edge"]]
            completed, expanded, completion = _vertical_completion(mask, candidate, target, peers)
            repair["completion"] = completion
            if completed is not None and not completion["integrity_unresolved"]:
                glyph, region = completed, expanded
                flags.discard("glyph_integrity_unresolved")
                flags.add("number_glyph_completed_with_pixel_evidence")
                repair["changed"] = True
        if removed:
            flags.add("colour_contact_residue_removed")
        candidate["bbox"] = list(glyph)
        candidate["glyph_bbox"] = list(glyph)
        candidate["region"] = list(region)
        candidate["flags"] = sorted(flags)
        candidate["number_crop_repair"] = repair
        x0, y0, x1, y1 = glyph
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask[y0:y1, x0:x1].astype(np.uint8), 8)
        metadata = candidate["cleanup_evidence"]
        metadata["glyph_pixels"] = int(np.count_nonzero(mask[y0:y1, x0:x1]))
        metadata["glyph_component_count"] = count - 1
        metadata["glyph_component_stats"] = stats[1:].tolist()
        metadata["black_components_removed"] += len(removed)
        if repair.get("completion") and not repair["completion"]["integrity_unresolved"]:
            # Preserve the old completion above, but report current integrity
            # consistently in the established field used by crop auditors.
            metadata["glyph_completion"] = copy.deepcopy(repair["completion"])
        metadata["number_crop_repair"] = copy.deepcopy(repair)
        out.append(candidate)
    return out, cleaned
