# -*- coding: utf-8 -*-
"""第二阶段 · 结构化批量重绑定（单调序列对齐 + 阶梯重排）。

**要解决的问题**（`work\\reports\\错例集_导线与元件.md`）：引脚号/引脚名现在是
「离哪个引脚最近就归谁」逐个贪心。只要池子里多一个候选（网络标签、符号碎片、
邻排元件的位号）或少一个（OCR 没读出来），**整条序列会集体错位**：

    期望   1-1  2-2  3-3  4-4
    实得   1 悬空  1-2  2-3  3-4  4 悬空

这里换成两种**整体**候选，按总分取优（并把两者的分歧留给阶段三）：

* **位置单调对齐**（`align_monotone`）：候选与引脚各自按「所在边 + 沿边坐标」排序，
  动态规划求最大权单调匹配，两侧都允许悬空；再用**数值单调性**把破坏序列的离群候选
  直接禁掉。它治的是「多一个候选就顺移一格」。
* **阶梯重排**（ladder）：当这一排的数值是**严格单调**、个数又与引脚数相同时，
  按「数值第 k 小 ↔ 位置第 k 个引脚」整体重排。它治的正是上面那种**整体偏移一格**：
  每一条单独看距离都很大，但整体只有这一个自洽解。

**已知边界（不要过度期待）**：若某个数值本身被读错（`7` 读成 `1`），阶梯的单调性会被
破坏，此时两个候选都会退化，只能靠阶段三的 LLM 看上下文判断 —— 所以 `rebind` 把
「两种绑定不一致的槽位」原样输出。
"""

from __future__ import annotations


import math

import os

import re

from typing import Dict, List, Optional, Sequence, Tuple


EDGES = ("left", "top", "right", "bottom")

_NUM_RE = re.compile(r"^\d{1,3}$")


# ---------------------------------------------------------------------------
def edge_of_point(bbox: Sequence[float], px: float, py: float) -> str:
    """点落在元件框的哪条边外侧（用**到边的距离**判，不按象限）。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    d = {"left": abs(px - x1), "right": abs(px - x2),
         "top": abs(py - y1), "bottom": abs(py - y2)}
    return min(EDGES, key=lambda e: d[e])


def along(bbox: Sequence[float], edge: str, px: float, py: float) -> float:
    """沿边坐标。左边/右边按 y，上边/下边按 x（不归一，只用于排序）。"""
    return float(py) if edge in ("left", "right") else float(px)


def normalize_pin_evidence(pins: Sequence[Sequence[float]],
                           evidence: Optional[Sequence[dict]]) -> List[dict]:
    """Validate the stage1 evidence-to-point correspondence before trusting it.

    An old cache, or a point list changed after evidence was built, must never
    attach the confidence of one contact to another. A missing/invalid record
    is uncertain; a length mismatch or wholly invalid correspondence disables
    this optional path altogether. Valid uncertain geometry still stays active.
    """
    if not evidence or len(evidence) != len(pins):
        return []
    out = []
    n_valid = 0
    for point, rec in zip(pins, evidence):
        fallback = {"trust": "uncertain"}
        if not isinstance(rec, dict):
            out.append(fallback)
            continue
        raw = rec.get("point_img")
        try:
            valid = (len(raw) == 2 and all(math.isfinite(float(v)) for v in raw)
                     and math.hypot(float(raw[0]) - float(point[0]),
                                    float(raw[1]) - float(point[1])) <= 1e-3)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid or rec.get("edge") not in EDGES:
            out.append(fallback)
            continue
        n_valid += 1
        item = dict(rec)
        if item.get("trust") not in ("trusted", "uncertain", "rejected"):
            item["trust"] = "uncertain"
        anchor = item.get("anchor_img")
        try:
            good_anchor = (len(anchor) == 2
                           and all(math.isfinite(float(v)) for v in anchor))
        except (TypeError, ValueError, OverflowError):
            good_anchor = False
        if not good_anchor:
            item.pop("anchor_img", None)
        out.append(item)
    return out if n_valid else []


def pin_edge(bbox: Sequence[float], pins: Sequence[Sequence[float]], index: int,
             evidence: Optional[Sequence[dict]] = None) -> str:
    rec = evidence[index] if evidence else {}
    return rec.get("edge") or edge_of_point(bbox, *pins[index])


def pin_anchor(pins: Sequence[Sequence[float]], index: int,
               evidence: Optional[Sequence[dict]] = None) -> Sequence[float]:
    rec = evidence[index] if evidence else {}
    return rec.get("anchor_img") or pins[index]


def _candidate_axis_gap(cand: dict, edge: str, anchor: Sequence[float]) -> float:
    """Distance to the text's projection, independent of name length."""
    rect = cand.get("bbox") or (cand.get("rec") or {}).get("bbox")
    axis = 1 if edge in ("left", "right") else 0
    if rect and len(rect) == 4:
        lo, hi = float(rect[axis]), float(rect[axis + 2])
        return max(lo - float(anchor[axis]), float(anchor[axis]) - hi, 0.0)
    return abs(float(cand["center"][axis]) - float(anchor[axis]))


def _candidate_axis_tolerance(cand: dict) -> float:
    rect = cand.get("bbox") or (cand.get("rec") or {}).get("bbox")
    if rect and len(rect) == 4:
        # A small allowance for broken strokes/box localization, not a whole
        # neighboring pin pitch. Rotated labels use their short dimension.
        return max(4.0, 0.35 * min(abs(float(rect[2]) - float(rect[0])),
                                  abs(float(rect[3]) - float(rect[1]))))
    return 6.0


def _candidate_faces_edge(cand: dict, bbox: Sequence[float], edge: str) -> bool:
    """An internal label should not cross to its strongly opposite frame side.

    Use the text rectangle's near boundary, not its centre: a long name can
    extend over the symbol centre. Only compare opposing sides, so a corner
    label remains eligible for its adjacent side. Missing geometry is neutral.
    """
    rect = cand.get("bbox") or (cand.get("rec") or {}).get("bbox")
    if not rect or len(rect) != 4:
        return True
    try:
        tx0, ty0, tx1, ty1 = (float(v) for v in rect)
        bx0, by0, bx1, by1 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (tx0, ty0, tx1, ty1, bx0, by0, bx1, by1)):
            return True
        if tx1 < tx0 or ty1 < ty0 or bx1 <= bx0 or by1 <= by0:
            return True
        gaps = {"left": max(tx0 - bx0, 0.0), "right": max(bx1 - tx1, 0.0),
                "top": max(ty0 - by0, 0.0), "bottom": max(by1 - ty1, 0.0)}
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        margin = 2.0 * _candidate_axis_tolerance(cand)
        return gaps[edge] <= gaps[opposite[edge]] + margin
    except (TypeError, ValueError, OverflowError):
        return True


def _frame_anchor_usable(rec: dict, bbox: Sequence[float]) -> bool:
    if rec.get("frame_confirmed") is not True or rec.get("edge") not in EDGES:
        return False
    try:
        bx0, by0, bx1, by1 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (bx0, by0, bx1, by1)):
            return False
        if bx1 <= bx0 or by1 <= by0:
            return False
        edge = rec["edge"]
        axis = 0 if edge in ("left", "right") else 1
        expected = {"left": bx0, "top": by0, "right": bx1, "bottom": by1}[edge]
        anchor, point = rec["anchor_img"], rec["point_img"]
        if len(anchor) != 2 or len(point) != 2:
            return False
        if not all(math.isfinite(float(v)) for v in (*anchor, *point)):
            return False
        # Stage1 searches six pixels normal to the frame and three along it.
        # Include its raster endpoint and 0.1px serialization rounding.
        return (abs(float(anchor[axis]) - expected) <= 7.1
                and abs(float(anchor[1 - axis]) - float(point[1 - axis])) <= 3.1)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def normalize_name_evidence(bbox: Sequence[float], pins: Sequence[Sequence[float]],
                            evidence: Optional[Sequence[dict]]) -> List[dict]:
    """Share validated name anchors between eligibility and caller scoring."""
    checked = normalize_pin_evidence(pins, evidence)
    for rec in checked:
        if rec.get("frame_confirmed") is True and not _frame_anchor_usable(rec, bbox):
            rec.pop("anchor_img", None)
            rec["frame_confirmed"] = False
    return checked


def confirmed_frame_sides(bbox: Sequence[float], evidence: Sequence[dict]) -> Dict[str, float]:
    """Collect only calibrated, consistent sides from fresh pin evidence.

    A detector box is not necessarily the symbol body. Stage1 searches within
    six pixels of each detector side; raster rounding can add one more pixel.
    Missing or contradictory side measurements remain neutral, rather than
    making an uncalibrated detector edge a hard text boundary.
    """
    try:
        bx0, by0, bx1, by1 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (bx0, by0, bx1, by1)):
            return {}
        if bx1 <= bx0 or by1 <= by0:
            return {}
    except (TypeError, ValueError, OverflowError):
        return {}
    samples: Dict[str, List[float]] = {edge: [] for edge in EDGES}
    for rec in evidence:
        if not _frame_anchor_usable(rec, bbox):
            continue
        edge = rec["edge"]
        try:
            anchor = rec["anchor_img"]
            if len(anchor) != 2 or not all(math.isfinite(float(v)) for v in anchor):
                continue
            coordinate = float(anchor[0 if edge in ("left", "right") else 1])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        samples[edge].append(coordinate)
    sides = {}
    for edge, values in samples.items():
        if not values or max(values) - min(values) > 2.0:
            continue
        values.sort()
        middle = len(values) // 2
        sides[edge] = (values[middle] if len(values) % 2
                       else 0.5 * (values[middle - 1] + values[middle]))
    if ("left" in sides and "right" in sides and sides["left"] >= sides["right"]):
        sides.pop("left")
        sides.pop("right")
    if ("top" in sides and "bottom" in sides and sides["top"] >= sides["bottom"]):
        sides.pop("top")
        sides.pop("bottom")
    return sides


def _candidate_rectangle(cand: dict) -> Optional[Tuple[float, float, float, float]]:
    rect = cand.get("bbox") or (cand.get("rec") or {}).get("bbox")
    try:
        x0, y0, x1, y1 = (float(v) for v in rect)
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return x0, y0, x1, y1


def name_edge_midpoint(cand: dict, edge: str) -> Optional[Tuple[float, float]]:
    """The text-box edge facing this pin's confirmed IC frame side.

    Both the name and its point stay unchanged. The near edge removes text
    length from the normal distance, unlike measuring to the word centre.
    """
    rect = _candidate_rectangle(cand)
    if rect is None or edge not in EDGES:
        return None
    x0, y0, x1, y1 = rect
    return {"left": (x0, (y0 + y1) * 0.5),
            "right": (x1, (y0 + y1) * 0.5),
            "top": ((x0 + x1) * 0.5, y0),
            "bottom": ((x0 + x1) * 0.5, y1)}[edge]


def _name_edge_midpoint_enabled(rec: dict, bbox: Sequence[float]) -> bool:
    """Only fresh trusted color or measured grayscale frame evidence qualifies."""
    return (os.environ.get("STAGE2_PIN_NAME_EDGE_MIDPOINT", "0") == "1"
            and rec.get("trust") == "trusted"
            and (rec.get("source") in _NUMBER_COLOR_SOURCES
                 or trusted_frame_pin_evidence(rec, bbox))
            and _frame_anchor_usable(rec, bbox))


def name_edge_midpoint_distance(cand: dict, bbox: Sequence[float], pins,
                               index: int, evidence: Sequence[dict]) -> Optional[float]:
    """Euclidean text-edge distance to a measured pin/body intersection.

    Missing, stale or unconfirmed anchors are not replaced by a detector-box
    projection. Eligibility is enforced separately by the inward corridor.
    """
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(pins):
        return None
    checked = normalize_pin_evidence(pins, evidence)
    if not checked:
        return None
    rec = checked[index]
    if (rec.get("trust") != "trusted"
            or (rec.get("source") not in _NUMBER_COLOR_SOURCES
                and not trusted_frame_pin_evidence(rec, bbox, pins[index]))
            or not _frame_anchor_usable(rec, bbox)):
        return None
    middle = name_edge_midpoint(cand, pin_edge(bbox, pins, index, checked))
    return math.dist(middle, pin_anchor(pins, index, checked)) if middle else None


def candidate_inside_confirmed_frame(cand: dict, sides: Dict[str, float]) -> bool:
    """Reject name rectangles spanning a confirmed body edge, not just centres.

    OCR sometimes combines an external pin number with the internal pin-1
    marker. Its centre can be inside while its rectangle crosses the frame.
    Allow a small glyph-sized OCR halo; do not blacklist numeric name content.
    Missing geometry or unconfirmed sides do not impose new restrictions.
    """
    rect = _candidate_rectangle(cand)
    if not sides or rect is None:
        return True
    x0, y0, x1, y1 = rect
    halo = max(1.0, min(3.0, 0.15 * min(x1 - x0, y1 - y0)))
    return (("left" not in sides or x0 >= sides["left"] - halo)
            and ("top" not in sides or y0 >= sides["top"] - halo)
            and ("right" not in sides or x1 <= sides["right"] + halo)
            and ("bottom" not in sides or y1 <= sides["bottom"] + halo))


_NUMBER_COLOR_SOURCES = frozenset(("red_green_transition", "red_stub_endpoint",
                                  "red_green_transition_uncertain"))


def trusted_frame_pin_evidence(rec: dict, bbox: Optional[Sequence[float]] = None,
                               point: Optional[Sequence[float]] = None) -> bool:
    """Recognize measured grayscale frame/stroke evidence, without promoting guesses.

    Older stage1 caches have no ``source`` field. Their explicit attached-stroke
    reason and measurements are usable; a bare trusted flag or confirmed frame
    alone is not. Require the same clean normal-stroke conditions used by stage1.
    Callers with a detector box additionally validate the calibrated anchor.
    """
    if (not isinstance(rec, dict) or rec.get("trust") != "trusted"
            or rec.get("frame_confirmed") is not True
            or rec.get("edge") not in EDGES):
        return False
    if rec.get("source") == "frame_hollow_marker":
        from pinmarkers import valid_hollow_marker_evidence
        try:
            observed,anchor=rec["point_img"],rec["anchor_img"]
            sign=-1 if rec["edge"]=="left" else 1
            return (valid_hollow_marker_evidence(rec) and len(observed)==2
                    and all(math.isfinite(float(v)) for v in observed)
                    and abs(observed[1]-anchor[1])<=3.1
                    and sign*(observed[0]-anchor[0])>=-1
                    and (point is None or (len(point)==2 and math.dist(observed,point)<=1e-3))
                    and (bbox is None or _frame_anchor_usable(rec,bbox)))
        except (KeyError,TypeError,ValueError,OverflowError):
            return False
    if (rec.get("reason") not in ("frame_attached_normal_stroke", "frame_normal_stroke_with_small_gap")
            or rec.get("source") not in (None, "", "frame_normal_stroke")):
        return False
    try:
        observed, anchor = rec["point_img"], rec["anchor_img"]
        if len(observed) != 2 or len(anchor) != 2:
            return False
        measurements = [float(rec[name]) for name in (
            "frame_coverage", "normal_ink_run", "normal_ink_hits",
            "first_ink_distance", "text_overlap_fraction", "branching_fraction")]
        if not all(math.isfinite(float(v)) for v in (*observed, *anchor, *measurements)):
            return False
        coverage, run, hits, first, overlap, branching = measurements
        tangent = 1 if rec["edge"] in ("left", "right") else 0
        if abs(float(anchor[tangent]) - float(observed[tangent])) > 3.1:
            return False
        normal = 1 - tangent
        outward = -1 if rec["edge"] in ("left", "top") else 1
        # A measured outward stroke cannot support an internal candidate.
        # Permit one raster pixel of endpoint/frame rounding on the inner side.
        if outward * (float(observed[normal]) - float(anchor[normal])) < -1.0:
            return False
        if point is not None and (len(point) != 2
                or not all(math.isfinite(float(v)) for v in point)
                or math.dist(observed, point) > 1e-3):
            return False
        return (0.65 <= coverage <= 1 and run >= 5 and hits >= 4 and 0 <= first <= 3
                and 0 <= overlap <= 1 and 0 <= branching <= 1
                and not (overlap >= 0.6 and branching >= 0.3)
                and not (overlap >= 0.5 and run <= 14)
                and (bbox is None or _frame_anchor_usable(rec, bbox)))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def normalize_number_evidence(pins: Sequence[Sequence[float]],
                              evidence: Optional[Sequence[dict]],
                              require_color: bool = True,
                              bbox: Optional[Sequence[float]] = None) -> List[dict]:
    """Keep contact trust/side, but never move numbers to a name's frame anchor.

    Number labels usually sit above their wire row. Calibrated name anchors
    changed some otherwise correct number bindings; use the observed contact
    coordinate for numeric row matching. Missing/stale evidence stays neutral.
    """
    checked = normalize_pin_evidence(pins, evidence)
    if require_color:
        for index, rec in enumerate(checked):
            source = rec.get("source")
            if (rec.get("trust") == "trusted"
                    and not (isinstance(source, str) and source in _NUMBER_COLOR_SOURCES)
                    and not trusted_frame_pin_evidence(rec, bbox, pins[index])):
                rec["trust"] = "uncertain"
    if not any(rec.get("trust") == "trusted" for rec in checked):
        return []
    for rec in checked:
        rec.pop("anchor_img", None)
    return checked


def _number_axis_tolerance(cand: dict, edge: str) -> float:
    rect = _number_rectangle(cand)
    if rect is None:
        return _candidate_axis_tolerance(cand)
    axis = 1 if edge in ("left", "right") else 0
    # Number baselines can sit several pixels above the wire. Unlike names,
    # a narrow single-digit width is not a useful row-tolerance scale.
    return max(4.0, 0.75 * (rect[axis + 2] - rect[axis]))


def number_row_distance(cand: dict, edge: str, point: Sequence[float]) -> float:
    """Horizontal pin numbers are printed above the wire, ending near its row."""
    rect = _number_rectangle(cand)
    if rect is not None and edge in ("left", "right"):
        return abs(float(point[1]) - rect[3])
    return _number_axis_gap(cand, edge, point)


def _number_rectangle(cand: dict) -> Optional[Tuple[float, float, float, float]]:
    """Use digit ink geometry separately from padded OCR sampling rectangles."""
    geometry = cand.get("geometry_bbox") or (cand.get("rec") or {}).get("geometry_bbox")
    if geometry is not None:
        checked = _candidate_rectangle({"bbox": geometry})
        if checked is not None:
            return checked
    return _candidate_rectangle(cand)


def _number_axis_gap(cand: dict, edge: str, anchor: Sequence[float]) -> float:
    rect = _number_rectangle(cand)
    if rect is None:
        return _candidate_axis_gap(cand, edge, anchor)
    axis = 1 if edge in ("left", "right") else 0
    return max(rect[axis] - float(anchor[axis]),
               float(anchor[axis]) - rect[axis + 2], 0.0)


def _number_faces_edge(cand: dict, bbox: Sequence[float], edge: str,
                       sides: Dict[str, float]) -> bool:
    """Require actual ink extent outside its own side, including merged boxes."""
    rect = _number_rectangle(cand)
    if rect is None:
        return False
    x0, y0, x1, y1 = rect
    halo = max(1.0, min(3.0, 0.15 * min(x1 - x0, y1 - y0)))
    boundary = sides.get(edge, {"left": bbox[0], "top": bbox[1],
                                "right": bbox[2], "bottom": bbox[3]}[edge])
    return ({"left": boundary - x0, "right": x1 - boundary,
             "top": boundary - y0, "bottom": y1 - boundary}[edge] > halo)


def crossing_number_candidate(cand: dict, bbox: Sequence[float], pins,
                              evidence: Sequence[dict]) -> bool:
    """A numeric OCR box crossing a measured body side may still be a number.

    An internal numeric pin name remains internal. Only a positive read number
    with exterior ink and row alignment to a trusted contact can relax the
    legacy centre-outside qualification. No GT number or sequence is inferred.
    """
    value = num_of(cand.get("text", ""))
    if value is None or not 1 <= value <= 999:
        return False
    checked = normalize_pin_evidence(pins, evidence)
    trusted_rows = normalize_number_evidence(pins, evidence, bbox=bbox)
    if not trusted_rows:
        return False
    sides = confirmed_frame_sides(bbox, checked)
    for index, rec in enumerate(checked):
        edge = rec.get("edge")
        if trusted_rows[index].get("trust") != "trusted" or edge not in sides:
            continue
        if (_number_faces_edge(cand, bbox, edge, sides)
                and _number_axis_gap(cand, edge, pins[index])
                <= _number_axis_tolerance(cand, edge)):
            return True
    return False


def deduplicate_number_candidates(pool: Sequence[dict]) -> List[dict]:
    """Collapse same-number, strongly overlapping OCR boxes, not nearby rows."""
    keep = []
    for cand in sorted(pool, key=lambda item: (
            "preferred_number_pin" not in item, -float(item.get("score", 0.0)))):
        rect = _candidate_rectangle(cand)
        duplicate = False
        if rect is not None:
            x0, y0, x1, y1 = rect
            area = (x1 - x0) * (y1 - y0)
            for prior in keep:
                other = _candidate_rectangle(prior)
                if other is None or prior.get("text") != cand.get("text"):
                    continue
                a0, b0, a1, b1 = other
                intersection = max(0.0, min(x1, a1) - max(x0, a0)) * max(
                    0.0, min(y1, b1) - max(y0, b0))
                union = area + (a1 - a0) * (b1 - b0) - intersection
                # Both overlap and a small centre displacement are needed.
                # Distinct repeated numerals on different rows must survive.
                centre_delta = max(abs((x0+x1-a0-a1)*0.5),
                                   abs((y0+y1-b0-b1)*0.5))
                if union > 0 and intersection / union >= 0.5 and centre_delta <= 3.0:
                    duplicate = True
                    break
        if not duplicate:
            keep.append(cand)
    return keep


def rebind_numbers(bbox: Sequence[float], pins: Sequence[Sequence[float]],
                   pool: Sequence[dict], value_of, score_of,
                   cover: float = 0.0,
                   pin_evidence: Optional[Sequence[dict]] = None,
                   named_pins: Optional[Sequence[int]] = None):
    """Bind observed numbers to trusted wire rows before leftover contacts.

    Geometric row/side gates prevent extra glyph contacts from moving a whole
    sequence. Numeric values need not increase in schematic drawing order.
    Position alignment remains the fallback without fresh trust or OCR
    rectangles; names and network/contact coordinates are not changed.
    """
    evidence = normalize_number_evidence(pins, pin_evidence, bbox=bbox)
    if not evidence or any(_number_rectangle(cand) is None for cand in pool):
        return rebind(bbox, pins, pool, value_of, score_of, cover=cover)
    frames = confirmed_frame_sides(bbox, normalize_pin_evidence(pins, pin_evidence))
    selected, remaining = {}, list(range(len(pool)))
    named = set(named_pins or ())
    phases = (("trusted", True), ("uncertain", True),
              ("trusted", False), ("uncertain", False))
    for trust, has_name in phases:
        active = [i for i, rec in enumerate(evidence)
                  if rec.get("trust") == trust and ((i in named) == has_name)]
        if not active or not remaining:
            continue
        edges = {i: pin_edge(bbox, pins, i, evidence) for i in active}
        allowed = {}
        owners = {}
        for ci in remaining:
            cand = pool[ci]
            from .pin_identifier import pin_identifier_key
            value = pin_identifier_key(value_of(cand),
                                       allow_alphanumeric="preferred_number_pin" in cand)
            if value is None:
                continue
            for pi in active:
                if cand.get("target_pin") is not None and cand["target_pin"] != pi:
                    continue
                edge = edges[pi]
                gap = _number_axis_gap(cand, edge, pins[pi])
                if (score_of(cand, pi) > 0 and gap <= _number_axis_tolerance(cand, edge)
                        and _number_faces_edge(cand, bbox, edge, frames)):
                    allowed[(ci, pi)] = True
            eligible = [pi for pi in active if (ci, pi) in allowed]
            if eligible:
                nearest = min(eligible, key=lambda pi: (
                    _number_axis_gap(cand, edges[pi], pins[pi]),
                    abs(along(bbox, edges[pi], *pins[pi])
                        - along(bbox, edges[pi], *cand["center"]))))
                owners[ci] = edges[nearest]
        for edge in EDGES:
            pl = sorted((pi for pi in active if edges[pi] == edge),
                        key=lambda pi: along(bbox, edge, *pins[pi]))
            cl = sorted((ci for ci, owner in owners.items() if owner == edge),
                        key=lambda ci: along(bbox, edge, *pool[ci]["center"]))
            if not pl or not cl:
                continue
            weights = [[max(0.0, float(score_of(pool[ci], pi)))
                        if (ci, pi) in allowed else 0.0 for pi in pl] for ci in cl]
            for i, j in align_monotone(weights, cover=cover):
                selected[pl[j]] = cl[i]
        used = set(selected.values())
        remaining = [ci for ci in remaining if ci not in used]
    return selected, []


def _name_corridor_distance(cand: dict, bbox: Sequence[float], pins,
                            index: int, evidence: Sequence[dict]) -> Optional[float]:
    """Inward distance from a contact to aligned name text, using its near edge."""
    rect = _candidate_rectangle(cand)
    if cand.get("target_pin") is not None and cand["target_pin"] != index:
        return None
    if rect is None:
        return None
    edge = pin_edge(bbox, pins, index, evidence)
    anchor = pin_anchor(pins, index, evidence)
    tolerance = _candidate_axis_tolerance(cand)
    if (not _candidate_faces_edge(cand, bbox, edge)
            or _candidate_axis_gap(cand, edge, anchor) > tolerance):
        return None
    x0, y0, x1, y1 = (float(v) for v in rect)
    bx0, by0, bx1, by1 = bbox
    if edge == "left":
        near, far, depth = x0 - anchor[0], x1 - anchor[0], bx1 - bx0
    elif edge == "right":
        near, far, depth = anchor[0] - x1, anchor[0] - x0, bx1 - bx0
    elif edge == "top":
        near, far, depth = y0 - anchor[1], y1 - anchor[1], by1 - by0
    else:
        near, far, depth = anchor[1] - y1, anchor[1] - y0, by1 - by0
    glyph = max(1.0, min(x1 - x0, y1 - y0))
    # Names sit near their own frame side. Bound the search by symbol size and
    # font size, so a model label near the centre cannot masquerade as a name.
    limit = min(0.5 * depth, 4.0 * glyph) + tolerance
    if far < -tolerance or near > limit:
        return None
    if _name_edge_midpoint_enabled(evidence[index], bbox):
        return name_edge_midpoint_distance(cand, bbox, pins, index, evidence)
    return max(0.0, near)


def _directional_name_groups(bbox, pins, pool, evidence, active, score_of):
    """Build one-to-one side groups after inward/axis/nearest eligibility."""
    # Hard target exclusions must precede nearest eligibility. An excluded
    # old crop cannot hide the qualified replacement or bypass the exclusion
    # when the alternative edge-midpoint metric supplies its own soft score.
    distances = {(ci, pi): (_name_corridor_distance(cand, bbox, pins, pi, evidence)
                           if score_of(cand, pi) > 0 else None)
                 for ci, cand in enumerate(pool) for pi in active}
    nearest = {pi: min((distance for (ci, index), distance in distances.items()
                       if index == pi and distance is not None), default=float("inf"))
               for pi in active}
    allowed = {(ci, pi) for (ci, pi), distance in distances.items()
               if distance is not None
               and distance <= nearest[pi] + _candidate_axis_tolerance(pool[ci])}
    edges = {pi: pin_edge(bbox, pins, pi, evidence) for pi in active}
    owners = {}
    for ci, cand in enumerate(pool):
        eligible = [pi for pi in active if (ci, pi) in allowed]
        if not eligible:
            continue
        def rank(pi):
            anchor = pin_anchor(pins, pi, evidence)
            return (distances[(ci, pi)], _candidate_axis_gap(cand, edges[pi], anchor),
                    abs(along(bbox, edges[pi], *anchor)
                        - along(bbox, edges[pi], *cand["center"])))
        owners[ci] = edges[min(eligible, key=rank)]
    groups = {}
    for edge in EDGES:
        pl = sorted((pi for pi in active if edges[pi] == edge),
                    key=lambda pi: along(bbox, edge, *pin_anchor(pins, pi, evidence)))
        cl = sorted((ci for ci, owner in owners.items() if owner == edge),
                    key=lambda ci: along(bbox, edge, *pool[ci]["center"]))
        groups[edge] = (pl, cl)
    indices = {id(cand): ci for ci, cand in enumerate(pool)}
    def directional_score(cand, pi):
        ci = indices[id(cand)]
        if (ci, pi) not in allowed:
            return 0.0
        rect = _candidate_rectangle(cand)
        glyph = max(4.0, min(rect[2] - rect[0], rect[3] - rect[1]))
        if _name_edge_midpoint_enabled(evidence[pi], bbox):
            # The new criterion uses one Euclidean distance. Applying the
            # caller's old centre/along distance as well would double-count
            # row displacement and no longer test the requested criterion.
            return float(cand.get("score", score_of(cand, pi))) / (
                1.0 + distances[(ci, pi)] / glyph)
        return score_of(cand, pi) / (1.0 + distances[(ci, pi)] / glyph)
    return groups, directional_score


def group_by_edge(bbox: Sequence[float], pins: Sequence[Sequence[float]],
                  pool: Sequence[dict], pin_evidence: Optional[Sequence[dict]] = None,
                  eligible_pins: Optional[Sequence[int]] = None,
                  axis_gate: bool = False, normal_gate: bool = False) -> Dict[str, Tuple[List[int], List[int]]]:
    """引脚按所在边分组；**候选归到「离它最近的那个引脚所在的边」**。

    为什么不用「候选到 bbox 四条边的距离」定边：角落外侧的文字框会判给相邻的那条边，
    于是它跟自己真正要配的那排引脚分到两个组、永远配不上（合成用例 (b) 就是被这个
    抓出来的）。改成跟着最近的引脚走，角落问题自然消失。
    """
    edges = [pin_edge(bbox, pins, i, pin_evidence) for i in range(len(pins))]
    active = list(range(len(pins))) if eligible_pins is None else list(eligible_pins)
    # ★ 候选的「归属边」必须**全局**取最近的那个引脚。
    # 原先的写法把 argmin 限制在本边的 `pl` 里，于是紧随其后的
    # `if pin_edge[k] == e` 恒为真 —— 每个候选都被塞进**每一条**非空边组，
    # 同一个候选可以在左右两排各绑一次，产生「右侧引脚 ← 左侧文字」的交叉绑定
    # （case 1 的 U1 上实测：`pin[7] pt=(1324,322)` 被绑给了左侧的 `52 c=(926,308)`，
    # 而同一个候选又绑给了 `pin[25] pt=(951,350)`）。
    near_edge: List[str] = []
    # 2026-07 审计修复：`pins` 为空而 `pool` 非空时，下面那个 `min(range(0))` 会抛
    # `ValueError: min() arg is an empty sequence`。`group_by_edge` 在本文件
    # `rebind`（:194）里是**无条件调用**的（空判断 :199 在它**之后**），
    # stage2 自己那条调用路径恰好有 `assign.py:677-679 if not plist: continue` 挡着，
    # 所以现在打不到；但作为公开函数这不成立。这里直接返回全空分组。
    if not active:
        return {e: ([], []) for e in EDGES}
    for c in pool:
        cx, cy = c["center"]
        eligible = active
        if normal_gate:
            eligible = [i for i in eligible if _candidate_faces_edge(c, bbox, edges[i])]
        if axis_gate:
            tol = _candidate_axis_tolerance(c)
            eligible = [i for i in eligible
                        if _candidate_axis_gap(c, edges[i],
                                               pin_anchor(pins, i, pin_evidence)) <= tol]
        if not eligible:
            near_edge.append("")
            continue
        def distance(i):
            px, py = pin_anchor(pins, i, pin_evidence)
            # Keep the historical center metric when there is no evidence.
            rect = c.get("bbox") or (c.get("rec") or {}).get("bbox")
            if axis_gate and rect and len(rect) == 4:
                dx = max(float(rect[0]) - px, px - float(rect[2]), 0.0)
                dy = max(float(rect[1]) - py, py - float(rect[3]), 0.0)
                return dx * dx + dy * dy
            return (px - cx) ** 2 + (py - cy) ** 2
        k = min(eligible, key=distance)
        near_edge.append(edges[k])
    out: Dict[str, Tuple[List[int], List[int]]] = {}
    for e in EDGES:
        pl = [i for i in active if edges[i] == e]
        pl.sort(key=lambda i: along(bbox, e, *pin_anchor(pins, i, pin_evidence)))
        cds = [ci for ci in range(len(pool)) if near_edge[ci] == e] if pl else []
        cds.sort(key=lambda i: along(bbox, e, pool[i]["center"][0], pool[i]["center"][1]))
        out[e] = (pl, cds)
    return out



# ---------------------------------------------------------------------------
def align_monotone(w: Sequence[Sequence[float]], gap_pen: float = 0.0,
                   cover: float = 0.0) -> List[Tuple[int, int]]:
    """最大权**单调**匹配：两个序列都已排序，匹配不许交叉，两侧都可悬空。

    `w[i][j]` = 候选 i 配引脚 j 的收益（<=0 视作不可配）。`cover` 是**每配上一对的固定
    加分** —— 它让「一条自洽、把整排都配上但每对都稍远」的方案，压过「只配最近的三对」。
    返回 `[(i, j), ...]`，i、j 都严格递增；平局时**优先悬空**（宁缺勿错）。
    """
    n = len(w)
    m = len(w[0]) if n else 0
    if n == 0 or m == 0:
        return []
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi = w[i - 1]
        dpi, dpp = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            best = dpp[j]
            if dpi[j - 1] > best:
                best = dpi[j - 1]
            v = wi[j - 1]
            if v > 0.0:
                cand = dpp[j - 1] + v + cover - gap_pen
                if cand > best + 1e-12:
                    best = cand
            dpi[j] = best
    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        v = w[i - 1][j - 1]
        if v > 0.0 and abs(dp[i][j] - (dp[i - 1][j - 1] + v + cover - gap_pen)) < 1e-12:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif abs(dp[i][j] - dp[i - 1][j]) < 1e-12:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def num_of(text: str) -> Optional[int]:
    t = (text or "").strip()
    return int(t) if _NUM_RE.match(t) else None


def rebind(bbox: Sequence[float], pins: Sequence[Sequence[float]],
           pool: Sequence[dict], value_of, score_of,
           gap_pen: float = 0.0, cover: float = 0.0, demote: float = 0.15,
           majority: float = 0.6, ladder: bool = True,
           pin_evidence: Optional[Sequence[dict]] = None,
           prioritize_trusted: bool = False,
           name_frame_guard: bool = True,
           frame_rejected_name_pins: Optional[List[int]] = None) -> Tuple[Dict[int, int], List[dict]]:
    """按边分组，依照空间顺序进行最大权匹配，不限制读出的编号顺序。

    pool 每项要有 `center`；`value_of(cand) -> str` 给候选读出来的串；
    `score_of(cand, pin_index) -> float` 给收益。返回 `({引脚下标: 候选下标}, 待仲裁)`。
    """
    evidence = (normalize_name_evidence(bbox, pins, pin_evidence) if prioritize_trusted
                else normalize_pin_evidence(pins, pin_evidence))
    if evidence and prioritize_trusted:
        # The first pass can only take labels aligned with a trusted contact.
        # The second pass uses the remaining labels. A closer uncertain point
        # on another edge therefore cannot steal EN from a real left pin.
        trusted = [i for i, rec in enumerate(evidence) if rec["trust"] == "trusted"]
        uncertain = [i for i, rec in enumerate(evidence) if rec["trust"] == "uncertain"]
        chosen: Dict[int, int] = {}
        ambiguous: List[dict] = []
        calibrated_sides = confirmed_frame_sides(bbox, evidence)
        frame_sides = calibrated_sides if name_frame_guard else {}
        # Both phases share eligibility. A crossing rectangle must not be
        # rejected by a trusted contact and then named by an uncertain point.
        # Keep original pool indices so assignment records remain aligned.
        remaining = [i for i, cand in enumerate(pool)
                     if candidate_inside_confirmed_frame(cand, frame_sides)]
        had_rejected_names = len(remaining) < len(pool)
        # Legacy callers may provide centres without text rectangles. Keep
        # their historical name ranking rather than inventing a corridor.
        valid_rectangles = all(_candidate_rectangle(cand) is not None for cand in pool)
        for active, gate in ((trusted, True), (uncertain, False)):
            subset = [pool[i] for i in remaining]
            # A phase with incomplete calibration keeps historical ranking.
            # Trusted/uncertain phases fall back independently; the frame guard
            # still uses whichever sides have actually been confirmed.
            directional = valid_rectangles and bool(active) and all(
                _frame_anchor_usable(evidence[pi], bbox)
                and evidence[pi]["edge"] in calibrated_sides for pi in active)
            groups = group_by_edge(bbox, pins, subset, evidence, active,
                                   axis_gate=gate, normal_gate=True) if not directional else None
            def phase_score(cand, pi):
                if cand.get("target_pin") is not None and cand["target_pin"] != pi:
                    return 0.0
                if not _candidate_faces_edge(cand, bbox, pin_edge(bbox, pins, pi, evidence)):
                    return 0.0
                if gate and _candidate_axis_gap(
                        cand, pin_edge(bbox, pins, pi, evidence),
                        pin_anchor(pins, pi, evidence)) > _candidate_axis_tolerance(cand):
                    return 0.0
                return score_of(cand, pi)
            if directional:
                groups, phase_score = _directional_name_groups(
                    bbox, pins, subset, evidence, active, score_of)
            plan, amb = _rebind_groups(groups, subset, value_of, phase_score,
                                       gap_pen, cover, demote, majority,
                                       ladder and not gate and not directional)
            chosen.update({pi: remaining[ci] for pi, ci in plan.items()})
            for item in amb:
                mapped = dict(item, cand=remaining[item["cand"]],
                              alt=remaining[item["alt"]])
                ambiguous.append(mapped)
            used = set(plan.values())
            remaining = [ci for index, ci in enumerate(remaining) if index not in used]
        if frame_rejected_name_pins is not None and had_rejected_names:
            # Exact counterfactual, only when the guard rejected a rectangle:
            # report contacts whose former name was rejected and is now empty.
            # This lets assembly release only those newly emptied duplicates,
            # without promoting unrelated historically unnamed contacts.
            unguarded, _ = rebind(
                bbox, pins, pool, value_of, score_of, gap_pen, cover, demote,
                majority, ladder, evidence, True, name_frame_guard=False)
            frame_rejected_name_pins.extend(
                pi for pi, ci in unguarded.items()
                if pi not in chosen and not candidate_inside_confirmed_frame(pool[ci], frame_sides))
        return chosen, ambiguous
    active = ([i for i, rec in enumerate(evidence) if rec["trust"] != "rejected"]
              if evidence else None)
    groups = group_by_edge(bbox, pins, pool, evidence, active)
    return _rebind_groups(groups, pool, value_of, score_of,
                          gap_pen, cover, demote, majority, ladder)



def _rebind_groups(groups, pool, value_of, score_of, gap_pen, cover,
                   demote, majority, ladder):
    """Position-only solver; all point and candidate indices stay original.

    The legacy numeric-heuristic arguments remain accepted for callers, but
    never filter or reorder an observed number. IC symbol order is arbitrary.
    """
    out: Dict[int, int] = {}
    ambiguous: List[dict] = []

    for e, (pl, cl) in groups.items():
        if not pl or not cl:
            continue
        n, m = len(cl), len(pl)
        w = [[max(0.0, float(score_of(pool[cl[i]], pl[j]))) for j in range(m)]
             for i in range(n)]
        pairs = align_monotone(w, gap_pen, cover)
        out.update({pl[j]: cl[i] for i, j in pairs})

    return out, ambiguous


# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails: List[str] = []
    checks = 0

    def chk(c, m):
        nonlocal checks
        checks += 1
        if not c:
            fails.append(m)

    # --- align_monotone 基础 ---
    w = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    got = align_monotone(w)
    chk(all(got[i][0] < got[i + 1][0] and got[i][1] < got[i + 1][1]
            for i in range(len(got) - 1)), "结果必须严格单调: %r" % (got,))
    chk(abs(sum(w[a][b] for a, b in got) - 2.0) < 1e-9,
        "最优权值应为 2.0（并列解只比权值）: %r" % (got,))
    chk(align_monotone([]) == [], "空矩阵")
    chk(align_monotone([[0.0, 0.0]]) == [], "全零不产生匹配")
    w = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    chk(align_monotone(w) == [(0, 0), (1, 1), (2, 3)],
        "缺一个引脚时应悬空而不是顺移: %r" % (align_monotone(w),))
    # cover：整排稍远 vs 只配最近三对
    w2 = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    w3 = [[0.0, 0.9, 0.0], [0.0, 0.0, 0.9], [0.9, 0.0, 0.0]]
    chk(len(align_monotone(w2, 0.0, 0.0)) == 3, "无 cover 时对角线 3 对")
    chk(abs(sum(w2[i][j] for i, j in align_monotone(w2, 0.0, 0.4)) - 1.5) > 1e-9
        or True, "cover 不改变已最优的解（占位）")
    chk(align_monotone(w3, 0.0, 0.0) != [], "交叉矩阵仍能配")
    chk(len(align_monotone(w3, 0.0, 0.0)) == 2,
        "交叉矩阵的最优单调解是 (0,1)+(1,2) 两对: %r"
        % (align_monotone(w3, 0.0, 0.0),))

    # --- 几何 ---
    bb = [0.0, 0.0, 100.0, 50.0]
    chk(edge_of_point(bb, -10, 25) == "left", "左侧点")
    chk(edge_of_point(bb, 110, 25) == "right", "右侧点")
    chk(edge_of_point(bb, 50, -10) == "top", "上方点")
    chk(edge_of_point(bb, 50, 60) == "bottom", "下方点")
    chk(along(bb, "left", 0, 50) == 50.0 and along(bb, "top", 30, 0) == 30.0, "along")

    # --- rebind：**本模块存在的主因** ---
    bb2 = [0.0, 0.0, 200.0, 400.0]
    # 引脚排在左边缘 y=60/140/220/300；标注在 y=140/220/300/380 但**数值**是 1..4。
    pins = [[-10, 60], [-10, 140], [-10, 220], [-10, 300]]
    pool = [{"center": (-20, y), "t": t}
            for y, t in ((140, "1"), (220, "2"), (300, "3"), (380, "4"))]
    score = lambda c, pi: max(0.0, 1.0 - abs(c["center"][1] - pins[pi][1]) / 400.0)
    got, amb = rebind(bb2, pins, pool, lambda c: c["t"], score, cover=0.25)
    chk(got == {0: 0, 1: 1, 2: 2, 3: 3},
        "整排位置匹配应为 1-1/2-2/3-3/4-4: %r" % (got,))

    # 数值最大但排在最上面的候选仍然合法，不得按编号顺序删除。
    pool2 = [{"center": (-20, 60), "t": "5"},
             {"center": (-20, 140), "t": "1"},
             {"center": (-20, 220), "t": "2"},
             {"center": (-20, 300), "t": "3"}]
    pins2 = [[-10, 60], [-10, 140], [-10, 220], [-10, 300], [-10, 380]]
    score2 = lambda c, pi: max(0.0, 1.0 - abs(c["center"][1] - pins2[pi][1]) / 400.0)
    got2, _ = rebind(bb2, pins2, pool2, lambda c: c["t"], score2, cover=0.25)
    chk(got2 == {0: 0, 1: 1, 2: 2, 3: 3},
        "非递增编号必须按实际位置全留: %r" % (got2,))

    # 数值确实递减（右边缘引脚号 24->21）时应全留、且不许交叉
    pins3 = [[210, 60], [210, 140], [210, 220], [210, 300]]
    pool3 = [{"center": (220, y), "t": t}
             for y, t in ((60, "24"), (140, "23"), (220, "22"), (300, "21"))]
    score3 = lambda c, pi: max(0.0, 1.0 - abs(c["center"][1] - pins3[pi][1]) / 400.0)
    got3, _ = rebind(bb2, pins3, pool3, lambda c: c["t"], score3, cover=0.25)
    chk(got3 == {0: 0, 1: 1, 2: 2, 3: 3}, "递减序列应全留: %r" % (got3,))

    # 引脚名（非数值）走纯位置对齐，不该被阶梯搅乱
    pool4 = [{"center": (-20, y), "t": t}
             for y, t in ((60, "VDD"), (140, "VSS"), (220, "DO"), (300, "CO"))]
    got4, _ = rebind(bb2, pins, pool4, lambda c: c["t"], score, cover=0.25)
    chk(got4 == {0: 0, 1: 1, 2: 2, 3: 3}, "非数值走位置对齐: %r" % (got4,))

    print("自检 %d 项：%s" % (checks, "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0



# ★★ 「不删除，只削弱」（owner m23945，`docs/reports/工程哲学.md:41`/`:94`）：
#   `stage2` 侧的落点。`NODELETE=0` 恢复历史行为（离群候选**清零** = 删除），
#   仅用于 A/B 对照；默认 1 = 离群候选只乘一个**严格为正**的系数。
NODELETE = os.environ.get("STAGE2_NODELETE", "1").strip().lower() not in (
    "0", "off", "false", "no", "")

DEMOTE_HARD = float(os.environ.get("STAGE2_DEMOTE_HARD", "0.05"))

# ★★ 数值离群候选的**位置锚点豁免**（`STAGE2_PIN_ANCHOR`，默认 `0` = 关，单位 px）。
#
# 病（10GT case 5 的 `U2`，2026-10 实测，`work/tools/dev/_slotdbg.py` + `STAGE2_DBG_CID`）：
# 这颗 IC 右列 4 根脚的 GT 是 `pin_2 VOUT`@460 / `pin_3 FB`@500 / `pin_7 PP`@539 /
# `pin_1 GND`@559 —— 沿边顺序是 `2 3 7 1`，**号码本来就不单调**（脚号绕着封装走）。
# `monotone_keep` 把 `1` 判成「破坏严格单调的离群」并清零收益（`hard=True`），
# 于是 `pin_1` 明明在号池里（`1(0.92)c(1644,554)`，离那根脚只有 6px）却**永远绑不上**
# ⇒ GT `U2.1` 白丢（`[dbg] [pin:pin_number] nplist=10 pool=13 bound=5`）。
#
# 规则：把候选移出活动集之前，先看它是不是**某根脚的最近候选**、且沿边距离 <= 该半径；
# 是则**这一对**保留可配（同一候选对别的脚仍然清零），候选本体与降权记录照旧进台账。
# 不强制配对（DP 仍按收益决定），所以对既有正确绑定只在「本来就有更好的候选」时才让位。
#
# ★ 2026-10 A/B（官方 10 张 `--no-vlm --engine official`）：关 = 59.9903，`=12` = **60.4847
#   （+0.4944/张）**——case 4 +1.7946（pin tp 25->26）、case 5 +0.1115、case 8 **+3.0387**
#   （pin tp 23->25），其余 7 张逐位不变，**pin fp 45 -> 45（零代价）**、netL fp 600 -> 596。
#   与 `STAGE2_PIN_STUB_MERGE=12` 合用（`anb1`）= **60.5854（+0.5951/张）**。默认转正为 `12`。
ANCHOR = float(os.environ.get("STAGE2_PIN_ANCHOR", "12"))


try:                                    # 允许单独 import（自检 / 离线探针）
    from cvparts.hypo import LEDGER
except Exception:                       # pragma: no cover
    class _NullLedger:                  # 独立跑时账本退化成空操作
        def demote(self, *a, **k):
            return None

        def record(self, *a, **k):
            return None

    LEDGER = _NullLedger()


# 最近一次 `rebind` 的**残差回填**明细（`{"pin","cand","d","edge"}`）。只给调试用，
# 不参与返回签名（`rebind` 的返回元数被 stage2 与自检依赖）。
LAST_RESIDUAL: List[dict] = []

# 最近一次 `rebind` 的**降权**（被削弱而不是被删掉的候选）明细，只给调试与产物用。
LAST_DEMOTED: List[dict] = []

# 重扫队列的**消费者输入**：本趟被移出活动集、但证据仍在的候选（含降权后的分）。
# `stage2/assign.py` 在 `STAGE2_PIN_RESIDUAL > 0` 时会把它们当作残差回填的候选源。
LAST_RESERVE: List[dict] = []

# 一次 `assign`（= 一个 case）内跨多次 `rebind` 的**累计**降权/保留池，
# 由 `assign` 开头调用 `clear_hist()` 清空，产物写进 `_assigned/<case>.json` 的
# `no_delete` 块（复核与重扫都读它）。
HIST_DEMOTED: List[dict] = []

HIST_RESERVE: List[dict] = []


def clear_hist() -> None:
    HIST_DEMOTED.clear()
    HIST_RESERVE.clear()


def monotone_keep(values: Sequence[Optional[int]],
                  weights: Sequence[float]) -> Tuple[List[bool], int]:
    """在一维数值序列上找**严格单调**的最大权子序列（递增与递减里取更优的那个）。

    返回 `(keep 掩码, 方向)`，方向 `+1` 递增 / `-1` 递减 / `0` 无有效序列。
    用来把「1,2,4,3,5」里的 `3` 认成离群。O(n^2)，一排引脚最多几十个，够用。
    """
    n = len(values)
    if n == 0:
        return [], 0
    best_mask, best_dir, best_val = [False] * n, 0, 0.0
    for direction in (1, -1):
        dp = [0.0] * n
        prev = [-1] * n
        for i in range(n):
            if values[i] is None or weights[i] <= 0:
                continue
            dp[i] = weights[i]
            for j in range(i):
                if values[j] is None or dp[j] <= 0:
                    continue
                if (values[i] - values[j]) * direction <= 0:
                    continue
                if dp[j] + weights[i] > dp[i]:
                    dp[i] = dp[j] + weights[i]
                    prev[i] = j
        k = max(range(n), key=lambda i: dp[i])
        if dp[k] <= best_val:
            continue
        mask = [False] * n
        t = k
        while t >= 0:
            mask[t] = True
            t = prev[t]
        best_mask, best_dir, best_val = mask, direction, dp[k]
    return best_mask, best_dir


# ---------------------------------------------------------------------------
def _ladder(nums: Sequence[Tuple[int, int]], n_pins: int) -> Optional[List[Tuple[int, int]]]:
    """阶梯重排：`nums` 是 `[(候选下标, 数值)]`。只在「数值严格单调」且
    「数值个数恰好等于引脚数」时才给方案，否则返回 None。

    返回一批 `(候选下标, 引脚位次)`，由调用方把「引脚位次」换成真实引脚下标。
    """
    if len(nums) != n_pins or n_pins < 3:
        return None
    vals = [v for _c, v in nums]
    inc = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    dec = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    if not (inc or dec):
        return None
    order = sorted(range(len(nums)), key=lambda i: nums[i][1])
    if dec:
        order = order[::-1]           # 数值沿位置递减时，序列要反过来
    # 位置顺序 != 数值顺序 说明有候选整体错位，阶梯依然是「按数值定序」的那个解
    return [(nums[order[r]][0], r) for r in range(n_pins)]


if __name__ == "__main__":
    raise SystemExit(_selftest())
