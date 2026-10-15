"""Complete pin text from measured monochrome strokes or calibrated color contacts.

The frame anchor and same-side physical rows define the search cells. OCR text
does not define a crop, and no numbering order or name dictionary is consulted.
Current chromatic strokes are rechecked before color text crops are opened.
The source image stays unchanged. Recognition copies may exclude measured long
wire pixels and their raster fringes; ambiguous glyph/line contact stays held.
"""

from __future__ import annotations


import math


import cv2

import numpy as np


from .structmatch import trusted_frame_pin_evidence


COLOR_CONTACT_SOURCES = frozenset(("red_green_transition", "red_stub_endpoint",
                                  "red_green_transition_uncertain"))


def measured_text_contact(record, frame, point):
    """Use calibrated color contacts without changing their evidence source."""
    if trusted_frame_pin_evidence(record, frame, point):
        return True
    if not isinstance(record, dict) or record.get("source") not in COLOR_CONTACT_SOURCES:
        return False
    anchor = _finite(record.get("anchor_img"), 2)
    observed = _finite(record.get("point_img"), 2)
    if (record.get("trust") != "trusted" or record.get("frame_confirmed") is not True
            or anchor is None or observed is None or math.dist(observed, point) > .001):
        return False
    measured = record.get("color_evidence") or {}
    numbers = _finite([measured.get("red_hits"), measured.get("red_span"),
                       measured.get("red_coverage"), measured.get("stroke_width")], 4)
    if numbers is None or not (numbers[0] >= 3 and numbers[1] >= 3
                              and .65 <= numbers[2] <= 1 and 1 <= numbers[3] <= 6):
        return False
    edge = record.get("edge")
    if edge not in ("left", "right", "top", "bottom"):
        return False
    normal = 0 if edge in ("left", "right") else 1
    side = {"left": 0, "top": 1, "right": 2, "bottom": 3}[edge]
    sign = -1 if edge in ("left", "top") else 1
    return (abs(anchor[normal]-frame[side]) <= 8
            and abs(anchor[1-normal]-point[1-normal]) <= 3.1
            and sign*(point[normal]-anchor[normal]) > 0)


def _finite(value, size):
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if len(result) == size and all(map(math.isfinite, result)) else None


def mono_observations(doc):
    """Yield calibrated left/right observations; crops recheck current ink."""
    for comp in doc.get("components") or []:
        if not isinstance(comp, dict) or comp.get("type") != "box":
            continue
        cid, frame = comp.get("id"), _finite(comp.get("bbox_img"), 4)
        if not isinstance(cid, str) or frame is None or frame[2] <= frame[0] or frame[3] <= frame[1]:
            continue
        pins = (doc.get("pins") or {}).get(cid) or []
        evidence = (doc.get("pin_evidence") or {}).get(cid) or []
        component_observations=[]
        for index, (raw, ev) in enumerate(zip(pins, evidence)):
            point = _finite(raw, 2)
            if (point is None or not isinstance(ev, dict) or ev.get("edge") not in ("left", "right")
                    or not measured_text_contact(ev, frame, point)):
                continue
            anchor = _finite(ev.get("anchor_img"), 2)
            sign = -1 if ev["edge"] == "left" else 1
            if anchor is None or (point[0] - anchor[0]) * sign <= 0:
                continue
            component_observations.append({"component_id": cid, "pin_index": index, "edge": ev["edge"],
                   "point_img": point, "anchor_img": anchor, "frame": frame,
                   "evidence": dict(ev)})
        # Two old contacts can sit on the same measured raster stroke. Keep
        # their stage1 identities, but use one geometric representative for
        # BOTH independent role crops; no text or numbering decides the winner.
        rows=(doc.get("pin_mono",{}).get(cid) or {}).get("stroke_rows") or []
        representatives={};unmatched=[]
        for ob in component_observations:
            candidates=[(i,r) for i,r in enumerate(rows) if r["edge"]==ob["edge"]
                        and r["tangent_band"][0]-3<=ob["point_img"][1]<=r["tangent_band"][1]+3
                        and abs(ob["anchor_img"][1]-r["tangent_center"])<=3]
            if len(candidates)!=1 or ob["evidence"].get("source")=="frame_hollow_marker":
                unmatched.append(ob)
                continue
            i,row=candidates[0]
            rank=(abs(ob["point_img"][1]-row["tangent_center"]),
                  abs(ob["anchor_img"][1]-row["tangent_center"]),
                  abs(abs(ob["point_img"][0]-ob["anchor_img"][0])-12),ob["pin_index"])
            if i not in representatives or rank<representatives[i][0]:
                representatives[i]=(rank,ob)
        yield from sorted(unmatched+[pair[1] for pair in representatives.values()],key=lambda o:o["pin_index"])


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


def _stroke_top(mask, observation, marker_mask=None):
    """Measure the attached stroke near the frame, away from external labels.

    Only a short normal probe between the current point and frame is tested.
    A dense continuous run through this probe must include the anchor row.
    The result sets the search cell. Later cleanup needs independent measured
    line evidence, and never reconstructs a numeral or a missing stroke.
    """
    point, anchor, edge = (observation[k] for k in ("point_img", "anchor_img", "edge"))
    if observation.get("evidence",{}).get("source")=="frame_hollow_marker":
        from pinmarkers import hollow_marker_bands
        frame=observation["frame"]
        current=mask if marker_mask is None else marker_mask
        ya,yb=int(math.ceil(frame[1]))+8,int(math.floor(frame[3]))-7
        x=int(math.floor(anchor[0]+.5))
        coverage=float(np.mean(current[max(0,ya):min(mask.shape[0],yb),x])) if 0<=x<mask.shape[1] else 0
        threshold=observation['evidence']['marker']['ink_threshold']
        fresh=hollow_marker_bands(current,frame,3 if edge=="left" else 1,
            (anchor[0],coverage>=.65,coverage),ink_threshold=threshold)
        matches=[b for b in fresh if abs(b["tangent_center"]-anchor[1])<=.6
                 and b["marker"]["hole_bbox"]==observation["evidence"]["marker"]["hole_bbox"]]
        if len(matches)!=1:
            return None
        measured=observation["evidence"]["marker"]["outer_line_probe"]
        if matches[0]["marker"]["outer_line_probe"]!=measured:
            return None
        probe=_clip(measured,mask.shape)
        if probe is None or probe!=measured:
            return None
        # The pale contour threshold includes the line's antialiased fringe.
        # Measure its actual dark core independently instead of requiring
        # every pale raster row to be black. A missing, detached or split core
        # still cannot authorize text reading from cached marker evidence.
        occupancy=np.mean(mask[probe[1]:probe[3],probe[0]:probe[2]],1)
        cores=[(probe[1]+a,probe[1]+b) for a,b in _groups(occupancy>=.85,0)]
        if len(cores)!=1:
            return None
        top,bottom=cores[0]
        if not top-1.5<=anchor[1]<=bottom+.5:
            return None
        black_probe=[probe[0],top,probe[2],bottom]
        # Marker geometry was established with pale original outline pixels.
        # This fresh probe still requires the black outer line in this image.
        return dict(probe=black_probe,stroke_top=top,stroke_bottom=bottom,
                    row_occupancy=occupancy.tolist(),marker_outer_line_probe=probe,
                    marker=matches[0]["marker"],
                    number_lane_markers=[b["marker"] for b in fresh
                                         if anchor[1]-80 < b["tangent_center"] <= anchor[1]+.6],
                    method="current_outer_line_of_measured_hollow_marker")
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


def _frame_inner_bounds(mask, light, observation):
    """Measure the persistent border band, including its raster fringe.

    A text stem near the frame is not enough: a candidate border column/row
    must persist across at least 80 percent of this measured frame side. The
    tight adjacent band must contain the current measured anchor coordinate.
    Missing/rough evidence falls back to the conservative old search cell;
    it never authorizes deletion of a short or detached text stroke.
    """
    frame, anchor, edge = (observation[k] for k in ("frame", "anchor_img", "edge"))
    bounds, evidence = list(frame), []
    height, width = mask.shape
    for side in range(4):
        vertical = side in (0, 2)
        coordinate = frame[side]
        if (side == 0 and edge == "left") or (side == 2 and edge == "right"):
            coordinate = anchor[0]
        length = frame[3] - frame[1] if vertical else frame[2] - frame[0]
        if length < 20:
            continue
        normal_size, tangent_size = (width, height) if vertical else (height, width)
        lo, hi = max(0, int(math.floor(coordinate))-8), min(normal_size, int(math.ceil(coordinate))+9)
        begin = max(0, int(math.ceil(frame[1] if vertical else frame[0]))+4)
        end = min(tangent_size, int(math.floor(frame[3] if vertical else frame[2]))-4)
        if end-begin < 12:
            continue
        core = np.mean(mask[begin:end, lo:hi], 0) if vertical else np.mean(mask[lo:hi, begin:end], 1)
        fringe = np.mean(light[begin:end, lo:hi], 0) if vertical else np.mean(light[lo:hi, begin:end], 1)
        bands = [(lo+a, lo+b) for a, b in _groups(fringe >= .80, 0)
                 if np.any(core[a:b] >= .80) and lo+a-2 <= coordinate <= lo+b+1]
        if len(bands) != 1:
            continue
        start, stop = bands[0]
        if stop-start > 12:
            continue
        # One white raster column/row after the actual persistent inner edge.
        inner = stop+1 if side in (0, 1) else start-1
        bounds[side] = inner
        evidence.append({"side": side, "persistent_band": [start, stop],
                         "inner_search_boundary": inner, "tangent_span": [begin, end],
                         "minimum_coverage": .80, "method": "current_long_side_ink_band"})
    return bounds, evidence


def _regions(observation, peers, mask, light_mask=None, marker_mask=None, stroke_mask=None):
    anchor, frame, edge = (observation[k] for k in ("anchor_img", "frame", "edge"))
    pitch = _pitch(observation, peers)
    row = anchor[1]
    above = [p["anchor_img"][1] for p in peers if p["anchor_img"][1] < row - 1]
    below = [p["anchor_img"][1] for p in peers if p["anchor_img"][1] > row + 1]
    measured, border_evidence = _frame_inner_bounds(mask, mask if light_mask is None else light_mask, observation)
    inner = [max(frame[0]+2, measured[0]), max(frame[1]+2, measured[1]),
             min(frame[2]-2, measured[2]), min(frame[3]-2, measured[3])]
    if observation["evidence"].get("source") in COLOR_CONTACT_SOURCES:
        for measured_side in border_evidence:
            side = measured_side["side"]
            inner[side] = measured_side["inner_search_boundary"]
    top = max(inner[1], row - min(24.0, pitch * .48))
    bottom = min(inner[3], row + min(24.0, pitch * .48))
    if above:
        top = max(top, (row + max(above)) / 2 + .5)
    if below:
        bottom = min(bottom, (row + min(below)) / 2 - .5)
    inside = ([max(anchor[0]+2, inner[0]), top, inner[2], bottom] if edge == "left"
              else [inner[0], top, min(anchor[0]-2, inner[2]), bottom])
    # Numbers occupy the space ABOVE their stroke, rather than the name's
    # symmetric row cell. A three-digit word can extend well beyond the pin
    # candidate point; measured pitch sets an independent external width.
    actual_strokes = mask if stroke_mask is None else stroke_mask
    stroke = _stroke_top(actual_strokes, observation, marker_mask)
    if stroke is not None:
        stroke["measured_frame_border"] = border_evidence
    outside = None
    if stroke is not None:
        depth = min(96.0, max(pitch * 2.5, abs(anchor[0] - observation["point_img"][0]) + pitch))
        y0 = row - min(44.0, pitch * .98)
        if above:
            previous_row = max(above)
            lower_limit = previous_row + 4
            previous = [p for p in peers if p["anchor_img"][1] == previous_row]
            # A fixed four-pixel gap cuts the top of dense, otherwise complete
            # labels. Measure the preceding PLAIN stroke in the current image
            # and leave one raster row after its entire black core instead.
            # Directional marker contours keep their existing guarded limit.
            if (len(previous) == 1
                    and _plain_stroke_observation(observation)
                    and _plain_stroke_observation(previous[0])):
                prior_stroke = _stroke_top(actual_strokes, previous[0], marker_mask)
                if prior_stroke is not None:
                    measured_limit = prior_stroke["stroke_bottom"] + 1
                    stroke["previous_current_stroke_limit"] = dict(
                        anchor_img=list(previous[0]["anchor_img"]),
                        stroke_top=prior_stroke["stroke_top"],
                        stroke_bottom=prior_stroke["stroke_bottom"],
                        crop_lower_limit=measured_limit)
            y0 = max(y0, lower_limit)
        # Keep the last numeral row. Raster fringes are audited from actual
        # long wire runs below, instead of deleting a fixed pixel from 1/4/6.
        stroke["upper_fringe_guard_pixels"] = 0
        outside = ([anchor[0] - depth, y0, anchor[0] - 2, stroke["stroke_top"]]
                   if edge == "left" else
                   [anchor[0] + 2, y0, anchor[0] + depth, stroke["stroke_top"]])
        prior_limit = (stroke.get("previous_current_stroke_limit") or {}).get("crop_lower_limit")
        if prior_limit is not None and prior_limit < y0:
            expanded = list(outside)
            expanded[1] = max(row-min(44.0,pitch*.98), prior_limit)
            stroke["number_region_with_previous_stroke_limit"] = _clip(expanded, mask.shape)
        if stroke.get("number_lane_markers"):
            stroke["number_lane_markers"]=[m for m in stroke["number_lane_markers"]
                                          if m["connected_marker_region"][3]>=y0-1
                                          and m["connected_marker_region"][1]<=stroke["stroke_top"]+1]
    return _clip(inside, mask.shape), _clip(outside, mask.shape) if outside else None, pitch, stroke


def _plain_stroke_observation(observation):
    return observation["evidence"].get("source") in (None, "", "frame_normal_stroke", *COLOR_CONTACT_SOURCES)


def _name_word_region(mask, region, anchor, pitch, include_enclosed_baseline=False, word_filter=None, word_gap=None):
    """Group horizontally adjacent glyphs with observed vertical overlap.

    Offset suffixes (subscripts/superscripts) may overlap only part of a
    prefix's height. Nearby independent labels with distinct baselines do not
    become one word solely because their one-dimensional row projections
    overlap. Every selected pixel remains original; detached semantic marks
    are retained or held using the same conservative distance audit.
    """
    if region is None:
        return None, {}
    x0, y0, x1, y1 = region
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask[y0:y1, x0:x1].astype(np.uint8), 8)
    pieces = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area:
            pieces.append({"label": index, "box": [x0+x, y0+y, x0+x+w, y0+y+h], "height": h})
    if not any(p["height"] >= 3 for p in pieces):
        return region, {}
    # A rasterized letter can have a detached foot/serif. Include those
    # observed pieces while linking adjacent glyphs; otherwise the foot is
    # added too late to bridge the real interletter gap and a full word can
    # silently become its first letter. Small-only groups cannot be selected.
    bodies = pieces
    gap = max(3, min(12, int(pitch*.28)))
    if word_gap is not None:
        gap = word_gap
    parents = list(range(len(bodies)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, first in enumerate(bodies):
        for j in range(i+1, len(bodies)):
            a, b = first["box"], bodies[j]["box"]
            normal_gap = max(a[0]-b[2], b[0]-a[2])
            overlap = max(0, min(a[3], b[3])-max(a[1], b[1]))
            # Separate rows over the same x range stay separate. Small marks
            # over a body receive their own full-word integrity audit below.
            widths = (a[2]-a[0], b[2]-b[0])
            horizontal_separation = abs((a[0]+a[2]-b[0]-b[2])/2)
            if (-max(1, .25*min(widths)) <= normal_gap <= gap
                    and horizontal_separation >= .25*sum(widths)
                    and overlap >= .30*min(first["height"], bodies[j]["height"])):
                parents[root(j)] = root(i)
    groups = {}
    for i, piece in enumerate(bodies):
        groups.setdefault(root(i), []).append(piece)
    groups = {key: group for key, group in groups.items()
              if any(p["height"] >= 3 for p in group)}
    def envelope(group):
        return [min(p["box"][0] for p in group), min(p["box"][1] for p in group),
                max(p["box"][2] for p in group), max(p["box"][3] for p in group)]

    # An underscore may be one white row below every adjacent letter. Attach
    # such observed small marks BEFORE linking the word envelopes: otherwise
    # GVDD_AB becomes two words and the frame-nearest suffix wins. Keep the
    # original letter baselines for the later merge, so a detached stroke
    # cannot bridge two independent text rows merely by expanding their boxes.
    def core(group):
        height = max(p["height"] for p in group)
        letters = [p for p in group if p["height"] > max(2, .35 * height)]
        return envelope(letters or group), height

    cores = {key: core(group) for key, group in groups.items()}
    letter_boxes = {key: [p["box"] for p in group
                          if p["height"] > max(2, .35 * cores[key][1])]
                    for key, group in groups.items()}
    owners = {p["label"]: key for key, group in groups.items() for p in group}
    attached_marks = []
    for piece in pieces:
        if piece["label"] in owners:
            continue
        a = piece["box"]
        choices = []
        for key, (b, height) in cores.items():
            if piece["height"] > max(2, .35 * height):
                continue
            normal_gap = max(b[0] - a[2], a[0] - b[2], 0)
            row_gap = max(b[1] - a[3], a[1] - b[3], 0)
            if normal_gap <= gap and row_gap <= max(4, .65 * height):
                choices.append((math.hypot(normal_gap, row_gap), key))
        if choices:
            _, key = min(choices)
            groups[key].append(piece)
            owners[piece["label"]] = key
            attached_marks.append(list(a))

    # A mark attached to one side can close the real interletter gap. Merge
    # only if the ORIGINAL letter cores overlap in y, including offset
    # suffixes; opposite-edge words with a genuine blank gap stay separate.
    changed = True
    while changed:
        changed = False
        keys = list(groups)
        for i, key in enumerate(keys):
            if key not in groups:
                continue
            for other in keys[i + 1:]:
                if other not in groups:
                    continue
                a, b = envelope(groups[key]), envelope(groups[other])
                ca, ha = cores[key]
                cb, hb = cores[other]
                normal_gap = max(a[0] - b[2], b[0] - a[2])
                # Compare the actual facing letters, not the union's taller
                # envelope. A low suffix plus a high prefix can span both
                # baselines while the nearest letters still belong to
                # independent words.
                if ca[0] + ca[2] <= cb[0] + cb[2]:
                    fa = max(letter_boxes[key], key=lambda box: box[2])
                    fb = min(letter_boxes[other], key=lambda box: box[0])
                else:
                    fa = min(letter_boxes[key], key=lambda box: box[0])
                    fb = max(letter_boxes[other], key=lambda box: box[2])
                overlap = max(0, min(fa[3], fb[3]) - max(fa[1], fb[1]))
                widths = (fa[2] - fa[0], fb[2] - fb[0])
                separation = abs((fa[0] + fa[2] - fb[0] - fb[2]) / 2)
                if (-max(1, .25 * min(widths)) <= normal_gap <= gap
                        and separation >= .25 * sum(widths)
                        and overlap >= .30 * min(fa[3] - fa[1], fb[3] - fb[1])):
                    groups[key].extend(groups.pop(other))
                    letter_boxes[key].extend(letter_boxes.pop(other))
                    cores[key] = ([min(ca[0], cb[0]), min(ca[1], cb[1]),
                                   max(ca[2], cb[2]), max(ca[3], cb[3])], max(ha, hb))
                    cores.pop(other)
                    for p in groups[key]:
                        owners[p["label"]] = key
                    changed = True

    def anchor_distance(group):
        b = envelope(group)
        normal = min(abs(b[0]-anchor[0]), abs(b[2]-anchor[0]))
        row = abs((b[1]+b[3])/2-anchor[1])
        return math.hypot(normal, row), row
    eligible = [key for key in groups if word_filter is None or word_filter(envelope(groups[key]))]
    if not eligible:
        return None, {}
    selected_key = min(eligible, key=lambda key: anchor_distance(groups[key]))
    selected = list(groups[selected_key])
    box = envelope(selected)
    body_height = max(p["height"] for p in selected)
    selected_labels = {p["label"] for p in selected}
    enclosed = []
    if include_enclosed_baseline:
        # An inversion bar can join some letters into a wide component while
        # other full-height letters lie inside its envelope. These are actual
        # same-baseline glyphs, not an independent text row above or below it.
        for piece in pieces:
            a = piece["box"]
            if (piece["label"] not in selected_labels and box[0] <= a[0] < a[2] <= box[2]
                    and piece["height"] >= .60 * body_height
                    and abs((a[1]+a[3]-box[1]-box[3])/2) <= max(1, .25*body_height)):
                selected.append(piece)
                selected_labels.add(piece["label"])
                enclosed.append(list(a))
    kept, ambiguous = [], []
    for piece in pieces:
        if piece["label"] in selected_labels or piece["height"] > max(3, .35*body_height):
            continue
        other = owners.get(piece["label"])
        if other != selected_key and other in cores and cores[other][1] > max(3, .50 * body_height):
            # A small serif/dot belonging to a different observed full text
            # row is not an ambiguous detached mark of this word.
            continue
        a = piece["box"]
        distance = max(box[1]-a[3], a[1]-box[3], 0)
        horizontal_overlap = min(a[2], box[2]) > max(a[0], box[0])
        horizontal_gap = max(a[0]-box[2], box[0]-a[2], 0)
        # A plus/minus, punctuation stroke or detached glyph piece can lie
        # beside the word on its observed baseline rather than above it.
        same_baseline_neighbor = distance == 0 and horizontal_gap <= gap
        if not horizontal_overlap and not same_baseline_neighbor:
            continue
        if distance <= max(4, .65*body_height):
            selected.append(piece)
            selected_labels.add(piece["label"])
            kept.append(list(a))
        else:
            ambiguous.append(list(a))
    box = envelope(selected)
    # Tight white margin prevents a separately grouped baseline from entering
    # the subsequent x projection. Any ink actually cut here stays held by
    # the original eight-neighbor boundary audit.
    chosen = [max(x0, box[0]-1), max(y0, box[1]-1), min(x1, box[2]+1), min(y1, box[3]+1)]
    unexpected = set(np.unique(labels[chosen[1]-y0:chosen[3]-y0, chosen[0]-x0:chosen[2]-x0]).tolist()) - {0} - selected_labels
    return chosen, {"method": "two_dimensional_observed_glyph_overlap",
                    "selected_glyph_boxes": [p["box"] for p in selected],
                    "small_marks_attached_before_word_merge": attached_marks,
                    "enclosed_same_baseline_glyphs": enclosed,
                    "retained_detached_small_ink_rows": kept,
                    "ambiguous_detached_small_ink_rows": ambiguous,
                    "independent_ink_inside_selected_rectangle": sorted(unexpected)}


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


def _known_external_lines(mask, region, observation, pitch, stroke, light_mask=None, gray=None):
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
    for marker in stroke.get("number_lane_markers") or ([stroke["marker"]] if stroke.get("marker") else []):
        mx0,my0,mx1,my1=_clip(marker["connected_marker_region"],mask.shape)
        contours=marker.get("hollow_edge_contours_img") or []
        pattern=np.zeros((my1-my0,mx1-mx0),np.uint8)
        for contour in contours:
            points=np.asarray(contour,np.int32)-[mx0,my0]
            cv2.polylines(pattern,[points],True,1,1)
        # Hole contours lie inside the visible stroke. Its measured raster
        # outline and antialiasing can extend two pixels beyond that contour.
        # This remains bounded by the actual marker region and text cores.
        fringe_pattern=cv2.dilate(pattern,np.ones((5,5),np.uint8)).astype(bool)
        pattern=cv2.dilate(pattern,np.ones((3,3),np.uint8)).astype(bool)
        # The independently measured outer line also proves the narrow bridge
        # joining opposing triangles; it lies below the number's body.
        probe=marker["outer_line_probe"]
        pattern[max(0,probe[1]-my0):min(my1-my0,probe[3]-my0),:]=True
        proposed=pattern & light[my0:my1,mx0:mx1]
        if gray is not None:
            # The larger radius only supplies pale raster fringes. It is not
            # evidence that a nearby black letter core touches the marker.
            pale_fringe=fringe_pattern & light[my0:my1,mx0:mx1] & (gray[my0:my1,mx0:mx1]>=100)
        else:
            pale_fringe=np.zeros_like(proposed)
        # Preserve the high-contrast cores of all observed text fragments,
        # including a one-pixel-wide "1". A core extending above the marker
        # is text evidence, never a dispensable part of the marker outline.
        core=np.zeros_like(mask)
        if gray is not None:
            cx0,cy0,cx1,cy1=x0,min(y0,my0),x1,max(y1,my1)
            count,labels,stats,_=cv2.connectedComponentsWithStats((gray[cy0:cy1,cx0:cx1]<100).astype(np.uint8),8)
            for index in range(1,count):
                sx,sy,sw,sh,_=stats[index]
                if sh>=3 and (cy0+sy < marker["hole_bbox"][1]-2
                              or cy0+sy+sh > marker["hole_bbox"][3]+2):
                    core[cy0:cy1,cx0:cx1]|=labels==index
        core_intersection=proposed & core[my0:my1,mx0:mx1]
        # A previous row's numeral can touch its own marker above this cell.
        # Keep that numeral protected, but do not make its unresolved contact
        # veto a separate current word. Only intersections inside the current
        # measured number cell can hold this crop; no text outside it is used.
        xa,ya=min(mx1-mx0,max(0,x0-mx0)),min(my1-my0,max(0,y0-my0))
        xb,yb=max(0,min(mx1-mx0,x1-mx0)),max(0,min(my1-my0,y1-my0))
        overlap=bool(xb>xa and yb>ya and np.any(core_intersection[ya:yb,xa:xb]))
        outside_intersections=int(core_intersection.sum())-int(core_intersection[ya:yb,xa:xb].sum())
        protected=cv2.dilate(core.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        if not contours or gray is None or overlap:
            stroke["marker_cleanup_unresolved"]=True
        else:
            selected=(proposed | pale_fringe) & ~protected[my0:my1,mx0:mx1]
            known[my0:my1,mx0:mx1]|=selected
            yy,xx=np.nonzero(selected)
            evidence.append(dict(axis="measured_hollow_marker",region=[mx0,my0,mx1,my1],
                                 method="measured_hollow_contours_with_text_core_protection",
                                 protected_text_core_intersection=False,
                                 protected_core_pixels_outside_current_cell=outside_intersections,
                                 measured_pixels=[[mx0+int(x),my0+int(y)] for y,x in zip(yy,xx)]))
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


def _complete_group(mask, region, observation, role, pitch, known_lines=None, original_mask=None,
                    stroke=None, line_evidence=None):
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
    marker_pixels={tuple(p) for e in (line_evidence or [])
                   if e.get("axis")=="measured_hollow_marker"
                   and e.get("protected_text_core_intersection") is False
                   for p in e.get("measured_pixels",[])}
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
                if any(original[int(y),column] and (column,int(y)) not in marker_pixels
                       for delta in (-1,0,1)
                       for y in np.clip(active+delta,0,mask.shape[0]-1)):
                    connected.append(side)
        else:
            row = y0 - 1 if side == 1 else y1
            border = y0 if side == 1 else y1 - 1
            if 0 <= row < mask.shape[0]:
                active = np.flatnonzero(mask[border, glyph[0]:glyph[2]]) + glyph[0]
                if any(original[row,int(x)] and (int(x),row) not in marker_pixels
                       for delta in (-1,0,1)
                       for x in np.clip(active+delta,0,mask.shape[1]-1)):
                    connected.append(side)
    contact, nearby_fringe = False, False
    if known_lines is not None:
        xa, ya, xb, yb = max(0, glyph[0]-1), max(0, glyph[1]-1), min(mask.shape[1], glyph[2]+1), min(mask.shape[0], glyph[3]+1)
        retained = np.zeros((yb-ya, xb-xa), np.uint8)
        retained[glyph[1]-ya:glyph[3]-ya, glyph[0]-xa:glyph[2]-xa] = mask[glyph[1]:glyph[3], glyph[0]:glyph[2]]
        adjacent = cv2.dilate(retained, np.ones((3, 3), np.uint8)) > 0
        contacted = adjacent & known_lines[ya:yb, xa:xb] & original[ya:yb, xa:xb]
        contact = bool(np.any(contacted))
        if contact and role=="pin_number":
            # Pale marker fringes can touch a numeral's antialiasing without
            # sharing its protected original core. Only these exactly measured
            # contour pixels qualify; ordinary wires and core intersections
            # retain the existing unresolved-contact policy.
            cy,cx=np.nonzero(contacted)
            if marker_pixels and all((xa+int(x),ya+int(y)) in marker_pixels for y,x in zip(cy,cx)):
                contact=False
        # A measured *upper raster fringe* may darken immediately beneath a
        # complete numeral. It is not a clipped digit merely by 3x3 proximity:
        # no source numeral pixel inside the selected box was excluded. This
        # narrow exception needs a complete cell margin and a separate current
        # stroke core. Vertical contact, actual core contact, internal erased
        # ink and a fringe's dark cap protruding past the word all stay held.
        if contact and not any(touch) and role == "pin_number" and isinstance(stroke, dict):
            fringe_row = stroke["stroke_top"]-1
            hits_y, hits_x = np.nonzero(contacted)
            hits_y, hits_x = hits_y+ya, hits_x+xa
            internal = np.any(known_lines[glyph[1]:glyph[3], glyph[0]:glyph[2]]
                              & original[glyph[1]:glyph[3], glyph[0]:glyph[2]])
            measured_fringe = [line for line in (line_evidence or [])
                               if line["axis"] == "horizontal" and line["row"] == fringe_row]
            vertical_contact = any(line["axis"].startswith("vertical")
                                   and any(x == line["column"] and line["span"][0] <= y < line["span"][1]
                                           for x, y in zip(hits_x, hits_y))
                                   for line in (line_evidence or []))
            nearby_fringe = (not internal and not vertical_contact and glyph[3] <= fringe_row
                             and bool(measured_fringe) and np.all(hits_y == fringe_row))
            if nearby_fringe:
                dark_runs = _groups(original[fringe_row], 0)
                for a, b in dark_runs:
                    if any(a <= x < b for x in hits_x) and not (glyph[0] <= a and b <= glyph[2]):
                        nearby_fringe = False
                        break
            if nearby_fringe:
                contact = False
            nearby_fringe = bool(nearby_fringe)
    # If a numeral meets a measured wire, erasing their connected lower ink is
    # speculative. Raw source pixels, rather than cleaned pixels, audit this.
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask[glyph[1]:glyph[3], glyph[0]:glyph[2]].astype(np.uint8), 8)
    return glyph, {"method": "complete_original_ink_group_inside_physical_row_cell",
                   "integrity_unresolved": bool(connected) or contact,
                   "removed_line_touches_retained_glyph": contact,
                   "nearby_measured_horizontal_fringe_only": nearby_fringe,
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
    marker_mask = (gray < 230) & (chroma < 30)
    pale_marker_mask = (gray < 245) & (chroma < 30)
    mono_masks = mask, light_mask, marker_mask, pale_marker_mask
    channels = image.astype(np.int16)
    blue, green, red = channels[:,:,0], channels[:,:,1], channels[:,:,2]
    red_core = (red-green >= 30) & (red-blue >= 30) & (gray < 190)
    green_core = (green-red >= 30) & (green-blue >= 30) & (gray < 190)
    wire_core = red_core | green_core
    wire_light = (((red-green >= 8) & (red-blue >= 8))
                  | ((green-red >= 8) & (green-blue >= 8))) & (gray < 245)
    color_light = wire_light | ((gray < 230) & (chroma < 30))
    color_masks = gray < 190, color_light, gray < 230, color_light
    observations, output = list(mono_observations(doc)), []
    for ob in observations:
        color = ob["evidence"].get("source") in COLOR_CONTACT_SOURCES
        mask, light_mask, marker_mask, pale_marker_mask = color_masks if color else mono_masks
        peers = [p for p in observations if p["component_id"] == ob["component_id"] and p["edge"] == ob["edge"]]
        current_marker_mask=(pale_marker_mask if ob['evidence'].get('marker',{}).get('ink_threshold')==245 else marker_mask)
        inside, outside, pitch, stroke = _regions(ob, peers, mask, light_mask, current_marker_mask,
                                                 red_core if color else None)
        if stroke is None:
            # Cached geometric evidence cannot open a name lane when its
            # claimed attached stroke is absent from the supplied image.
            continue
        for role, region in (("pin_name", inside), ("pin_number", outside)):
            known, line_evidence = (None, [])
            local_mask = mask
            audit_mask = mask
            if role == "pin_number" and region is not None:
                wire_mask = wire_core if color else mask
                current_wire_light = wire_light if color else light_mask
                known, line_evidence = _known_external_lines(wire_mask, region, ob, pitch, stroke, current_wire_light, gray)
                local_mask = mask & ~known
                if color:
                    # Audit after removing only the independently measured
                    # chromatic wire bands. Neutral digit strokes, including
                    # their pale raster pixels, are never classified as wires.
                    audit_mask = local_mask
            full_region = list(region) if region is not None else None
            selector = _name_word_region if role == "pin_name" or color else _text_row_region
            grouping_pitch = pitch
            if color:
                other_rows = sorted(set(p["anchor_img"][1] for p in observations
                                        if p["component_id"] == ob["component_id"]))
                spacings = [b-a for a,b in zip(other_rows,other_rows[1:]) if b-a > 3]
                grouping_pitch = min([pitch]+spacings)
            region, row_evidence = (selector(local_mask, region, ob["anchor_img"], grouping_pitch,
                                            include_enclosed_baseline=color) if role == "pin_name" or color
                                    else selector(local_mask, region, ob["anchor_img"], role, pitch))
            complete = _complete_group(local_mask, region, ob, role, grouping_pitch, known_lines=known, original_mask=audit_mask,
                                       stroke=stroke, line_evidence=line_evidence)
            expanded = stroke.get("number_region_with_previous_stroke_limit")
            if (role == "pin_number" and complete is not None and expanded is not None
                    and 1 in complete[1].get("connected_ink_beyond_region_sides", [])):
                # Open the measured extension only for a visibly clipped top.
                # A complete old cell cannot import the preceding wire fringe.
                known, line_evidence = _known_external_lines(wire_mask, expanded, ob, pitch, stroke, current_wire_light, gray)
                local_mask = mask & ~known
                audit_mask = local_mask if color else mask
                full_region = list(expanded)
                region, row_evidence = (_name_word_region(local_mask, expanded, ob["anchor_img"], grouping_pitch,
                                                         include_enclosed_baseline=True) if color
                                       else _text_row_region(local_mask, expanded, ob["anchor_img"], role, pitch))
                complete = _complete_group(local_mask, region, ob, role, grouping_pitch, known_lines=known,
                                           original_mask=audit_mask, stroke=stroke, line_evidence=line_evidence)
            if complete is None:
                continue
            glyph, integrity = complete
            if role=="pin_number" and stroke.get("marker_cleanup_unresolved"):
                integrity["integrity_unresolved"]=True
                integrity["marker_cleanup_unresolved"]=True
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
            if row_evidence.get("independent_ink_inside_selected_rectangle"):
                integrity["integrity_unresolved"] = True
                integrity["unresolved_reason"] = "independent_name_baseline_inside_selected_rectangle"
            if known is not None and full_region is not None:
                a, b, c, d = full_region
                yy, xx = np.nonzero(known[b:d, a:c])
                integrity["excluded_region_pixel_count"] = len(xx)
                integrity["excluded_region_ink_bbox"] = ([a + int(xx.min()), b + int(yy.min()), a + int(xx.max()) + 1, b + int(yy.max()) + 1] if len(xx) else None)
            if not color and np.any((chroma[glyph[1]:glyph[3], glyph[0]:glyph[2]] >= 50)
                      & (gray[glyph[1]:glyph[3], glyph[0]:glyph[2]] < 230)):
                integrity["integrity_unresolved"] = True
                integrity["unresolved_reason"] = "non_monochrome_ink_inside_original_crop"
            target = {k: ob[k] for k in ("component_id", "pin_index", "edge", "point_img", "anchor_img")}
            flags = ["glyph_integrity_unresolved"] if integrity["integrity_unresolved"] else []
            output.append({"bbox": glyph, "geometry_bbox": list(glyph), "glyph_bbox": list(glyph),
                           "region": region, "role": role, "vertical": False,
                           "pin_target": target, "flags": flags,
                           "cleanup_evidence": dict(integrity, stroke_probe=stroke,
                               sampling_color_mode="original_grayscale_of_color_contact" if color else "monochrome",
                               pitch_pixels=pitch, physical_evidence=ob["evidence"],
                               same_side_rows=[p["anchor_img"][1] for p in peers],
                               independent_of_ocr_text=True)})
    return output, gray
