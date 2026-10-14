"""Image-supported terminals of axial passives and single-port power symbols.

This module never opens labels or reads text. Coloured symbols use measured
pin/wire interfaces, connected to the detected body, as their coordinates.
Red/green boundaries, filled junction centres and shared one-pixel contacts
follow the user's image rules. Additional colour roles and measured symmetry
are handled by passive_terminal_geometry. Monochrome circuits, ICs and
ambiguous contacts retain their previous candidates.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .pincolor import _colour_masks, _green_path, _round, _runs


AXIAL_TYPES = frozenset({"r", "c", "l", "d", "led", "fuse"})
SINGLE_TYPES = frozenset({"gnd", "v", "vcc"})


def _blended_red_green_pixels(bgr):
    """Dark red/green mixtures have both warm channels above blue."""
    blue, green, red = np.moveaxis(bgr.astype(np.int16), -1, 0)
    return ((np.minimum(red, green) - blue >= 18) & (np.abs(red - green) < 18)
            & (np.minimum(red, green) >= 40))


def _blend_bridges(red, blends):
    """Link red strokes only across one to three observed mixed pixels."""
    bridge = np.zeros_like(red, dtype=bool)
    for gap in (1, 2, 3):
        distance = gap + 1
        if red.shape[1] > distance:
            width = red.shape[1] - distance
            valid = red[:, :width] & red[:, distance:]
            for step in range(1, distance):
                valid &= blends[:, step:step + width]
            for step in range(1, distance):
                bridge[:, step:step + width] |= valid
        if red.shape[0] > distance:
            height = red.shape[0] - distance
            valid = red[:height, :] & red[distance:, :]
            for step in range(1, distance):
                valid &= blends[step:step + height, :]
            for step in range(1, distance):
                bridge[step:step + height, :] |= valid
    return bridge


def _box(value):
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _ray(mask, point, direction, length=12, start=1):
    h, w = mask.shape
    dx, dy = direction
    values = []
    for step in range(start, start + length):
        x, y = point[0] + dx * step, point[1] + dy * step
        values.append(any(0 <= _round(x - dy * q) < w
                          and 0 <= _round(y + dx * q) < h
                          and mask[_round(y + dx * q), _round(x - dy * q)]
                          for q in (-1, 0, 1)))
    return sum(values) / max(1, len(values))


def _curved_lead_path(body_red, point, direction, raw, cross):
    """A short inductor lead can enter a connected coil instead of a line."""
    h, w = body_red.shape
    dx, dy = direction
    x, y = _round(point[0]), _round(point[1])
    radius = int(math.ceil(max(2, min(8, cross / 2 + 2))))

    def observed(step, lateral):
        px, py = x + dx * step - dy * lateral, y + dy * step + dx * lateral
        return 0 <= px < w and 0 <= py < h and body_red[py, px]

    def column_path(step, entries):
        reachable = {q for q in entries if observed(step, q)}
        # Steep arcs can move along a column before entering the next one.
        # Extend only through adjacent red pixels in that same column.
        for _ in range(2 * radius + 1):
            adjacent = {q for q in range(-radius, radius + 1)
                        if observed(step, q) and (q - 1 in reachable or q + 1 in reachable)}
            if adjacent <= reachable:
                break
            reachable |= adjacent
        return reachable

    reachable = column_path(0, (-1, 0, 1))
    for step in range(1, 7):
        # Every axial column/row must contain an adjacent observed red pixel.
        # A wider probe alone could jump a white gap to an unrelated stroke.
        entries = {q for q in range(-radius, radius + 1)
                   if any(abs(q - prev) <= 1 for prev in reachable)}
        reachable = column_path(step, entries)
        if not reachable:
            return None
    if not any(raw[0] - 2 <= x + dx * 6 - dy * q <= raw[2] + 2
               and raw[1] - 2 <= y + dy * 6 + dx * q <= raw[3] + 2 for q in reachable):
        return None
    return {'kind': 'curved_red_body_path', 'length_pixels': 6, 'max_lateral_pixels': radius}


def _transition_point(red, green, point, direction, max_gap=3):
    """Midpoint of facing coloured pixels, including up to two neutral pixels."""
    h, w = red.shape
    dx, dy = direction
    pairs = []
    for q in (-1, 0, 1):
        for step in range(-4, 5):
            x = _round(point[0] + dx * step - dy * q)
            y = _round(point[1] + dy * step + dx * q)
            if not (0 <= x < w and 0 <= y < h and red[y, x]):
                continue
            for gap in range(1, max_gap + 1):
                gx, gy = x + dx * gap, y + dy * gap
                if 0 <= gx < w and 0 <= gy < h and green[gy, gx]:
                    pairs.append((gap, ((x + gx) / 2, (y + gy) / 2)))
                    break
    if not pairs:
        return None
    closest = min(gap for gap, _ in pairs)
    return tuple(np.mean([p for gap, p in pairs if gap == closest], axis=0).tolist())


def _opposed_red_lead(body_red, point, direction, raw):
    """An observed opposite LED stem can corroborate an offset detector axis."""
    h, w = body_red.shape
    axis = 0 if direction[0] else 1
    origin = list(point)
    origin[axis] = raw[axis] if direction[axis] > 0 else raw[axis + 2]
    dx, dy = -direction[0], -direction[1]
    for q in (-1, 0, 1):
        pixels = [(_round(origin[0] + dx * step - dy * q),
                   _round(origin[1] + dy * step + dx * q)) for step in range(1, 7)]
        if all(0 <= x < w and 0 <= y < h and body_red[y, x] for x, y in pixels):
            return pixels
    return None


def _wire_path(green, point, direction):
    """Reuse IC wire/NC evidence, allowing an observed orthogonal continuation."""
    dx, dy = direction
    edge = 1 if dx > 0 else 3 if dx < 0 else 2 if dy > 0 else 0
    frame, tangent = (point[0], point[1]) if dx else (point[1], point[0])
    path = _green_path(green, frame, tangent, edge, 0)
    if path["marker_near_tip"] or path["first"] is None:
        return None
    if path["connected"] is True:
        return {"kind": "straight_green_wire", "hits": path["hits"]}
    # A short normal run can meet a wire that immediately turns or branches.
    # Start the transverse probes on actual green pixels of that same run;
    # do not bridge a white gap to an unrelated parallel wire.
    h, w = green.shape
    left, top = max(0, int(point[0]) - 36), max(0, int(point[1]) - 36)
    local = green[top:min(h, int(point[1]) + 37), left:min(w, int(point[0]) + 37)]
    _, labels = cv2.connectedComponents(local.astype(np.uint8), 8)
    initial_labels = set()
    for lateral in (-1, 0, 1):
        x = _round(point[0] + dx * path['first'] - dy * lateral)
        y = _round(point[1] + dy * path['first'] + dx * lateral)
        if left <= x < left + local.shape[1] and top <= y < top + local.shape[0]:
            initial_labels.add(int(labels[y - top, x - left]))
    initial_labels.discard(0)
    measured_green = np.isin(labels, list(initial_labels))
    for step in range(path["first"], min(path["last"], path["first"] + 3) + 1):
        for lateral in (-1, 0, 1):
            x = _round(point[0] + dx * step - dy * lateral)
            y = _round(point[1] + dy * step + dx * lateral)
            if not (0 <= x < w and 0 <= y < h and green[y, x]):
                continue
            if int(labels[y - top, x - left]) not in initial_labels:
                continue
            for bx, by in ((-dy, dx), (dy, -dx)):
                branch_edge = 1 if bx > 0 else 3 if bx < 0 else 2 if by > 0 else 0
                branch_frame, branch_tangent = (x - left, y - top) if bx else (y - top, x - left)
                branch = _green_path(measured_green, branch_frame, branch_tangent, branch_edge, 0)
                if branch["first"] == 1 and branch["connected"] is True:
                    return {"kind": "turning_green_wire", "hits": branch["hits"],
                            "turn_img": [x, y], "branch_direction": [bx, by]}
    return None


def _short_wire_to_peer(red, green, labels, point, direction, raw, other_boxes, blends=None):
    """A brief wire needs a continuous facing red lead attached to a peer."""
    dx, dy = direction
    edge = 1 if dx > 0 else 3 if dx < 0 else 2 if dy > 0 else 0
    frame, tangent = point if dx else (point[1], point[0])
    path = _green_path(green, frame, tangent, edge, 0)
    if path['marker_near_tip'] or path['first'] is None or path['hits'] >= 6:
        return None
    h, w = red.shape
    axis = 0 if dx else 1
    cross_axis = 1 - axis
    evidence = {}
    for q in (-1, 0, 1):
        def pixel(step):
            # Probe the adjacent raster cell, not a full pixel beyond a
            # half-pixel boundary (which would skip its first green cell).
            return (_round(point[0] + dx * (step - .5) - dy * q),
                    _round(point[1] + dy * (step - .5) + dx * q))

        first = next((step for step in range(1, 5)
                      if 0 <= pixel(step)[0] < w and 0 <= pixel(step)[1] < h
                      and green[pixel(step)[1], pixel(step)[0]]), None)
        if first is None:
            continue
        pixels = []
        for step in range(first, first + 6):
            x, y = pixel(step)
            if not (0 <= x < w and 0 <= y < h and green[y, x]):
                break
            pixels.append([x, y])
        if not 2 <= len(pixels) <= 5:
            continue
        end = first + len(pixels)
        mixed = []
        if blends is not None:
            for _ in range(3):
                x, y = pixel(end)
                if not (0 <= x < w and 0 <= y < h and blends[y, x]):
                    break
                mixed.append([x, y])
                end += 1
        facing = [pixel(step) for step in range(end, end + 6)]
        if not all(0 <= x < w and 0 <= y < h and red[y, x] for x, y in facing):
            continue
        label = int(labels[facing[0][1], facing[0][0]])
        if not label or any(labels[y, x] != label for x, y in facing):
            continue
        for cid, peer in other_boxes:
            if (min(raw[2], peer[2]) > max(raw[0], peer[0])
                    and min(raw[3], peer[3]) > max(raw[1], peer[1])):
                continue
            near_edge = peer[axis] if direction[axis] > 0 else peer[axis + 2]
            distance = (near_edge - point[axis]) * direction[axis]
            if not end <= distance <= 48:
                continue
            if not peer[cross_axis] - 2 <= point[cross_axis] <= peer[cross_axis + 2] + 2:
                continue
            into_body = [pixel(step) for step in range(end, int(math.ceil(distance)) + 1)]
            if not all(0 <= x < w and 0 <= y < h and red[y, x] and labels[y, x] == label
                       for x, y in into_body):
                continue
            target = list(point)
            target[axis] = near_edge
            if _intervening_body(target, direction, raw, [(k, b) for k, b in other_boxes if k != cid]):
                continue
            x0, y0 = max(0, int(peer[0])), max(0, int(peer[1]))
            x1, y1 = min(w, int(math.ceil(peer[2])) + 1), min(h, int(math.ceil(peer[3])) + 1)
            yy, xx = np.where(labels[y0:y1, x0:x1] == label)
            transverse = yy if dx else xx
            # A detection over a narrow straight stem is not another body.
            if not transverse.size or np.ptp(transverse) < 4:
                continue
            evidence[cid] = {'kind': 'short_green_wire_to_detected_body',
                             'green_pixels_img': pixels, 'peer_id': cid,
                             'facing_red_pixels_img': [list(v) for v in facing]}
            if mixed:
                evidence[cid]['blend_pixels_img'] = mixed
    return next(iter(evidence.values())) if len(evidence) == 1 else None


def _short_wire_to_junction(red, green, point, direction):
    """A compact red node needs two other observed long green wire arms."""
    dx, dy = direction
    edge = 1 if dx > 0 else 3 if dx < 0 else 2 if dy > 0 else 0
    frame, tangent = point if dx else (point[1], point[0])
    path = _green_path(green, frame, tangent, edge, 0)
    if path['marker_near_tip'] or path['first'] is None or path['hits'] >= 6:
        return None
    h, w = red.shape
    nodes = []
    for size in (5, 3):
        filled = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        _, blobs, stats, centres = cv2.connectedComponentsWithStats(filled, 8)
        nodes.extend((stat, centre, blobs == label, size == 3)
                     for label, (stat, centre) in enumerate(zip(stats[1:], centres[1:]), 1))
    _, red_labels = cv2.connectedComponents(red.astype(np.uint8), 8)
    for q in (-1, 0, 1):
        def pixel(step):
            return (_round(point[0] + dx * (step - .5) - dy * q),
                    _round(point[1] + dy * (step - .5) + dx * q))

        first = next((step for step in range(1, 5)
                      if 0 <= pixel(step)[0] < w and 0 <= pixel(step)[1] < h
                      and green[pixel(step)[1], pixel(step)[0]]), None)
        if first is None:
            continue
        pixels = []
        for step in range(first, first + 6):
            x, y = pixel(step)
            if not (0 <= x < w and 0 <= y < h and green[y, x]):
                break
            pixels.append([x, y])
        if not 2 <= len(pixels) <= 5:
            continue
        face = pixel(first + len(pixels))
        fx, fy = face
        if not (0 <= fx < w and 0 <= fy < h and red[fy, fx]):
            continue
        red_label = int(red_labels[fy, fx])
        for stat, centre, blob, needs_core in nodes:
            x, y, bw, bh, area = map(int, stat)
            if not (5 <= min(bw, bh) <= max(bw, bh) <= 12
                    and min(bw, bh) >= .7 * max(bw, bh) and area >= .55 * bw * bh):
                continue
            # A six-pixel filled diamond can lose its whole core to a 5px
            # opening. A smaller opening retains it, but a thick red cross
            # also survives that opening: require its observed filled centre.
            core_size = max(3, int(math.ceil(.6 * min(bw, bh))))
            core_x, core_y = x + (bw - core_size) // 2, y + (bh - core_size) // 2
            if needs_core and not np.all(blob[core_y:core_y + core_size, core_x:core_x + core_size]):
                continue
            rim = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if not rim[fy, fx] or not np.any(blob & (red_labels == red_label)):
                continue
            node_red = red & rim & (red_labels == red_label)
            branches = []
            for bx, by in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (bx, by) == (-dx, -dy):
                    continue
                tip = ((x + bw - 1 if bx > 0 else x) if bx else centre[0],
                       (y + bh - 1 if by > 0 else y) if by else centre[1])
                boundary = _transition_point(node_red, green, tip, (bx, by), max_gap=1)
                if boundary is None:
                    continue
                branch_edge = 1 if bx > 0 else 3 if bx < 0 else 2 if by > 0 else 0
                branch_frame, branch_tangent = boundary if bx else (boundary[1], boundary[0])
                wire = _green_path(green, branch_frame, branch_tangent, branch_edge, 0)
                if wire['connected'] is True:
                    branches.append({'direction': [bx, by], 'point_img': list(boundary), 'hits': wire['hits']})
            if len(branches) >= 2:
                return {'kind': 'short_green_wire_to_red_junction', 'green_pixels_img': pixels,
                        'wire_junction_img': centre.tolist(), 'junction_wire_arms': branches}
    return None


def _short_observed_wire(red, green, blends, point, direction):
    """A one-to-five-pixel green span is valid without a detected peer body.

    Require the observed span to end at red ink, allowing only actual mixed
    boundary pixels. This does not bridge a white break or infer a peer/net.
    The caller has already checked the component's continuous inward lead.
    """
    dx, dy = direction
    edge = 1 if dx > 0 else 3 if dx < 0 else 2 if dy > 0 else 0
    frame, tangent = point if dx else (point[1], point[0])
    if _green_path(green, frame, tangent, edge, 0)['marker_near_tip']:
        return None
    h, w = green.shape
    for q in (-1, 0, 1):
        def pixel(step):
            return (_round(point[0] + dx * (step - .5) - dy * q),
                    _round(point[1] + dy * (step - .5) + dx * q))

        def is_mask(mask, step):
            x, y = pixel(step)
            return 0 <= x < w and 0 <= y < h and mask[y, x]

        first = next((s for s in range(1, 5) if is_mask(green, s)), None)
        if first is None:
            continue
        pixels = []
        for step in range(first, first + 6):
            if not is_mask(green, step):
                break
            pixels.append(list(pixel(step)))
        if not 1 <= len(pixels) <= 5:
            continue
        end = first + len(pixels)
        mixed = []
        if blends is not None:
            while len(mixed) < 3 and is_mask(blends, end):
                mixed.append(list(pixel(end)))
                end += 1
        if not all(is_mask(red, s) for s in (end, end + 1)):
            continue
        evidence = {'kind': 'short_observed_green_wire', 'green_pixels_img': pixels,
                    'facing_red_pixels_img': [list(pixel(end)), list(pixel(end + 1))]}
        if mixed:
            evidence['blend_pixels_img'] = mixed
        return evidence
    return None


def _single_pixel_contacts(red, green, blends, component, box, axis_y,
                           other_boxes, ground_axes=(), ground_stems=None):
    """Directly opposing detected pins share an actual one-pixel colour mark.

    A dark red/green mixture is an observed colour mark too. Only a single
    axial pixel, sandwiched by continuous red leads into two distinct bodies,
    is shared. Longer green spans retain their separate boundary positions.
    """
    raw = _box(component.get('bbox_raw_img')) or box
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h, w = red.shape
    left, top = max(0, int(box[0] - 48)), max(0, int(box[1] - 48))
    right, bottom = min(w, math.ceil(box[2] + 48)), min(h, math.ceil(box[3] + 48))
    r, g = red[top:bottom, left:right], green[top:bottom, left:right]
    mixed = blends[top:bottom, left:right] if blends is not None else np.zeros_like(r)
    mark = g | mixed
    verticals = (axis_y,) if component['type'] in AXIAL_TYPES else tuple(ground_axes) or (False, True)
    candidates = []
    ground_stems = ground_stems or {}
    for vertical in verticals:
        sandwich = np.zeros_like(r)
        if vertical and r.shape[0] > 2:
            sandwich[1:-1] = mark[1:-1] & r[:-2] & r[2:]
        elif not vertical and r.shape[1] > 2:
            sandwich[:, 1:-1] = mark[:, 1:-1] & r[:, :-2] & r[:, 2:]
        _, labels, stats, centres = cv2.connectedComponentsWithStats(sandwich.astype(np.uint8), 8)
        for label, (stat, centre) in enumerate(zip(stats[1:], centres[1:]), 1):
            bw, bh = int(stat[2]), int(stat[3])
            if (bh if vertical else bw) != 1 or (bw if vertical else bh) > 6:
                continue
            point = (float(centre[0] + left), float(centre[1] + top))
            axis, cross_axis = (1, 0) if vertical else (0, 1)
            axial = point[axis] - (cy if vertical else cx)
            lateral = abs(point[cross_axis] - (cx if vertical else cy))
            cross = box[cross_axis + 2] - box[cross_axis]
            half = (box[axis + 2] - box[axis]) / 2
            centred_limit = max(5, cross * .3)
            limit = max(centred_limit, cross * .75) if component['type'] in AXIAL_TYPES else centred_limit
            if abs(axial) < max(4, half * .65) or lateral > limit:
                continue
            direction = (0, 1 if axial > 0 else -1) if vertical else (1 if axial > 0 else -1, 0)
            inward = (-direction[0], -direction[1])
            own_stems = ground_stems.get(component.get('id'), [])
            own_edge = raw[axis + 2] if direction[axis] > 0 else raw[axis]
            if own_stems:
                facing = [stem for stem in own_stems if stem['axis_y'] == vertical
                          and stem['outward_side'] == direction[axis]
                          and abs(stem['point'][cross_axis] - point[cross_axis]) <= 2.5
                          and (point[axis] - stem['point'][axis]) * direction[axis] > 0]
                if not facing:
                    continue
                own_edge = min((stem['point'][axis] for stem in facing), key=lambda value: abs(value - point[axis]))
            own_distance = (point[axis] - own_edge) * direction[axis]
            if not 1 <= own_distance <= 48:
                continue
            own_path = None
            for q in (-1, 0, 1):
                stop = max(0, int(math.floor((_round(own_edge) - point[axis]) * inward[axis] + 1e-6)))
                trail = [(_round(point[0] + inward[0] * s - inward[1] * q),
                          _round(point[1] + inward[1] * s + inward[0] * q))
                         for s in range(1, stop + 1)]
                if trail and all(0 <= x < w and 0 <= y < h and red[y, x] for x, y in trail):
                    own_path = trail
                    break
            if own_path is None:
                continue
            if _intervening_body(point, direction, raw, other_boxes) is not None:
                continue
            evidence = {}
            for peer_id, peer in other_boxes:
                if min(raw[2], peer[2]) > max(raw[0], peer[0]) and min(raw[3], peer[3]) > max(raw[1], peer[1]):
                    continue
                edge = peer[axis] if direction[axis] > 0 else peer[axis + 2]
                stems = ground_stems.get(peer_id, [])
                if stems:
                    facing_stems = [stem for stem in stems if stem['axis_y'] == vertical
                                    and stem['outward_side'] == -direction[axis]
                                    and abs(stem['point'][cross_axis] - point[cross_axis]) <= 2.5
                                    and (stem['point'][axis] - point[axis]) * direction[axis] > 0]
                    if not facing_stems:
                        continue
                    edge = min((stem['point'][axis] for stem in facing_stems),
                               key=lambda value: abs(value - point[axis]))
                distance = (edge - point[axis]) * direction[axis]
                if not 1 <= distance <= 48 or not peer[cross_axis] - 2 <= point[cross_axis] <= peer[cross_axis + 2] + 2:
                    continue
                target = list(point)
                target[axis] = edge
                if _intervening_body(target, direction, raw, [(cid, b) for cid, b in other_boxes if cid != peer_id]):
                    continue
                dx, dy = direction
                for q in (-1, 0, 1):
                    stop = max(0, int(math.floor((_round(edge) - point[axis]) * direction[axis] + 1e-6)))
                    path = [(_round(point[0] + dx * s - dy * q),
                             _round(point[1] + dy * s + dx * q)) for s in range(1, stop + 1)]
                    if not path or not all(0 <= x < w and 0 <= y < h and red[y, x] for x, y in path):
                        continue
                    yy, xx = np.where(labels == label)
                    pixels = [[int(x + left), int(y + top)] for x, y in zip(xx, yy)]
                    evidence[peer_id] = {'kind': 'direct_pin_single_color_pixel', 'peer_id': peer_id,
                                         'marker_pixels_img': pixels,
                                         'marker_color': 'green' if np.any(g & (labels == label)) else 'red_green_blend',
                                         'facing_red_pixels_img': [list(p) for p in path],
                                         'own_red_pixels_img': [list(p) for p in own_path],
                                         'coordinate_img': list(point)}
                    break
            if len(evidence) == 1:
                candidates.append({'point': point, 'side': 1 if axial > 0 else -1,
                                   'source': 'shared_single_green_pixel', 'off_center': lateral > centred_limit,
                                   'blocked_by': None, 'wire': next(iter(evidence.values()))})
    return candidates


def _peer_red_body_path(red, start, direction, peer, typ, stem=None):
    """Reach observed body ink, including an outlined net-port tip.

    The endpoint is a raster cell overlapping the measured box. Fractional
    box edges must not require one additional, white cell outside the body.
    Net-port outlines can turn before entering an inset detector box.
    """
    h, w = red.shape
    dx, dy = direction
    axis, cross_axis = (0, 1) if dx else (1, 0)
    edge = stem['point'][axis] if stem else peer[axis] if direction[axis] > 0 else peer[axis + 2]
    distance = (edge - start[axis]) * direction[axis]
    if not -1 <= distance <= 48:
        return None
    end = _round(edge)
    curved = typ in {'net_input', 'net_output', 'l'}
    radius = min(8, math.ceil((peer[cross_axis + 2] - peer[cross_axis]) / 2 + 2)) if curved else 1
    tangent = stem['point'][cross_axis] if stem else start[cross_axis]
    queue, visited = deque(), set()
    for q in (-1, 0, 1):
        cell = (_round(start[0] - dy * q), _round(start[1] + dx * q))
        if 0 <= cell[0] < w and 0 <= cell[1] < h and red[cell[1], cell[0]]:
            queue.append((cell, [list(cell)]))
            visited.add(cell)
    while queue:
        (x, y), path = queue.popleft()
        cell = (x, y)
        if abs(cell[axis] - end) <= .5:
            if stem or (peer[0] - .5 <= x <= peer[2] + .5 and peer[1] - .5 <= y <= peer[3] + .5):
                return path if len(path) >= 3 else None
        moves = ((dx, dy), (dx - dy, dy + dx), (dx + dy, dy - dx))
        if curved:
            moves += ((-dy, dx), (dy, -dx))
        for mx, my in moves:
            nxt = (x + mx, y + my)
            if (nxt in visited or not (0 <= nxt[0] < w and 0 <= nxt[1] < h)
                    or (nxt[axis] - end) * direction[axis] > 0
                    or abs(nxt[cross_axis] - tangent) > radius
                    or not red[nxt[1], nxt[0]]):
                continue
            visited.add(nxt)
            queue.append((nxt, path + [list(nxt)]))
    return None


def _mixed_span_contacts(red, green, blends, component, box, axis_y,
                         other_boxes, ground_axes, ground_stems, peer_types, mix_context):
    """Short colour spans need two continuous, independently detected leads.

    Red-dominant marks are accepted by their local green/blue contrast, not
    by reclassifying every reddish pixel. Longer spans publish the boundary
    next to this lead; only one axial pixel can be shared by both bodies.
    """
    raw = _box(component.get('bbox_raw_img')) or box
    h, w = red.shape
    left, top = max(0, int(box[0] - 48)), max(0, int(box[1] - 48))
    right, bottom = min(w, math.ceil(box[2] + 48)), min(h, math.ceil(box[3] + 48))
    tinted, path_red, warm = mix_context
    # A stroke may be two pixels wide with one green and one mixed cell.
    # Measure the whole mark; a mixed subset is not a second terminal.
    mark = (green[top:bottom, left:right] | blends[top:bottom, left:right]
            | tinted[top:bottom, left:right])
    _, labels, stats, centres = cv2.connectedComponentsWithStats(mark.astype(np.uint8), 8)
    verticals = (axis_y,) if component['type'] in AXIAL_TYPES else tuple(ground_axes) or (False, True)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    candidates = []
    for vertical in verticals:
        axis, cross_axis = (1, 0) if vertical else (0, 1)
        for label, (stat, centre) in enumerate(zip(stats[1:], centres[1:]), 1):
            bx, by, bw, bh, _ = map(int, stat)
            length, thickness = (bh, bw) if vertical else (bw, bh)
            if not 1 <= length <= 3 or thickness > 6:
                continue
            bounds = (bx + left, by + top, bx + left + bw - 1, by + top + bh - 1)
            middle = (float(centre[0] + left), float(centre[1] + top))
            axial = middle[axis] - (cy if vertical else cx)
            side = 1 if axial > 0 else -1
            cross = box[cross_axis + 2] - box[cross_axis]
            half = (box[axis + 2] - box[axis]) / 2
            lateral = abs(middle[cross_axis] - (cx if vertical else cy))
            centred_limit = max(5, cross * .3)
            limit = max(centred_limit, cross * .75) if component['type'] in AXIAL_TYPES else centred_limit
            if abs(axial) < max(4, half * .65) or lateral > limit:
                continue
            direction = (0, side) if vertical else (side, 0)
            near = bounds[axis] if side > 0 else bounds[axis + 2]
            far = bounds[axis + 2] if side > 0 else bounds[axis]
            own_start, peer_start = list(middle), list(middle)
            own_start[axis], peer_start[axis] = near - side, far + side
            own_edge = raw[axis + 2] if side > 0 else raw[axis]
            own_stems = ground_stems.get(component.get('id'), [])
            if own_stems:
                facing = [s for s in own_stems if s['axis_y'] == vertical and s['outward_side'] == side
                          and abs(s['point'][cross_axis] - middle[cross_axis]) <= 2.5]
                if not facing:
                    continue
                own_edge = min((s['point'][axis] for s in facing), key=lambda v: abs(v - near))
            if not 1 <= (near - own_edge) * side <= 48:
                continue
            target = list(middle)
            target[axis] = _round(own_edge)
            own_path = _continuous_red_segment(path_red, own_start, target, vertical)
            if not own_path or len(own_path) < 3:
                continue
            yy, xx = np.where(labels == label)
            pixels = [[int(x + left), int(y + top)] for x, y in zip(xx, yy)]
            has_green = any(green[y, x] for x, y in pixels)
            if length > 1 and has_green:
                # A real green span already has separate wire boundaries.
                continue
            has_tint = any(tinted[y, x] for x, y in pixels)
            if has_tint:
                # Ordinary red antialiasing has matching green/blue channels.
                # Both facing strokes must differ from this compact mark.
                contrast = min(int(warm[y, x]) for x, y in pixels)
                facing_warm = []
                for start in (own_start, peer_start):
                    for q in (-1, 0, 1):
                        x, y = _round(start[0] - direction[1] * q), _round(start[1] + direction[0] * q)
                        if 0 <= x < w and 0 <= y < h and path_red[y, x]:
                            facing_warm.append(int(warm[y, x]))
                if len(facing_warm) < 2 or contrast - max(facing_warm) < 14:
                    continue
            coordinate = list(middle)
            if length > 1:
                coordinate[axis] = near - side * .5
            if _intervening_body(coordinate, direction, raw, other_boxes) is not None:
                continue
            evidence = {}
            for peer_id, peer in other_boxes:
                if min(raw[2], peer[2]) > max(raw[0], peer[0]) and min(raw[3], peer[3]) > max(raw[1], peer[1]):
                    continue
                if not peer[cross_axis] - 2 <= middle[cross_axis] <= peer[cross_axis + 2] + 2:
                    continue
                stems = ground_stems.get(peer_id, [])
                facing_stems = [s for s in stems if s['axis_y'] == vertical and s['outward_side'] == -side
                                and abs(s['point'][cross_axis] - middle[cross_axis]) <= 2.5]
                if stems and not facing_stems:
                    continue
                stem = min(facing_stems, key=lambda s: abs(s['point'][axis] - far)) if facing_stems else None
                facing_path = _peer_red_body_path(path_red, peer_start, direction, peer, peer_types.get(peer_id), stem)
                if not facing_path:
                    continue
                # Demand actual body ink with transverse extent, not a box
                # placed on the same narrow straight lead.
                x0, y0 = max(0, int(peer[0])), max(0, int(peer[1]))
                x1, y1 = min(w, math.ceil(peer[2]) + 1), min(h, math.ceil(peer[3]) + 1)
                py, px = np.where(red[y0:y1, x0:x1])
                transverse = px if vertical else py
                if not stem and (not transverse.size or np.ptp(transverse) < 4):
                    continue
                target = list(middle)
                target[axis] = stem['point'][axis] if stem else peer[axis] if side > 0 else peer[axis + 2]
                if _intervening_body(target, direction, raw, [(cid, b) for cid, b in other_boxes if cid != peer_id]):
                    continue
                evidence[peer_id] = {'kind': 'direct_pin_mixed_colour_span', 'peer_id': peer_id,
                                     'marker_pixels_img': pixels, 'marker_bounds_img': list(bounds),
                                     'marker_axis_y': vertical, 'outward_direction_img': list(direction),
                                     'marker_color': ('red_dominant_blend' if has_tint else
                                                      'green' if has_green else 'red_green_blend'),
                                     'own_red_pixels_img': own_path, 'facing_red_pixels_img': facing_path,
                                     'coordinate_img': coordinate, 'axial_colour_pixels': length}
            if len(evidence) == 1:
                candidates.append({'point': tuple(coordinate), 'side': side,
                                   'source': 'shared_single_green_pixel' if length == 1 else 'mixed_colour_span_boundary',
                                   'off_center': lateral > centred_limit, 'blocked_by': None,
                                   'wire': next(iter(evidence.values()))})
    return candidates


def _filled_red_junctions(red):
    """Locate compact filled nodes, including six-pixel raster diamonds."""
    nodes = []
    for size in (5, 3):
        opened = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        for label, stat in enumerate(stats[1:], 1):
            x, y, bw, bh, area = map(int, stat)
            if not (5 <= min(bw, bh) <= max(bw, bh) <= 12
                    and min(bw, bh) >= .7 * max(bw, bh) and area >= .55 * bw * bh):
                continue
            core = max(3, math.ceil(.6 * min(bw, bh)))
            core_x, core_y = x + (bw - core) // 2, y + (bh - core) // 2
            if size == 3 and not np.all(labels[core_y:core_y + core, core_x:core_x + core] == label):
                # A thin curved lead can survive opening at one end of the
                # dot. Recover its filled, symmetric square core without
                # moving the centre into that appendage.
                diameter = min(bw, bh)
                bulbs = []
                if max(bw, bh) - diameter <= 4:
                    for oy in range(bh - diameter + 1):
                        for ox in range(bw - diameter + 1):
                            bulb = red[y + oy:y + oy + diameter, x + ox:x + ox + diameter]
                            inset = (diameter - core) // 2
                            if (not np.all(bulb[inset:inset + core, inset:inset + core])
                                    or np.mean(bulb) < .55
                                    or max(np.mean(bulb != bulb[::-1]), np.mean(bulb != bulb[:, ::-1])) > .1):
                                continue
                            bulbs.append((int(bulb.sum()), ox, oy))
                if not bulbs:
                    continue
                _, ox, oy = max(bulbs)
                x, y, bw, bh = x + ox, y + oy, diameter, diameter
            centre = (x + (bw - 1) / 2, y + (bh - 1) / 2)
            if any(math.dist(centre, node['point']) <= 1 for node in nodes):
                continue
            nodes.append({'point': centre, 'bounds': (x, y, x + bw - 1, y + bh - 1)})
    # A three-pixel coil can remain connected to the bulb after opening.
    # Fit a fully observed diamond around a filled core rather than taking
    # the bounding box of that whole connected coil. Thin appendages may
    # occupy a few corner cells; a solid square or a T is not this shape.
    core_cells = cv2.erode(red.astype(np.uint8), np.ones((4, 4), np.uint8))
    ys, xs = np.where(core_cells)
    for diameter in (6, 8, 10):
        grid_y, grid_x = np.indices((diameter, diameter))
        mid = (diameter - 1) / 2
        diamond = np.abs(grid_x - mid) + np.abs(grid_y - mid) <= diameter / 2
        for px, py in zip(xs, ys):
            x, y = int(px - diameter // 2), int(py - diameter // 2)
            if x < 0 or y < 0 or x + diameter > red.shape[1] or y + diameter > red.shape[0]:
                continue
            centre = (x + mid, y + mid)
            nearby = [node for node in nodes if math.dist(centre, node['point']) <= 2]
            if any(diameter <= max(node['bounds'][2] - node['bounds'][0] + 1,
                                   node['bounds'][3] - node['bounds'][1] + 1) for node in nearby):
                continue
            bulb = red[y:y + diameter, x:x + diameter]
            if not np.all(bulb[diamond]) or np.count_nonzero(bulb & ~diamond) > .17 * diameter * diameter:
                continue
            nodes = [node for node in nodes if node not in nearby]
            nodes.append({'point': centre, 'bounds': (x, y, x + diameter - 1, y + diameter - 1)})
    return nodes


def _continuous_red_segment(red, origin, target, vertical):
    """Observe every row/column of the lead; colour transitions stop the walk."""
    h, w = red.shape
    axis = 1 if vertical else 0
    direction = 1 if target[axis] > origin[axis] else -1
    start, end = _round(origin[axis]), _round(target[axis])
    tangent = target[1 - axis]
    paths = {}
    for coord in range(start, end + direction, direction):
        following = {}
        for q in (-1, 0, 1):
            point = (_round(tangent + q), coord) if vertical else (coord, _round(tangent + q))
            x, y = point
            if not (0 <= x < w and 0 <= y < h and red[y, x]):
                continue
            if coord == start:
                following[q] = [list(point)]
            else:
                previous = next((p for p in (q, q - 1, q + 1) if p in paths), None)
                if previous is not None:
                    following[q] = paths[previous] + [list(point)]
        paths = following
        if not paths:
            return None
    return next(iter(paths.values()), None)


def _node_wire_arms(red, other_colours, node, incoming):
    """A filled dot must also join an independently observed straight arm.

    Symbol corners and digits overlapping a stem can survive opening just
    like dots. A continuous external wire distinguishes the actual junction.
    Blue ink supports the shape check without becoming a red/green boundary.
    """
    x0, y0, x1, y1 = node['bounds']
    cx, cy = node['point']
    h, w = red.shape
    ink = red | other_colours
    core = red[y0:y1 + 1, x0:x1 + 1]
    round_dot = max(float(np.mean(core != core[::-1, :])),
                    float(np.mean(core != core[:, ::-1]))) <= .1
    arms = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if (dx, dy) == incoming:
            continue
        tip = (x1 if dx > 0 else x0 if dx < 0 else cx,
               y1 if dy > 0 else y0 if dy < 0 else cy)
        for q in (-1, 0, 1):
            trail = [(_round(tip[0] + dx * s - dy * q),
                      _round(tip[1] + dy * s + dx * q)) for s in range(1, 7)]
            if not all(0 <= x < w and 0 <= y < h and ink[y, x] for x, y in trail):
                continue
            colour_hits = sum(bool(other_colours[y, x]) for x, y in trail)
            alignment = dx * incoming[0] + dy * incoming[1]
            round_collinear_dot = alignment == -1 and round_dot
            # A thick T at a symbol's edge also has perpendicular red arms.
            # Pure-red arms need the filled bulb's symmetry in both axes.
            if colour_hits < 3 and (not round_dot or (alignment != 0 and not round_collinear_dot)):
                continue
            arms.append({'direction': [dx, dy], 'pixels_img': [list(p) for p in trail],
                         'non_red_colour_hits': colour_hits})
            break
    return arms


def _first_node_contacts(red, component, box, axis_y, other_boxes, nodes,
                         ground_stems, peer_types, other_colours):
    """The first node on a continuous red lead owns the terminal coordinate.

    No green/mixed pixel or white gap is traversed: an earlier colour contact
    has priority over a later junction. Ground stems start at the longest bar,
    allowing a real node inside a loose detector box to be reached correctly.
    """
    raw = _box(component.get('bbox_raw_img')) or box
    typ = component['type']
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    own_stems = ground_stems.get(component.get('id'), [])
    candidates = []
    for node in nodes:
        point = node['point']
        if not (box[0] - 48 <= point[0] <= box[2] + 48 and box[1] - 48 <= point[1] <= box[3] + 48):
            continue
        vertical = axis_y if typ in AXIAL_TYPES else abs(point[1] - centre[1]) > abs(point[0] - centre[0])
        axis, cross_axis = (1, 0) if vertical else (0, 1)
        axial = point[axis] - centre[axis]
        side = 1 if axial > 0 else -1
        direction = (0, side) if vertical else (side, 0)
        lateral = abs(point[cross_axis] - centre[cross_axis])
        cross = box[cross_axis + 2] - box[cross_axis]
        centred_limit = max(5, cross * .3)
        limit = max(centred_limit, cross * .75) if typ in AXIAL_TYPES else centred_limit
        if lateral > limit:
            continue
        if own_stems:
            stems = [stem for stem in own_stems if stem['axis_y'] == vertical
                     and stem['outward_side'] == side
                     and (point[axis] - stem['point'][axis]) * side >= 2
                     and abs(point[cross_axis] - stem['point'][cross_axis]) <= 2.5]
            origins = [stem['point'] for stem in stems]
        else:
            edge = raw[axis + 2] if side > 0 else raw[axis]
            if (point[axis] - edge) * side < 2:
                continue
            origins = []
            for shift in (0, -1, 1, -2, 2):
                origin = list(point)
                origin[axis] = edge + shift
                origins.append(origin)
        path = next((p for origin in origins
                     if (p := _continuous_red_segment(red, origin, point, vertical)) and len(p) >= 3), None)
        if path is None and typ == 'l':
            path = _peer_red_body_path(red, point, (-direction[0], -direction[1]), raw, typ)
        if path is None:
            continue
        arms = _node_wire_arms(red, other_colours, node, (-direction[0], -direction[1]))
        if not arms:
            continue
        blocker = _intervening_body(point, direction, raw, other_boxes)
        if blocker is not None:
            # A ground detector can include its exposed stem and the node.
            # Reaching that stem is valid; crossing its actual bars is not.
            peer_stems = ground_stems.get(blocker, []) if peer_types.get(blocker) == 'gnd' else []
            if not any(stem['axis_y'] == vertical
                       and (stem['point'][axis] - point[axis]) * side >= 2
                       and abs(stem['point'][cross_axis] - point[cross_axis]) <= 2.5
                       for stem in peer_stems):
                continue
        candidates.append({'point': point, 'side': side, 'source': 'first_red_junction_center',
                           'off_center': lateral > centred_limit, 'blocked_by': None,
                           'wire': {'kind': 'continuous_red_lead_to_first_node',
                                    'coordinate_img': list(point), 'red_junction_img': list(point),
                                    'node_bounds_img': list(node['bounds']), 'red_lead_pixels_img': path,
                                    'junction_wire_arms': arms}})
    first = []
    for side in (-1, 1):
        options = [candidate for candidate in candidates if candidate['side'] == side]
        if options:
            distance = min(abs(candidate['point'][1 if axis_y else 0] - centre[1 if axis_y else 0])
                           if typ in AXIAL_TYPES else math.dist(candidate['point'], centre) for candidate in options)
            first.extend(candidate for candidate in options
                         if abs((abs(candidate['point'][1 if axis_y else 0] - centre[1 if axis_y else 0])
                                 if typ in AXIAL_TYPES else math.dist(candidate['point'], centre)) - distance) < .1)
    return first


def _enclosing_body_labels(labels, stats, raw, left, top):
    """Anchor an inset detector box to an enclosing observed red outline."""
    cx, cy = (raw[0] + raw[2]) / 2 - left, (raw[1] + raw[3]) / 2 - top
    rw, rh = raw[2] - raw[0], raw[3] - raw[1]
    found = set()
    for label, stat in enumerate(stats[1:], 1):
        x, y, w, h, area = map(int, stat)
        if not (x < cx < x + w and y < cy < y + h):
            continue
        # A distant/oversized enclosing component is not this small symbol.
        if w > 6 * max(8, rw) or h > 6 * max(8, rh):
            continue
        mask = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if any(cv2.pointPolygonTest(contour, (cx - x, cy - y), False) > 0
               for contour in contours):
            found.add(label)
    return found


def _ground_body_labels(red, labels, raw, left, top, with_axes=False):
    """Find the stem-bearing bar from three aligned, tapering ground bars."""
    h, w = red.shape
    found = {}
    bar_axes = {}
    stem_anchors = []
    for vertical in (False, True):
        normal_center = (raw[1] + raw[3]) / 2 - top if vertical else (raw[0] + raw[2]) / 2 - left
        tangent_center = (raw[0] + raw[2]) / 2 - left if vertical else (raw[1] + raw[3]) / 2 - top
        normal_size = raw[3] - raw[1] if vertical else raw[2] - raw[0]
        tangent_size = raw[2] - raw[0] if vertical else raw[3] - raw[1]
        normal_lo = max(0, int(normal_center - normal_size))
        normal_hi = min(h if vertical else w, math.ceil(normal_center + normal_size) + 1)
        tangent_lo = max(0, int(tangent_center - 1.5 * tangent_size))
        tangent_hi = min(w if vertical else h, math.ceil(tangent_center + 1.5 * tangent_size) + 1)
        rows = []
        for coord in range(normal_lo, normal_hi):
            pixels = red[coord, tangent_lo:tangent_hi] if vertical else red[tangent_lo:tangent_hi, coord]
            runs = [(a + tangent_lo, b + tangent_lo) for a, b in _runs(pixels)
                    if b - a >= max(3, .18 * tangent_size)
                    and (abs((a + b) / 2 + tangent_lo - tangent_center) <= max(2, .2 * tangent_size)
                         or (b - a >= 1.6 * tangent_size and a + tangent_lo <= tangent_center <= b + tangent_lo))]
            if len(runs) != 1:
                continue
            a, b = runs[0]
            # Antialiasing can shorten the first row of one thick bar by more
            # than two pixels. Contiguous, centred rows still form one bar.
            if (rows and coord == rows[-1]['last'] + 1
                    and abs((a + b - 1) / 2 - rows[-1]['tangent']) <= 2
                    and min(b - a, rows[-1]['width']) >= .6 * max(b - a, rows[-1]['width'])):
                rows[-1]['last'] = coord
                if b - a > rows[-1]['width']:
                    rows[-1]['width'] = b - a
                    rows[-1]['tangent'] = (a + b - 1) / 2
            else:
                rows.append(dict(first=coord, last=coord, width=b - a, tangent=(a + b - 1) / 2))
        bars = [row for row in rows if row['last'] - row['first'] + 1 <= max(4, .2 * normal_size)]
        for a, b, c in zip(bars, bars[1:], bars[2:]):
            gaps = (b['first'] - a['last'], c['first'] - b['last'])
            if not all(2 <= gap <= max(8, .65 * normal_size) for gap in gaps):
                continue
            if a['width'] >= 1.2 * b['width'] and b['width'] >= 1.2 * c['width']:
                longest = a
            elif c['width'] >= 1.2 * b['width'] and b['width'] >= 1.2 * a['width']:
                longest = c
            else:
                continue
            if longest['width'] < .8 * tangent_size:
                continue
            normal = _round((longest['first'] + longest['last']) / 2)
            reference_tangent = longest['tangent']
            if abs(reference_tangent - tangent_center) > max(2, .2 * tangent_size):
                shorter = [row for row in (a, b, c) if row is not longest]
                if abs(shorter[0]['tangent'] - shorter[1]['tangent']) > 2:
                    continue
                reference_tangent = sum(row['tangent'] for row in shorter) / 2
            tangent = _round(reference_tangent)
            label = int(labels[normal, tangent] if vertical else labels[tangent, normal])
            if label:
                tangent_lo_bar = longest['tangent'] - (longest['width'] - 1) / 2
                tangent_hi_bar = longest['tangent'] + (longest['width'] - 1) / 2
                bar_bounds = (tangent_lo_bar + left, longest['first'] + top,
                              tangent_hi_bar + left, longest['last'] + top) if vertical else (
                              longest['first'] + left, tangent_lo_bar + top,
                              longest['last'] + left, tangent_hi_bar + top)
                stem_anchors.append({'point': (float(tangent + left), float(normal + top)) if vertical
                                     else (float(normal + left), float(tangent + top)),
                                     'axis_y': vertical, 'outward_side': -1 if longest is a else 1,
                                     'bar_bounds': bar_bounds})
                found.setdefault(label, set()).add(vertical)
                for bar in (a, b, c):
                    normal = _round((bar['first'] + bar['last']) / 2)
                    tangent = _round(bar['tangent'])
                    bar_label = int(labels[normal, tangent] if vertical else labels[tangent, normal])
                    if bar_label:
                        bar_axes.setdefault(bar_label, set()).add(vertical)
    if with_axes:
        return {'body_labels': set(found),
                'bar_axes': {label: tuple(sorted(axes)) for label, axes in bar_axes.items()},
                'stem_anchors': stem_anchors}
    return set(found)


def _compact_junction(body_red, contact_mask, stat):
    """A filled compact bulb can have a larger contact than a thin lead."""
    x, y, width, height, _ = map(int, stat)
    if max(width, height) > 12:
        return None
    h, w = body_red.shape
    left, top = max(0, x - 6), max(0, y - 6)
    right, bottom = min(w, x + width + 6), min(h, y + height + 6)
    local = body_red[top:bottom, left:right].astype(np.uint8)
    filled = cv2.morphologyEx(local, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, labels, stats, centres = cv2.connectedComponentsWithStats(filled, 8)
    contact = contact_mask[top:bottom, left:right]
    for label, blob in enumerate(stats[1:], 1):
        _, _, bw, bh, area = map(int, blob)
        if (5 <= min(bw, bh) <= max(bw, bh) <= 12 and min(bw, bh) >= .7 * max(bw, bh)
                and area >= .55 * bw * bh and np.any(contact & (labels == label))):
            return (float(centres[label][0] + left), float(centres[label][1] + top))
    return None


def _intervening_body(point, direction, raw, other_boxes):
    """A terminal cannot be reached by looking through another detected body."""
    axis = 0 if direction[0] else 1
    cross_axis = 1 - axis
    positive = direction[axis] > 0
    start = raw[axis + 2] if positive else raw[axis]
    end = float(point[axis])
    if (end - start) * direction[axis] <= 0:
        return None
    lo, hi = sorted((start, end))
    tangent = float(point[cross_axis])
    for cid, peer in other_boxes:
        if (min(raw[2], peer[2]) > max(raw[0], peer[0])
                and min(raw[3], peer[3]) > max(raw[1], peer[1])):
            # Two overlapping detector proposals can refer to the same body.
            continue
        if not peer[cross_axis] + 1 <= tangent <= peer[cross_axis + 2] - 1:
            continue
        # Ignore a mere touch at the detector edge: the path must pass into
        # the interior before it reaches the proposed red-green boundary.
        overlap = min(hi, peer[axis + 2] - 1) - max(lo, peer[axis] + 1)
        if overlap > 1:
            return cid
    return None


def _colour_contacts(red, green, component, box, axis_y, blends=None, other_boxes=(),
                     junctions=(), ground_stems=None, peer_types=None, other_colours=None, mix_context=None):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = red.shape
    left, top = max(0, min(w, int(x0 - 48))), max(0, min(h, int(y0 - 48)))
    right = max(0, min(w, int(math.ceil(x1 + 48))))
    bottom = max(0, min(h, int(math.ceil(y1 + 48))))
    r, g = red[top:bottom, left:right], green[top:bottom, left:right]
    local_blends = blends[top:bottom, left:right] if blends is not None else None
    if r.size == 0 or not r.any():
        return []
    _, bodies, body_stats, _ = cv2.connectedComponentsWithStats(r.astype(np.uint8), 8)
    raw = _box(component.get("bbox_raw_img")) or box
    sx0 = max(0, min(r.shape[1], int(raw[0] - left - 2)))
    sy0 = max(0, min(r.shape[0], int(raw[1] - top - 2)))
    sx1 = max(0, min(r.shape[1], int(raw[2] - left + 3)))
    sy1 = max(0, min(r.shape[0], int(raw[3] - top + 3)))
    seeds = set(np.unique(bodies[sy0:sy1, sx0:sx1])) - {0}
    ground_axes = {}
    if component.get("type") in AXIAL_TYPES:
        seeds |= _enclosing_body_labels(bodies, body_stats, raw, left, top)
    elif component.get("type") == 'gnd':
        ground = _ground_body_labels(r, bodies, raw, left, top, with_axes=True)
        ground_axes = ground['bar_axes']
        seeds |= ground['body_labels']
    if not seeds:
        return []
    body_red = r & np.isin(bodies, list(seeds))
    added_body = np.zeros_like(r, dtype=bool)
    if blends is not None:
        bridges = _blend_bridges(r, blends[top:bottom, left:right])
        if bridges.any():
            _, linked = cv2.connectedComponents((r | bridges).astype(np.uint8), 8)
            anchored = set(np.unique(linked[body_red])) - {0}
            linked_red = r & np.isin(linked, list(anchored))
            added_body = linked_red & ~body_red
            body_red = linked_red
    contact = body_red & cv2.dilate(
        # Up to two neutral antialias pixels can lie between red and green.
        g.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    _, contact_labels, stats, centres = cv2.connectedComponentsWithStats(contact.astype(np.uint8), 8)
    normals = {axis for axes in ground_axes.values() for axis in axes}
    shared_contacts = _single_pixel_contacts(red, green, blends, component, box, axis_y,
                                             other_boxes, normals, ground_stems)
    if mix_context is not None:
        span_contacts = _mixed_span_contacts(red, green, blends, component, box, axis_y, other_boxes,
                                             normals, ground_stems or {}, peer_types or {}, mix_context)
        for span in span_contacts:
            bounds = span['wire']['marker_bounds_img']
            shared_contacts = [c for c in shared_contacts if not (
                bounds[0] <= c['point'][0] <= bounds[2] and bounds[1] <= c['point'][1] <= bounds[3])]
        shared_contacts += span_contacts
    node_contacts = _first_node_contacts(red, component, box, axis_y, other_boxes, junctions,
                                        ground_stems or {}, peer_types or {},
                                        other_colours if other_colours is not None else
                                        green | (blends if blends is not None else np.zeros_like(red)))
    if component.get('type') == 'led':
        local_raw = (raw[0] - left, raw[1] - top, raw[2] - left, raw[3] - top)
        for candidate in node_contacts:
            if not candidate['off_center']:
                continue
            point = (candidate['point'][0] - left, candidate['point'][1] - top)
            owned = int(bodies[_round(point[1]), _round(point[0])])
            direction = (0, candidate['side']) if axis_y else (candidate['side'], 0)
            opposed = _opposed_red_lead(body_red & (bodies == owned), point, direction, local_raw)
            if opposed is not None:
                candidate['wire']['opposed_red_lead_img'] = [[x + left, y + top] for x, y in opposed]
    candidates = []
    single = component.get("type") in SINGLE_TYPES
    for label, (stat, centre) in enumerate(zip(stats[1:], centres[1:]), 1):
        area = int(stat[-1])
        if area < 1 or area > 64:
            continue
        junction = _compact_junction(body_red, contact_labels == label, stat) if area > 24 else None
        if area > 24 and junction is None:
            continue
        p = (float(centre[0] + left), float(centre[1] + top))
        ox, oy = p[0] - cx, p[1] - cy
        vertical = abs(oy) > abs(ox) if single else axis_y
        initial_axis = vertical
        if ground_axes:
            owned_labels = np.unique(bodies[contact_labels == label])
            normals = {axis for owned in owned_labels for axis in ground_axes.get(int(owned), ())}
            if len(normals) > 1:
                continue
            if normals:
                # A ground pin leaves normal to its bars, not along the bar
                # towards an unrelated green wire passing its outer end.
                vertical = next(iter(normals))
        axial = oy if vertical else ox
        lateral = abs(ox) if vertical else abs(oy)
        cross = (x1 - x0) if vertical else (y1 - y0)
        body_half = ((y1 - y0) if vertical else (x1 - x0)) / 2
        centred_limit = max(5, cross * 0.3)
        search_limit = centred_limit if single else max(centred_limit, cross * 0.75)
        if abs(axial) < max(4, body_half * 0.65) or lateral > search_limit:
            continue
        direction = (0, 1 if axial > 0 else -1) if vertical else (1 if axial > 0 else -1, 0)
        if junction is not None:
            # A side branch biases the contact centroid away from the lead.
            # The filled bulb supplies its observed transverse centre.
            p = (junction[0] + left, p[1]) if vertical else (p[0], junction[1] + top)
        inward = (-direction[0], -direction[1])
        curved_lead = None
        if _ray(red, p, inward, 6, 1) < 0.65:
            if component.get('type') == 'l' and junction is None:
                local_raw = (raw[0] - left, raw[1] - top, raw[2] - left, raw[3] - top)
                curved_lead = _curved_lead_path(body_red, (p[0] - left, p[1] - top),
                                              inward, local_raw, cross)
            if curved_lead is None:
                continue
        p = _transition_point(red, green, p, direction)
        if p is None:
            continue
        blocker = _intervening_body(p, direction, raw, other_boxes)
        if area > 24 and _ray(red, p, (-direction[0], -direction[1]), 12, 6) < .75:
            continue
        wire = _wire_path(green, p, direction)
        if wire is None:
            local_raw = (raw[0] - left, raw[1] - top, raw[2] - left, raw[3] - top)
            local_peers = [(cid, (b[0] - left, b[1] - top, b[2] - left, b[3] - top))
                           for cid, b in other_boxes]
            wire = _short_wire_to_peer(r, g, bodies, (p[0] - left, p[1] - top), direction,
                                       local_raw, local_peers,
                                       blends[top:bottom, left:right] if blends is not None else None)
            if wire is not None:
                for key in ('green_pixels_img', 'facing_red_pixels_img', 'blend_pixels_img'):
                    if key not in wire:
                        continue
                    wire[key] = [[x + left, y + top] for x, y in wire[key]]
            else:
                wire = _short_wire_to_junction(r, g, (p[0] - left, p[1] - top), direction)
                if wire is not None:
                    wire['green_pixels_img'] = [[x + left, y + top] for x, y in wire['green_pixels_img']]
                    wire['wire_junction_img'] = [wire['wire_junction_img'][0] + left,
                                                wire['wire_junction_img'][1] + top]
                    for arm in wire['junction_wire_arms']:
                        arm['point_img'] = [arm['point_img'][0] + left, arm['point_img'][1] + top]
        if wire is None:
            wire = _short_observed_wire(red, green, blends, p, direction)
            if wire is None:
                continue
        if curved_lead is not None:
            wire['red_lead'] = curved_lead
        if ground_axes and initial_axis != vertical:
            wire['ground_body_axis'] = {'kind': 'normal_to_tapering_red_bars', 'axis_y': vertical}
        if component.get('type') == 'led' and lateral > centred_limit:
            # LED arrows can move the detector centre off its actual stems.
            # Independent ink seeded in the same box is not axis evidence:
            # the opposite stem must share this contact's red component.
            contact_bodies = np.unique(bodies[contact_labels == label])
            same_body = body_red & np.isin(bodies, contact_bodies)
            local_raw = (raw[0] - left, raw[1] - top, raw[2] - left, raw[3] - top)
            opposed = _opposed_red_lead(same_body, (p[0] - left, p[1] - top), direction, local_raw)
            if opposed is not None:
                wire['opposed_red_lead_img'] = [[x + left, y + top] for x, y in opposed]
        if junction is not None:
            wire['red_junction_img'] = [junction[0] + left, junction[1] + top]
        if np.any(added_body & (contact_labels == label)):
            wire['red_body_link'] = {'kind': 'short_red_green_blend', 'max_gap_pixels': 3}
        candidates.append({"point": p, "side": 1 if axial > 0 else -1,
                           "source": "body_attached_red_green_transition",
                           "off_center": lateral > centred_limit,
                           "blocked_by": blocker,
                           "wire": wire})
    if shared_contacts:
        shared_sides = {candidate['side'] for candidate in shared_contacts}
        candidates = [candidate for candidate in candidates if candidate['side'] not in shared_sides]
    candidates += shared_contacts
    if node_contacts:
        def boundary_precedes_node(contact, node):
            vertical = axis_y if component['type'] in AXIAL_TYPES else abs(node['point'][1] - cy) > abs(node['point'][0] - cx)
            axis, cross_axis = (1, 0) if vertical else (0, 1)
            direction = node['side']
            bounds = node['wire']['node_bounds_img']
            if (bounds[axis] <= contact['point'][axis] <= bounds[axis + 2]
                    and contact['source'] == 'body_attached_red_green_transition'):
                # Green at the dot's side is a branch, not an earlier cut
                # across this lead. A real centred colour mark still wins.
                centred = list(contact['point'])
                centred[cross_axis] = node['point'][cross_axis]
                normal = (0, direction) if vertical else (direction, 0)
                if _transition_point(red, green, centred, normal) is None:
                    return False
            return (contact['blocked_by'] is None and contact['side'] == node['side']
                    and abs(contact['point'][cross_axis] - node['point'][cross_axis]) <= 2.5
                    and (node['point'][axis] - contact['point'][axis]) * direction > .15)
        reached_nodes = []
        for node in node_contacts:
            earlier = [contact for contact in candidates if boundary_precedes_node(contact, node)]
            if not earlier:
                reached_nodes.append(node)
                continue
            vertical = axis_y if component['type'] in AXIAL_TYPES else abs(node['point'][1] - cy) > abs(node['point'][0] - cx)
            axis, cross_axis = (1, 0) if vertical else (0, 1)
            first = min(earlier, key=lambda contact: contact['point'][axis] * node['side'])
            # Thick red strokes can bypass a colour pixel laterally. The
            # junction's far boundary is then in the same red component,
            # but still belongs beyond this pin's first observed interface.
            candidates = [contact for contact in candidates
                          if not (contact['side'] == node['side']
                                  and abs(contact['point'][cross_axis] - first['point'][cross_axis]) <= 2.5
                                  and (contact['point'][axis] - first['point'][axis]) * node['side'] > .15)]
        node_contacts = reached_nodes
        node_sides = {candidate['side'] for candidate in node_contacts}
        candidates = [candidate for candidate in candidates if candidate['side'] not in node_sides]
    candidates += node_contacts
    unique = []
    for candidate in candidates:
        same = next((r for r in unique if r['side'] == candidate['side']
                     and math.dist(r['point'], candidate['point']) < .15), None)
        if same is None:
            unique.append(candidate)
        else:
            same['off_center'] = same['off_center'] and candidate['off_center']
    if component['type'] in {'v', 'vcc'}:
        owned = []
        for candidate in unique:
            point = candidate['point']
            vertical = abs(point[1] - cy) > abs(point[0] - cx)
            axis = 1 if vertical else 0
            direction = 1 if point[axis] > (cy if vertical else cx) else -1
            origin = list(point)
            origin[axis] = raw[axis + 2] if direction > 0 else raw[axis]
            target = list(point)
            if candidate['source'] == 'body_attached_red_green_transition':
                target[axis] -= direction * .5
            path = candidate['wire'].get('own_red_pixels_img') or _continuous_red_segment(red, origin, target, vertical)
            crosses_ground_bar = False
            for peer_id, stems in (ground_stems or {}).items():
                if peer_id == component.get('id'):
                    continue
                for stem in stems:
                    bounds = stem.get('bar_bounds')
                    # The final red cell can be the interface itself. Only
                    # a bar traversed before that cell is an intervening body.
                    if bounds and any(bounds[0] - .5 <= x <= bounds[2] + .5
                                      and bounds[1] - .5 <= y <= bounds[3] + .5 for x, y in (path or [])[:-1]):
                        crosses_ground_bar = True
            if not crosses_ground_bar:
                owned.append(candidate)
        unique = owned
    return unique


def refine_pins(components, pins, bgr):
    """Return new candidates plus per-component provenance; never mutate inputs."""
    result = {cid: list(points) for cid, points in pins.items()}
    report = {}
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return result, report
    red, green = _colour_masks(bgr)
    blends = _blended_red_green_pixels(bgr) & ~red & ~green
    blue_channel, green_channel, red_channel = np.moveaxis(bgr.astype(np.int16), -1, 0)
    tinted = red & (green_channel - blue_channel >= 18)
    mix_context = (tinted, red & ~tinted, green_channel - blue_channel)
    blue = ((blue_channel - red_channel >= 18) & (blue_channel - green_channel >= 18)
            & (blue_channel >= 45))
    other_colours = green | blends | blue
    if not red.any():
        from .passive_terminal_geometry import extend_terminals
        return extend_terminals(components, pins, result, report, bgr)
    junctions = _filled_red_junctions(red)
    _, red_labels = cv2.connectedComponents(red.astype(np.uint8), 8)
    ground_stems = {}
    peer_types = {component.get('id'): component.get('type') for component in components}
    for component in components:
        if component.get('type') == 'gnd':
            raw = _box(component.get('bbox_raw_img')) or _box(component.get('bbox_img'))
            if raw is not None:
                ground_stems[component.get('id')] = _ground_body_labels(
                    red, red_labels, raw, 0, 0, with_axes=True)['stem_anchors']
    body_boxes = [(c.get('id'), _box(c.get('bbox_raw_img')) or _box(c.get('bbox_img')))
                  for c in components]
    for comp in components:
        typ, cid = comp.get("type"), comp.get("id")
        if typ not in AXIAL_TYPES | SINGLE_TYPES or cid not in result:
            continue
        box = _box(comp.get("bbox_img"))
        raw = _box(comp.get("bbox_raw_img")) or box
        if box is None or raw is None:
            continue
        h, w = bgr.shape[:2]
        if box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
            continue
        rw, rh = raw[2] - raw[0], raw[3] - raw[1]
        uncertain_axis = typ in AXIAL_TYPES and max(rw, rh) < 1.15 * min(rw, rh)
        axis_y = rh > rw
        axes = (axis_y, not axis_y) if typ in AXIAL_TYPES else (axis_y,)
        alternatives = []
        peers = [(other, bounds) for other, bounds in body_boxes if other != cid and bounds is not None]
        for vertical in axes:
            chain_contacts = _colour_contacts(red, green, comp, box, vertical, blends, peers,
                                             junctions, ground_stems, peer_types, other_colours, mix_context)
            contacts = [r for r in chain_contacts if r['blocked_by'] is None]
            if typ in AXIAL_TYPES:
                if vertical != axis_y and not uncertain_axis and not any(
                        r['source'] in {'first_red_junction_center', 'shared_single_green_pixel'} for r in contacts):
                    continue
                centred_sides = {r['side'] for r in contacts if not r['off_center']}
                contacts = [r for r in contacts if not r['off_center'] or r['side'] not in centred_sides]
                if not contacts or any(sum(r["side"] == side for r in contacts) > 1 for side in (-1, 1)):
                    continue
                # Box shape alone cannot decide a square or offset symbol.
                # Two opposed, independently observed contacts on one line can.
                if uncertain_axis or vertical != axis_y or any(r["off_center"] for r in contacts):
                    axis_contacts = contacts
                    if len(contacts) != 2:
                        if len(contacts) == 1 and contacts[0]['source'] == 'shared_single_green_pixel':
                            contacts[0]['wire']['axis_support'] = 'continuous_opposed_red_pins_at_shared_pixel'
                            alternatives.append((vertical, contacts))
                            continue
                        if (not uncertain_axis and len(contacts) == 1
                                and contacts[0]['wire'].get('opposed_red_lead_img')):
                            contacts[0]['wire']['axis_support'] = 'opposed_continuous_red_lead'
                            alternatives.append((vertical, contacts))
                            continue
                        # The red chain through a neighbour can corroborate a
                        # reliable detector axis without publishing its tip as
                        # this body's terminal. This preserves the clean end
                        # of an offset LED whose other end is red-red connected.
                        if (uncertain_axis or len(contacts) != 1 or len(chain_contacts) != 2
                                or {r['side'] for r in chain_contacts} != {-1, 1}):
                            continue
                        axis_contacts = chain_contacts
                    cross_axis = 0 if vertical else 1
                    if abs(axis_contacts[0]["point"][cross_axis] - axis_contacts[1]["point"][cross_axis]) > 2.5:
                        continue
                    if len(contacts) == 1:
                        contacts[0]['wire']['axis_support'] = 'aligned_red_chain_with_intervening_body'
            elif len(contacts) > 1:
                established = [r for r in contacts if r['wire']['kind'] != 'short_observed_green_wire']
                if len(established) == 1:
                    contacts = established
            if contacts:
                alternatives.append((vertical, contacts))
        if len(alternatives) != 1:
            continue
        axis_y, measured = alternatives[0]
        if typ in SINGLE_TYPES:
            if len(measured) != 1:
                continue
        else:
            if not measured or any(sum(r["side"] == side for r in measured) > 1 for side in (-1, 1)):
                continue
            measured.sort(key=lambda r: r["side"])
        new = [r["point"] for r in measured]
        # Always record confirmed coordinates, even when the old points were
        # already correct: stage2 must not stamp them back to centre +/-20.
        report[cid] = {"type": typ, "before": result[cid], "after": new,
                       "sources": [r["source"] for r in measured],
                       "axis_y": axis_y,
                       "connection_evidence": [r["wire"] for r in measured],
                       "coordinate_rule": ("image_color_terminal_rules" if any(r['source'] != 'body_attached_red_green_transition' for r in measured)
                                           else "red_green_pixel_boundary")}
        result[cid] = new
    from .passive_terminal_geometry import extend_terminals
    return extend_terminals(components, pins, result, report, bgr)
