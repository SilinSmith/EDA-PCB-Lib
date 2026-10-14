"""Unmarked straight-pin contacts from observed corners and local symbol lengths.

Lengths are learned from measured terminals in the current image. Detector
boxes only identify a facing symbol; every connection still needs real ink.
"""
from __future__ import annotations

import math

import numpy as np

DERIVED = frozenset({'symmetric_opposed_terminal', 'peer_shared_terminal',
                     'same_symbol_terminal_length'})


def _axis(geo):
    return 1 if geo['axis_y'] else 0


def _edge(geo, side):
    from .passive_terminal_geometry import _body_edge
    return _body_edge(geo, side)


def _point(geo, normal, tangent=None):
    tangent = geo['tangent'] if tangent is None else tangent
    return (float(tangent), float(normal)) if geo['axis_y'] else (float(normal), float(tangent))


def _stroke(kinds, geo, side, legacy, target, allow_weak=None):
    from .passive_terminal_geometry import _trace
    samples = _trace(kinds, geo, side)
    if len(samples) < 4:
        return None
    base = samples[0]['pigment']
    if base == 0 or any(s['pigment'] != base for s in samples[:4]):
        return None
    mask = kinds == base
    if allow_weak is not None:
        mask = mask | allow_weak
    origin = samples[0]['point']
    path = legacy._continuous_red_segment(mask, origin, target, geo['axis_y'])
    return (base, path) if path else None


def _facing_peer(kinds, geo, side, point, cid, components, raw_boxes, legacy, weak=None, *, allow_ic=False):
    """A single-port peer needs an uninterrupted stem ending in its real body."""
    from .passive_terminal_geometry import _line_runs
    axis, cross = _axis(geo), 1 - _axis(geo)
    candidates = []
    own = _stroke(kinds, geo, side, legacy, point, weak)
    if own is None:
        return None
    base, _ = own
    for peer in components:
        pid, typ = peer.get('id'), peer.get('type')
        if pid == cid or typ not in ({'gnd', 'v', 'vcc', 'box'} if allow_ic else {'gnd', 'v', 'vcc'}) or pid not in raw_boxes:
            continue
        raw = raw_boxes[pid]
        facing = raw[axis] if side > 0 else raw[axis + 2]
        if not (2 <= (facing - point[axis]) * side <= 48
                and raw[cross] - 2 <= point[cross] <= raw[cross + 2] + 2):
            continue
        limit = int(math.ceil(abs(facing - point[axis]))) + 7
        # Half-pixel contacts must still visit every raster column/row.
        # Rounding successive .5 values would skip all odd positions.
        start = math.ceil(point[axis]) if side > 0 else math.floor(point[axis])
        for step in range(2, limit + 1):
            normal = start + side * step
            radius = max(24, raw[cross + 2] - raw[cross]) if typ == 'box' else 24
            runs = [(a, b) for a, b in _line_runs(kinds == base, normal, point[cross], geo['axis_y'], radius)
                    if a <= point[cross] <= b and b - a >= 5
                    and (b - a >= .6 * (raw[cross + 2] - raw[cross]) if typ == 'box'
                         else abs((a + b - 1) / 2 - point[cross]) <= 2)]
            if len(runs) != 1:
                continue
            if not raw[axis] - 6 <= normal <= raw[axis + 2] + 6:
                continue
            if typ == 'box' and abs(normal - facing) > 3:
                continue
            target = _point(geo, normal, point[cross])
            connection = _stroke(kinds, geo, side, legacy, target, weak)
            if connection is None:
                break
            blockers = [(other, box) for other, box in raw_boxes.items() if other not in {cid, pid}]
            if legacy._intervening_body(point, (0, side) if axis else (side, 0), raw_boxes[cid], blockers) is not None:
                break
            root = dict(axis_y=geo['axis_y'], low=normal - .5, high=normal + .5,
                        tangent=point[cross], only_side=-side, kind='observed_single_port_body',
                        body_bands_img=[dict(first=normal, last=normal, width=runs[0][1] - runs[0][0],
                                             tangent=point[cross])])
            candidates.append((pid, connection[1], root))
            break
    return candidates[0] if len(candidates) == 1 else None


def _corner(kinds, geo, side, cid, raw_boxes, measured, legacy):
    """An L formed by two same-red straight pins, ending at a detected peer."""
    from .passive_terminal_geometry import _trace, _line_runs
    samples = _trace(kinds, geo, side)
    if len(samples) < 6 or samples[0]['pigment'] != 1:
        return None
    first_change = next((i for i, s in enumerate(samples) if s['pigment'] != 1), len(samples))
    samples = samples[:first_change]
    if len(samples) < 6:
        return None
    axis, cross = _axis(geo), 1 - _axis(geo)
    tangent = samples[0]['point'][cross]
    proposals = []
    wide = next((s for s in samples if s['width'] >= 5), samples[-1])
    for normal in range(wide['pos'] - 3, wide['pos'] + 4):
        distance = (normal - _edge(geo, side)) * side
        if not 5 <= distance <= 48:
            continue
        # The corner's transverse stroke is the actual second straight pin.
        runs = [(a, b) for a, b in _line_runs(kinds == 1, normal, tangent, geo['axis_y'], 24)
                if a - 1 <= tangent <= b and b - a >= 6]
        if len(runs) != 1:
            continue
        low, high = runs[0]
        directions = [d for d in (-1, 1) if (tangent - low if d < 0 else high - 1 - tangent) >= 5]
        if len(directions) != 1:
            continue
        turn = directions[0]
        point = _point(geo, normal, tangent)
        own = _stroke(kinds, geo, side, legacy, point)
        if own is None:
            continue
        # A line continuing forward is a T, not two pins meeting at an L.
        forward = [_point(geo, normal + side * step, tangent) for step in (2, 3, 4)]
        if any(0 <= int(x) < kinds.shape[1] and 0 <= int(y) < kinds.shape[0]
               and kinds[int(y), int(x)] == 1 for x, y in forward):
            continue
        peers = []
        for peer, raw in raw_boxes.items():
            if peer == cid or not raw[axis] - 2 <= normal <= raw[axis + 2] + 2:
                continue
            facing = raw[cross] if turn > 0 else raw[cross + 2]
            if not 4 <= (facing - tangent) * turn <= 48:
                continue
            target = list(point)
            target[cross] = math.ceil(facing) if turn > 0 else math.floor(facing)
            path = legacy._continuous_red_segment(kinds == 1, point, target, not geo['axis_y'])
            if path:
                peers.append((peer, path))
        if len(peers) == 1:
            proposals.append(dict(point=point, side=side, source='direct_pin_corner',
                                  wire=dict(kind='two_straight_pins_at_corner', coordinate_img=list(point),
                                            own_lead_pixels_img=own[1], peer_id=peers[0][0],
                                            perpendicular_peer_pixels_img=peers[0][1],
                                            body_geometry=geo)))
    if not proposals:
        return None
    # Adjacent thick raster columns represent one elbow; use their centre.
    if max(p['point'][axis] for p in proposals) - min(p['point'][axis] for p in proposals) > 2:
        return None
    chosen = proposals[len(proposals) // 2]
    corner_candidates = [list(p['point']) for p in proposals]
    value = float(np.mean([p['point'][axis] for p in proposals]))
    chosen['point'] = _point(geo, value, tangent)
    chosen['wire']['coordinate_img'] = list(chosen['point'])
    chosen['wire']['corner_candidates_img'] = corner_candidates
    return chosen


def _shape(kinds, geo):
    """Normalise observed body dimensions, not detector-box dimensions."""
    from .passive_terminal_geometry import _line_runs
    if 'only_side' not in geo:
        widths = [band['width'] for band in geo['body_bands_img'] if 'width' in band]
        return (geo['high'] - geo['low'], max(widths))
    edge = int(round(_edge(geo, geo['only_side']) - .5 * geo['only_side']))
    profile = []
    for step in range(12):
        pos = edge - geo['only_side'] * step
        bars = [(a, b) for a, b in _line_runs(kinds != 0, pos, geo['tangent'], geo['axis_y'], 24)
                if a - 1 <= geo['tangent'] <= b and b - a <= 40]
        profile.append(max((b - a for a, b in bars), default=0))
    while profile and profile[-1] == 0:
        profile.pop()
    return (len(profile), max(profile, default=0))


def _library(components, kinds, geometries, measured, nodes=None):
    from . import passive_pins as legacy
    from .passive_terminal_geometry import _filled_nodes, _trace
    if nodes is None:
        nodes = _filled_nodes(kinds)
    rows = []
    for comp in components:
        cid = comp.get('id')
        if cid not in geometries:
            continue
        geo = geometries[cid][0]
        axis = _axis(geo)
        shape = _shape(kinds, geo)
        for contact in measured.get(cid, []):
            if contact['source'] in DERIVED:
                continue
            side = contact['side']
            distance = (contact['point'][axis] - _edge(geo, side)) * side
            if not 4 <= distance <= 48 or abs(contact['point'][1 - axis] - geo['tangent']) > 1.5:
                continue
            # A legacy colour contact beyond a directly reached filled node
            # measures the dot's far edge as extra pin length. It cannot be
            # a reliable template, even if the old output still contains it.
            stem = _trace(kinds, geo, side, 4)
            base = stem[0]['pigment'] if stem else None
            earlier = [n for n in nodes if n['pigment'] == base
                       and 3 <= (n['point'][axis] - _edge(geo, side)) * side < distance - 2
                       and abs(n['point'][1 - axis] - contact['point'][1 - axis]) <= 1.5]
            if stem and any(
                    legacy._continuous_red_segment(kinds == base, stem[0]['point'], n['point'], geo['axis_y'])
                    and legacy._node_wire_arms(kinds == base, (kinds != 0) & (kinds != base), n,
                                              (0, -side) if axis else (-side, 0))
                    for n in earlier):
                continue
            rows.append(dict(id=cid, type=comp['type'], geometry=geo, shape=shape,
                             side=side, length=distance, source=contact['source'], point=contact['point']))
    return rows


def extend_unmarked(components, kinds, geometries, raw_boxes, measured, weak_near, legacy, *, weak_pixels=None, nodes=None):
    """Fill missing sides without moving any previously observed coordinate."""
    if weak_pixels is None:
        weak_pixels = weak_near & (kinds != 0)
    # The explicit red/green drawing convention makes a red L a pin contact.
    red_wire_style = any(
        contact['source'] == 'body_attached_red_green_transition'
        or (contact['source'] == 'lead_wire_colour_boundary'
            and contact['wire'].get('lead_pigment') == 'red'
            and contact['wire'].get('wire_pigment') == 'green')
        for contacts in measured.values() for contact in contacts)
    if red_wire_style:
        for comp in components:
            cid = comp.get('id')
            if comp.get('type') not in {'r', 'c', 'gnd', 'v', 'vcc'} or cid not in geometries:
                continue
            geo = geometries[cid][0]
            for side in (geo['only_side'],) if 'only_side' in geo else (-1, 1):
                if any(c['side'] == side for c in measured.get(cid, [])):
                    continue
                contact = _corner(kinds, geo, side, cid, raw_boxes, measured, legacy)
                if contact:
                    measured.setdefault(cid, []).append(contact)

    # A reliable opposite coordinate is sufficient even if the facing
    # ground's bars rasterise together or the peer is a power symbol.
    for comp in components:
        cid = comp.get('id')
        contacts = measured.get(cid, [])
        if comp.get('type') not in {'r', 'c'} or cid not in geometries or len(contacts) != 1:
            continue
        geo = geometries[cid][0]
        anchor = contacts[0]
        if anchor['source'] in DERIVED:
            continue
        axis, side = _axis(geo), -anchor['side']
        point = list(anchor['point'])
        point[axis] = geo['low'] + geo['high'] - point[axis]
        stroke = _stroke(kinds, geo, side, legacy, point)
        if stroke is None or any(weak_near[y, x] for x, y in stroke[1]):
            continue
        peer = _facing_peer(kinds, geo, side, point, cid, components, raw_boxes, legacy)
        if peer is None:
            continue
        pid, path, peer_geo = peer
        distance = (point[axis] - _edge(geo, side)) * side
        if not 4 <= distance <= 48:
            continue
        inferred = dict(point=tuple(point), side=side, source='symmetric_opposed_terminal',
                        wire=dict(kind='equal_opposed_lead_lengths', coordinate_img=point,
                                  anchor_point_img=list(anchor['point']), anchor_source=anchor['source'],
                                  measured_lead_length=distance, body_geometry=geo,
                                  own_lead_pixels_img=path, peer_id=pid))
        contacts.append(inferred)
        if pid not in geometries:
            geometries[pid] = [peer_geo]
        if not measured.get(pid):
            measured[pid] = [dict(point=tuple(point), side=-side, source='peer_shared_terminal',
                                 wire=dict(kind='direct_peer_symmetric_contact', coordinate_img=point,
                                           peer_id=cid, anchor_source=inferred['source'],
                                           own_lead_pixels_img=path, body_geometry=peer_geo))]

    references = _library(components, kinds, geometries, measured, nodes)
    for comp in components:
        cid, typ = comp.get('id'), comp.get('type')
        if typ not in {'r', 'c', 'gnd', 'v', 'vcc'} or cid not in geometries:
            continue
        geo = geometries[cid][0]
        shape = _shape(kinds, geo)
        axis = _axis(geo)
        for side in (geo['only_side'],) if 'only_side' in geo else (-1, 1):
            if any(c['side'] == side for c in measured.get(cid, [])):
                continue
            refs = [r for r in references if r['id'] != cid and r['type'] == typ
                    and r['geometry'].get('symbol_signature') == geo.get('symbol_signature')
                    and abs(r['shape'][0] - shape[0]) <= (1 if 'only_side' in geo else .25)
                    and abs(r['shape'][1] - shape[1]) <= 1.5]
            if not refs:
                continue
            # Corresponding ends preserve raster half-cell offsets. Other
            # ends are a fallback only when no same-end example is measured.
            same = [r for r in refs if r['side'] == side]
            refs = same or refs
            # Prefer matching nearby instances of a symbol bank before
            # combining similar glyphs from other drawing styles on a sheet.
            def separation(ref):
                rg = ref['geometry']
                return math.dist(_point(geo, (geo['low'] + geo['high']) / 2),
                                 _point(rg, (rg['low'] + rg['high']) / 2))
            nearest = min(map(separation, refs))
            if nearest < 128:
                refs = [r for r in refs if separation(r) <= nearest + max(32, 2 * shape[1])]
            lengths = [r['length'] for r in refs]
            if max(lengths) - min(lengths) > 1.5:
                continue
            distance = float(np.median(lengths))
            point = _point(geo, _edge(geo, side) + side * distance)
            stroke = _stroke(kinds, geo, side, legacy, point, weak_pixels)
            if stroke is None:
                continue
            # Neutral-only symbols stay deferred even if the sheet has
            # coloured labels elsewhere.
            if stroke[0] == 7 and not measured.get(cid):
                continue
            # The IC is only a facing body supporting this non-IC stem.
            # This branch never adds IC geometry or changes an IC coordinate.
            peer = _facing_peer(kinds, geo, side, point, cid, components, raw_boxes, legacy, weak_pixels,
                                allow_ic=True)
            if peer is None:
                # A ground/power stem can face an observed axial body.
                candidates = []
                for pid, fits in geometries.items():
                    if pid == cid or 'only_side' in fits[0]:
                        continue
                    pg = fits[0]
                    if pg['axis_y'] != geo['axis_y'] or abs(pg['tangent'] - point[1 - axis]) > 1.5:
                        continue
                    facing = pg['low'] if side > 0 else pg['high']
                    if not 3 <= (facing - point[axis]) * side <= 48:
                        continue
                    trail = _stroke(kinds, geo, side, legacy, _point(geo, facing), weak_pixels)
                    if trail:
                        candidates.append((pid, trail[1], pg))
                if len(candidates) == 1:
                    peer = candidates[0]
                elif not candidates and typ == 'gnd':
                    # A verified same-symbol length identifies the ground's
                    # own straight stem even when the wire later turns away.
                    # A facing but unconfirmed body remains an ownership
                    # conflict; it must not be bypassed by this fallback.
                    facing_boxes = [box for pid, box in raw_boxes.items() if pid != cid
                                    and 2 <= ((box[axis] if side > 0 else box[axis + 2]) - point[axis]) * side <= 48
                                    and box[1 - axis] - 2 <= point[1 - axis] <= box[3 - axis] + 2]
                    blockers = [(pid, box) for pid, box in raw_boxes.items() if pid != cid]
                    if (facing_boxes or legacy._intervening_body(point, (0, side) if axis else (side, 0),
                                                               raw_boxes[cid], blockers) is not None):
                        continue
                    peer = (None, stroke[1], None)
                else:
                    continue
            pid, path, _ = peer
            measured.setdefault(cid, []).append(dict(
                point=point, side=side, source='same_symbol_terminal_length',
                wire=dict(kind='measured_same_image_symbol_length', coordinate_img=list(point),
                          measured_lead_length=distance, body_geometry=geo, own_lead_pixels_img=path,
                          peer_id=pid, connection_validation='facing_body' if pid else 'observed_own_stem',
                          weak_colour_fallback=bool(any(weak_near[y, x] for x, y in path)),
                          reference_terminals=[dict(id=r['id'], point_img=list(r['point']), source=r['source'],
                                                    lead_length=r['length'], body_shape=list(r['shape'])) for r in refs])))
