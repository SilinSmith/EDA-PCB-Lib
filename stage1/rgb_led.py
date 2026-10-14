"""Recover one RGB package from three coloured, internally joined LED parts.

Three nearby LEDs alone are not enough. They need RGB filled bodies, opposing
outer leads, and one continuous red internal conductor with an observed fourth
external terminal. Original detections and terminals remain in a group ledger.
"""
from __future__ import annotations

import copy
import itertools
import math

import cv2
import numpy as np


def _body_pigment(kinds, comp):
    box = comp.get('bbox_raw_img') or comp.get('bbox_img')
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(kinds.shape[1], math.ceil(box[2])), min(kinds.shape[0], math.ceil(box[3]))
    patch = kinds[y0:y1, x0:x1]
    # The red outline and emission arrows can cover more pixels than a blue
    # or green triangle fill. Use a substantial non-red fill before the
    # outline's area, rather than calling every outlined LED red.
    fills = [pigment for pigment in (3, 5) if np.sum(patch == pigment) >= max(12, .075 * patch.size)]
    if len(fills) == 1:
        return fills[0]
    if len(fills) > 1:
        return None
    scores = [(int(np.sum(patch == pigment)), pigment) for pigment in (1, 3, 5)]
    area, pigment = max(scores)
    return pigment if area >= max(10, .12 * patch.size) else None


def _common_terminal(kinds, labels, label, raw, vertical, legacy, bgr):
    ys, xs = np.where(labels == label)
    if not len(xs):
        return None
    axis, cross = (1, 0) if vertical else (0, 1)
    normals, tangents = (ys, xs) if vertical else (xs, ys)
    blue, green, red = np.moveaxis(bgr.astype(np.int16), -1, 0)
    weak = (green - blue >= 8) & (np.abs(red - green) <= 25) & (green >= 40)
    proposals = []
    for side, normal in [(-1, int(normals.min())), (1, int(normals.max()))]:
        if not (normal < raw[axis] - 2 if side < 0 else normal > raw[axis + 2] + 2):
            continue
        cells = tangents[normals == normal]
        if not 1 <= len(cells) <= 4 or cells.max() - cells.min() > 3:
            continue
        tangent = float((cells.min() + cells.max()) / 2)
        def point(n, t):
            return (int(round(t)), n) if vertical else (n, int(round(t)))
        own = [point(normal - side * step, tangent) for step in range(5)]
        if not all(0 <= x < kinds.shape[1] and 0 <= y < kinds.shape[0] and labels[y, x] == label for x, y in own):
            continue
        neighbour = [point(normal + side, t) for t in cells]
        if not all(0 <= x < kinds.shape[1] and 0 <= y < kinds.shape[0] for x, y in neighbour):
            continue
        if all(kinds[y, x] == 3 for x, y in neighbour):
            source = 'body_attached_red_green_transition'
            direction = (0, side) if vertical else (side, 0)
            xy = (tangent, normal + .5 * side) if vertical else (normal + .5 * side, tangent)
            if legacy._wire_path(kinds == 3, xy, direction) is None:
                continue
        elif any(weak[y, x] for x, y in neighbour):
            # This is the boundary belonging to the RGB side of the weak
            # span, rather than the resistor's boundary on the far side.
            source = 'mixed_colour_span_boundary'
            xy = (tangent, normal + .5 * side) if vertical else (normal + .5 * side, tangent)
            span = 0
            for step in range(1, 4):
                x, y = point(normal + side * step, tangent)
                if not (0 <= x < kinds.shape[1] and 0 <= y < kinds.shape[0]) or not weak[y, x]:
                    break
                span += 1
            if not span:
                continue
            peer = point(normal + side * (span + 1), tangent)
            if not (0 <= peer[0] < kinds.shape[1] and 0 <= peer[1] < kinds.shape[0] and kinds[peer[1], peer[0]] == 1):
                continue
        else:
            continue
        proposals.append(dict(point=xy, source=source,
                              proof=dict(kind='rgb_common_external_boundary', coordinate_img=list(xy),
                                         colour_span_length=span if source=='mixed_colour_span_boundary' else 0,
                                         own_lead_pixels_img=[list(p) for p in own],
                                         interface_pixels_img=[list(p) for p in neighbour],
                                         internal_red_component_bounds_img=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])))
    return proposals[0] if len(proposals) == 1 else None


def group_components(components, pins, report, bgr):
    """Return grouped geometry and an immutable ledger; unrelated/ICs untouched."""
    from . import passive_pins as legacy, passive_terminal_geometry as geometry

    leds = [c for c in components if c.get('type') == 'led' and len(report.get(c.get('id'), {}).get('after', [])) == 1]
    if len(leds) < 3:
        return components, pins, report, []
    kinds = geometry._pigments(bgr)
    candidates = []
    for comp in leds:
        pigment = _body_pigment(kinds, comp)
        if pigment is not None:
            candidates.append((comp, pigment))
    if not {1, 3, 5} <= {pigment for _, pigment in candidates}:
        return components, pins, report, []
    red_mask = (kinds == 1).astype(np.uint8)
    for comp in leds:
        raw = comp.get('bbox_raw_img') or comp['bbox_img']
        red_mask[max(0, int(raw[1])):math.ceil(raw[3]) + 1,
                 max(0, int(raw[0])):math.ceil(raw[2]) + 1] = 0
    _, labels = cv2.connectedComponents(red_mask, 8)
    proposals = []
    for members in itertools.combinations(candidates, 3):
        if {paint for _, paint in members} != {1, 3, 5}:
            continue
        bodies = [c for c, _ in members]
        boxes = [c.get('bbox_raw_img') or c['bbox_img'] for c in bodies]
        raw = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes)]
        widths = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
        if max(widths) > 1.4 * min(widths) or max(raw[2] - raw[0], raw[3] - raw[1]) > 6 * min(widths):
            continue
        ports, internal = [], []
        for comp, paint in members:
            rec = report[comp['id']]
            point = rec['after'][0]
            box = comp.get('bbox_raw_img') or comp['bbox_img']
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            vertical = abs(point[1] - cy) > abs(point[0] - cx)
            axis = 1 if vertical else 0
            side = 1 if point[axis] > (cy if vertical else cx) else -1
            normal = box[axis] if side > 0 else box[axis + 2]
            neighbours = []
            for step in (2, 3, 4):
                for shift in (-1, 0, 1):
                    position = normal - side * step
                    x, y = ((int(round(point[0] + shift)), int(round(position))) if vertical
                            else (int(round(position)), int(round(point[1] + shift))))
                    if 0 <= x < kinds.shape[1] and 0 <= y < kinds.shape[0] and labels[y, x]:
                        neighbours.append((int(labels[y, x]), [x, y]))
            if not neighbours or len({n[0] for n in neighbours}) != 1:
                break
            internal.append(neighbours[0])
            ports.append(dict(id=comp['id'], point=point, pigment=paint, axis_y=vertical, side=side,
                              source=rec['sources'][0], proof=rec['connection_evidence'][0]))
        if len(ports) != 3 or len({p['axis_y'] for p in ports}) != 1 or len({i[0] for i in internal}) != 1:
            continue
        # Two outer leads face one way, the third faces the opposite way.
        if sorted(p['side'] for p in ports) not in [[-1, -1, 1], [-1, 1, 1]]:
            continue
        common = _common_terminal(kinds, labels, internal[0][0], raw, ports[0]['axis_y'], legacy, bgr)
        if common is None:
            continue
        proposals.append((bodies, raw, ports, common, internal))
    used, groups = set(), []
    result_components = list(components)
    result_pins, result_report = dict(pins), dict(report)
    for bodies, raw, ports, common, internal in proposals:
        ids = {c['id'] for c in bodies}
        if ids & used or any(sum(c['id'] in {b['id'] for b in candidate[0]} for candidate in proposals) > 1 for c in bodies):
            continue
        used.update(ids)
        leader = min(bodies, key=lambda c: c['id'])
        cid = leader['id']
        terminals = [dict(id=None, point=common['point'], source=common['source'], proof=common['proof'])]
        # Deterministic geometry order; actual pin numbers still come from
        # stage2 OCR or its existing generic positional fallback.
        centre = ((raw[0] + raw[2]) / 2, (raw[1] + raw[3]) / 2)
        reference = math.atan2(common['point'][1] - centre[1], common['point'][0] - centre[0])
        ports.sort(key=lambda p: (math.atan2(p['point'][1] - centre[1], p['point'][0] - centre[0]) - reference) % (2 * math.pi))
        terminals.extend(ports)
        aggregate = dict(leader, subtype='rgb', terminal_group='rgb_led',
                         bbox_raw_img=raw, bbox_img=raw, center_img=list(centre),
                         bbox_json=[raw[0], bgr.shape[0] - raw[3], raw[2], bgr.shape[0] - raw[1]])
        result_components = [aggregate if c['id'] == cid else c for c in result_components if c['id'] not in ids or c['id'] == cid]
        for member in ids:
            result_pins.pop(member, None)
            result_report.pop(member, None)
        result_pins[cid] = [tuple(t['point']) for t in terminals]
        result_report[cid] = dict(type='led', before=pins.get(cid, []), after=result_pins[cid],
                                  sources=[t['source'] for t in terminals], axis_y=ports[0]['axis_y'],
                                  connection_evidence=[t['proof'] for t in terminals], coordinate_rule='image_color_terminal_rules')
        groups.append(dict(kind='rgb_led', id=cid, member_ids=sorted(ids), bbox_img=raw,
                           original_components=copy.deepcopy(bodies), original_pins={i:copy.deepcopy(pins.get(i, [])) for i in ids},
                           original_reports={i:copy.deepcopy(report.get(i, {})) for i in ids},
                           terminals=copy.deepcopy(terminals), internal_ports_img=[p for _, p in internal],
                           numbering='OCR or generic positional fallback; not independently verified'))
    return result_components, result_pins, result_report, groups


def clip_external_rays(segments, groups):
    """Other symbols' long rays must not traverse the recovered package."""
    result=list(segments)
    for group in groups:
        box=group['bbox_img']
        lo=[box[0]-.5,box[1]-.5];hi=[box[2]+.5,box[3]+.5]
        clipped=[]
        for segment in result:
            a,b=segment;delta=[b[0]-a[0],b[1]-a[1]]
            start,end=0.,1.
            intersects=True
            for axis in (0,1):
                if abs(delta[axis])<1e-9:
                    if not lo[axis]<a[axis]<hi[axis]:intersects=False;break
                    continue
                near,far=sorted([(lo[axis]-a[axis])/delta[axis],(hi[axis]-a[axis])/delta[axis]])
                start,end=max(start,near),min(end,far)
                if start>=end:intersects=False;break
            if not intersects:
                clipped.append(segment);continue
            point=lambda t:[float(a[0]+t*delta[0]),float(a[1]+t*delta[1])]
            if start>0 and math.dist(a,point(start))>=.5:clipped.append([list(a),point(start)])
            if end<1 and math.dist(point(end),b)>=.5:clipped.append([point(end),list(b)])
        result=clipped
    return result
