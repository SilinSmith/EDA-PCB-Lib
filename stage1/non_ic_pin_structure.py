"""Reconcile monochrome terminal counts without defining exact pin positions.

Ground bars and power caps describe one electrical port, even when wire or
glyph tracing leaves several candidates. Retain one existing estimate on the
observed stem side. An axial passive may recover a missing opposite candidate
only from a real uninterrupted stroke, using the existing box-outside estimate.
Colour terminal reports and all IC candidates remain authoritative.
"""
from __future__ import annotations

import math

import numpy as np

from . import passive_pins, passive_terminal_geometry as geometry, pins as legacy


def _supported_body(geo, raw):
    axis = int(geo['axis_y'])
    edge = geometry._body_edge(geo, geo['only_side'])
    return raw[axis] - 4 <= edge <= raw[axis + 2] + 4


def _power_geometries(kinds, raw):
    """A neutral cap and narrow stem may share the label's ink colour."""
    ink = kinds != 0
    grounds = [g for g in geometry._ground_geometries(kinds, raw, passive_pins)
               if _supported_body(g, raw)]
    fits = []
    for vertical in (False, True):
        axis, cross = (1, 0) if vertical else (0, 1)
        tangent = (raw[cross] + raw[cross + 2]) / 2
        width = raw[cross + 2] - raw[cross]
        bands = []
        for pos in range(max(0, math.floor(raw[axis] - 4)), math.ceil(raw[axis + 2] + 5)):
            for a, b in geometry._line_runs(ink, pos, tangent, vertical, max(12, 1.6 * width)):
                centre = (a + b - 1) / 2
                if not max(6, .6 * width) <= b - a <= max(16, 4 * width):
                    continue
                if abs(centre - tangent) > max(3, .4 * width):
                    continue
                if (bands and bands[-1]['last'] == pos - 1
                        and abs(bands[-1]['width'] - (b - a)) <= 1
                        and abs(bands[-1]['tangent'] - centre) <= 1):
                    bands[-1]['last'] = pos
                else:
                    bands.append(dict(first=pos, last=pos, width=b - a, tangent=centre))
        for band in bands:
            if band['last'] - band['first'] > 3:
                continue
            for side in (-1, 1):
                middle = (band['first'] + band['last']) / 2
                geo = dict(axis_y=vertical, low=middle - .5, high=middle + .5,
                           tangent=band['tangent'], only_side=side, kind='observed_neutral_power_cap',
                           body_bands_img=[band])
                if (not _supported_body(geo, raw)
                        or any(g['axis_y'] == vertical and abs(g['tangent'] - geo['tangent']) <= 2
                               and abs(geometry._body_edge(g, g['only_side']) - geometry._body_edge(geo, side)) <= 2
                               for g in grounds)):
                    continue
                edge = band['last'] if side > 0 else band['first']
                other = band['first'] - 1 if side > 0 else band['last'] + 1
                if any(a <= geo['tangent'] <= b - 1
                       for a, b in geometry._line_runs(ink, other, geo['tangent'], vertical, 5)):
                    continue
                samples = geometry._trace(kinds, geo, side, 4)
                if len(samples) == 4 and all(s['pigment'] == 7 and s['width'] <= 4 for s in samples):
                    fits.append(geo)
    unique = []
    for fit in fits:
        if not any(fit['axis_y'] == old['axis_y'] and fit['only_side'] == old['only_side']
                   and abs(fit['tangent'] - old['tangent']) <= 1
                   and abs(geometry._body_edge(fit, fit['only_side']) - geometry._body_edge(old, old['only_side'])) <= 1
                   for old in unique):
            unique.append(fit)
    return unique


def _stroke_to_estimate(ink, box, point, axis, side):
    """Follow an inset zigzag body to a real straight arm, without white gaps."""
    edge = box[axis + 2] if side > 0 else box[axis]
    start = math.ceil(edge) if side > 0 else math.floor(edge)
    cross = 1 - axis
    tangent = point[cross]
    width = box[cross + 2] - box[cross]
    pixels, previous, narrow = [], None, []
    limit = math.ceil(max(24, 3 * width))
    passed_estimate = False
    for step in range(limit):
        pos = start + side * step
        runs = [(a, b) for a, b in geometry._line_runs(ink, pos, tangent, bool(axis), max(8, width))
                if abs((a + b - 1) / 2 - tangent) <= max(4, width)
                and b - a <= max(8, 1.5 * width)]
        if len(runs) != 1:
            return None
        a, b = runs[0]
        if previous is not None and max(a, previous[0]) > min(b, previous[1]):
            return None
        previous = (a, b)
        centre = (a + b - 1) / 2
        if b - a <= 5:
            narrow.append(centre)
        else:
            narrow = []
        pixels.extend([[q, pos] if axis else [pos, q] for q in range(a, b)])
        if abs(pos - point[axis]) <= 1 and a - 1 <= point[cross] <= b:
            passed_estimate = True
        if passed_estimate and len(narrow) >= 6 and max(narrow[-6:]) - min(narrow[-6:]) <= 1:
            return pixels
    return None


def reconcile(components, pins, bgr, colour_reports=None):
    """Return count-reconciled candidates plus an honest structural ledger."""
    result = {cid: list(points) for cid, points in pins.items()}
    report = {}
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return result, report
    kinds = geometry._pigments(bgr)
    ink = kinds == 7
    for comp in components:
        cid, typ = comp.get('id'), comp.get('type')
        old = result.get(cid, [])
        if (cid in (colour_reports or {}) or typ not in {'gnd', 'v', 'vcc', 'r', 'c'}
                or (typ in {'r', 'c'} and len(old) != 1)
                or (typ in {'gnd', 'v', 'vcc'} and len(old) <= 1)):
            continue
        raw = passive_pins._box(comp.get('bbox_raw_img')) or passive_pins._box(comp.get('bbox_img'))
        box = passive_pins._box(comp.get('bbox_img'))
        if raw is None or box is None:
            continue
        h, w = kinds.shape
        left, top = max(0, math.floor(raw[0] - 24)), max(0, math.floor(raw[1] - 24))
        right, bottom = min(w, math.ceil(raw[2] + 25)), min(h, math.ceil(raw[3] + 25))
        local = kinds[top:bottom, left:right]
        if not local.size or np.any((local != 0) & (local != 7)):
            continue
        if typ in {'gnd', 'v', 'vcc'}:
            local_raw = (raw[0] - left, raw[1] - top, raw[2] - left, raw[3] - top)
            fits = (geometry._ground_geometries(local, local_raw, passive_pins) if typ == 'gnd'
                    else _power_geometries(local, local_raw))
            fits = [g for g in fits if len(geometry._trace(local, g, g['only_side'], 4)) == 4]
            if len(fits) != 1:
                continue
            geo = fits[0];axis = int(geo['axis_y']);side = geo['only_side']
            edge = geometry._body_edge(geo, side) + (top if axis else left)
            tangent = geo['tangent'] + (left if axis else top)
            widths = [b.get('width', b['bounds'][3 - axis] - b['bounds'][1 - axis] + 1)
                      if 'bounds' in b else b['width'] for b in geo['body_bands_img']]
            eligible = [i for i, p in enumerate(old)
                        if (-4 if typ == 'gnd' else 1) <= (p[axis] - edge) * side <= 48
                        and abs(p[1 - axis] - tangent) <= max(3, .55 * max(widths))]
            if not eligible:
                continue
            chosen = min(eligible, key=lambda i: (abs(old[i][1 - axis] - tangent), abs(old[i][axis] - edge), i))
            result[cid] = [old[chosen]]
            report[cid] = dict(type=typ, kind='single_port_count_reconciliation', before=list(old),
                               after=result[cid], retained_raw_indices=[chosen],
                               discarded_raw_indices=[i for i in range(len(old)) if i != chosen],
                               observed_axis_y=bool(axis), observed_outward_side=side,
                               body_geometry_local=geo, local_origin_img=[left, top],
                               coordinate_status='unchanged_existing_estimate_not_precision_verified')
        else:
            axis = int(box[3] - box[1] > box[2] - box[0]);cross = 1 - axis
            span, width = box[axis + 2] - box[axis], box[cross + 2] - box[cross]
            if span < 1.5 * width:
                continue
            middle = (box[axis] + box[axis + 2]) / 2
            side = 1 if old[0][axis] > middle else -1
            stubs = []
            for point in legacy._stub_points(box, ink, legacy.PIN_OUT):
                facing = 1 if point[axis] > middle else -1
                edge = box[axis + 2] if facing > 0 else box[axis]
                if not 3 <= (point[axis] - edge) * facing <= 20 or not box[cross] - 1 <= point[cross] <= box[cross + 2] + 1:
                    continue
                path = _stroke_to_estimate(ink, box, point, axis, facing)
                if path:
                    stubs.append((facing, point, path))
            own = [s for s in stubs if s[0] == side and math.dist(s[1], old[0]) <= 3]
            missing = [s for s in stubs if s[0] == -side]
            if len(own) != 1 or len(missing) != 1:
                continue
            result[cid] = old + [missing[0][1]]
            report[cid] = dict(type=typ, kind='observed_opposed_stub_count_recovery', before=list(old),
                               after=result[cid], retained_raw_indices=[0], new_raw_indices=[1],
                               own_stroke_pixels_img=own[0][2], recovered_stroke_pixels_img=missing[0][2],
                               observed_axis_y=bool(axis),
                               coordinate_status='legacy_box_outside_estimate_not_precision_verified')
    return result, report
