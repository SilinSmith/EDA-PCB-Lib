"""Additional colour terminals, using body geometry to establish ink roles.

Detector boxes are search windows, not symmetry centres. Parallel capacitor
plates and closed resistor outlines supply observed body edges; ground bars
supply an outward stem. Only the external stem is searched for a wire boundary
or a filled junction. An unmarked end needs a measured opposite end, a symmetric
body and a continuous connection to a detected peer before it can be inferred.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


PIGMENTS = {1: 'red', 2: 'warm', 3: 'green', 4: 'cyan', 5: 'blue',
            6: 'purple', 7: 'neutral'}
NEW_SOURCES = frozenset({'first_filled_junction_center', 'lead_wire_colour_boundary',
                         'symmetric_opposed_terminal', 'peer_shared_terminal',
                         'direct_pin_corner', 'same_symbol_terminal_length'})


def _pigments(bgr):
    """Group antialiased ink by hue, with a separate dark neutral layer."""
    pixels = bgr.astype(np.int16)
    lo, hi = pixels.min(axis=2), pixels.max(axis=2)
    coloured = (hi - lo >= 18) & (lo < 235)
    hue = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    kinds = np.zeros(lo.shape, np.uint8)
    for value, mask in ((1, (hue < 15) | (hue >= 172)), (2, (hue >= 15) & (hue < 35)),
                        (3, (hue >= 35) & (hue < 85)), (4, (hue >= 85) & (hue < 105)),
                        (5, (hue >= 105) & (hue < 137)), (6, (hue >= 137) & (hue < 172))):
        kinds[coloured & mask] = value
    kinds[~coloured & (hi < 220)] = 7
    return kinds


def _runs(values):
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _line_runs(ink, pos, tangent, vertical, radius):
    h, w = ink.shape
    if not 0 <= pos < (h if vertical else w):
        return []
    low = max(0, int(math.floor(tangent - radius)))
    high = min(w if vertical else h, int(math.ceil(tangent + radius)) + 1)
    cells = ink[pos, low:high] if vertical else ink[low:high, pos]
    return [(int(a + low), int(b + low)) for a, b in _runs(cells)]


def _body_edge(geometry, side):
    """Start a single-port stem outside its actual longest bar, not its centre."""
    if 'only_side' not in geometry:
        return geometry['high'] if side > 0 else geometry['low']
    axis = 1 if geometry['axis_y'] else 0
    bands = []
    for band in geometry['body_bands_img']:
        if 'first' in band:
            bands.append((band['first'], band['last'], band.get('width', 0)))
        else:
            box = band['bounds']
            bands.append((box[axis], box[axis + 2], box[3 - axis] - box[1 - axis] + 1))
    first, last, _ = max(bands, key=lambda value: value[2])
    return last + .5 if side > 0 else first - .5


def _axial_body(ink, raw, typ, vertical):
    """Measure straight plates or a closed rectangle, rejecting body guesses."""
    axis, cross = (1, 0) if vertical else (0, 1)
    centre = (raw[axis] + raw[axis + 2]) / 2
    tangent = (raw[cross] + raw[cross + 2]) / 2
    span, width = raw[axis + 2] - raw[axis], raw[cross + 2] - raw[cross]
    bands = []
    margin = max(6, min(12, .35 * span))
    for pos in range(max(0, int(raw[axis] - margin)), int(raw[axis + 2] + margin) + 1):
        bars = [(a, b) for a, b in _line_runs(ink, pos, tangent, vertical, max(8, width))
                if b - a >= max(4, .6 * width)
                and abs((a + b - 1) / 2 - tangent) <= max(3, .25 * width)]
        if len(bars) != 1:
            continue
        a, b = bars[0]
        if (bands and pos == bands[-1]['last'] + 1
                and abs((a + b - 1) / 2 - bands[-1]['tangent']) <= 1.5):
            bands[-1]['last'] = pos
            if b - a > bands[-1]['width']:
                bands[-1].update(width=b - a, tangent=(a + b - 1) / 2, cross_low=a, cross_high=b - 1)
        else:
            bands.append(dict(first=pos, last=pos, width=b - a, tangent=(a + b - 1) / 2,
                              cross_low=a, cross_high=b - 1))
    fits = []
    for left, right in zip(bands, bands[1:]):
        thickness = max(left['last'] - left['first'] + 1, right['last'] - right['first'] + 1)
        if thickness > max(4, .25 * width):
            continue
        if (abs(left['tangent'] - right['tangent']) > 1.5
                or min(left['width'], right['width']) < .75 * max(left['width'], right['width'])):
            continue
        low, high = left['first'] - .5, right['last'] + .5
        middle = (low + high) / 2
        if abs(middle - centre) > max(6, .45 * span):
            continue
        gap = right['first'] - left['last'] - 1
        if typ == 'c':
            if not 1 <= gap <= max(12, .8 * width):
                continue
            # Ordinary parallel plates have an empty gap along their stem.
            t = int(round((left['tangent'] + right['tangent']) / 2))
            gap_cells = ink[left['last'] + 1:right['first'], t] if vertical else ink[t, left['last'] + 1:right['first']]
            if not gap_cells.size or np.mean(gap_cells) > .2:
                continue
        else:
            if not max(5, .5 * span) <= high - low <= 1.8 * span:
                continue
            # A resistor rectangle needs both observed side walls. Text and
            # a pair of disconnected bars cannot supply symmetry evidence.
            edges = (left['cross_low'], left['cross_high'])
            if max(abs(right['cross_low'] - edges[0]), abs(right['cross_high'] - edges[1])) > 1:
                continue
            walls = []
            for t in edges:
                hits = [any(ink[pos, q] if vertical else ink[q, pos]
                            for q in range(max(0, t - 1), min(ink.shape[1 if vertical else 0], t + 2)))
                        for pos in range(left['last'] + 1, right['first'])]
                walls.append(bool(hits) and np.mean(hits) >= .9)
            if not all(walls):
                continue
        fits.append(dict(axis_y=vertical, low=low, high=high,
                         tangent=(left['tangent'] + right['tangent']) / 2,
                         kind='parallel_capacitor_plates' if typ == 'c' else 'closed_resistor_outline',
                         body_bands_img=[left, right]))
    fits.sort(key=lambda fit: abs((fit['low'] + fit['high']) / 2 - centre))
    if len(fits) > 1 and abs(abs((fits[0]['low'] + fits[0]['high']) / 2 - centre)
                              - abs((fits[1]['low'] + fits[1]['high']) / 2 - centre)) < 2:
        return None
    if fits:
        return fits[0]
    if typ == 'r':
        return _filled_resistor_body(ink, raw, vertical)
    return None


def _filled_resistor_body(ink, raw, vertical):
    """A solid rectangular resistor is also an observed symmetric body."""
    axis, cross = (1, 0) if vertical else (0, 1)
    span, width = raw[axis + 2] - raw[axis], raw[cross + 2] - raw[cross]
    if span < 1.25 * width:
        return None
    x0,y0=max(0,int(raw[0]-3)),max(0,int(raw[1]-3))
    x1,y1=min(ink.shape[1],math.ceil(raw[2]+4)),min(ink.shape[0],math.ceil(raw[3]+4))
    local=ink[y0:y1,x0:x1].astype(np.uint8)
    opened=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(opened,8)
    candidates=[]
    for x,y,w,h,area in stats[1:]:
        if min(w,h)<4 or area<.9*w*h:
            continue
        box=(x+x0,y+y0,x+x0+w-1,y+y0+h-1)
        normal_span=(h if vertical else w)
        cross_span=(w if vertical else h)
        if not (.65*span<=normal_span<=1.35*span and .6*width<=cross_span<=1.5*width):
            continue
        middle=(box[axis]+box[axis+2])/2
        tangent=(box[cross]+box[cross+2])/2
        if abs(middle-(raw[axis]+raw[axis+2])/2)>3 or abs(tangent-(raw[cross]+raw[cross+2])/2)>2:
            continue
        low,high=box[axis]-.5,box[axis+2]+.5
        candidates.append(dict(axis_y=vertical,low=low,high=high,tangent=tangent,
                               kind='filled_resistor_body',body_bands_img=[
                                   dict(first=box[axis],last=box[axis],width=int(cross_span),tangent=tangent),
                                   dict(first=box[axis+2],last=box[axis+2],width=int(cross_span),tangent=tangent)]))
    return candidates[0] if len(candidates)==1 else None


def _filled_nodes(kinds):
    """Compact filled bulbs in any ink layer; wire arms are checked later."""
    nodes = []
    for pigment in PIGMENTS:
        mask = (kinds == pigment).astype(np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        for label, stat in enumerate(stats[1:], 1):
            x, y, bw, bh, area = map(int, stat)
            large_blue = pigment == 5 and 12 < max(bw, bh) <= 16
            if large_blue:
                # A painted pin tip can extend one narrow row beyond a bulb.
                # Trim that tip only for the previously unsupported large dots;
                # the established small-dot coordinates remain unchanged.
                bulb = mask[y:y + bh, x:x + bw]
                rows, cols = bulb.sum(axis=1), bulb.sum(axis=0)
                xlo, xhi, ylo, yhi = 0, bw - 1, 0, bh - 1
                if cols[0] <= 3 < cols[-1]:
                    xlo += 1
                elif cols[-1] <= 3 < cols[0]:
                    xhi -= 1
                if rows[0] <= 3 < rows[-1]:
                    ylo += 1
                elif rows[-1] <= 3 < rows[0]:
                    yhi -= 1
                x, y, bw, bh = x + xlo, y + ylo, xhi - xlo + 1, yhi - ylo + 1
            aspect = .65 if large_blue else (.55 if pigment == 5 and max(bw, bh) <= 6 else .7)
            if not (3 <= min(bw, bh) <= max(bw, bh) <= (16 if large_blue else 12)
                    and min(bw, bh) >= aspect * max(bw, bh) and area >= .45 * bw * bh):
                continue
            bulb = mask[y:y + bh, x:x + bw]
            if np.mean(bulb) < .5:
                continue
            # Five-pixel blue dots often lose one antialiased edge where a
            # red stem meets them. Their independent wire arms are checked
            # by _contact; perfect raster symmetry is not required.
            symmetry = .35 if pigment == 5 and max(bw, bh) <= 6 else (.3 if max(bw, bh) <= 4 else .2)
            if max(np.mean(bulb != bulb[::-1]), np.mean(bulb != bulb[:, ::-1])) > symmetry:
                continue
            # A solid square of text is not a circular or diamond node.
            if min(bw, bh) > 3 and np.mean(bulb) > .96:
                continue
            if pigment == 5 and not large_blue and min(bw, bh) < .7 * max(bw, bh):
                # A thin painted tip belongs to the outgoing stroke. Keep
                # the filled bulb's overlapping 3x3 cores for its centre.
                cores = cv2.erode(bulb, np.ones((3, 3), np.uint8),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
                cy, cx = np.nonzero(cores)
                if not len(cx) or max(np.ptp(cx), np.ptp(cy)) > 2:
                    continue
                x, y = x + int(cx.min()) - 1, y + int(cy.min()) - 1
                bw, bh = int(np.ptp(cx)) + 3, int(np.ptp(cy)) + 3
            point = (x + (bw - 1) / 2, y + (bh - 1) / 2)
            node = dict(point=point, bounds=(x, y, x + bw - 1, y + bh - 1), pigment=pigment)
            if large_blue:
                node['large_blue'] = True
            nodes.append(node)
    return nodes


def _single_port_geometries(kinds, raw, typ):
    """Observe a power T or compact ground taper and its one narrow stem.

    A detector box may include a label and be offset from the real axis. The
    actual centred bars and uninterrupted stem establish that axis instead.
    Neither a box centre nor a nearby text stroke supplies a terminal.
    """
    found = []
    for pigment in PIGMENTS:
        ink = kinds == pigment
        for vertical in (False, True):
            axis, cross = (1, 0) if vertical else (0, 1)
            tangent = (raw[cross] + raw[cross + 2]) / 2
            width = raw[cross + 2] - raw[cross]
            bands = []
            for pos in range(max(0, int(raw[axis] - 6)), int(raw[axis + 2] + 7)):
                bars = [(a, b) for a, b in _line_runs(ink, pos, tangent, vertical, max(12, 1.5 * width))
                        if 3 <= b - a <= max(16, 2 * width)
                        and abs((a + b - 1) / 2 - tangent) <= max(4, .75 * width)]
                for a, b in bars:
                    centre = (a + b - 1) / 2
                    if (bands and pos == bands[-1]['last'] + 1
                            and abs(centre - bands[-1]['tangent']) <= 1
                            and abs(b - a - bands[-1]['width']) <= 1):
                        bands[-1]['last'] = pos
                    else:
                        bands.append(dict(first=pos, last=pos, width=b - a, tangent=centre))
            choices = []
            if typ in {'v', 'vcc'}:
                choices = [(band, [band], side) for band in bands for side in (-1, 1)
                           if band['width'] >= max(6, .55 * width)
                           and band['last'] - band['first'] <= 3]
            else:
                for a, b, c in zip(bands, bands[1:], bands[2:]):
                    if (any(band['last'] - band['first'] > 2 for band in (a, b, c))
                            or not all(1 <= gap <= 2 for gap in (b['first'] - a['last'], c['first'] - b['last']))
                            or max(band['tangent'] for band in (a, b, c)) - min(band['tangent'] for band in (a, b, c)) > 1):
                        continue
                    longest, side = ((a, -1) if a['width'] >= 1.3 * b['width'] >= 1.69 * c['width'] else
                                     (c, 1) if c['width'] >= 1.3 * b['width'] >= 1.69 * a['width'] else (None, 0))
                    if longest is not None and longest['width'] >= max(6, .55 * width):
                        choices.append((longest, [a, b, c], side))
            for longest, body, side in choices:
                edge = longest['last'] if side > 0 else longest['first']
                tangent = longest['tangent']
                stem = []
                for step in range(1, 5):
                    runs = [(a, b) for a, b in _line_runs(ink, edge + side * step, tangent, vertical, 6)
                            if a <= tangent <= b - 1 and b - a <= 4
                            and abs((a + b - 1) / 2 - tangent) <= 1]
                    if len(runs) != 1:
                        break
                    stem.append(runs[0])
                if len(stem) != 4:
                    continue
                if typ in {'v', 'vcc'}:
                    other = longest['first'] - 1 if side > 0 else longest['last'] + 1
                    if any(a <= tangent <= b - 1 for a, b in _line_runs(kinds != 0, other, tangent, vertical, 6)):
                        continue
                    if any(a <= tangent <= b - 1 and b - a >= 3
                           for step in range(1, 13)
                           for a, b in _line_runs(ink, other - side * step, tangent, vertical, max(12, width))):
                        continue
                kind = 'single_power_bar' if typ in {'v', 'vcc'} else 'compact_ground_bars'
                normal = (longest['first'] + longest['last']) / 2
                found.append(dict(axis_y=vertical, low=normal - .5, high=normal + .5,
                                  tangent=tangent, only_side=side, kind=kind,
                                  body_bands_img=body, symbol_signature=kind))
    unique = []
    for geo in found:
        if not any(geo['axis_y'] == old['axis_y'] and geo['only_side'] == old['only_side']
                   and abs(geo['tangent'] - old['tangent']) <= 1
                   and abs(_body_edge(geo, geo['only_side']) - _body_edge(old, old['only_side'])) <= 1
                   for old in unique):
            unique.append(geo)
    return unique


def _ground_geometries(kinds, raw, legacy):
    """Tapering bars can be adjacent raster rows in a small colour drawing."""
    found = []
    for pigment in PIGMENTS:
        ink = kinds == pigment
        if not ink.any():
            continue
        _, labels = cv2.connectedComponents(ink.astype(np.uint8), 8)
        stems = legacy._ground_body_labels(ink, labels, raw, 0, 0, with_axes=True)['stem_anchors']
        for stem in stems:
            axis = 1 if stem['axis_y'] else 0
            found.append(dict(axis_y=stem['axis_y'], low=stem['point'][axis] - .5,
                              high=stem['point'][axis] + .5, tangent=stem['point'][1 - axis],
                              only_side=stem['outward_side'], kind='tapering_ground_bars',
                              body_bands_img=[dict(bounds=list(stem['bar_bounds']))]))
        for vertical in (False, True):
            axis, cross = (1, 0) if vertical else (0, 1)
            tangent = (raw[cross] + raw[cross + 2]) / 2
            width = raw[cross + 2] - raw[cross]
            bands = []
            for pos in range(max(0, int(raw[axis] - 8)), int(raw[axis + 2] + 9)):
                bars = [(a, b) for a, b in _line_runs(ink, pos, tangent, vertical, max(8, width))
                        if b - a >= 3 and abs((a + b - 1) / 2 - tangent) <= max(3, .2 * width)]
                if len(bars) != 1:
                    continue
                a, b = bars[0]
                if (bands and pos == bands[-1]['last'] + 1
                        and abs((a + b - 1) / 2 - bands[-1]['tangent']) <= 1.5
                        and min(b - a, bands[-1]['width']) >= .6 * max(b - a, bands[-1]['width'])):
                    bands[-1]['last'] = pos
                    if b - a > bands[-1]['width']:
                        bands[-1].update(width=b - a, tangent=(a + b - 1) / 2)
                else:
                    bands.append(dict(first=pos, last=pos, width=b - a, tangent=(a + b - 1) / 2))
            for a, b, c in zip(bands, bands[1:], bands[2:]):
                if (any(bar['last'] - bar['first'] + 1 > 6 for bar in (a, b, c))
                        or not all(1 <= gap <= 7 for gap in (b['first'] - a['last'], c['first'] - b['last']))
                        or max(bar['tangent'] for bar in (a, b, c)) - min(bar['tangent'] for bar in (a, b, c)) > 1.5):
                    continue
                longest, side = (a, -1) if a['width'] >= 1.2 * b['width'] >= 1.44 * c['width'] else (
                    (c, 1) if c['width'] >= 1.2 * b['width'] >= 1.44 * a['width'] else (None, 0))
                if longest is None or longest['width'] < .7 * width:
                    continue
                normal = (longest['first'] + longest['last']) / 2
                found.append(dict(axis_y=vertical, low=normal - .5, high=normal + .5,
                                  tangent=longest['tangent'], only_side=side,
                                  kind='tapering_ground_bars', body_bands_img=[a, b, c]))
    unique = []
    # Four-bar grounds produce nested three-bar fits. Use the outermost
    # longest bar, never a shorter bar that the stem would have to cross.
    found.sort(key=lambda geo: geo['low'] * geo['only_side'], reverse=True)
    for geo in found:
        if not any(geo['axis_y'] == old['axis_y'] and geo['only_side'] == old['only_side']
                   and abs(geo['tangent'] - old['tangent']) < 2
                   and abs(geo['low'] - old['low']) <= 8 for old in unique):
            unique.append(geo)
    return unique


def _short_turn_contact(kinds, geometry, side, samples, first, index, legacy, raw, peers):
    """A short painted elbow joins this straight stem to a perpendicular pin."""
    axis, cross = int(geometry['axis_y']), 1 - int(geometry['axis_y'])
    tangent = samples[index - 1]['point'][cross]
    end = index
    while end < len(samples) and samples[end]['pigment'] in {2, 3}:
        end += 1
    span = samples[index:end]
    if not 1 <= len(span) <= 3:
        return None
    normal = span[0]['pos'] if len(span) == 1 else (samples[index - 1]['pos'] + span[0]['pos']) / 2
    point = _point(geometry, normal, tangent)
    proposals = []
    for peer_id, box in peers:
        for turn in (-1, 1):
            facing = box[cross] if turn > 0 else box[cross + 2]
            distance = (facing - tangent) * turn
            if not 4 <= distance <= 48:
                continue
            paths = []
            for sample in span:
                if not box[axis] - 2 <= sample['pos'] <= box[axis + 2] + 2:
                    continue
                origin = _point(geometry, sample['pos'], tangent)
                target = list(origin)
                target[cross] = math.ceil(facing) if turn > 0 else math.floor(facing)
                mask = kinds == 1
                # Only the actual painted contact cells may bridge the red
                # peer pin. No white pixels or coloured wire beyond it enter.
                for cell in span:
                    for x, y in cell['pixels']:
                        if abs((x, y)[cross] - tangent) <= 2 and kinds[y, x] in {2, 3}:
                            mask[y, x] = True
                path = legacy._continuous_red_segment(mask, origin, target, not geometry['axis_y'])
                forward = [_point(geometry, samples[end - 1]['pos'] + side * step, tangent) for step in (1, 2, 3)]
                if path and not any(0 <= int(x) < kinds.shape[1] and 0 <= int(y) < kinds.shape[0]
                                    and kinds[int(y), int(x)] != 0 for x, y in forward):
                    paths.extend(path)
            if paths:
                proposals.append((peer_id, paths))
    if len(proposals) != 1 or legacy._intervening_body(point, (0, side) if axis else (side, 0), raw, peers) is not None:
        return None
    source = ('shared_single_green_pixel' if len(span) == 1 else
              'mixed_colour_span_boundary' if any(s['pigment'] == 2 for s in span) else
              'body_attached_red_green_transition')
    return dict(point=point, side=side, source=source,
                wire=dict(kind='observed_turning_short_colour_span', coordinate_img=list(point),
                          own_lead_pixels_img=[p for s in samples[first:index] for p in s['pixels']],
                          colour_span_pixels_img=[p for s in span for p in s['pixels']],
                          perpendicular_peer_pixels_img=proposals[0][1], peer_id=proposals[0][0],
                          body_geometry=geometry))


def _trace(kinds, geometry, side, length=96):
    ink = kinds != 0
    vertical = geometry['axis_y']
    edge = _body_edge(geometry, side)
    start = math.floor(edge) + 1 if side > 0 else math.ceil(edge) - 1
    tangent = geometry['tangent']
    if 'only_side' in geometry:
        widths = [band.get('width', abs(band['bounds'][3 - int(vertical)] - band['bounds'][1 - int(vertical)]) + 1)
                  if 'bounds' in band else band.get('width', 0) for band in geometry['body_bands_img']]
        width = max(widths)
        # A differently tinted antialias fringe can lie just outside the
        # dominant-colour bar bounds. It is still the bar, not a stem or a
        # gap: skip at most two broad, centred, physically connected rows.
        for _ in range(2):
            bars = [(a, b) for a, b in _line_runs(ink, start, tangent, vertical, max(6, width))
                    if a <= tangent <= b - 1 and b - a >= max(5, .6 * width)
                    and abs((a + b - 1) / 2 - tangent) <= 2]
            if len(bars) != 1:
                break
            start += side
    samples = []
    for step in range(length):
        pos = start + side * step
        runs = [(a, b) for a, b in _line_runs(ink, pos, tangent, vertical, 6)
                if a - 1 <= tangent <= b and b - a <= 12]
        if len(runs) != 1:
            break
        a, b = runs[0]
        centre = (a + b - 1) / 2
        if step < 4:
            if b - a > 4 or abs(centre - tangent) > 3:
                break
            tangent = centre
        elif b - a <= 4 and abs(centre - tangent) > 1.5:
            break
        cells = kinds[pos, a:b] if vertical else kinds[a:b, pos]
        # Use the strongest central stroke rather than its antialias fringe.
        t = min(b - 1, max(a, int(round(tangent))))
        pigment = int(kinds[pos, t] if vertical else kinds[t, pos])
        point = (float(tangent), float(pos)) if vertical else (float(pos), float(tangent))
        samples.append(dict(pos=pos, point=point, pigment=pigment,
                            pixels=[[int(q), pos] if vertical else [pos, int(q)] for q in range(a, b)], width=b - a))
    return samples


def _point(geometry, normal, tangent):
    return (float(tangent), float(normal)) if geometry['axis_y'] else (float(normal), float(tangent))


def _contact(kinds, geometry, side, nodes, legacy, raw, peers):
    samples = _trace(kinds, geometry, side)
    if len(samples) < 3:
        return None
    axis = 1 if geometry['axis_y'] else 0
    cross = 1 - axis
    direction = (0, side) if axis else (side, 0)
    base = samples[0]['pigment']
    first = 0
    body_band = geometry['body_bands_img'][0]
    if 'first' in body_band:
        body_pos = (body_band['first'] + body_band['last']) // 2
        body_pt = _point(geometry, body_pos, round(geometry['tangent']))
        body_pigment = int(kinds[int(body_pt[1]), int(body_pt[0])])
    else:
        body_pigment = base
    change = next((i for i, s in enumerate(samples) if s['pigment'] != base), None)
    neutral_start = next((i for i in range(change, min(change + 3, len(samples)))
                          if len(samples[i:i + 4]) == 4 and all(s['pigment'] == 7 for s in samples[i:i + 4])), None) if change is not None else None
    # A coloured body drawing can protrude a few pixels past its plates or
    # outline, then meet the actual black pin. That internal change is not
    # the pin/wire interface. Require a complete neutral stem and a later
    # chromatic connection; otherwise leave the role ambiguous.
    if (geometry['kind'] in {'parallel_capacitor_plates', 'closed_resistor_outline'}
            and base == body_pigment and base != 7 and neutral_start is not None):
        prefix_limit = max(4, min(12, .8 * (geometry['high'] - geometry['low'])))
        if neutral_start > prefix_limit + 2:
            return None
        first, base = neutral_start, 7
    # A filled node can broaden the ink beyond the narrow trace's window.
    # Its incoming face, rather than its centre, determines whether it is
    # immediately reached. Do not skip a white gap or an earlier wire colour.
    edge = _body_edge(geometry, side)
    tangent = samples[first]['point'][cross]
    near_nodes = sorted((n for n in nodes
                         if 3 <= (n['point'][axis] - edge) * side <= len(samples) + 8
                         and abs(n['point'][cross] - tangent) <= 1.5),
                        key=lambda n: n['point'][axis] * side)
    for node in near_nodes:
        incoming = node['bounds'][axis] if side > 0 else node['bounds'][axis + 2]
        gap = (incoming - samples[-1]['pos']) * side
        missing = (node['point'][axis] - samples[-1]['pos']) * side
        if not 0 <= gap <= (3 if node.get('large_blue') else 2) or not 0 < missing <= 8:
            continue
        additions = []
        for step in range(1, int(math.ceil(missing)) + 1):
            normal = samples[-1]['pos'] + side * step
            runs = [(a, b) for a, b in _line_runs(kinds != 0, normal, tangent, geometry['axis_y'], 6)
                    if a - 1 <= tangent <= b]
            if len(runs) != 1:
                break
            a, b = runs[0]
            t = min(b - 1, max(a, int(round(tangent))))
            pigment = int(kinds[normal, t] if axis else kinds[t, normal])
            additions.append(dict(pos=normal, point=_point(geometry, normal, tangent), pigment=pigment,
                                  pixels=[[q, normal] if axis else [normal, q] for q in range(a, b)],
                                  width=b - a))
        if len(additions) == int(math.ceil(missing)):
            samples.extend(additions)
        break
    prefix = 4
    if 'only_side' in geometry:
        first_change = next((i for i in range(first, len(samples)) if samples[i]['pigment'] != base), len(samples))
        if first_change - first in {2, 3} and any(
                n['bounds'][axis] - 2 <= samples[first_change]['pos'] <= n['bounds'][axis + 2] + 2
                for n in near_nodes):
            prefix = first_change - first
    if len(samples[first:first + prefix]) != prefix or any(s['pigment'] != base for s in samples[first:first + prefix]):
        return None
    tangent = samples[first + prefix - 1]['point'][cross]
    for index in range(first + prefix, len(samples)):
        sample = samples[index]
        node = next((n for n in near_nodes
                     if n['bounds'][axis] - 1 <= sample['pos'] <= n['bounds'][axis + 2] + 1), None)
        if node is None and sample['pigment'] != base:
            # A tiny blue dot can touch a text row after rasterisation. Its
            # filled 3x3 core remains visible, although the whole connected
            # pigment blob is too large. Independent other-colour wire arms
            # are mandatory for this small-core recovery.
            # Preserve the established direct-core path for a node already
            # reached by the trace. The wider search is only for a painted
            # interface that otherwise hides a small blue bulb.
            pigment = sample['pigment']
            for delta in (-1, 0, 1):
                for shift in (-1, 0, 1):
                    point = _point(geometry, sample['pos'] + side * delta, round(tangent) + shift)
                    x, y = map(int, point)
                    if x < 1 or y < 1 or x + 1 >= kinds.shape[1] or y + 1 >= kinds.shape[0]:
                        continue
                    if not np.all(kinds[y - 1:y + 2, x - 1:x + 2] == pigment):
                        continue
                    proposal = dict(point=point, bounds=(x - 1, y - 1, x + 1, y + 1), pigment=pigment)
                    arms = legacy._node_wire_arms(kinds == pigment, (kinds != 0) & (kinds != pigment),
                                                  proposal, (-direction[0], -direction[1]))
                    if any(arm['non_red_colour_hits'] >= 3 for arm in arms):
                        node = proposal
                        break
                if node is not None:
                    break
            proposals = []
            for delta in (() if node is not None else (-1, 0, 1, 2, 3)):
                for shift in (-1, 0, 1):
                    point = _point(geometry, sample['pos'] + side * delta, round(tangent) + shift)
                    x, y = map(int, point)
                    if x < 1 or y < 1 or x + 1 >= kinds.shape[1] or y + 1 >= kinds.shape[0]:
                        continue
                    pigment = int(kinds[y, x])
                    if pigment != 5 or not np.all(kinds[y - 1:y + 2, x - 1:x + 2] == pigment):
                        continue
                    proposal = dict(point=point, bounds=(x - 1, y - 1, x + 1, y + 1), pigment=pigment)
                    arms = legacy._node_wire_arms(kinds == pigment, (kinds != 0) & (kinds != pigment),
                                                  proposal, (-direction[0], -direction[1]))
                    if any(arm['non_red_colour_hits'] >= 3 for arm in arms):
                        proposals.append(proposal)
            if proposals:
                points = np.array([p['point'] for p in proposals])
                if np.max(np.ptp(points, axis=0)) <= 2:
                    low = np.min(points, axis=0).astype(int) - 1
                    high = np.max(points, axis=0).astype(int) + 1
                    node = dict(point=tuple(np.mean(points, axis=0)), bounds=tuple(map(int, (*low, *high))), pigment=5)
        if node is not None:
            # A continuous stem that directly enters a dot has no earlier
            # pin/wire interface. Its first actual junction supplies the centre.
            origin = _point(geometry, samples[first]['pos'], tangent)
            lead_mask = kinds == base
            if axis:
                lead_mask[node['bounds'][1]:node['bounds'][3] + 1, :] |= kinds[node['bounds'][1]:node['bounds'][3] + 1, :] == node['pigment']
            else:
                lead_mask[:, node['bounds'][0]:node['bounds'][2] + 1] |= kinds[:, node['bounds'][0]:node['bounds'][2] + 1] == node['pigment']
            # A direct differently coloured dot can have up to two painted
            # fringe cells at the incoming interface. Include actual ink
            # there only; an earlier wire segment remains a boundary.
            connector = []
            incoming_edge = node['bounds'][axis] if side > 0 else node['bounds'][axis + 2]
            for step in (1, 2):
                normal = incoming_edge - side * step
                for shift in (-1, 0, 1):
                    point = _point(geometry, normal, round(tangent) + shift)
                    x, y = map(int, point)
                    if (0 <= x < kinds.shape[1] and 0 <= y < kinds.shape[0]
                            and kinds[y, x] not in {0, 3, 7}):
                        lead_mask[y, x] = True
                        connector.append([x, y])
            path = legacy._continuous_red_segment(lead_mask, origin, node['point'], geometry['axis_y'])
            arms = legacy._node_wire_arms(kinds == node['pigment'], (kinds != 0) & (kinds != node['pigment']), node,
                                           (-direction[0], -direction[1]))
            if (path and arms and legacy._intervening_body(node['point'], direction, raw, peers) is None):
                # Coloured labels elsewhere do not make a grayscale circuit
                # eligible. A neutral stem and node need an actual coloured
                # wire arm; otherwise monochrome coordinate work is deferred.
                if base == node['pigment'] == 7 and not any(arm['non_red_colour_hits'] >= 3 for arm in arms):
                    return None
                # The caller skips already observed sides. A red dot missed
                # by the older body search still owns its exact centre here.
                return dict(point=node['point'], side=side, source='first_filled_junction_center',
                            wire=dict(kind='observed_stem_to_filled_junction', node_bounds_img=list(node['bounds']),
                                      coordinate_img=list(node['point']), node_pigment=PIGMENTS[node['pigment']],
                                      own_lead_pixels_img=path, junction_wire_arms=arms,
                                      node_connector_pixels_img=connector,
                                      internal_body_stem_pixels_img=[p for s in samples[:first] for p in s['pixels']],
                                      body_geometry=geometry))
            return None
        if sample['pigment'] == base:
            continue
        # An observed body can meet another red pin through only one to
        # three coloured cells. Filled resistor glyphs often prevented the
        # older body search from reaching this otherwise visible interface.
        if base == 1 and sample['pigment'] in {2, 3}:
            end = index
            while end < len(samples) and samples[end]['pigment'] == sample['pigment']:
                end += 1
            opposite = samples[end:end + 4]
            opposite_pixels = []
            valid_opposite = len(opposite) == 4
            for s in opposite:
                runs = [(a, b) for a, b in _line_runs(kinds == base, s['pos'], tangent, geometry['axis_y'], 6)
                        if a - 1 <= tangent <= b and b - a <= 4]
                if len(runs) != 1:
                    valid_opposite = False
                    break
                a, b = runs[0]
                opposite_pixels.extend([[q, s['pos']] if axis else [s['pos'], q] for q in range(a, b)])
            if 1 <= end - index <= 3 and valid_opposite:
                span = samples[index:end]
                normal = span[0]['pos'] if len(span) == 1 else (samples[index - 1]['pos'] + span[0]['pos']) / 2
                point = _point(geometry, normal, tangent)
                nearby=[]
                for peer_id,box in peers:
                    facing=box[axis]if side>0 else box[axis+2]
                    distance=(facing-point[axis])*side
                    if 0<=distance<=64 and box[cross]-2<=point[cross]<=box[cross+2]+2:
                        nearby.append((distance,box))
                if nearby:
                    closest=min(d for d,_ in nearby)
                    closest_boxes=[box for d,box in nearby if d<=closest+2]
                    if any(max(a[0],b[0])<min(a[2],b[2])and max(a[1],b[1])<min(a[3],b[3])
                           for i,a in enumerate(closest_boxes)for b in closest_boxes[i+1:]):
                        return None
                if legacy._intervening_body(point, direction, raw, peers) is None:
                    source = ('shared_single_green_pixel' if len(span) == 1 else
                              'mixed_colour_span_boundary' if sample['pigment'] == 2 else
                              'body_attached_red_green_transition')
                    return dict(point=point, side=side, source=source,
                                wire=dict(kind='observed_body_short_opposed_colour_span', coordinate_img=list(point),
                                          own_lead_pixels_img=[p for s in samples[first:index] for p in s['pixels']],
                                          colour_span_pixels_img=[p for s in span for p in s['pixels']],
                                          opposite_lead_pixels_img=opposite_pixels,
                                          body_geometry=geometry))
            if geometry['kind'] in {'single_power_bar', 'compact_ground_bars'}:
                turning = _short_turn_contact(kinds, geometry, side, samples, first, index, legacy, raw, peers)
                if turning is not None:
                    return turning
        # These contacts need an independently measured body and straight
        # stem. Short, blended spans retain the specialised existing path.
        for offset in (0, 1, 2):
            start = index + offset
            wire_samples = samples[start:start + 6]
            if len(wire_samples) != 6:
                continue
            other = wire_samples[0]['pigment']
            if other == base:
                continue
            # Differently coloured labels can touch the wire's raster row.
            # Measure the wire's own pigment, not the combined ink width.
            wire_cells = []
            valid_wire = True
            for s in wire_samples:
                runs = [(a, b) for a, b in _line_runs(kinds == other, s['pos'], tangent, geometry['axis_y'], 6)
                        if a - 1 <= tangent <= b and b - a <= 4]
                if len(runs) != 1:
                    valid_wire = False
                    break
                a, b = runs[0]
                wire_cells.extend([[q, s['pos']] if axis else [s['pos'], q] for q in range(a, b)])
            if not valid_wire:
                continue
            normal = (samples[index - 1]['pos'] + wire_samples[0]['pos']) / 2
            point = _point(geometry, normal, tangent)
            if legacy._intervening_body(point, direction, raw, peers) is not None:
                return None
            wire = legacy._wire_path(kinds == other, point, direction)
            if wire is None:
                continue
            return dict(point=point, side=side, source='lead_wire_colour_boundary',
                        wire=dict(kind='external_stem_wire_boundary', coordinate_img=list(point),
                                  lead_pigment=PIGMENTS[base], wire_pigment=PIGMENTS[other],
                                  own_lead_pixels_img=[p for s in samples[first:index] for p in s['pixels']],
                                  internal_body_stem_pixels_img=[p for s in samples[:first] for p in s['pixels']],
                                  wire_pixels_img=wire_cells,
                                  transition_span_img=[samples[index - 1]['point'], wire_samples[0]['point']],
                                  wire_path=wire, body_geometry=geometry))
        return None
    return None


def extend_terminals(components, original_pins, result, report, bgr):
    """Add only independently supported missing terminals to the colour path."""
    from . import passive_pins as legacy

    kinds = _pigments(bgr)
    if not np.any((kinds != 0) & (kinds != 7)):
        return result, report
    ink = kinds != 0
    blue, green, red = np.moveaxis(bgr.astype(np.int16), -1, 0)
    weak_mixture = (green - blue >= 8) & (np.abs(red - green) <= 25) & (green >= 40)
    weak_near = cv2.dilate(weak_mixture.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    nodes = _filled_nodes(kinds)
    geometries, raw_boxes = {}, {}
    for comp in components:
        cid, typ = comp.get('id'), comp.get('type')
        box = legacy._box(comp.get('bbox_img'))
        if box is None or box[2] <= 0 or box[3] <= 0 or box[0] >= kinds.shape[1] or box[1] >= kinds.shape[0]:
            continue
        raw = legacy._box(comp.get('bbox_raw_img')) or legacy._box(comp.get('bbox_img'))
        if raw is None:
            continue
        raw_boxes[cid] = raw
        if typ in {'r', 'c'}:
            fits = [fit for vertical in (False, True)
                    if (fit := _axial_body(ink, raw, typ, vertical)) is not None
                    and all(len(_trace(kinds, fit, side, 4)) == 4 for side in (-1, 1))]
            if len(fits) == 1:
                geometries[cid] = fits
        elif typ in {'gnd', 'v', 'vcc'}:
            h, w = kinds.shape
            left, top = max(0, int(raw[0] - 24)), max(0, int(raw[1] - 24))
            right, bottom = min(w, math.ceil(raw[2] + 25)), min(h, math.ceil(raw[3] + 25))
            local_raw = (raw[0] - left, raw[1] - top, raw[2] - left, raw[3] - top)
            local = kinds[top:bottom, left:right]
            fits = _ground_geometries(local, local_raw, legacy) if typ == 'gnd' else []
            if not fits:
                fits = _single_port_geometries(local, local_raw, typ)
            for fit in fits:
                normal, tangent = (top, left) if fit['axis_y'] else (left, top)
                fit['low'] += normal
                fit['high'] += normal
                fit['tangent'] += tangent
                for band in fit['body_bands_img']:
                    if 'bounds' in band:
                        band['bounds'] = [v + (left if i % 2 == 0 else top) for i, v in enumerate(band['bounds'])]
                    else:
                        band['first'] += normal
                        band['last'] += normal
                        band['tangent'] += tangent
            if len(fits) == 1:
                geometries[cid] = fits
    measured = {}
    for comp in components:
        cid, typ = comp.get('id'), comp.get('type')
        if cid not in geometries or cid not in result:
            continue
        geo = geometries[cid][0]
        old = report.get(cid, {})
        if old and old['axis_y'] != geo['axis_y'] and typ in legacy.AXIAL_TYPES:
            continue
        axis = 1 if geo['axis_y'] else 0
        middle = (geo['low'] + geo['high']) / 2
        retained = [dict(point=tuple(point), side=1 if point[axis] > middle else -1,
                         source=source, wire=proof)
                    for point, source, proof in zip(old.get('after', []), old.get('sources', []),
                                                   old.get('connection_evidence', []))]
        peers = [(other, raw) for other, raw in raw_boxes.items() if other != cid]
        options = []
        for side in (geo['only_side'],) if 'only_side' in geo else (-1, 1):
            if any(r['side'] == side for r in retained):
                continue
            contact = _contact(kinds, geo, side, nodes, legacy, raw_boxes[cid], peers)
            if contact is not None:
                options.append(contact)
        measured[cid] = retained + options

    # Symmetry is a fallback for an otherwise unmarked connected stem, after
    # every observed boundary and junction has had priority.
    from . import passive_terminal_templates
    passive_terminal_templates.extend_unmarked(
        components, kinds, geometries, raw_boxes, measured, weak_near, legacy,
        weak_pixels=weak_mixture, nodes=nodes)

    for comp in components:
        cid, typ = comp.get('id'), comp.get('type')
        contacts = measured.get(cid, [])
        if typ not in {'r', 'c'} or len(contacts) != 1:
            continue
        geo = geometries[cid][0]
        anchor = contacts[0]
        if anchor['source'] in {'symmetric_opposed_terminal', 'peer_shared_terminal', 'same_symbol_terminal_length'}:
            continue
        side = -anchor['side']
        axis = 1 if geo['axis_y'] else 0
        cross = 1 - axis
        reflected = list(anchor['point'])
        reflected[axis] = geo['low'] + geo['high'] - reflected[axis]
        samples = _trace(kinds, geo, side)
        edge = geo['high'] if side > 0 else geo['low']
        distance = (reflected[axis] - edge) * side
        if (distance < 4 or distance > 48 or not samples
                or abs(reflected[cross] - samples[0]['point'][cross]) > 1.5):
            continue
        prior = [s for s in samples if (s['pos'] - reflected[axis]) * side <= .5]
        # A colour interface or filled node before the reflection must be
        # handled by its observed coordinate, never by an equal-length guess.
        if (len(prior) < distance - 1 or any(s['pigment'] != prior[0]['pigment'] for s in prior)
                or any(weak_near[y, x] for s in prior for x, y in s['pixels'])
                or any(n['bounds'][axis] - 1 <= reflected[axis] <= n['bounds'][axis + 2] + 1
                       and abs(n['point'][cross] - reflected[cross]) <= 2 for n in nodes)
                or any(0 < (n['point'][axis] - edge) * side < distance
                       and abs(n['point'][cross] - reflected[cross]) <= 2 for n in nodes)):
            continue
        peers = []
        for peer, fits in geometries.items():
            if peer == cid:
                continue
            pg = fits[0]
            if pg['axis_y'] != geo['axis_y'] or abs(pg['tangent'] - reflected[cross]) > 2:
                continue
            facing_edge = pg['low'] if side > 0 else pg['high']
            if not 3 <= (facing_edge - reflected[axis]) * side <= 48:
                continue
            if 'only_side' in pg and pg['only_side'] != -side:
                continue
            if 'only_side' not in pg:
                peer_contacts = measured.get(peer, [])
                facing = [r['point'] for r in peer_contacts if r['side'] == -side]
                opposite = [r['point'] for r in peer_contacts if r['side'] == side]
                if not facing and len(opposite) == 1:
                    reflected_peer = list(opposite[0])
                    reflected_peer[axis] = pg['low'] + pg['high'] - reflected_peer[axis]
                    facing = [reflected_peer]
                if len(facing) != 1 or math.dist(facing[0], reflected) > 1.5:
                    continue
            origin = _point(geo, samples[0]['pos'], samples[0]['point'][cross])
            target = _point(geo, math.floor(facing_edge) if side > 0 else math.ceil(facing_edge), reflected[cross])
            path = legacy._continuous_red_segment(kinds == prior[0]['pigment'], origin, target, geo['axis_y'])
            if path and legacy._intervening_body(reflected, (0, side) if axis else (side, 0),
                                                raw_boxes[cid], [(k, b) for k, b in raw_boxes.items() if k not in {cid, peer}]) is None:
                peers.append((peer, path))
        if len(peers) != 1:
            continue
        peer, path = peers[0]
        inferred = dict(point=tuple(reflected), side=side, source='symmetric_opposed_terminal',
                        wire=dict(kind='equal_opposed_lead_lengths', coordinate_img=reflected,
                                  anchor_point_img=list(anchor['point']), anchor_source=anchor['source'],
                                  measured_lead_length=distance, body_geometry=geo,
                                  own_lead_pixels_img=path, peer_id=peer))
        contacts.append(inferred)
        # The same physical contact belongs to a directly facing one-port
        # peer. Its measured bars and uninterrupted stem establish ownership.
        peer_geo = geometries[peer][0]
        if 'only_side' in peer_geo and not measured.get(peer):
            measured[peer] = [dict(point=tuple(reflected), side=-side, source='peer_shared_terminal',
                                  wire=dict(kind='direct_peer_symmetric_contact', coordinate_img=reflected,
                                            peer_id=cid, anchor_source=inferred['source'],
                                            body_geometry=peer_geo, own_lead_pixels_img=path))]

    for comp in components:
        cid, typ = comp.get('id'), comp.get('type')
        contacts = measured.get(cid, [])
        if (not contacts or (not any(r['source'] in NEW_SOURCES for r in contacts)
                             and len(contacts) <= len(report.get(cid, {}).get('after', [])))):
            continue
        if typ in legacy.SINGLE_TYPES and len(contacts) != 1:
            continue
        if any(sum(r['side'] == side for r in contacts) > 1 for side in (-1, 1)):
            continue
        contacts.sort(key=lambda r: r['side'])
        new = [r['point'] for r in contacts]
        report[cid] = dict(type=typ, before=original_pins.get(cid, []), after=new,
                           sources=[r['source'] for r in contacts], axis_y=geometries[cid][0]['axis_y'],
                           connection_evidence=[r['wire'] for r in contacts], coordinate_rule='image_color_terminal_rules')
        result[cid] = new
    return result, report
