"""Independent local readings for weak or missing non-IC designators.

This reuses the verified single-crop PP-OCR backend without changing IC text
processing. Only two agreeing, high-confidence views of source caption pixels
can add a designator candidate. Unassigned captions must have one geometric
owner. It does not infer a label from other labels.
"""
from __future__ import annotations

import math
import re
import hashlib

import cv2
import numpy as np

from .pin_name_ocr import get_pin_name_ocr

PREFIXES = {'r': 'R', 'c': 'C', 'l': 'L', 'd': 'D', 'led': 'LED', 'fuse': 'F'}
PRIMARY_LIMIT = .85
SECONDARY_MIN = .97
SMALL_CAPTION_PRIMARY_MIN = .90


def _normalized(value, prefix):
    if not isinstance(value, str):
        return None
    text = re.sub(r'\s+', '', value).upper()
    return text if re.fullmatch(re.escape(prefix) + r'[0-9]{1,5}', text) else None


def _box(value):
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 and all(math.isfinite(v) for v in values) and values[2] > values[0] and values[3] > values[1] else None


def _distance(first, second):
    return math.hypot(max(first[0] - second[2], second[0] - first[2], 0),
                      max(first[1] - second[3], second[1] - first[3], 0))


def _unassigned_caption_boxes(doc, cid):
    """Recover an unowned caption only when its detector geometry has one owner."""
    # Match the same boxes and centre-distance ownership as stage2.assign.
    # Ground/power, connectors and ICs can also be the nearest symbol.
    bodies = [(c.get('id'), _box(c.get('bbox_img')))
              for c in doc.get('components', [])]
    bodies = [(pid, box) for pid, box in bodies if box is not None and pid is not None]
    own = next((box for pid, box in bodies if pid == cid), None)
    if own is None:
        return []

    def unique_owner(box):
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for region in (box, (*centre, *centre)):
            ranked = sorted(((_distance(region, body), pid) for pid, body in bodies), key=lambda pair: pair[0])
            if ranked[0][1] != cid or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= 2):
                return False
        return True

    boxes = []
    for text in doc.get('texts', []):
        box = _box(text.get('bbox_img') if isinstance(text, dict) else text)
        if box is not None:
            boxes.append(box)
    candidates = []
    for box in boxes:
        distance = _distance(box, own)
        if distance > max(48, 1.8 * max(own[2] - own[0], own[3] - own[1])):
            continue
        if not unique_owner(box):
            continue
        # A detector may split the last digit from an otherwise complete word.
        # Join only immediately adjacent text boxes on the same physical row.
        merged = list(box)
        for _ in range(2):
            additions = []
            for peer in boxes:
                if peer == box:
                    continue
                mh, ph = merged[3] - merged[1], peer[3] - peer[1]
                if min(mh, ph) < .65 * max(mh, ph) or abs((merged[1] + merged[3] - peer[1] - peer[3]) / 2) > max(2, .25 * min(mh, ph)):
                    continue
                gap = max(peer[0] - merged[2], merged[0] - peer[2], 0)
                if gap > max(3, .45 * min(mh, ph)):
                    continue
                additions.append(peer)
            updated = [min([merged[0]] + [b[0] for b in additions]), min([merged[1]] + [b[1] for b in additions]),
                       max([merged[2]] + [b[2] for b in additions]), max([merged[3]] + [b[3] for b in additions])]
            if updated == merged:
                break
            merged = updated
        if not unique_owner(merged):
            continue
        if tuple(merged) not in [value[1] for value in candidates]:
            candidates.append((distance, tuple(merged)))
    return [box for _, box in sorted(candidates)[:4]]


def _retry_small_cap_caption(glyph, colour_glyph, provider, text, threshold, trace):
    """Fixed small-text views; every read must corroborate the original text.

    Blank crop margins can make a ten-pixel caption smaller inside PP-OCR's
    input. Locate the blue text core and keep its original rectangular pixels,
    then try two margins and, if necessary,
    the established 32/48px recognition heights. Never choose a different
    identifier or select the maximum-confidence view from a search.
    """
    color = colour_glyph.astype(np.int16)
    blue = ((color[:, :, 0] - color[:, :, 1] >= 20)
            & (color[:, :, 0] - color[:, :, 2] >= 20) & (glyph < 210))
    dark = glyph < 170
    if np.count_nonzero(blue) < 8 or np.count_nonzero(blue & dark) < .8 * np.count_nonzero(dark):
        return None
    ys, xs = np.nonzero(blue)
    if not len(xs):
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    if not (0 < x0 < x1 < glyph.shape[1] and 0 < y0 < y1 < glyph.shape[0]
            and 6 <= y1 - y0 <= 16 and .75 <= (x1 - x0) / (y1 - y0) <= 6):
        return None
    tight = glyph[y0:y1, x0:x1].copy()
    trace['small_caption_normalization'] = dict(
        foreground_bbox_local=[x0, y0, x1, y1], foreground_threshold=210,
        foreground_colour='blue', blue_channel_difference_minimum=20, dark_ink_blue_fraction_minimum=.8,
        foreground_shape=list(tight.shape), foreground_sha256=hashlib.sha256(tight.tobytes()).hexdigest(),
        interpolation='cubic', normalized_heights=[32, 48],
        original_view_minimum=SMALL_CAPTION_PRIMARY_MIN,
        rule='all original and retry reads must agree; two accepted views each meet the original threshold')
    trace['retry_views'] = []
    for family in ('tight_original_pixels', 'scale_normalized_caption'):
        views = []
        for size in (2, 4) if family == 'tight_original_pixels' else (32, 48):
            pad = size if family == 'tight_original_pixels' else 2
            view = tight if family == 'tight_original_pixels' else cv2.resize(
                tight, (round(tight.shape[1] * size / tight.shape[0]), size), interpolation=cv2.INTER_CUBIC)
            if max(view.shape) + 2 * pad > 384:
                return None
            reading = dict(provider.read_glyph(np.pad(view, pad, constant_values=255)),
                           view_family=family, padding=pad, target_foreground_height=view.shape[0])
            views.append(reading)
            trace['retry_views'].append(reading)
        readings = [_normalized(v.get('text'), 'C') for v in views]
        confidences = [float(v['confidence']) for v in views]
        if readings != [text, text] or not all(math.isfinite(v) and 0 <= v <= 1 for v in confidences):
            trace['held_reason'] = 'small_caption_retry_disagrees'
            return None
        if min(confidences) >= threshold:
            trace['accepted_view_family'] = family
            return min(confidences)
    return None


def read_non_ic_captions(doc, assigned, bgr, *, min_conf=0., mode='auto'):
    """Return additional OCR candidates and a trace of accepted/held views."""
    if mode not in {'off', 'auto', 'on'}:
        raise ValueError('non-IC caption OCR mode must be off, auto or on')
    if mode == 'off' or bgr is None:
        return [], []
    min_conf = float(min_conf)
    if not math.isfinite(min_conf):
        raise ValueError('min_conf must be finite')
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        return [], []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ic_boxes = [c['bbox_img'] for c in doc.get('components', []) if c.get('type') == 'box' and c.get('bbox_img')]
    pending = []
    for comp in doc.get('components', []):
        prefix = PREFIXES.get(comp.get('type'))
        if prefix is None:
            continue
        cid = comp.get('id')
        primary = (assigned.get('comps', {}).get(cid, {}) or {}).get('designator') or {}
        try:
            confidence = float(primary.get('conf', 0.))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(confidence) or confidence >= PRIMARY_LIMIT:
            continue
        selected = _box(primary.get('box')) if primary.get('text') else None
        boxes = [selected] if selected is not None else _unassigned_caption_boxes(doc, cid)
        for box in boxes:
            x0, y0, x1, y1 = map(round, box)
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                    and min(x1 - x0, y1 - y0) >= 6 and max(x1 - x0, y1 - y0) + 8 <= 384):
                continue
            if any(min(x1, b[2]) > max(x0, b[0]) and min(y1, b[3]) > max(y0, b[1]) for b in ic_boxes):
                continue
            # Detector text boxes can cut the outer stroke. Recover actual
            # source pixels first; white margins cannot repair a clipped glyph.
            gx0, gy0, gx1, gy1 = max(0, x0 - 2), max(0, y0 - 2), min(w, x1 + 2), min(h, y1 + 2)
            if any(min(gx1, b[2]) > max(gx0, b[0]) and min(gy1, b[3]) > max(gy0, b[1]) for b in ic_boxes):
                continue
            glyph = gray[gy0:gy1, gx0:gx1].copy()
            if max(glyph.shape) + 8 > 384:
                continue
            ink = glyph < 170
            if not ink.any():
                continue
            trace = dict(component_id=cid, type=comp['type'], bbox=list(box), glyph_bbox=[gx0, gy0, gx1, gy1],
                         primary_text=primary.get('text', ''), primary_confidence=confidence,
                         intended_role='designator', views=[], accepted=False,
                         ownership_source='existing_assignment' if selected else 'unique_nearest_caption_geometry')
            if ink[0].any() or ink[-1].any() or ink[:, 0].any() or ink[:, -1].any():
                trace['held_reason'] = 'source_glyph_touches_crop_border'
                pending.append((None, prefix, trace))
            else:
                pending.append((glyph, prefix, trace))
    if not pending:
        return [], []
    provider = get_pin_name_ocr(mode) if any(glyph is not None for glyph, _, _ in pending) else None
    extras, audit = [], []
    threshold = max(SECONDARY_MIN, min_conf)
    for glyph, prefix, trace in pending:
        audit.append(trace)
        if glyph is None:
            continue
        if provider is None:
            trace['held_reason'] = 'optional_provider_unavailable'
            continue
        try:
            for pad in (2, 4):
                view = provider.read_glyph(np.pad(glyph, pad, constant_values=255))
                trace['views'].append(dict(view, padding=pad))
            texts = [_normalized(v.get('text'), prefix) for v in trace['views']]
            confidences = [float(v['confidence']) for v in trace['views']]
        except Exception as exc:
            trace['held_reason'] = 'secondary_recognition_failed'
            trace['error'] = type(exc).__name__ + ': ' + str(exc)
            continue
        valid = (len(texts) == 2 and texts[0] is not None and texts[0] == texts[1]
                 and all(math.isfinite(v) and 0 <= v <= 1 for v in confidences))
        confidence = min(confidences) if valid else None
        if not valid or confidence < threshold:
            trace['held_reason'] = 'views_disagree_or_below_threshold'
            # Only small capacitor captions whose original reads already
            # agree may retry. Other symbol types and conflicting reads stay
            # on the existing path.
            if valid and trace['type'] == 'c' and confidence >= SMALL_CAPTION_PRIMARY_MIN:
                try:
                    gx0, gy0, gx1, gy1 = trace['glyph_bbox']
                    confidence = _retry_small_cap_caption(
                        glyph, bgr[gy0:gy1, gx0:gx1], provider, texts[0], threshold, trace)
                except Exception as exc:
                    trace['held_reason'] = 'secondary_recognition_failed'
                    trace['error'] = type(exc).__name__ + ': ' + str(exc)
                    confidence = None
            else:
                confidence = None
            if confidence is None:
                continue
        text = texts[0]
        if text == _normalized(trace['primary_text'], prefix):
            trace['held_reason'] = 'primary_already_agrees'
            continue
        trace.update(accepted=True, text=text, confidence=confidence,
                     confidence_definition='minimum of the two accepted PP-OCR character-probability means')
        trace.pop('held_reason', None)
        extras.append(dict(bbox=list(trace['bbox']), reads={'designator': (text, confidence)},
                           source='non_ic_caption_ppocr', non_ic_caption_ocr=trace))
    # An unassigned component with two distinct nearby identifiers remains
    # ambiguous. A highest-score guess must not invent caption ownership.
    for cid in {row['component_id'] for row in audit if row['ownership_source'] != 'existing_assignment'}:
        candidates = [row for row in audit if row['component_id'] == cid and row['accepted']]
        if len({row['text'] for row in candidates}) > 1:
            for row in candidates:
                row.update(accepted=False, held_reason='multiple_caption_regions_disagree')
    extras = [row for row in extras if row['non_ic_caption_ocr']['accepted']]
    return extras, audit
