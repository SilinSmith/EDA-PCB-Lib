"""Recover a compact caption using an observed prefix and isolated digits.

The full word's actual CTC tokens supply case-sensitive prefix evidence.
Only a real blank gap can open the independent numeric read. No ambiguous
letter is converted to a digit, and no expected identity is supplied.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .mono_pin_crops import _finite, _groups


PADDINGS = (2, 4)
THRESHOLD = .90
METHOD = 'literal_ctc_prefix_and_separated_original_numeric_suffix'


def _certainty(value):
    try:
        confidence = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return confidence if math.isfinite(confidence) and THRESHOLD <= confidence <= 1 else None


def _prefix(views, allowed):
    if not isinstance(views, list) or len(views) != 2:
        return None
    observed = []
    for padding, view in zip((4, 6), views):
        if not isinstance(view, dict) or view.get('padding') != padding:
            return None
        tokens, word = view.get('ctc_tokens'), view.get('text')
        if (not isinstance(tokens, list) or not tokens or not isinstance(word, str)
                or any(not isinstance(t, dict) or not isinstance(t.get('text'), str) for t in tokens)
                or ''.join(t['text'] for t in tokens) != word):
            return None
        # This conservative route handles one physically separated letter
        # followed by numeric glyphs. Multi-letter unit identities retain
        # the complete-word route until their segmentation is established.
        first = tokens[0]
        if (len(first['text']) != 1 or not first['text'].isascii()
                or first['text'].upper() not in allowed
                or _certainty(first.get('confidence')) is None):
            return None
        observed.append((first['text'], _certainty(first['confidence'])))
    return (observed[0][0], min(p[1] for p in observed)) if observed[0][0] == observed[1][0] else None


def _suffix_conflicts(views, prefix, digits):
    for view in views:
        tail = view['ctc_tokens'][1:]
        if len(tail) != len(digits) or not all(len(t['text']) == 1 and t['text'].isascii()
                                               and t['text'].isalnum() for t in tail):
            return True
        if any(t['text'] != digit and _certainty(t.get('confidence')) is not None
               for t, digit in zip(tail, digits)):
            return True
    return False


def _suffix_geometry(gray, bgr, box, body):
    caption, frame = _finite(box, 4), _finite(body, 4)
    if caption is None or frame is None:
        return None
    a, b, c, d = int(math.floor(caption[0])), int(math.floor(caption[1])), int(math.ceil(caption[2])), int(math.ceil(caption[3]))
    if not (0 <= a < c <= gray.shape[1] and 0 <= b < d <= gray.shape[0]):
        return None
    pixels, colors = gray[b:d, a:c], bgr[b:d, a:c]
    chroma = colors.max(2).astype(np.int16)-colors.min(2).astype(np.int16) if colors.ndim == 3 else np.zeros(pixels.shape, np.int16)
    dark = (pixels < 190) & (chroma < 30)
    spans = _groups(dark.any(0), 0)
    if not 2 <= len(spans) <= 5:
        return None
    left_end, right_start = spans[0][1], spans[1][0]
    if not 2 <= right_start-left_end <= min(10, d-b):
        return None
    def envelope(x0, x1):
        yy, xx = np.nonzero(dark[:, x0:x1])
        return [a+x0+int(xx.min()), b+int(yy.min()), a+x0+int(xx.max())+1, b+int(yy.max())+1]
    prefix = envelope(*spans[0])
    suffix = envelope(right_start, spans[-1][1])
    if min(prefix[3]-prefix[1], suffix[3]-suffix[1]) < 3:
        return None
    number = [max(prefix[2], suffix[0]-2), max(0, suffix[1]-2),
              min(gray.shape[1], suffix[2]+2), min(gray.shape[0], suffix[3]+2)]
    if not (caption[3] <= frame[1] and number[3] < frame[1]
            or caption[1] >= frame[3] and number[1] > frame[3]):
        return None
    xa, ya, xb, yb = number
    if max(yb-ya, xb-xa)+2*max(PADDINGS) > 384:
        return None
    local, rgb = gray[ya:yb, xa:xb], bgr[ya:yb, xa:xb]
    chroma = rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16) if rgb.ndim == 3 else np.zeros(local.shape, np.int16)
    ink = (local < 190) & (chroma < 30)
    outside = np.ones(ink.shape, bool)
    outside[max(0,b-ya):min(yb-ya,d-ya), max(0,a-xa):min(xb-xa,c-xa)] = False
    if np.any(ink & outside):
        return None
    return dict(prefix_body_bbox=prefix, suffix_body_bbox=suffix, suffix_bbox=number,
                caption_bbox=list(caption), component_bbox=list(frame), blank_gap_pixels=right_start-left_end)


def read_segmented_caption(gray, bgr, box, body, views, provider, allowed):
    observed_prefix = _prefix(views, allowed)
    geometry = _suffix_geometry(gray, bgr, box, body) if observed_prefix else None
    if geometry is None:
        return None
    a, b, c, d = geometry['suffix_bbox']
    digits_views = [dict(provider.read_glyph(np.pad(gray[b:d, a:c], p, constant_values=255)), padding=p)
                    for p in PADDINGS]
    values = [v.get('text') for v in digits_views]
    certainties = [_certainty(v.get('confidence')) for v in digits_views]
    if (not all(isinstance(t, str) and re.fullmatch(r'[0-9]{1,4}', t) for t in values)
            or values[0] != values[1] or not all(certainties)):
        return None
    prefix, confidence = observed_prefix
    if _suffix_conflicts(views, prefix, values[0]):
        return None
    return dict(geometry, method=METHOD, prefix=prefix, suffix=values[0],
                suffix_paddings=list(PADDINGS), suffix_reads=digits_views,
                confidence=min(confidence, *certainties), unchanged_source_pixels=True)


def qualified_segmented_caption(metadata, text, confidence):
    """Validate both original reading branches before literal assignment."""
    split = metadata.get('segmented_caption')
    if not isinstance(split, dict) or split.get('method') != METHOD or split.get('unchanged_source_pixels') is not True:
        return None
    prefix, digits = split.get('prefix'), split.get('suffix')
    if (not isinstance(prefix, str) or not isinstance(digits, str)
            or not re.fullmatch(r'[A-Za-z][0-9]{1,4}', text)
            or prefix+digits != text or split.get('suffix_paddings') != list(PADDINGS)):
        return None
    full_views = metadata.get('raw_reads')
    observed = _prefix(full_views, {prefix.upper()})
    if observed is None or observed[0] != prefix or _suffix_conflicts(full_views, prefix, digits):
        return None
    views = split.get('suffix_reads')
    if not isinstance(views, list) or len(views) != 2:
        return None
    certainties = [observed[1]]
    for padding, view in zip(PADDINGS, views):
        if not isinstance(view, dict) or view.get('padding') != padding or view.get('text') != digits:
            return None
        value = _certainty(view.get('confidence'))
        if value is None:
            return None
        certainties.append(value)
    boxes = [_finite(split.get(k), 4) for k in ('caption_bbox','prefix_body_bbox','suffix_body_bbox','suffix_bbox','component_bbox')]
    if any(b is None or b[0] >= b[2] or b[1] >= b[3] for b in boxes):
        return None
    caption, first, last, number, body = boxes
    if (caption != _finite(metadata.get('bbox'), 4) or first[2] >= last[0]
            or number[0] < first[2] or number[0] > last[0] or number[2] < last[2]
            or number[1] > last[1] or number[3] < last[3]
            or any(first[i] < caption[i] or last[i] < caption[i] for i in (0,1))
            or any(first[i] > caption[i] or last[i] > caption[i] for i in (2,3))
            or any(abs(number[i]-last[i]) > 2 for i in range(4))
            or not (caption[3] <= body[1] and number[3] < body[1]
                    or caption[1] >= body[3] and number[1] > body[3])
            or confidence != min(certainties) or split.get('confidence') != confidence):
        return None
    return text
