"""Keep a complete detector word when old boxes are its aligned fragments.

This does not relax merge_boxes' area gate. A second detector must already
have grouped the original ink into one compact row, and at least two old
boxes must support its span. Old boxes are retained for downstream choice.
"""
from __future__ import annotations

import math


def _box(value):
    try:
        vals = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (len(vals) != 4 or not all(math.isfinite(x) for x in vals)
            or vals[2] <= vals[0] or vals[3] <= vals[1]):
        return None
    return vals


def recover_fragmented_boxes(base, det, *, limit=64):
    """Return extra complete words supported by distinct old row fragments.

    Coordinates alone cannot prove semantics. These conservative scale,
    alignment and coverage gates verify the same detected text row rather
    than admitting every larger/new detector rectangle.
    """
    if limit <= 0:
        return []
    old = list(dict.fromkeys(b for x in base if (b := _box(x)) is not None))
    additions = []
    for raw in det:
        d = _box(raw)
        if d is None or d in old or d in additions:
            continue
        w, h = d[2] - d[0], d[3] - d[1]
        if not (5 <= h <= 48 and 1.4 <= w / h <= 8):
            continue
        fragments = []
        for b in old:
            bw, bh = b[2] - b[0], b[3] - b[1]
            if not (.55 * h <= bh <= 1.35 * h and bw <= .70 * w):
                continue
            overlap = (max(0, min(d[2], b[2]) - max(d[0], b[0]))
                       * max(0, min(d[3], b[3]) - max(d[1], b[1])))
            if (overlap < .90 * bw * bh
                    or abs((b[1] + b[3] - d[1] - d[3]) / 2) > .35 * h):
                continue
            fragments.append(b)
        if not 2 <= len(fragments) <= 8:
            continue
        fragments.sort(key=lambda b: (b[0], b[1]))
        # Distinct spatial pieces, not duplicate crops of the same glyph.
        gaps = [b[0] - a[2] for a, b in zip(fragments, fragments[1:])]
        if any(gap < 1 or gap > 2.25 * h for gap in gaps):
            continue
        span = fragments[-1][2] - fragments[0][0]
        if span < .75 * w or d[0] < fragments[0][0] - .5 * h or d[2] > fragments[-1][2] + .5 * h:
            continue
        additions.append(d)
        if len(additions) >= limit:
            break
    return additions
