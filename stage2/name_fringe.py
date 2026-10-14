"""A bounded original-pixel view retaining the pale edge of a complete name."""
from __future__ import annotations

import math

import numpy as np

from .mono_pin_crops import _finite


FRINGE_THRESHOLD = 230
FRINGE_PADDINGS = (0, 2)


def antialiased_name_candidate(candidate, gray, bgr):
    """Retain at most one pale raster pixel around a measured dark word.

    A new dark body outside the word, an unresolved crop, line cleanup or
    contact with the physical ownership boundary cannot use this retry.
    No source pixel is erased or synthesized. The row and pin remain fixed.
    """
    evidence = candidate.get('cleanup_evidence') or {}
    if (candidate.get('role') != 'pin_name' or candidate.get('flags')
            or evidence.get('integrity_unresolved')
            or evidence.get('known_line_exclusions') or candidate.get('recognition_view')):
        return None
    core = _finite(candidate.get('bbox'), 4)
    cell = _finite(evidence.get('physical_search_region'), 4)
    if core is None or cell is None or any(int(v) != v for v in core):
        return None
    x0, y0, x1, y1 = map(int, core)
    xa, ya = max(0, math.ceil(cell[0]), x0-1), max(0, math.ceil(cell[1]), y0-1)
    xb, yb = min(gray.shape[1], math.floor(cell[2]), x1+1), min(gray.shape[0], math.floor(cell[3]), y1+1)
    if not (xa <= x0 < x1 <= xb and ya <= y0 < y1 <= yb):
        return None
    pixels = gray[ya:yb, xa:xb]
    colors = bgr[ya:yb, xa:xb]
    chroma = colors.max(2).astype(np.int16) - colors.min(2).astype(np.int16)
    pale = (pixels < FRINGE_THRESHOLD) & (chroma < 30)
    outside = np.ones(pale.shape, bool)
    outside[y0-ya:y1-ya, x0-xa:x1-xa] = False
    if np.any((pixels < 190) & (chroma < 30) & outside):
        return None
    yy, xx = np.nonzero(pale)
    if not len(xx) or not np.any(pale & outside):
        return None
    box = [min(x0, xa+int(xx.min())), min(y0, ya+int(yy.min())),
           max(x1, xa+int(xx.max())+1), max(y1, ya+int(yy.max())+1)]
    # A retry must stay inside the same measured row cell with a real margin.
    if any(box[i] <= cell[i] for i in (0, 1)) or any(box[i] >= cell[i] for i in (2, 3)):
        return None
    fringe = dict(method='one_pixel_original_pale_name_fringe', core_bbox=list(core),
                  bbox=box, physical_search_region=list(cell), ink_threshold=FRINGE_THRESHOLD,
                  added_pale_pixels=int(np.sum(pale & outside)),
                  new_dark_body_pixels=0, unchanged_source_pixels=True)
    return dict(candidate, bbox=box, geometry_bbox=box, glyph_bbox=box,
                region=[max(int(cell[0]), box[0]-1), max(int(cell[1]), box[1]-1),
                        min(int(cell[2]), box[2]+1), min(int(cell[3]), box[3]+1)],
                recognition_view='original_antialiased_complete_name',
                cleanup_evidence=dict(evidence, name_fringe=fringe))


def qualified_fringe_view(metadata, box):
    """Recheck the narrow-padding authorization at assignment's boundary."""
    evidence = (metadata.get('cleanup_evidence') or {}).get('name_fringe') or {}
    core, fringe, cell = (_finite(evidence.get(key), 4)
                          for key in ('core_bbox', 'bbox', 'physical_search_region'))
    return (metadata.get('role') == 'pin_name'
            and metadata.get('view') == 'original_antialiased_complete_name'
            and core is not None and fringe is not None and cell is not None
            and fringe == _finite(box, 4)
            and all(0 <= core[i]-fringe[i] <= 1 for i in (0, 1))
            and all(0 <= fringe[i]-core[i] <= 1 for i in (2, 3))
            and fringe != core
            and all(fringe[i] > cell[i] for i in (0, 1))
            and all(fringe[i] < cell[i] for i in (2, 3))
            and evidence.get('method') == 'one_pixel_original_pale_name_fringe'
            and evidence.get('ink_threshold') == FRINGE_THRESHOLD
            and evidence.get('unchanged_source_pixels') is True
            and evidence.get('new_dark_body_pixels') == 0
            and isinstance(evidence.get('added_pale_pixels'), int)
            and evidence['added_pale_pixels'] > 0)
