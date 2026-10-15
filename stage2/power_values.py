"""Publish power-symbol values from owned OCR, without moving pins or nets."""
from __future__ import annotations

import math
import re

POWER_TYPES = frozenset({'gnd', 'v', 'vcc', 'vdd', 'power'})
_RAIL = re.compile(r'(?:[A-Z]*GND[A-Z0-9_]*|V(?:CC|DD|SS|BAT|BUS|IN|OUT|REF|USB|AA|EE)[A-Z0-9_]*(?:\.[A-Z0-9_]+)*|[+-]?\d+(?:\.\d+)?V\d*)', re.I)


def _accepted(text, confidence, ctype):
    if not isinstance(text, str) or not math.isfinite(float(confidence)) or float(confidence) < .8:
        return False
    if text != text.strip() or not _RAIL.fullmatch(text):
        return False
    return ctype != 'gnd' or bool(re.fullmatch(r'(?:[A-Z]*GND[A-Z0-9_]*|VSS[A-Z0-9_]*)', text, re.I))


def _records(ctype, info):
    records = []
    for field in ('power_value', 'partname', 'value'):
        rec = info.get(field) or {}
        if not isinstance(rec, dict):
            continue
        try:
            valid = _accepted(rec.get('text'), rec.get('conf', 0), ctype)
        except (ValueError, TypeError):
            valid = False
        if valid:
            records.append(rec)
    return records


def published_power_value(ctype, info):
    """No voltage defaults or spelling repairs; conflicting OCR stays unknown.

    Standard ground carries GND semantics even when no label is printed.
    Explicit AGND/DGND/VSS observations take precedence over that type default.
    """
    if ctype not in POWER_TYPES:
        return None
    values = {rec['text'] for rec in _records(ctype, info)}
    if len(values) == 1:
        return next(iter(values))
    if len(values) > 1 or info.get('power_value_conflict'):
        return None
    return 'GND' if ctype == 'gnd' else None


def assign_power_values(components, owned, assigned):
    """Read only the already-owned label boxes, never IC pin crops or captions.

    Raw literal text and confidence are retained in the internal assigned
    evidence. No new OCR calls, geometric changes or cross-component borrowing.
    """
    for comp in components:
        cid, ctype = comp['id'], comp.get('type')
        if ctype not in POWER_TYPES:
            continue
        candidates = _records(ctype, assigned.get(cid) or {})
        for item in owned.get(cid, []):
            if (item.get('designator_only') or item.get('pin_target')
                    or item.get('color_crop') or item.get('wide')):
                continue
            if item.get('d', float('inf')) > 40:
                continue
            for role in ('net_label', 'partname', 'value'):
                read = (item.get('reads') or {}).get(role)
                if not isinstance(read, (tuple, list)) or len(read) != 2:
                    continue
                try:
                    valid = _accepted(read[0], read[1], ctype)
                except (ValueError, TypeError):
                    valid = False
                if valid:
                    candidates.append({'text': read[0], 'conf': float(read[1]),
                                       'box': list(item['bbox']), 'role': role,
                                       'source': 'owned_power_label'})
        record = assigned[cid]
        if len({r['text'] for r in candidates}) > 1:
            record['power_value_conflict'] = True
        elif candidates:
            record['power_value'] = max(candidates, key=lambda r: r['conf'])
