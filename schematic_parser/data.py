# -*- coding: utf-8 -*-
"""Dataset discovery, pairing and JSON parsing for the EDA-Race schematic task.

Expected on-disk layout (the layout shipped with the competition data)::

    <root>/
      10GT/<1..10>/<n>.png + <n>.json
      10_GTcase/<id>-<source>/<id>-<source>.png + <id>-<source>_target.json
      200_train_cases/<id>/<id>-<source>.png + <id>-<source>_target_named.json

Every case directory holds exactly one PNG and exactly one JSON, so pairing is
done by globbing rather than by parsing the file-name convention.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- constants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(
    REPO_ROOT, 'yolo_part', 'Dataset(waitforpreprocess)')

SPLITS = ('10GT', '10_GTcase', '200_train_cases')

# Component types whose `Name` field participates in the component-text score.
# The problem statement calls these "ic"/"boxic"/"box"; the dataset labels them
# `box`.
IC_TYPES = frozenset({'box', 'ic', 'boxic'})
# Component types whose `value` field participates in the component-text score.
VALUE_TYPES = frozenset({'r', 'resistor', 'c', 'capacitor'})


@dataclass(frozen=True)
class Case:
    """One schematic sample: an image plus its ground-truth JSON."""
    split: str
    case_id: str
    png_path: str
    json_path: Optional[str]

    @property
    def key(self) -> str:
        return '%s/%s' % (self.split, self.case_id)


# ---------------------------------------------------------------- discovery

def discover(data_root: str = DEFAULT_DATA_ROOT,
             splits: Tuple[str, ...] = SPLITS) -> List[Case]:
    """Return every case found under *data_root*, ordered by split then id."""
    cases: List[Case] = []
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            case_dir = os.path.join(split_dir, name)
            if not os.path.isdir(case_dir):
                continue
            pngs = sorted(f for f in os.listdir(case_dir)
                          if f.lower().endswith('.png'))
            jsons = sorted(f for f in os.listdir(case_dir)
                           if f.lower().endswith('.json'))
            # A case with **only** the ground-truth JSON is a valid case for the
            # scorer: ``schematic_parser.evaluate`` never reads the image.  The
            # repository therefore ships ``*.json`` without the PNGs (see
            # ``.gitignore``), and a fresh clone can score predictions straight
            # away.  Pipelines that need the raster must check ``png_path``.
            if not pngs and not jsons:
                continue
            cases.append(Case(
                split=split,
                case_id=name,
                png_path=os.path.join(case_dir, pngs[0]) if pngs else None,
                json_path=os.path.join(case_dir, jsons[0]) if jsons else None,
            ))
    return cases


def find_case(case_id: str, data_root: str = DEFAULT_DATA_ROOT) -> Optional[Case]:
    """Look a case up by its directory name (e.g. ``'0001'`` or ``'1'``)."""
    for c in discover(data_root):
        if c.case_id == case_id:
            return c
    return None


# ---------------------------------------------------------------- json io

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- parsing

_HG_SPLIT = re.compile(r'\s*,\s*')


def parse_hypergraph(text: str) -> List[Tuple[str, str]]:
    """``"(U1.A5,R1.1)"`` -> ``[('U1','A5'), ('R1','1')]``.

    Returns the pin references in the order written, de-duplicated but with
    order preserved.  A malformed / empty string yields ``[]``.
    """
    if not text:
        return []
    s = text.strip()
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    out: List[Tuple[str, str]] = []
    seen = set()
    for tok in _HG_SPLIT.split(s):
        tok = tok.strip()
        if not tok:
            continue
        # split on the LAST dot so designators that contain a dot survive
        if '.' not in tok:
            continue
        comp, _, pinnum = tok.rpartition('.')
        ref = (comp.strip(), pinnum.strip())
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def build_hypergraph(refs) -> str:
    return '(' + ','.join('%s.%s' % (c, p) for c, p in refs) + ')'


def pin_key_to_number(pin_key: str) -> str:
    """``'pin_A5'`` -> ``'A5'``; anything else is returned unchanged."""
    return pin_key[4:] if pin_key.startswith('pin_') else pin_key


def number_to_pin_key(pin_number: str) -> str:
    return 'pin_%s' % pin_number


def net_pin_set(net: dict) -> set:
    """The set of ``(component, pin_number)`` references a net connects."""
    return set(parse_hypergraph(net.get('hyperGraph', '')))


def pin_set_of(components: dict, pins: dict) -> set:
    """All pin references implied by a prediction's components/pins sections."""
    return {(comp, pin_key_to_number(pk))
            for comp, group in pins.items()
            for pk in group}


# ---------------------------------------------------------------- normalise

def normalise_doc(doc: dict) -> dict:
    """Coerce a document into the canonical in-memory shape.

    Anything that is not a ``components``/``pins``/``nets`` mapping is dropped,
    so a partially-formed prediction still scores instead of raising.
    """
    out = {}
    for section in ('components', 'pins', 'nets'):
        value = doc.get(section)
        out[section] = value if isinstance(value, dict) else {}
    return out


# ---------------------------------------------------------------- reporting

def describe_doc(doc: dict) -> Dict[str, int]:
    c, p, n = doc['components'], doc['pins'], doc['nets']
    return {
        'components': len(c),
        'pins': sum(len(g) for g in p.values()),
        'pin_groups': len(p),
        'nets': len(n),
        'edges': sum(len(net.get('edges') or {}) for net in n.values()),
    }
