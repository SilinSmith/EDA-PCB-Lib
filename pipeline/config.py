# -*- coding: utf-8 -*-
"""Shared paths, constants and the y-axis flip for the schematic parsing pipeline.

Coordinate conventions (measured, see ``docs/勘察报告.md``):

* image pixels, origin top-left, y down  -- everything in this package that is called
  ``*_img`` or comes straight out of cv2 / PIL
* competition JSON, origin **bottom-left**, y up -- ``y_json = H - y_img``

The flip happens exactly once, in :func:`json_to_img` / :func:`img_to_json`; no other
module is allowed to write `H - y` by hand.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- paths

# 路径按 __file__ 推导（本文件在 <repo>/pipeline/config.py）；以前硬编码
# r'Y:\workspace\EDA\eda-pcb'，容器里那个路径不存在。仍可用 EDA_PCB_REPO /
# EDA_ROOT 覆盖（诊断脚本换数据集时会用）。
REPO = os.environ.get('EDA_PCB_REPO') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
EDA_ROOT = os.environ.get('EDA_ROOT') or os.path.dirname(REPO)
DATA_ROOT = os.path.join(REPO, 'yolo_part', 'Dataset(waitforpreprocess)')
REF_ROOT = os.path.join(REPO, 'yolo_part', 'Dataset300我不知道干嘛用的先放置')
SYNTH_ROOT = os.path.join(EDA_ROOT, '合成原理图demo')
OUT_ROOT = os.path.join(REPO, 'pipeline', 'out')
DOCS = os.path.join(EDA_ROOT, 'docs')

YOLO_BEST = os.path.join(
    REPO, 'yolo_part', 'runs', 'detect', 'pcb43_yolo11m_im1024-2', 'weights', 'best.pt')


# ---------------------------------------------------------------- geometry

def json_to_img(pts, h):
    """``[(x, y_json), ...]`` -> ``[(x, y_img), ...]`` (accepts a single pair too)."""
    if pts and isinstance(pts[0], (int, float)):
        return (float(pts[0]), float(h) - float(pts[1]))
    return [(float(x), float(h) - float(y)) for x, y in pts]


def img_to_json(pts, h):
    if pts and isinstance(pts[0], (int, float)):
        return (float(pts[0]), float(h) - float(pts[1]))
    return [(float(x), float(h) - float(y)) for x, y in pts]


def bbox_json_to_img(bbox, h):
    """``[x1, y1, x2, y2]`` bottom-left origin -> ``(x1, y1, x2, y2)`` top-left origin."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (min(x1, x2), h - max(y1, y2), max(x1, x2), h - min(y1, y2))


def bbox_img_to_json(box, h):
    x1, y1, x2, y2 = (float(v) for v in box)
    return [min(x1, x2), h - max(y1, y2), max(x1, x2), h - min(y1, y2)]


# ---------------------------------------------------------------- text roles

#: The six roles the router has to tell apart.  Typo `netLabel` occurs 44x in the
#: reference intermediate representation and is folded into `net_label`.
TEXT_ROLES = ('designator', 'value', 'pin_number', 'pin_name', 'net_label', 'note')

ROLE_ALIASES = {
    'designator': 'designator',
    'value': 'value',
    'pinNumber': 'pin_number',
    'pin_number': 'pin_number',
    'pinName': 'pin_name',
    'pin_name': 'pin_name',
    'netlabel': 'net_label',
    'netLabel': 'net_label',
    'net_label': 'net_label',
    'searchName': 'note',
    'note': 'note',
    '': 'note',
    None: 'note',
}

#: Component types whose ``Name`` carries a part number rather than a designator.
NAME_IS_PARTNO = frozenset({'box', 'block'})
#: Component types with no designator text at all (the corpus spells their key `∅N`).
ANONYMOUS_TYPES = frozenset({'gnd', 'v', 'net_short', 'net_input', 'net_output',
                             'net_bidirection'})
#: Types whose ``value`` field is compared by the scorer.
VALUE_TYPES = frozenset({'r', 'resistor', 'c', 'capacitor'})
#: Types whose ``Name`` field is compared by the scorer.
IC_TYPES = frozenset({'box', 'ic', 'boxic'})


def role_of(raw) -> str:
    return ROLE_ALIASES.get(raw, 'note')
