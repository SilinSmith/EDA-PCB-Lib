# -*- coding: utf-8 -*-
"""§5.115 取证：case 8 的 box 引脚「逐边点位」三方对照（GT / 我方 sjndef / 队友 pr3b）。

把每个引脚点按最近的框边归边，量它的**切向坐标**（沿边方向）与**法向偏移**
（框外为正，px）。同一侧按切向排序后三方并排打印 —— 用来判断
「我方 y（切向）偏 10px、队友的实测切向坐标对不对」。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_c8pins.py --cases 8 --runs sjndef,pr3b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, rewrite_keys)

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="8")
ap.add_argument("--runs", default="sjndef,pr3b")
a = ap.parse_args()

SIDES = ("top", "bottom", "left", "right")


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get("x"), raw.get("y"))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _bbox(comp):
    for key in ("bbox_img", "bbox", "box"):
        b = (comp or {}).get(key)
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            try:
                return [float(v) for v in b[:4]]
            except (TypeError, ValueError):
                pass
    return None


def gather(doc, cid):
    """-> side -> list[(num, name, tangential, normal)]，按切向排序。"""
    comp = (doc.get("components") or {}).get(cid) or {}
    bb = _bbox(comp)
    out = {s: [] for s in SIDES}
    if bb is None:
        return out
    x0, y0, x1, y1 = bb
    for num, info in ((doc.get("pins") or {}).get(cid) or {}).items():
        if not isinstance(info, dict):
            continue
        p = _pt(info.get("point"))
        if p is None:
            continue
        x, y = p
        cand = {
            "top": (abs(y - y0), x, y0 - y),
            "bottom": (abs(y - y1), x, y - y1),
            "left": (abs(x - x0), y, x0 - x),
            "right": (abs(x - x1), y, x - x1),
        }
        side = min(cand, key=lambda s: cand[s][0])
        _, t, n = cand[side]
        out[side].append((str(num), str(info.get("name") or ""), t, n))
    for s in SIDES:
        out[s].sort(key=lambda r: r[2])
    return out


cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c

runs = a.runs.split(",")
for cid in a.cases.split(","):
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    box = {k for k, v in gt["components"].items()
           if canon_type((v or {}).get("type")) in IC_TYPES}
    cfg = EvalConfig()
    srcs = {"GT": (gt, box)}
    for r in runs:
        pp = ROOT / "work" / ("_fast_out_%s" % r) / "final_results" / ("%s_Pred.json" % cid)
        if not pp.exists():
            print("（没有 %s）" % pp)
            continue
        pred = normalise_doc(json.load(open(str(pp), encoding="utf-8")))
        _, keymap = assign_components(gt, pred, cfg, {})
        pred2 = rewrite_keys(pred, keymap)
        gkey = {v: k for k, v in keymap.items()}
        keys = {k for k in box if k in (pred2.get("components") or {})}
        srcs[r] = (pred2, keys)
    common = set(box)
    for r in runs:
        if r in srcs:
            common &= set(srcs[r][1])
    for cid2 in sorted(common):
        print("=" * 96)
        print("case %s  comp %s  type=%s" % (cid, cid2,
              canon_type((gt["components"].get(cid2) or {}).get("type"))))
        print("  GT bbox = %s" % [round(v, 1) for v in (_bbox(gt["components"][cid2]) or [])])
        for r in runs:
            if r in srcs:
                bb = _bbox(srcs[r][0]["components"].get(cid2)) or []
                print("  %-7s bbox = %s" % (r, [round(v, 1) for v in bb]))
        tabs = {name: gather(doc, cid2) for name, (doc, _) in srcs.items()}
        for side in SIDES:
            rows = {name: tabs[name][side] for name in srcs}
            if not any(rows.values()):
                continue
            print("  -- %s --" % side)
            width = max(len(r) for r in rows.values())
            print("     %-22s | %-22s | %-22s" % tuple(
                ("%s(切向,法向)" % n)[:22] for n in srcs))
            for i in range(width):
                cells = []
                for name in srcs:
                    rr = rows[name]
                    if i < len(rr):
                        num, nm, t, n = rr[i]
                        cells.append("%-7s %7.1f %6.1f" % (num[:7], t, n))
                    else:
                        cells.append(" " * 22)
                print("     %s | %s | %s" % tuple(cells))
