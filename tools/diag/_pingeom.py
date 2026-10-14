# -*- coding: utf-8 -*-
"""§5.111 取证（二）：**引脚点该落在框外多远** —— GT 与我们的「框外距离」逐脚对照。

`protocol2` 的 pin 通道用 `pin_tol=5px` 比对 (元件, 脚号) 的点。`stage1/pins.py` 的
设计约定是「接触点朝框外推 `PIN_OUT_BOX=12px`」（GT 实测 p50=10 / p75=13），但如果
某条代码路径没推、或元件框本身偏了，点就会整体离开 GT 10px 上下 —— 恰好越过 5px 容差。

本探针把「框外距离」定义成：点到**所选最近那条框边**的垂直距离（框内为负、框外为正），
对 GT 与预测分别算，并只统计两边键与名都对上的**该判 tp 的脚**（这样偏移归因最干净）。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_pingeom.py --run gsd9 [--cases 1,...,10] [--show 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, rewrite_keys)


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get('x'), raw.get('y'))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _outward(pt, bbox):
    """(框外距离, 边名)；框内为负。边名 = l/r/t/b。"""
    x, y = pt
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cand = (("l", x0 - x), ("r", x - x1), ("t", y0 - y), ("b", y - y1))
    edge, d = max(cand, key=lambda kv: kv[1])
    return d, edge


ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--show", type=int, default=40)
args = ap.parse_args()

want = set(args.cases.split(","))
cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = EvalConfig()
hist = Counter()
per_case = defaultdict(list)
print("case | 元件 | gt外推 | pred外推 | 差 | 框偏移(x0,y0,x1,y1)")
print("-" * 78)
for cid in sorted(cases, key=lambda s: (len(s), s)):
    if cid not in want:
        continue
    pred_p = ROOT / "work" / ("_fast_out_%s" % args.run) / "final_results" / ("%s_Pred.json" % cid)
    if not pred_p.exists():
        continue
    gt_doc = json.load(open(str(cases[cid].json_path), encoding="utf-8"))
    pred_doc = json.load(open(str(pred_p), encoding="utf-8"))
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    comp, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)
    box = {k for k, v in gt['components'].items()
           if canon_type((v or {}).get('type')) in IC_TYPES}
    for c in sorted(box):
        gb = gt['components'][c].get('bbox') or gt['components'][c].get('box')
        pb = (pred2['components'].get(c) or {}).get('bbox') or \
             (pred2['components'].get(c) or {}).get('box')
        if not gb or not pb:
            continue
        gpins = gt['pins'].get(c) or {}
        ppins = pred2['pins'].get(c) or {}
        rows = []
        for pk, g in gpins.items():
            p = ppins.get(pk)
            if not isinstance(g, dict) or not isinstance(p, dict):
                continue
            if p.get('pinname') != g.get('pinname'):
                continue                      # 名字都不对，几何没什么可说的
            a, b = _pt(g.get('point')), _pt(p.get('point'))
            if a is None or b is None:
                continue
            dg, eg = _outward(a, gb)
            dp, ep = _outward(b, pb)
            rows.append((pk, a, b, dg, eg, dp, ep))
        if not rows:
            continue
        shift = [float(pb[i]) - float(gb[i]) for i in range(4)]
        print("%-4s | %-4s | 框 %s → %s  dx=%+.0f dy=%+.0f dw=%+.0f dh=%+.0f"
              % (cid, c, ["%.0f" % v for v in gb], ["%.0f" % v for v in pb],
                 shift[0], shift[1], shift[2], shift[3]))
        for pk, a, b, dg, eg, dp, ep in rows:
            hist[(eg, "%+.0f" % (dg // 5 * 5), ep, "%+.0f" % (dp // 5 * 5))] += 1
            hist[(eg, ep, "edge" if eg == ep else "cross")] += 1
            per_case[cid].append((pk, a, b, dg, eg, dp, ep))
            if args.show and len(per_case[cid]) <= args.show:
                print("       %-7s gt=(%.0f,%.0f) %s %+.0f  |  pred=(%.0f,%.0f) %s %+.0f  "
                      "Δ=(%+.0f,%+.0f)"
                      % (pk, a[0], a[1], eg, dg, b[0], b[1], ep, dp,
                         b[0] - a[0], b[1] - a[1]))
print("-" * 78)
print("全 10 张「键名都对」的脚的框外距离分布（GT边→GT距离档, pr边→pr距离档）：")
for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
    if len(k) == 4:
        print("    %-40s %d" % (str(k), v))
print("边的对应关系：")
for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
    if len(k) == 3:
        print("    %-28s %d" % (str(k), v))
