# -*- coding: utf-8 -*-
"""§5.114 取证：引脚**点几何**的独立比较（不看脚号，只量「点落在哪」）。

引脚通道只统计 GT 里 type=box 的元件。把 GT 的引脚点集合与各跑法的预测点
集合做**最近邻**配对，量点的几何距离 —— 这样即使脚号读错、或对方按
「未读出的脚号一律省略」少了脚，也能看出谁的**几何约定**更接近 GT。
判分器本身按「(元件, 脚号) 同键且点距 > pin_tol=5」记 fp+fn，本探针把
「号」这一维去掉，只看几何。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_nnpin.py --runs gsd9,pr3b,pr3v [--hist]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, rewrite_keys)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="gsd9,pr3b")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--hist", action="store_true")
a = ap.parse_args()


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get('x'), raw.get('y'))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def pts(section, only):
    out = []
    for c, g in (section or {}).items():
        if c not in only or not isinstance(g, dict):
            continue
        for pk, info in g.items():
            if isinstance(info, dict):
                p = _pt(info.get('point'))
                if p:
                    out.append(p)
    return out


def nn(xs, ys):
    if not ys:
        return []
    return [min(((x - u) ** 2 + (y - v) ** 2) ** 0.5 for u, v in ys) for x, y in xs]


cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c

runs = a.runs.split(",")
print("列 = 最近邻距离 中位/均值/个数（gt-pr：每个 GT 脚点到最近预测脚点；pr-gt 反之）")
print("%-4s | %s" % ("case", " | ".join("%-40s" % r for r in runs)))
print("-" * (7 + 43 * len(runs)))
tot = {r: ([], []) for r in runs}
for cid in a.cases.split(","):
    if cid not in cases:
        continue
    gt_doc = json.load(open(str(cases[cid].json_path), encoding="utf-8"))
    gt = normalise_doc(gt_doc)
    box = {k for k, v in gt['components'].items()
           if canon_type((v or {}).get('type')) in IC_TYPES}
    gpts = pts(gt['pins'], box)
    cfg = EvalConfig()
    cells = []
    for r in runs:
        pp = (ROOT / "work" / ("_fast_out_%s" % r) / "final_results"
              / ("%s_Pred.json" % cid))
        if not pp.exists():
            cells.append("（没有预测 %s）" % pp.name)
            continue
        pred = normalise_doc(json.load(open(str(pp), encoding="utf-8")))
        _, keymap = assign_components(gt, pred, cfg, {})
        pred2 = rewrite_keys(pred, keymap)
        ppts = pts(pred2['pins'], box)
        d1, d2 = nn(gpts, ppts), nn(ppts, gpts)
        tot[r][0].extend(d1)
        tot[r][1].extend(d2)
        fmt = (lambda d: "%.1f/%.1f/%d" % (statistics.median(d), statistics.mean(d), len(d))
               if d else "-")
        cells.append("%-12s | %-12s | gt%d pr%d" % (fmt(d1), fmt(d2), len(gpts), len(ppts)))
        if a.hist and d1:
            h = collections.Counter(int(x // 4) * 4 for x in d1)
            print("      %s %s hist(4px): %s" % (cid, r, dict(sorted(h.items()))))
    print("%-4s | %s" % (cid, " | ".join("%-40s" % c for c in cells)))
print("-" * (7 + 43 * len(runs)))
for r in runs:
    d1, d2 = tot[r]
    fmt = (lambda d: "%.2f/%.2f/%d" % (statistics.median(d), statistics.mean(d), len(d))
           if d else "-")
    print("汇总 %-8s gt-pr %s | pr-gt %s" % (r, fmt(d1), fmt(d2)))
