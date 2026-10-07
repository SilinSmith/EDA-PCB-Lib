# -*- coding: utf-8 -*-
"""missing 的 GT 元件 → 没被配上的预测元件，到底是什么关系。

官方口径下（`unkeyed_match='gt'`）**只有 GT 侧是 ∅NNN 的元件才允许坐标匹配**，
所以一个**带名**的 GT 元件只要键对不上，就必然是 missing + 一个 extra。
这个脚本把 missing 的 GT 元件与「剩下的」预测元件按中心距最近配起来，
看它们是不是类型标错、还是纯粹键读错。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, load_json  # noqa: E402
from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, assign_components, bbox_ok, canon_type, is_unkeyed)

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "stage2_v17"
cfg = EvalConfig()


def ctr(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


same_type = Counter()
diff_type = Counter()
dist_hist = Counter()
keyed_missing = Counter()
unkeyed_missing = Counter()
for c in discover():
    if not c.json_path:
        continue
    f = PRED / ("%s.json" % c.case_id)
    if not f.exists():
        f = PRED / ("%s_Pred.json" % c.case_id)   # final_results 的命名
    if not f.exists():
        continue
    gt = load_json(c.json_path)
    pr = json.loads(f.read_text(encoding="utf-8"))
    gc, pc = gt["components"], pr.get("components") or {}
    _f1, keymap = assign_components(gt, pr, cfg, {})
    matched_gt = set(keymap.values())
    left_p = [(k, v) for k, v in pc.items() if k not in keymap and v.get("bbox")]
    for gk, g in gc.items():
        if gk in matched_gt or not g.get("bbox"):
            continue
        gt_t = canon_type(g.get("type"))
        (unkeyed_missing if is_unkeyed(gk) else keyed_missing)[gt_t] += 1
        gc_ = ctr(g["bbox"])
        best = None
        for pk, p in left_p:
            pc_ = ctr(p["bbox"])
            d = math.hypot(gc_[0] - pc_[0], gc_[1] - pc_[1])
            if best is None or d < best[0]:
                best = (d, pk, canon_type(p.get("type")), bbox_ok(g["bbox"], p["bbox"], cfg) is True)
        if best is None:
            dist_hist["没有剩余预测"] += 1
            continue
        d, pk, pt, bok = best
        if d <= 20:
            dist_hist["<=20px"] += 1
        elif d <= 40:
            dist_hist["20~40px"] += 1
        elif d <= 80:
            dist_hist["40~80px"] += 1
        else:
            dist_hist[">80px"] += 1
        if gt_t == pt:
            same_type[gt_t] += 1
        else:
            diff_type[(gt_t, pt)] += 1

print("预测目录 %s" % PRED)
print()
print("--- missing 的 GT 元件里，最近的剩余预测在多元处？---")
for k in ("<=20px", "20~40px", "40~80px", ">80px", "没有剩余预测"):
    print("  %-12s %5d" % (k, dist_hist[k]))
print()
print("--- **类型一致**但没配上的（纯键问题）---")
for t, n in same_type.most_common(20):
    print("  %-14s %5d" % (t, n))
print()
print("--- **类型不一致**（最近的剩余预测是别的类型）top 24 ---")
for (a, b), n in diff_type.most_common(24):
    print("  GT %-12s -> 预测 %-12s %5d" % (a, b, n))
print()
print("带名 GT 元件 missing 合计 %d；∅NNN 的 missing 合计 %d"
      % (sum(keyed_missing.values()), sum(unkeyed_missing.values())))
print("带名 missing 的类型分布 top12：", keyed_missing.most_common(12))
print("∅NNN missing 的类型分布 top12：", unkeyed_missing.most_common(12))
