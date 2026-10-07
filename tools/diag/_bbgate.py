# -*- coding: utf-8 -*-
"""量「差一点点没过 IoU 0.5 闸门」的元件有多少。

不靠官方 metrics，直接在 GT 与预测之间按类型 + 中心最近配对，算 IoU 分布。
如果一大批 GT 元件的最佳 IoU 落在 0.3~0.5，那就是纯粹的**包围盒形状**问题，
补上就是白捡的 tp（每个 tp 同时省掉一个 fp 和一个 fn）。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, load_json  # noqa: E402
from schematic_parser.metrics import canon_type  # noqa: E402

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "stage2_v17"


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


buckets = Counter()
by_type = defaultdict(lambda: Counter())
size_pairs = Counter()
worst = []
tot = 0
for c in discover():
    if not c.json_path:
        continue
    f = PRED / ("%s.json" % c.case_id)
    if not f.exists():
        continue
    gt = load_json(c.json_path)
    pr = json.loads(f.read_text(encoding="utf-8"))
    preds = []
    for k, v in (pr.get("components") or {}).items():
        if v.get("bbox"):
            preds.append((canon_type(v.get("type")), v["bbox"], k))
    for k, g in (gt.get("components") or {}).items():
        if not g.get("bbox"):
            continue
        ct = canon_type(g.get("type"))
        tot += 1
        best = 0.0; bk = None; bb = None
        for pt, pb, pk in preds:
            if pt != ct:
                continue
            v = iou(g["bbox"], pb)
            if v > best:
                best = v; bk = pk; bb = pb
        if best >= 1.0:
            tag = "1.0"
        elif best >= 0.7:
            tag = "0.7~1.0"
        elif best >= 0.5:
            tag = "0.5~0.7"
        elif best >= 0.35:
            tag = "0.35~0.5"
        elif best >= 0.2:
            tag = "0.2~0.35"
        elif best > 0:
            tag = "0~0.2"
        else:
            tag = "0（同类型都没配到）"
        buckets[tag] += 1
        by_type[ct][tag] += 1
        if 0.3 <= best < 0.5 and bb is not None:
            gw = g["bbox"][2] - g["bbox"][0]; gh = g["bbox"][3] - g["bbox"][1]
            pw = bb[2] - bb[0]; ph = bb[3] - bb[1]
            size_pairs[(round(gw), round(gh), round(pw), round(ph))] += 1
            worst.append((best, c.case_id, k, ct, g["bbox"], bb))

print("GT 元件合计 %d（预测目录 %s）" % (tot, PRED))
print("%-18s %6s %7s" % ("最佳 IoU 桶", "个数", "占比"))
for tag in ("1.0", "0.7~1.0", "0.5~0.7", "0.35~0.5", "0.2~0.35", "0~0.2", "0（同类型都没配到）"):
    n = buckets[tag]
    print("%-18s %6d %7.4f" % (tag, n, n / max(1, tot)))
print()
print("--- 各类型在 0.35~0.5 桶里的个数（题目最大的一块）---")
rows = [(by_type[t]["0.35~0.5"], t, sum(by_type[t].values())) for t in by_type]
rows.sort(reverse=True)
for n, t, m in rows[:18]:
    print("  %-16s %5d / %5d = %.3f   该类型 0.5 以上占 %.3f"
          % (t, n, m, n / max(1, m), (by_type[t]["0.5~0.7"] + by_type[t]["0.7~1.0"] + by_type[t]["1.0"]) / max(1, m)))
print()
print("--- 0.35~0.5 桶里最常见的 (GT 宽,GT 高)->(预测 宽,预测 高) ---")
for (gw, gh, pw, ph), n in size_pairs.most_common(20):
    print("  GT %3dx%-3d -> 预测 %3dx%-3d   %5d 个" % (gw, gh, pw, ph, n))
print()
print("--- 最接近闸门的 25 个（IoU 0.45~0.5）---")
worst.sort(key=lambda x: -x[0])
for b, cid, k, ct, gb, pb in [w for w in worst if w[0] >= 0.45][:25]:
    print("  %.4f %-12s %-6s %-12s GT %s -> 预测 %s" % (b, cid, k, ct, [round(x) for x in gb], [round(x) for x in pb]))
