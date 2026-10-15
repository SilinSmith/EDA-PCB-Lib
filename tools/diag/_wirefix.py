# -*- coding: utf-8 -*-
"""_wirefix.py —— 三个「只改一处」的反事实测量（全部离线重算评测器，不动源码/产物）：

  A 基线：现产物原样
  B 去掉输出里 edges 为空的网（我们的输出本来就不该有这些「单脚空网」）
  C 基线 + 只把评测器的 line_tol 从 5 放到 10 / 20 / 40（衡量「端点差一点」的天花板）
  D 基线 + 把每条 GT 边的 y 翻转关掉（排除坐标口径错的怀疑）
输出 mean score / net_line F1 / tp fp fn。
"""
import copy
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS, discover, load_json,  # noqa: E402
                                   normalise_doc)
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)

W = ROOT / "work"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
D19 = ["0092", "0048", "0019", "0024", "0086", "0042", "0075", "0083", "0090", "0057",
       "0062", "0032", "0028", "0175", "0198", "0105", "0106", "0036", "0120"]
CASES = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
         "0092", "0024", "0032", "0120", "0057"]


def gt_of(cid):
    return load_json(exact[cid].json_path)


def pred_of(cid):
    return json.loads((W / "_f220_A" / ("%s.json" % cid)).read_text(encoding="utf-8"))


def strip_empty(doc):
    d = copy.deepcopy(doc)
    nets = d.get("nets") or {}
    d["nets"] = {k: v for k, v in nets.items() if (v.get("edges") or {})}
    return d


def run(label, cases, tf=None, cfg=None):
    tf = tf or pred_of
    cfg = cfg or EvalConfig()
    agg = {"tp": 0, "fp": 0, "fn": 0}
    sc = []
    per = {}
    for cid in cases:
        gt, pr = gt_of(cid), tf(cid)
        s = evaluate_case(gt, pr, cfg)
        c = s.net_line.as_dict()
        agg["tp"] += c["tp"]
        agg["fp"] += c["fp"]
        agg["fn"] += c["fn"]
        sc.append(s.score)
        per[cid] = (round(s.score, 2), round(c.get("f1", 0.0), 4), c["tp"], c["fp"], c["fn"])
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
    p = tp / max(1e-9, tp + fp)
    r = tp / max(1e-9, tp + fn)
    print("%-34s mean score %6.2f ｜ net_line F1 %.4f (P %.4f R %.4f) tp %d fp %d fn %d"
          % (label, sum(sc) / len(sc), f1, p, r, tp, fp, fn))
    return per


print("### 官方 9 张")
base_gt9 = run("A 基线（现产物）", GT9)
noemp_gt9 = run("B 去掉 0 边网", GT9, lambda c: strip_empty(pred_of(c)))
for tol in (10.0, 20.0, 40.0):
    run("C line_tol=%.0f" % tol, GT9, cfg=EvalConfig(line_tol=tol))
run("D line_tol=20 + 去掉 0 边网", GT9, lambda c: strip_empty(pred_of(c)), cfg=EvalConfig(line_tol=20.0))

print()
print("### 派生 19 张")
run("A 基线（现产物）", D19)
run("B 去掉 0 边网", D19, lambda c: strip_empty(pred_of(c)))
for tol in (10.0, 20.0, 40.0):
    run("C line_tol=%.0f" % tol, D19, cfg=EvalConfig(line_tol=tol))
run("D line_tol=20 + 去掉 0 边网", D19, lambda c: strip_empty(pred_of(c)), cfg=EvalConfig(line_tol=20.0))

print()
print("### 逐 case：基线 → 去掉 0 边网（score / lineF1 / tp fp fn）")
for cid in CASES:
    b = base_gt9.get(cid)
    n = noemp_gt9.get(cid)
    if b is None:
        continue
    print("  %-5s 基线 %6.2f/%.4f tp%-4d fp%-4d fn%-4d  →  去空网 %6.2f/%.4f tp%-4d fp%-4d fn%-4d"
          % (cid, b[0], b[1], b[2], b[3], b[4], n[0], n[1], n[2], n[3], n[4]))

