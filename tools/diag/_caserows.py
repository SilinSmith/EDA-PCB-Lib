# -*- coding: utf-8 -*-
"""一次性：把某个 case 的元件对账里某一批键打出来。"""
import sys
import json
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "debug"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import ledger                                        # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT  # noqa: E402
from stage2_eval import build_index, resolve, load_json   # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "1"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "C15"
PRED = ROOT / "work" / "_submit_out220z" / "final_results"

exact, digits = build_index(DEFAULT_DATA_ROOT)
case, how = resolve(CASE, exact, digits)
gt = load_json(case.json_path)
pred = load_json(PRED / ("%s_Pred.json" % CASE))

led = ledger.build_ledger(gt, pred, case_id=CASE, split=case.split)
comp = [p for p in led["score"]["parts"] if p["key"] == "component"][0]
print("case %s (%s)  split=%s  总分 %.4f（官方 %.4f）"
      % (CASE, how, case.split, led["score"]["total"], led["score"]["official"]))
print("component: F1=%.4f tp=%d fp=%d fn=%d" % (comp["f1"], comp["tp"], comp["fp"], comp["fn"]))
print("buckets:", json.dumps(comp["stats"].get("buckets", {}), ensure_ascii=False))
print()
print("%-10s %-10s %-8s %-8s %-14s %-7s %s" %
      ("GT键", "我们键", "GT类型", "我们类型", "判决", "中心距", "原因"))
for r in led["components"]:
    gk, pk = r.get("gt_key") or "", r.get("pred_key") or ""
    if gk.startswith(PREFIX) or pk.startswith(PREFIX):
        print("%-10s %-10s %-8s %-8s %-14s %-7s %s" % (
            gk or "-", pk or "-", r.get("gt_type") or "-", r.get("pred_type") or "-",
            r["verdict"], r.get("centre_dist"), (r.get("reason") or "")[:70]))

print()
print("pred 里 %s* :" % PREFIX, sorted(k for k in pred["components"] if k.startswith(PREFIX)))
print()
print("gt   里 %s* :" % PREFIX, sorted(k for k in gt["components"] if k.startswith(PREFIX)))
print()
for k in sorted(k for k in pred["components"] if k.startswith(PREFIX)):
    c = pred["components"][k]
    print("  pred %-8s type=%-6s value=%-8s bbox=%s" %
          (k, c.get("type"), c.get("value"), c.get("bbox")))
for k in sorted(k for k in gt["components"] if k.startswith(PREFIX)):
    c = gt["components"][k]
    print("  gt   %-8s type=%-6s value=%-8s bbox=%s" %
          (k, c.get("type"), c.get("value"), c.get("bbox")))
