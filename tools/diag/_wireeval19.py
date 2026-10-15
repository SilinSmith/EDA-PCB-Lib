# -*- coding: utf-8 -*-
"""_wireeval19.py —— 用官方评测器口径对派生 19 张汇总（stage2_eval 的 --cases 按 stem 过滤，这里自己配）。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
from schematic_parser.data import DEFAULT_DATA_ROOT, load_json            # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case            # noqa: E402
from schematic_parser.data import SPLITS                                  # noqa: E402

D19 = ["0092", "0048", "0019", "0024", "0086", "0042",
       "0075", "0083", "0090", "0057", "0062", "0032", "0028",
       "0175", "0198", "0105", "0106", "0036", "0120"]
import re                                                                 # noqa: E402
from schematic_parser.data import discover                                # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)

cfg = EvalConfig()
rows = []
det = {}
for cid in D19:
    c = exact.get(cid)
    p = ROOT / "work" / "_f220_A" / ("%s.json" % cid)
    if c is None or not p.exists():
        print("skip", cid)
        continue
    s = evaluate_case(load_json(c.json_path), json.loads(p.read_text(encoding="utf-8")), cid, cfg)
    rows.append(s)
    for k, n in (s.net_line.detail or {}).items():
        det[k] = det.get(k, 0) + n
    print("%-6s score %6.2f comp %.4f pin %.4f netH %.4f line %.4f  (tp %d fp %d fn %d)"
          % (cid, s.score, s.component.f1, s.pin.f1, s.net_hypergraph.f1, s.net_line.f1,
             s.net_line.tp, s.net_line.fp, s.net_line.fn))
tp = sum(s.net_line.tp for s in rows); fp = sum(s.net_line.fp for s in rows)
fn = sum(s.net_line.fn for s in rows)
print("---")
print("D19 n=%d  mean score %.4f  net_line F1 %.4f  P %.4f  R %.4f  tp %d fp %d fn %d"
      % (len(rows), sum(s.score for s in rows) / len(rows),
         2 * tp / (2 * tp + fp + fn), tp / max(1e-9, tp + fp), tp / max(1e-9, tp + fn), tp, fp, fn))
print("失配明细:", json.dumps(det, ensure_ascii=False))
