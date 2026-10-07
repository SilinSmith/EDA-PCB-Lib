# -*- coding: utf-8 -*-
"""给**单个 case** 的预测文档打分（`fast.py --score-only` 只认 `_fast_out`，这张表要单点）。

    python work\\tools\\diag\\_score1.py --case 0143 --pred <path.json>

打印总分与四个分项的 F1/P/R/tp/fp/fn，以及引脚失分的粗分解。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[3] / "eda-pcb"
sys.path[:0] = [str(EDA), str(Path(__file__).resolve().parents[1] / "eval")]

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    if a.case not in cases:
        print("找不到 case", a.case)
        return 2
    c = cases[a.case]
    gt = load_json(c.json_path)
    pr = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    sc = evaluate_case(gt, pr, case_id=a.case, cfg=EvalConfig())
    print("case %s split=%s score=%.4f" % (a.case, c.split, sc.score))
    for k in ("component", "pin", "net_hypergraph", "net_line"):
        f = getattr(sc, k, None)
        if f is None:
            continue
        print("  %-14s F1=%.4f  P=%.4f  R=%.4f  tp=%s fp=%s fn=%s"
              % (k, f.f1, f.precision, f.recall, f.tp, f.fp, f.fn))
    d = getattr(sc, "detail", None) or {}
    if d:
        print("  detail:", json.dumps(d, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
