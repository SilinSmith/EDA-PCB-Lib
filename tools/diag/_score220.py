# -*- coding: utf-8 -*-
"""把任意产物目录 (`<out>/final_results/*_Pred.json`) 打成 `_220_*` 那种形状的分数文件，
这样 `_stages.py` 的能量（漏斗 / 增量 / 归因 / 最差）就能用在任何一次跑上。

用法：
    python work\\tools\\diag\\_score220.py --out Y:\\workspace\\EDA\\work\\_submit_out220r36 --save _cur220_novlm
    python work\\tools\\diag\\_score220.py --out ... --save x --split 10GT
    python work\\tools\\diag\\_score220.py --out ... --print       # 打完顺手打一张表

`--save` 写到 `work/_fast_base/<name>.json`（不带 .json 后缀会自动补上）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case          # noqa: E402
from stage2_eval import build_index, resolve                            # noqa: E402
import _stages as ST                                                    # noqa: E402

BASE = ROOT / "work" / "_fast_base"


def lookup(cid, exact, digits):
    case, how = resolve(cid, exact, digits)
    if case is None:
        case, how = resolve(cid.lstrip("0") or "0", exact, digits)
    return case, how


def find_pred(out_dir, case_id):
    for name in (case_id + "_Pred.json", case_id + ".json"):
        p = out_dir / "final_results" / name
        if p.exists():
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="给任意产物目录打分")
    ap.add_argument("--out", required=True, help="产物目录（含 final_results/）")
    ap.add_argument("--save", default="", help="存成 work/_fast_base/<名字>.json")
    ap.add_argument("--cases", default="", help="只看这些 case（逗号分隔）")
    ap.add_argument("--print", action="store_true", help="打完顺手打表")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    if not (out_dir / "final_results").is_dir():
        print("! 没有 final_results：%s" % out_dir)
        return 1

    want = {c.strip() for c in a.cases.split(",") if c.strip()}
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    rows, miss = [], []
    for pred in sorted((out_dir / "final_results").glob("*_Pred.json")):
        cid = pred.name[:-len("_Pred.json")]
        if want and cid not in want:
            continue
        case, _ = lookup(cid, exact, digits)
        if case is None:
            miss.append((cid, "找不到 GT"))
            continue
        gt = load_json(Path(case.json_path))
        pr = load_json(pred)
        sc = evaluate_case(gt, pr, case_id=case.case_id, cfg=cfg)
        counts = {}
        for m in ST.METRICS:
            o = getattr(sc, m)
            counts[m] = {"tp": o.tp, "fp": o.fp, "fn": o.fn,
                         "precision": o.precision, "recall": o.recall, "f1": o.f1}
        rows.append({
            "case_id": cid, "score": sc.score,
            "component_f1": sc.component.f1, "pin_f1": sc.pin.f1,
            "net_hypergraph_f1": sc.net_hypergraph.f1, "net_line_f1": sc.net_line.f1,
            "counts": counts, "diagnostic": dict(sc.diagnostic or {}),
            "split": case.split,
        })
    print("打分 %d 张，跳过 %d 张%s" % (len(rows), len(miss),
                                       ("：" + str(miss[:5])) if miss else ""))
    if not rows:
        return 1
    n = len(rows)
    summ = {"cases": n, "mean_score": sum(r["score"] for r in rows) / n}
    for m in ST.METRICS:
        summ["%s_f1" % m] = sum(r["%s_f1" % m] for r in rows) / n
    print("均值 %.4f  |  " % summ["mean_score"] +
          "  ".join("%s %.4f" % (ST.ZH[m], summ["%s_f1" % m]) for m in ST.METRICS))
    if a.save:
        BASE.mkdir(parents=True, exist_ok=True)
        p = BASE / (a.save + ("" if a.save.endswith(".json") else ".json"))
        p.write_text(json.dumps({"config": cfg.to_dict() if hasattr(cfg, "to_dict") else {},
                                 "summary": summ, "cases": rows},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print("已存：%s" % p)
    if a.print:
        ST.summarize(rows, out_dir.name)
        ST.agg_diag(rows, 25)
        ST.worst(rows, 20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
