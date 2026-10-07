# -*- coding: utf-8 -*-
"""候选「输出规范」规则的离线 A/B —— 直接跑官方 metrics。

    & 'K:\\python\\311\\python.exe' Y:\\workspace\\EDA\\work\\tools\\diag\\_val_rule.py --pred work\\stage2_v7

发现：官方 `component_text_ok` 对 `VALUE_TYPES`（r/c）比 `value` 逐字符相等。而 GT 里
r/c 的 `value` **只有两种形态**：`None`（29.3%）或**含字母**的串（`4.99k`/`100nF`/`10kΩ`），
**没有一个纯数字**。我们却输出了 489 个单字符（`1`/`4`/`5`…）和 77 个纯数字 ——
这些串在任何情况下都不可能是对的：
  * GT 是 None 时，我们写了个垃圾 -> 本来能对的元件变成错；
  * GT 是含字母的值时，我们本来就错。
所以「r/c 的值只有含字母才写，否则写 null」是**严格不亏**的。

本脚本把这个变换施加在预测 JSON 上（不动磁盘），复用 `stage2_eval.py` 的配对与
`metrics.evaluate_case`，逐规则打印总分。

规则：
  base        原样
  noletter    r/c 的 value 不含字母（A-Za-z/Ω/μ）-> None
  noletter_ws 在 noletter 之上再把 value/Name 的空格去掉
  stripall    base + value/Name 去空格（不做 noletter）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS, discover,  # noqa: E402
                                   load_json, VALUE_TYPES, IC_TYPES)
from schematic_parser.metrics import EvalConfig, aggregate, evaluate_case  # noqa: E402

LETTER = re.compile(r"[A-Za-z\u03A9\u2126\u00B5\u03BC]")   # 含 Ω(2126) μ(00B5/03BC)


def gt_index(data_root):
    idx = {}
    for c in discover(data_root, SPLITS):
        if not c.json_path:
            continue
        idx.setdefault(c.case_id, c)
        idx.setdefault(re.sub(r"\D", "", c.case_id) or c.case_id, c)
    return idx


def transform(pred, rule):
    comps = pred.get("components") or {}
    if not isinstance(comps, dict):
        return pred, 0
    n = 0
    for key, c in comps.items():
        ct = c.get("type")
        if rule in ("noletter", "noletter_ws") and ct in VALUE_TYPES:
            v = c.get("value")
            if v is not None and not LETTER.search(str(v)):
                if str(v) != "":
                    c["value"] = None
                    n += 1
                else:
                    c["value"] = None
                    n += 1
        if rule in ("stripall", "noletter_ws"):
            for f in ("value", "Name"):
                s = c.get(f)
                if isinstance(s, str) and s != s.strip():
                    c[f] = s.strip()
                    n += 1
    return pred, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v7"))
    ap.add_argument("--rules", default="base,noletter,noletter_ws,stripall")
    args = ap.parse_args(argv)

    pred_dir = Path(args.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / args.pred
    idx = gt_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(pred_dir.glob("*.json")) if not f.name.startswith("_")]

    gcache = {}
    for f in files:
        case = idx.get(f.stem) or idx.get(re.sub(r"\D", "", f.stem))
        if case is not None:
            gcache[f.stem] = (load_json(case.json_path), f)

    print("配对 %d 张" % len(gcache))
    print("%-13s %9s %8s %8s %9s %8s %6s %8s" %
          ("rule", "score", "compF1", "pinF1", "netHF1", "lineF1", "改了", "text_mm"))
    for rule in args.rules.split(","):
        scores, nch = [], 0
        for stem, (gt, f) in gcache.items():
            pred = json.loads(f.read_text(encoding="utf-8"))
            pred, n = transform(pred, rule)
            nch += n
            scores.append(evaluate_case(gt, pred, stem, cfg))
        s = aggregate(scores)
        det = {}
        for sc in scores:
            for k, v in (sc.component.detail or {}).items():
                det[k] = det.get(k, 0) + v
        print("%-13s %9.4f %8.4f %8.4f %9.4f %8.4f %6d %8d" %
              (rule, s.get("mean_score", 0.0), s.get("component_f1", 0.0),
               s.get("pin_f1", 0.0), s.get("net_hypergraph_f1", 0.0),
               s.get("net_line_f1", 0.0), nch,
               det.get("component.text_mismatch", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
