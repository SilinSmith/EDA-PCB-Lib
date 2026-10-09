# -*- coding: utf-8 -*-
"""把交付产物的**失分拆到「哪一类、哪一种错」**上——只读已有产物，不跑管线。

动机：`component` 权重 0.30，而官方 `component_text_ok()`（`schematic_parser/metrics.py:213-221`）
规定 **`r`/`c` 比 `value`、`box` 比 `Name`**，其它类型不比文字。所以「元件 F1 掉了多少」
必须能拆成三块：bbox 没配上 / 配上了但 value（或 Name）不对 / 别的。

    python -X utf8 work\\tools\\diag\\_valhead.py <pred_dir> [case ...]
    python -X utf8 work\\tools\\diag\\_valhead.py work\\_f9_B_pin --json work\\_valhead.json

产物：按类汇总表 + 逐 case 表；`--json` 落原始 detail。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, str(ROOT))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="交付产物失分拆解（类 × 错因）")
    ap.add_argument("pred")
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    pred_dir = Path(args.pred)
    idx = {}
    for c in discover(DEFAULT_DATA_ROOT, SPLITS):
        if c.json_path:
            idx.setdefault(c.case_id, c)

    cfg = EvalConfig()
    only = set(args.cases)
    per_case = []
    bucket = defaultdict(Counter)          # type -> Counter(错因)
    dsum = Counter()
    score_sum = 0.0
    n = 0
    for p in sorted(pred_dir.glob("*.json")):
        stem = p.name[:-5]
        if stem.endswith("_Pred"):
            stem = stem[:-5]
        if only and stem not in only:
            continue
        c = idx.get(stem)
        if c is None:
            print("!! 找不到 GT:", stem)
            continue
        gt = load_json(c.json_path)
        pred = json.loads(p.read_text(encoding="utf-8"))
        sc = evaluate_case(gt, pred, stem, cfg)
        d = sc.diagnostic or {}
        for k, v in d.items():
            dsum[k] += v if isinstance(v, (int, float)) else 0
        # 逐类：用 GT 的类型统计（只看比文字的类最有信息）
        gtypes = Counter(v.get("type") for v in gt["components"].values())
        detail_c = {k: v for k, v in d.items() if k.startswith("comp")}
        per_case.append((stem, sc.component.f1, sc.score, detail_c, gtypes))
        score_sum += sc.score
        n += 1
        for t in ("r", "c", "box", "d", "led", "l", "v", "gnd"):
            if gtypes.get(t):
                bucket[t]["gt"] += gtypes[t]
        for k, v in detail_c.items():
            t = k.split(".")[-2] if k.count(".") >= 2 else ""
            if t in bucket:
                bucket[t][k.split(".")[-1]] += v if isinstance(v, (int, float)) else 0

    print("== %s：%d 张，均分 %.4f ==" % (pred_dir, n, score_sum / max(1, n)))
    print("\n-- 元件 detail 全量（跨 case 求和，只列非零）--")
    for k in sorted(dsum):
        if dsum[k]:
            print("   %-34s %8.1f" % (k, dsum[k]))
    print("\n-- 逐 case --")
    for stem, cf1, sc, dc, gt_t in per_case:
        extra = " ".join("%s=%g" % (k.split(".")[-1], v) for k, v in sorted(dc.items()) if v)
        print("   %-16s compF1 %.4f  score %7.3f   %s" % (stem, cf1, sc, extra[:110]))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"cases": [{"case": s, "comp_f1": f, "score": sc, "detail": dc}
                       for s, f, sc, dc, _ in per_case],
             "detail_sum": dict(dsum)}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n-> " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
