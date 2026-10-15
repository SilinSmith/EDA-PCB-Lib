# -*- coding: utf-8 -*-
"""逐对拆 `component` 的失分：**text_mismatch 到底是 IC 的 `Name` 还是 r/c 的 `value`**。

背景（`schematic_parser/metrics.py:224 assign_components` + `:376-394`）：
* 键相同的元件（同一个位号）**直接成对**，然后逐对判 type → bbox → text；
* `component_text_ok`（`:224`）对 IC 比 `Name`、对 **r/c 比 `value`，且是字符串完全相等**
  （大小写、单位符号、`Ω` 都要一模一样）；
* 任一项不过就 `fp+fn`，所以在官方 9 张上「7 对 text_mismatch」= 14 个错。

结论要回答的问题是：这 7 对里几对是 IC 名、几对是 r/c 的值，具体字符串差在哪。

用法：
    python -X utf8 work\\tools\\diag\\_comptext.py --dir work\\_fast_out_wl_on
    python -X utf8 work\\tools\\diag\\_comptext.py --dir work\\_full220 --glob "*.json" --cases-file work\\_cases220.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA\eda-pcb")
WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(WORK / "tools" / "eval"))

from schematic_parser.data import IC_TYPES, VALUE_TYPES, load_json               # noqa: E402
from schematic_parser.metrics import (EvalConfig, assign_components,             # noqa: E402
                                      bbox_ok, canon_type, component_text_ok)
import stage2_eval as SE                                                         # noqa: E402


def kind(ctype: str) -> str:
    if ctype in IC_TYPES:
        return "IC(Name)"
    if ctype in VALUE_TYPES:
        return "RC(value)"
    return "other(type=%s)" % ctype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="final_results/*_Pred.json")
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--cases-file", default=None)
    ap.add_argument("--max-print", type=int, default=40)
    a = ap.parse_args()

    if a.cases_file:
        names = [s.strip() for s in Path(a.cases_file).read_text(
            encoding="utf-8").strip().split(",") if s.strip()]
    else:
        names = [s.strip() for s in a.cases.split(",") if s.strip()]

    cfg = EvalConfig()
    idx, digits = SE.build_index(SE.DEFAULT_DATA_ROOT)
    d = Path(a.dir)

    buckets = Counter()
    shown = 0
    for name in names:
        hit = None
        for f in sorted(d.glob(a.glob)):
            if f.stem == name or f.stem.startswith(name + "_"):
                hit = f
                break
        if hit is None:
            print("[warn] %s 没找到（glob=%s）" % (name, a.glob))
            continue
        case, _how = SE.resolve(name, idx, digits)
        if case is None:
            print("[warn] %s 没有 GT" % name)
            continue
        gt, pred = load_json(case.json_path), json.loads(hit.read_text(encoding="utf-8"))
        gc, pc = gt.get("components") or {}, pred.get("components") or {}
        res, keymap = assign_components(gt, pred, cfg)
        for pk, gk in keymap.items():
            g, p = gc.get(gk) or {}, pc.get(pk) or {}
            ct = canon_type(g.get("type"))
            k = kind(ct)
            if canon_type(p.get("type")) != ct:
                buckets[k + "|type_mismatch"] += 1
                continue
            if bbox_ok(g.get("bbox"), p.get("bbox"), cfg) is not True:
                buckets[k + "|bbox_mismatch"] += 1
                continue
            if not component_text_ok(g, p, cfg):
                buckets[k + "|text_mismatch"] += 1
                if shown < a.max_print:
                    gv = g.get("Name") if ct in IC_TYPES else g.get("value")
                    pv = p.get("Name") if ct in IC_TYPES else p.get("value")
                    print("  [%s %s] GT %r  ->  pred %r   (%s)" % (name, gk, gv, pv, k))
                    shown += 1
                continue
            buckets[k + "|ok"] += 1
        gvals = set(keymap.values())
        for gk, g in gc.items():
            if gk not in gvals:
                buckets[kind(canon_type(g.get("type"))) + "|fn(没配上)"] += 1
                if shown < a.max_print:
                    print("  [%s] FN GT %s type=%s value=%r bbox=%s"
                          % (name, gk, g.get("type"), g.get("value"),
                             [round(float(v), 1) for v in (g.get("bbox") or [])]))
                    shown += 1
        for pk, p in pc.items():
            if pk not in keymap:
                buckets[kind(canon_type(p.get("type"))) + "|fp(多出来)"] += 1
                if shown < a.max_print:
                    print("  [%s] FP pred %s type=%s value=%r bbox=%s"
                          % (name, pk, p.get("type"), p.get("value"),
                             [round(float(v), 1) for v in (p.get("bbox") or [])]))
                    shown += 1

    print("\n== 汇总（%d 张）==" % len(names))
    for k in sorted(buckets):
        print("  %-34s %d" % (k, buckets[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
