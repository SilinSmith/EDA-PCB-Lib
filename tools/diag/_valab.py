# -*- coding: utf-8 -*-
"""`valbase` vs `valonly` 的完整对账：在**键匹配上的 r/c** 上，把 (GT 值, 管线值, VLM 值)
三元组分桶，看「覆盖」到底赚在哪、亏在哪。

用法：
    python work\\tools\\diag\\_valab.py --base work\\stage2_v17 --vlm work\\stage2_v22_val
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import stage2_eval  # noqa: E402
from schematic_parser.data import load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, canon_type  # noqa: E402


def norm(s):
    return s if isinstance(s, str) else ("None" if s is None else str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--types", default="r,c")
    a = ap.parse_args()
    keep = set(a.types.split(","))
    base, vlm = Path(a.base), Path(a.vlm)

    exact, digits = stage2_eval.build_index(stage2_eval.DEFAULT_DATA_ROOT)
    buckets = Counter()
    samples = {}
    tot = Counter()
    rows = []
    for f in sorted(vlm.glob("*.json")):
        if f.name.startswith("_"):
            continue
        case, _how = stage2_eval.resolve(f.stem, exact, digits)
        if case is None:
            continue
        g = load_json(case.json_path)
        pv = json.loads(f.read_text(encoding="utf-8"))
        pb = json.loads((base / f.name).read_text(encoding="utf-8"))
        _f1, km = assign_components(g, pv, EvalConfig(), {})
        # 注意方向：`keymap` 是 **预测键 -> GT 键**（metrics.py:293/370）
        for pk, gk in km.items():
            gc = g["components"].get(gk)
            if gc is None:
                continue
            if canon_type(gc.get("type")) not in keep:
                continue
            bc = (pb.get("components") or {}).get(pk) or {}
            vc = (pv.get("components") or {}).get(pk) or {}
            gt, b, v = norm(gc.get("value")), norm(bc.get("value")), norm(vc.get("value"))
            rows.append((gt, b, v))
            tot["n"] += 1
            tot["keyed"] += 1 if pk == gk else 0
            tot["gt_empty"] += 1 if gt in ("None", "") else 0
            tot["b_ok"] += 1 if b == gt else 0
            tot["v_ok"] += 1 if v == gt else 0
            if b == v:
                key = "未改动"
            elif b == gt:
                key = "改坏（管线本来对）"
            elif v == gt:
                key = "改好（VLM 对了）"
            elif gt in ("None", ""):
                key = "改坏（GT 本来空，我们写了值）"
            else:
                key = "改坏（两个都不对，值变了）"
            buckets[key] += 1
            samples.setdefault(key, [])
            if len(samples[key]) < 12:
                samples[key].append((case.case_id, gk, gt, b, v))

    print("键匹配上的 %s 元件共 %d 个（键完全相同的 %d 个，GT 值为空 %d 个）"
          % (a.types, tot["n"], tot["keyed"], tot["gt_empty"]))
    print("  管线（base）值对     %d = %.4f" % (tot["b_ok"], tot["b_ok"] / max(1, tot["n"])))
    print("  VLM（valonly）值对   %d = %.4f" % (tot["v_ok"], tot["v_ok"] / max(1, tot["n"])))
    # 几种「采纳策略」的离线重算（不重新调 VLM）
    pol = {}

    def pick(policy, gt, b, v):
        if policy == "all":
            return v
        if policy == "only_empty":
            return v if b in ("None", "") else b
        if policy == "keep_ci":                 # 只差大小写时保留管线
            return b if b.lower() == v.lower() else v
        if policy == "empty_or_ci":
            if b in ("None", ""):
                return v
            return b if b.lower() == v.lower() else v
        return b

    for p in ("all", "only_empty", "keep_ci", "empty_or_ci"):
        pol[p] = (sum(1 for r in rows if pick(p, *r) == r[0]), tot["n"])
    print()
    for p in ("all", "only_empty", "keep_ci", "empty_or_ci"):
        c, n = pol[p]
        print("  策略 %-12s 值对 %5d = %.4f" % (p, c, c / max(1, n)))
    print()
    for k, v in buckets.most_common():
        print("  %-28s %5d" % (k, v))
    print()
    for k in ("改好（VLM 对了）", "改坏（管线本来对）", "改坏（GT 本来空，我们写了值）",
              "改坏（两个都不对，值变了）"):
        if k in samples:
            print("--- %s ---" % k)
            for cid, gk, gt, b, v in samples[k]:
                print("  %-8s %-6s GT=%-14s 管线=%-14s VLM=%s" % (cid, gk, gt, b, v))


if __name__ == "__main__":
    main()
