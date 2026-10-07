# -*- coding: utf-8 -*-
"""按 GT 类型拆开 component 的错误预算。

用官方 `assign_components` 拿到配对表（keymap 的 values 就是被配上的 GT 键），
再按官方三条判据逐对判 tp / type_mismatch / bbox_mismatch / text_mismatch，
另外统计 missing / extra。这样每一分该去哪儿就一目了然。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, load_json  # noqa: E402
from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, assign_components, bbox_ok, canon_type, component_text_ok,
    IC_TYPES, VALUE_TYPES)

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "stage2_v17"
cfg = EvalConfig()

cnt = defaultdict(Counter)
for c in discover():
    if not c.json_path:
        continue
    f = PRED / ("%s.json" % c.case_id)
    if not f.exists():
        f = PRED / ("%s_Pred.json" % c.case_id)      # final_results 的命名
    if not f.exists():
        continue
    gt = load_json(c.json_path)
    pr = json.loads(f.read_text(encoding="utf-8"))
    gc, pc = gt["components"], pr.get("components") or {}
    _f1, keymap = assign_components(gt, pr, cfg, {})
    matched_gt = set(keymap.values())
    for gk, g in gc.items():
        t = canon_type(g.get("type"))
        if gk not in matched_gt:
            cnt[t]["missing"] += 1
            continue
        pk = next((p for p, q in keymap.items() if q == gk), None)
        p = pc.get(pk) or {}
        if canon_type(p.get("type")) != canon_type(g.get("type")):
            cnt[t]["type_mm"] += 1
        elif bbox_ok(g.get("bbox"), p.get("bbox"), cfg) is not True:
            cnt[t]["bbox_mm"] += 1
        elif not component_text_ok(g, p, cfg):
            cnt[t]["text_mm"] += 1
        else:
            cnt[t]["tp"] += 1
    for pk in pc:
        if pk not in keymap:
            cnt[canon_type((pc[pk] or {}).get("type"))]["extra"] += 1

KEYS = ("tp", "text_mm", "bbox_mm", "type_mm", "missing", "extra")
tot = Counter()
print("%-14s %7s %7s %7s %7s %7s %7s   %s" % ("GT 类型", *KEYS, "该类型的判据"))
rows = sorted(cnt.items(), key=lambda kv: -sum(kv[1].values()))
for t, c in rows:
    for k in KEYS:
        tot[k] += c[k]
    if t in IC_TYPES:
        how = "比 Name"
    elif t in VALUE_TYPES:
        how = "比 value"
    else:
        how = "不比文字（配上即对）"
    print("%-14s %7d %7d %7d %7d %7d %7d   %s"
          % (t, c["tp"], c["text_mm"], c["bbox_mm"], c["type_mm"], c["missing"], c["extra"], how))
print("%-14s %7d %7d %7d %7d %7d %7d" % ("合计", *[tot[k] for k in KEYS]))
print()
print("component 总分里各类型的占比（tp 越多越好）：")
for t, c in rows[:16]:
    print("  %-14s tp %5d  丢分（text+bbox+type+missing+extra）%6d"
          % (t, c["tp"], c["text_mm"] + c["bbox_mm"] + c["type_mm"] + c["missing"] + c["extra"]))
print()
print("不比文字的类型的 missing 合计 %d，比文字的（r/c/box）missing 合计 %d"
      % (sum(c["missing"] for t, c in cnt.items() if t not in IC_TYPES and t not in VALUE_TYPES),
         sum(c["missing"] for t, c in cnt.items() if t in IC_TYPES or t in VALUE_TYPES)))
print("比文字的（r/c/box）text_mm 合计 %d（r %d / c %d / box %d）"
      % (tot["text_mm"], cnt["r"]["text_mm"], cnt["c"]["text_mm"], cnt["box"]["text_mm"]))
