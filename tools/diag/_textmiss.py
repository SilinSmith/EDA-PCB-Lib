# -*- coding: utf-8 -*-
"""配对上、类型对、bbox 对，但**文字不对**的那些，到底是什么类型、错在哪。

官方 metrics：`component_text_ok` 只在 IC（box）比 `Name`、在 VALUE_TYPES
（r/c/l/d…）比 `value`，其它类型直接返回 True。文字不对 ⇒ fp+1 fn+1（丢一个 tp）。

用法：python work\tools\diag\_textmiss.py --pred work\stage2_v13
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES, VALUE_TYPES,  # noqa: E402
                                      bbox_ok, canon_type, component_text_ok)
from stage2_eval import build_index, resolve  # noqa: E402

UKEY = "\u2205"
CFG = EvalConfig()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v13"))
    ap.add_argument("--samples", type=int, default=30)
    a = ap.parse_args()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cat = Counter()
    samples = []
    for f in sorted(Path(a.pred).glob("*.json")):
        if f.name.startswith("_"):
            continue
        case, _ = resolve(f.stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = gt.get("components") or {}, pred.get("components") or {}
        for k in gc:
            if k not in pc:
                continue
            g, p = gc[k], pc[k]
            gty = canon_type(g.get("type"))
            pty = canon_type(p.get("type"))
            if gty != pty:
                cat["type_mismatch"] += 1
                continue
            if bbox_ok(g.get("bbox"), p.get("bbox"), CFG) is not True:
                cat["bbox_mismatch"] += 1
                continue
            if component_text_ok(g, p, CFG):
                cat["ok"] += 1
                continue
            if gty in IC_TYPES:
                cat["text_mismatch box(Name)"] += 1
                if len(samples) < a.samples:
                    samples.append((f.stem, k, gty, "Name",
                                    (g.get("Name") or "")[:26],
                                    (p.get("Name") or "")[:26]))
            elif gty in VALUE_TYPES:
                cat["text_mismatch value(%s)" % gty] += 1
                pv = p.get("value")
                if pv is None:
                    cat["V 我们没写值(None)"] += 1
                elif pv == "":
                    cat["V 我们写了空串"] += 1
                elif (g.get("value") or "") == "":
                    cat["V GT 本来就是空、我们多写"] += 1
                else:
                    cat["V 写了但写错"] += 1
                if len(samples) < a.samples:
                    samples.append((f.stem, k, gty, "value",
                                    (g.get("value") or "")[:26],
                                    str(p.get("value"))[:26]))
            else:
                cat["text_mismatch 其它"] += 1
    print("键对上的元件判定结果：")
    for k, v in cat.most_common():
        print("   %-28s %d" % (k, v))
    print("\n样例（case, key, type, 比什么, GT值, 我们值）：")
    for s in samples:
        print("   %-8s %-6s %-8s %-6s %-28s %s" % s)


if __name__ == "__main__":
    main()
