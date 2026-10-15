# -*- coding: utf-8 -*-
"""把单个 case 的 GT 元件 / 预测元件 / keymap 并排打印，用来定位「为什么普通位号配不上」。

用法：
  python -X utf8 work\tools\dev\_compdump.py --case 0001 --dir work\_ro220b
  python -X utf8 work\tools\dev\_compdump.py --case 0001 --dir work\_ro220b --limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(WORK / "tools" / "dev"))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import mk_official_textset as M            # noqa: E402
import stage2_eval as SE                   # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()

    idx, digits = M.build_index()
    case, how = SE.resolve(a.case, idx, digits)
    print("case=%s how=%s json=%s" % (a.case, how, getattr(case, "json_path", None)))
    gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
    pred = json.loads((Path(a.dir) / ("%s.json" % a.case)).read_text(encoding="utf-8"))
    gc, pc = gt.get("components") or {}, pred.get("components") or {}
    f1, keymap = assign_components(gt, pred, EvalConfig())
    covered = set(keymap.values())
    print("comp_f1=%.4f (tp %d fp %d fn %d)  GT %d  pred %d  keymap %d"
          % (f1.f1, f1.tp, f1.fp, f1.fn, len(gc), len(pc), len(keymap)))
    print("-- keymap（pred → gt）--")
    for k, v in list(keymap.items())[:a.limit]:
        g = gc.get(v) or {}
        p = pc.get(k) or {}
        print("   %-10s -> %-10s  gt type=%-6s bbox=%s   pred type=%-6s bbox=%s"
              % (k, v, g.get("type"), g.get("bbox"), p.get("type"), p.get("bbox")))
    print("-- GT 没配上 --")
    n = 0
    for k, g in gc.items():
        if k in covered:
            continue
        n += 1
        if n <= a.limit:
            print("   GT %-10s type=%-6s bbox=%s value=%r" % (k, g.get("type"), g.get("bbox"), g.get("value")))
    print("   ... 共 %d 个" % n)
    print("-- pred 没配上 --")
    m = 0
    for k, p in pc.items():
        if k in keymap:
            continue
        m += 1
        if m <= a.limit:
            print("   PRED %-10s type=%-6s bbox=%s value=%r" % (k, p.get("type"), p.get("bbox"), p.get("value")))
    print("   ... 共 %d 个" % m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
