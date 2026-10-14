# -*- coding: utf-8 -*-
"""§5.112 取证：**我方的 gnd/v 符号到底发了什么** —— 元件、pins 段、以及网里的引用。

用法（从 Y:\\workspace\\EDA\\eda-pcb 跑）：
    python -X utf8 ..\\work\\_pwrpred.py --run work/_fast_out_gsd9 --case 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import parse_hypergraph  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--case", default="5")
args = ap.parse_args()

p = ROOT / args.run / "final_results" / ("%s_Pred.json" % args.case)
doc = json.load(open(str(p), encoding="utf-8"))
comps = doc.get('components') or {}
pins = doc.get('pins') or {}
nets = doc.get('nets') or {}
print("Pred: %s" % p)
print("段：components %d  pins %d  nets %d" % (len(comps), len(pins), len(nets)))
pwr = [k for k, v in comps.items()
       if str((v or {}).get('type', '')).lower() in ('gnd', 'v', 'ground', 'power')]
print("电源类元件 %d 个：%s" % (len(pwr), sorted(pwr)[:14]))
for k in sorted(pwr)[:8]:
    print("  %-10s type=%-4s value=%-6s bbox=%s"
          % (k, (comps[k] or {}).get('type'), (comps[k] or {}).get('value'),
             [round(float(v)) for v in ((comps[k] or {}).get('bbox') or [])]))
    print("        pins 段: %s" % ({pk: (v or {}).get('pinname')
                                   for pk, v in (pins.get(k) or {}).items()}))
    innets = []
    for nname, net in nets.items():
        if not isinstance(net, dict):
            continue
        refs = parse_hypergraph(net.get('hyperGraph', ''))
        if any(c == k for c, _ in refs):
            innets.append((nname, sorted("%s.%s" % (a, b) for a, b in refs)))
    print("        出现在网: %s" % (innets or "（无）"))
print("\n所有网的 hyperGraph（前 20）：")
for nname, net in list(nets.items())[:20]:
    if isinstance(net, dict):
        print("   %-8s %s" % (nname, net.get('hyperGraph')))
