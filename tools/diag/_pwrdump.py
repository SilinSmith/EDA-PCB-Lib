# -*- coding: utf-8 -*-
"""§5.112 取证：**GT 里电源符号（∅NNN）的键名与脚号是什么关系** —— 决定能不能复现。

打印某个 case 的 GT 文档里所有 `∅NNN` 元件：type / bbox / value / 它的 pins 段条目 /
它出现在哪些网的 hyperGraph 里，以及「键名数字」与「脚号」的差。
用法（从 Y:\\workspace\\EDA\\eda-pcb 跑）：python -X utf8 ..\\work\\_pwrdump.py --case 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc, parse_hypergraph  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="5")
ap.add_argument("--all", action="store_true")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
NUM = re.compile(r"(\d+)$")
diffs = {}
for cid in sorted(cases, key=lambda s: (len(s), s)):
    if not args.all and cid != args.case:
        continue
    raw = json.load(open(str(cases[cid].json_path), encoding="utf-8"))
    d = normalise_doc(raw)
    print("########## case %s  (%s)" % (cid, Path(cases[cid].json_path).name))
    n_pwr = 0
    for k in sorted(d['components']):
        if not k.startswith("∅"):
            continue
        c = d['components'][k] or {}
        pgroup = d['pins'].get(k) or {}
        nets = []
        for nname, net in d['nets'].items():
            if not isinstance(net, dict):
                continue
            refs = parse_hypergraph(net.get('hyperGraph', ''))
            if any(cc == k for cc, _ in refs):
                nets.append((nname, sorted("%s.%s" % (a, b) for a, b in refs)))
        m = NUM.search(k)
        kn = int(m.group(1)) if m else None
        for pk in pgroup:
            pn = pk[4:] if pk.startswith("pin_") else pk
            diffs.setdefault((kn - int(pn)) if (kn is not None and pn.isdigit()) else None,
                             []).append("%s.%s@%s" % (k, pn, cid))
        if args.all or True:
            print("  %-8s type=%-4s bbox=%s value=%s pins=%s"
                  % (k, c.get('type'), [round(float(v)) for v in (c.get('bbox') or [])],
                     c.get('value'), {pk: (v or {}).get('pinname')
                                      for pk, v in pgroup.items()}))
            for nname, refs in nets:
                print("        %-8s %s" % (nname, " ".join(refs)))
            n_pwr += 1
    print("    共 %d 个 ∅ 元件；网数 %d；pin 总数 %d"
          % (n_pwr, len(d['nets']), sum(len(v) for v in d['pins'].values() if isinstance(v, dict))))
print("\n== 「键名数字 − 脚号」分布 ==")
for k, v in sorted(diffs.items(), key=lambda kv: -len(kv[1])):
    print("   差 %-6s n=%-4d e.g. %s" % (k, len(v), ", ".join(v[:6])))
