# -*- coding: utf-8 -*-
"""看 GT 数据里**一个元件**到底有哪些字段（尤其有没有位号文字框）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
import stage2_eval as SE                                                # noqa: E402

exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
want = sys.argv[1] if len(sys.argv) > 1 else "0001"
case, how = SE.resolve(want, exact, digits)
print("case:", case.case_id, "how:", how, "json:", case.json_path)
gt = load_json(case.json_path)
print("顶层键:", list(gt.keys()))
comps = gt.get("components") or {}
print("元件数:", len(comps))
gpin = gt.get("pins") or {}
for i, (k, v) in enumerate(comps.items()):
    if i >= 3:
        break
    print("\n--- components[%r] ---" % k)
    print(json.dumps(v, ensure_ascii=False, indent=2)[:1800])
    print("  pins:", json.dumps(gpin.get(k), ensure_ascii=False)[:400])
nets = gt.get("nets") or {}
print("\n网络数:", len(nets))
for i, (k, v) in enumerate(nets.items()):
    if i >= 2:
        break
    print("  nets[%r] = %s" % (k, json.dumps(v, ensure_ascii=False)[:300]))
