# -*- coding: utf-8 -*-
"""把 GT / pred 的 components 与 pins 两个 section 的真实形状打出来。"""
import json, os, sys
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import find_case, load_json

case = sys.argv[1] if len(sys.argv) > 1 else "0001"
PRED = sys.argv[2] if len(sys.argv) > 2 else r"Y:\workspace\EDA\work\stage2_v14"

c = find_case(case)
gt = load_json(c.json_path)
pred = load_json(os.path.join(PRED, case + ".json"))

for tag, doc in (("GT", gt), ("PRED", pred)):
    print("=" * 20, tag, "=" * 20)
    comps = doc.get("components")
    print("components:", type(comps).__name__, len(comps) if comps else 0)
    if isinstance(comps, dict):
        for k, v in list(comps.items())[:2]:
            print("   [%s] =" % k, json.dumps(v, ensure_ascii=False)[:300])
    elif isinstance(comps, list):
        for v in comps[:2]:
            print("   ", json.dumps(v, ensure_ascii=False)[:300])
    pins = doc.get("pins")
    print("pins:", type(pins).__name__, len(pins) if pins else 0)
    if isinstance(pins, dict):
        for k, v in list(pins.items())[:4]:
            print("   [%s] =" % k, json.dumps(v, ensure_ascii=False)[:300])
    elif isinstance(pins, list):
        for v in pins[:4]:
            print("   ", json.dumps(v, ensure_ascii=False)[:300])
