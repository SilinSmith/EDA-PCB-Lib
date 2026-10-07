"""打印 GT 与预测 JSON 的结构，供后续 dedup 实验用。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "1"
PRED_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"Y:\workspace\EDA\work\_fusion_out220no4")

c = D.find_case(CASE)
gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
pp = PRED_DIR / f"{c.case_id}.json"
pd = json.loads(pp.read_text(encoding="utf-8"))

print("case:", c.case_id, "split:", c.split)
print("GT 顶层键 :", list(gt.keys()))
print("PRED 顶层键:", list(pd.keys()))


def probe(doc, tag):
    comps = doc.get("components")
    print(f"[{tag}] components 类型 = {type(comps).__name__}")
    if isinstance(comps, dict):
        k0 = next(iter(comps))
        print(f"[{tag}] 首键 {k0!r} → {json.dumps(comps[k0], ensure_ascii=False)[:300]}")
        print(f"[{tag}] n_components = {len(comps)}")
    elif isinstance(comps, list):
        print(f"[{tag}] 首元素 → {json.dumps(comps[0], ensure_ascii=False)[:300]}")
        print(f"[{tag}] n_components = {len(comps)}")
    pins = doc.get("pins")
    print(f"[{tag}] pins 类型 = {type(pins).__name__}")
    if isinstance(pins, dict) and pins:
        k0 = next(iter(pins))
        print(f"[{tag}] pins 首键 {k0!r} → {json.dumps(pins[k0], ensure_ascii=False)[:300]}")
        print(f"[{tag}] n_pins = {len(pins)}")
    elif isinstance(pins, list) and pins:
        print(f"[{tag}] pins 首元素 → {json.dumps(pins[0], ensure_ascii=False)[:300]}")
        print(f"[{tag}] n_pins = {len(pins)}")
    for k in doc:
        if k not in ("components", "pins"):
            v = doc[k]
            s = json.dumps(v, ensure_ascii=False)
            print(f"[{tag}] {k} = {s[:200]}")


probe(gt, "GT")
print("-" * 80)
probe(pd, "PRED")
