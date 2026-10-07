# -*- coding: utf-8 -*-
"""探查两件事：
  ① `_assigned.comps` 的 cid 前缀（`c*` vs `b*`）与 stage1 `components[].id` 是否同一空间；
  ② GT 文档里 `pins` 的真实结构（键是什么、值是什么形状）。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

R = ROOT / "work/_submit_out220r36/runs"

for c in ("0048", "1", "0001"):
    asg = json.loads((R / "stage2/_assigned" / ("%s.json" % c)).read_text(encoding="utf-8"))
    s1 = json.loads((R / "stage1" / ("%s.stage1.json" % c)).read_text(encoding="utf-8"))
    ks = sorted((asg.get("comps") or {}).keys())
    pre = {}
    for k in ks:
        pre[k[0]] = pre.get(k[0], 0) + 1
    s1ids = sorted(str(x.get("id")) for x in (s1.get("components") or []))
    s1pre = {}
    for k in s1ids:
        s1pre[k[0]] = s1pre.get(k[0], 0) + 1
    print("==", c, "| asg comps", len(ks), pre, "| stage1 comps", len(s1ids), s1pre)
    print("   asg keys head:", ks[:6], "tail:", ks[-3:])
    b = [k for k in ks if k[0] == "b"]
    if b:
        k0 = b[0]
        print("   asg[%s] fields:" % k0, sorted((asg["comps"][k0] or {}).keys()))
        print("   asg[%s].designator:" % k0,
              json.dumps((asg["comps"][k0] or {}).get("designator"), ensure_ascii=False)[:300])
        print("   desig_slots[%s]:" % k0,
              json.dumps((asg.get("desig_slots") or {}).get(k0), ensure_ascii=False)[:300])
    print("   stage1 b-prefixed:", len([k for k in s1ids if k[0] == "b"]),
          "| asg c-prefix ⊆ stage1 ids:", set(k for k in ks if k[0] == "c") <= set(s1ids))

print()
from schematic_parser import data as D                                            # noqa: E402
cs = {x.case_id: x for x in D.discover()}
for c in ("0048", "0001"):
    cse = cs.get(c)
    if cse is None:
        print("GT 找不到", c)
        continue
    gt = json.loads(pathlib.Path(cse.json_path).read_text(encoding="utf-8"))
    print("== GT", c, "keys:", sorted(gt.keys()))
    comps = gt.get("components") or {}
    print("   components type:", type(gt.get("components")).__name__, "n=", len(comps))
    pins = gt.get("pins")
    print("   pins type:", type(pins).__name__, "n=", len(pins or {}))
    if isinstance(pins, dict):
        k0 = next(iter(pins), None)
        print("   pins key sample:", k0, "->", json.dumps(pins.get(k0), ensure_ascii=False)[:320])
    elif isinstance(pins, list):
        print("   pins[0]:", json.dumps(pins[0], ensure_ascii=False)[:320] if pins else None)
    ck = next(iter(comps), None)
    print("   components key sample:", ck, "->",
          json.dumps(comps.get(ck), ensure_ascii=False)[:320])
