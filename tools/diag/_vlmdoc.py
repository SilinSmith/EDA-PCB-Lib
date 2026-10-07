# -*- coding: utf-8 -*-
"""看一份 VLM 补强后的文档结构，并与 stage2 原文逐字段比较。

用法：python _vlmdoc.py [case]
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

R32 = Path(r"Y:\workspace\EDA\work\_submit_out220r32\runs")
R36 = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs")
case = sys.argv[1] if len(sys.argv) > 1 else "0001"

vlm = json.loads((R32 / "stage3_vlm" / ("%s.json" % case)).read_text(encoding="utf-8"))
s2 = json.loads((R36 / "stage2" / ("%s.json" % case)).read_text(encoding="utf-8"))
asm = json.loads((R36 / "stage2" / "_assigned" / ("%s.json" % case)).read_text(encoding="utf-8"))

print("stage3_vlm 顶层键：", list(vlm.keys()))
comps = vlm.get("components")
print("components 类型 %s，长度 %s" % (type(comps).__name__,
                                       len(comps) if hasattr(comps, "__len__") else "?"))
if isinstance(comps, dict):
    k = list(comps)[0]
    print("样例 %s = %s" % (k, json.dumps(comps[k], ensure_ascii=False)[:700]))
else:
    print("样例 0 = %s" % json.dumps(comps[0], ensure_ascii=False)[:700])
print()
print("stage2 的 components 类型 %s" % type(s2.get("components")).__name__)
s2c = s2.get("components")
if isinstance(s2c, dict):
    k = list(s2c)[0]
    print("样例 %s = %s" % (k, json.dumps(s2c[k], ensure_ascii=False)[:700]))
print()
print("assigned.comps 样例 =",
      json.dumps(list((asm.get("comps") or {}).items())[:1], ensure_ascii=False)[:500])
print()
# 比较 vlm 与 stage2 在 components 上的字段差异
if isinstance(comps, dict) and isinstance(s2c, dict):
    n_diff = 0
    for cid, cv in comps.items():
        cs = s2c.get(cid) or {}
        for f in set(cv) | set(cs):
            if cv.get(f) != cs.get(f):
                n_diff += 1
                if n_diff <= 12:
                    print("  %s.%s: stage2=%r -> vlm=%r"
                          % (cid, f, cs.get(f), cv.get(f)))
    print("字段差异共 %d 处" % n_diff)
print()
print("vlm 的 pins 样例：",
      json.dumps(list((vlm.get("pins") or {}).items())[:1], ensure_ascii=False)[:400])
print("vlm 的 nets 样例：",
      json.dumps(list((vlm.get("nets") or {}).items())[:1], ensure_ascii=False)[:400])
