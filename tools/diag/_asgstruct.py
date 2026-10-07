# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
p = Path("Y:/workspace/EDA/work/_submit_out220r36/runs/stage2/_assigned/1.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("top keys:", list(d.keys()))
comps = d.get("comps") or {}
print("n comps:", len(comps))
k0 = list(comps)[0]
c0 = comps[k0]
print("comp keys:", list(c0.keys()))
for k, v in c0.items():
    if isinstance(v, list):
        print("  %s: list n=%d" % (k, len(v)))
        if v:
            print("     [0] type=%s" % type(v[0]).__name__)
            if isinstance(v[0], dict):
                print("     [0] keys=%s" % list(v[0].keys()))
                print("     [0] =%s" % json.dumps(v[0], ensure_ascii=False)[:600])
    elif isinstance(v, dict):
        print("  %s: dict n=%d keys=%s" % (k, len(v), list(v)[:8]))
    else:
        print("  %s: %r" % (k, v))
for kk in ("desig_slots", "texts", "slots"):
    if kk in d:
        print("top-level %s: n=%s" % (kk, len(d[kk])))
