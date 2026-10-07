# -*- coding: utf-8 -*-
import json
from pathlib import Path
p = Path("Y:/workspace/EDA/work/_submit_out220r36/runs/stage2/_assigned/1.json")
d = json.loads(p.read_text(encoding="utf-8"))
ds = d["desig_slots"]
print("desig_slots n=%d" % len(ds))
k0 = list(ds)[0]
print("key0=%r  val type=%s keys=%s" % (k0, type(ds[k0]).__name__,
      list(ds[k0]) if isinstance(ds[k0], dict) else "n=%d" % len(ds[k0])))
print(json.dumps(ds[k0], ensure_ascii=False)[:1200])
print("n_texts=%s n_owned=%s n_free=%s" % (d["n_texts"], d["n_owned"], d["n_free"]))
ss = d.get("struct_slots")
print("struct_slots type=%s n=%s" % (type(ss).__name__, len(ss) if ss is not None else None))
if ss:
    print(json.dumps(ss[0] if isinstance(ss, list) else list(ss.values())[0], ensure_ascii=False)[:400])
