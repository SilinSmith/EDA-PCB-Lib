import json, sys
from pathlib import Path
pr = json.loads(Path("work/_gt10_pb10/runs/stage2/1.json").read_text(encoding="utf-8"))
comps = pr.get("components") or {}
for k, v in comps.items():
    if str(v.get("type")) in ("gnd", "v"):
        print("  ", k, v.get("type"), v.get("value"), v.get("bbox"))
print()
for nk, n in list((pr.get("nets") or {}).items())[:14]:
    print("  ", nk, str(n.get("hyperGraph"))[:110])
