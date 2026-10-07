import json
from pathlib import Path
for tag in ("sym0", "sym1"):
    d = json.loads(Path("work/_gt10_%s/1.json" % tag).read_text(encoding="utf-8"))
    nets = d.get("nets") or {}
    withref = sum(1 for v in nets.values() if v.get("hyperGraph") and v["hyperGraph"] != "()")
    print(tag, "nets=", len(nets), "withref=", withref)
    if tag == "sym1":
        big = sorted(nets.items(), key=lambda kv: -len(str(kv[1].get("hyperGraph"))))[:3]
        for k, v in big:
            print("   ", k, str(v.get("hyperGraph"))[:200])
