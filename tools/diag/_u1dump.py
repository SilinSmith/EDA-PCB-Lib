import json, sys
from pathlib import Path
for tag, p in (("OLD(_gt10_s2h)", "work/_gt10_s2h/1.json"), ("NEW(pinbox)", "work/_gt10_pinbox/runs/stage2/1.json")):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    ps = d.get("pins", {}).get("U1") or {}
    rows = []
    for k, v in ps.items():
        pt = v.get("point") or {}
        rows.append((k, v.get("number"), v.get("pinname") or v.get("name"),
                     round(float(pt.get("x", 0))), round(float(pt.get("y", 0)))))
    rows.sort(key=lambda r: (r[3], r[4]))
    print("==", tag, "n=%d" % len(rows))
    for r in rows:
        print("   %-10s num=%-6s name=%-8s (%s,%s)" % r)
