import json
from pathlib import Path
A = json.loads(Path("work/_fast_base/pinfix19.json").read_text(encoding="utf-8"))
B = json.loads(Path("work/_fast_base/edgemerge_off19.json").read_text(encoding="utf-8"))
key = lambda r: r.get("case") or r.get("case_id") or r.get("id")
ma = {key(r): r for r in A}; mb = {key(r): r for r in B}
print("%-10s %8s %8s | %s" % ("case", "pinfix", "off19", "delta(comp/pin/netH/line)"))
for k in ma:
    ra, rb = ma[k], mb.get(k)
    if not rb: continue
    sa, sb = float(ra.get("score", 0)), float(rb.get("score", 0))
    f = lambda r: [float((r.get(x) or {}).get("f1", 0)) for x in ("component", "pin", "net_hypergraph", "net_line")]
    fa, fb = f(ra), f(rb)
    print("%-10s %8.2f %8.2f | %s" % (k, sa, sb, " ".join("%+.3f" % (y - x) for x, y in zip(fa, fb))))
