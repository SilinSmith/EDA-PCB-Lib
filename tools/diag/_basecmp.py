import json
from pathlib import Path
for n in ("pinfix19", "minedge19", "edgemerge19", "edgemerge_off19"):
    p = Path("work/_fast_base/%s.json" % n)
    if not p.exists():
        print(n, "MISSING"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        m = sum(float(r.get("score", 0) or 0) for r in d) / max(1, len(d))
        tp = {k: sum(int((r.get(k) or {}).get("tp", 0) or 0) for r in d)
              for k in ("component", "pin", "net_hypergraph", "net_line")}
        print("%-18s mean=%.4f  n=%d  tp=%s" % (n, m, len(d), tp))
    else:
        print(n, list(d.keys())[:8])
