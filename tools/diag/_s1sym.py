import json, sys, collections
from pathlib import Path
s1 = json.loads(Path("work/_gt10_pb10/runs/stage1/1.stage1.json").read_text(encoding="utf-8"))
comps = s1.get("components") or []
gid = {c["id"]: (c.get("type") or "?") for c in comps}
print("n comps", len(comps), "n nets", len(s1.get("nets") or []))
cnt = collections.Counter()
types = collections.Counter()
for c in comps: types[c.get("type")] += 1
print("types:", dict(types))
for k, grp in (s1.get("pins") or {}).items():
    cnt[gid.get(k, "?")] += len(grp)
print("pins per type:", dict(cnt))
for n in (s1.get("nets") or [])[:6]:
    print("   net pins:", [(gid.get(a), b) for a, b in (n.get("pins") or [])][:10])
# 有多少条 net 引用了 gnd/v
ref = sum(1 for n in (s1.get("nets") or []) if any(gid.get(a) in ("gnd","v") for a, _ in (n.get("pins") or [])))
print("nets referencing gnd/v:", ref, "/", len(s1.get("nets") or []))
