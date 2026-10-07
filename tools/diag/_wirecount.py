# -*- coding: utf-8 -*-
"""stage1 提出来的线 vs 最终进网络的边：差在哪一步？"""
import json, sys, os
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

case = sys.argv[1] if len(sys.argv) > 1 else "0001"
s1p = rf"Y:\workspace\EDA\work\stage1_v6\{case}.stage1.json"
d = json.load(open(s1p, encoding="utf-8"))

w = d.get("wires") or []
print("=== stage1 ===")
print("wires:", len(w), type(w).__name__)
it = (w.items() if isinstance(w, dict) else enumerate(w))
it = list(it)
if it:
    print("wire[%s]:" % it[0][0], json.dumps(it[0][1], ensure_ascii=False)[:400])
print("wire_detail:", type(d.get("wire_detail")).__name__,
      (list(d["wire_detail"])[:6] if isinstance(d.get("wire_detail"), dict) else ""))
det = d.get("wire_detail") or {}
for k in ("segments", "endpoints", "junctions", "groups"):
    v = det.get(k)
    print("  detail[%s]: %s %s" % (k, len(v) if hasattr(v, "__len__") else v,
                                   type(v).__name__))
    if v:
        it2 = (list(v.items()) if isinstance(v, dict) else list(enumerate(v)))
        print("     [0]:", json.dumps(it2[0][1], ensure_ascii=False)[:220])
print("  n_rays:", det.get("n_rays"))
# stage1 的 net 里一共引用了多少个 segment
seg_refs = set()
for n in (d.get("nets") or []):
    for i in (n.get("seg_index") or []):
        seg_refs.add(i)
print("stage1 被引用的 segment 数:", len(seg_refs))
nets = d.get("nets") or []
print("nets:", len(nets))
if nets:
    print("net[0]:", json.dumps(nets[0], ensure_ascii=False)[:500])
print("stats:", json.dumps(d.get("stats"), ensure_ascii=False)[:500])

# 最终 pred
pp = rf"Y:\workspace\EDA\work\stage2_v14\{case}.json"
pd = json.load(open(pp, encoding="utf-8"))
print("=== pred (stage2_v14) ===")
print("nets:", len(pd.get("nets") or {}))
e = 0
ne = 0
for k, v in (pd.get("nets") or {}).items():
    ne += len(v.get("edges") or {})
    e += 1
print("nets with edges:", e, " total edges:", ne)
k0 = next(iter(pd.get("nets") or {}), None)
if k0:
    print("net[%s]:" % k0, json.dumps(pd["nets"][k0], ensure_ascii=False)[:500])
