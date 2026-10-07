import json, sys
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
d = json.load(open(r"work\_submit_out220r36\runs\stage2\0001.json", encoding="utf-8"))
print("顶层:", sorted(d.keys()))
c = d["components"]["U1"] if "U1" in d.get("components",{}) else list(d["components"].values())[0]
print("comp 字段:", sorted(c.keys()))
print("comp 样例:", json.dumps(c, ensure_ascii=False)[:400])
p = d.get("pins") or {}
k0 = list(p.keys())[0]
print("pins 顶层键样例:", k0, type(p[k0]).__name__)
v = p[k0]
if isinstance(v, list):
    print("pins[cid] 是 list，长度", len(v), "首元素:", json.dumps(v[0], ensure_ascii=False)[:300])
else:
    kk = list(v.keys())[0]
    print("pins[cid] 是 dict，键样例:", kk, "->", json.dumps(v[kk], ensure_ascii=False)[:300])
n = d.get("nets") or {}
nk = list(n.keys())[0] if isinstance(n, dict) else None
print("nets 类型:", type(n).__name__, "首键:", nk)
print("nets 首项:", json.dumps(n[nk] if nk else n[0], ensure_ascii=False)[:300] if n else "")
