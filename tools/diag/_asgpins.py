import json
a = json.load(open(r"work\_submit_out220r36\runs\stage2\_assigned\0001.json", encoding="utf-8"))
d = json.load(open(r"work\_submit_out220r36\runs\stage2\0001.json", encoding="utf-8"))
print("_assigned 顶层:", sorted(a.keys()))
comps = a["comps"]
for cid, rec in comps.items():
    pm = rec.get("pins") or {}
    if pm:
        print("样例 cid:", cid, "| type:", rec.get("type"), "| pins 键类型:", type(list(pm.keys())[0]).__name__, "| 键:", list(pm.keys())[:6])
        k0 = list(pm.keys())[0]
        print("  slot 字段:", sorted(pm[k0].keys()))
        print("  slot 样例:", json.dumps(pm[k0], ensure_ascii=False)[:400])
        break
print("---- doc pins 键 ----")
for dk, dv in list(d["pins"].items())[:3]:
    print(" ", dk, "->", list(dv.keys())[:6])
print("---- 能否用 cid 找 doc pins ----")
print("  pins.get('c0000'):", d["pins"].get("c0000"))
print("---- assigned comps key 形态 ----", list(comps.keys())[:5])
r0 = comps[list(comps.keys())[0]]
print("  rec 字段:", sorted(r0.keys()))
