import json, sys, collections
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
d = json.load(open(r"work\_submit_out220r36\runs\stage2\0001.json", encoding="utf-8"))
a = json.load(open(r"work\_submit_out220r36\runs\stage2\_assigned\0001.json", encoding="utf-8"))
# cid -> designator 键：用 stage1 + 框重叠太麻烦，先看 assigned 里有没有线索
rec0 = a["comps"]["c0003"]
print("c0003 designator:", json.dumps(rec0.get("designator"), ensure_ascii=False)[:200])
print("c0003 pins keys:", list((rec0.get("pins") or {}).keys()))
for k, v in list((rec0.get("pins") or {}).items())[:4]:
    print("  k=%s number=%s name=%s" % (k, (v.get("number") or {}).get("text"), (v.get("name") or {}).get("text")))
print("---- stage1 是否有 cid -> key ----")
s1 = json.load(open(r"work\_submit_out220r36\runs\stage1\0001.stage1.json", encoding="utf-8"))
c = [x for x in s1["components"] if x["id"] == "c0003"][0]
print("stage1 comp c0003 字段:", sorted(c.keys()))
print("  ", json.dumps({k: c[k] for k in c if k not in ("bbox_img","bbox_json","bbox_raw_img")}, ensure_ascii=False)[:300])
