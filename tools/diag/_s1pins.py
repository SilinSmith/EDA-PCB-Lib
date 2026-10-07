import json
p = r"work\_submit_out220r36\runs\stage1\0001.stage1.json"
d = json.load(open(p, encoding="utf-8"))
pins = d.get("pins") or {}
print("stage1 pins 顶级键前 6 个:", list(pins)[:6])
c = list(pins)[0]
print("样例 %s ->" % c, json.dumps(pins[c][:3]))
print("是不是 cid 形态:", all(str(k).startswith("c") for k in list(pins)[:20]))
print("元件数:", len(d.get("components") or []), "  有引脚点集的元件:", len(pins))
print("components[0] 的键:", sorted((d.get("components") or [{}])[0].keys()))
