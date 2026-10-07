import json
p = r"work\_submit_out220r36\runs\stage2\_assigned\0001.json"
d = json.load(open(p, encoding="utf-8"))
comps = d["comps"]
n = 0
for cid, rec in comps.items():
    for fld in ("value", "partname"):
        v = rec.get(fld)
        if v:
            print("%s.%s -> %s" % (cid, fld, json.dumps(v, ensure_ascii=False)))
            n += 1
        if n >= 6: break
    if n >= 6: break
print("--- 字段统计 ---")
import collections
c = collections.Counter()
tot = 0
for cid, rec in comps.items():
    for fld in ("value", "partname", "designator"):
        v = rec.get(fld)
        if v:
            tot += 1
            c[(fld, tuple(sorted(v.keys())))] += 1
for k, v in c.most_common():
    print("  %-12s keys=%-40s n=%d" % (k[0], k[1], v))
print("有内容的字段总数", tot)
