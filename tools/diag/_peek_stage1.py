import json, sys
sys.stdout.reconfigure(encoding="utf-8")
p = r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage1\0001.stage1.json"
d = json.load(open(p, encoding="utf-8"))
ks = {}
for t in d["texts"]:
    ks[tuple(sorted(t.keys()))] = ks.get(tuple(sorted(t.keys())), 0) + 1
for k, n in ks.items():
    print("texts 键形 %d 个: %s" % (n, k))
for t in d["texts"]:
    if len(t) > 2:
        print("样例:", json.dumps(t, ensure_ascii=False)[:400]); break
a = json.load(open(r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2\_assigned\0001.json", encoding="utf-8"))
c = a["comps"]["c0000"]
print("assigned comps[c0000] 键:", list(c.keys()))
for k, v in c.items():
    if isinstance(v, list):
        print("   %-14s list[%d] %s" % (k, len(v), json.dumps(v[0], ensure_ascii=False)[:200] if v else ""))
    elif isinstance(v, dict):
        print("   %-14s dict[%d] %s" % (k, len(v), list(v.keys())[:6]))
    else:
        print("   %-14s %r" % (k, v))
