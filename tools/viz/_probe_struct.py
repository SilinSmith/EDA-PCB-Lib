import json, pathlib
for c in ("0048","0001"):
    d = json.loads(pathlib.Path("work/viz/cases/%s.json" % c).read_text(encoding="utf-8"))
    print("=====", c, "keys:", sorted(d.keys()))
    print("  score:", json.dumps(d.get("score"), ensure_ascii=False)[:400])
    for k in ("gt","ours"):
        sub = d.get(k) or {}
        print("  %s keys:" % k, sorted(sub.keys()))
        arr = sub.get("comps") or []
        print("   comps[0]:", json.dumps(arr[0], ensure_ascii=False) if arr else None)
    for k in ("texts","slots","regions","pinwins","labels","ops"):
        a = d.get(k) or []
        print("  %s n=%d [0]=" % (k, len(a)), json.dumps(a[0], ensure_ascii=False)[:340] if a else None)
    cp = d.get("comp_pins") or {}
    k0 = next(iter(cp), None)
    print("  comp_pins n=%d sample key=%s:" % (len(cp), k0), json.dumps(cp.get(k0), ensure_ascii=False)[:400] if k0 else None)
    print("  cid2gt:", json.dumps(d.get("cid2gt"), ensure_ascii=False)[:200])
    print("  cid2key:", json.dumps(d.get("cid2key"), ensure_ascii=False)[:200])
    for k in ("case_id","split","img","ow","oh","scale","assets","counts","diag"):
        if k in d: print("  %s:" % k, json.dumps(d[k], ensure_ascii=False)[:220])
