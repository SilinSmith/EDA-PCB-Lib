import json, pathlib
for c in ("1","0048","0001","8"):
    d = json.loads(pathlib.Path("work/viz/cases/%s.json"%c).read_text(encoding="utf-8"))
    g = d.get("cid2gt") or {}
    print("==", c, "cid2gt n=%d"%len(g), "b-cids:", [k for k in g if k.startswith("b")])
    # 我方元件判定分布
    oc = (d.get("ours") or {}).get("comps") or []
    vd = {}
    for x in oc: vd[x["verdict"]] = vd.get(x["verdict"],0)+1
    print("   ours verdicts:", vd, "| n_slots=%d"%len(d.get("slots") or []),
          "| slots with gt:", sum(1 for s in (d.get("slots") or []) if s.get("gt")))
    print("   gt.comps with pred_key:", sum(1 for x in ((d.get("gt") or {}).get("comps") or []) if x.get("pred_key")))
