import json, pathlib
R = pathlib.Path("work/_submit_out220r36/runs")
for c in ("0048","1"):
    s1 = json.loads((R/"stage1"/("%s.stage1.json"%c)).read_text(encoding="utf-8"))
    for x in (s1.get("components") or []):
        if str(x.get("id")) == "b0000":
            print("==", c, "stage1 b0000 fields:", sorted(x.keys()))
            print("   ", json.dumps({k: v for k, v in x.items() if k != "pins"}, ensure_ascii=False)[:500])
