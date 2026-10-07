import json, sys
from pathlib import Path
pr = json.loads(Path("work/_gt10_pb10/runs/stage2/1.json").read_text(encoding="utf-8"))
for k in ("p012","p013","p018","p020","C151","C156"):
    print(" ", k, "comp=", json.dumps(pr.get("components",{}).get(k), ensure_ascii=False)[:130])
    print("       pins=", json.dumps(pr.get("pins",{}).get(k), ensure_ascii=False)[:200])
s1 = json.loads(Path("work/_gt10_pb10/runs/stage1/1.stage1.json").read_text(encoding="utf-8"))
print("stage1 comps:", len(s1.get("components") or []))
for c in (s1.get("components") or []):
    if c.get("id") in ("p012","p013","p018","p020"):
        print("  s1", c.get("id"), c.get("type"), c.get("bbox_img"), "pins=", s1.get("pins",{}).get(c.get("id")))
