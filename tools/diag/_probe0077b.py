import json
from pathlib import Path
for d in ("_fusion_out220nl","_fusion_out220nl2"):
    p = Path(r"Y:\workspace\EDA\work")/d
    for f in p.glob("*.json"):
        if f.stem in ("0077",):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f, "读取失败", e); continue
            print(d, f.name, "components:", len(doc.get("components") or {}))
            break
    else:
        print(d, "没有 0077.json")
