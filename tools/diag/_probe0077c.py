import json
from pathlib import Path

p = Path(r"Y:\workspace\EDA\work\_fusion_out220nl2\_fusion_report.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("report type:", type(d).__name__, "len", len(d))
if isinstance(d, dict):
    print("keys:", list(d.keys())[:6])
    rows = d.get("cases") or d.get("rows") or []
else:
    rows = d
for r in rows:
    if str(r.get("case")) == "0077":
        print(json.dumps({k: v for k, v in r.items() if k != "rounds"},
                         ensure_ascii=False, indent=1)[:2500])
        break
