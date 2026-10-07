import json
from pathlib import Path

p = Path(r"Y:\workspace\EDA\work\_fusion_out220nl\_score.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("top keys:", list(d.keys()))
print("n cases:", len(d["cases"]))
c = d["cases"][0]
print("case keys:", list(c.keys()))
print(json.dumps(c, ensure_ascii=False)[:1200])
