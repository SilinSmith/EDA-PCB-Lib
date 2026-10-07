import json, re
from pathlib import Path
R = Path(r"Y:\workspace\EDA")
for name in ("C180","C181","C185","C186","C77","C78"):
    hits = []
    for p in (R/"work/_fusion_vlm/0077.json", R/"work/_submit_out220r36/runs/stage1/0077.stage1.json"):
        t = p.read_text(encoding="utf-8")
        hits.append((p.name, t.count('"%s"' % name), t.count(name)))
    print(name, hits)
# 融合报告里 0077 一节
rep = json.loads((R/"work/_fusion_out220nl2/_report.json").read_text(encoding="utf-8"))
print("report type:", type(rep).__name__)
