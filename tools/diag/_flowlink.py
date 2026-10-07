"""打印 link schema、groups 与关键节点坐标。"""
import json
import re
from pathlib import Path

t = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
d = json.loads(re.search(
    r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', t, re.S).group(2))
print("links[:4] =", json.dumps(d["links"][:4], ensure_ascii=False))
print("groups =", json.dumps(d["groups"], ensure_ascii=False))
for n in d["nodes"]:
    if n["id"] in ("pool", "llm_judge", "rescan", "iterate", "know",
                   "ev_ocr", "ev_vlm", "plausible", "conflict", "fixstr"):
        print("%-12s x=%-6s y=%-6s w=%-5s group=%s" % (
            n["id"], n.get("x"), n.get("y"), n.get("w"), n.get("group")))
print("node keys =", sorted({k for n in d["nodes"] for k in n}))
print("link keys =", sorted({k for l in d["links"] for k in l}))
