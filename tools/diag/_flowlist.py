"""列出 workflow.html 里的节点 id / title / 进度行。"""
import json
import re
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1
         else r"Y:\workspace\EDA\work\flow\workflow.html")
t = P.read_text(encoding="utf-8")
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', t, re.S)
d = json.loads(m.group(2))
print("keys =", list(d.keys()))
for n in d.get("nodes", []):
    note = str(n.get("note") or "")
    prog = ""
    mm = re.search(r"\*\*进度：([^*]{0,40})", note)
    if mm:
        prog = mm.group(1).strip()
    print("%-14s %-22s %-26s %s" % (n.get("id"), n.get("title"), n.get("label") or "",
                                    prog))
print("links =", len(d.get("links") or []))
print("groups =", [g.get("id") or g.get("title") for g in (d.get("groups") or [])])
