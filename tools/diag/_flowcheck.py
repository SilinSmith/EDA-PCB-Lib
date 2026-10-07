"""Sanity-check work/flow/workflow.html after patching: JSON validity, link/port integrity,
and that the new notes landed."""
import json
import re
from pathlib import Path

HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)

s = HTML.read_text(encoding="utf-8")
print("file bytes:", len(s.encode("utf-8")))
print("graphdata tags:", len(PAT.findall(s)))
m = PAT.search(s)
g = json.loads(m.group(2))
nodes = {n["id"]: n for n in g["nodes"]}
print("nodes=%d links=%d groups=%d" % (len(g["nodes"]), len(g["links"]), len(g["groups"])))
print("name:", g["name"])

def split(end):
    if isinstance(end, dict):
        return end.get("id"), end.get("port")
    s = str(end)
    if "." in s:
        a, b = s.split(".", 1)
        return a, b
    return s, None


bad = []
for l in g["links"]:
    f, t = split(l["from"]), split(l["to"])
    if f[0] not in nodes or t[0] not in nodes:
        bad.append(("missing node", l))
        continue
    fo = {o["id"] for o in nodes[f[0]].get("outs", [])}
    ti = {i["id"] for i in nodes[t[0]].get("ins", [])}
    if f[1] is not None and f[1] not in fo:
        bad.append(("bad out port", l))
    if t[1] is not None and t[1] not in ti:
        bad.append(("bad in port", l))
print("link problems:", len(bad))
for b in bad[:10]:
    print("   ", b)

for nid in ("ev_text", "ev_ocr"):
    n = nodes[nid]
    print("%s note=%d ch params=%s" % (nid, len(n.get("note") or ""), n.get("params")))
print("top note=%d ch" % len(g.get("note") or ""))
print("MARK in top:", "【尺度归一这一轮】" in (g.get("note") or ""))
