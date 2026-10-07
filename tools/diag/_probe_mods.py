"""Quick module reconnaissance + workflow node geometry dump."""
import json
import re
from pathlib import Path

HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)

print("=== stage2/expect.py ===")
p = Path(r"Y:\workspace\EDA\eda-pcb\stage2\expect.py")
s = p.read_text(encoding="utf-8")
print(s[:600].replace("\n", " | "))
print("defs:", re.findall(r"^(?:def|class)\s+(\w+)", s, re.M))

print()
print("=== stage1/reconcile.py ===")
p = Path(r"Y:\workspace\EDA\eda-pcb\stage1\reconcile.py")
s = p.read_text(encoding="utf-8")
print(s[:600].replace("\n", " | "))
print("defs:", re.findall(r"^(?:def|class)\s+(\w+)", s, re.M))

print()
print("=== 节点几何（用于插新节点）===")
g = json.loads(PAT.search(HTML.read_text(encoding="utf-8")).group(2))
for n in g["nodes"]:
    print("%-11s grp=%-3s x=%-5s y=%-5s w=%-4s title=%s" % (
        n["id"], n.get("group"), n.get("x"), n.get("y"), n.get("w"), n.get("title")))
print()
print("ins/outs of ev_text / ev_ocr:")
for n in g["nodes"]:
    if n["id"] in ("ev_text", "ev_ocr"):
        print(" ", n["id"], "ins=", n.get("ins"), "outs=", n.get("outs"))
print()
print("links touching ev_text / ev_ocr:")
for l in g["links"]:
    if "ev_text" in str(l) or "ev_ocr" in str(l):
        print("  ", l)
print()
print("node keys:", sorted(g["nodes"][0].keys()))
