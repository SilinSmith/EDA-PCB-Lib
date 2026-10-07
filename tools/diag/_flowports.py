import json, re
p = r"work\flow\workflow.html"
s = open(p, encoding="utf-8").read()
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
g = json.loads(m.group(2))
for nid in ("pool","llm_judge","rescan","know","solve"):
    n = [x for x in g["nodes"] if x["id"]==nid][0]
    print("%-10s ins=%s" % (nid, [(i["id"], i["label"]) for i in n.get("ins") or []]))
    print("%-10s outs=%s  x=%s y=%s" % ("", [(o["id"], o["label"]) for o in n.get("outs") or []], n["x"], n["y"]))
print()
print("--- 所有 link ---")
for l in g["links"]:
    print("  %-26s -> %-26s %s" % (l["from"], l["to"], l.get("kind")))
