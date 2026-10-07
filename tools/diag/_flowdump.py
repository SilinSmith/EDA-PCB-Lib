import json, re, sys
p = r"work\flow\workflow.html"
s = open(p, encoding="utf-8").read()
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
g = json.loads(m.group(2))
print("顶层键:", sorted(g.keys()), "| nodes:", len(g["nodes"]), "| links:", len(g.get("links") or []))
for n in g["nodes"]:
    nt = n.get("note") or ""
    prog = re.search(r"\*\*进度：\*\*\s*([^\n]*)", nt)
    print("%-12s kind=%-5s file=%-34s prog=%s | note=%d字" % (
        n["id"], n.get("kind"), (n.get("file") or "-")[:34],
        (prog.group(1).strip()[:22] if prog else "（无）"), len(nt)))
