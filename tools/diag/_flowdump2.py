import json, re
p = r"work\flow\workflow.html"
s = open(p, encoding="utf-8").read()
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
g = json.loads(m.group(2))
RX = re.compile(r"\*\*进度：([^*]*)\*\*")
for n in g["nodes"]:
    nt = n.get("note") or ""
    mm = RX.search(nt)
    print("%-12s %-5s %-40s %s" % (n["id"], n.get("kind"), (n.get("file") or "-")[:40],
          ("| " + mm.group(1).strip()[:30]) if mm else ("| 无进度标记  (批次标记: %s)" % (re.findall(r"【第.批】", nt) or "无"))))
