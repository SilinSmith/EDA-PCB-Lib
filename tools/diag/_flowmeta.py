import json, re
p = r"work\flow\workflow.html"
s = open(p, encoding="utf-8").read()
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
g = json.loads(m.group(2))
print("groups:", json.dumps(g["groups"], ensure_ascii=False)[:600])
print()
print("第一条 link:", json.dumps(g["links"][0], ensure_ascii=False))
print("link 键全集:", sorted({k for l in g["links"] for k in l}))
print()
n = [x for x in g["nodes"] if x["id"]=="audit"][0]
print("audit 节点（去掉 note）:", json.dumps({k:v for k,v in n.items() if k!="note"}, ensure_ascii=False))
print()
print("与 audit 相关的 link:", json.dumps([l for l in g["links"] if "audit" in (l.get("from",""),l.get("to","")) or "audit" in str(l)], ensure_ascii=False)[:500])
print()
print("score:", json.dumps(g.get("score"), ensure_ascii=False)[:300])
print("note:", (g.get("note") or "")[:400])
