import json, re
p = r"work\flow\workflow.html"
s = open(p, encoding="utf-8").read()
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
g = json.loads(m.group(2))
for nid in ("pool","know","llm_judge","cropfix","ev_crop"):
    n = [x for x in g["nodes"] if x["id"]==nid][0]
    print("="*20, nid, "keys:", sorted(n.keys()))
    print((n.get("note") or "")[:900])
    print()
print("=== 全文里 '进度' 出现次数:", s.count("进度"))
print("=== 全文里 '已落地' 出现次数:", s.count("已落地"))
print("=== 全文里 '待更新' 出现次数:", s.count("待更新"))
