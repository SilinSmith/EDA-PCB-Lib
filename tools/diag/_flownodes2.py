import json, re
from pathlib import Path
t = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', t, re.S)
d = json.loads(m.group(2))
print("顶层键:", list(d.keys()))
for k, v in d.items():
    if k != "nodes":
        print("  %s = %s" % (k, json.dumps(v, ensure_ascii=False)[:300]))
print()
for nd in d["nodes"]:
    if nd.get("id") in ("input", "ev_crop", "know"):
        print("----", nd.get("id"))
        for k, v in nd.items():
            s = json.dumps(v, ensure_ascii=False)
            print("   %-12s %s" % (k, s[:400]))
print()
print("=== 全文里出现的状态词 ===")
for w in ("未落地", "已落地", "已证否", "待更新", "TODO", "TBD", "未实现", "缺失", "待补"):
    print("  %-8s %d 次" % (w, t.count(w)))
