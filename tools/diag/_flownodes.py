import json, re
from pathlib import Path
t = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', t, re.S)
d = json.loads(m.group(2))
print("nodes:", len(d["nodes"]), " edges:", len(d.get("edges") or []))
print()
for nd in d["nodes"]:
    note = (nd.get("note") or "")
    # 抓「进度：」后面到 ** 或 （ 之前的状态词
    mm = re.search(r"\*\*进度：\s*([^\*（(]*)", note)
    st = mm.group(1).strip() if mm else "(无进度标记)"
    flag = ""
    if "待更新" in note or "待更新" in st: flag = "  <<<< 待更新"
    print("%-14s %-22s len=%-5d%s" % (nd.get("id"), st[:22], len(note), flag))
