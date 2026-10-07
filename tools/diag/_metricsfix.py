import json, re
from pathlib import Path
HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)
s = HTML.read_text(encoding="utf-8"); m = PAT.search(s); g = json.loads(m.group(2))
OLD = "官方 Q&A 说这 10 张 `10GT` 是评测案例：现在 `10GT` **49.29**、`10_GTcase` 29.83、`200_train_cases` 26.34（总分 27.33）。"
NEW = "官方 Q&A 说这 10 张 `10GT` 是评测案例。~~曾记录 `10GT` 49.29、`10_GTcase` 29.83、`200_train_cases` 26.34（总分 27.33）~~ —— **这是旧快照，已作废**，正确口径见下方【第二批】。"
hit = 0
for n in g["nodes"]:
    if n["id"] == "metrics" and OLD in (n.get("note") or ""):
        n["note"] = n["note"].replace(OLD, NEW); hit += 1
blob = json.dumps(g, ensure_ascii=False, indent=1)
HTML.write_text(s[:m.start(2)] + blob + s[m.end(2):], encoding="utf-8")
print("replaced:", hit)
print("nodes=%d links=%d" % (len(g["nodes"]), len(g["links"])))
