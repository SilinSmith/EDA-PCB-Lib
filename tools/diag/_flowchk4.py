import json, re
from pathlib import Path
t = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', t, re.S)
d = json.loads(m.group(2))
print("nodes:", len(d["nodes"]))
for nd in d["nodes"]:
    note = nd.get("note") or ""
    if "【第四批】" in note:
        print("  ", nd["id"], "| 第四批 出现", note.count("【第四批】"), "次 | len", len(note))
print("--- 我的 NODE_IDS 命中情况 ---")
ids = ("priors","pool","solve","plausible","llm_judge","rescan","iterate","conflict","fixstr")
for nd in d["nodes"]:
    if nd.get("id") in ids:
        note = nd.get("note") or ""
        has_mine = "28.4200" in note
        print("  %-10s 第四批:%s 含28.4200:%s" % (nd["id"], "【第四批】" in note, has_mine))
