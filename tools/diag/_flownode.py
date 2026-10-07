"""打印若干节点的完整 JSON 与一条 link，用来照抄 schema。"""
import json
import re
import sys
from pathlib import Path

P = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
t = P.read_text(encoding="utf-8")
d = json.loads(re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)',
                         t, re.S).group(2))
want = sys.argv[1:] or ["rescan", "plausible"]
for n in d["nodes"]:
    if n.get("id") in want:
        c = {k: v for k, v in n.items() if k != "note"}
        print(json.dumps(c, ensure_ascii=False, indent=1))
        print("  note 长度 =", len(str(n.get("note") or "")))
print("--- link 样例")
print(json.dumps(d["links"][:3], ensure_ascii=False))
print("--- groups")
print(json.dumps(d["groups"], ensure_ascii=False))
