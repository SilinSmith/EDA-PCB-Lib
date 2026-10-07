import json, re
from pathlib import Path
s = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
g = json.loads(re.search(r'<script type="application/json" id="graphdata">(.*?)</script>', s, re.S).group(1))
for n in g["nodes"]:
    for field in ("note",):
        t = n.get(field) or ""
        for tok in ("0.30h", "27.33", "49.29"):
            for mm in re.finditer(re.escape(tok), t):
                a = max(0, mm.start()-70); b = mm.end()+50
                print("[%s/%s] …%s…" % (n["id"], tok, t[a:b].replace("\n"," ")))
