import json, collections
from pathlib import Path
R = Path(r"Y:\workspace\EDA\work\_submit_out220r32\runs")
p = R/"stage3_vlm"/"0001.json"
d = json.loads(p.read_text(encoding="utf-8"))
print("top keys:", list(d.keys())[:30])
def walk(o, path=""):
    if isinstance(o, dict):
        for k,v in o.items(): yield from walk(v, path+"/"+str(k))
    elif isinstance(o, list):
        for i,v in enumerate(o[:3]): yield from walk(v, path+"[]")
    else:
        yield path, o
vals = list(walk(d))
print("leaf count:", len(vals))
# look for provenance-ish keys
prov = collections.Counter()
for k,v in vals:
    if any(t in k.lower() for t in ("by","src","source","vlm","llm","origin","from")):
        prov[(k,v if not isinstance(v,str) or len(str(v))<40 else "…")] += 1
for k,c in prov.most_common(20): print("  %-60s %d" % (str(k), c))
