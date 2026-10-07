import json, sys
from pathlib import Path
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
d = json.loads(Path(r"Y:\workspace\EDA\work\_gt10_pb10\runs\stage1\1.stage1.json").read_text(encoding="utf-8"))
wd = d.get("wire_detail") or {}
print("wire_detail keys:", list(wd.keys())[:12])
segs = wd.get("segments") or []
print("segments:", len(segs), "sample:", segs[0] if segs else None)
import numpy as np
from stage1 import labels as L
S = L._as_segs(segs)
g = L.wire_groups(S)
print("groups:", len(set(int(x) for x in g)))
# 只看右半边 x>=1250
m = (S[:, 0] >= 1250) & (S[:, 2] >= 1250)
idx = np.where(m)[0]
print("右半边段数:", len(idx))
rows = []
for k in idx:
    x1, y1, x2, y2 = S[k][:4]
    L_ = ((x2-x1)**2 + (y2-y1)**2) ** .5
    rows.append((int(g[k]), round(L_), round(x1), round(y1), round(x2), round(y2)))
rows.sort()
from collections import defaultdict
byg = defaultdict(list)
for r in rows: byg[r[0]].append(r[1:])
print("\n右半边按组：")
for k in sorted(byg):
    v = byg[k]
    print("  组 %-4d 段数 %-3d : %s" % (k, len(v), v[:6]))
