import json, sys
from pathlib import Path
sys.path.insert(0, "eda-pcb")
from stage1 import labels as L
import numpy as np
s1 = json.loads(Path("work/_gt10_pb10/runs/stage1/1.stage1.json").read_text(encoding="utf-8"))
segs = (s1.get("wire_detail") or {}).get("segments") or []
print("segments:", len(segs))
import itertools
for ext, weld, tol, nearest in itertools.product((8.0, 20.0, 35.0), (16.0, 40.0), (1.5, 4.0), (False, True)):
    g = L.wire_groups(segs, tol=tol, ext=ext, ext_nearest=nearest, weld=weld,
                      perp_tol=L.WIRE_PERP, cos_tol=L.WIRE_COS)
    import collections
    c = collections.Counter(g)
    big = sorted(c.values(), reverse=True)[:6]
    print("ext=%-5.1f weld=%-5.1f tol=%-3.1f nearest=%-5s -> 组 %3d  单段组 %3d  最大 %s"
          % (ext, weld, tol, nearest, len(c), sum(1 for v in c.values() if v == 1), big))
