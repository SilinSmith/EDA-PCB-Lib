# -*- coding: utf-8 -*-
"""fusion 层第一轮实测：拿真实产物建池 + 跑先验，看位号候选变成什么样。

用法：python _fz1.py [case]   默认 0001（= case 1，C151..C156 那组）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
sys.stdout.reconfigure(encoding="utf-8")

from fusion import claims as CL          # noqa: E402
from fusion import priors as PR          # noqa: E402

RUN = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs")
case = sys.argv[1] if len(sys.argv) > 1 else "0001"

s1 = RUN / "stage1" / ("%s.stage1.json" % case)
s2 = RUN / "stage2" / "_assigned" / ("%s.json" % case)
s2b = RUN / "stage2" / ("%s.json" % case)

pool = CL.from_file(case, s1, s2)
print("case %s：槽位 %d，取景框 %d，图 %s" % (
    case, len(pool.hy.slots), len(pool.regions),
    (pool.doc.get("image") or {}).get("path", "")[-60:]))

P = PR.Priors()
st = P.annotate(pool)
print("先验：%s" % st)

keys = [s for s in pool.slot_ids() if s.startswith("comp.") and s.endswith(".key")]
print("\n位号槽位 %d 个：" % len(keys))
for sid in sorted(keys):
    b = pool.best(sid)
    ctx = (pool.hy.slots[sid].context or {}).get("连号窗口候选")
    print("  %-18s best=%-10s windows=%s" % (sid, b.value if b else None, ctx))
    for a in pool.alts(sid)[:6]:
        print("        %-10s %.3f  %s" % (a.value, a.score, a.source))

bad = P.check(pool)
print("\n常识检查 %d 条：" % len(bad))
for b in bad[:20]:
    print("  %-18s %s" % (b["kind"], b.get("why")))

# 该图全部元件的 GT 位号（从官方 json 取）
gt = {}
for p in (Path(r"Y:\workspace\EDA\eda-pcb\yolo_part").glob("Dataset*")):
    for f in p.glob("**/annotation.json"):
        pass
print("\n取景框示例：")
for sid in sorted(keys)[:3]:
    r = pool.region(sid)
    if r:
        print("  %-18s box=%s note=%s" % (sid, [round(v) for v in r.box], r.note))
