# -*- coding: utf-8 -*-
"""形状闸诊断 v2：把 comp.c0003.key / comp.c0048.key 的**全部候选**连证据打全。

用法：python work\\tools\\diag\\_shapekeys2.py [case] [sid...]
"""
import json
import sys
from pathlib import Path

REPO = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(REPO / "eda-pcb"))

from fusion import claims as CL, priors as PR, solve as SV   # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "0001"
SIDS = sys.argv[2:] or ["comp.c0003.key", "comp.c0048.key"]
RUN = REPO / "work" / "_submit_out220r36" / "runs"
S1 = json.loads((RUN / "stage1" / ("%s.stage1.json" % CASE)).read_text(encoding="utf-8"))
ASG = json.loads((RUN / "stage2" / "_assigned" / ("%s.json" % CASE)).read_text(encoding="utf-8"))
VP = REPO / "work" / "_fusion_vlm" / ("%s.json" % CASE)
VLM = json.loads(VP.read_text(encoding="utf-8")) if VP.exists() else None

for gate in (True, False):
    PR.SHAPE_GATE = gate
    asg = json.loads(json.dumps(ASG))
    pool = CL.from_stage2(CASE, S1, asg, bgr=None, vlm=VLM)
    sol = SV.solve(pool)
    print("=" * 78, "SHAPE_GATE =", gate)
    for sid in SIDS:
        print("  ---", sid, "（solve 之前）best =", pool.best(sid))
        for c in pool.alts(sid):
            ev = c.evidence or {}
            print("      %-10s sc=%-8s src=%-22s 文本=%-12r 形状闸=%s also=%s" % (
                c.value, round(float(c.score), 3), c.source,
                ev.get("文本"), PR.desig_shape_ok(ev.get("文本") or c.value),
                [(d.get("source"), round(float(d.get("score") or 0), 3))
                 for d in (ev.get("also_from") or [])]))
    sol = SV.solve(pool)
    for sid in SIDS:
        print("  ---", sid, "（solve 之后）best =", pool.best(sid),
              " sol =", (sol.get(sid) if isinstance(sol, dict) else sol))
    # VLM 那一层到底加了什么
    for nt in pool.notes:
        if nt.get("kind") == "vlm_ingest":
            print("  vlm_ingest:", {k: v for k, v in nt.items() if k != "rows"})
    print("  闸掉的槽位:", [n for n in pool.notes if n.get("kind") == "shape_gate"])
