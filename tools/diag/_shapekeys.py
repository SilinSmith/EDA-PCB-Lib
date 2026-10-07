# -*- coding: utf-8 -*-
"""形状闸诊断：0001 的 comp.*.key 候选表 + solve 结果，闸开/闸关各一遍。

用法：python work\\tools\\diag\\_shapekeys.py [case]
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(REPO / "eda-pcb"))

from fusion import claims as CL, priors as PR, solve as SV   # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "0001"
RUN = REPO / "work" / "_submit_out220r36" / "runs"
S1 = json.loads((RUN / "stage1" / ("%s.stage1.json" % CASE)).read_text(encoding="utf-8"))
ASG = json.loads((RUN / "stage2" / "_assigned" / ("%s.json" % CASE)).read_text(encoding="utf-8"))
VP = REPO / "work" / "_fusion_vlm" / ("%s.json" % CASE)
VLM = json.loads(VP.read_text(encoding="utf-8")) if VP.exists() else None
print("case %s  vlm_doc=%s" % (CASE, bool(VLM)))

for gate in (True, False):
    PR.SHAPE_GATE = gate
    asg = json.loads(json.dumps(ASG))
    pool = CL.from_stage2(CASE, S1, asg, bgr=None, vlm=VLM)
    print("=" * 70, "SHAPE_GATE =", gate, " notes:", pool.notes)
    sol = SV.solve(pool)
    for sid in sorted(pool.hy.slots):
        if not sid.startswith("comp."):
            continue
        b = pool.best(sid)
        v = str(b.value) if b else None
        if v is None or not v.startswith("U"):
            continue
        cs = sorted(((str(c.value), round(float(c.score), 3), c.source)
                     for c in pool.alts(sid)), key=lambda x: -x[1])
        mark = "  <== 被拒" if PR.SHAPE_GATE and any(
            not PR.desig_shape_ok(str((c.evidence or {}).get("文本") or c.value))
            for c in pool.alts(sid) if str(c.value) == v) else ""
        print("  %-18s best=%-6s sol=%-6s %s%s" % (
            sid, v, (sol.get(sid) if isinstance(sol, dict) else None), cs[:4], mark))
    print("  --- U1/U11 归属：")
    for sid in sorted(pool.hy.slots):
        if not sid.startswith("comp."):
            continue
        b = pool.best(sid)
        if b and str(b.value) in ("U1", "U11", "U580", "U0", "U2"):
            print("     ", sid, "->", b.value, round(float(b.score), 3), b.source,
                  "| sol =", (sol.get(sid) if isinstance(sol, dict) else None))
