# -*- coding: utf-8 -*-
"""看融合层把哪些位号改掉了，以及每个改动槽位的候选与来源。

    python work\tools\diag\_keydbg.py 0001 [0005 ...]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
PCB = ROOT / "eda-pcb"
sys.path.insert(0, str(PCB))

from fusion import claims as CL          # noqa: E402
from fusion import doc as DC             # noqa: E402
from fusion import priors as PR          # noqa: E402
from fusion import solve as SV           # noqa: E402

RUN = Path(os.environ.get("FUSION_RUN_DIR",
                          str(ROOT / "work" / "_submit_out220r36" / "runs")))
VLM = Path(os.environ.get("FUSION_VLM_DIR", str(ROOT / "work" / "_fusion_vlm")))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv):
    for case in argv or ["0001"]:
        doc = load(RUN / "stage1" / ("%s.stage1.json" % case))
        assigned = load(RUN / "stage2" / "_assigned" / ("%s.json" % case))
        assigned0 = json.loads(json.dumps(assigned))
        vlm = load(VLM / ("%s.json" % case)) if (VLM / ("%s.json" % case)).exists() else None
        pool = CL.from_stage2(case, doc, assigned, vlm=vlm)
        PR.Priors().annotate(pool)
        sol = SV.solve(pool)
        before = DC.invert_keymap(vlm, doc, assigned0) if vlm else {}
        dsrc = DC._decided_src(sol)
        print("== %s" % case)
        n = 0
        for sid, row in sorted((sol.get("assign") or {}).items()):
            if not sid.endswith(".key"):
                continue
            cid = sid.split(".")[1]
            cur = before.get(cid)
            got = str(row[0])
            if cur == got:
                continue
            n += 1
            print("  %s  %s -> %s   (由 %s 定, margin=%.3f)" % (
                sid, cur, got, row[2] if len(row) > 2 else "?", float(row[1])))
            for a in pool.alts(sid):
                mark = "*" if str(a.value) == got else " "
                print("      %s %-8s %.3f  %s" % (mark, a.value, a.score, a.source))
        print("  （%d 个槽位的键变了）" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
