# -*- coding: utf-8 -*-
"""裁切方向反馈的现场试验：对可疑槽位问方向、按方向做外扩阶梯、看能不能读回完整串。

    python work\tools\diag\_cropfixtest.py [case ...] [--limit N] [--dry]
"""
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
PCB = ROOT / "eda-pcb"
sys.path.insert(0, str(PCB))

from fusion import claims as CL          # noqa: E402
from fusion import cropfix as CF         # noqa: E402
from fusion import llm_judge as LJ       # noqa: E402
from fusion import priors as PR          # noqa: E402

RUN = Path(os.environ.get("FUSION_RUN_DIR",
                          str(ROOT / "work" / "_submit_out220r36" / "runs")))
VLM = Path(os.environ.get("FUSION_VLM_DIR", str(ROOT / "work" / "_fusion_vlm")))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv):
    limit = 6
    dry = "--dry" in argv
    args = [a for a in argv if not a.startswith("--")]
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    cases = [a.zfill(4) for a in args] or ["0001"]
    if not dry:
        ok = LJ.preflight(strict=True)
        print("VLM 预检：%r" % (ok,))
    tot = {"sus": 0, "fix": 0, "gain": 0}
    for case in cases:
        s1 = RUN / "stage1" / ("%s.stage1.json" % case)
        asg = RUN / "stage2" / "_assigned" / ("%s.json" % case)
        if not s1.exists() or not asg.exists():
            print("跳过 %s（缺产物）" % case)
            continue
        vlm = VLM / ("%s.json" % case)
        pool = CL.from_stage2(case, load(s1), load(asg),
                              vlm=load(vlm) if vlm.exists() else None)
        PR.Priors().annotate(pool)
        sus = CF.suspects(pool)
        print("== %s  可疑 %d 条 / 槽位 %d" % (case, len(sus), len(pool.slot_ids())))
        shown = 0
        for sid in sus:
            b = pool.best(sid)
            seen = str(b.value) if b else ""
            if dry:
                print("   [dry] %-18s 「%s」 type=%s" % (
                    sid, seen, (pool.meta.get(sid) or {}).get("type")))
                shown += 1
                if shown >= limit:
                    break
                continue
            if shown >= limit:
                break
            row = CF.fix_slot(pool, sid, seen, log=None)
            print("   %-18s 「%s」 DIR=%s -> 「%s」 gain=%.2f %.1fs" % (
                sid, seen, row["dir"] or "-", row["ans"], row["gain"], row["sec"]))
            for x in row["ladder"]:
                print("        k%-4g -> %-10s score %5.2f  %.1fs" % (
                    x["k"], "「%s」" % x["ans"], x["score"], x["sec"]))
            tot["fix"] += 1
            tot["gain"] += 1 if row["gain"] > 0 else 0
            shown += 1
        tot["sus"] += len(sus)
    print("合计：可疑 %d，试了 %d，有增益 %d" % (tot["sus"], tot["fix"], tot["gain"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
