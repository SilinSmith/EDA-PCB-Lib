# -*- coding: utf-8 -*-
"""用真实的 stage3.build 量「真歧义槽位」数，并统计 MARGIN 敏感性。

    python work\tools\diag\_slotdiff.py [stage2_dir] [--limit N]
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from cvparts.hypo import MARGIN, CaseHypotheses  # noqa: E402
from stage3.arbitrate import build as build_hy  # noqa: E402


def main():
    s2 = r"Y:\workspace\EDA\work\stage2_v14"
    limit = 0
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        s2 = args[0]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    s1dir = r"Y:\workspace\EDA\work\stage1_v6"
    fs = sorted(glob.glob(os.path.join(s2, "_assigned", "*.json")))
    if limit:
        fs = fs[:limit]
    print("stage2 =", s2, " MARGIN =", MARGIN, " cases:", len(fs))

    tot_alts = tot_slots = tot_multi = tot_unres = 0
    unres2 = 0
    for f in fs:
        case = os.path.basename(f)[:-5]
        with open(f, encoding="utf-8") as fh:
            assigned = json.load(fh)
        p = os.path.join(s1dir, case + ".stage1.json")
        doc = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        hy = build_hy(case, assigned, doc.get("components"))
        ds = assigned.get("desig_slots") or {}
        tot_slots += len(ds)
        tot_multi += sum(1 for c in ds.values() if c and len(c) > 1)
        tot_alts += sum(len(c or []) for c in ds.values())
        u = hy.unresolved(margin=MARGIN)
        tot_unres += len(u)
        unres2 += len(hy.unresolved(margin=0.5))
    n = len(fs)
    print("总槽位 %d (%.1f/张)" % (tot_slots, tot_slots / n))
    print("多候选槽位 %d (%.1f/张)" % (tot_multi, tot_multi / n))
    print("候选总数 %d (%.1f/张)" % (tot_alts, tot_alts / n))
    print("真歧义(MARGIN=%.2f) %d (%.1f/张)" % (MARGIN, tot_unres, tot_unres / n))
    print("真歧义(MARGIN=0.50) %d (%.1f/张)" % (unres2, unres2 / n))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
