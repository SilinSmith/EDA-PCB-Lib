# -*- coding: utf-8 -*-
"""数一数 stage2 产物里到底有多少个真歧义槽位（决定 LLM 预算）。"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from cvparts.hypo import MARGIN  # noqa: E402


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage2_v14\_assigned"
    fs = sorted(glob.glob(os.path.join(d, "*.json")))
    print("assigned 文件:", len(fs), " MARGIN =", MARGIN)
    tot = multi = alts = 0
    per = []
    for f in fs:
        with open(f, encoding="utf-8") as fh:
            a = json.load(fh)
        ds = a.get("desig_slots") or {}
        m = [c for c in ds.values() if c and len(c) > 1]
        tot += len(ds)
        multi += len(m)
        alts += sum(len(c or []) for c in ds.values())
        per.append((len(ds), len(m), os.path.basename(f)[:-5]))
    print("总槽位 %d  多候选槽位 %d  候选总数 %d" % (tot, multi, alts))
    print("每张平均：槽位 %.1f  多候选 %.1f" % (tot / len(fs), multi / len(fs)))
    per.sort(key=lambda x: -x[1])
    print("多候选最多的 10 张 (槽位, 多候选, case):")
    for row in per[:10]:
        print("   ", row)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
