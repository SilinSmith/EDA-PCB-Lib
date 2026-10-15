# -*- coding: utf-8 -*-
"""交付折线的点数分布（只读）。

判分器 `net_line_f1` 里每条 edge 只比 `pe[:2]` 与 `ge[:2]`（unordered）。
GT 的 edge 平均只有 2.0 点（即单段直线）；若我方交付的 edge 是 3+ 点的折线，
那「覆盖但没配上 25.4%」很可能是**折线被头两点截断**造成的，而不是端点摆错。
"""
from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.metrics import _edge_points, normalise_doc  # noqa: E402


def edge_points(edge):
    return [(float(p[0]), float(p[1])) for p in (_edge_points(edge) or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args(argv)
    d = os.path.join(a.run, "final_results")
    hist = collections.Counter()
    npts = []
    for cid in [c for c in a.cases.split(",") if c]:
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if not os.path.exists(fp):
            cand = [p for p in glob.glob(os.path.join(d, "%s*.json" % cid))
                    if "Pred" in os.path.basename(p)]
            if not cand:
                print("case %s 没有产物" % cid)
                continue
            fp = cand[0]
        with open(fp, "r", encoding="utf-8") as fh:
            doc = normalise_doc(json.load(fh))
        n = 0
        for net in (doc.get("nets") or {}).values():
            for edge in (net.get("edges") or {}).values():
                pts = edge_points(edge)
                if len(pts) < 2:
                    continue
                n += 1
                hist[len(pts)] += 1
                npts.append(len(pts))
        print("case %-3s edge %3d" % (cid, n))
    if npts:
        arr = np.asarray(npts, dtype=float)
        print("\n交付 edge 点数分布：")
        for k in sorted(hist):
            print("   %2d 点: %5d  (%.1f%%)" % (k, hist[k], 100.0 * hist[k] / len(npts)))
        print("   合计 %d 条 edge；均值 %.2f 点；中位 %.1f；最长 %d"
              % (len(npts), float(arr.mean()), float(np.median(arr)), int(arr.max())))
        two = sum(v for k, v in hist.items() if k == 2)
        print("   ⇒ 恰好 2 点（判分器能完整比较）%d = %.1f%%；**3+ 点（被 pe[:2] 截断）%d = %.1f%%**"
              % (two, 100.0 * two / len(npts), len(npts) - two,
                 100.0 * (len(npts) - two) / len(npts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
