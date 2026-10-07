# -*- coding: utf-8 -*-
"""`net_line` 为什么爆 0 —— 逐条看边的端点距离。

官方口径（`schematic_parser\\metrics.py`）：

* `net_line_f1` 只在 `match_nets` 配上的**网络对**里比边；
* 一条边算 tp 的条件是 `_seg_distance(我们的边, GT 的边) <= cfg.line_tol(5.0)`；
* 而 `_seg_distance` **不是线段到线段的距离**，它是
  `max(hypot(p0-q0), hypot(p1-q1))`（`unordered` 时再试一次反向）
  ⇒ **两个端点都必须落在 5px 内**。

所以只要我们把 GT 的一条长边切成 5 段，5 段全都算 fp，而 GT 那条算 fn。

用法::

    python work\\tools\\diag\\_linewhy.py --cases 1,2
    python work\\tools\\diag\\_linewhy.py --cases 1 --show 6
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, normalise_doc  # noqa: E402
from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, _net_sets, _edge_points, match_nets,
)
from stage2_eval import build_index, resolve, load_json  # noqa: E402


def segd(p, q, unordered=True):
    def m(a, b):
        return max(math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]),
                   math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]))
    f = m(p, q)
    return min(f, m(p, q[::-1])) if unordered else f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2")
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    pred_dir = Path(a.pred)
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case, _how = resolve(cid, exact, digits)
        if case is None:
            continue
        pf = pred_dir / ("%s_Pred.json" % case.case_id)
        if not pf.exists():
            pf = pred_dir / ("%s.json" % case.case_id)
        gt = normalise_doc(load_json(case.json_path))
        pr = normalise_doc(load_json(pf))
        g_names = list((gt.get("nets") or {}))
        p_names = list((pr.get("nets") or {}))
        g_sets = _net_sets(gt["nets"])
        p_sets = _net_sets(pr["nets"])
        pairs = match_nets(g_sets, p_sets)
        print("== case %s：GT net %d / 我们 net %d，配上 %d 对"
              % (case.case_id, len(g_sets), len(p_sets), len(pairs)))
        if not pairs:
            print("   ⇒ 一对都没配上，`net_line` 必然 0（全部边按 fp+fn 计）")
            continue
        dist = []
        plen, glen = [], []
        tp = 0
        shown = 0
        for i, j in pairs:
            pe = [(_edge_points(v)) for v in (pr["nets"][p_names[i]].get("edges") or {}).values()]
            ge = [(_edge_points(v)) for v in (gt["nets"][g_names[j]].get("edges") or {}).values()]
            pe = [e for e in pe if e]
            ge = [e for e in ge if e]
            for e in pe:
                plen.append(math.hypot(e[0][0] - e[-1][0], e[0][1] - e[-1][1]))
            for e in ge:
                glen.append(math.hypot(e[0][0] - e[-1][0], e[0][1] - e[-1][1]))
            for e in pe:
                best = min((segd(e[:2], g[:2]) for g in ge), default=float("inf"))
                dist.append(best)
                if best <= a.tol:
                    tp += 1
                if shown < a.show and len(ge):
                    nb = min(ge, key=lambda g: segd(e[:2], g[:2]))
                    print("     我们边 %s..%s 长 %6.1f | 最近 GT 边 %s..%s 长 %6.1f | 端距 %8.2f"
                          % (tuple(round(x, 1) for x in e[0]), tuple(round(x, 1) for x in e[-1]),
                             math.hypot(e[0][0] - e[-1][0], e[0][1] - e[-1][1]),
                             tuple(round(x, 1) for x in nb[0]), tuple(round(x, 1) for x in nb[-1]),
                             math.hypot(nb[0][0] - nb[-1][0], nb[0][1] - nb[-1][1]),
                             segd(e[:2], nb[:2])))
                    shown += 1
        n = len(dist)
        if not n:
            print("   配上的网络对里我们没有边")
            continue
        hist = collections.Counter()
        for d in dist:
            hist["<=5" if d <= 5 else "5~15" if d <= 15 else "15~40" if d <= 40
                 else "40~120" if d <= 120 else ">120"] += 1
        ngt = sum(len([1 for v in (gt["nets"][g_names[j]].get("edges") or {}).values()]) for _i, j in pairs)
        print("   我们边 %d 条（配上 tp %d），对应 GT 边 %d 条" % (n, tp, ngt))
        print("   端距直方 " + "  ".join("%s %d(%.2f)" % (k, hist[k], hist[k] / n)
                                       for k in ("<=5", "5~15", "15~40", "40~120", ">120") if hist[k]))
        if plen and glen:
            plen.sort(); glen.sort()
            print("   边长 我们 p50 %.1f p90 %.1f | GT p50 %.1f p90 %.1f"
                  % (plen[len(plen) // 2], plen[int(len(plen) * .9)],
                     glen[len(glen) // 2], glen[int(len(glen) * .9)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
