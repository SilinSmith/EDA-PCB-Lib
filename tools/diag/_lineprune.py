# -*- coding: utf-8 -*-
"""离线 A/B：**删掉多余的导线段**能拿多少（可交付的结构判据）。

`_lineaudit.py` 的账里，最大的两块不是「端点偏」而是**段账**：
  fp 多余段 264（A1 重复 3 / A2 网没配上 3 / **A3 其它 261**）
  fn 缺失段 253（B1 被更长段跨过 40 / B2 被更短段盖住 42 / B3 没画 171）
而神谕读数：**只删掉多余的 264 段** ⇒ 0.5796 → **0.7328（+1.53 加权分）**；
只补上缺失的 253 段 ⇒ 0.8212（+2.42）。删比补便宜得多，而且 fp 是**我们自己发出来的**。

本探针只改交付文档，试几条**纯结构、可交付**的删段判据（都不用 GT）：
  len    段长 < L 的碎段
  sub    整条落在**同网另一条段内部**（冗余子段：两端都 in-line）
  bridge 两端都落在（同网）别的段的**内部**（把两条段"串"起来的桥段）
  stub   一端落在别的段内部、另一端又贴在别的段端点 3px 内（悬空短枝）
  all    上面并集（按长度优先删）
并给出每条的删段数与 F1。

用法：
    python -X utf8 tools\\diag\\_lineprune.py --len 8,12,20,30
"""
from __future__ import print_function

import argparse
import collections
import copy
import math
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover   # noqa: E402
from schematic_parser.metrics import (EvalConfig, _edge_points, net_line_f1,
                                      normalise_doc)                       # noqa: E402
from _dotsnap import score                                                 # noqa: E402


def elen(e):
    return float(math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))


def online(p, a, b, tol=2.0):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay) <= tol
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / L2
    if t < -0.02 or t > 1.02:
        return False
    px, py = ax + t * vx, ay + t * vy
    return math.hypot(p[0] - px, p[1] - py) <= tol


def prune(pred2, kinds, L, tols=(2.0, 3.0)):
    doc = copy.deepcopy(pred2)
    tot = collections.Counter()
    for _nk, net in (doc.get("nets") or {}).items():
        edges = net.get("edges") or {}
        segs = {}
        for k, v in edges.items():
            pts = _edge_points(v) or []
            if len(pts) >= 2:
                segs[k] = [(float(pts[0][0]), float(pts[0][1])),
                           (float(pts[-1][0]), float(pts[-1][1]))]
        drop = set()
        keys = list(segs)
        for k in keys:
            e = segs[k]
            if "len" in kinds and elen(e) < L:
                drop.add(k)
                tot["len"] += 1
                continue
            others = [segs[j] for j in keys if j != k]
            if "sub" in kinds and any(online(e[0], o[0], o[1], tols[0])
                                      and online(e[1], o[0], o[1], tols[0]) for o in others):
                drop.add(k)
                tot["sub"] += 1
                continue
            if "bridge" in kinds and others:
                t0 = [o for o in others if online(e[0], o[0], o[1], tols[1])]
                t1 = [o for o in others if online(e[1], o[0], o[1], tols[1])]
                if t0 and t1:
                    drop.add(k)
                    tot["bridge"] += 1
                    continue
            if "stub" in kinds and others:
                end_on = [o for o in others if online(e[0], o[0], o[1], tols[1])]
                if end_on:
                    near = any(min(math.hypot(e[1][0] - o[0][0], e[1][1] - o[0][1]),
                                   math.hypot(e[1][0] - o[1][0], e[1][1] - o[1][1])) <= tols[1]
                               for o in others)
                    if near:
                        drop.add(k)
                        tot["stub"] += 1
                        continue
                end_on = [o for o in others if online(e[1], o[0], o[1], tols[1])]
                if end_on:
                    near = any(min(math.hypot(e[0][0] - o[0][0], e[0][1] - o[0][1]),
                                   math.hypot(e[0][0] - o[1][0], e[0][1] - o[1][1])) <= tols[1]
                               for o in others)
                    if near:
                        drop.add(k)
                        tot["stub"] += 1
        for k in drop:
            edges.pop(k, None)
    return doc, tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--len", default="8,12,20,30")
    ap.add_argument("--kinds", default="len,sub,bridge,stub")
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    Ls = [float(x) for x in a.len.split(",") if x.strip()]
    kinds = [k for k in a.kinds.split(",") if k.strip()]
    variants = []
    for L in Ls:
        if "len" in kinds:
            variants.append(("len%g" % L, ["len"], L))
    for k in ("sub", "bridge", "stub"):
        if k in kinds:
            variants.append((k, [k], 0.0))
    if all(k in kinds for k in ("sub", "bridge", "stub")) and "len" in kinds:
        variants.append(("all%g" % Ls[-1], kinds, Ls[-1]))
    rows = []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        gt_doc = load_json(case.json_path)
        gt_norm = normalise_doc(gt_doc)
        pred_doc = load_json(fp)
        base, pred2, pairs = score(gt_doc, pred_doc, cfg)
        rec = {"case": cid, "base": base.f1, "v": {}}
        for name, ks, L in variants:
            doc, tot = prune(pred2, ks, L)
            sc = net_line_f1(gt_norm, doc, cfg, pairs, {})
            rec["v"][name] = (sc.f1, sum(tot.values()))
        rows.append(rec)
        line = "case %-3s 基线 %.4f" % (cid, rec["base"])
        for name, _ks, _L in variants:
            f1, nd = rec["v"][name]
            line += "  %s %.4f(%+.4f,-%d)" % (name, f1, f1 - rec["base"], nd)
        print(line)
    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张）--" % len(rows))
        print("   基线                        %.4f" % mb)
        for name, _ks, _L in variants:
            m = sum(r["v"][name][0] for r in rows) / n
            nd = sum(r["v"][name][1] for r in rows)
            print("   %-10s (可交付)         %.4f  (%+.4f)  ⇒ 加权 %+.2f 分  [共删 %d 段]"
                  % (name, m, m - mb, (m - mb) * 10.0, nd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
