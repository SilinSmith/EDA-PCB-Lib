# -*- coding: utf-8 -*-
"""_nlmove.py —— 对比两个产物目录：**边端点移动了多少**（不只看线 F1）。

用途：分离「交付产物里一条边的端点是由哪一步决定的」。对每个 case，把两边
`nets[*].edges[*]` 的端点做无向贪心配对（按中点近邻），统计：
  moved   两端任一端位移 >0.5px 的边数
  dx_med  移动端的位移中位（px）
  n_edge  总边数
另给整个文件的线段数差（配对不上的一侧）。

用法：
  python -X utf8 tools/diag/_nlmove.py --a Y:/workspace/EDA/work/_fast_out_abl_base5 \
      --b Y:/workspace/EDA/work/_fast_out_abl_ext0 --cases 2,3,6,9,10
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os


def _pts_of(v):
    if isinstance(v, dict):
        return v.get("points") or v.get("pts") or []
    if isinstance(v, (list, tuple)):
        return v
    return []


def _edges(fp):
    doc = json.load(io.open(fp, encoding="utf-8"))
    out = []
    for nk, net in (doc.get("nets") or {}).items():
        e = (net or {}).get("edges") or {}
        vals = list(e.values()) if isinstance(e, dict) else list(e)
        for v in vals:
            pts = _pts_of(v)
            if len(pts) >= 2:
                p0, p1 = pts[0], pts[-1]
                if isinstance(p0, dict):
                    a = (float(p0["x"]), float(p0["y"]))
                    b = (float(p1["x"]), float(p1["y"]))
                else:
                    a = (float(p0[0]), float(p0[1]))
                    b = (float(p1[0]), float(p1[1]))
                out.append((nk, a, b))
    return out


def _mid(e):
    return ((e[1][0] + e[2][0]) / 2.0, (e[1][1] + e[2][1]) / 2.0)


def _d(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def cmp_case(fa, fb):
    A, B = _edges(fa), _edges(fb)
    cand = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if a[0] != b[0]:              # 只比同一个网内的边
                continue
            cand.append((_d(_mid(a), _mid(b)), i, j))
    cand.sort()
    ua, ub = set(), set()
    pairs = []
    for _d0, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        pairs.append((i, j))
    moved, deltas = 0, []
    for i, j in pairs:
        a, b = A[i], B[j]
        d0 = _d(a[1], b[1])
        d1 = _d(a[2], b[2])
        d2 = _d(a[1], b[2])
        d3 = _d(a[2], b[1])
        if min(max(d0, d1), max(d2, d3)) > 0.5:
            moved += 1
            deltas.append(min(max(d0, d1), max(d2, d3)))
    ds = sorted(deltas)
    return dict(nA=len(A), nB=len(B), paired=len(pairs), moved=moved,
                med=(ds[len(ds) // 2] if ds else 0.0),
                mx=(ds[-1] if ds else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()
    print("%-6s %6s %6s %7s %7s %8s %8s" % ("case", "nA", "nB", "paired", "moved", "位移中位", "最大"))
    tot = dict(nA=0, nB=0, paired=0, moved=0)
    for cid in [x.strip() for x in a.cases.split(",") if x.strip()]:
        fa = os.path.join(a.a, "final_results", "%s_Pred.json" % cid)
        fb = os.path.join(a.b, "final_results", "%s_Pred.json" % cid)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print("%-6s 缺文件" % cid)
            continue
        r = cmp_case(fa, fb)
        for k in tot:
            tot[k] += r[k]
        print("%-6s %6d %6d %7d %7d %8.1f %8.1f"
              % (cid, r["nA"], r["nB"], r["paired"], r["moved"], r["med"], r["mx"]))
    print("%-6s %6d %6d %7d %7d" % ("ALL", tot["nA"], tot["nB"], tot["paired"], tot["moved"]))
    print("\nA = %s\nB = %s" % (a.a, a.b))


if __name__ == "__main__":
    main()
