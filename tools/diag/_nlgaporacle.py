# -*- coding: utf-8 -*-
"""§5.126 5.2 离线神谕：**只治欠伸**的沿轴闭合（extend-only）上界。

父 agent 规格（m00396 第 2 条）：只治欠伸 —— `axial>0`（朝段外）且 `axial<=R` 且
`perp<=P` 才把端点搬到「两轴交点」；把**过伸/回缩**那条路整条排除在该开关之外。

本脚本在**已交付产物**上离线模拟两种候选规则，用 `_wirecut.score` 重算 `net_line_f1`，
在不跑管线的前提下扫完参数网格，判定该规则是否值得实现。

规则 A（cross，T/十字闭合）：端点的**自身轴延长线**与另一条边的**轴直线**相交于 X，
    要求 `0 < t <= R`（只外推）、端点到对方轴的垂距 `<= P`，且 X 落在对方跨度内
    （或离对方某端点 <= R）。
规则 B（collinear，共线缺口闭合）：另一条边与自身夹角 <= `--ang` 度，且对方**近端**
    在朝外方向 `0 < t <= R`、垂距 `<= P` 处 —— 把端点搬到对方近端。

CLI:
  python -X utf8 tools/diag/_nlgaporacle.py --dir work/_fast_out_deterministic_default
  python -X utf8 tools/diag/_nlgaporacle.py --dir work/_fast_out_delivery_default --rule B
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, load_json, DEFAULT_DATA_ROOT   # noqa: E402
from schematic_parser.metrics import EvalConfig                            # noqa: E402
from tools.diag import _wirecut                                            # noqa: E402


def _segs(doc):
    """[(netkey, edgekey, (p0, p1))] —— 只取边的前两点（判分口径）。"""
    out = []
    for nk, net in (doc.get("nets") or {}).items():
        eo = net.get("edges") if isinstance(net, dict) else None
        if not isinstance(eo, dict):
            continue
        for k, v in eo.items():
            if (isinstance(v, list) and len(v) >= 2
                    and isinstance(v[0], dict) and "x" in v[0]):
                out.append((nk, k, ((float(v[0]["x"]), float(v[0]["y"])),
                                    (float(v[1]["x"]), float(v[1]["y"])))))
    return out


def _anchored(p, segs, own, tol=3.0):
    """该端已经贴在别的边的某个顶点上 ⇒ 不是自由端，不动。"""
    for nk, k, (a, b) in segs:
        if (nk, k) == own:
            continue
        for q in (a, b):
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= tol:
                return True
    return False


def close_gaps(doc, rule="A", R=12.0, P=2.0, ang=12.0, cross=True,
               span_gate=True, free_only=True):
    """原地改 doc；返回 (移动端点数, 候选端点数)。"""
    segs = _segs(doc)
    ref = {}
    for nk, net in (doc.get("nets") or {}).items():
        eo = net.get("edges") if isinstance(net, dict) else None
        if isinstance(eo, dict):
            for k, v in eo.items():
                ref[(nk, k)] = v
    want = {}
    nfree = 0
    cosang = math.cos(math.radians(ang))
    for nk, k, (A0, A1) in segs:
        for idx, (p, q) in ((0, (A0, A1)), (1, (A1, A0))):
            ux, uy = p[0] - q[0], p[1] - q[1]
            L = math.hypot(ux, uy)
            if L < 1e-6:
                continue
            ux, uy = ux / L, uy / L
            if free_only and _anchored(p, segs, (nk, k)):
                continue
            nfree += 1
            best = None
            for nk2, k2, (C0, C1) in segs:
                if (nk2, k2) == (nk, k):
                    continue
                if not cross and nk2 != nk:
                    continue
                vx, vy = C1[0] - C0[0], C1[1] - C0[1]
                L2 = math.hypot(vx, vy)
                if L2 < 1e-6:
                    continue
                if rule == "B":
                    # 共线缺口：对方近端落在朝外方向 R 内、垂距 P 内
                    if abs(ux * vx + uy * vy) / L2 < cosang:
                        continue
                    for C in (C0, C1):
                        rx, ry = C[0] - p[0], C[1] - p[1]
                        t = rx * ux + ry * uy
                        if not (0.0 < t <= R):
                            continue
                        if abs(rx * (-uy) + ry * ux) > P:
                            continue
                        cand = (t, C)
                        if best is None or cand[0] < best[0]:
                            best = cand
                else:
                    vx, vy = vx / L2, vy / L2
                    cr = ux * vy - uy * vx
                    if abs(cr) < 1e-6:
                        continue
                    t = ((C0[0] - p[0]) * vy - (C0[1] - p[1]) * vx) / cr
                    if not (0.0 < t <= R):
                        continue
                    # 端点到对方**轴直线**的垂距
                    rx, ry = p[0] - C0[0], p[1] - C0[1]
                    if abs(rx * (-vy) + ry * vx) > P:
                        continue
                    X = (p[0] + t * ux, p[1] + t * uy)
                    if span_gate:
                        tj = ((X[0] - C0[0]) * vx + (X[1] - C0[1]) * vy) / L2
                        ok = -0.05 <= tj <= 1.05
                        if not ok:
                            for C in (C0, C1):
                                if 0.0 < math.hypot(C[0] - X[0], C[1] - X[1]) <= R:
                                    ok = True
                                    break
                        if not ok:
                            continue
                    cand = (t, X)
                    if best is None or cand[0] < best[0]:
                        best = cand
            if best is not None:
                want[(nk, k, idx)] = best[1]
    for (nk, k, idx), X in want.items():
        v = ref.get((nk, k))
        if v is None:
            continue
        v[idx]["x"] = round(X[0], 2)
        v[idx]["y"] = round(X[1], 2)
    return len(want), nfree


def shift_free_ends(doc, k, free_only=True):
    """把所有**自由端**沿自身轴向外推 k px（k<0 = 回缩）。返回动过的端点数。

    这是「端点沿轴位置」的**局部梯度**探针：不区分好坏端、不做任何门控，
    给出 `net_line_f1` 对自由端轴向位置的一阶灵敏度，用来判断当前状态是否
    已在局部最优、以及往哪个方向偏。
    """
    segs = _segs(doc)
    ref = {}
    for nk, net in (doc.get("nets") or {}).items():
        eo = net.get("edges") if isinstance(net, dict) else None
        if isinstance(eo, dict):
            for kk, v in eo.items():
                ref[(nk, kk)] = v
    want = {}
    for nk, kk, (A0, A1) in segs:
        for idx, (p, q) in ((0, (A0, A1)), (1, (A1, A0))):
            ux, uy = p[0] - q[0], p[1] - q[1]
            L = math.hypot(ux, uy)
            if L < 1e-6:
                continue
            ux, uy = ux / L, uy / L
            if free_only and _anchored(p, segs, (nk, kk)):
                continue
            want[(nk, kk, idx)] = (p[0] + k * ux, p[1] + k * uy)
    for (nk, kk, idx), X in want.items():
        v = ref.get((nk, kk))
        if v is None:
            continue
        v[idx]["x"] = round(X[0], 2)
        v[idx]["y"] = round(X[1], 2)
    return len(want)


def sweep_axial(a, cases, index, cfg):
    print("%-14s %8s %8s | mv=%s" % ("k(px)", "baseF1", "newF1", "动过的端"))
    ks = ([-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
           0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] if not a.ks
          else [float(x) for x in a.ks.split(",")])
    for k in ks:
        base_all, new_all, mv = [], [], 0
        for cid in cases:
            case = index.get(cid)
            fp = os.path.join(a.dir, "final_results", "%s_Pred.json" % cid)
            if case is None or not os.path.exists(fp):
                continue
            gt = load_json(case.json_path)
            pred = load_json(fp)
            base_all.append(float(_wirecut.score(gt, pred, cfg)[0].f1))
            mv += shift_free_ends(pred, k)
            new_all.append(float(_wirecut.score(gt, pred, cfg)[0].f1))
        if not base_all:
            continue
        bb = sum(base_all) / len(base_all)
        nn = sum(new_all) / len(new_all)
        print("%-14s %8.4f %8.4f | %+8.4f mv=%d"
              % ("k=%+.2f" % k, bb, nn, nn - bb, mv))


def quantise_ends(doc, mode="int", shift=0.0, free_only=False):
    """把所有边端点做**量化**：mode=``int`` 取整、``half`` 取到 0.5 格、``none`` 不动；
    ``shift`` 再沿自身轴平移（px）。返回动过的端点数。

    动机：骨架端点是像素域产物（常落在 x.3/x.5/x.7），而 GT 端点多为整数 ——
    量化可能是一次**稳健的**收窄；也可能只是噪声（用两口径同看可分辨）。
    """
    segs = _segs(doc)
    ref = {}
    for nk, net in (doc.get("nets") or {}).items():
        eo = net.get("edges") if isinstance(net, dict) else None
        if isinstance(eo, dict):
            for kk, v in eo.items():
                ref[(nk, kk)] = v
    want = {}
    for nk, kk, (A0, A1) in segs:
        for idx, (p, q) in ((0, (A0, A1)), (1, (A1, A0))):
            if free_only and _anchored(p, segs, (nk, kk)):
                continue
            x, y = p
            if mode == "int":
                x, y = round(x), round(y)
            elif mode == "half":
                x, y = round(x * 2.0) / 2.0, round(y * 2.0) / 2.0
            if shift:
                ux, uy = p[0] - q[0], p[1] - q[1]
                L = math.hypot(ux, uy)
                if L > 1e-6:
                    x, y = x + shift * ux / L, y + shift * uy / L
            want[(nk, kk, idx)] = (x, y)
    for (nk, kk, idx), X in want.items():
        v = ref.get((nk, kk))
        if v is None:
            continue
        v[idx]["x"] = round(X[0], 2)
        v[idx]["y"] = round(X[1], 2)
    return len(want)


def sweep_quant(a, cases, index, cfg):
    print("%-8s %-7s %-13s %8s %8s | %s"
          % ("mode", "shift", "free_only", "baseF1", "newF1", "delta"))
    for mode in ("int", "half"):
        for shift in (0.0, -0.5, -1.0):
            for fo in (False, True):
                base_all, new_all, mv = [], [], 0
                for cid in cases:
                    case = index.get(cid)
                    fp = os.path.join(a.dir, "final_results", "%s_Pred.json" % cid)
                    if case is None or not os.path.exists(fp):
                        continue
                    gt = load_json(case.json_path)
                    pred = load_json(fp)
                    base_all.append(float(_wirecut.score(gt, pred, cfg)[0].f1))
                    mv += quantise_ends(pred, mode, shift, free_only=fo)
                    new_all.append(float(_wirecut.score(gt, pred, cfg)[0].f1))
                if not base_all:
                    continue
                bb = sum(base_all) / len(base_all)
                nn = sum(new_all) / len(new_all)
                print("%-8s %-7.2f %-13s %8.4f %8.4f | %+8.4f mv=%d"
                      % (mode, shift, fo, bb, nn, nn - bb, mv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/_fast_out_deterministic_default")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--data", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--rule", default="A", choices=["A", "B", "AB", "AX", "Q"])
    ap.add_argument("--ks", default="")
    a = ap.parse_args()
    cases = [x.strip() for x in a.cases.split(",") if x.strip()]
    index = {}
    for c in discover(a.data):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()
    if a.rule == "AX":
        sweep_axial(a, cases, index, cfg)
        return
    if a.rule == "Q":
        sweep_quant(a, cases, index, cfg)
        return

    grid = []
    if a.rule in ("A", "AB"):
        for R in (5.0, 8.0, 12.0, 16.0):
            for P in (1.0, 2.0, 3.0):
                grid.append(("A", R, P, 12.0, True))
        grid.append(("A", 16.0, 2.0, 12.0, False))
    if a.rule in ("B", "AB"):
        for R in (8.0, 12.0, 16.0, 20.0):
            for P in (1.0, 2.0, 3.0):
                for ang in (8.0, 15.0, 30.0):
                    grid.append(("B", R, P, ang, True))

    hdr = "%-30s %8s %8s %8s | %s" % ("rule R/P/ANG(gate)", "baseF1", "newF1", "delta", "mv")
    print(hdr)
    print("-" * len(hdr))
    for (rule, R, P, ang, gate) in grid:
        base_all, new_all, per = [], [], []
        mv = cand = 0
        for cid in cases:
            case = index.get(cid)
            fp = os.path.join(a.dir, "final_results", "%s_Pred.json" % cid)
            if case is None or not os.path.exists(fp):
                continue
            gt = load_json(case.json_path)
            pred = load_json(fp)
            kb = float(_wirecut.score(gt, pred, cfg)[0].f1)
            m, nf = close_gaps(pred, rule=rule, R=R, P=P, ang=ang,
                               span_gate=gate, free_only=True)
            mv += m
            cand += nf
            kn = float(_wirecut.score(gt, pred, cfg)[0].f1)
            base_all.append(kb)
            new_all.append(kn)
            per.append((cid, round(kn - kb, 4)))
        if not base_all:
            continue
        bb = sum(base_all) / len(base_all)
        nn = sum(new_all) / len(new_all)
        print("%-30s %8.4f %8.4f %+8.4f | mv=%-4d free=%-4d %s"
              % ("%s R%.0f P%.0f A%.0f g%d" % (rule, R, P, ang, gate),
                 bb, nn, nn - bb, mv, cand,
                 " ".join("%s:%+.3f" % t for t in per
                          if abs(t[1]) > 1e-9) or "全 0"))


if __name__ == "__main__":
    main()
