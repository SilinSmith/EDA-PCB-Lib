# -*- coding: utf-8 -*-
"""把「没配上的段」摊开看：到底是碎片、重复，还是位置错？（只读）

`_lineaudit.py` 给的是账：官方 10 张 net_line_f1 0.5796，fp 264（A1 重复 3 / A2 网没配上 3 /
A3 其它 261）、fn 253（B1 被更长段跨过 40 / B2 被更短段盖住 42 / B3 没画 171）；
而神谕「把未配上段的端点瞬移到 20px 内最近 GT 端点」能到 0.9224（+3.43 加权分）。
`_dotsnap.py` 进一步否证了「吸结点圆点」这条路（R8~12 只 +0.23，R>=15 转负）。

本探针回答「那 261/253 到底是什么形状」：
  · 长度分布（配上 / 多余 / 缺失 三类分开）—— 碎片应该短，重复应该与某条已配上段共线
  · 每条多余段：是否整条落在**某条 GT 段内部**（碎片/子段）还是**跨过** GT 段（过长）
  · 每条缺失 GT 段：是否整条落在**某条我方段内部**（只是没切）还是**我们根本没有**
  · 多余段与其最近 GT 段的端点距离（判断是「差 5~20px」还是「差得离谱」）

用法：
    python -X utf8 tools\\diag\\_linedump.py --cases 1 --top 15
"""
from __future__ import print_function

import argparse
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover       # noqa: E402
from schematic_parser.metrics import (EvalConfig, _assign, _edge_points,
                                      _seg_distance, assign_components,
                                      net_hypergraph_f1, net_line_f1,
                                      normalise_doc, rewrite_keys)             # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT as _DR                     # noqa: E402

INF = float("inf")


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


def dpe(p, e):
    """点 p 到段 e 两端点的最近距离与是哪一端。"""
    d0 = math.hypot(p[0] - e[0][0], p[1] - e[0][1])
    d1 = math.hypot(p[0] - e[1][0], p[1] - e[1][1])
    return (d0, 0) if d0 <= d1 else (d1, 1)


def bucket(x, edges=(10.0, 30.0, 80.0)):
    if x < edges[0]:
        return "  <10"
    if x < edges[1]:
        return "10-30"
    if x < edges[2]:
        return "30-80"
    return " >80"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        gt = normalise_doc(load_json(case.json_path))
        pred = normalise_doc(load_json(fp))
        det = {}
        _c, keymap = assign_components(gt, pred, cfg, det)
        pred2 = rewrite_keys(pred, keymap)
        _n, pairs = net_hypergraph_f1(gt, pred2, cfg, det)
        base = net_line_f1(gt, pred2, cfg, pairs, det).f1
        g_nets = list(gt["nets"].items())
        p_nets = list(pred2["nets"].items())

        # 全部 GT 段端点（全局），用于「最近 GT 端点」
        all_gt = []
        for _k, net in g_nets:
            for v in (net.get("edges") or {}).values():
                e = _edge_points(v)
                if e:
                    all_gt.append(e[:2])

        fp_rows, fn_rows = [], []
        hb = {"fp": collections.Counter(), "fn": collections.Counter(),
              "tp_p": collections.Counter(), "tp_g": collections.Counter()}
        frag = collections.Counter()
        for i, j in pairs:
            pnet, gnet = p_nets[i][1], g_nets[j][1]
            pe = [e[:2] for e in (_edge_points(v) for v in (pnet.get("edges") or {}).values()) if e]
            ge = [e[:2] for e in (_edge_points(v) for v in (gnet.get("edges") or {}).values()) if e]
            cost = [[(_seg_distance(x, y, cfg.line_unordered)
                      if _seg_distance(x, y, cfg.line_unordered) <= cfg.line_tol else INF)
                     for y in ge] for x in pe]
            got = _assign(cost)
            mi = {r for r, _ in got}
            mj = {c for _, c in got}
            for r, x in enumerate(pe):
                hb["tp_p" if r in mi else "fp"][bucket(elen(x))] += 1
            for c, y in enumerate(ge):
                hb["tp_g" if c in mj else "fn"][bucket(elen(y))] += 1
                if c in mj:
                    continue
                # 缺失段：落在哪条我方段内部 / 最近的我方段端点距离
                holder = [x for x in pe if online(y[0], x[0], x[1]) and online(y[1], x[0], x[1])]
                near = min(pe, key=lambda x: max(dpe(y[0], x)[0], dpe(y[1], x)[0])) if pe else None
                if holder:
                    frag["fn_inside_pred"] += 1
                else:
                    inside = [x for x in pe if online(x[0], y[0], y[1]) and online(x[1], y[0], y[1])]
                    frag["fn_pred_inside" if inside else "fn_no_overlap"] += 1
                if near is not None:
                    dd = [dpe(y[k], near) for k in (0, 1)]
                    fn_rows.append((elen(y), dd[0][0], dd[1][0], elen(near), bool(holder)))
            for r, x in enumerate(pe):
                if r in mi:
                    continue
                holder = [y for y in ge if online(x[0], y[0], y[1]) and online(x[1], y[0], y[1])]
                spans = [y for y in ge if online(y[0], x[0], x[1]) and online(y[1], x[0], x[1])]
                best = min(ge, key=lambda y: _seg_distance(x, y, cfg.line_unordered)) if ge else None
                if holder:
                    frag["fp_inside_gt"] += 1
                elif spans:
                    frag["fp_spans_gt"] += 1
                else:
                    frag["fp_partial"] += 1
                if best is not None:
                    dd = [dpe(x[k], best) for k in (0, 1)]
                    fp_rows.append((elen(x), dd[0][0], dd[1][0], elen(best),
                                    bool(holder), bool(spans)))

        print("\n===== case %s  基线 %.4f  （配对网 %d）=====" % (cid, base, len(pairs)))
        print("  长度分布  配上 pred %s | 多余 pred %s | 配上 GT %s | 缺失 GT %s"
              % (dict(hb["tp_p"]), dict(hb["fp"]), dict(hb["tp_g"]), dict(hb["fn"])))
        print("  多余段归类：整条落在某 GT 段内（碎片）%d / 跨过某 GT 段（过长）%d / 其它 %d"
              % (frag["fp_inside_gt"], frag["fp_spans_gt"], frag["fp_partial"]))
        print("  缺失段归类：整条在我方某段内（只是没切）%d / 我方段在它内部（短了）… %d / "
              "无重叠 %d" % (frag["fn_inside_pred"], frag["fn_pred_inside"], frag["fn_no_overlap"]))
        fp_rows.sort(key=lambda r: -r[0])
        fn_rows.sort(key=lambda r: -r[0])
        print("  -- 最长 %d 条**多余**段（长, 端到最近GT段两端距离, 最近GT段长, 在其内?, 跨过?）--"
              % a.top)
        for r in fp_rows[:a.top]:
            print("     len %7.1f  d0 %6.1f d1 %6.1f   GT len %7.1f  inside %-5s span %-5s"
                  % r)
        print("  -- 最长 %d 条**缺失**GT 段（长, 端到最近我方段两端距离, 最近我方段长, 在我方段内?）--"
              % a.top)
        for r in fn_rows[:a.top]:
            print("     len %7.1f  d0 %6.1f d1 %6.1f   pred len %7.1f  inside_pred %-5s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
