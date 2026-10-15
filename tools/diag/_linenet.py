# -*- coding: utf-8 -*-
"""把**一个网**的多余/缺失段连坐标摊开（只读），用来判断偏移的方向与成因。

`_linedump.py` 已给出形状账（我方段比 GT 段长 4~6px、端点偏 6px；
吸圆点 `_dotsnap.py` 与端点聚类 `_nodecluster.py` 在 case 1 都是 **0.0000 变化**）。
本探针把「最差的那个网」逐段打坐标，并给出每个问题端点附近有什么：
  最近的 GT 端点 / 最近的**我方其它段端点**（能否聚类）/ 最近的**结点圆点**（能否吸附）/
  最近的**脚点** / 最近的**骨架分叉点**
⇒ 直接看出该用哪种可交付信号去校正，而不是继续猜。

用法：
    python -X utf8 tools\\diag\\_linenet.py --case 1 --nets 3
"""
from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover     # noqa: E402
from schematic_parser.metrics import (EvalConfig, _assign, _edge_points,
                                      _seg_distance, assign_components,
                                      net_hypergraph_f1, net_line_f1,
                                      normalise_doc, rewrite_keys)           # noqa: E402
from pipeline import pcore, prun                                            # noqa: E402
from _dotprobe import dots_of, flip                                         # noqa: E402

INF = float("inf")


def elen(e):
    return float(math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))


def dmin(p, arr):
    if len(arr) == 0:
        return float("inf")
    return float(np.min(np.hypot(arr[:, 0] - p[0], arr[:, 1] - p[1])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--case", default="1")
    ap.add_argument("--nets", type=int, default=2)
    ap.add_argument("--min-dt", type=float, default=3.0)
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    case = index[a.case]
    gt = normalise_doc(load_json(case.json_path))
    pred = normalise_doc(load_json(os.path.join(a.run, "final_results", "%s_Pred.json" % a.case)))
    det = {}
    _c, keymap = assign_components(gt, pred, cfg, det)
    pred2 = rewrite_keys(pred, keymap)
    _n, pairs = net_hypergraph_f1(gt, pred2, cfg, det)
    base = net_line_f1(gt, pred2, cfg, pairs, det).f1
    print("case %s 基线 %.4f" % (a.case, base))

    pc = pcore.find_case(a.case)
    inp = prun.inputs_of(pc)
    h = inp.h
    res = prun.skeleton_of(inp, "adaptive")
    _dt, dots = dots_of(np.asarray(res.ink).astype(bool), a.min_dt)
    dot_arr = np.asarray([[c[0], c[1]] for c in dots], dtype=float) if dots else np.zeros((0, 2))
    pin_arr = (np.asarray([[p.x, p.y] for p in inp.truth.pins.values()], dtype=float)
               if inp.truth.pins else np.zeros((0, 2)))
    jun_arr = (np.asarray([[p[0], p[1]] for p in res.junctions], dtype=float)
               if len(res.junctions) else np.zeros((0, 2)))

    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    scored = []
    for i, j in pairs:
        pnet, gnet = p_nets[i][1], g_nets[j][1]
        pe = [e[:2] for e in (_edge_points(v) for v in (pnet.get("edges") or {}).values()) if e]
        ge = [e[:2] for e in (_edge_points(v) for v in (gnet.get("edges") or {}).values()) if e]
        cost = [[(_seg_distance(x, y, cfg.line_unordered)
                  if _seg_distance(x, y, cfg.line_unordered) <= cfg.line_tol else INF)
                 for y in ge] for x in pe]
        got = _assign(cost)
        n_fp = len(pe) - len(got)
        n_fn = len(ge) - len(got)
        scored.append((n_fp + n_fn, i, j, pe, ge, got))
    scored.sort(key=lambda r: -r[0])
    for _bad, i, j, pe, ge, got in scored[:a.nets]:
        print("\n=== pred 网 %s / GT 网 %s：我方 %d 段 / GT %d 段，配上 %d ==="
              % (p_nets[i][0], g_nets[j][0], len(pe), len(ge), len(got)))
        mi = {r for r, _ in got}
        mj = {c for _, c in got}
        gt_img = [flip(e, h) for e in ge]
        pr_img = [flip(e, h) for e in pe]
        all_gt_ep = np.asarray([q for e in gt_img for q in e], dtype=float)
        all_pr_ep = np.asarray([q for e in pr_img for q in e], dtype=float)
        for c, (e, ei) in enumerate(zip(ge, gt_img)):
            tag = "配上" if c in mj else "缺失"
            print("  GT%02d %s len %6.1f  img (%.1f,%.1f)->(%.1f,%.1f)"
                  % (c, tag, elen(e), ei[0][0], ei[0][1], ei[1][0], ei[1][1]))
        for r, (e, ei) in enumerate(zip(pe, pr_img)):
            tag = "配上" if r in mi else "多余"
            extra = ""
            if r not in mi:
                # 到最近 GT 段两端
                cost2 = []
                for yy in gt_img:
                    m = max(min(math.hypot(ei[k][0] - yy[0][0], ei[k][1] - yy[0][1]),
                                math.hypot(ei[k][0] - yy[1][0], ei[k][1] - yy[1][1]))
                            for k in (0, 1))
                    cost2.append((m, yy))
                cost2.sort(key=lambda t: t[0])
                extra = "  最近GT段两端 max-min %5.1f" % cost2[0][0]
            print("  P%02d  %s len %6.1f  img (%.1f,%.1f)->(%.1f,%.1f)%s"
                  % (r, tag, elen(e), ei[0][0], ei[0][1], ei[1][0], ei[1][1], extra))
            if r not in mi:
                for k in (0, 1):
                    q = ei[k]
                    print("       端%d (%.1f,%.1f)：最近GT端 %5.1f | 最近我方其它端 %5.1f | "
                          "最近圆点 %5.1f | 最近骨架分叉 %5.1f | 最近脚点 %5.1f"
                          % (k, q[0], q[1],
                             dmin(q, all_gt_ep),
                             _dmin_other(q, pr_img, r),
                             dmin(q, dot_arr), dmin(q, jun_arr), dmin(q, pin_arr)))
    return 0


def _dmin_other(q, pr_img, skip):
    best = float("inf")
    for r2, e in enumerate(pr_img):
        if r2 == skip:
            continue
        for p in e:
            best = min(best, math.hypot(q[0] - p[0], q[1] - p[1]))
    return best


if __name__ == "__main__":
    sys.exit(main())
