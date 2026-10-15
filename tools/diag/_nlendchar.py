# -*- coding: utf-8 -*-
"""_nlendchar.py — 未配上 GT 段的「单端就位」端点特征分解（§5.125 取证）

口径同 `metrics.py:588-628 net_line_f1`。对每条未配上的 GT 段，取**最近的我方段**
（不设 tol）并按有序配对算出两端误差；只统计「一端 <=5px、另一端 >5px」的
*单端就位* 类，然后回答：**错的那一端，可观测特征是什么**。

对错端 `q` 输出：
    cls       q 的 GT 结构类别（pin/join/box/blank，判据同 `_nlerr._end_class`）
    perp      q 到「我方段所在直线」的垂距     （>5 ⇒ 横向偏，沿轴补没用）
    t         q 沿我方段方向的投影弧长（<0 或 >L ⇒ 落在我方段跨度**外**）
    over      超出我方段跨度的距离（t<0 或 t>L 时的绝对值，否则 0）
    deg       我方段端点附近**骨架图**的度数（需要 --skel 时才算）
    dot       骨架分叉点附近是否有结点圆点墨（`stage1/labels.py _has_dot` 的对角判据）

用法：python -X utf8 tools\\diag\\_nlendchar.py --run ..\\work\\_fast_out_delivery_default
"""
from __future__ import print_function

import argparse
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, _assign, _seg_distance, assign_components,
    net_hypergraph_f1, net_line_f1, normalise_doc, rewrite_keys,
)

import _nlerr as N                                                       # noqa: E402

INF = 1e12
OFFICIAL = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def proj(p, a, b):
    """p 沿 a->b 的投影弧长 t（可越界）与到直线的垂距 perp。"""
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(vx, vy)
    if L < 1e-9:
        return 0.0, math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / (L * L)
    perp = math.hypot(a[0] + vx * t - p[0], a[1] + vy * t - p[1])
    return t * L, perp


def _bucket(v, edges=(2, 5, 8, 12, 20)):
    for e in edges:
        if v <= e:
            return "<=%g" % e
    return ">%g" % edges[-1]


def audit(gt_doc, pred_doc, cfg, far=30.0, tol=3.0, cap=40):
    gt = normalise_doc(gt_doc)
    pred = normalise_doc(pred_doc)
    _c, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)
    _n, pairs = net_hypergraph_f1(gt, pred2, cfg, {})
    lf = net_line_f1(gt, pred2, cfg, pairs, {})
    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    g_pins = N._pins_of(gt)
    g_boxes = N._boxes_of(gt)
    stat = collections.Counter()
    samples = []
    for _pi, _gi in pairs:                      # pairs 是 (pred_net_idx, gt_net_idx)
        i, j = _gi, _pi
        if i >= len(g_nets) or j >= len(p_nets):
            continue
        ge = N._edges_of(g_nets[i][1])
        pe = N._edges_of(p_nets[j][1])
        if not ge or not pe:
            continue
        raw = [[_seg_distance(p, g, cfg.line_unordered) for g in ge] for p in pe]
        cost = [[(v if v <= cfg.line_tol else INF) for v in row] for row in raw]
        got = set(_assign(cost))
        matched_g = set(gj for _pi2, gj in got)
        for gi, g in enumerate(ge):
            if gi in matched_g:
                continue
            bi = min(range(len(pe)), key=lambda k: raw[k][gi])
            if raw[bi][gi] > far:
                continue
            p = pe[bi]

            def dd(a, b):
                return math.hypot(a[0] - b[0], a[1] - b[1])

            # 有序：e0 对 g[0]，e1 对 g[1]；否则反向
            fwd = dd(p[0], g[0]) + dd(p[1], g[1])
            rev = dd(p[0], g[1]) + dd(p[1], g[0])
            if fwd <= rev:
                e0, e1 = dd(p[0], g[0]), dd(p[1], g[1])
            else:
                e0, e1 = dd(p[0], g[1]), dd(p[1], g[0])
            stat["unmatched"] += 1
            if max(e0, e1) <= cfg.line_tol:
                stat["would_pass_but_unmatched"] += 1
                continue
            if min(e0, e1) > cfg.line_tol:
                stat["both_ends_off"] += 1
                continue
            # ★ 单端就位
            stat["one_end_ok"] += 1
            bad = 0 if e0 > e1 else 1
            q = g[bad]
            cls = N._end_class(q, g_pins, ge, gi, g_boxes, tol)
            t, perp = proj(q, p[0], p[1])
            Lp = dd(p[0], p[1])
            outside = (t < 0.0) or (t > Lp)
            over = (-t) if t < 0.0 else ((t - Lp) if t > Lp else 0.0)
            stat["cls_" + cls] += 1
            stat["cls_%s_perp_%s" % (cls, _bucket(perp))] += 1
            stat["cls_%s_%s" % (cls, "outside" if outside else "inside")] += 1
            if outside:
                stat["cls_%s_out_%s" % (cls, _bucket(over))] += 1
            if len(samples) < cap:
                samples.append({"case": None, "cls": cls, "err": round(max(e0, e1), 1),
                                "perp": round(perp, 2), "t": round(t, 1),
                                "Lp": round(Lp, 1), "over": round(over, 1),
                                "gt_bad": [round(v, 1) for v in q],
                                "pred": [[round(v, 1) for v in p[0]],
                                         [round(v, 1) for v in p[1]]]})
    return stat, samples, lf.f1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_delivery_default")
    ap.add_argument("--cases", default=",".join(OFFICIAL))
    ap.add_argument("--far", type=float, default=30.0)
    ap.add_argument("--end-tol", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()
    d = os.path.join(a.run, "final_results")

    total = collections.Counter()
    rows = []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        stat, samples, base = audit(load_json(case.json_path), load_json(fp),
                                    cfg, a.far, a.end_tol, a.samples)
        for k, v in stat.items():
            total[k] += v
        for s in samples:
            s["case"] = cid
        rows.append({"case": cid, "line": round(base, 4),
                     "stat": dict(stat), "samples": samples})
        print("case %-3s line=%.4f  one_end_ok=%2d both_off=%2d | join=%2d blank=%2d pin=%2d box=%2d"
              % (cid, base, stat["one_end_ok"], stat["both_ends_off"],
                 stat["cls_join"], stat["cls_blank"], stat["cls_pin"], stat["cls_box"]))

    print("\n== 合计（%d 张）==" % len(rows))
    for k in ("one_end_ok", "both_ends_off", "would_pass_but_unmatched"):
        print("   %-26s %4d" % (k, total[k]))
    print("\n   -- 单端就位：错端的结构类别 --")
    for k in ("join", "blank", "pin", "box"):
        if total["cls_" + k]:
            print("      %-6s %4d   跨度外 %3d / 跨度内 %3d"
                  % (k, total["cls_" + k], total["cls_%s_outside" % k],
                     total["cls_%s_inside" % k]))
    print("\n   -- 错端垂距（到本方段直线）--")
    for k in sorted(k for k in total if "_perp_" in k):
        print("      %-26s %4d" % (k, total[k]))
    print("\n   -- 跨度外的超出量 --")
    for k in sorted(k for k in total if "_out_" in k):
        print("      %-26s %4d" % (k, total[k]))
    if a.out:
        import json
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, ensure_ascii=False, indent=1)
        print("\n   已写出 %s" % a.out)


if __name__ == "__main__":
    main()
