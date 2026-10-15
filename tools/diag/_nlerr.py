# -*- coding: utf-8 -*-
"""_nlerr.py — 未配上的 GT 导线段：误差性质分解（沿轴 / 横向 / 没画）

口径与 `schematic_parser/metrics.py:588-628 net_line_f1` 完全一致：
    cost = _seg_distance(pe[:2], ge[:2], cfg.line_unordered)   # 两端点距离取 max
    > cfg.line_tol(5.0) 记 INF；_assign = 匈牙利最大基数最小代价

本探针只回答一个问题：**判分配不上的 GT 段，到底错在哪**。
对每条未配上的 GT 段，在**同一个网对**里找真正距离最小的 pred 段（不设 tol）：

    no_cand   该网对里没有可比较的 pred 段
    far       最近 pred 段 cost > FAR(默认 30px)  ⇒ 真·没画
    near      最近 pred 段 cost <= FAR ⇒ 继续拆端点误差：
        perp_max   两端到 pred 段所在**直线**的垂距的较大者
        along_max  端点在直线上投影落到线段外的距离（>0 即线头越过/没到）
        ⇒ along_only（perp<=5 且 along>5：线是准的、只是长度/切点不对）
           perp_only （perp>5 且 along<=5：端点位置对、整条线偏了）
           both / tiny

同时给 GT 两端定结构类别（只看 GT 自己的图形）：
    pin   <=tol 到最近 GT 引脚      join  <=tol 到同网**别的** GT 导线段（真 T 接）
    box   <=tol 到最近 GT 元件框     blank 都不挨着（悬空头）

用法：python -X utf8 tools\\diag\\_nlerr.py --run ..\\work\\_fast_out_delivery_default
"""
from __future__ import print_function

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, _assign, _edge_points, _seg_distance, assign_components,
    net_hypergraph_f1, net_line_f1, normalise_doc, rewrite_keys,
)

INF = 1e12


def _pt_seg(p, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(ax + vx * t - p[0], ay + vy * t - p[1])


def _perp_along(p, a, b):
    """p 到直线 ab 的垂距；以及 p 在直线上的投影落到线段外的距离（0 = 在线段内）"""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L = math.hypot(vx, vy)
    if L <= 1e-9:
        return math.hypot(p[0] - ax, p[1] - ay), 0.0
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / (L * L)
    perp = math.hypot(ax + vx * t - p[0], ay + vy * t - p[1])
    off = 0.0
    if t < 0.0:
        off = -t * L
    elif t > 1.0:
        off = (t - 1.0) * L
    return perp, off


def _edges_of(net):
    """nets[*]['edges'] 是 **dict**（metrics.py:598 用 .values()），也兼容 list"""
    e = net.get("edges") or {}
    vals = e.values() if isinstance(e, dict) else e
    out = []
    for v in vals:
        p = _edge_points(v)
        if p is not None and len(p) >= 2:
            out.append(p[:2])
    return out


def _pins_of(doc):
    """GT / 交付结构：pins[comp][pin] = {'point': {'x','y'}}（doc 坐标）"""
    out = []
    for _ck, cv in (doc.get("pins") or {}).items():
        if not isinstance(cv, dict):
            continue
        if isinstance(cv.get("point"), dict) and "x" in cv["point"]:
            out.append((float(cv["point"]["x"]), float(cv["point"]["y"])))
            continue
        for _pk, pv in cv.items():
            if isinstance(pv, dict) and isinstance(pv.get("point"), dict) \
                    and "x" in pv["point"]:
                out.append((float(pv["point"]["x"]), float(pv["point"]["y"])))
    return out


def _boxes_of(doc):
    out = []
    for _k, c in (doc.get("components") or {}).items():
        bb = c.get("bbox") if isinstance(c, dict) else None
        if not isinstance(bb, dict):
            continue
        x1 = bb.get("x1", bb.get("left", bb.get("xmin")))
        y1 = bb.get("y1", bb.get("top", bb.get("ymin")))
        x2 = bb.get("x2", bb.get("right", bb.get("xmax")))
        y2 = bb.get("y2", bb.get("bottom", bb.get("ymax")))
        if None in (x1, y1, x2, y2):
            continue
        out.append((min(float(x1), float(x2)), min(float(y1), float(y2)),
                    max(float(x1), float(x2)), max(float(y1), float(y2))))
    return out


def _box_perim_d(p, boxes):
    best = 1e18
    for (x1, y1, x2, y2) in boxes:
        dx = max(x1 - p[0], 0.0, p[0] - x2)
        dy = max(y1 - p[1], 0.0, p[1] - y2)
        if dx > 0.0 or dy > 0.0:
            d = math.hypot(dx, dy)
        else:
            d = min(p[0] - x1, x2 - p[0], p[1] - y1, y2 - p[1])
        if d < best:
            best = d
    return best


def _end_class(p, gt_pins, own, gi, gt_all_nets, tol):
    """own = 本网 GT 边列表（每项是点序列），gi = 本条边的下标（排除自己）"""
    for q in gt_pins:
        if math.hypot(p[0] - q[0], p[1] - q[1]) <= tol:
            return "pin"
    for k, e in enumerate(own):
        if k == gi or len(e) < 2:
            continue
        if _pt_seg(p, e[0], e[1]) <= tol:
            return "join"
    if _box_perim_d(p, gt_all_nets) <= tol:
        return "box"
    return "blank"


def audit_case(gt_doc, pred_doc, cfg, far=30.0, tol=3.0, cap=6):
    gt = normalise_doc(gt_doc)
    pred = normalise_doc(pred_doc)
    _c, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)
    _n, pairs = net_hypergraph_f1(gt, pred2, cfg, {})
    lf = net_line_f1(gt, pred2, cfg, pairs, {})
    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    g_pins = _pins_of(gt)
    g_boxes = _boxes_of(gt)

    stat = collections.Counter()
    samples = []
    # 注意：net_hypergraph_f1 返回的 pairs 是 (pred_net_idx, gt_net_idx)
    for _pi, _gi in pairs:
        i, j = _gi, _pi
        if i >= len(g_nets) or j >= len(p_nets):
            continue
        ge = _edges_of(g_nets[i][1])
        pe = _edges_of(p_nets[j][1])
        if not ge:
            continue
        raw = [[_seg_distance(p, g, cfg.line_unordered) for g in ge] for p in pe]
        cost = [[(v if v <= cfg.line_tol else INF) for v in row] for row in raw]
        got = set(_assign(cost)) if (pe and ge) else set()
        matched_g = set(gj for _pi, gj in got)
        pred_used = {}
        for _pi, _gj in got:
            pred_used.setdefault(_pi, []).append(_gj)
        for gi, g in enumerate(ge):
            if gi in matched_g:
                stat["matched"] += 1
                continue
            stat["fn"] += 1
            ends = "%s+%s" % (_end_class(g[0], g_pins, ge, gi, g_boxes, tol),
                              _end_class(g[1], g_pins, ge, gi, g_boxes, tol))
            if not pe:
                stat["no_cand"] += 1
                stat["%s|no_cand" % ends] += 1
                continue
            bi = min(range(len(pe)), key=lambda k: raw[k][gi])
            best = raw[bi][gi]
            if best > far:
                stat["far"] += 1
                stat["%s|far" % ends] += 1
                continue
            stat["near"] += 1
            perp_max = 0.0
            along_max = 0.0
            for e in g:
                pp, oo = _perp_along(e, pe[bi][0], pe[bi][1])
                perp_max = max(perp_max, pp)
                along_max = max(along_max, oo)
            if perp_max > cfg.line_tol:
                kind = "lateral"      # 整条线横向偏了
            elif along_max > cfg.line_tol:
                kind = "underspan"    # GT 段端点戳出我方段之外 ⇒ 我方段短/沿轴错位
            else:
                kind = "overspan"     # GT 段整个落在我方段内部 ⇒ 我方段长
            stat[kind] += 1
            stat["%s|%s" % (ends, kind)] += 1
            stat["bcost_%s" % ("<=10" if best <= 10 else "<=20" if best <= 20 else "<=30")] += 1
            # ★ 关键区分：最近的 pred 段是不是**已经被别的 GT 段占掉**了
            #   compete ⇒ 我方一段跨了 GT 的好几段（切点不对，不是端点不准）
            #   free_pred ⇒ 那条 pred 段没配上任何 GT 段 ⇒ 纯粹端点误差/真没画
            if bi in pred_used:
                stat["compete"] += 1
                stat["compete|%s" % kind] += 1
                stat["compete_gap_%s" % ("le5" if best <= 5 else "le10" if best <= 10 else "le20")] += 1
            else:
                stat["free_pred"] += 1
                stat["freepred|%s" % kind] += 1
                stat["freepred_cost_%s" % ("le5" if best <= 5 else "le10" if best <= 10
                                           else "le20" if best <= 20 else "le30")] += 1
            if len(samples) < cap:
                samples.append({"gt": [[round(v, 1) for v in e] for e in g],
                                "pred": [[round(v, 1) for v in e] for e in pe[bi]],
                                "best": round(best, 2),
                                "perp": round(perp_max, 2),
                                "along": round(along_max, 2),
                                "ends": ends, "kind": kind})
    return stat, samples, lf.f1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_delivery_default")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--far", type=float, default=30.0)
    ap.add_argument("--end-tol", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=6)
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
        stat, samples, base = audit_case(load_json(case.json_path), load_json(fp),
                                         cfg, a.far, a.end_tol, a.samples)
        for k, v in stat.items():
            total[k] += v
        rows.append({"case": cid, "line": round(base, 4),
                     "stat": dict(stat), "samples": samples})
        print("case %-3s line=%.4f fn=%3d | near %3d (沿轴 %3d / 横 %3d / 都错 %3d / 微 %3d)"
              " | far %3d no_cand %3d"
              % (cid, base, stat["fn"], stat["near"], stat["along_only"],
                 stat["perp_only"], stat["both"], stat["tiny"],
                 stat["far"], stat["no_cand"]))

    print("\n== 合计 fn %d 条 GT 段 ==" % total["fn"])
    for k in ("matched", "fn", "near", "far", "no_cand", "along_only",
              "perp_only", "both", "tiny"):
        v = total[k]
        if v:
            print("   %-11s %4d  %5.1f%%" % (k, v, 100.0 * v / max(1, total["fn"])))
    print("   最近 pred cost 分布：" + "  ".join(
        "%s %d" % (k, total[k]) for k in ("bcost_<=10", "bcost_<=20", "bcost_<=30")
        if total[k]))
    print("\n   -- 端点结构 × 误差性质 --")
    keys = sorted(k for k in total if "|" in k)
    for k in keys:
        print("      %-22s %4d" % (k, total[k]))
    print("\n   -- ★ 最近的 pred 段是否已被占用（compete = 切点不对；free_pred = 端点误差）--")
    for k in sorted(k for k in total if k.startswith(("compete", "freepred"))):
        print("      %-24s %4d" % (k, total[k]))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, ensure_ascii=False, indent=1)
        print("\n   已写出 %s" % a.out)


if __name__ == "__main__":
    main()
