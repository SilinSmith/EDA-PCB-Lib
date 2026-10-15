# -*- coding: utf-8 -*-
"""配不上的导线端点，到底该落在「引脚」还是「导线 T 接」上？（决定提取器先改哪一类结点）

口径与判分器一致（`schematic_parser/metrics.py:588-628 net_line_f1`）：
每个网对内 `cost = _seg_distance(pe[:2], ge[:2], unordered)`，`<= line_tol(5px)` 才算边，
再用 `_assign`（匈牙利、最大基数最小代价）配对；配不上的 pred 边计 fp、GT 边计 fn。

本脚本把**配不上的**端点分类：
  pin  = 最近的 GT 引脚在 3px 内
  join = 同网另一条 GT 导线（不是自己）在 3px 内（导线—导线 T 接/交叉）
  free = 两者都不是（真自由端，例如走向图幅外/被元件挡住）
同时记录「最近的 GT 端点距它多远」的分布，以及它离我方端点最近多远。

用法：python -X utf8 tools\\diag\\_endcause.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from pipeline import pcore, prun                                            # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json    # noqa: E402
from schematic_parser.metrics import (EvalConfig, _assign, _edge_points,     # noqa: E402
                                      _seg_distance, assign_components,
                                      net_hypergraph_f1, normalise_doc,
                                      rewrite_keys)


def pts_of(doc):
    """{net_key: [(edge_key, [(x,y)_img, ...]), ...]}"""
    out = {}
    for nkey, net in (doc.get("nets") or {}).items():
        rows = []
        for ekey, e in (net.get("edges") or {}).items():
            p = _edge_points(e)
            if p is None or len(p) < 2:
                continue
            rows.append((ekey, [(float(q[0]), float(q[1])) for q in p]))
        out[nkey] = rows
    return out


def pins_img(doc, h):
    out = []
    for ck, v in (doc.get("pins") or {}).items():
        if not isinstance(v, dict):
            continue
        for pk, p in v.items():
            pt = p.get("point") if isinstance(p, dict) else None
            if isinstance(pt, dict) and "x" in pt:
                out.append((float(pt["x"]), h - float(pt["y"])))
    return np.asarray(out, dtype=float) if out else np.zeros((0, 2))


def pt_seg_d(p, a, b):
    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    L2 = ab[0] * ab[0] + ab[1] * ab[1]
    if L2 < 1e-9:
        return float(np.hypot(ap[0], ap[1]))
    t = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1]) / L2))
    return float(np.hypot(ap[0] - t * ab[0], ap[1] - t * ab[1]))


def classify(p, gt_rows, gpin, tol=3.0):
    """p 是图像坐标；gt_rows = [(ekey, pts)] 属于这个网对。"""
    if gpin.shape[0]:
        d = np.hypot(gpin[:, 0] - p[0], gpin[:, 1] - p[1]).min()
        if d <= tol:
            return "pin", float(d)
    best = 1e9
    for _ek, pts in gt_rows:
        for i in range(len(pts) - 1):
            d = pt_seg_d(p, pts[i], pts[i + 1])
            if d < best:
                best = d
    if best <= tol:
        return "join", best
    return "free", best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    tally = {"fp": {}, "fn": {}}
    offs = {"fp": [], "fn": []}
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            continue
        inp = prun.inputs_of(pcore.find_case(cid))
        h = inp.h
        gt_raw = load_json(case.json_path)
        gt = normalise_doc(gt_raw)
        pred = normalise_doc(load_json(fp))
        _c, km = assign_components(gt, pred, cfg, {})
        pred = rewrite_keys(pred, km)
        _n, pairs = net_hypergraph_f1(gt, pred, cfg, {})
        G, P = pts_of(gt), pts_of(pred)
        g_nets, p_nets = list(gt["nets"].items()), list(pred["nets"].items())
        gpin = pins_img(gt, h)
        c_tally = {"fp": {}, "fn": {}}
        for i, j in pairs:
            p_rows = P.get(p_nets[i][0], [])
            g_rows = G.get(g_nets[j][0], [])
            if not p_rows or not g_rows:
                continue
            cost = []
            for _pk, pe in p_rows:
                row = []
                for _gk, ge in g_rows:
                    dd = _seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                    row.append(dd if dd <= cfg.line_tol else float("inf"))
                cost.append(row)
            got = set((int(x), int(y)) for x, y in _assign(cost))
            gp = np.asarray([q for _k, pts in g_rows for q in pts], dtype=float) if g_rows else np.zeros((0, 2))
            g_flat = [pts for _k, pts in g_rows for _q in pts]
            for pi, (ek, pe) in enumerate(p_rows):
                if any(pi == x for x, _y in got):
                    continue
                for end in (0, -1):
                    p = pe[end]
                    dd = np.hypot(gp[:, 0] - p[0], gp[:, 1] - p[1]) if gp.size else np.array([1e9])
                    k = int(np.argmin(dd))
                    near = g_flat[k]
                    cls, _d = classify(p, [("", near)], gpin)
                    c_tally["fp"][cls] = c_tally["fp"].get(cls, 0) + 1
                    offs["fp"].append(float(dd[k]))
            for gi, (ek, ge) in enumerate(g_rows):
                if any(gi == y for _x, y in got):
                    continue
                # ★ 必须把「自己的那条 GT 边」排除，否则端点到自己边的距离恒为 0，
                #   所有 fn 端点都会被判成 `join`（2026-10-14 首版就是这样，98.6% 的
                #   join 是这么来的假象）。
                others = [(k2, pts2) for i2, (k2, pts2) in enumerate(g_rows) if i2 != gi]
                for end in (0, -1):
                    p = ge[end]
                    cls, _d = classify(p, others, gpin)
                    c_tally["fn"][cls] = c_tally["fn"].get(cls, 0) + 1
                    if p_rows:
                        pa = np.asarray([[q[0], q[1]] for _k, pts in p_rows for q in pts], float)
                        offs["fn"].append(float(np.hypot(pa[:, 0] - p[0], pa[:, 1] - p[1]).min()))
        for tag in ("fp", "fn"):
            for k, v in c_tally[tag].items():
                tally[tag][k] = tally[tag].get(k, 0) + v
        print("case %-3s 端点分类 fp %s   fn %s" % (cid, c_tally["fp"], c_tally["fn"]))

    print("\n== 合计（官方 10 张，端点计数）==")
    for tag, name in (("fp", "我方多余边"), ("fn", "GT 缺失边")):
        tot = sum(tally[tag].values())
        print("  %s：共 %d 个端点" % (name, tot))
        for k in ("pin", "join", "free"):
            v = tally[tag].get(k, 0)
            print("     %-5s %4d  (%.1f%%)" % (k, v, 100.0 * v / max(tot, 1)))
    for tag, name in (("fp", "我方多余边端→最近 GT 端点"), ("fn", "GT 缺失边端→最近我方端点")):
        arr = np.asarray(offs[tag], dtype=float)
        if arr.size:
            print("  %s：中位 %.1f  p90 %.1f  <=5px %.1f%%  <=20px %.1f%%"
                  % (name, np.median(arr), np.percentile(arr, 90),
                     100.0 * float((arr <= 5).mean()), 100.0 * float((arr <= 20).mean())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
