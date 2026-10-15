# -*- coding: utf-8 -*-
"""离线 A/B：把交付导线端点沿自身轴**回收到「粗墨团（焊盘/结点圆点）的轴线质心」**能不能拿回亏损？

动机（§5.121）：net_line_f1 基线 0.5796，`_lineaudit.py` 量到亏损的**性质**是「端点偏 5~20px」：
- 端点瞬移到 **5px** 内最近 GT 端点只 +0.0036，**20px** 内 +3.43 ⇒ 瓶颈在 5~20px 这一段；
- 案例（case 1 排针扇出区）：GT `(1336,203)-(1432,203)`，我方 `(1429,202)-(1328,202.8)`
  —— 右端差 3.1px（能配上），**左端越过 8px**（判 fp+fn）。GT 端 = 焊盘中心，
  我方端 = 骨架追到**焊盘墨团的外缘**（焊盘直径≈16px ⇒ 外缘离中心正好 8px）。
- 失败的「吸附引脚」(`_wiretrim.py` −0.38) 用的是**我方引脚点**（pin_f1 0.8166，自己就偏），
  本探针改用**图像侧的墨宽**：端点若陷在一个「粗墨团」里，就把端点搬到该墨团沿轴的质心
  —— 这正是 GT 的语义（焊盘中心 / 结点圆点中心），且不需要引脚通道。

判定规则（每段每端，全部图像侧、可交付）：
1. 以该端点为原点、沿该段轴线取 `t ∈ [-24, +4]` px（负=往段内），垂直方向取 `s ∈ [-12, +12]` px；
2. 每行 `t` 量「包含轴线采样点的**连续墨段**宽度」⇒ 半宽 `w(t)`（轴线不在墨上则 0）；
3. 细线基准 `w_thin` = 全图所有**已配上导线**轴线上 `w(t)≥2` 的中位半宽（近似：本段自身中位）；
   粗区判据 `w(t) ≥ max(3.0, thr_mul × w_thin)`；
4. 找**包含端点（t=0）的那一段连续粗区** `[t0, t1]`；若粗区长 ≥ `min_run` px，
   把端点搬到轴线上的 `(t0+t1)/2`（移动量 ≤ `max_move` px；否则不动）。

用法：
    python -X utf8 tools\\diag\\_wireendtrim.py                       # 默认多组参数
    python -X utf8 tools\\diag\\_wireendtrim.py --cases 1,2,3
    python -X utf8 tools\\diag\\_wireendtrim.py --mode gt             # 只用神谕校验打分链
"""
from __future__ import print_function

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import pcore, prun                                          # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover   # noqa: E402
from schematic_parser.metrics import (EvalConfig, _edge_points, assign_components,
                                      net_hypergraph_f1, net_line_f1,
                                      normalise_doc, rewrite_keys)         # noqa: E402
from _dotprobe import flip                                               # noqa: E402


def score(gt_doc, pred_doc, cfg):
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    detail = {}
    _, keymap = assign_components(gt, pred, cfg, detail)
    pred2 = rewrite_keys(pred, keymap)
    _net, pairs = net_hypergraph_f1(gt, pred2, cfg, detail)
    line = net_line_f1(gt, pred2, cfg, pairs, detail)
    return line, pred2, pairs


def edge_ends_img(pred2, h):
    """[(p_img, other_img, handle)]，handle = (nkey, ekey, which)。"""
    out = []
    for nkey, net in (pred2.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge) or []
            if len(pts) < 2:
                continue
            for which, ref in ((0, 1), (-1, -2)):
                p = pts[which]
                q = pts[ref]
                out.append(((float(p[0]), h - float(p[1])),
                            (float(q[0]), h - float(q[1])),
                            (nkey, ekey, which)))
    return out


def apply_moves(doc, moves, h):
    n = 0
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge) or []
            if len(pts) < 2:
                continue
            new = [[float(q[0]), float(q[1])] for q in pts]
            touched = False
            for which in (0, -1):
                tgt = moves.get((nkey, ekey, which))
                if tgt is None:
                    continue
                new[which] = [float(tgt[0]), h - float(tgt[1])]
                touched = True
                n += 1
            if touched:
                net["edges"][ekey] = new
    return n


TS = np.arange(-24.0, 4.5, 1.0)
SS = np.arange(-12.0, 12.5, 1.0)
CI = int(np.argmin(np.abs(SS)))          # 轴线所在的列号


def width_profile(ink, p, q):
    """沿 p→(p - dir*q) 方向的半宽剖面：返回 (t 数组, w 数组, 轴线采样点是否在墨上)。"""
    dx, dy = p[0] - q[0], p[1] - q[1]
    L = float(np.hypot(dx, dy))
    if L < 1e-6:
        return TS, np.zeros_like(TS)
    dx, dy = dx / L, dy / L
    px, py = -dy, dx
    xs = p[0] + TS[:, None] * dx + SS[None, :] * px
    ys = p[1] + TS[:, None] * dy + SS[None, :] * py
    hgt, wid = ink.shape
    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)
    ok = (xi >= 0) & (xi < wid) & (yi >= 0) & (yi < hgt)
    vals = np.zeros(TS.shape[0], dtype=np.float64)
    np.clip(xi, 0, wid - 1, out=xi)
    np.clip(yi, 0, hgt - 1, out=yi)
    row_ink = ink[yi, xi] & ok
    for i in range(row_ink.shape[0]):
        row = row_ink[i]
        if not row[CI]:
            continue
        lo = CI
        while lo - 1 >= 0 and row[lo - 1]:
            lo -= 1
        hi = CI
        while hi + 1 < row.shape[0] and row[hi + 1]:
            hi += 1
        vals[i] = 0.5 * (SS[hi] - SS[lo]) + 0.5      # 半宽（px）
    return TS, vals


def blob_trim(ink, p, q, thr_mul, min_run, max_move, w_thin):
    """返回新端点 (x,y) 或 None。"""
    t, w = width_profile(ink, p, q)
    thr = max(3.0, thr_mul * max(w_thin, 1.0))
    big = w >= thr
    i0 = int(np.argmin(np.abs(t)))               # t≈0 的位置
    if not big[i0]:
        return None
    lo = i0
    while lo - 1 >= 0 and big[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < big.shape[0] and big[hi + 1]:
        hi += 1
    run = t[hi] - t[lo]
    if run < min_run:
        return None
    tm = 0.5 * (t[lo] + t[hi])
    if abs(tm) > max_move:
        return None
    dx, dy = p[0] - q[0], p[1] - q[1]
    L = float(np.hypot(dx, dy))
    if L < 1e-6:
        return None
    return (p[0] + tm * dx / L, p[1] + tm * dy / L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--mode", default="both", choices=["blob", "gt", "both"])
    ap.add_argument("--params", default="1.8/6/12,2.2/6/12,2.6/6/16,1.8/4/20,2.2/8/20")
    ap.add_argument("--gt-r", type=float, default=20.0)
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    combos = []
    for chunk in a.params.split(","):
        if not chunk.strip():
            continue
        tm, mr, mm = [float(x) for x in chunk.split("/")]
        combos.append((tm, mr, mm))
    rows = []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        pc = pcore.find_case(cid)
        inp = prun.inputs_of(pc)
        res = prun.skeleton_of(inp, "adaptive")
        h = inp.h
        ink = np.asarray(res.ink).astype(bool)
        gt_doc = load_json(case.json_path)
        pred_doc = load_json(fp)
        base, pred2, pairs = score(gt_doc, pred_doc, cfg)
        ends = edge_ends_img(pred2, h)
        # 细线基准：所有端点往段内 12px 处的半宽中位（多数段是细线）
        ws = []
        for p, q, _hd in ends:
            _t, w = width_profile(ink, p, q)
            k = int(np.argmin(np.abs(TS + 12.0)))
            if w[k] > 0:
                ws.append(w[k])
        w_thin = float(np.median(ws)) if ws else 2.0
        rec = {"case": cid, "base": base.f1, "n_ep": len(ends), "w_thin": w_thin, "blob": {}}
        gt_eps = []
        for _n, net in (normalise_doc(gt_doc).get("nets") or {}).items():
            for _e, edge in (net.get("edges") or {}).items():
                pl = _edge_points(edge) or []
                if len(pl) >= 2:
                    f = flip([pl[0], pl[-1]], h)
                    gt_eps.append(f[0])
                    gt_eps.append(f[1])
        gt_arr = np.asarray(gt_eps, dtype=float) if gt_eps else np.zeros((0, 2))
        if a.mode in ("blob", "both"):
            for combo in combos:
                pd = copy.deepcopy(pred2)
                moves = {}
                for p, q, hd in ends:
                    newp = blob_trim(ink, p, q, combo[0], combo[1], combo[2], w_thin)
                    if newp is not None:
                        moves[hd] = newp
                n_moved = apply_moves(pd, moves, h)
                sc = net_line_f1(normalise_doc(gt_doc), pd, cfg, pairs, {})
                rec["blob"][combo] = (sc.f1, n_moved)
        if a.mode in ("gt", "both") and len(gt_arr):
            pd = copy.deepcopy(pred2)
            moves = {}
            for p, _q, hd in ends:
                dd = np.hypot(gt_arr[:, 0] - p[0], gt_arr[:, 1] - p[1])
                k = int(np.argmin(dd))
                if dd[k] <= a.gt_r:
                    moves[hd] = (float(gt_arr[k, 0]), float(gt_arr[k, 1]))
            n_moved = apply_moves(pd, moves, h)
            sc = net_line_f1(normalise_doc(gt_doc), pd, cfg, pairs, {})
            rec["gt"] = (sc.f1, n_moved)
        rows.append(rec)
        parts = ["case %-3s 基线 %.4f  端点 %3d  w_thin %.2f" % (cid, rec["base"], rec["n_ep"],
                                                                rec["w_thin"])]
        for combo in combos:
            if combo in rec["blob"]:
                f1, nm = rec["blob"][combo]
                parts.append("%s %.4f(%+.4f,mv%d)" % (combo, f1, f1 - rec["base"], nm))
        if "gt" in rec:
            parts.append("| GT神谕 %.4f(%+.4f,mv%d)" % (rec["gt"][0], rec["gt"][0] - rec["base"],
                                                        rec["gt"][1]))
        print("  ".join(parts))

    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张，基线 %.4f）--" % (len(rows), mb))
        for combo in combos:
            if all(combo in r["blob"] for r in rows):
                mv = sum(r["blob"][combo][1] for r in rows)
                m = sum(r["blob"][combo][0] for r in rows) / n
                print("   墨团回收到轴线质心 (thr×%.1f, run≥%g, move≤%g)  %.4f  (%+.4f)"
                      "  ⇒ 加权 %+.2f 分  [移动 %d 端点]"
                      % (combo[0], combo[1], combo[2], m, m - mb, (m - mb) * 10.0, mv))
        if all("gt" in r for r in rows):
            m = sum(r["gt"][0] for r in rows) / n
            mv = sum(r["gt"][1] for r in rows)
            print("   GT 端点 R<=%g (神谕)                        %.4f  (%+.4f)"
                  "  ⇒ 加权 %+.2f 分  [移动 %d 端点]"
                  % (a.gt_r, m, m - mb, (m - mb) * 10.0, mv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
