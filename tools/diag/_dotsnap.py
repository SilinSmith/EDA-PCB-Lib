# -*- coding: utf-8 -*-
"""离线 A/B：把交付端点吸到「结点圆点质心」能不能拿回端点偏差的亏损？

`_lineaudit.py` 量到：官方 10 张 net_line_f1 = 0.5796，未配上 264 段 / 缺 253 段，
但把未配上段的端点**瞬移到 20px 内最近的 GT 端点**（神谕）就能到 0.9224（+3.43 加权分）；
5px 内瞬移只 +0.0036 ⇒ 亏损 = 端点偏 5~20px。

`_dotprobe.py` 量到：GT 的 T 接端点 **79.8%** 命中「对角 4 探针」圆点判据（对照 5.6%），
**71.8%** 落在检出圆点质心 3px 内（p50 1.8px）；而脚点端点只有 8.6% / 0.0% ⇒
**圆点只标导线—导线结点，是图像侧可判定的结点锚点**（不是 GT 神谕）。

本探针不动流水线，只在**已交付文档**上做后处理再评分（口径与判分器一致）：
  --mode dot   把每条导线端点吸到 R 像素内最近的**检出圆点质心**（可交付）
  --mode gt    把端点吸到 20px 内最近的 **GT 端点**（神谕上界，用来校验离线打分链）
  --mode both  两种都算

只改 `nets[*].edges` 的端点坐标，网配对 `pairs` 沿用基线 ⇒ 与 `_wirecut.py` 同一套离线口径。

用法：
    python -X utf8 tools\\diag\\_dotsnap.py
"""
from __future__ import print_function

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import pcore, prun                                         # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover  # noqa: E402
from schematic_parser.metrics import (EvalConfig, _edge_points, assign_components,
                                      net_hypergraph_f1, net_line_f1,
                                      normalise_doc, rewrite_keys)        # noqa: E402
from _dotprobe import dots_of, flip                                     # noqa: E402


def score(gt_doc, pred_doc, cfg):
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    detail = {}
    _, keymap = assign_components(gt, pred, cfg, detail)
    pred2 = rewrite_keys(pred, keymap)
    _net, pairs = net_hypergraph_f1(gt, pred2, cfg, detail)
    line = net_line_f1(gt, pred2, cfg, pairs, detail)
    return line, pred2, pairs


def endpoints_img(pred2, h):
    """交付文档里所有导线端点（图像坐标）+ 回写句柄 (net, ekey, which)。"""
    out = []
    for nkey, net in (pred2.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge) or []
            if len(pts) < 2:
                continue
            for which in (0, -1):
                p = pts[which]
                out.append(((float(p[0]), h - float(p[1])), (nkey, ekey, which)))
    return out


def apply_snap(doc, moves, h):
    """moves: { (nkey, ekey, which): (x_img, y_img) } ⇒ 写回 doc 坐标（y 向上）。"""
    n = 0
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge) or []
            if len(pts) < 2:
                continue
            new = [[float(q[0]), float(q[1])] for q in pts]
            for which in (0, -1):
                tgt = moves.get((nkey, ekey, which))
                if tgt is None:
                    continue
                new[which] = [float(tgt[0]), h - float(tgt[1])]
                n += 1
            if n:
                net["edges"][ekey] = new
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--mode", default="both", choices=["dot", "gt", "both"])
    ap.add_argument("--rs", default="4,6,8,10,12,15,20")
    ap.add_argument("--gt-r", type=float, default=20.0)
    ap.add_argument("--min-dt", type=float, default=3.0)
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    rs = [float(x) for x in a.rs.split(",") if x.strip()]
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
        _dt, dots = dots_of(np.asarray(res.ink).astype(bool), a.min_dt)
        gt_doc = load_json(case.json_path)
        pred_doc = load_json(fp)
        base, pred2, pairs = score(gt_doc, pred_doc, cfg)
        eps = endpoints_img(pred2, h)
        gt_eps = []
        for _n, net in (normalise_doc(gt_doc).get("nets") or {}).items():
            for _e, edge in (net.get("edges") or {}).items():
                pl = _edge_points(edge) or []
                if len(pl) >= 2:
                    gt_eps.append(flip([pl[0], pl[-1]], h)[0])
                    gt_eps.append(flip([pl[0], pl[-1]], h)[1])
        gt_arr = np.asarray(gt_eps, dtype=float) if gt_eps else np.zeros((0, 2))
        rec = {"case": cid, "base": base.f1, "n_ep": len(eps), "n_dot": len(dots),
               "dot": {}, "gt": None, "gt_moved": 0}
        if a.mode in ("dot", "both") and len(dots):
            arr = np.asarray([[c[0], c[1]] for c in dots], dtype=float)
            for R in rs:
                pd = copy.deepcopy(pred2)
                moves = {}
                for p_img, handle in eps:
                    dd = np.hypot(arr[:, 0] - p_img[0], arr[:, 1] - p_img[1])
                    k = int(np.argmin(dd))
                    if dd[k] <= R:
                        moves[handle] = (float(arr[k, 0]), float(arr[k, 1]))
                n_moved = apply_snap(pd, moves, h)
                sc = net_line_f1(normalise_doc(gt_doc), pd, cfg, pairs, {})
                rec["dot"][R] = (sc.f1, n_moved)
        if a.mode in ("gt", "both") and len(gt_arr):
            pd = copy.deepcopy(pred2)
            moves = {}
            for p_img, handle in eps:
                dd = np.hypot(gt_arr[:, 0] - p_img[0], gt_arr[:, 1] - p_img[1])
                k = int(np.argmin(dd))
                if dd[k] <= a.gt_r:
                    moves[handle] = (float(gt_arr[k, 0]), float(gt_arr[k, 1]))
            n_moved = apply_snap(pd, moves, h)
            sc = net_line_f1(normalise_doc(gt_doc), pd, cfg, pairs, {})
            rec["gt"] = sc.f1
            rec["gt_moved"] = n_moved
        rows.append(rec)
        line = "case %-3s 基线 %.4f  端点 %3d 圆点 %3d" % (cid, rec["base"], rec["n_ep"], rec["n_dot"])
        for R in rs:
            if R in rec["dot"]:
                f1, nm = rec["dot"][R]
                line += "  R%-3g %.4f(%+.4f,mv%d)" % (R, f1, f1 - rec["base"], nm)
        if rec["gt"] is not None:
            line += "  | GT神谕 %.4f(%+.4f,mv%d)" % (rec["gt"], rec["gt"] - rec["base"],
                                                     rec["gt_moved"])
        print(line)

    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张）--" % len(rows))
        print("   基线                        %.4f" % mb)
        for R in rs:
            if all(R in r["dot"] for r in rows):
                mv = sum(r["dot"][R][1] for r in rows)
                m = sum(r["dot"][R][0] for r in rows) / n
                print("   吸圆点 R<=%-4g (可交付)     %.4f  (%+.4f)  ⇒ 加权 %+.2f 分  [移动 %d 端点]"
                      % (R, m, m - mb, (m - mb) * 10.0, mv))
        if all(r["gt"] is not None for r in rows):
            m = sum(r["gt"] for r in rows) / n
            mv = sum(r["gt_moved"] for r in rows)
            print("   吸 GT 端点 R<=%-4g (神谕)   %.4f  (%+.4f)  ⇒ 加权 %+.2f 分  [移动 %d 端点]"
                  % (a.gt_r, m, m - mb, (m - mb) * 10.0, mv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
