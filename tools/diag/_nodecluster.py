# -*- coding: utf-8 -*-
"""离线 A/B：把「同一结点上的我方端点」聚类取质心（可交付，不用 GT）

`_linedump.py` 摊开后真相很清楚（case 1）：
  多余段  len 105.0  d0 2.0  d1 6.0   最近 GT 段 len 101.0  span=True
  缺失段  len 101.0  d0 2.0  d1 6.0   最近我方段 len 105.0  inside_pred=True
⇒ 形状全对，**我方段比 GT 段长 4px、端点偏 6px**；而 `metrics._seg_distance` 取两端最大距离、
容差只有 5px ⇒ 6px 就整条判 fp+fn。案 2 更极端（我方段长 20px 以上，端点偏 39~42px）。

`_dotsnap.py` 已否证「吸结点圆点」（case 1 完全 0 变化，均值最好 +0.23）。本探针试另一条
**可交付**路子：真结点上本来有多条我方段汇聚，它们的端点应当重合；把互相 ≤C 的端点做并查集
聚类、各自移到簇质心 ⇒ 短/长的骨架分叉点被拉回共同交点。纯我方几何，不需要 GT、不需要圆点。

  --mode cluster  端点聚类取质心（可交付）
  --mode gt5      端点吸到 5px 内最近 GT 端点（参照，理论上 ≈0）
  --mode both

用法：
    python -X utf8 tools\\diag\\_nodecluster.py --cs 6,8,10,12,16,20
"""
from __future__ import print_function

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover  # noqa: E402
from schematic_parser.metrics import EvalConfig, net_line_f1, normalise_doc  # noqa: E402
from _dotsnap import score, apply_snap, endpoints_img                      # noqa: E402
from _dotprobe import flip                                                # noqa: E402
from pipeline import pcore, prun                                          # noqa: E402


class UF(object):
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def clusters(pts, C):
    """返回 size>=2 的簇：[(质心, [idx...])]"""
    n = len(pts)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(pts[i][0] - pts[j][0]) > C or abs(pts[i][1] - pts[j][1]) > C:
                continue
            if float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])) <= C:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    out = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        cx = float(np.mean([pts[i][0] for i in idx]))
        cy = float(np.mean([pts[i][1] for i in idx]))
        out.append(((cx, cy), idx))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3")
    ap.add_argument("--mode", default="both", choices=["cluster", "gt5", "both"])
    ap.add_argument("--cs", default="6,8,10,12,16,20")
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    cs = [float(x) for x in a.cs.split(",") if x.strip()]
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
        eps = endpoints_img(pred2, h_ := prun.inputs_of(pcore.find_case(cid)).h)
        pts = [e[0] for e in eps]
        handles = [e[1] for e in eps]
        rec = {"case": cid, "base": base.f1, "n_ep": len(eps), "cl": {}, "gt5": None}
        if a.mode in ("cluster", "both"):
            for C in cs:
                cl = clusters(pts, C)
                pd = copy.deepcopy(pred2)
                moves = {}
                n_in = 0
                for cen, idx in cl:
                    for i in idx:
                        moves[handles[i]] = cen
                        n_in += 1
                nm = apply_snap(pd, moves, h_)
                sc = net_line_f1(gt_norm, pd, cfg, pairs, {})
                rec["cl"][C] = (sc.f1, len(cl), nm)
        if a.mode in ("gt5", "both"):
            gt_eps = []
            for _n, net in (gt_norm.get("nets") or {}).items():
                for _e, edge in (net.get("edges") or {}).items():
                    from schematic_parser.metrics import _edge_points
                    pl = _edge_points(edge) or []
                    if len(pl) >= 2:
                        gt_eps.append(flip([pl[0], pl[-1]], h_)[0])
                        gt_eps.append(flip([pl[0], pl[-1]], h_)[1])
            arr = np.asarray(gt_eps, dtype=float)
            pd = copy.deepcopy(pred2)
            moves = {}
            for p_img, handle in eps:
                dd = np.hypot(arr[:, 0] - p_img[0], arr[:, 1] - p_img[1])
                k = int(np.argmin(dd))
                if dd[k] <= 5.0:
                    moves[handle] = (float(arr[k, 0]), float(arr[k, 1]))
            nm = apply_snap(pd, moves, h_)
            sc = net_line_f1(gt_norm, pd, cfg, pairs, {})
            rec["gt5"] = (sc.f1, nm)
        rows.append(rec)
        line = "case %-3s 基线 %.4f 端点 %3d" % (cid, rec["base"], rec["n_ep"])
        for C in cs:
            if C in rec["cl"]:
                f1, ncl, nm = rec["cl"][C]
                line += "  C%-3g %.4f(%+.4f,%d簇/%d端)" % (C, f1, f1 - rec["base"], ncl, nm)
        if rec["gt5"]:
            line += "  | 吸GT5px %.4f(%+.4f)" % (rec["gt5"][0], rec["gt5"][0] - rec["base"])
        print(line)
    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张）--" % len(rows))
        print("   基线                      %.4f" % mb)
        for C in cs:
            if all(C in r["cl"] for r in rows):
                m = sum(r["cl"][C][0] for r in rows) / n
                npull = sum(r["cl"][C][2] for r in rows)
                print("   端点聚类 C<=%-4g (可交付)  %.4f  (%+.4f)  ⇒ 加权 %+.2f 分  [拉动 %d 端点]"
                      % (C, m, m - mb, (m - mb) * 10.0, npull))
    return 0


if __name__ == "__main__":
    sys.exit(main())
