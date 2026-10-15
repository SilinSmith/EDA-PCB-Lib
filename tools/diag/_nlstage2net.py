# -*- coding: utf-8 -*-
"""§5.126：量 stage2 端点链对「配不上的坏端」的**净位移**。

做法：坏端在 `final` 快照里的坐标 = 出口坐标（已验证逐位一致）；
在 `pre` 快照里（= stage1/pskel 直出）找同一条边的同名端，
两端之差就是整条 stage2 链的净位移（正 = 向段外推，负 = 向段内收）。

CLI: python -X utf8 tools/diag/_nlstage2net.py --tracedir ..\\work\\_trace_all
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser import metrics as M                       # noqa: E402
from schematic_parser.data import discover, load_json, normalise_doc  # noqa: E402

TOL = 5.0


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tag(path):
    p = os.path.basename(path).split(".")
    return p[-2] if len(p) >= 4 else ""


def _load(tracedir, cid):
    out = {}
    for fp in sorted(glob.glob(os.path.join(tracedir, "%s.*.json" % cid))):
        t = _tag(fp)
        if not t:
            continue
        try:
            out[t] = json.load(io.open(fp, encoding="utf-8")).get("edges") or []
        except Exception:
            pass
    return out


def _ends(rows):
    return [(r[0], (float(r[2][0]), float(r[2][1])), (float(r[3][0]), float(r[3][1])))
            for r in rows]


def _pre_match(q, nk, cand, maxd):
    """在 `pre` 里找同 net 的最近端。返回 (dist, 端点, 另一端)。"""
    best = None
    for n2, p0, p1 in cand:
        if n2 != nk:
            continue
        for ei, p in ((0, p0), (1, p1)):
            d = _d(q, p)
            if d <= maxd and (best is None or d < best[0]):
                best = (d, p, p1 if ei == 0 else p0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracedir", required=True)
    ap.add_argument("--run-dir",
                    default=r"Y:\workspace\EDA\work\_fast_out_deterministic_default")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()
    idx = {}
    for c in discover(__import__("schematic_parser.data",
                                 fromlist=["x"]).DEFAULT_DATA_ROOT):
        idx.setdefault(c.case_id, c)
    cfg = M.EvalConfig()
    cfg.line_tol = TOL
    n_all = n_tr = 0
    h = collections.Counter()
    signed = []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = idx.get(cid)
        if case is None:
            continue
        st = _load(a.tracedir, cid)
        if "pre" not in st or "final" not in st:
            print("case %s 缺快照" % cid)
            continue
        pre, fin = _ends(st["pre"]), _ends(st["final"])
        gt = normalise_doc(load_json(case.json_path))
        pred = normalise_doc(load_json(
            os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid)))
        _c, km = M.assign_components(gt, pred, cfg, None)
        pred2 = M.rewrite_keys(pred, km)
        _n, pairs = M.net_hypergraph_f1(gt, pred2, cfg, None)
        g_nets, p_nets = list(gt["nets"].items()), list(pred2["nets"].items())
        bad = 0
        for i, j in pairs:
            pnk, pv = p_nets[i]
            _gk, gv = g_nets[j]
            pe = [M._edge_points(v) for v in (pv.get("edges") or {}).values()]
            ge = [M._edge_points(v) for v in (gv.get("edges") or {}).values()]
            pe = [e for e in pe if e]
            ge = [e for e in ge if e]
            if not pe or not ge:
                continue
            cost = []
            for e in pe:
                row = []
                for g in ge:
                    d = M._seg_distance(e[:2], g[:2], cfg.line_unordered)
                    row.append(d if d <= cfg.line_tol else float("inf"))
                cost.append(row)
            cand = []
            for ii, row in enumerate(cost):
                for jj, cc in enumerate(row):
                    if cc <= cfg.line_tol:
                        cand.append((cc, ii, jj))
            cand.sort(key=lambda z: z[0])
            ui, uj = set(), set()
            for _cc, ii, jj in cand:
                if ii in ui or jj in uj:
                    continue
                ui.add(ii)
                uj.add(jj)
            for pi, e in enumerate(pe):
                if pi in ui:
                    continue
                for ei in (0, 1):
                    q = (float(e[0 if ei == 0 else -1][0]),
                         float(e[0 if ei == 0 else -1][1]))
                    bg = min((_d(q, g[0 if ei == 0 else 1]) for g in ge
                              if len(g) >= 2), default=None)
                    if bg is None or bg <= TOL:
                        continue
                    bad += 1
                    m = _pre_match(q, pnk, pre, 40.0)
                    if m is None:
                        continue
                    n_tr += 1
                    mv = _d(q, m[1])
                    h[int(mv // 2) * 2] += 1
                    # 方向：投影到 final 段自身的轴上
                    o = e[0] if ei == 1 else e[-1]
                    vx, vy = float(o[0]) - q[0], float(o[1]) - q[1]
                    L = math.hypot(vx, vy) or 1.0
                    proj = ((q[0] - m[1][0]) * vx + (q[1] - m[1][1]) * vy) / L
                    signed.append(proj)
        n_all += bad
        print("case %2s: 坏端 %3d" % (cid, bad))
    print("\n合计坏端 %d，能在 pre 里追到 %d" % (n_all, n_tr))
    if signed:
        s = sorted(signed)
        pos = sum(1 for x in s if x > 0.5)
        neg = sum(1 for x in s if x < -0.5)
        zero = len(s) - pos - neg
        print("整条 stage2 链对坏端的**净位移**（带符号，正=外推）：")
        print("  中位 %+.2f  均值 %+.2f  外推 %d  回缩 %d  几乎没动 %d"
              % (s[len(s) // 2], sum(s) / len(s), pos, neg, zero))
        print("  |净位移| 分箱：%s"
              % " ".join("%d:%d" % (k, h[k]) for k in sorted(h)))
        for thr in (2, 4, 6, 8, 12):
            print("   |净位移| ≤%2d : %3d （占追到的 %.0f%%）"
                  % (thr, sum(1 for x in s if abs(x) <= thr),
                     100.0 * sum(1 for x in s if abs(x) <= thr) / len(s)))


if __name__ == "__main__":
    main()
