# -*- coding: utf-8 -*-
"""_nlmoveax.py —— **自由端沿轴位移**归因：把基线里「配不上 GT 段的那个错端」在两个产物目录
之间逐端对齐，统计每个 stage2 算子把该端推了多少 px、推完之后是否落进 `line_tol = 5.0`
（`schematic_parser/metrics.py:88`），以及 `net_line_f1` 的差。

与 `_nlmove.py` 的差别：后者只报「边的两端最大位移」；这里**只报那些本来就错的端**，
从而回答「哪一步把它们搬到了现在的位置」。

用法：
  python -X utf8 tools/diag/_nlmoveax.py --base work/_fast_out_deterministic_default \
      --arms "trim0=work/_fast_out_deterministic_nltrim0,corner0=work/_fast_out_deterministic_nlcorner0"
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, load_json, DEFAULT_DATA_ROOT   # noqa: E402
from schematic_parser.metrics import EvalConfig, normalise_doc, _edge_points  # noqa: E402
from tools.diag._nlerr import _boxes_of, _pins_of, _end_class  # noqa: E402
from tools.diag._nlpair2 import _gt_nets, _ends, _d            # noqa: E402
from tools.diag import _wirecut                                # noqa: E402

TOL = 5.0
FREE_GAP = 10.0
NEAR = 20.0


def _edge_pts(v):
    if isinstance(v, list):
        return [(float(p["x"]), float(p["y"])) for p in v if isinstance(p, dict)]
    if isinstance(v, dict):
        return [(float(p["x"]), float(p["y"]))
                for p in (v.get("points") or []) if isinstance(p, dict)]
    return []


def _load(fp):
    """返回 (原始 doc, [(raw_net_key, [(x,y),...]), ...])，边序与 `_nlerr._edges_of` 一致。"""
    doc = load_json(fp)
    out = []
    for nk, net in (doc.get("nets") or {}).items():
        if not isinstance(net, dict):
            continue
        e = net.get("edges") or {}
        vals = e.values() if isinstance(e, dict) else e
        for v in vals:
            p = _edge_points(v)
            if p is not None and len(p) >= 2:
                out.append((nk, [(float(q[0]), float(q[1])) for q in p]))
    return doc, out


def _pair_by_mid(A, B):
    """同 net 内按中点近邻贪心配对，返回 {i: (j, swap)}。A=[(nk,pts)], B=[(nk,pts)]。"""
    cand = []
    for i, (nk, pa) in enumerate(A):
        ma = ((pa[0][0] + pa[-1][0]) / 2.0, (pa[0][1] + pa[-1][1]) / 2.0)
        for j, (mk, pb) in enumerate(B):
            if mk != nk:
                continue
            mb = ((pb[0][0] + pb[-1][0]) / 2.0, (pb[0][1] + pb[-1][1]) / 2.0)
            cand.append((_d(ma, mb), i, j))
    cand.sort()
    ua, ub, out = set(), set(), {}
    for _c, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        pa, pb = A[i][1], B[j][1]
        d0 = _d(pa[0], pb[0]) + _d(pa[-1], pb[-1])
        d1 = _d(pa[0], pb[-1]) + _d(pa[-1], pb[0])
        out[i] = (j, 1 if d1 < d0 else 0)
    return out


def _free_flags(pts_list):
    allp = [(i, p) for i, ps in enumerate(pts_list) for p in (ps[0], ps[-1])]
    flags = []
    for i, ps in enumerate(pts_list):
        flags.append([not any(_d(pt, r) <= FREE_GAP and k != i for k, r in allp)
                      for pt in (ps[0], ps[-1])])
    return flags


def bad_ends(case, pred_dir, cid, gt_pins, gt_boxes, cfg):
    """返回基线产物里「该边配不上 GT、且该端是自由端」的错端清单。

    匹配口径**完全复用** `metrics.py:588-628`：`assign_components` → `rewrite_keys`
    → `net_hypergraph_f1` 得 net 配对 → 同 net 内 `_seg_distance`（`:266`，取两端
    距离较大者，> `line_tol` 记 INF）+ `_assign`（`:278`）匈牙利。
    返回 (gt_doc, B, ends, line)。
    """
    from schematic_parser.metrics import _seg_distance, _assign

    gt = load_json(case.json_path)
    pred_raw = load_json(os.path.join(pred_dir, "final_results", "%s_Pred.json" % cid))
    line, pred, pairs = _wirecut.score(gt, pred_raw, cfg)
    g_nets = list(normalise_doc(gt)["nets"].items())
    p_nets = list(pred["nets"].items())
    # B 直接取自**归一化** pred，保证下标与 `pairs` / `used_pred` / `paired_pred` 同空间
    B = []
    net_span = {}                                    # p_nets 下标 -> B 里的边下标列表
    for ni, (nk, net) in enumerate(p_nets):
        e = net.get("edges") or {}
        vals = e.values() if isinstance(e, dict) else e
        idxs = []
        for v in vals:
            p = _edge_points(v)
            if p is not None and len(p) >= 2:
                idxs.append(len(B))
                B.append((nk, [(float(q[0]), float(q[1])) for q in p]))
        net_span[ni] = idxs

    gnets = dict(_gt_nets(gt))                       # 原始 key -> [pts,...]（已截前两点）
    praw_edges = _load(os.path.join(pred_dir, "final_results", "%s_Pred.json" % cid))[1]
    praw_nets = []
    for _nk, _p in praw_edges:
        if not praw_nets or praw_nets[-1][0] != _nk:
            praw_nets.append((_nk, []))
        praw_nets[-1][1].append(_p)
    praw_by_net = dict(praw_nets)
    unmatched_gt = []                                # [(edge_pts, own_list, key)]
    used_pred = set()
    matched_gt = {j for _i, j in pairs}
    for i, j in pairs:
        p_edges = [_edge_points(v) for v in (p_nets[i][1].get("edges") or {}).values()]
        g_edges = [_edge_points(v) for v in (g_nets[j][1].get("edges") or {}).values()]
        p_edges = [e for e in p_edges if e is not None]
        g_edges = [e for e in g_edges if e is not None]
        gkey = g_nets[j][0]
        own = gnets.get(gkey, [])
        pkey = praw_nets[i][0] if i < len(praw_nets) else p_nets[i][0]
        if not own:
            own = [e[:2] for e in g_edges if len(e) >= 2]
        got = set()
        if p_edges and g_edges:
            INF = float("inf")
            cost = []
            for pe in p_edges:
                row = []
                for ge in g_edges:
                    if len(pe) < 2 or len(ge) < 2:
                        row.append(INF)
                        continue
                    d = _seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                    row.append(d if d <= cfg.line_tol else INF)
                cost.append(row)
            got = set(_assign(cost))
        for b in range(len(g_edges)):
            if not any(bb == b for _a, bb in got):
                unmatched_gt.append((g_edges[b], own, pkey))
        for a, _b in got:
            if a < len(net_span[i]):
                used_pred.add(net_span[i][a])
    # 未匹配到任何 net 的 GT 网里的段也算（这些段没有对应我方网，用 GT 自己的 key）
    for j, (gkey, net) in enumerate(g_nets):
        if j in matched_gt:
            continue
        own = gnets.get(gkey, [])
        for v in (net.get("edges") or {}).values():
            e = _edge_points(v)
            if e is not None and len(e) >= 2:
                unmatched_gt.append((e, own or [e[:2]], gkey))

    bfree = _free_flags([p for _nk, p in B])
    paired_pred = {i for i, _j in pairs}     # 只治「网络已配上、段没配上」的网
    _paired_e = {k for ni in paired_pred for k in net_span.get(ni, [])}
    _keep = [i for i, (_nk, _ps) in enumerate(B) if i in _paired_e]
    # `ends["bi"]` 用 B 的**全局**下标（与 `_load` 同序），臂侧按 `_keep` 对齐
    ends = []
    for i, (nk, ps) in enumerate(B):
        if i in used_pred or i not in _paired_e:
            continue
        for eidx in (0, 1):
            if not bfree[i][eidx]:
                continue
            q = ps[0] if eidx == 0 else ps[-1]
            best = None
            for gpts, own, gkey in unmatched_gt:
                if gkey != nk:
                    continue
                ga, gb = _ends(gpts)
                for fl in (0, 1):
                    q0, q1 = (ga, gb) if fl == 0 else (gb, ga)
                    c = max(_d(q0, ps[0]), _d(q1, ps[-1]))
                    if best is None or c < best[0]:
                        best = (c, q0, q1, own)
            if best is None or best[0] > NEAR:
                continue
            _c, q0, q1, own = best
            if _d(ps[0], q0) >= _d(ps[-1], q1):
                GB, GO = q0, q1
            else:
                GB, GO = q1, q0
            if _d(GB, q) <= TOL:
                continue
            gi = 0
            bestg = None
            for k, e in enumerate(own):
                if len(e) < 2:
                    continue
                c = _d(GB, e[0]) + _d(GO, e[1])
                if bestg is None or c < bestg[0]:
                    bestg = (c, k)
            if bestg is not None:
                gi = bestg[1]
            _cls = _end_class(GB, gt_pins, own, gi, gt_boxes, 3.0)
            _other_pt = ps[-1] if eidx == 0 else ps[0]
            ends.append(dict(bi=i, eidx=eidx, q=q, GB=GB, other=_other_pt,
                             d0=_d(GB, q), d_other=_d(_other_pt, GO), cls=_cls))
    return gt, B, ends, line, [(B[i][0], i) for i in _keep], len(unmatched_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", required=True, help='"名字=目录,名字=目录"')
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--data", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--only-single", type=int, default=1,
                    help="1=只统计「另一端已就位」的端（默认），0=全部自由错端")
    a = ap.parse_args()
    arms = []
    for part in a.arms.split(","):
        nm, _, dr = part.partition("=")
        arms.append((nm.strip(), dr.strip()))
    cases = [x.strip() for x in a.cases.split(",") if x.strip()]
    index = {}
    for c in discover(a.data):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()

    cache = {}
    for cid in cases:
        case = index.get(cid)
        if case is None:
            continue
        try:
            cache[cid] = (case,) + bad_ends(case, a.base, cid,
                                            _pins_of(load_json(case.json_path)),
                                            _boxes_of(load_json(case.json_path)), cfg)
        except Exception as exc:      # noqa: BLE001
            import traceback
            print("case %s 跳过：%r" % (cid, exc))
            traceback.print_exc()
    n_bad = sum(len(v[2]) for v in cache.values())
    print("基线自由错端合计 %d 个（%d 张图）" % (n_bad, len(cache)))
    _b0 = [v[4].f1 for v in cache.values()]
    if _b0:
        print("基线 line_f1（本工具重算）均值 %.4f  [逐案 %s]"
              % (sum(_b0) / len(_b0), " ".join("%.4f" % x for x in _b0)))

    for nm, dr in arms:
        f1b, f1a = [], []
        agg = {}
        for cid, (_case, _gt, B, ends, _ln, bmap, _nfn) in cache.items():
            fb = os.path.join(a.base, "final_results", "%s_Pred.json" % cid)
            fa = os.path.join(dr, "final_results", "%s_Pred.json" % cid)
            if not os.path.exists(fa):
                continue
            db, A = _load(fa)
            f1b.append(float(_wirecut.score(_gt, _load(fb)[0], cfg)[0].f1))
            f1a.append(float(_wirecut.score(_gt, db, cfg)[0].f1))
            BMAP = {nk for nk, _i in bmap}
            _bk = [i for _nk, i in bmap]
            Bc = [(nk, B[i][1]) for nk, i in bmap if i < len(B)]
            _bk = _bk[:len(Bc)]
            _Aidx = [k for k, (nk, _p) in enumerate(A) if nk in BMAP]
            Ac = [A[k] for k in _Aidx]
            mp = _pair_by_mid(Bc, Ac)
            # mp 用 Bc/Ac 的**位置**做键，换回基线下标与 A 的下标
            m = {_bk[pos]: (_Aidx[j], sw) for pos, (j, sw) in mp.items()}
            for e in ends:
                if e["bi"] not in m:
                    continue
                if a.only_single:
                    # 只保留「另一端已就位」的端（即 `_nlpair2` 的 one_end_ok 类）
                    if e["d_other"] > TOL:
                        continue
                j, sw = m[e["bi"]]
                pa, pb = A[j][1][0], A[j][1][-1]
                if sw:
                    pa, pb = pb, pa
                nq = pa if e["eidx"] == 0 else pb
                d_after = _d(e["GB"], nq)
                shift = _d(nq, e["q"])
                # 沿该端所在轴的带符号位移（朝段内为正）
                o = e["other"]
                L = _d(e["q"], o) or 1.0
                ux, uy = (o[0] - e["q"][0]) / L, (o[1] - e["q"][1]) / L
                s = (nq[0] - e["q"][0]) * ux + (nq[1] - e["q"][1]) * uy
                g = agg.setdefault(e["cls"], dict(n=0, fixed=0, worse=0, same=0, sh=[], s=[]))
                g["n"] += 1
                if d_after <= TOL:
                    g["fixed"] += 1
                elif d_after > e["d0"] + 0.5:
                    g["worse"] += 1
                else:
                    g["same"] += 1
                g["sh"].append(shift)
                g["s"].append(s)
        if not f1a:
            continue
        print("\n== 臂 %s ==  line_f1 %.4f → %.4f (%+.4f)" % (
            nm, sum(f1b) / len(f1b), sum(f1a) / len(f1a),
            sum(f1a) / len(f1a) - sum(f1b) / len(f1b)))
        print("   %-6s %5s %6s %6s %6s %9s %10s %10s"
              % ("cls", "n", "修好", "更差", "未动", "位移中位", "位移均值", "带符号中位"))
        for k in sorted(agg, key=lambda z: -agg[z]["n"]):
            g = agg[k]
            sh = sorted(g["sh"])
            ss = sorted(g["s"])
            print("   %-6s %5d %6d %6d %6d %9.1f %10.1f %10.2f"
                  % (k, g["n"], g["fixed"], g["worse"], g["same"],
                     sh[len(sh) // 2], sum(sh) / len(sh), ss[len(ss) // 2]))
        ta = dict(n=0, fixed=0, worse=0, same=0)
        for g in agg.values():
            for q in ta:
                ta[q] += g[q]
        print("   %-6s %5d %6d %6d %6d" % ("合计", ta["n"], ta["fixed"], ta["worse"], ta["same"]))


if __name__ == "__main__":
    main()
