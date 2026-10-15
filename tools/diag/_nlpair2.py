# -*- coding: utf-8 -*-
"""_nlpair2.py —— 未配上 GT 段的**严格**端点级归因（按 metrics.net_line_f1 的真实口径）。

与 `_nlendchar.py` 的差别（后者用「最近单端点」找伙伴，会串到同一条 GT 段在**同一 net 的
另一份拷贝**上，把「两端都错」误报成「单端错」）：
  * tp 集合按 `metrics.py:588-628` 的真实代价重建：一对 (gt,pred) 代价 = max(两端距离)，
    > line_tol(5) 记 INF；贪心取最小非 INF 得 tp，剩下的才是 fn。
  * 对每条 fn，再找 **max(两端距离) 最小**的 pred 段（不管是否已被占用）当「最像的伙伴」，
    因为 `_seg_distance`（`:266`）要两端同时进 5px ⇒ 必须两头一起看。
  * 每条 fn 归三类：one_end（一端 ≤5 另一端 >5）/ both_end / no_near（20px 内无 pred 段）。
  * 每端给出：轴向差 axial（沿 pred 轴、从 good 端起算：0..L 在段内、<0 越过 good、>L 越过 other）、
    横向差 perp、GT 端结构类别（pin/join/box/blank，判据复用 `_nlerr._end_class`）。

用法：
  python -X utf8 tools/diag/_nlpair2.py --dir work/_fast_out_delivery_default --cases 2,3,9 \
      --dump work/_nlpair2.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, load_json, DEFAULT_DATA_ROOT   # noqa: E402
from schematic_parser.metrics import EvalConfig                # noqa: E402
from tools.diag._nlerr import (_edges_of, _boxes_of, _pins_of, _end_class)  # noqa: E402

TOL = 5.0
RAD = 20.0


def _ends(seg):
    return (tuple(float(x) for x in seg[0]), tuple(float(x) for x in seg[1]))


def _d(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _ax_perp(bad, good, other):
    """bad 端相对 pred 段 good->other 的（轴向坐标 axial, 横向 perp）。"""
    vx, vy = other[0] - good[0], other[1] - good[1]
    L = math.hypot(vx, vy)
    if L < 1e-9:
        return 0.0, _d(bad, good)
    wx, wy = bad[0] - good[0], bad[1] - good[1]
    return (wx * vx + wy * vy) / L, abs(wx * vy - wy * vx) / L


def _gt_nets(doc):
    """返回 [(net_id, [pts,...])]，pts 已过滤成 len>=2 的前两点。"""
    out = []
    for nk, net in (doc.get("nets") or {}).items():
        es = _edges_of(net) if isinstance(net, dict) else []
        if es:
            out.append((nk, es))
    return out


def analyse(cases, run_dir, data_root):
    index = {}
    for c in discover(data_root):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()
    rows = []
    for cid in cases:
        case = index.get(cid)
        fp = os.path.join(run_dir, "final_results", "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物（%s）" % (cid, fp))
            continue
        gt = load_json(case.json_path)
        pred = load_json(fp)
        gnets = _gt_nets(gt)
        pnets = _gt_nets(pred)
        ge = [(nk, i, e) for nk, es in gnets for i, e in enumerate(es)]
        pe = [(nk, i, e) for nk, es in pnets for i, e in enumerate(es)]
        gt_boxes, gt_pins = _boxes_of(gt), _pins_of(gt)
        # --- tp：最小非 INF 代价贪心配对
        cand = []
        for gi, (gnk, gidx, g) in enumerate(ge):
            ga, gb = _ends(g)
            for pi, (pnk, pidx, p) in enumerate(pe):
                pa, pb = _ends(p)
                c1 = max(_d(ga, pa), _d(gb, pb))
                c2 = max(_d(ga, pb), _d(gb, pa))
                c = min(c1, c2)
                if c <= TOL:
                    cand.append((c, gi, pi, 0 if c1 <= c2 else 1))
        cand.sort(key=lambda z: z[0])
        used_g, used_p = set(), set()
        for c, gi, pi, fl in cand:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
        fn = [i for i in range(len(ge)) if i not in used_g]
        pc = dict(case=cid, n_gt=len(ge), n_pred=len(pe), n_fn=len(fn),
                  n_tp=len(used_g), one_end=0, both_end=0, no_near=0, samples=[])
        per_net = {nk: es for nk, es in gnets}
        for gi in fn:
            gnk, gidx, g = ge[gi]
            ga, gb = _ends(g)
            best = None
            for p in pe:
                pa, pb = _ends(p[2])
                for fl in (0, 1):
                    q0, q1 = (pa, pb) if fl == 0 else (pb, pa)
                    c = max(_d(ga, q0), _d(gb, q1))
                    if best is None or c < best[0]:
                        best = (c, p, q0, q1)
            if best is None or best[0] > RAD:
                pc["no_near"] += 1
                continue
            c, p, q0, q1 = best
            d0, d1 = _d(ga, q0), _d(gb, q1)
            nbad = (d0 > TOL) + (d1 > TOL)
            kind = "one_end" if nbad == 1 else "both_end"
            pc[kind] += 1
            Lp = _d(q0, q1)
            rec = dict(case=cid, kind=kind, cost=round(c, 2), gt_net=gnk, gt_idx=gidx,
                       pnet=p[0], gt=[list(ga), list(gb)], pred=[list(q0), list(q1)])
            for nm, GB, GP, OT in (("a", ga, q0, q1), ("b", gb, q1, q0)):
                ax, pp = _ax_perp(GB, GP, OT)
                rec[nm] = dict(d=round(_d(GB, GP), 2), axial=round(ax, 2), perp=round(pp, 2),
                               cls=_end_class(GB, gt_pins, per_net.get(gnk, []), gidx,
                                              gt_boxes, 3.0),
                               Lp=round(Lp, 1),
                               out=round(max(0.0, -ax, ax - Lp), 2),
                               inside=bool(-TOL <= ax <= Lp + TOL))
            pc["samples"].append(rec)
        rows.append(pc)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/_fast_out_delivery_default")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--dump", default="")
    ap.add_argument("--data", default=DEFAULT_DATA_ROOT)
    a = ap.parse_args()
    cases = [x.strip() for x in a.cases.split(",") if x.strip()]
    rows = analyse(cases, a.dir, a.data)
    tot = dict(one_end=0, both_end=0, no_near=0, n_fn=0, n_gt=0, n_tp=0)
    print("%-6s %5s %6s %5s %5s | %7s %8s %8s" % ("case", "nGT", "nPred", "nTP", "nFN",
                                                   "one_end", "both_end", "no_near"))
    for r in rows:
        for k in ("one_end", "both_end", "no_near", "n_fn", "n_gt", "n_tp"):
            tot[k] += r[k]
        print("%-6s %5d %6d %5d %5d | %7d %8d %8d" % (r["case"], r["n_gt"], r["n_pred"],
                                                      r["n_tp"], r["n_fn"], r["one_end"],
                                                      r["both_end"], r["no_near"]))
    print("%-6s %5d %6s %5d %5d | %7d %8d %8d" % ("ALL", tot["n_gt"], "", tot["n_tp"],
                                                  tot["n_fn"], tot["one_end"], tot["both_end"],
                                                  tot["no_near"]))
    agg = {}
    side = dict(inside=0, outside=0)
    for r in rows:
        for s in r["samples"]:
            if s["kind"] != "one_end":
                continue
            bad = "a" if s["a"]["d"] > TOL else "b"
            b = s[bad]
            side["inside" if b["inside"] else "outside"] += 1
            d = agg.setdefault(b["cls"], dict(n=0, short=0, long=0, perp2=0, ds=[], over2=0))
            d["n"] += 1
            if not b["inside"]:            # GT 端落在我方段**外** => 我方欠伸
                d["short"] += 1
            else:                          # 落在段内但离端点 >5px => 我方过伸
                d["long"] += 1
            if b["perp"] <= 2.0:
                d["perp2"] += 1
            if b["out"] <= 2.0:
                d["over2"] += 1
            d["ds"].append(round(b["d"], 1))
    print("\n单端错（one_end）按**错端**的 GT 结构类别：")
    print("  %-6s %5s %7s %7s %10s %8s %s" % ("cls", "n", "欠伸", "过伸", "perp<=2", "out<=2", "距离中位"))
    for k in sorted(agg, key=lambda z: -agg[z]["n"]):
        d = agg[k]
        ds = sorted(d["ds"])
        print("  %-6s %5d %7d %7d %10d %8d %s" % (k, d["n"], d["short"], d["long"],
                                                  d["perp2"], d["over2"],
                                                  ds[len(ds) // 2] if ds else 0))
    print("\none_end 的错端相对我方段：段外(欠伸) %d / 段内(过伸) %d"
          % (side["outside"], side["inside"]))
    if a.dump:
        with io.open(a.dump, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows), fh, ensure_ascii=False, indent=1)
        print("-> %s" % a.dump)


if __name__ == "__main__":
    main()
