# -*- coding: utf-8 -*-
"""导线段亏损分解（判分口径感知，只读）。

`net_line_f1`（metrics.py:588）的配对规则很短：**每条 edge 只比前两点**，
`_seg_distance` 取「两个端点距离的较大者」，unordered（正反都试），
tol = 5px（`EvalConfig.line_tol`），配对是 `_assign` 的全局最优一一匹配。

所以一条 edge 能不能配上，只看**它的两个端点是不是都落在 GT 那条段的两个端点 5px 内**。
本探针不再看「覆盖」，只看判分真正在意的东西，把亏损拆成可归因的五份：

  A 精度损失（fp）：我们多出来的段
      A1 与某条**已配上**的 GT 段重复（两端都落在同一条 GT 段内部 ⇒ 冗余重复段）
      A2 所在网没有 GT 对应（net 分组错，段本身可能没错）
      A3 其余多余段（短碎段/串线段）
  B 召回损失（fn）：GT 有、我们没配上
      B1 被我们一条**更长的段**盖住（GT 两端都在我们段的内部 ⇒ 需要在该点切段）
      B2 被我们一条**更短的段**盖住（我们两端都在 GT 段内部 ⇒ 需要延长）
      B3 完全没画到

再给出四个**神谕上界**（不可交付，只用来判断「该修哪一侧」）：

  drop_fp   把未配上的 pred 段全删掉（precision→1）
  add_fn    把未配上的 GT 段全加进去（recall→1）
  snap_5    配对上的段端点吸附到 GT 端点（端点级后处理的上界；<5px *)
  snap_20   同上，但允许把端点吸到 20px 内最近的 GT 端点
  snap_best 每条 pred 段两端各自吸到最近的 GT 端点，无半径限制（端点级绝对上界）

用法：
    python -X utf8 tools\\diag\\_lineaudit.py --run work\\_fast_out_sjndef --cases 1,2,3
"""
from __future__ import print_function

import argparse
import collections
import copy
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


def _maxdist(a, b):
    return max(math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]),
               math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]))


def _online(p, a, b, tol=2.0):
    """p 是否落在线段 ab 上（含 2px 容差），且**严格在内部**（不是端点）。"""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return False
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / L2
    if not (0.02 < t < 0.98):
        return False
    px, py = ax + vx * t, ay + vy * t
    return math.hypot(px - p[0], py - p[1]) <= tol


def _nearest_gt_endpoint(p, pts):
    best, bd = None, float("inf")
    for q in pts:
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if d < bd:
            best, bd = q, d
    return best, bd


def audit_case(gt_doc, pred_doc, cfg, detail):
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    _comp, keymap = assign_components(gt, pred, cfg, detail)
    pred2 = rewrite_keys(pred, keymap)
    _net, pairs = net_hypergraph_f1(gt, pred2, cfg, detail)
    base = net_line_f1(gt, pred2, cfg, pairs, detail).f1

    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    paired_pred = {i for i, _ in pairs}

    stat = collections.Counter()
    all_gt_pts = []
    for _k, net in g_nets:
        for v in (net.get("edges") or {}).values():
            e = _edge_points(v)
            if e:
                all_gt_pts.extend(e[:2])

    # ---- 逐网配对，收集 A/B 分类 ----------------------------------------
    drop = copy.deepcopy(pred2)      # 删掉未配上的 pred 段
    add = copy.deepcopy(pred2)       # 加上未配上的 GT 段
    snap5 = copy.deepcopy(pred2)
    snap20 = copy.deepcopy(pred2)
    snapbest = copy.deepcopy(pred2)

    for i, j in pairs:
        pnet, gnet = p_nets[i][1], g_nets[j][1]
        pkeys = list((pnet.get("edges") or {}).keys())
        gkeys = list((gnet.get("edges") or {}).keys())
        p_edges = [_edge_points(pnet["edges"][k]) for k in pkeys]
        g_edges = [_edge_points(gnet["edges"][k]) for k in gkeys]
        p_ok = [k for k, e in zip(pkeys, p_edges) if e]
        g_ok = [k for k, e in zip(gkeys, g_edges) if e]
        p_edges = [_edge_points(pnet["edges"][k]) for k in p_ok]
        g_edges = [_edge_points(gnet["edges"][k]) for k in g_ok]
        INF = float("inf")
        cost = [[(_seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                  if _seg_distance(pe[:2], ge[:2], cfg.line_unordered) <= cfg.line_tol
                  else INF)
                 for ge in g_edges] for pe in p_edges]
        got = _assign(cost)
        mi = {r for r, _ in got}
        mj = {c for _, c in got}

        for r, k in enumerate(p_ok):
            if r in mi:
                continue
            stat["A_fp"] += 1
            pe = p_edges[r]
            dup = False
            for c in mj:
                ge = g_edges[c]
                if _online(pe[0], ge[0], ge[1]) and _online(pe[1], ge[0], ge[1]):
                    dup = True
                    break
            if dup:
                stat["A1_dup_of_matched"] += 1
            else:
                stat["A3_other_extra"] += 1
            drop["nets"][p_nets[i][0]]["edges"].pop(k, None)

        for c, k in enumerate(g_ok):
            if c in mj:
                continue
            stat["B_fn"] += 1
            ge = g_edges[c]
            kind = None
            for pe in p_edges:
                if _online(ge[0], pe[0], pe[1]) and _online(ge[1], pe[0], pe[1]):
                    kind = "B1_spanned"
                    break
                if _online(pe[0], ge[0], ge[1]) and _online(pe[1], ge[0], ge[1]):
                    kind = "B2_underspan"
                    break
            if kind is None:
                kind = "B3_missing"
                stat[kind] += 1
            else:
                stat[kind] += 1
            add["nets"][p_nets[i][0]]["edges"]["_audit_%d" % c] = [
                {"x": ge[0][0], "y": ge[0][1]}, {"x": ge[1][0], "y": ge[1][1]}]

        # 吸附类神谕
        for r, k in enumerate(p_ok):
            pe = p_edges[r]
            tgt = [k]
            for nm, rad in (("snap5", 5.0), ("snap20", 20.0)):
                for idx in (0, 1):
                    if r in mi:
                        c = [cc for rr, cc in got if rr == r][0]
                        q = g_edges[c][idx]
                    else:
                        q, d = _nearest_gt_endpoint(pe[idx], all_gt_pts)
                        if d > rad:
                            continue
                    doc = {"snap5": snap5, "snap20": snap20}[nm]
                    doc["nets"][p_nets[i][0]]["edges"][k][idx] = {
                        "x": q[0], "y": q[1]}
            for idx in (0, 1):
                q, _d = _nearest_gt_endpoint(pe[idx], all_gt_pts)
                if q is not None:
                    snapbest["nets"][p_nets[i][0]]["edges"][k][idx] = {
                        "x": q[0], "y": q[1]}

    # 没有 GT 对应的 pred 网：段全是 fp
    for i, (nkey, net) in enumerate(p_nets):
        if i in paired_pred:
            continue
        for v in (net.get("edges") or {}).values():
            if _edge_points(v):
                stat["A2_net_unpaired"] += 1
        for k in list((drop["nets"][nkey].get("edges") or {}).keys()):
            drop["nets"][nkey]["edges"].pop(k, None)

    scores = {
        "base": base,
        "drop_fp": net_line_f1(gt, drop, cfg, pairs).f1,
        "add_fn": net_line_f1(gt, add, cfg, pairs).f1,
        "snap_5": net_line_f1(gt, snap5, cfg, pairs).f1,
        "snap_20": net_line_f1(gt, snap20, cfg, pairs).f1,
        "snap_best": net_line_f1(gt, snapbest, cfg, pairs).f1,
    }
    return scores, stat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()
    d = os.path.join(a.run, "final_results")

    total = collections.Counter()
    acc = collections.Counter()
    rows = []
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    for cid in cases:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        scores, stat = audit_case(load_json(case.json_path), load_json(fp),
                                  cfg, {})
        for k, v in stat.items():
            total[k] += v
        for k, v in scores.items():
            acc[k] += v
        rows.append((cid, scores, stat))
        print("case %-3s line=%.4f  drop_fp=%.4f  add_fn=%.4f  snap5=%.4f  "
              "snap20=%.4f  snapbest=%.4f | fp %3d (dup %3d / 散 %3d)  fn %3d "
              "(跨 %2d / 短 %2d / 缺 %2d)"
              % (cid, scores["base"], scores["drop_fp"], scores["add_fn"],
                 scores["snap_5"], scores["snap_20"], scores["snap_best"],
                 stat["A_fp"], stat["A1_dup_of_matched"], stat["A3_other_extra"],
                 stat["B_fn"], stat["B1_spanned"], stat["B2_underspan"],
                 stat["B3_missing"]))

    n = len(rows)
    if not n:
        return 1
    print("\n== 官方 %d 张合计（net_line_f1 均值） ==" % n)
    for k in ("base", "drop_fp", "add_fn", "snap_5", "snap_20", "snap_best"):
        print("   %-10s %.4f  (%+.4f  ⇒ 加权 %+.2f 分)"
              % (k, acc[k] / n, acc[k] / n - acc["base"] / n,
                 10.0 * (acc[k] / n - acc["base"] / n)))
    fp_tot = total["A_fp"]
    print("\n-- 段账（合计 %d 条 pred 段 / %d 条 GT 段） --"
          % (fp_tot + (total["A_fp"] - total["A_fp"]), total["B_fn"]))
    print("   fp 多余段 %3d：A1 与已配上段重复 %3d / A2 网没配上 %3d / A3 其它 %3d"
          % (fp_tot, total["A1_dup_of_matched"], total["A2_net_unpaired"],
             total["A3_other_extra"]))
    print("   fn 缺失段 %3d：B1 被更长段跨过 %3d / B2 被更短段盖住 %3d / B3 没画 %3d"
          % (total["B_fn"], total["B1_spanned"], total["B2_underspan"],
             total["B3_missing"]))
    if a.out:
        import json
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"cases": [r[0] for r in rows],
                       "scores": {r[0]: r[1] for r in rows},
                       "stat": dict(total)}, fh, ensure_ascii=False, indent=1)
        print("\n已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
