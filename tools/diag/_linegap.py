# -*- coding: utf-8 -*-
"""§5.113 取证：**导线通道（net_line，权重 0.10）的亏空结构**。

口径（`metrics.net_line_f1`，已读源码）：只在**配对成功的网对**内部比边；
每条边只用 `points[:2]`（两个端点），`_seg_distance` = 两端点距离的 **max**（允许反转），
≤ `line_tol=5.0` 才算能配，网内匈牙利一对一；配不上的 GT 边记 fn、我方边记 fp。
未配对的网的所有边直接记 fn/fp。

本探针逐配对输出 tp/fp/fn，并把配不上的边分类：
  cover   这条 GT 边的两个端点都落在**我方某条边的内部**（≥3px 离端点）⇒ 我方把几段并成一条
  near    到最近的边距离 ≤40px（几何在附近但端点点位差 >5px）
  no_ours 40px 内没有任何我方边 ⇒ 根本没有几何
我方多出的边同理反向分类。

用法（从 Y:\\workspace\\EDA\\eda-pcb 跑）：
    python -X utf8 ..\\work\\_linegap.py --run work/_fast_out_gsd9 --cases 1,2,3,4,5,6,7,8,9,10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--show", type=int, default=8)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()


def seg_pt_dist(seg, q):
    """点 q 到线段 seg 的距离。"""
    (x1, y1), (x2, y2) = seg
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return math.hypot(q[0] - x1, q[1] - y1)
    t = max(0.0, min(1.0, ((q[0] - x1) * dx + (q[1] - y1) * dy) / L2))
    return math.hypot(q[0] - (x1 + t * dx), q[1] - (y1 + t * dy))


def classify(e, others):
    """e = [(x,y),(x,y)]，others = 我方/GT 的边列表。"""
    if not others:
        return 'no_ours', 9e9
    best = min(M._seg_distance(e, o, True) for o in others)
    if best <= 40.0:
        return 'near', best
    # 端点是否落在我方某条边内部（并段）
    for o in others:
        d = max(seg_pt_dist(o, e[0]), seg_pt_dist(o, e[1]))
        de = min(math.hypot(e[0][0] - o[0][0], e[0][1] - o[0][1]),
                 math.hypot(e[0][0] - o[1][0], e[0][1] - o[1][1]),
                 math.hypot(e[1][0] - o[0][0], e[1][1] - o[0][1]),
                 math.hypot(e[1][0] - o[1][0], e[1][1] - o[1][1]))
        if d <= 3.0 and de >= 3.0:
            return 'cover', best
    return 'no_ours', best


tot = Counter()
sum_f1 = 0.0
rows = []
for cid in [s.strip() for s in args.cases.split(",") if s.strip()]:
    if cid not in cases:
        print("!! 没有 case %s" % cid); continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    _, km = M.assign_components(gt, pred, cfg, {})
    pred = M.rewrite_keys(pred, km)
    g_nets = list(gt['nets'].items()); p_nets = list(pred['nets'].items())
    pairs = M.match_nets(M._net_sets(gt['nets'], gt['components'], False),
                         M._net_sets(pred['nets'], pred['components'], False), soft=True)
    f1 = M.net_line_f1(gt, pred, cfg, pairs)
    sum_f1 += f1.f1
    cat = Counter(); ex = []
    mset_p = {i for i, _ in pairs}; mset_g = {j for _, j in pairs}
    for i, j in pairs:
        pe = [e for e in (M._edge_points(v) for v in (p_nets[i][1].get('edges') or {}).values()) if e]
        ge = [e for e in (M._edge_points(v) for v in (g_nets[j][1].get('edges') or {}).values()) if e]
        if not pe or not ge:
            cat['empty.gt'] += len(ge); cat['empty.pr'] += len(pe); continue
        cost = [[(M._seg_distance(p[:2], g[:2], cfg.line_unordered)
                  if min(len(p), len(g)) >= 2 else float('inf')) for g in ge] for p in pe]
        cost = [[(d if d <= cfg.line_tol else float('inf')) for d in row] for row in cost]
        got = M._assign(cost)
        gp = {gi for _, gi in got}; pp = {pi for pi, _ in got}
        for gi, g in enumerate(ge):
            if gi in gp:
                cat['gt.hit'] += 1; continue
            tag, d = classify(g[:2], [p[:2] for p in pe])
            cat['gt.' + tag] += 1
            if len(ex) < args.show:
                ex.append("GT %s %s %s(%.0f,%.0f)->(%.0f,%.0f) d=%.1f"
                          % (cid, g_nets[j][0], tag, g[0][0], g[0][1], g[1][0], g[1][1], d))
        for pi, p in enumerate(pe):
            if pi in pp:
                cat['pr.hit'] += 1; continue
            tag, d = classify(p[:2], [g[:2] for g in ge])
            cat['pr.' + tag] += 1
            if len(ex) < args.show:
                ex.append("PR %s %s %s(%.0f,%.0f)->(%.0f,%.0f) d=%.1f"
                          % (cid, p_nets[i][0], tag, p[0][0], p[0][1], p[1][0], p[1][1], d))
    for i, (_, net) in enumerate(p_nets):
        if i not in mset_p:
            cat['pr.unpaired_net'] += len([v for v in (net.get('edges') or {}).values()
                                           if M._edge_points(v)])
    for j, (_, net) in enumerate(g_nets):
        if j not in mset_g:
            cat['gt.unpaired_net'] += len([v for v in (net.get('edges') or {}).values()
                                           if M._edge_points(v)])
    rows.append((cid, f1.f1, f1.tp, f1.fp, f1.fn))
    print("== case %-3s net_line F1 %.4f  tp %d fp %d fn %d" % (cid, f1.f1, f1.tp, f1.fp, f1.fn))
    print("   %s" % {k: v for k, v in sorted(cat.items()) if v})
    for e in ex[:4]:
        print("     " + e)
    tot.update(cat)
n = max(1, len(rows))
print("\n===== 汇总（%d 张）net_line 均值 %.4f =====" % (len(rows), sum_f1 / n))
for k, v in sorted(tot.items()):
    print("   %-18s %d" % (k, v))
