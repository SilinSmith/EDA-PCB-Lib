# -*- coding: utf-8 -*-
"""§5.112 取证：**netH 的天花板**——把 GT 里所有 ∅ 电源脚的 refs 删掉当预测来评。

理由（`work/_netgap.py` 实测）：我方缺的 115 个网成员里 **108 个是 ∅ 电源脚**
（gnd 59 / v 49），只有 7 个是真几何错（box, pin_pos_far）；而 ∅ 的脚号 = 标注器
全局计数器（键名−2），**不可从图里复现**。所以「网络结构全对、只丢电源脚」就是
netH 的现实天花板；本探针把它算出来，用来判断这条通道还剩多少真实净空。

用法（从 Y:\\workspace\\EDA 跑）：python -X utf8 work\\_netceil.py --arm gsd9
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda-pcb"))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import (DEFAULT_DATA_ROOT, build_hypergraph,  # noqa: E402
                                   discover, load_json, normalise_doc)

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="gsd9")
ap.add_argument("--cases", default=os.path.join(ROOT, "work", "_cases10gt.txt"))
args = ap.parse_args()

IDX = {}
for c in discover(DEFAULT_DATA_ROOT):
    if c.split == "10GT" or (c.split == "200_train_cases" and c.json_path.endswith("target_named.json")):
        IDX[c.case_id] = c
PRED = os.path.join(ROOT, "work", "_fast_out_" + args.arm, "final_results")
cfg = M.EvalConfig()


def strip_power(doc):
    out = dict(doc)
    out['components'] = {k: v for k, v in (doc.get('components') or {}).items()
                         if not k.startswith("∅")}
    out['pins'] = {k: v for k, v in (doc.get('pins') or {}).items() if not k.startswith("∅")}
    nets = {}
    for name, net in (doc.get('nets') or {}).items():
        if not isinstance(net, dict):
            nets[name] = net; continue
        nn = dict(net)
        nn['hyperGraph'] = build_hypergraph(
            [(c, p) for c, p in M.parse_hypergraph(net.get('hyperGraph', ''))
             if not c.startswith("∅")])
        nets[name] = nn
    out['nets'] = nets
    return out


ours_l, ceil_l = [], []
for cid in [x.strip() for x in io.open(args.cases, encoding="utf-8").read().strip().split(",") if x.strip()]:
    case = IDX.get(cid)
    pp = os.path.join(PRED, cid + "_Pred.json")
    if case is None or not os.path.exists(pp):
        print("!! 缺 case %s" % cid); continue
    gt = normalise_doc(load_json(case.json_path))
    pred = load_json(pp)
    _, km = M.assign_components(gt, pred, cfg, {})
    pred = M.rewrite_keys(pred, km)
    f_ours = M.net_hypergraph_f1(gt, pred, cfg)[0]
    gtp = strip_power(gt)
    f_ceil = M.net_hypergraph_f1(gt, gtp, cfg)[0]
    ours_l.append(f_ours.f1); ceil_l.append(f_ceil.f1)
    print("case %-3s 我方 netH %.4f (tp %.2f) | 天花板 %.4f (tp %.2f) | 净空 %.4f = %.4f 分"
          % (cid, f_ours.f1, f_ours.tp, f_ceil.f1, f_ceil.tp,
             f_ceil.f1 - f_ours.f1, (f_ceil.f1 - f_ours.f1) * 0.35))
n = max(1, len(ours_l))
mo, mc = sum(ours_l) / n, sum(ceil_l) / n
print("\n== 均值（%d 张）：我方 %.4f | 天花板 %.4f | 真实净空 %.4f netH = **%.4f 分**"
      % (len(ours_l), mo, mc, mc - mo, (mc - mo) * 0.35))
