# -*- coding: utf-8 -*-
"""网络合并的离线实验：不动管线，只改预测文档的 `nets`，看 `net_hypergraph` / `net_line` 涨不涨。

动机（实测，`work\\_submit_out220z\\final_results`）：

    case 1    GT 10 net / 60 pins / 102 edges  |  我们 19 net /  25 pins / 371 edges
    case 0092 GT 51 net / 143 pins / 156 edges |  我们  2 net /   2 pins /   2 edges
    case 0048 GT 30 net / 78 pins / 128 edges  |  我们  0 net /   0 pins /   0 edges
    case 0086 GT 52 net / 154 pins / 223 edges |  我们  2 net /   2 pins /   8 edges

而且 case 1 的 `net_1..net_4` 里 **同一个 `C15.2` 同时出现在 net_1 / net_3 / net_4** ——
一个引脚只能属于一个网络，说明装配时**没有做全局连通闭合**，碎片化 + 重复归属。

`net_line_f1` 只在 `match_nets` 配上的网络对内比边，而 `match_nets` 是按**引脚集合**配的
⇒ 引脚集合碎了，`net_line` 自然塌成 0。所以两个指标是同一个病。

用法::

    python work\\tools\\diag\\_netmerge.py                     # 快测集 19 张
    python work\\tools\\diag\\_netmerge.py --cases 1,2,0092
    python work\\tools\\diag\\_netmerge.py --all
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, normalise_doc  # noqa: E402
from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, evaluate_case, net_pin_set, parse_hypergraph,
)
from stage2_eval import build_index, resolve, load_json  # noqa: E402

GND_RE = re.compile(r"^(/?)(GND|VSS|AGND|DGND|PGND|GNDA|GNDD|GNDIO|VSSA|GNDS)", re.I)
PWR_RE = re.compile(r"^(/?)(VDD|VCC|VIN|VBAT|VBUS|3V3|5V|VOUT|VDC)", re.I)


def fmt(refs):
    return "(" + ",".join("%s.%s" % (c, p) for c, p in sorted(refs)) + ")"


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def merge_nets(pred, rail=False, pins=None):
    """把**共享引脚**的网络做传递闭包合并；`rail=True` 时再按「接地/电源」名归并。"""
    nets = pred.get("nets") or {}
    if not nets:
        return pred, 0, 0
    names = list(nets)
    dsu = DSU()
    owner = {}
    pin_owner = {}
    for n in names:
        dsu.find(n)
        for ref in net_pin_set(nets[n]):
            key = (ref[0], ref[1])
            pin_owner.setdefault(key, []).append(n)
            owner.setdefault(key, n)
    for key, ns in pin_owner.items():
        for x in ns[1:]:
            dsu.union(ns[0], x)

    if rail and pins:
        rail_of = {}
        for n in names:
            members = net_pin_set(nets[n])
            if not members:
                continue
            tags = []
            for c, p in members:
                nm = str(((pins.get(c) or {}).get("pin_%s" % p) or {}).get("pinname") or "")
                m = GND_RE.match(nm)
                if m:
                    tags.append("GND"); continue
                if PWR_RE.match(nm):
                    tags.append("PWR")
            if len(tags) >= 2 and len(set(tags)) == 1:
                rail_of.setdefault(tags[0], []).append(n)
        for _tag, ns in rail_of.items():
            for x in ns[1:]:
                dsu.union(ns[0], x)

    groups = {}
    for n in names:
        groups.setdefault(dsu.find(n), []).append(n)
    if len(groups) == len(names):
        return pred, 0, 0

    out = copy.deepcopy(pred)
    newnets = {}
    k = 0
    for _r, members in groups.items():
        k += 1
        refs = set()
        edges = {}
        e = 0
        for n in members:
            refs |= net_pin_set(nets[n])
            for ev in (nets[n].get("edges") or {}).values():
                e += 1
                edges["edge_%d" % e] = ev
        newnets["net_%d" % k] = {"hyperGraph": fmt(refs), "edges": edges}
    out["nets"] = newnets
    return out, len(names), len(newnets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10,0092,0048,0019,0024,0075,0086,0002,0001,0042,0103")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pred", default=str(ROOT / "work" / "_submit_out220z" / "final_results"))
    ap.add_argument("--rail", action="store_true", help="再加「接地/电源」名归并")
    a = ap.parse_args()

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cids = None if a.all else [s.strip() for s in a.cases.split(",") if s.strip()]
    if a.all:
        cids = sorted({c.case_id for c in build_index(DEFAULT_DATA_ROOT)[0].values()})
    pred_dir = Path(a.pred)

    tot = {"n": 0, "b": [0.0, 0.0, 0.0, 0.0], "a": [0.0, 0.0, 0.0, 0.0],
           "nb": [0, 0], "na": [0, 0]}
    rows = []
    for cid in cids:
        case, _how = resolve(cid, exact, digits)
        if case is None:
            continue
        pf = pred_dir / ("%s_Pred.json" % case.case_id)
        if not pf.exists():
            pf = pred_dir / ("%s.json" % case.case_id)
        gt = normalise_doc(load_json(case.json_path))
        pr = normalise_doc(load_json(pf))
        s0 = evaluate_case(gt, pr, case.case_id, EvalConfig())
        pr2, n0, n1 = merge_nets(pr, a.rail, pr.get("pins"))
        s1 = evaluate_case(gt, pr2, case.case_id, EvalConfig())
        b = [s0.component.f1, s0.pin.f1, s0.net_hypergraph.f1, s0.net_line.f1]
        c = [s1.component.f1, s1.pin.f1, s1.net_hypergraph.f1, s1.net_line.f1]
        d0 = 100 * (0.30 * b[0] + 0.25 * b[1] + 0.35 * b[2] + 0.10 * b[3])
        d1 = 100 * (0.30 * c[0] + 0.25 * c[1] + 0.35 * c[2] + 0.10 * c[3])
        tot["n"] += 1
        for i in range(4):
            tot["b"][i] += b[i]; tot["a"][i] += c[i]
        tot["nb"][0] += n0; tot["nb"][1] += len(pr.get("nets") or {})
        tot["na"][0] += n1; tot["na"][1] += len(pr2.get("nets") or {})
        rows.append((case.case_id, d0, d1, b, c, n0, n1))

    rows.sort(key=lambda r: r[2] - r[1])
    print("%-8s %8s %8s %8s | %s" % ("case", "原分", "合并后", "Δ", "net数 原->新"))
    for cid, d0, d1, b, c, n0, n1 in rows[:40]:
        print("%-8s %8.2f %8.2f %+8.2f | %5d -> %-5d  netH %.3f->%.3f  line %.3f->%.3f"
              % (cid, d0, d1, d1 - d0, n0, n1, b[2], c[2], b[3], c[3]))
    if not rows:
        print("没有可用样本"); return 0
    n = tot["n"]
    print("\n== 合计 %d 张" % n)
    for i, nm in enumerate(("元件", "引脚", "网络", "导线")):
        print("   %s F1 %.4f -> %.4f  (%+.4f)" % (nm, tot["b"][i] / n, tot["a"][i] / n, (tot["a"][i] - tot["b"][i]) / n))
    w = (0.30, 0.25, 0.35, 0.10)
    s0 = 100 * sum(w[i] * tot["b"][i] / n for i in range(4))
    s1 = 100 * sum(w[i] * tot["a"][i] / n for i in range(4))
    print("   总分 %.2f -> %.2f  (%+.2f)" % (s0, s1, s1 - s0))
    print("   网络数 %d -> %d" % (tot["nb"][1], tot["na"][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
