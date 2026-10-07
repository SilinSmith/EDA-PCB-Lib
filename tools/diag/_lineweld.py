# -*- coding: utf-8 -*-
"""`net_line`（权重 0.10）爆 0 的对策实验：把「共线且首尾相接」的导线片段焊成整条直线。

`_linewhy.py` 已查明官方口径：`_seg_distance(p, q, unordered)` **不是线段到线段的距离**，
而是 `max(hypot(p0-q0), hypot(p1-q1))` —— **两个端点都要在 `cfg.line_tol = 5.0` 内**才算一条 tp。
所以「我们一条 67px 的直线被切成了 3 段 22px」时，GT 那条 67px 的边一条都配不上（全 fp+fn）。

实测（`_linewhy.py`，case `1`）：网络配上了 10 对，307 条边 tp=0，
端距直方 `5~15 16% / 15~40 10% / 40~120 17% / >120 58%`；
边长 我们 p50 **41** / GT p50 **67** ⇒ **我们的边整体偏短、数量偏多**（371 条 vs GT 102 条）。

做法（离线，只改预测文档的 `edges` 再按官方口径打分）：
对每个网络，把端点相接（≤ `--snap` px）且方向夹角很小（≤ `--ang` 度）的片段串成链，
再按链的整体方向重新取端点。`--ang 180` 表示只看「端点相接」不看方向。

用法：`python work\\tools\\diag\\_lineweld.py [--pred 目录] [--cases 1,2,3] [--ang 25] [--snap 2.0]`
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402

TEN = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def _pts(ev):
    return [(float(p["x"]), float(p["y"])) for p in (ev or []) if "x" in p and "y" in p]


def _key(p, snap):
    return (round(p[0] / snap), round(p[1] / snap))


def _ang(a, b, c):
    """b 处、从 a 到 c 的转角（度）；0 = 完全反向（一条直线），180 = 折回。"""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def weld_edges(edges: dict, ang_tol: float = 25.0, snap: float = 2.0) -> dict:
    """把首尾相接、且共线的片段焊成一条。返回新的 `edge_N -> [p0, p1]`。"""
    segs = []
    for ev in (edges or {}).values():
        p = _pts(ev)
        if len(p) >= 2:
            segs.append((p[0], p[-1]))
    if len(segs) < 2:
        return edges
    used = [False] * len(segs)
    ends = collections.defaultdict(list)
    for i, (a, b) in enumerate(segs):
        ends[_key(a, snap)].append(i)
        ends[_key(b, snap)].append(i)

    out = {}
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        a, b = segs[i]
        chain = [a, b]
        grew = True
        while grew:
            grew = False
            for endidx in (0, -1):
                tip = chain[endidx]
                for j in ends.get(_key(tip, snap), []):
                    if used[j]:
                        continue
                    q0, q1 = segs[j]
                    tail = q1 if _key(q0, snap) == _key(tip, snap) else q0
                    if _key(tail, snap) == _key(tip, snap):
                        continue                       # 退化
                    if endidx == 0:
                        nxt = [tail] + chain
                        tst = (tail, chain[0], chain[1])
                    else:
                        nxt = chain + [tail]
                        tst = (chain[-2], chain[-1], tail)
                    if len(chain) >= 2 and _ang(*tst) > ang_tol:
                        continue                       # 拐弯了，不焊
                    chain = nxt
                    used[j] = True
                    grew = True
                    break
                if grew:
                    break
        out["edge_%d" % (len(out) + 1)] = [{"x": round(chain[0][0], 2), "y": round(chain[0][1], 2)},
                                           {"x": round(chain[-1][0], 2), "y": round(chain[-1][1], 2)}]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--ang", type=float, default=25.0)
    ap.add_argument("--snap", type=float, default=2.0)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or TEN
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()

    agg = collections.Counter()
    print("%-7s %8s %8s %8s | %-16s %s" %
          ("case", "原总分", "焊后", "Δ", "line 原->新", "边数 原->新"))
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        s0 = evaluate_case(gt, pr, case_id=cid, cfg=cfg)
        d = copy.deepcopy(pr)
        n0 = n1 = 0
        for rec in (d.get("nets") or {}).values():
            n0 += len(rec.get("edges") or {})
            rec["edges"] = weld_edges(rec.get("edges") or {}, a.ang, a.snap)
            n1 += len(rec["edges"])
        s1 = evaluate_case(gt, d, case_id=cid, cfg=cfg)
        for k, v0, v1 in (("s", s0.score, s1.score), ("l", s0.net_line.f1, s1.net_line.f1),
                          ("tp", s0.net_line.tp, s1.net_line.tp), ("fp", s0.net_line.fp, s1.net_line.fp),
                          ("fn", s0.net_line.fn, s1.net_line.fn), ("n", s0.net_hypergraph.f1, s1.net_hypergraph.f1)):
            agg[k + "0"] += v0
            agg[k + "1"] += v1
        agg["c"] += 1
        print("%-7s %8.2f %8.2f %+8.2f | %.4f -> %.4f    %d -> %d" %
              (cid, s0.score, s1.score, s1.score - s0.score,
               s0.net_line.f1, s1.net_line.f1, n0, n1))

    n = max(1, agg["c"])
    print()
    print("合计 %d 张（ang=%.0f° snap=%.1f）：总分 %.2f -> %.2f (%+.2f) | line %.4f -> %.4f (%+.4f)"
          % (n, a.ang, a.snap, agg["s0"] / n, agg["s1"] / n, (agg["s1"] - agg["s0"]) / n,
             agg["l0"] / n, agg["l1"] / n, (agg["l1"] - agg["l0"]) / n))
    print("line 明细  tp %d->%d  fp %d->%d  fn %d->%d"
          % (agg["tp0"], agg["tp1"], agg["fp0"], agg["fp1"], agg["fn0"], agg["fn1"]))
    print("netH       %.4f -> %.4f" % (agg["n0"] / n, agg["n1"] / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
