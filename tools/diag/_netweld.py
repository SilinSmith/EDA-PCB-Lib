# -*- coding: utf-8 -*-
"""`netH` 的 `D_在网里但配错`（10GT 上 96 个 / 0.229）对策实验：**按几何把网络焊起来**。

背景链：
- `_netwhy.py`：10GT 的 `fn` 里 `C + D = 146 / 277`。
- `_lineweld.py`：把共线片段焊成整条直线，`net_line` 只 `+0.0006`、`tp 22→22`、`fn 531→531`
  ⇒ **`net_line` 不是独立的病，它继承了「网络配错」这个上游问题**。
- 我们的网络是**过碎片化**的（case `1`：我们 20 个网络 vs GT 10 个；`net_line` 端距 `>120px` 占 58%）。

已经做过的三种合并（`merge_by_label` 按标签、`merge_shared_pins` 按共享引脚、
`attach_orphan_rails` 按轨名救悬空引脚）都**不覆盖**这一种：
**两个网络的导线片段在几何上首尾相接（同一个 junction），却被 stage1 分成了两个网络。**
「一个 junction 上的线是连的」和「一个引脚只能属于一个网络」一样是电路的基本事实。

做法（离线，只改预测文档的 `nets` 再按官方口径打分）：
把每个网络的**全部边端点**收集起来，任意两个网络的端点集合里有距离 ≤ `--snap` 的一对，
就把这两个网络并起来（DSU），边并集、引脚并集、`hyperGraph` 重写。

用法：`python work\\tools\\diag\\_netweld.py [--pred 目录] [--cases 1,2,3] [--snap 2.0] [--min-len 0]`
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
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


def refs_of(hg: str):
    out = []
    for tok in str(hg or "").strip().strip("()").split(","):
        tok = tok.strip()
        if "." in tok:
            c, _, p = tok.rpartition(".")
            out.append((c.strip(), p.strip()))
    return out


def ends_of(rec):
    out = set()
    for ev in (rec.get("edges") or {}).values():
        pts = [(round(float(p["x"]), 1), round(float(p["y"]), 1))
               for p in (ev or []) if "x" in p and "y" in p]
        if pts:
            out.add(pts[0])
            out.add(pts[-1])
    return out


def weld(doc: dict, snap: float = 2.0, min_len: int = 0) -> tuple:
    d = copy.deepcopy(doc)
    nets = d.get("nets") or {}
    names = list(nets)
    if len(names) < 2:
        return d, 0
    parent = {k: k for k in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ends = {n: ends_of(nets[n]) for n in names}
    grid = collections.defaultdict(list)          # 端点 -> 有哪些网络
    for n in names:
        for pt in ends[n]:
            grid[(round(pt[0] / max(snap, 0.5)), round(pt[1] / max(snap, 0.5)))].append(n)
    # 同一格里的网络互连；再查相邻 8 格（处理刚好跨格的情况）
    for (gx, gy), members in list(grid.items()):
        near = list(members)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    near += grid.get((gx + dx, gy + dy), [])
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                ra, rb = find(near[i]), find(near[j])
                if ra != rb:
                    parent[ra] = rb

    groups = collections.defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    if len(groups) == len(names):
        return d, 0

    out = {}
    for members in groups.values():
        edges, refs, seen_r, seen_e = {}, [], set(), set()
        for n in members:
            for ref in refs_of(nets[n].get("hyperGraph", "")):
                if ref not in seen_r:
                    seen_r.add(ref)
                    refs.append(ref)
            for ev in (nets[n].get("edges") or {}).values():
                pts = tuple((round(float(p["x"]), 1), round(float(p["y"]), 1))
                            for p in (ev or []) if "x" in p and "y" in p)
                if len(pts) < 2 or pts in seen_e:
                    continue
                seen_e.add(pts)
                edges["edge_%d" % (len(edges) + 1)] = [
                    {"x": p[0], "y": p[1]} for p in pts]
        if len(members) > 1 and min_len and len(edges) < min_len:
            pass
        rec = dict(nets[members[0]])
        rec["edges"] = edges
        rec["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
        out["net_%d" % (len(out) + 1)] = rec
    d["nets"] = out
    return d, len(names) - len(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--snap", type=float, default=2.0)
    ap.add_argument("--min-len", type=int, default=0)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or TEN
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    agg = collections.Counter()
    print("%-7s %8s %8s %8s | %-16s %-16s %s" %
          ("case", "原总分", "焊后", "Δ", "netH 原->新", "line 原->新", "网数 原->新"))
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        s0 = evaluate_case(gt, pr, case_id=cid, cfg=cfg)
        d, merged = weld(pr, a.snap, a.min_len)
        s1 = evaluate_case(gt, d, case_id=cid, cfg=cfg)
        for k, v0, v1 in (("s", s0.score, s1.score),
                          ("h", s0.net_hypergraph.f1, s1.net_hypergraph.f1),
                          ("tp", s0.net_hypergraph.tp, s1.net_hypergraph.tp),
                          ("fp", s0.net_hypergraph.fp, s1.net_hypergraph.fp),
                          ("fn", s0.net_hypergraph.fn, s1.net_hypergraph.fn),
                          ("l", s0.net_line.f1, s1.net_line.f1),
                          ("ltp", s0.net_line.tp, s1.net_line.tp)):
            agg[k + "0"] += v0
            agg[k + "1"] += v1
        agg["c"] += 1
        print("%-7s %8.2f %8.2f %+8.2f | %.4f -> %.4f  %.4f -> %.4f   %d -> %d%s" %
              (cid, s0.score, s1.score, s1.score - s0.score,
               s0.net_hypergraph.f1, s1.net_hypergraph.f1,
               s0.net_line.f1, s1.net_line.f1,
               len(pr.get("nets") or {}), len(d.get("nets") or {}),
               "" if merged == 0 else "  (并掉 %d)" % merged))
    n = max(1, agg["c"])
    print()
    print("合计 %d 张（snap=%.1f）：总分 %.2f -> %.2f (%+.2f) | netH %.4f -> %.4f (%+.4f) | line %.4f -> %.4f"
          % (n, a.snap, agg["s0"] / n, agg["s1"] / n, (agg["s1"] - agg["s0"]) / n,
             agg["h0"] / n, agg["h1"] / n, (agg["h1"] - agg["h0"]) / n,
             agg["l0"] / n, agg["l1"] / n))
    print("netH 明细 tp %d->%d fp %d->%d fn %d->%d ｜ line tp %d->%d"
          % (agg["tp0"], agg["tp1"], agg["fp0"], agg["fp1"], agg["fn0"], agg["fn1"],
             agg["ltp0"], agg["ltp1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
