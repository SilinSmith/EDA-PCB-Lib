"""_nldiag.py —— `net_line_f1` 为什么只有 0.06？

`metrics.py:522 net_line_f1` 的算法是：
  1. 只对 `net_hypergraph_f1` **配对成功**的 (GT net, pred net) 比边；
  2. 每对里做一对一匹配，边的代价是 `_seg_distance(pe[:2], ge[:2], unordered)`
     —— **两端点各自的最大距离**（不是一个端点靠上就算），`> line_tol(5px)` 就 INF；
  3. 没配上的边分别记 fp / fn。

所以 0.06 有三种可能，本脚本负责分清是哪一种：
  A 我们的边**根本没跟 GT 的边对上**（端点分叉 / 分段方式不同）→ 是「表示」问题；
  B 边对得上但**距离超 5px**（描线偏差、线宽中心差）→ 是「容差」问题；
  C 我们**压根没输出边**（`edges` 空 / 没配对上的 net）→ 是「上游没做」问题。

  python work\\tools\\diag\\_nldiag.py [--out <产物目录>] [case ...]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                     # Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, evaluate_case,       # noqa: E402
                                      normalise_doc, net_hypergraph_f1,
                                      rewrite_keys, assign_components,
                                      _edge_points, _seg_distance)
from stage2_eval import build_index, resolve                            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

OFFICIAL = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
INF = float("inf")


def one(case, outdir, cfg, verbose):
    gt_doc = load_json(Path(case.json_path))
    pp = outdir / "final_results" / (case.case_id + "_Pred.json")
    if not pp.exists():
        print("  ! 没有预测 %s" % pp)
        return None
    pred_doc = load_json(pp)
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    detail = {}
    comp, keymap = assign_components(gt, pred, cfg, detail)
    pred2 = rewrite_keys(pred, keymap)
    _net, pairs = net_hypergraph_f1(gt, pred2, cfg, detail)

    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    g_edges_all = sum(len([1 for v in (n.get("edges") or {}).values()
                           if _edge_points(v) is not None]) for _, n in g_nets)
    p_edges_all = sum(len([1 for v in (n.get("edges") or {}).values()
                           if _edge_points(v) is not None]) for _, n in p_nets)

    st = Counter()
    dists = []           # 已配对 net 里，每条 pred 边到最近 GT 边的距离
    near = []            # 最近距离（不设限），用来判「差多远才能对上」
    for i, j in pairs:
        p_e = [e for e in (_edge_points(v) for v in (p_nets[i][1].get("edges") or {}).values())
               if e is not None]
        g_e = [e for e in (_edge_points(v) for v in (g_nets[j][1].get("edges") or {}).values())
               if e is not None]
        if not p_e or not g_e:
            st["配对 net 缺边"] += 1
            continue
        for pe in p_e:
            if len(pe) < 2:
                st["我方边点数<2"] += 1
                continue
            best = INF
            for ge in g_e:
                if len(ge) < 2:
                    continue
                d = _seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                if d < best:
                    best = d
            near.append(best)
            if best <= cfg.line_tol:
                dists.append(best)
    if verbose:
        for i, (_, n) in enumerate(p_nets):
            pass
    return dict(case=case.case_id, split=case.split, pairs=len(pairs),
                g_nets=len(g_nets), p_nets=len(p_nets),
                g_edges=g_edges_all, p_edges=p_edges_all,
                st=dict(st), near=near)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work\\_fast_out_novlm9")
    ap.add_argument("--cases", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    cases = [c for c in a.cases.split(",") if c] or OFFICIAL
    cfg = EvalConfig()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    rows = []
    print("线宽容差 line_tol=%.1f  unordered=%s" % (cfg.line_tol, cfg.line_unordered))
    print("%-6s %-6s %6s %6s %6s %7s %7s | %s"
          % ("case", "pairs", "gN", "pN", "gE", "pE", "命中", "最近距离分位(命中/未命中)"))
    print("-" * 100)
    for cid in cases:
        case, how = resolve(cid, exact, digits)
        if case is None:
            case, how = resolve(cid.lstrip("0") or "0", exact, digits)
        if case is None:
            print("! 找不到 %s" % cid)
            continue
        r = one(case, outdir, cfg, a.verbose)
        if r is None:
            continue
        rows.append(r)
        n = r["near"]
        hit = [d for d in n if d <= cfg.line_tol]
        miss = sorted(d for d in n if d > cfg.line_tol)
        pct = ("%.0f/%.0f/%.0f" % (miss[0], statistics.median(miss), miss[-1])
               if miss else "-")
        print("%-6s %-6d %6d %6d %6d %7d %7d | 未命中 %d 条，最近距离 min/中位/max = %s px"
              % (r["case"], r["pairs"], r["g_nets"], r["p_nets"], r["g_edges"],
                 r["p_edges"], len(hit), len(miss), pct))
        if r["st"]:
            print("        %s" % r["st"])
    print("-" * 100)
    he = sum(r["g_edges"] for r in rows)
    pe = sum(r["p_edges"] for r in rows)
    print("合计 GT 边 %d / 我方边 %d（多了 %.1f%%）"
          % (he, pe, 100.0 * (pe - he) / max(1, he)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
