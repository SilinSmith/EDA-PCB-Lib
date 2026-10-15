"""net_line 失分分解：把 tp/fp/fn 拆成「网没配上」「网配上了但边数不对」「边数对但端点超差」。

用法（从仓库根跑）：
    python -X utf8 work\\tools\\diag\\_nlfork.py --pred work\\_fnew
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.metrics import (  # noqa: E402
    _edge_points, _seg_distance, match_nets, _net_sets, EvalConfig)

GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
CASES = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


def gt_path(case: str) -> Path:
    return GT_DIR / case / f"{case}.json"


def edges_of(net):
    out = []
    for v in (net.get("edges") or {}).values():
        e = _edge_points(v)
        if e is not None:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir

    tot = Counter()
    dist_all = []
    rows = []
    for case in a.cases.split(","):
        case = case.strip()
        gt = json.loads(gt_path(case).read_text(encoding="utf-8"))
        pr = json.loads((pred_dir / f"{case}.json").read_text(encoding="utf-8"))
        g_nets = list(gt["nets"].items())
        p_nets = list(pr["nets"].items())
        cfg = EvalConfig()
        g_sets = _net_sets(gt["nets"], gt.get("components"), cfg.net_pin_collapse_nonbox)
        p_sets = _net_sets(pr["nets"], pr.get("components"), cfg.net_pin_collapse_nonbox)
        pairs = match_nets(g_sets, p_sets)
        pm = {i for i, _ in pairs}
        gm = {j for _, j in pairs}
        c = Counter()
        c["gt_nets"] = len(g_nets)
        c["pred_nets"] = len(p_nets)
        c["matched"] = len(pairs)
        c["gt_edges"] = sum(len(edges_of(n)) for _, n in g_nets)
        c["pred_edges"] = sum(len(edges_of(n)) for _, n in p_nets)
        c["pred_edges_unmatched_net"] = sum(
            len(edges_of(n)) for i, (_, n) in enumerate(p_nets) if i not in pm)
        c["gt_edges_unmatched_net"] = sum(
            len(edges_of(n)) for j, (_, n) in enumerate(g_nets) if j not in gm)
        per_net = []
        for i, j in pairs:
            pe = edges_of(p_nets[i][1])
            ge = edges_of(g_nets[j][1])
            got = 0
            best = []
            used = set()
            for p in pe:
                cand = []
                for k, g in enumerate(ge):
                    if len(p) < 2 or len(g) < 2:
                        continue
                    d = _seg_distance(p[:2], g[:2], True)
                    cand.append((d, k))
                    best.append(d)
                cand.sort()
                for d, k in cand:
                    if d <= a.tol and k not in used:
                        used.add(k)
                        got += 1
                        break
            c["tp"] += got
            c["fp_matched"] += len(pe) - got
            c["fn_matched"] += len(ge) - got
            c["p_edges_matched"] += len(pe)
            c["g_edges_matched"] += len(ge)
            dist_all.extend(best)
            if len(pe) != len(ge) or got != len(ge):
                per_net.append((len(pe), len(ge), got, p_nets[i][0], g_nets[j][0],
                                sorted(best)[:4]))
        per_net.sort(key=lambda r: (r[2] - r[1]))
        rows.append((case, c, per_net[:a.top]))
        tot.update(c)

    print(f"pred = {pred_dir}")
    print(f"{'case':>5} {'gtN':>4} {'pdN':>4} {'mt':>3} {'gtE':>5} {'pdE':>5} "
          f"{'tp':>4} {'fpM':>4} {'fnM':>4} {'pdE_un':>7} {'gtE_un':>7}")
    for case, c, _ in rows:
        print(f"{case:>5} {c['gt_nets']:>4} {c['pred_nets']:>4} {c['matched']:>3} "
              f"{c['gt_edges']:>5} {c['pred_edges']:>5} {c['tp']:>4} {c['fp_matched']:>4} "
              f"{c['fn_matched']:>4} {c['pred_edges_unmatched_net']:>7} "
              f"{c['gt_edges_unmatched_net']:>7}")
    print(f"{'ALL':>5} {tot['gt_nets']:>4} {tot['pred_nets']:>4} {tot['matched']:>3} "
          f"{tot['gt_edges']:>5} {tot['pred_edges']:>5} {tot['tp']:>4} "
          f"{tot['fp_matched']:>4} {tot['fn_matched']:>4} "
          f"{tot['pred_edges_unmatched_net']:>7} {tot['gt_edges_unmatched_net']:>7}")
    print(f"\nnetL 理论 f1 = 2*tp/(2*tp+fp+fn) ，fp={tot['fp_matched'] + tot['pred_edges_unmatched_net']}"
          f" fn={tot['fn_matched'] + tot['gt_edges_unmatched_net']}")

    if dist_all:
        ds = sorted(dist_all)
        n = len(ds)
        q = lambda p: ds[min(n - 1, int(p * n))]
        print(f"\n每条 pred 边到最近 GT 边的端点距离（同网配对里，n={n}）："
              f" p10={q(.1):.1f} p25={q(.25):.1f} 中位={q(.5):.1f} p75={q(.75):.1f} p90={q(.9):.1f}")
        buckets = Counter()
        for d in ds:
            if d <= 5:
                buckets["<=5 命中"] += 1
            elif d <= 8:
                buckets["5~8"] += 1
            elif d <= 13:
                buckets["8~13"] += 1
            elif d <= 25:
                buckets["13~25"] += 1
            elif d <= 60:
                buckets["25~60"] += 1
            else:
                buckets[">60"] += 1
        print("  距离分桶：", dict(buckets))

    print("\n最差的若干网（pred 边数 / GT 边数 / 命中 / predKey / gtKey / 最近距离前 4）：")
    for case, _, per in rows:
        if not per:
            continue
        print(f"  --- case {case} ---")
        for pe, ge, got, pk, gk, near in per:
            print(f"    p={pe:>2} g={ge:>2} hit={got:>2}  {pk:<14} {gk:<14} {[round(x,1) for x in near]}")


if __name__ == "__main__":
    main()
