"""net_line 结构归因：把「我们边太多」拆成 ①双描边（平行近重复）②过分割（GT 长边被切成几段）。

用法（从仓库根跑）：
    python -X utf8 work\\tools\\diag\\_nlfork2.py --pred work\\_fnew
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.metrics import (  # noqa: E402
    _edge_points, match_nets, _net_sets, EvalConfig)

GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
CASES = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


def edges_of(net):
    out = []
    for v in (net.get("edges") or {}).values():
        e = _edge_points(v)
        if e is not None and len(e) >= 2:
            out.append((e[0], e[1]))
    return out


def pt_seg(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def ang(a, b):
    ax, ay = a[1][0] - a[0][0], a[1][1] - a[0][1]
    bx, by = b[1][0] - b[0][0], b[1][1] - b[0][1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return abs((ax * bx + ay * by) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--near", type=float, default=4.0, help="双描边判据：平行且端点都近")
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir

    tot = Counter()
    for case in a.cases.split(","):
        case = case.strip()
        gt = json.loads((GT_DIR / case / f"{case}.json").read_text(encoding="utf-8"))
        pr = json.loads((pred_dir / f"{case}.json").read_text(encoding="utf-8"))
        g_nets = list(gt["nets"].items())
        p_nets = list(pr["nets"].items())
        cfg = EvalConfig()
        pairs = match_nets(_net_sets(gt["nets"], gt.get("components"), cfg.net_pin_collapse_nonbox),
                           _net_sets(pr["nets"], pr.get("components"), cfg.net_pin_collapse_nonbox))
        c = Counter()
        c["gt_edges"] = sum(len(edges_of(n)) for _, n in g_nets)
        c["pred_edges"] = sum(len(edges_of(n)) for _, n in p_nets)
        for i, j in pairs:
            pe = edges_of(p_nets[i][1])
            ge = edges_of(g_nets[j][1])
            c["p_in"] += len(pe)
            c["g_in"] += len(ge)
            # ① 双描边：同一 net 内两条边平行且两个端点都近
            dup = 0
            for x in range(len(pe)):
                for y in range(x + 1, len(pe)):
                    if ang(pe[x], pe[y]) < 0.985:
                        continue
                    d = max(min(pt_seg(pe[x][0], *pe[y]), pt_seg(pe[x][1], *pe[y])),
                            min(pt_seg(pe[y][0], *pe[x]), pt_seg(pe[y][1], *pe[x])))
                    if d <= a.near:
                        dup += 1
            c["pred_dup_pairs"] += dup
            # ② 过分割：我们的边两端都落在**某条 GT 边的内部**（不在 GT 端点附近）
            split = 0
            for p in pe:
                if any(min(math.hypot(p[0][0] - g[0][0], p[0][1] - g[0][1]),
                           math.hypot(p[0][0] - g[1][0], p[0][1] - g[1][1])) <= 5.0
                       and min(math.hypot(p[1][0] - g[0][0], p[1][1] - g[0][1]),
                               math.hypot(p[1][0] - g[1][0], p[1][1] - g[1][1])) <= 5.0
                       for g in ge):
                    c["p_both_ends_at_gt_ends"] += 1
                elif any(pt_seg(p[0], *g) <= 5.0 and pt_seg(p[1], *g) <= 5.0 for g in ge):
                    split += 1
            c["pred_split_pieces"] += split
            # ③ GT 边被我们覆盖（两端都在我们某条边内部或端点）
            cov = 0
            for g in ge:
                if any((pt_seg(g[0], *p) <= 5.0 and pt_seg(g[1], *p) <= 5.0) for p in pe):
                    cov += 1
            c["gt_covered_by_one_pred"] += cov
            # ④ 反向：我们的边落在 GT 边上（一端也行的松口径）
            loose = 0
            for p in pe:
                if any(pt_seg(p[0], *g) <= 5.0 and pt_seg(p[1], *g) <= 5.0 for g in ge):
                    loose += 1
            c["p_on_gt_seg_loose"] += loose
        tot.update(c)
        print(f"case {case:>2}: gtE {c['gt_edges']:>4} pdE {c['pred_edges']:>4} | 配对网内 "
              f"p {c['p_in']:>4} g {c['g_in']:>4} | 双描边对 {c['pred_dup_pairs']:>4} | "
              f"过分割片 {c['pred_split_pieces']:>4} | 两端对上 {c['p_both_ends_at_gt_ends']:>3} | "
              f"落在GT段上(松) {c['p_on_gt_seg_loose']:>4} | 被单条我们边盖住 {c['gt_covered_by_one_pred']:>3}")
    print("\n合计：")
    for k in ("gt_edges", "pred_edges", "p_in", "g_in", "pred_dup_pairs",
              "pred_split_pieces", "p_both_ends_at_gt_ends", "p_on_gt_seg_loose",
              "gt_covered_by_one_pred"):
        print(f"  {k:<26} {tot[k]}")
    if tot["p_in"]:
        print(f"  ⇒ 配对网内：我们的边 {tot['p_in']}、GT {tot['g_in']}（比值 {tot['p_in']/max(1,tot['g_in']):.2f}）")
        print(f"  ⇒ 双描边嫌疑 {tot['pred_dup_pairs']} 对；过分割片 {tot['pred_split_pieces']}")
        print(f"  ⇒ 我们的边落在 GT 线段上（松）{tot['p_on_gt_seg_loose']}；"
              f"GT 边被我们单条边盖住 {tot['gt_covered_by_one_pred']}")


if __name__ == "__main__":
    main()
