"""wire 通道的「完美切分」上界：把我们的线段在 **GT 边的端点**处切开后重打分。

动机（`_wirediag.py --cover` 的读数）：我们的导线**几何是对的**（GT 边平均被覆盖
89.1%、512/639 全覆），但 wire F1 只有 0.276 —— 失分主要在**边怎么切**：
GT 的 `edges` 在每个分叉/引脚处切成 2 点短边，而我们的骨架把整条走线接成一段长边，
官方 matcher 要求**两个端点都在 5px 内**，长边只能命中其中一小截。

这个脚本作弊：拿 GT 自己的端点当切刀，把我们的边切开，看 wire F1 能到多少。
若这个上界很高 ⇒ 值得去做「按分叉/引脚切边」；若不高 ⇒ 问题在几何/配网，别白干。

用法（在 Y:\\workspace\\EDA 下）：

    python -X utf8 eda-pcb\\tools\\dev\\_wireceiling.py --run work\\_fast_out_dropunk
    python -X utf8 eda-pcb\\tools\\dev\\_wireceiling.py --run <臂> --tol 4
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import official as OFF  # noqa: E402

GT_ROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def resplit(segs, cuts, tol):
    """把 segs 里每条边在 cuts（投影落在线段内部且垂距 <= tol）处切开。"""
    out = []
    for a, b in segs:
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 <= 1e-9:
            out.append((a, b))
            continue
        us = []
        for cx, cy in cuts:
            t = ((cx - ax) * dx + (cy - ay) * dy) / L2
            if t <= 1e-3 or t >= 1.0 - 1e-3:
                continue
            qx, qy = ax + t * dx, ay + t * dy
            if math.hypot(cx - qx, cy - qy) <= tol:
                us.append((t, (qx, qy)))
        if not us:
            out.append((a, b))
            continue
        us.sort()
        prev = a
        for _t, q in us:
            if math.hypot(q[0] - prev[0], q[1] - prev[1]) <= 1e-6:
                continue
            out.append((prev, q))
            prev = q
        if math.hypot(b[0] - prev[0], b[1] - prev[1]) > 1e-6:
            out.append((prev, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work\\_fast_out_dropunk")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--tol", type=float, default=4.0, help="切刀投影容差")
    ap.add_argument("--gt-cuts", action="store_true",
                    help="用 GT 边端点当切刀（作弊上界，默认开）")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    base_sum = ceil_sum = 0.0
    n = 0
    for cid in cases:
        gt_p = GT_ROOT / cid / ("%s.json" % cid)
        pred_p = ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)
        if not (gt_p.exists() and pred_p.exists()):
            print("skip %s" % cid)
            continue
        gt, pred = load(gt_p), load(pred_p)
        _conn, pairs = OFF.connectivity_f1(gt, pred, collapse_nonbox=False)
        ge, pe = OFF._edges_of(gt), OFF._edges_of(pred)
        pmap = dict(pairs)
        pred2 = copy.deepcopy(pred)
        n_before = sum(len(v) for v in pe.values())
        n_after = 0
        for gid, pid in pairs:
            cuts = [p for e in (ge.get(gid) or []) for p in e]
            segs = pe.get(pid) or []
            new = resplit(segs, cuts, args.tol)
            n_after += len(new)
            rec = (pred2.get("nets") or {}).get(pid)
            if rec is None:
                continue
            rec["edges"] = {str(i): [[p[0][0], p[0][1]], [p[1][0], p[1][1]]]
                            for i, p in enumerate(new)}
        # 没被配上的网保持原样，也要计入新总数
        for pid, segs in pe.items():
            if pid not in set(pmap.values()):
                n_after += len(segs)
        inter = OFF._edges_of(pred2)
        w0 = OFF.wire_f1(gt, pred, pairs)
        w1 = OFF.wire_f1(gt, pred2, pairs)
        base_sum += w0.f1
        ceil_sum += w1.f1
        n += 1
        print("case %-3s pred_edge %3d → 切分后 %3d | wireF1 %.4f → %.4f (tp %d→%d fp %d→%d fn %d→%d)"
              % (cid, n_before, sum(len(v) for v in inter.values()), w0.f1, w1.f1,
                 w0.tp, w1.tp, w0.fp, w1.fp, w0.fn, w1.fn))
    if n:
        print()
        print("均值：原 %.4f  →  GT 端点切分上界 %.4f（+%.4f，×0.10 = %+.2f 分）"
              % (base_sum / n, ceil_sum / n, (ceil_sum - base_sum) / n,
                 10.0 * (ceil_sum - base_sum) / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
