# -*- coding: utf-8 -*-
"""wire 通道的**离线后处理**候选：不改管线，只改 pred 的 `nets[*]["edges"]` 首尾两点。

背景（`_wirediag.py` / `_wireceiling.py` 的读数）：
  * GT 边平均被我们覆盖 89.1%（几何是对的），但 wire F1 只有 0.28；
  * 拿 **GT 自己的端点**当切刀切开我们的边 ⇒ 0.4156（+1.35 分）⇒ 潜力在「端点落在哪」；
  * `STAGE1_SPLIT_TOL` 1.5→8 端到端只 +0.06 ⇒ **只切不动端点没用**：我们的分叉点本身
    就偏了 5~15px（骨架在拐角处走斜、端点停短），所以切口虽然对、端点仍在 5px 外。

本脚本试的直接就是「把端点搬到该去的地方」：
  1) `snap`  —— 端点 p 落在**同网另一条边的内部**（垂距 ≤ D）⇒ p 搬到投影点 q；
  2) `cut`   —— 搬完之后，别的边的端点落在本条内部 ⇒ 在本条那里切开（现在切口是精确的）；
  3) `dedup` —— 被另一条边几乎完全覆盖的碎边丢掉（切完留下的残桩）。

用法：
    python -X utf8 eda-pcb\\tools\\dev\\_wirepost.py                    # 全部候选
    python -X utf8 eda-pcb\\tools\\dev\\_wirepost.py --modes snap,cut
    python -X utf8 eda-pcb\\tools\\dev\\_wirepost.py --d 8 --cases 1,5
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
EPS = 1e-6


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def foot(p, a, b):
    """p 到线段 ab 的 (垂距, 参数 t, 投影点 q)。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return math.hypot(p[0] - a[0], p[1] - a[1]), 0.0, a
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2
    tc = min(1.0, max(0.0, t))
    q = (a[0] + tc * dx, a[1] + tc * dy)
    return math.hypot(p[0] - q[0], p[1] - q[1]), t, q


def segs_of(rec):
    """pred/gt 网的边 ⇒ [(a, b)]，只取首尾两点（官方只看这两点）。"""
    out = []
    for e in (rec.get("edges") or {}).values():
        if len(e) >= 2:
            out.append(((float(e[0]["x"]), float(e[0]["y"])),
                        (float(e[-1]["x"]), float(e[-1]["y"]))))
    return out


def do_snap(segs, D, rounds=2):
    """端点搬到同网另一条边的内部（垂距 ≤ D）。"""
    segs = [list(s) for s in segs]
    for _ in range(rounds):
        moved = 0
        for i, si in enumerate(segs):
            for k in (0, 1):
                p = si[k]
                best = None
                for j, sj in enumerate(segs):
                    if j == i:
                        continue
                    d, t, q = foot(p, sj[0], sj[1])
                    if d <= EPS or d > D:
                        continue
                    if not (1e-3 < t < 1.0 - 1e-3):
                        continue            # 端点对端点不归这里管
                    if best is None or d < best[0]:
                        best = (d, q)
                if best is not None:
                    si[k] = best[1]
                    moved += 1
        if not moved:
            break
    return [(tuple(s[0]), tuple(s[1])) for s in segs]


def do_cut(segs, tol=1.5):
    """别的边的端点落在本条内部（垂距 ≤ tol）⇒ 在本条那里切开。"""
    ends = [(s[0], s[1], i) for i, s in enumerate(segs)]
    out = []
    for i, (a, b) in enumerate(segs):
        us = []
        for p, _q, j in ends:
            if j == i:
                continue
            d, t, q = foot(p, a, b)
            if d <= tol and 1e-3 < t < 1.0 - 1e-3:
                us.append((t, q))
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


def cover_frac(inner, outer, step=2.0):
    """inner 被单条 outer 覆盖的比例（沿线采样，容差 2px）。"""
    a, b = inner
    oa, ob = outer
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L <= EPS:
        return 0.0
    n = max(2, int(L / step))
    hit = 0
    for i in range(n + 1):
        t = i / n
        p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        d, _tt, _q = foot(p, oa, ob)
        if d <= 2.0:
            hit += 1
    return hit / (n + 1)


def do_dedup(segs, frac=0.9):
    """被另一条**单条**边覆盖 ≥ frac 的边丢掉（残桩），并去掉零长边。"""
    keep = []
    for i, s in enumerate(segs):
        L = math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1])
        if L <= 1.0:
            continue
        drop = False
        for j, o in enumerate(segs):
            if j == i:
                continue
            Lo = math.hypot(o[1][0] - o[0][0], o[1][1] - o[0][1])
            if Lo <= L + 1.0:
                continue                    # 只让「更长的」吃掉短的
            if cover_frac(s, o) >= frac:
                drop = True
                break
        if not drop:
            keep.append(s)
    return keep


def apply_mode(rec, mode, D, cut_tol):
    segs = segs_of(rec)
    if not segs:
        return 0
    if mode == "snap":
        segs = do_snap(segs, D)
    elif mode == "cut":
        segs = do_cut(segs, cut_tol)
    elif mode == "snap,cut":
        segs = do_cut(do_snap(segs, D), cut_tol)
    elif mode == "snap,cut,dedup":
        segs = do_dedup(do_cut(do_snap(segs, D), cut_tol))
    elif mode == "snap,dedup":
        segs = do_dedup(do_snap(segs, D))
    rec["edges"] = {str(i): [[round(p[0][0], 2), round(p[0][1], 2)],
                             [round(p[1][0], 2), round(p[1][1], 2)]]
                    for i, p in enumerate(segs)}
    return len(segs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work\\_fast_out_dropunk")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--modes", default="snap,cut,snap+cut,snap+cut+dedup,snap+dedup")
    ap.add_argument("--d", type=float, default=8.0, help="端点吸附距离上限")
    ap.add_argument("--cut-tol", type=float, default=1.5)
    ap.add_argument("--per-case", action="store_true")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    modes = [m.replace("+", ",") for m in args.modes.split(",") if m.strip()]
    # "--modes a,b" 会把逗号当分隔符，这里再手工拼回常见三项
    if args.modes.count("+") and len(modes) > 3:
        modes = ["snap", "cut", "snap,cut", "snap,cut,dedup", "snap,dedup"]

    docs = []
    for cid in cases:
        gt_p = GT_ROOT / cid / ("%s.json" % cid)
        pred_p = ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)
        if not (gt_p.exists() and pred_p.exists()):
            print("skip %s" % cid)
            continue
        docs.append((cid, load(gt_p), load(pred_p)))

    def score(preds):
        tot = {"w": 0.0, "tp": 0, "fp": 0, "fn": 0}
        per = []
        for cid, gt, _p in docs:
            pr = preds[cid]
            _conn, pairs = OFF.connectivity_f1(gt, pr, collapse_nonbox=False)
            w = OFF.wire_f1(gt, pr, pairs)
            tot["w"] += w.f1
            tot["tp"] += w.tp
            tot["fp"] += w.fp
            tot["fn"] += w.fn
            per.append((cid, w.f1, w.tp, w.fp, w.fn))
        n = max(1, len(docs))
        return tot["w"] / n, tot["tp"], tot["fp"], tot["fn"], per

    base = {cid: copy.deepcopy(p) for cid, _g, p in docs}
    f0, tp0, fp0, fn0, _per = score(base)
    print("基线（%d 张）：wireF1 %.4f   tp %d  fp %d  fn %d" % (len(docs), f0, tp0, fp0, fn0))
    for mode in modes:
        preds = {cid: copy.deepcopy(p) for cid, _g, p in docs}
        ne = 0
        for cid, _g, _p in docs:
            rec_nets = preds[cid].get("nets") or {}
            for rec in rec_nets.values():
                ne += apply_mode(rec, mode, args.d, args.cut_tol)
        f1, tp, fp, fn, per = score(preds)
        print("%-16s wireF1 %.4f  (%+.4f)  tp %d (%+d)  fp %d (%+d)  fn %d (%+d)  边 %d"
              % (mode, f1, f1 - f0, tp, tp - tp0, fp, fp - fp0, fn, fn - fn0, ne))
        if args.per_case:
            print("    " + "  ".join("%s:%.3f" % (c, v) for c, v, _a, _b, _c in per))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
