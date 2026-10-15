# -*- coding: utf-8 -*-
"""神谕目标可及性普查：`_endoracle.py --mode end` 把 211 条坏边的一端挪到 GT 端点，
这些「目标点」在我们自己的产物里有依可循吗？
分类：落在某条我方边的内部（⇒ 该切一刀）／贴近某条我方边的端点（⇒ 挪端点）／
贴近某个引脚锚点（⇒ 吸附到引脚）／无所依。
另外统计目标点相对我方边的**横向偏移**与**位移**，供设计规则。
用法：python -X utf8 work/_pinorac.py --base <dir> --orac <dir> --cases 1,...,10
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # archived copy: work/ -> tools/diag/
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import metrics as M  # noqa: E402


def _edges(doc):
    out = []
    for nid, net in (doc.get("nets") or {}).items():
        for k, v in (net.get("edges") or {}).items():
            e = M._edge_points(v)
            if e:
                out.append((nid, k, e[0], e[1]))
    return out


def _pins(doc):
    pts = []
    for _cname, pd in ((doc.get("pins") or {})).items():
        if not isinstance(pd, dict):
            continue
        for _pn, pp in pd.items():
            if isinstance(pp, dict):
                q = pp.get("point") or pp
                if isinstance(q, dict) and "x" in q and "y" in q:
                    pts.append((float(q["x"]), float(q["y"])))
    for comp in (doc.get("components") or {}).values():
        for pp in (comp.get("pins") or []):
            q = pp.get("point") if isinstance(pp, dict) else None
            if isinstance(q, dict) and "x" in q:
                pts.append((float(q["x"]), float(q["y"])))
    return pts


def _seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    tc = min(1.0, max(0.0, t))
    cx, cy = ax + tc * dx, ay + tc * dy
    return math.hypot(px - cx, py - cy), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "work" / "_fast_out_finalrd8"))
    ap.add_argument("--orac", default=str(ROOT / "work" / "_fast_out_endorc_end2"))
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--interior", type=float, default=1.5)
    ap.add_argument("--endnear", type=float, default=3.0)
    ap.add_argument("--pinnear", type=float, default=10.0)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    want = set(a.cases.split(","))
    gt_by_cid = {}
    from schematic_parser.data import DEFAULT_DATA_ROOT, discover
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
        gt_by_cid[cid] = json.loads(Path(c.json_path).read_text(encoding="utf-8"))

    tally = {"interior": 0, "end": 0, "pin": 0, "none": 0}
    disp, lat, per_case, same_net, xnet = [], [], [], 0, 0
    node_hit = cross_hit = no_ev = 0
    ev = []
    rows = []
    for cid in sorted(gt_by_cid):
        if cid not in want:
            continue
        fa = Path(a.base) / "final_results" / (cid + "_Pred.json")
        fb = Path(a.orac) / "final_results" / (cid + "_Pred.json")
        if not (fa.exists() and fb.exists()):
            continue
        A, B = json.loads(fa.read_text(encoding="utf-8")), json.loads(fb.read_text(encoding="utf-8"))
        ea, eb = _edges(A), _edges(B)
        pins = _pins(A)
        moved = 0
        this = {"interior": 0, "end": 0, "pin": 0, "none": 0}
        for (nid, k, a0, a1), (_n2, _k2, b0, b1) in zip(ea, eb):
            tgt = None
            mov = 0.0
            if math.hypot(b0[0] - a0[0], b0[1] - a0[1]) > 1e-6:
                tgt, keep, mov = b0, a1, math.hypot(b0[0] - a0[0], b0[1] - a0[1])
            elif math.hypot(b1[0] - a1[0], b1[1] - a1[1]) > 1e-6:
                tgt, keep, mov = b1, a0, math.hypot(b1[0] - a1[0], b1[1] - a1[1])
            if tgt is None:
                continue
            moved += 1
            d = mov
            disp.append(mov)
            best = None                      # (横向距离, n2, k2, t, c0, c1)
            for (n2, k2, c0, c1) in ea:
                dd, t = _seg_dist(tgt, c0, c1)
                if best is None or dd < best[0]:
                    best = (dd, n2, k2, t, c0, c1)
            latv = best[0]
            lat.append(latv)
            # 分类
            kind = "none"
            dd, n2, k2, t, c0, c1 = best
            L = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
            if dd <= a.interior and L > 1e-6 and 2.0 / L < t < 1.0 - 2.0 / L:
                kind = "interior"
                if n2 == nid:
                    same_net += 1
                else:
                    xnet += 1
            if kind == "none":
                for (n2, k2, c0, c1) in ea:
                    if min(math.hypot(tgt[0] - c0[0], tgt[1] - c0[1]),
                           math.hypot(tgt[0] - c1[0], tgt[1] - c1[1])) <= a.endnear:
                        kind = "end"
                        break
            if kind == "none":
                for p in pins:
                    if math.hypot(tgt[0] - p[0], tgt[1] - p[1]) <= a.pinnear:
                        kind = "pin"
                        break
            tally[kind] += 1
            this[kind] += 1
            # --- 目标点上有没有我们自己的图节点/交越（可实现的依据）---
            nh = min([math.hypot(tgt[0] - p0[0], tgt[1] - p0[1]) for (_a, _b, p0, p1) in ea
                      if (_a, _b) != (nid, k)] +
                     [math.hypot(tgt[0] - p1[0], tgt[1] - p1[1]) for (_a, _b, p0, p1) in ea
                      if (_a, _b) != (nid, k)] or [1e9])
            ch = 1e9
            for (_a, _b, p0, p1) in ea:
                if (_a, _b) == (nid, k):
                    continue
                dd2, t2 = _seg_dist(tgt, p0, p1)
                if 1e-3 < t2 < 1.0 - 1e-3:
                    ch = min(ch, dd2)
            depth = min(math.hypot(tgt[0] - a0[0], tgt[1] - a0[1]),
                        math.hypot(tgt[0] - a1[0], tgt[1] - a1[1]))
            ev.append((mov, nh, ch, depth, kind))
            if nh <= 2.0:
                node_hit += 1
            elif ch <= 2.0:
                cross_hit += 1
            else:
                no_ev += 1
            if len(rows) < a.show or kind == "none":
                rows.append((cid, nid, k, keep, tgt, kind, latv, d))
        per_case.append((cid, moved, this))

    print("位移像素：n=%d  p50=%.1f  p90=%.1f  max=%.1f" % (
        len(disp), st.median(disp), sorted(disp)[int(0.9 * (len(disp) - 1))], max(disp)))
    print("目标点到最近我方边的横向距离：p50=%.1f  p90=%.1f" % (
        st.median(lat), sorted(lat)[int(0.9 * (len(lat) - 1))]))
    print("归属：", tally, "（interior 里同网 %d / 跨网 %d）" % (same_net, xnet))
    if ev:
        mv = [x[0] for x in ev]
        dp = [x[3] for x in ev]
        print("实际位移（被挪端走了多远）：p50=%.1f  p90=%.1f  max=%.1f   ≤3px:%d  ≤8px:%d  ≤15px:%d  >15px:%d" % (
            st.median(mv), sorted(mv)[int(0.9 * (len(mv) - 1))], max(mv),
            sum(1 for v in mv if v <= 3), sum(1 for v in mv if v <= 8),
            sum(1 for v in mv if v <= 15), sum(1 for v in mv if v > 15)))
        print("目标点到我方该边**本端**的深度：p50=%.1f  p90=%.1f" % (
            st.median(dp), sorted(dp)[int(0.9 * (len(dp) - 1))]))
        print("目标点处我方图里的依据：别的边端点 ≤2px %d ｜ 别的边跨过 ≤2px %d ｜ 都没有 %d"
              % (node_hit, cross_hit, no_ev))
    print("逐案：")
    for cid, moved, this in per_case:
        print("  case %-3s 目标 %2d 条  interior %2d | 我方端点 %2d | 引脚 %2d | 无 %2d"
              % (cid, moved, this["interior"], this["end"], this["pin"], this["none"]))
    print("样例（'无' 的全列）：")
    for cid, nid, k, keep, tgt, kind, latv, d in rows[:a.show + 12]:
        print("  %-3s net %-8s %-9s %-8s 移动 %5.1f  (%.0f,%.0f)→(%.0f,%.0f) 横距 %.1f"
              % (cid, nid, k, kind, d, keep[0], keep[1], tgt[0], tgt[1], latv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
