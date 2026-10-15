# -*- coding: utf-8 -*-
"""结点圆点（junction dot）能不能当「真结点」锚点？（只读）

为什么问这个：`tools/diag/_lineaudit.py` 刚量出官方 10 张 net_line_f1 = 0.5796，
但把**未配上**的我方段端点瞬移到 20px 内最近的 GT 端点，就能到 **0.9224（+3.43 分加权）**；
5px 内瞬移只 +0.0036 ⇒ 亏损几乎全是**端点位置偏 5~20px**，不是边形状。
历史结论同向（`_wfbullets.py` 里 `_nlpair/_nledge/_nldup`）：每条边形状都对，
**骨架分叉点比真结点短 8~13px**，而容差只有 5px。

本探针只回答一件事：**GT 的 T 接端点是否落在图像上画出来的实心结点圆点上，圆点是否可测。**
可测的话，就能用「圆点质心」当结点锚点去校正端点 —— 这是**图像侧可判定**的信号，
不是 GT 神谕（对比：切点后处理已被否证，见 RESULTS §5.120）。

三层证据：
  1. 对角 4 探针判据（复刻 `stage1/labels.py:498 _has_dot`）量 GT 端点命中率，
     对照组 = GT 导线内部的采样点（那里应该没有圆点）；
  2. 墨掩膜距离变换找「粗块」（走线 ~2px 宽，圆点直径 9~10px ⇒ dt≥3 基本只有圆点），
     报检出圆点数 / 半径，并量 GT 端点到最近圆点质心的距离分布；
  3. 把 GT 端点分 T 接 / 脚点 / 其它三类，分别给命中率与距离。

用法：
    python -X utf8 tools\\diag\\_dotprobe.py --cases 1,2,3,4,5
"""
from __future__ import print_function

import argparse
import collections
import math
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import pcore, prun                                         # noqa: E402
from schematic_parser.metrics import _edge_points, normalise_doc         # noqa: E402


def gt_wire_edges(gt_doc):
    doc = normalise_doc(gt_doc)
    out = []
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge)
            if pts and len(pts) >= 2:
                out.append((nkey, ekey, pts))
    return out


def flip(pts, h):
    return [(float(p[0]), float(h) - float(p[1])) for p in pts]


def d_point_seg(p, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / L2))
    return math.hypot(p[0] - (ax + t * vx), p[1] - (ay + t * vy))


def has_dot(ink, p, r=3.0, need=3):
    """复刻 stage1/labels.py:498 `_has_dot`：对角 45/135/225/315 处有墨就算圆点。"""
    h, w = ink.shape[:2]
    hit = 0
    for k in range(4):
        a = math.pi * (0.25 + 0.5 * k)
        x = int(round(p[0] + r * math.cos(a)))
        y = int(round(p[1] + r * math.sin(a)))
        if 0 <= x < w and 0 <= y < h and ink[y, x]:
            hit += 1
    return hit >= need


def dots_of(ink, min_dt=3.0):
    dt = ndimage.distance_transform_edt(ink)
    lab, n = ndimage.label(dt >= min_dt)
    cents = ndimage.center_of_mass(dt >= min_dt, lab, range(1, n + 1))
    out = []
    for i, (cy, cx) in enumerate(cents):
        rr = float(dt[lab == (i + 1)].max())
        out.append((float(cx), float(cy), rr))
    return dt, out


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--tol", type=float, default=5.0, help="T 接判据：离别的导线 <= tol")
    ap.add_argument("--min-dt", type=float, default=3.0)
    ap.add_argument("--ctrl-per-edge", type=int, default=5)
    a = ap.parse_args(argv)

    tot = collections.Counter()
    hit_stat = collections.Counter()
    dist_dot = collections.defaultdict(list)      # 类 -> [到最近圆点质心的距离]
    ctrl_hits = [0, 0]
    print("case   GT端点  T接  脚点  其它 | 对角命中(T接/脚点) | 圆点数(半径中位) | "
          "T接到最近圆点 p50/p90/<=3px | 对照命中")
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = pcore.find_case(cid)
        if case is None:
            print("  ! 找不到 case %s" % cid)
            continue
        inp = prun.inputs_of(case)
        h = inp.h
        res = prun.skeleton_of(inp, "adaptive")
        ink = res.ink
        if ink is None:
            print("  ! case %s 没有 ink" % cid)
            continue
        ink = np.asarray(ink).astype(bool)
        _dt, dots = dots_of(ink, a.min_dt)

        edges = gt_wire_edges(inp.gt_doc)
        flat = [flip(pts, h) for _n, _e, pts in edges]
        pin_pts = np.array([[p.x, p.y] for p in inp.truth.pins.values()], dtype=float) \
            if inp.truth.pins else np.zeros((0, 2))
        cnt = collections.Counter()
        hits = collections.Counter()
        ctrl = [0, 0]
        for idx, (_n, _e, pts) in enumerate(edges):
            pl = flat[idx]
            for end_i in (0, -1):
                p = pl[end_i]
                other = []
                for jdx, q in enumerate(flat):
                    if jdx == idx:
                        continue
                    for i in range(len(q) - 1):
                        other.append((q[i], q[i + 1]))
                d_wire = min([d_point_seg(p, s0, s1) for s0, s1 in other]
                             or [float("inf")])
                d_pin = (float(np.min(np.hypot(pin_pts[:, 0] - p[0],
                                               pin_pts[:, 1] - p[1])))
                         if len(pin_pts) else float("inf"))
                if d_wire <= a.tol:
                    kind = "T"
                elif d_pin <= a.tol:
                    kind = "pin"
                else:
                    kind = "other"
                cnt[kind] += 1
                if has_dot(ink, p):
                    hits[kind] += 1
                if dots:
                    dd = min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in dots)
                    dist_dot[kind].append(dd)
            # 对照组：段内部采样
            for k in range(1, a.ctrl_per_edge + 1):
                t = k / float(a.ctrl_per_edge + 1)
                q = (pl[0][0] + (pl[1][0] - pl[0][0]) * t,
                     pl[0][1] + (pl[1][1] - pl[0][1]) * t)
                ctrl[1] += 1
                if has_dot(ink, q):
                    ctrl[0] += 1
        ctrl_hits[0] += ctrl[0]
        ctrl_hits[1] += ctrl[1]
        for k in ("T", "pin", "other"):
            tot[k] += cnt[k]
            hit_stat[k] += hits[k]
        rad = [d[2] for d in dots]
        dT = dist_dot["T"]
        print("case %-3s %5d %5d %5d %5d | %3d/%-3d %3d/%-3d | %4d (r=%.1f) | "
              "%5.1f/%5.1f/%4.1f%% | %d/%d"
              % (cid, sum(cnt.values()), cnt["T"], cnt["pin"], cnt["other"],
                 hits["T"], cnt["T"], hits["pin"], cnt["pin"],
                 len(dots), float(np.median(rad)) if rad else float("nan"),
                 pct(dT, 50), pct(dT, 90),
                 100.0 * sum(1 for x in dT if x <= 3.0) / max(1, len(dT)),
                 ctrl[0], ctrl[1]))

    print("\n== 合计 ==")
    n_end = sum(tot.values())
    print("  GT 端点 %d：T 接 %d（%.1f%%）/ 脚点 %d（%.1f%%）/ 其它 %d（%.1f%%）"
          % (n_end, tot["T"], 100.0 * tot["T"] / max(1, n_end),
             tot["pin"], 100.0 * tot["pin"] / max(1, n_end),
             tot["other"], 100.0 * tot["other"] / max(1, n_end)))
    for k in ("T", "pin", "other"):
        print("  %-5s 对角 4 探针命中 %3d/%3d = %5.1f%%   到最近圆点质心 p50 %5.1f / "
              "p90 %5.1f / ≤3px %5.1f%%"
              % (k, hit_stat[k], tot[k],
                 100.0 * hit_stat[k] / max(1, tot[k]),
                 pct(dist_dot[k], 50), pct(dist_dot[k], 90),
                 100.0 * sum(1 for x in dist_dot[k] if x <= 3.0)
                 / max(1, len(dist_dot[k]))))
    print("  对照（GT 导线内部采样点）对角命中 %d/%d = %.1f%%"
          % (ctrl_hits[0], ctrl_hits[1], 100.0 * ctrl_hits[0] / max(1, ctrl_hits[1])))
    print("  ⇒ 判据：T 接命中率远高于对照 ⇒ 圆点可测、可当结点锚点；若两者接近 ⇒ 不可分。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
