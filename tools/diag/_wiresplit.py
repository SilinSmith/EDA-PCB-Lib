# -*- coding: utf-8 -*-
"""导线段「覆盖但没配上」的形态分解（只读）。

`work/_endstruct.py` 的线段层给出：639 条 GT edge 里 394 配上、162 覆盖但没配上、83 没覆盖。
本探针把这 162 条按**形态**再分类，判断有没有可执行的规则：

  overspan   我方这条段**跨过了**这条 GT 段：GT 两个端点都落在我方段的内部
             ⇒ 需要在 GT 端点处**把我方段切开**（该点就是图像上的 T 接/结点）
  underspan  我方这条段**短了**：我方两个端点都落在 GT 段内部
             ⇒ 需要沿同一方向**延长**到 GT 端点（结构 = 脚点/框边/T 接）
  lateral    两端**横向对不齐**（不是长短问题），或形态不属于上面两类
  no_pred    找不到能覆盖它的我方段（理论上不该出现，出现说明覆盖判据用了 2px、配对用了别的段）

同时统计：这些 GT 端点在图像上的结构归属（沿用 `_endstruct` 的分类），
用来判断「切断点/延端点」是否本来就有可观测锚点。
"""
from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb\tools\eval")
from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover   # noqa: E402
from schematic_parser.metrics import _edge_points, normalise_doc           # noqa: E402


def poly(edge, h):
    pts = _edge_points(edge) or []
    out = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((float(p[0]), h - float(p[1])))
    return out


def edges_of(gt_doc, h):
    doc = normalise_doc(gt_doc)
    out = []
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pl = poly(edge, h)
            if len(pl) >= 2:
                out.append({"net": nkey, "edge": ekey, "pts": pl})
    return out


def dist_point_seg(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 1e-12:
        return float(np.hypot(*(p - a)))
    t = float((p - a) @ ab) / L2
    tc = min(1.0, max(0.0, t))
    proj = a + ab * tc
    return float(np.hypot(*(proj - p))), t


def point_to_polyline(p, pts):
    """p 到折线 pts（点序列）的最短距离。"""
    best = float("inf")
    for i in range(len(pts) - 1):
        d, _ = dist_point_seg(p, pts[i], pts[i + 1])
        if d < best:
            best = d
    return best


def param_of(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 1e-12:
        return 0.5
    return float((p - a) @ ab) / L2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--min-frac", type=float, default=0.9,
                    help="覆盖判据：GT 段上采样点落在 pred 折线 2px 内的比例")
    a = ap.parse_args(argv)

    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")

    stat = collections.Counter()
    per_case = collections.OrderedDict()
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        with open(case.png_path, "rb"):
            pass
        from PIL import Image  # noqa: E402
        h = Image.open(case.png_path).size[1]
        gt_doc = load_json(case.json_path)
        pred_doc = load_json(fp)
        g = edges_of(gt_doc, h)
        p = edges_of(pred_doc, h)
        c = collections.Counter()
        for e in g:
            gpl = np.asarray(e["pts"], dtype=float)
            a0, b0 = gpl[0], gpl[1]
            # 采样 GT 段
            samples = []
            for i in range(len(gpl) - 1):
                seg = gpl[i + 1] - gpl[i]
                L = float(np.hypot(*seg))
                if L <= 1e-9:
                    continue
                k = max(2, int(L / 3.0) + 1)
                for t in np.linspace(0.0, 1.0, k):
                    samples.append(gpl[i] + seg * t)
            # 已配上？（unordered、头两点）
            matched = False
            best = None
            for q in p:
                qpl = np.asarray(q["pts"], dtype=float)
                if len(qpl) < 2:
                    continue
                for (u, v) in ((qpl[0], qpl[1]), (qpl[1], qpl[0])):
                    if (max(abs(a0[0] - u[0]), abs(a0[1] - u[1])) <= a.tol
                            and max(abs(b0[0] - v[0]), abs(b0[1] - v[1])) <= a.tol):
                        matched = True
                        break
                if matched:
                    break
                # 覆盖分 + 形态
                hit = sum(1 for s in samples
                          if point_to_polyline(s, qpl) <= 2.0)
                frac = hit / float(len(samples)) if samples else 0.0
                if frac >= a.min_frac:
                    # 用 pred 段的**首段直线**判断长短（交付边都是 2 点）
                    A, B = qpl[0], qpl[1]
                    ta0 = param_of(a0, A, B)
                    tb0 = param_of(b0, A, B)
                    tap = param_of(A, a0, b0)
                    tbp = param_of(B, a0, b0)
                    in_a = 0.05 < ta0 < 0.95
                    in_b = 0.05 < tb0 < 0.95
                    p_inA = 0.05 < tap < 0.95
                    p_inB = 0.05 < tbp < 0.95
                    if in_a and in_b:
                        kind = "overspan"          # 我方跨过整条 GT 段 ⇒ 要切开
                    elif p_inA and p_inB:
                        kind = "underspan"         # 我方整段落在 GT 段内 ⇒ 要延长
                    elif in_a != in_b:
                        kind = "cut1"              # 只有一端被跨过 ⇒ 那一端要切
                    elif p_inA != p_inB:
                        kind = "ext1"              # 只有一端悬空 ⇒ 那一端要延
                    else:
                        kind = "lateral"           # 两端都错位（真几何/标注噪声）
                    score = frac
                    if best is None or score > best[0]:
                        best = (score, kind, float(np.hypot(*(A - a0))),
                                float(np.hypot(*(B - b0))))
            if matched:
                c["matched"] += 1
                stat["matched"] += 1
            elif best is None:
                c["no_pred"] += 1
                stat["no_pred"] += 1
            else:
                c[best[1]] += 1
                stat[best[1]] += 1
        per_case[cid] = c
        print("case %-3s GT edge %3d  配上 %3d  跨过 %2d  跨一端 %2d  短了 %2d  悬一端 %2d  "
              "横向/其它 %2d  无覆盖 %2d"
              % (cid, len(g), c["matched"], c["overspan"], c["cut1"], c["underspan"],
                 c["ext1"], c["lateral"], c["no_pred"]))
    tot = sum(stat.values())
    if tot:
        print("\n-- 合计 %d 条 GT edge --" % tot)
        for k in ("matched", "overspan", "cut1", "underspan", "ext1", "lateral", "no_pred"):
            print("   %-10s %4d  %5.1f%%" % (k, stat[k], 100.0 * stat[k] / tot))
        cut = stat["overspan"] + stat["cut1"]
        ext = stat["underspan"] + stat["ext1"]
        print("   ⇒ **该切段 %d（%.1f%%）** + **该延长 %d（%.1f%%）** ⇒ 端点规则能覆盖 %.1f%%；"
              "横向 %d（%.1f%%）是真几何；无覆盖 %d（%.1f%%）是缺线"
              % (cut, 100.0 * cut / tot, ext, 100.0 * ext / tot,
                 100.0 * (cut + ext) / tot, stat["lateral"],
                 100.0 * stat["lateral"] / tot, stat["no_pred"],
                 100.0 * stat["no_pred"] / tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
