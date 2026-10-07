# -*- coding: utf-8 -*-
"""自由端的来源分解 —— 决定导线图该怎么修。

`all13` 上一张图 324 条段却并出约 211 个连通分量、422 个自由端，于是
"自由端" 完全不等价于 "走线的端"（网络标签定位因此失败）。在动手修之前必须先知道
这些假自由端是**哪一类**造成的，因为三类的修法完全不同：

  A. **断口**：两个本该连在一起的分量，端点相距只有几个像素。修法 = 桥接。
  B. **毛刺**：骨架在拐角/笔画末端分出来的短线，或者射线跑到一半停下。修法 = 剪枝。
  C. **真端**：走线本身就在这里结束。这类要留着（标签就在这儿）。

本脚本用「该自由端到**别的分量**的最近距离」把三者分开，并且用 GT 交叉验证：
一个自由端是不是真的走线端，可以拿它到最近 GT 走线**端点** / **整条走线**的距离来判。

用法：
    python Y:\workspace\EDA\work\_free_end_diag.py --cases 30
"""
import argparse
import json
import os
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import numpy as np

for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1.labels import _as_segs, wire_groups, free_ends  # noqa: E402
from stage1_conn_eval import _dist_pts_to_segs  # noqa: E402

MATCH_TOL = 6.0


def _xy(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def gt_ends_and_segs(doc, H):
    ends, segs = [], []
    for net in (doc.get("nets") or {}).values():
        for e in (net.get("edges") or {}).values():
            if len(e) < 2:
                continue
            x1, y1 = _xy(e[0])
            x2, y2 = _xy(e[-1])
            ends.append((x1, H - y1))
            ends.append((x2, H - y2))
            segs.append((x1, H - y1, x2, H - y2))
    return (np.asarray(ends, np.float64).reshape(-1, 2),
            np.asarray(segs, np.float64).reshape(-1, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=r"Y:\workspace\EDA\work\stage1_all13")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--skeleton-only", action="store_true", default=True)
    a = ap.parse_args()

    cases = sorted(discover(a.data_root), key=lambda c: c.case_id)
    idx = list(range(0, len(cases), max(1, len(cases) // a.cases)))[:a.cases]

    buckets = [(0.0, 2.0, "<=2px 断口"), (2.0, 5.0, "2~5px 断口"),
               (5.0, 12.0, "5~12px 近邻"), (12.0, 1e9, ">12px 远")]
    agg = {b[2]: dict(n=0, on_gt_end=0, on_gt_seg=0, near_comp=0, near_text=0,
                      grp_len=[], nseg=[]) for b in buckets}
    n_free = n_comp = n_seg = n_case = 0
    grp_hist = {}

    for ci in idx:
        case = cases[ci]
        sp = os.path.join(a.stage1, case.case_id + ".stage1.json")
        if not os.path.exists(sp):
            continue
        sd = json.load(open(sp, encoding="utf-8"))
        H = float(sd["image"]["h"])
        segs = _as_segs(sd["wire_detail"]["segments"])
        n_ray = int((sd["wire_detail"] or {}).get("n_rays") or 0)
        if a.skeleton_only and n_ray:
            segs = segs[:len(segs) - n_ray]
        if not len(segs):
            continue
        comps = sd.get("components") or []
        texts = sd.get("texts") or []
        bounds = [(float((c.get("bbox_img") or [0, 0, 0, 0])[0]),
                   float((c.get("bbox_img") or [0, 0, 0, 0])[1]),
                   float((c.get("bbox_img") or [0, 0, 0, 0])[2]),
                   float((c.get("bbox_img") or [0, 0, 0, 0])[3])) for c in comps]
        tbound = [(float((t.get("bbox_img") or [0, 0, 0, 0])[0]),
                   float((t.get("bbox_img") or [0, 0, 0, 0])[1]),
                   float((t.get("bbox_img") or [0, 0, 0, 0])[2]),
                   float((t.get("bbox_img") or [0, 0, 0, 0])[3])) for t in texts]

        g = wire_groups(segs)
        fe = free_ends(segs, g)
        glen = {}
        slen = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        gcnt = {}
        for i, gg in enumerate(g):
            glen[int(gg)] = glen.get(int(gg), 0.0) + float(slen[i])
            gcnt[int(gg)] = gcnt.get(int(gg), 0) + 1
        ncomp = len(glen)
        grp_hist[ncomp] = grp_hist.get(ncomp, 0) + 1

        gdoc = json.load(open(case.json_path, encoding="utf-8"))
        GE, GS = gt_ends_and_segs(gdoc, H)
        gsegs = GS if len(GS) else None

        fe_xy = np.asarray([(t[0], t[1]) for t in fe], np.float64).reshape(-1, 2)
        fe_grp = [t[2] for t in fe]
        if not len(fe_xy):
            continue
        # 到「别的分量」的最近距离
        D = _dist_pts_to_segs(fe_xy, segs)
        for i in range(len(fe_xy)):
            D[i, np.nonzero(np.asarray(g) == fe_grp[i])[0]] = np.inf
        gap = D.min(axis=1) if D.size else np.full(len(fe_xy), 1e9)
        dgt_e = (np.hypot(fe_xy[:, 0][:, None] - GE[:, 0][None, :],
                          fe_xy[:, 1][:, None] - GE[:, 1][None, :]).min(axis=1)
                 if len(GE) else np.full(len(fe_xy), 1e9))
        dgt_s = (_dist_pts_to_segs(fe_xy, gsegs).min(axis=1)
                 if gsegs is not None and len(gsegs) else np.full(len(fe_xy), 1e9))

        for i in range(len(fe_xy)):
            x, y = fe_xy[i]
            b = None
            for lo, hi, name in buckets:
                if lo <= gap[i] < hi:
                    b = agg[name]
                    break
            if b is None:
                continue
            b["n"] += 1
            if dgt_e[i] <= MATCH_TOL:
                b["on_gt_end"] += 1
            if dgt_s[i] <= 2.0:
                b["on_gt_seg"] += 1
            for cb in bounds:
                if cb[0] - 6 <= x <= cb[2] + 6 and cb[1] - 6 <= y <= cb[3] + 6:
                    b["near_comp"] += 1
                    break
            for tb in tbound:
                if tb[0] - 6 <= x <= tb[2] + 6 and tb[1] - 6 <= y <= tb[3] + 6:
                    b["near_text"] += 1
                    break
            b["grp_len"].append(glen.get(fe_grp[i], 0.0))
            b["nseg"].append(gcnt.get(fe_grp[i], 0))
        n_free += len(fe_xy)
        n_comp += ncomp
        n_seg += len(segs)
        n_case += 1

    print("=== 自由端来源分解（{} 张，{}）===".format(
        n_case, "只用骨架段" if a.skeleton_only else "骨架+射线"))
    print("段 {}  连通分量 {}  自由端 {}  ⇒ 每张 段 {:.1f} / 分量 {:.1f} / 自由端 {:.1f}".format(
        n_seg, n_comp, n_free, n_seg / n_case, n_comp / n_case, n_free / n_case))
    print("\n{:<14} {:>6} {:>7} {:>11} {:>11} {:>9} {:>9} {:>10} {:>6}".format(
        "到别分量距离", "条数", "占比", "落在GT端点", "落在GT走线", "贴元件", "贴文字",
        "分量均长", "段数"))
    for lo, hi, name in buckets:
        b = agg[name]
        n = b["n"]
        if not n:
            continue
        print("{:<14} {:>6} {:>6.1f}% {:>10.1f}% {:>10.1f}% {:>8.1f}% {:>8.1f}% {:>10.1f} {:>6.2f}".format(
            name, n, 100.0 * n / n_free,
            100.0 * b["on_gt_end"] / n, 100.0 * b["on_gt_seg"] / n,
            100.0 * b["near_comp"] / n, 100.0 * b["near_text"] / n,
            float(np.mean(b["grp_len"])), float(np.mean(b["nseg"]))))
    print("\n注：'落在GT端点' = 该自由端在某个 GT 走线端点 6px 内（真端）；")
    print("    '落在GT走线' = 在 GT 走线 2px 内但多半不在端点（停在走线中段）。")


if __name__ == "__main__":
    main()
