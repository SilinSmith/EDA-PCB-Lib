# -*- coding: utf-8 -*-
"""结点位置重解（`PSKEL_NODE_SOLVE`）到底有没有把切点搬到 GT 的 T 接结点上？

只跑 stage1 的骨架（不跑整条流水线、不判分），直接量：
  对每个 GT「导线—导线 T 接」端点，到最近的**我方段端点**的距离 p50 / p90 / ≤5px 占比。
开/关 `PSKEL_NODE_SOLVE` 各算一遍做对照（`pipeline/prun.py:104 skeleton_of` 有缓存，
所以这里自己 new `WireExtractor`，与 `prun.py:115` 参数逐字一致）。

用法：python -X utf8 tools\\diag\\_nodesolvechk.py --cases 1,2,3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from pipeline import pcore, prun, pskel                                       # noqa: E402
from schematic_parser.metrics import _edge_points, normalise_doc              # noqa: E402


def gt_join_ends(gt_doc, h, tol=5.0):
    """GT 里落在**别的** GT 导线上的导线端点（图像坐标）。"""
    gt = normalise_doc(gt_doc)
    edges = []
    for _nk, net in (gt.get("nets") or {}).items():
        for _ek, e in (net.get("edges") or {}).items():
            p = _edge_points(e)
            if p is not None and len(p) >= 2:
                edges.append([(float(q[0]), h - float(q[1])) for q in p])
    out = []
    for i, pts in enumerate(edges):
        for end in (0, -1):
            p = pts[end]
            best = 1e9
            for j, other in enumerate(edges):
                if j == i:
                    continue
                for k in range(len(other) - 1):
                    a, b = other[k], other[k + 1]
                    ab = (b[0] - a[0], b[1] - a[1])
                    L2 = ab[0] * ab[0] + ab[1] * ab[1]
                    if L2 < 1e-9:
                        continue
                    t = max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L2))
                    d = float(np.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1]))
                    if d < best:
                        best = d
            if best <= tol:
                out.append(p)
    return np.asarray(out, dtype=float) if out else np.zeros((0, 2))


def seg_ends(segs):
    pts = []
    for a, b in segs:
        pts.append((float(a[0]), float(a[1])))
        pts.append((float(b[0]), float(b[1])))
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def run_case(cid, on):
    os.environ["PSKEL_NODE_SOLVE"] = "1" if on else "0"
    inp = prun.inputs_of(pcore.find_case(cid))
    ex = pskel.WireExtractor(use_colour=False, close_ink=True, snap=False, ink_mode="adaptive")
    res = ex.extract(inp.bgr)
    ge = gt_join_ends(inp.gt_doc, inp.h)
    se = seg_ends(res.segments)
    if ge.shape[0] and se.shape[0]:
        d = np.hypot(ge[:, 0][:, None] - se[:, 0][None, :], ge[:, 1][:, None] - se[:, 1][None, :]).min(axis=1)
    else:
        d = np.zeros(0)
    return dict(n_seg=len(res.segments), n_gt_join=len(ge),
                solved=res.n_nodes_solved, shift=res.node_shift_mean,
                solved3=int(getattr(ex, "last_node_solved3", 0)),
                shift3=float(getattr(ex, "last_node_shift3", 0.0)),
                p50=float(np.median(d)) if d.size else float("nan"),
                p90=float(np.percentile(d, 90)) if d.size else float("nan"),
                hit5=float((d <= 5.0).mean() * 100.0) if d.size else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3")
    a = ap.parse_args(argv)
    print("case  臂     段数  GT_T接 结点重解 平均位移  ≥3度重解  ≥3度位移  d(p50) d(p90) <=5px")
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        for on in (False, True):
            r = run_case(cid, on)
            print("%-5s %-5s %5d %6d %8d %8.2f %9d %9.2f %6.2f %6.2f %5.1f%%"
                  % (cid, "ON" if on else "off", r["n_seg"], r["n_gt_join"], r["solved"],
                     r["shift"], r["solved3"], r["shift3"], r["p50"], r["p90"], r["hit5"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
