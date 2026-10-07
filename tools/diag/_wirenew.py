# -*- coding: utf-8 -*-
"""找出「把 `WIRE_TOL` 从 1.5 抬到 N 之后**新产生**的段并」是哪几对。

动机：`WIRE_TOL=10` 在官方 9 张上净 +0.82，但 case 10 上把一个网从 5 个脚
吹成 11 个脚的**巨块**（过度并网，−0.96）。要设计一条只吃「端点微越过目标段」
（case 1 的电容引线比母线低 5px）而不吃「只是离得近」的局部规则，
就得先看清楚新并上的到底是哪些段、几何差在哪。

用法：
    python work\\tools\\diag\\_wirenew.py --dir work\\_fast_out_wtol_def --case 10 --tol 10
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "eda-pcb", "stage1"))

import labels as L  # noqa: E402


def as_segs(raw):
    out = []
    for s in raw:
        arr = np.asarray(s, np.float64).reshape(-1, 2)
        if len(arr) >= 2:
            out.append(arr)
    return out


def groups_of(segs, tol):
    flat = np.array([[a[0][0], a[0][1], a[-1][0], a[-1][1]] for a in segs], np.float64)
    g = L.wire_groups(flat, tol=tol, weld=L.WIRE_WELD, perp_tol=L.WIRE_PERP,
                      cos_tol=L.WIRE_COS, ext=L.WIRE_EXT, ext_perp=L.WIRE_EXT_PERP,
                      ext_cos=1.0, ext_nearest=bool(L.WIRE_EXT_NEAREST))
    return g


def side(pt, a, b):
    """点在有向直线 a->b 的哪一侧（叉积）。"""
    ab = b - a
    return float(ab[0] * (pt[1] - a[1]) - ab[1] * (pt[0] - a[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "_fast_out_wtol_def"))
    ap.add_argument("--case", default="10")
    ap.add_argument("--tol", type=float, default=10.0)
    a = ap.parse_args()

    p = os.path.join(a.dir, "runs", "stage1", a.case + ".stage1.json")
    s1 = json.load(open(p, encoding="utf-8"))
    segs = as_segs((s1.get("wire_detail") or {}).get("segments") or [])
    g0, g1 = groups_of(segs, 1.5), groups_of(segs, a.tol)

    # 新并上的段对：在 1.5 时不同组，在 tol 时同组；距离用 labels 自己的 _dist_pt_segs
    # （**必须用它**：自己算会忘了把 t 夹到 [0,1]，把「同一条直线上相隔 78px 的两段」
    #  算成 0px，于是整张列出来的都是假的）。
    flat = np.array([[a[0][0], a[0][1], a[-1][0], a[-1][1]] for a in segs], np.float64)
    ends = np.stack([flat[:, :2], flat[:, 2:]], axis=1).reshape(-1, 2)
    own = np.repeat(np.arange(len(flat)), 2)
    D = L._dist_pt_segs(ends, flat)

    pairs = []
    for m in range(len(ends)):
        i = int(own[m])
        row = D[m].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        d = float(row[j])
        if 1.5 < d <= a.tol and g0[i] != g0[j] and g1[i] == g1[j]:
            A, B = segs[i], segs[j]
            cross = side(A[0], B[0], B[-1]) * side(A[-1], B[0], B[-1]) < 0
            aa, bb = B[0], B[-1]
            ab = bb - aa
            t = float((ends[m] - aa) @ ab / max(1e-9, float(ab @ ab)))
            tag = "A0" if m % 2 == 0 else "A1"
            pairs.append((d, i, j, "端点-线段 %s t=%.2f 穿越=%s" % (tag, t, cross)))
    pairs.sort()
    print("把 tol 从 1.5 抬到 %g：新并上 %d 对" % (a.tol, len(pairs)))
    for d, i, j, why in pairs[:40]:
        A, B = segs[i], segs[j]
        print("  %6.1fpx  seg%-4d [%6.0f,%6.0f]->[%6.0f,%6.0f]  seg%-4d [%6.0f,%6.0f]->[%6.0f,%6.0f]  %s"
              % (d, i, A[0][0], A[0][1], A[-1][0], A[-1][1],
                 j, B[0][0], B[0][1], B[-1][0], B[-1][1], why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
