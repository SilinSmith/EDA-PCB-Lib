# -*- coding: utf-8 -*-
"""_nlspan.py —— 「单端错」的错端到底该落在哪：沿轴墨迹剖面探测。

对 `_nlpair2.py` 找出的**单端错**（一端 ≤5px、另一端 >5px）样本，沿 pred 段的轴带向外
采样图像墨迹强度，看：
  * 墨迹沿轴延伸到哪（是我们欠伸 = 墨迹更长，还是我们过伸 = 墨迹更短）；
  * 墨迹端点更接近 **GT 端** 还是 **我方端**；
  * 端点外 4px 内是否有**横向墨迹**（= T 接/交叉，说明该切在交叉处）。

坐标：GT/产物都是 doc 坐标（y 向上），图像像素用 `im[H - y, x]`。
墨迹强度不用交付层的颜色分类器：直接取灰度暗度 `255 - min(b,g,r)`，在垂直于轴的方向取
`half` 像素带内的中位/最大值，抗 1~2px 的横向偏位。

用法：
  python -X utf8 tools/diag/_nlspan.py --dir Y:/workspace/EDA/work/_fast_out_delivery_default \
      --cases 2,3,6,9,10 --cls blank --samples 8
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, DEFAULT_DATA_ROOT   # noqa: E402

TOL = 5.0


def _prof(gray, H, good, other, rmax=45, half=7, step=1.0, thr=90.0):
    """沿 good->other 轴的剖面：每 t 给 (墨迹强度 max, 暗连续宽度 px)。

    宽度用**轴心两侧连续暗像素**数：中心像素必须暗，然后向两侧扩到第一个不暗的像素。
    这样「旁边 5px 外另有一条线」不会被算成本线（宽度会膨胀）⇒ 长度用**width<=wmax**筛。
    """
    vx, vy = other[0] - good[0], other[1] - good[1]
    L = math.hypot(vx, vy)
    ux, uy = vx / L, vy / L
    nx, ny = -uy, ux
    out = []
    t = -rmax
    while t <= L + rmax + 1e-9:
        px0, py0 = good[0] + t * ux, good[1] + t * uy
        vals = []
        for o in range(-half, half + 1):
            x = int(round(px0 + o * nx))
            y = int(round(py0 + o * ny))
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                vals.append(255 - int(gray[H - y, x]))
            else:
                vals.append(0.0)
        c = half
        w = 0
        if vals[c] >= thr:
            w = 1
            k = c - 1
            while k >= 0 and vals[k] >= thr:
                w += 1
                k -= 1
            k = c + 1
            while k < len(vals) and vals[k] >= thr:
                w += 1
                k += 1
        out.append((t, float(max(vals)), w))
        t += step
    return L, out, (ux, uy, nx, ny)


def _span(prof, thr, wmax):
    """墨迹（>=thr 且暗连续宽度<=wmax）在轴上的连通区间里，包含段中点的那一段 -> (t_lo, t_hi)。"""
    def on(t):
        return [z for z in prof if z[0] == t][0]
    mid_t = (prof[0][0] + prof[-1][0]) / 2.0
    runs = []
    cur = None
    for t, v, w in prof:
        if v >= thr and (wmax <= 0 or 0 < w <= wmax):
            if cur is None:
                cur = [t, t]
            else:
                cur[1] = t
        else:
            if cur is not None:
                runs.append(tuple(cur))
                cur = None
    if cur is not None:
        runs.append(tuple(cur))
    if not runs:
        return None, None
    best = None
    for r in runs:
        d = 0.0 if r[0] <= mid_t <= r[1] else min(abs(r[0] - mid_t), abs(r[1] - mid_t))
        if best is None or d < best[0]:
            best = (d, r)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Y:/workspace/EDA/work/_fast_out_delivery_default")
    ap.add_argument("--cases", default="2,3,6,9,10")
    ap.add_argument("--cls", default="")            # 只看某类错端，空 = 全部
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--thr", type=float, default=90.0)
    ap.add_argument("--wmax", type=int, default=9, help="暗连续宽度上限 px，0=不限")
    ap.add_argument("--json", default="Y:/workspace/EDA/work/_nlpair2.json")
    a = ap.parse_args()

    from PIL import Image
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    rows = json.load(io.open(a.json, encoding="utf-8"))["rows"]
    want = set(x.strip() for x in a.cases.split(",") if x.strip())

    agg = {}
    shown = {}
    for r in rows:
        cid = r["case"]
        if cid not in want:
            continue
        case = index.get(cid)
        im = Image.open(case.png_path).convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        gray = arr.mean(axis=2)
        H = arr.shape[0]
        for s in r["samples"]:
            if s["kind"] != "one_end":
                continue
            bad = "a" if s["a"]["d"] > TOL else "b"
            b = s[bad]
            if a.cls and b["cls"] != a.cls:
                continue
            GP = tuple(s["pred"][0] if bad == "a" else s["pred"][1])   # 我方错端
            GB = tuple(s["gt"][0] if bad == "a" else s["gt"][1])       # GT 真端
            GO = tuple(s["pred"][1] if bad == "a" else s["pred"][0])   # 我方对端（好端）
            L, prof, (ux, uy, nx, ny) = _prof(gray, H, GP, GO, thr=a.thr)
            t_gt = (GB[0] - GP[0]) * ux + (GB[1] - GP[1]) * uy
            lo, hi = _span(prof, a.thr, a.wmax)
            if lo is None:
                key = (b["cls"], "no_ink")
                agg[key] = agg.get(key, 0) + 1
                continue

            def at(tt):
                return min(prof, key=lambda z: abs(z[0] - tt))[1]

            beyond = at(t_gt + 6.0) if t_gt > L else at(t_gt - 6.0)
            t_ink_end = hi if t_gt > L else lo
            d_ink_gt = abs(t_ink_end - t_gt)
            d_ink_our = abs(t_ink_end - (L if t_gt > L else 0.0))
            kind = "ink_beyond" if (hi > L + 2 or lo < -2) else "ink_inside"
            key = (b["cls"], kind)
            d = agg.setdefault(key, dict(n=0, ink_closer_gt=0, ink_closer_our=0,
                                         sum_gt=0.0, sum_our=0.0, sum_x=0.0))
            d["n"] += 1
            d["sum_gt"] += d_ink_gt
            d["sum_our"] += d_ink_our
            d["sum_x"] += beyond
            if d_ink_gt < d_ink_our:
                d["ink_closer_gt"] += 1
            else:
                d["ink_closer_our"] += 1
            lst = shown.setdefault(cid, [])
            if len(lst) < a.samples:
                lst.append(dict(cls=b["cls"], d=b["d"], t_gt=round(t_gt, 1), L=round(L, 1),
                                ink=[round(lo, 1), round(hi, 1)],
                                d_ink_gt=round(d_ink_gt, 1), d_ink_our=round(d_ink_our, 1),
                                beyond=round(beyond, 1),
                                profile=[(round(t, 1), round(v), w) for t, v, w in prof
                                         if abs(t) < 4 or abs(t - L) < 4 or abs(t - t_gt) < 4]))
    print("%-8s %-12s %5s %10s %10s %8s %8s" % ("cls", "kind", "n", "墨迹更近GT", "更近我方", "均值|GT|", "端点外墨迹"))
    for k in sorted(agg, key=lambda z: -agg[z]["n"]):
        d = agg[k]
        print("%-8s %-12s %5d %10d %10d %8.1f %8.1f"
              % (k[0], k[1], d["n"], d["ink_closer_gt"], d["ink_closer_our"],
                 d["sum_gt"] / d["n"], d["sum_x"] / d["n"]))
    print("\n样例（t_gt = GT 端沿轴坐标；L = 我方段长；ink = 墨迹区间沿轴坐标；"
          "beyond = GT 端再往外 6px 处的墨迹强度）：")
    for cid, lst in sorted(shown.items()):
        print(" case %s" % cid)
        for s in lst:
            print("   %-6s d=%-5s t_gt=%-7s L=%-6s ink=%-16s d_ink(GT/our)=%s/%s beyond=%s"
                  % (s["cls"], s["d"], s["t_gt"], s["L"], s["ink"], s["d_ink_gt"],
                     s["d_ink_our"], s["beyond"]))
            print("      prof %s" % s["profile"])


if __name__ == "__main__":
    main()
