# -*- coding: utf-8 -*-
"""_nlext.py —— 判断「单端错」的错端到底是**墨迹也短**还是**墨迹更长**。

做法（比 `_nlspan.py` 更硬）：不用垂直带采样，而是在错端附近找一个「干净的轴对齐窗口」，
用**暗像素行程（run）**沿轴方向量出这条线的真实起点/终点：
  * 先沿轴找到最近的、与我的轴垂直方向一致的位置（用我的端点所在直线，取其在窗口内的暗行程）；
  * 行程长度 = 该处暗连续像素数，判为「同一条线」当长度 <= wmax；
  * 沿轴逐步走，直到行程断开或爆宽（爆宽 = 走到交叉/拐角/别的结构）⇒ 得到墨迹端 t_ink。

输出：对每条 one_end 样本，错端处 `t_gt`（GT 端沿轴坐标）与 `t_ink`（墨迹端）之差：
  * |t_ink - t_gt| <= 4  ⇒ **墨迹能指出 GT 端**（可救）
  * 否则墨迹端更靠近我方端 ⇒ 我方没错/墨迹不可信

用法：python -X utf8 tools/diag/_nlext.py --cases 2,3,6,9,10 --samples 6
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


def _runlen(gray, H, x, y, nx, ny, thr, half=14):
    """(x,y) 处沿垂直方向 (nx,ny) 的暗连续行程长度（中心必须暗）。"""
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]):
        return 0
    if 255 - gray[H - yi, xi] < thr:
        return 0
    n = 1
    for s in (1, -1):
        for k in range(1, half + 1):
            x2, y2 = int(round(xi + s * k * nx)), int(round(yi + s * k * ny))
            if not (0 <= x2 < gray.shape[1] and 0 <= y2 < gray.shape[0]):
                break
            if 255 - gray[H - y2, x2] < thr:
                break
            n += 1
    return n


def _axis_extent(gray, H, px, py, ux, uy, thr, wmax, rmax=40, step=1.0):
    """从 (px,py) 沿 (ux,uy) 单方向走，返回最后一个「行程<=wmax 且连续」的位置坐标 t。"""
    nx, ny = -uy, ux
    last = 0.0
    t = 0.0
    while t <= rmax:
        t += step
        x, y = px + t * ux, py + t * uy
        w = _runlen(gray, H, x, y, nx, ny, thr)
        if w == 0 or w > wmax:
            break
        last = t
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Y:/workspace/EDA/work/_fast_out_delivery_default")
    ap.add_argument("--json", default="Y:/workspace/EDA/work/_nlpair2.json")
    ap.add_argument("--cases", default="2,3,6,9,10")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--thr", type=float, default=90.0)
    ap.add_argument("--wmax", type=int, default=8)
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
        arr = np.asarray(Image.open(case.png_path).convert("RGB"), dtype=np.uint8)
        gray = arr.mean(axis=2)
        H = arr.shape[0]
        for s in r["samples"]:
            if s["kind"] != "one_end":
                continue
            bad = "a" if s["a"]["d"] > TOL else "b"
            b = s[bad]
            GP = tuple(s["pred"][0] if bad == "a" else s["pred"][1])
            GB = tuple(s["gt"][0] if bad == "a" else s["gt"][1])
            GO = tuple(s["pred"][1] if bad == "a" else s["pred"][0])
            vx, vy = GO[0] - GP[0], GO[1] - GP[1]
            L = math.hypot(vx, vy)
            if L < 1e-6:
                continue
            ux, uy = vx / L, vy / L
            t_gt = (GB[0] - GP[0]) * ux + (GB[1] - GP[1]) * uy
            # 从错端往外（沿 -u）量墨迹；也从好端往里量，交叉验证这条线是不是直的
            t_out = _axis_extent(gray, H, GP[0], GP[1], -ux, -uy, a.thr, a.wmax)
            t_in = _axis_extent(gray, H, GP[0], GP[1], ux, uy, a.thr, a.wmax)
            # GT 端往外还有多少墨迹（GT 端真的结束了吗）
            gx, gy = GP[0] + t_gt * ux, GP[1] + t_gt * uy
            t_gt_out = _axis_extent(gray, H, gx, gy, -ux, -uy, a.thr, a.wmax, rmax=20)
            t_gt_in = _axis_extent(gray, H, gx, gy, ux, uy, a.thr, a.wmax, rmax=20)
            dirn = "out" if t_gt < 0 else "in"      # GT 端在我方段的哪一侧
            t_ink = -t_out if t_gt < 0 else L + t_in
            d_ink_gt = abs(t_ink - t_gt)
            d_ink_our = abs(t_ink - (0.0 if t_gt < 0 else L))
            key = (b["cls"], dirn)
            d = agg.setdefault(key, dict(n=0, ink_at_gt=0, ink_at_our=0, mid=0,
                                         s=0.0, s2=0.0))
            d["n"] += 1
            d["s"] += d_ink_gt
            d["s2"] += d_ink_our
            if d_ink_gt <= 4.0:
                d["ink_at_gt"] += 1
            elif d_ink_our <= 4.0:
                d["ink_at_our"] += 1
            else:
                d["mid"] += 1
            lst = shown.setdefault(cid, [])
            if len(lst) < a.samples:
                lst.append(dict(cls=b["cls"], d=b["d"], t_gt=round(t_gt, 1), L=round(L, 1),
                                t_out=round(t_out, 1), t_in=round(t_in, 1),
                                t_gt_out=round(t_gt_out, 1), t_gt_in=round(t_gt_in, 1),
                                ink_gt=round(d_ink_gt, 1), ink_our=round(d_ink_our, 1)))
    print("%-7s %-5s %5s %10s %10s %6s %8s %8s"
          % ("cls", "侧", "n", "墨迹=GT端", "墨迹=我方端", "居中", "均|GT|", "均|我方|"))
    for k in sorted(agg, key=lambda z: -agg[z]["n"]):
        d = agg[k]
        print("%-7s %-5s %5d %10d %10d %6d %8.1f %8.1f"
              % (k[0], k[1], d["n"], d["ink_at_gt"], d["ink_at_our"], d["mid"],
                 d["s"] / d["n"], d["s2"] / d["n"]))
    print("\nt_gt=GT 端沿轴坐标(负=在我方起点外) L=我方段长 t_out/t_in=从错端沿轴两个方向的"
          "墨迹延伸 t_gt_out/t_gt_in=从 GT 端两方向的墨迹延伸")
    for cid, lst in sorted(shown.items()):
        print(" case %s" % cid)
        for s in lst:
            print("   %-6s d=%-5s t_gt=%-7s L=%-6s t_out=%-5s t_in=%-5s gt_out=%-5s gt_in=%-5s"
                  " -> |ink-GT|=%-5s |ink-our|=%s"
                  % (s["cls"], s["d"], s["t_gt"], s["L"], s["t_out"], s["t_in"],
                     s["t_gt_out"], s["t_gt_in"], s["ink_gt"], s["ink_our"]))


if __name__ == "__main__":
    main()
