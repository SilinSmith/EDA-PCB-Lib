"""testpoint 的 CV 可行性探针。

GT 的 testpoint 是恒定 20x20 图章。肉眼看真实裁片，符号是「同心双圆环」（靶心）。
本探针问三件事：
  1. 环能不能被找到（contour 圆度 / Hough）
  2. 环心相对 GT 框中心的偏移是不是常数（若是，CV 定位就成立）
  3. 环半径分布
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
import schematic_parser as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))
from gt_look import to_img  # noqa: E402


def circles_in(gray):
    """返回 [(cx, cy, r, circ)]，用 contour 圆度筛。"""
    inv = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p <= 0 or a < 12:
            continue
        circ = 4 * math.pi * a / (p * p)
        if circ < 0.55:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        out.append((float(cx), float(cy), float(r), float(circ), float(a)))
    return out


def main():
    deltas = []
    radii = []
    hit_ring = 0
    n_inst = 0
    per_src = Counter()
    miss_cases = Counter()
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        tps = [(k, v) for k, v in (doc.get("components") or {}).items()
               if str(v.get("type", "")).strip() == "testpoint"]
        if not tps:
            continue
        g = cv2.imread(str(c.png_path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        src = c.case_id
        for k, v in tps:
            b = v.get("bbox")
            if not b:
                continue
            x1, y1, x2, y2 = to_img(b, H)
            gcx, gcy = (x1 + x2) / 2, (y1 + y2) / 2
            pad = 4.0
            px1 = max(0, int(gcx - 20 * pad)); px2 = min(W, int(gcx + 20 * pad))
            py1 = max(0, int(gcy - 20 * pad)); py2 = min(H, int(gcy + 20 * pad))
            if px2 - px1 < 20 or py2 - py1 < 20:
                continue
            sub = g[py1:py2, px1:px2]
            n_inst += 1
            cs = circles_in(sub)
            if not cs:
                miss_cases[c.case_id] += 1
                continue
            # 取圆心离 GT 中心最近的环
            best = min(cs, key=lambda t: (t[0] + px1 - gcx) ** 2 + (t[1] + py1 - gcy) ** 2)
            cx, cy, r, circ, a = best
            cx += px1; cy += py1
            hit_ring += 1
            deltas.append((cx - gcx, cy - gcy, r, circ))
            radii.append(r)
            per_src[str(c.case_id).split("-")[-1]] += 1
    print(f"testpoint 实例 {n_inst}，找到环 {hit_ring}，找不到 {n_inst - hit_ring}")
    if not deltas:
        print("没有任何环被找到 —— CV 这条路需要换特征")
        return
    d = np.array(deltas)
    dx, dy, r, circ = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

    def st(v, nm):
        q = np.percentile(v, [10, 25, 50, 75, 90])
        print(f"  {nm:<6} mean {v.mean():7.2f}  p10 {q[0]:6.1f} p25 {q[1]:6.1f} "
              f"p50 {q[2]:6.1f} p75 {q[3]:6.1f} p90 {q[4]:6.1f}  sd {v.std():6.2f}")

    print("\n环心 - GT框中心（图像坐标，y 向下）:")
    st(dx, "dx"); st(dy, "dy")
    print("\n环半径 / 圆度:")
    st(r, "r"); st(circ, "circ")
    print("\n环心偏移落在 [-20,20] 内的比例:",
          f"{float(np.mean((np.abs(dx) <= 20) & (np.abs(dy) <= 20))):.3f}",
          "  落在 [-10,10] 内:", f"{float(np.mean((np.abs(dx) <= 10) & (np.abs(dy) <= 10))):.3f}")
    print("按 case:", dict(per_src))
    if miss_cases:
        print("找不到环的 case:", dict(miss_cases))


if __name__ == "__main__":
    main()
