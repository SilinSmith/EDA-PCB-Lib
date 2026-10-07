# -*- coding: utf-8 -*-
"""图章形状勘察：官方 GT 的每类 bbox 到底是「正方形」还是「长方形」。

动机（owner 2026-xx）：「图章那个虽然在误差内但是明显你用长方形得到的误差更小。」

`eda-pcb\\schematic_parser\\stamp.py` 的 `STAMP_SIDE` 取每个实例的 `max(w, h)`，
把全部图章类统一压成**正方形**。本脚本量的是：如果把图章做成 **(长, 短)** 的
长方形（并按实例的轴向摆放），`bbox_ok` 的角点误差能省多少。

用法：
    python work\\tools\\diag\\_stampshape.py            # A 部分：派生长方形模板
    python work\\tools\\diag\\_stampshape.py --crop      # B 部分：导出 pin/testpoint 对照图
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "eda-pcb"))

from schematic_parser import data as D                       # noqa: E402
from schematic_parser import stamp as ST                     # noqa: E402
from schematic_parser.metrics import EvalConfig, bbox_ok     # noqa: E402

PREVIEW = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "preview")


def _iter_comps():
    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if not c.json_path:
            continue
        g = json.load(open(c.json_path, encoding="utf-8"))
        for k, cv in (g.get("components") or {}).items():
            bb = cv.get("bbox")
            if not bb or len(bb) != 4:
                continue
            w, h = float(bb[2]) - float(bb[0]), float(bb[3]) - float(bb[1])
            if w <= 0 or h <= 0:
                continue
            yield (cv.get("type") or "").strip(), w, h


def part_a():
    cfg = EvalConfig()
    shapes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t, w, h in _iter_comps():
        shapes[t][(round(max(w, h)), round(min(w, h)))] += 1
    print(f"{'type':<16}{'实例':>6}{'模板(长,短)':>14}{'占该类型':>9}"
          f"{'正方形边长':>11}{'方形最大误差':>13}{'矩形最大误差':>13}")
    tot_sq = tot_rr = 0.0
    n_tot = 0
    for t in sorted(shapes, key=lambda z: -sum(shapes[z].values())):
        cnt = shapes[t]
        n = sum(cnt.values())
        if t not in ST.SNAP_TYPES:
            continue
        (L, S), top = cnt.most_common(1)[0]
        side = ST.STAMP_SIDE.get(t)
        # 用「完美居中的真值框」比较两种贴法：方形 vs 矩形（轴向对）
        gt = [0.0, 0.0, float(L), float(S)]
        sq = [-(side - L) / 2.0, -(side - S) / 2.0,
              L + (side - L) / 2.0, S + (side - S) / 2.0]
        rr = list(gt)
        e_sq = 0.0 if bbox_ok(gt, sq, cfg) is True else 1.0
        e_rr = 0.0 if bbox_ok(gt, rr, cfg) is True else 1.0
        # 真正的误差量：角点切比雪夫距离
        d_sq = max(abs(sq[0] - gt[0]), abs(sq[1] - gt[1]),
                   abs(sq[2] - gt[2]), abs(sq[3] - gt[3]))
        d_rr = 0.0
        tot_sq += d_sq * top
        tot_rr += d_rr * top
        n_tot += top
        print(f"{t:<16}{n:>6}{str((L, S)):>14}{top / n:>8.1%}"
              f"{side:>11}{d_sq:>13.1f}{d_rr:>13.1f}")
    if n_tot:
        print(f"\n按实例数加权的平均角点误差：方形 {tot_sq / n_tot:.2f}px"
              f" / 矩形 {tot_rr / n_tot:.2f}px")
    # 轴向分布：长边是横的还是竖的
    print("\n每类「长边竖直」的比例：")
    ax: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t, w, h in _iter_comps():
        if t in ST.SNAP_TYPES:
            ax[t]["竖" if h >= w else "横"] += 1
    for t in sorted(ax, key=lambda z: -sum(ax[z].values())):
        c = ax[t]
        print(f"  {t:<16}{dict(c)}  竖 {c['竖'] / sum(c.values()):.1%}")


def part_b(per=6):
    import cv2
    import numpy as np
    rows = []
    want = {"pin": per, "testpoint": per}
    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if not c.json_path or not any(want.values()):
            continue
        g = json.load(open(c.json_path, encoding="utf-8"))
        img = cv2.imread(c.png_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        for k, cv in (g.get("components") or {}).items():
            t = (cv.get("type") or "").strip()
            if t not in want or want[t] <= 0:
                continue
            bb = cv["bbox"]
            x0, y0 = int(bb[0]) - 14, int(H - bb[3]) - 14
            x1, y1 = int(bb[2]) + 14, int(H - bb[1]) + 14
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = img[y0:y1, x0:x1]
            sc = max(1, int(64 / max(1, min(crop.shape[:2]))))
            crop = cv2.resize(crop, None, fx=sc, fy=sc,
                              interpolation=cv2.INTER_NEAREST)
            canvas = np.full((96, 160, 3), 255, np.uint8)
            hh = min(88, crop.shape[0])
            ww = min(152, crop.shape[1])
            ox, oy = (160 - ww) // 2, 4 + (88 - hh) // 2
            canvas[oy:oy + hh, ox:ox + ww] = crop[:hh, :ww]
            cv2.putText(canvas, f"{t} {k} {c.case_id}", (3, 93),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1)
            rows.append((t, canvas))
            want[t] -= 1
    if not rows:
        print("没有找到可裁剪的实例")
        return
    out = np.vstack([r[1] for r in rows])
    os.makedirs(PREVIEW, exist_ok=True)
    p = os.path.join(PREVIEW, "_tp_vs_pin.png")
    cv2.imwrite(p, out)
    print("已写出", p, out.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", action="store_true")
    a = ap.parse_args()
    if a.crop:
        part_b()
    else:
        part_a()
