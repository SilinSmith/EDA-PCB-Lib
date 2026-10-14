# -*- coding: utf-8 -*-
"""§5.111 取证（四）：引脚点到底该落在哪 —— **桩尖（ink run 末端）** 假设检验。

对每个（GT 里键名都对上的）框元件引脚：从框边沿该脚的外法线方向往外走，量墨迹
（灰 < `--thr`，允许 ≤`--gap` 的空白）连续到多远，得到「桩尖」坐标；再把三个候选点
放在一起比较：GT 点 / 我方点 / 桩尖。若「GT ≈ 桩尖」而「我方 = 框边+12 常数」，
那病灶就是外推距离常数而不是桩归属。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_pinprobe.py --case 8 --comp U1 [--run gsd9] [--thr 128] [--gap 2]
    python -X utf8 work\\_pinprobe.py --all [--run gsd9]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,      # noqa: E402
                                   normalise_doc)
from schematic_parser.metrics import (EvalConfig, assign_components,          # noqa: E402
                                      canon_type, rewrite_keys)


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get('x'), raw.get('y'))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return float(raw[0]), float(raw[1])


def ink_run(gray, x, y, dx, dy, limit=80, thr=128, gap=2):
    """从 (x,y) 起沿 (dx,dy) 走，返回墨迹最远延伸长度（允许 gap 个白像素）。"""
    H, W = gray.shape[:2]
    last = 0
    blank = 0
    for d in range(1, limit + 1):
        xi, yi = int(round(x + dx * d)), int(round(y + dy * d))
        if not (0 <= xi < W and 0 <= yi < H):
            break
        if gray[yi, xi] < thr:
            last = d
            blank = 0
        else:
            blank += 1
            if blank > gap:
                break
    return last


def outward(bbox, pt):
    """返回 (边名, 外法线)。用「点相对框心的方向」定边，比最近边稳。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ux, uy = pt[0] - cx, pt[1] - cy
    wx, wy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    sx, sy = abs(ux) / wx, abs(uy) / wy
    if sx >= sy:
        return ("r", (1.0, 0.0)) if ux > 0 else ("l", (-1.0, 0.0))
    return ("b", (0.0, -1.0)) if uy > 0 else ("t", (0.0, 1.0))


ap = argparse.ArgumentParser()
ap.add_argument("--case", default="8")
ap.add_argument("--comp", default="")
ap.add_argument("--run", default="gsd9")
ap.add_argument("--thr", type=int, default=128)
ap.add_argument("--gap", type=int, default=2)
ap.add_argument("--all", action="store_true")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c

cfg = EvalConfig()
print("case comp pin      edge  gt_dist  pr_dist  tip_len  gt-tip  pr-tip  (gt / pr / tip)")
print("-" * 96)
stat = {}
for cid in sorted(cases, key=lambda s: (len(s), s)):
    if not args.all and cid != args.case:
        continue
    pred_p = (ROOT / "work" / ("_fast_out_%s" % args.run) / "final_results"
              / ("%s_Pred.json" % cid))
    if not pred_p.exists():
        continue
    gt_doc = json.load(open(str(cases[cid].json_path), encoding="utf-8"))
    pred_doc = json.load(open(str(pred_p), encoding="utf-8"))
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    comp, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)
    img = cv2.imread(str(cases[cid].png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    H = img.shape[0]
    for c in sorted(gt['components']):
        if canon_type((gt['components'][c] or {}).get('type')) not in IC_TYPES:
            continue
        if args.comp and c != args.comp:
            continue
        gb = (gt['components'][c] or {}).get('bbox')
        if not gb:
            continue
        x0, y0, x1, y1 = [float(v) for v in gb]
        gpins = gt['pins'].get(c) or {}
        ppins = pred2['pins'].get(c) or {}
        for pk in sorted(gpins, key=lambda s: (len(str(s)), str(s))):
            g = gpins[pk]
            a = _pt((g or {}).get('point'))
            if not a:
                continue
            p = ppins.get(pk)
            b = _pt((p or {}).get('point')) if isinstance(p, dict) else None
            edge, (dx, dy) = outward(gb, a)
            # 图坐标：y 翻转；外法线的图坐标方向
            ex, ey = a[0], H - a[1]
            dxi, dyi = dx, -dy
            # 从框边出发测量
            if edge == 'l':
                sx, sy = x0 - 1.0, ey
            elif edge == 'r':
                sx, sy = x1 + 1.0, ey
            elif edge == 't':
                sx, sy = ex, (H - y1) + 1.0
            else:
                sx, sy = ex, (H - y0) - 1.0
            dxi2, dyi2 = (1.0, 0.0) if edge == 'r' else \
                         (( -1.0, 0.0) if edge == 'l' else
                          (( 0.0, 1.0) if edge == 'b' else (0.0, -1.0)))
            tip = ink_run(img, sx, sy, dxi2, dyi2, thr=args.thr, gap=args.gap)
            gt_d = {"l": x0 - a[0], "r": a[0] - x1,
                    "t": y0 - a[1], "b": a[1] - y1}[edge]
            pr_d = None
            if b:
                pr_d = {"l": x0 - b[0], "r": b[0] - x1,
                        "t": y0 - b[1], "b": b[1] - y1}[edge]
            d_gt_tip = abs(gt_d - tip)
            d_pr_tip = abs(pr_d - tip) if pr_d is not None else None
            key = (edge, int(round(gt_d / 5.0) * 5), int(round((pr_d or 0) / 5.0) * 5))
            st = stat.setdefault(key, [0, 0.0, 0, 0.0])
            st[0] += 1
            st[1] += d_gt_tip
            if d_pr_tip is not None:
                st[2] += 1
                st[3] += d_pr_tip
            print("%-4s %-4s %-8s %-5s %7.1f %8s %8d %7.1f %7s   (%.0f,%.0f)/(%s)/tip@%.0f"
                  % (cid, c, pk, edge, gt_d,
                     ("%.1f" % pr_d) if pr_d is not None else "-",
                     tip, d_gt_tip,
                     ("%.1f" % d_pr_tip) if d_pr_tip is not None else "-",
                     a[0], a[1], ("%.0f,%.0f" % (b[0], b[1])) if b else "-", tip))
print("-" * 96)
print("边/GT档/我方档 → 样本数 · |GT−桩尖| 均值 · 有对儿样本数 · |我方−桩尖| 均值")
for k, v in sorted(stat.items(), key=lambda kv: -kv[1][0]):
    print("   %-22s n=%-3d  gt_tip=%.1f   n_pr=%-3d  pr_tip=%.1f"
          % (str(k), v[0], v[1] / max(v[0], 1), v[2], v[3] / max(v[2], 1)))
