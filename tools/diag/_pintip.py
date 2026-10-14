# -*- coding: utf-8 -*-
"""§5.111 取证（六）：**桩尖**是不是比「框边 + 常数」更好的引脚点预测器？

对每个（键名都对上的）框元件引脚，沿外法线方向量「暗红（符号桩）」段的**外端**：
从框边外 3px 起扫（跳过框线本身），允许 ≤2px 间断，连续 6px 无桩则停。然后比较三个数：
  d_gt  = GT 点到框边的距离      tip = 桩尖到框边的距离      d_pr = 我方点到框边的距离
统计 |d_gt−tip| 与 |d_gt−12|（当前常数策略）的均值与 ≤5px 命中率。

用法：python -X utf8 work\\_pintip.py [--run gsd9] [--all] [--case 8] [--show 24]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

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


def outward(bbox, pt):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ux, uy = pt[0] - cx, pt[1] - cy
    wx, wy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    if abs(ux) / wx >= abs(uy) / wy:
        return "r" if ux > 0 else "l"
    return "b" if uy > 0 else "t"


def stub_tip(mask, x0, y0, x1, y1, edge, e, H, limit=45, skip=3, gap=2, need=6):
    """返回 (桩尖距离, 桩起距离, 桩长)；找不到桩返回 None。"""
    if edge == 'l':
        sx, sy, dx, dy = x0 - 1.0 - skip, e[1], -1.0, 0.0
    elif edge == 'r':
        sx, sy, dx, dy = x1 + 1.0 + skip, e[1], 1.0, 0.0
    elif edge == 't':
        sx, sy, dx, dy = e[0], (H - y1) + 1.0 + skip, 0.0, -1.0
    else:
        sx, sy, dx, dy = e[0], (H - y0) - 1.0 - skip, 0.0, 1.0
    first = None
    last = None
    miss = 0
    start_off = skip
    for d in range(limit):
        xi = int(round(sx + dx * d))
        yi = int(round(sy + dy * d))
        if not (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]):
            break
        if mask[yi, xi]:
            if first is None:
                first = start_off + d
            last = start_off + d
            miss = 0
        else:
            miss += 1
            if miss > gap:
                break
    if first is None or (last - first + 1) < need:
        return None
    return last, first, last - first + 1


ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gsd9")
ap.add_argument("--case", default="")
ap.add_argument("--all", action="store_true")
ap.add_argument("--show", type=int, default=24)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = EvalConfig()
agg_tip, agg_const, agg_pr = [], [], []
per = {}
print("case comp pin     edge  d_gt  d_pr   tip   起  长  |gt-tip| |gt-12| |gt-pr|")
print("-" * 88)
for cid in sorted(cases, key=lambda s: (len(s), s)):
    if not args.all and args.case and cid != args.case:
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
    img = cv2.imread(str(cases[cid].png_path), cv2.IMREAD_COLOR)
    if img is None:
        continue
    H = img.shape[0]
    r = img[:, :, 2].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 0].astype(np.int16)
    maroon = ((r > 90) & (g < 115) & (b < 115) &
              (r - np.maximum(g, b) > 20)).astype(np.uint8)
    shown = 0
    for c in sorted(gt['components']):
        if canon_type((gt['components'][c] or {}).get('type')) not in IC_TYPES:
            continue
        gb = (gt['components'][c] or {}).get('bbox')
        if not gb:
            continue
        x0, y0, x1, y1 = [float(v) for v in gb]
        gpins = gt['pins'].get(c) or {}
        ppins = pred2['pins'].get(c) or {}
        for pk in sorted(gpins, key=lambda s: (len(str(s)), str(s))):
            a = _pt((gpins[pk] or {}).get('point'))
            if not a:
                continue
            p = ppins.get(pk)
            bb = _pt((p or {}).get('point')) if isinstance(p, dict) else None
            if bb is None or (p or {}).get('pinname') != (gpins[pk] or {}).get('pinname'):
                continue
            edge = outward(gb, a)
            e = (a[0], H - a[1])
            d_gt = {"l": x0 - a[0], "r": a[0] - x1, "t": y0 - a[1], "b": a[1] - y1}[edge]
            d_pr = {"l": x0 - bb[0], "r": bb[0] - x1,
                    "t": y0 - bb[1], "b": bb[1] - y1}[edge]
            tip = stub_tip(maroon, x0, y0, x1, y1, edge, e, H)
            agg_const.append(abs(d_gt - 12.0))
            agg_pr.append(abs(d_gt - d_pr))
            d = per.setdefault(cid, [0, 0.0, 0, 0.0, 0, 0.0])
            d[0] += 1
            d[1] += abs(d_gt - d_pr)
            if tip:
                agg_tip.append(abs(d_gt - tip[0]))
                d[2] += 1
                d[3] += abs(d_gt - tip[0])
            if args.show and (args.case or args.all) and shown < args.show:
                print("%-4s %-4s %-7s %-5s %5.1f %5.1f %6s %4s %3s %8s %6.1f %6.1f"
                      % (cid, c, pk, edge, d_gt, d_pr,
                         ("%.1f" % tip[0]) if tip else "-",
                         tip[1] if tip else "-", tip[2] if tip else "-",
                         ("%.1f" % abs(d_gt - tip[0])) if tip else "-",
                         abs(d_gt - 12.0), abs(d_gt - d_pr)))
                shown += 1
print("-" * 88)
print("GT 点预测误差（全 10 张、键名都对上的脚）")
print("  当前（常数 12 / 我方实际点）: |gt-12| 均值 %.2f   |gt-pr| 均值 %.2f  n=%d"
      % (float(np.mean(agg_const)), float(np.mean(agg_pr)), len(agg_pr)))
if agg_tip:
    arr = np.asarray(agg_tip)
    print("  桩尖预测器                  : |gt-tip| 均值 %.2f  ≤5px 命中 %.1f%%  n=%d"
          % (float(arr.mean()), 100.0 * float((arr <= 5).mean()), arr.size))
    print("  常数 12 在同一子集上        : ≤5px 命中 %.1f%%"
          % (100.0 * float((np.asarray(agg_const[:arr.size]) <= 5).mean())))
print("  逐案（n, |gt-pr| 均值, 有桩尖 n, |gt-tip| 均值）：")
for cid in sorted(per, key=lambda s: (len(s), s)):
    d = per[cid]
    print("    case %-3s n=%-3d |gt-pr|=%.2f   n_tip=%-3d |gt-tip|=%s"
          % (cid, d[0], d[1] / d[0], d[2],
             ("%.2f" % (d[3] / d[2])) if d[2] else "-"))
