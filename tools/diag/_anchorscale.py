# -*- coding: utf-8 -*-
"""外伸量能不能从**图像可观测的尺度**预测？（逐案 / 逐元件）

背景（`work\\_anchorfit.py` 实测）：
  · GT 的脚点外伸量在（元件, 边）内**严格恒定** —— 10GT 37/37 组极差≤1px，
    池子 1770/1770 组（max 1.0px）；
  · 神谕：逐（元件,边）已知 ⇒ 引脚 F1 0.8166→0.9251（**+2.71 分**）；
    逐案一个常数 ⇒ +1.06 分；逐脚精确 ⇒ +3.47 分。
  ⇒ 问题从「逐脚预测」降成「逐元件逐边预测一个标量」。

本探针问：这个标量**是不是尺度**？
  池子里的外伸量跨度 3~68px（10GT 只有 6~22），若不同案例的示意图是不同缩放渲染的，
  那么任何几何尺度（框高、引脚间距、字高）都应当与它强相关 —— 那就能直接算出来。

对每个案例（10GT + 池子）打印：中位外伸量 ｜ 中位框高/框宽 ｜ 中位引脚节距 ｜ 脚数，
再给跨案例的相关系数（Pearson / Spearman）与单变量最小二乘拟合。

用法：
    python -X utf8 work\\_anchorscale.py [--limit 200] [--cases ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import canon_type                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--splits", default="10GT,200_train_cases")
a = ap.parse_args()


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get("x"), raw.get("y"))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def box_of(comp):
    bb = (comp or {}).get("bbox")
    if isinstance(bb, dict):
        return (float(bb["x0"]), float(bb["y0"]), float(bb["x1"]), float(bb["y1"]))
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return tuple(float(v) for v in bb)
    return None


def edge_dist(p, box):
    x0, y0, x1, y1 = box
    return max([("x-", x0 - p[0]), ("x+", p[0] - x1), ("y-", y0 - p[1]), ("y+", p[1] - y1)],
               key=lambda kv: kv[1])


rows = []
for c in discover(DEFAULT_DATA_ROOT):
    if c.split not in a.splits.split(",") or not c.json_path:
        continue
    gt = normalise_doc(json.load(open(str(c.json_path), encoding="utf-8")))
    groups = defaultdict(list)          # (comp, side) -> [(off, tangential)]
    heights, widths = [], []
    for cid_, comp in (gt.get("components") or {}).items():
        if canon_type((comp or {}).get("type")) not in IC_TYPES:
            continue
        bb = box_of(comp)
        if not bb:
            continue
        heights.append(bb[3] - bb[1])
        widths.append(bb[2] - bb[0])
        for pk, info in ((gt.get("pins") or {}).get(cid_) or {}).items():
            p = _pt((info or {}).get("point"))
            if not p:
                continue
            side, off = edge_dist(p, bb)
            tang = p[1] if side.startswith("x") else p[0]
            groups[(cid_, side)].append((off, tang))
    if not groups:
        continue
    offs = [g[0] for items in groups.values() for g in items]
    pitches = []
    for items in groups.values():
        tt = sorted(t for _, t in items)
        pitches += [b - x for x, b in zip(tt, tt[1:]) if 0 < b - x < 200]
    rows.append({
        "split": c.split, "cid": str(c.case_id),
        "off": statistics.median(offs), "n": len(offs),
        "h": statistics.median(heights) if heights else 0.0,
        "w": statistics.median(widths) if widths else 0.0,
        "pitch": statistics.median(pitches) if pitches else 0.0,
        "gmed": [statistics.median([x[0] for x in v]) for v in groups.values()],
    })

rows.sort(key=lambda r: (r["split"], r["cid"]))
print("%-16s %-8s %8s %5s %8s %8s %8s" % ("split", "case", "外伸量", "脚数", "框高", "框宽", "节距"))
for r in rows:
    print("%-16s %-8s %8.1f %5d %8.1f %8.1f %8.1f" % (
        r["split"], r["cid"], r["off"], r["n"], r["h"], r["w"], r["pitch"]))

off = np.array([r["off"] for r in rows], dtype=float)
print("\n跨案例相关（n=%d）：" % len(rows))
for key, label in (("h", "框高中位"), ("w", "框宽中位"), ("pitch", "引脚节距"), ("n", "脚数")):
    x = np.array([r[key] for r in rows], dtype=float)
    ok = x > 0
    if ok.sum() < 3 or np.std(x[ok]) == 0:
        print("  %-8s 样本不足" % label)
        continue
    pear = float(np.corrcoef(off[ok], x[ok])[0, 1])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    spear = float(np.corrcoef(rank(off[ok]), rank(x[ok]))[0, 1])
    k, b = np.polyfit(x[ok], off[ok], 1)
    resid = off[ok] - (k * x[ok] + b)
    print("  %-8s Pearson %+.3f  Spearman %+.3f ｜ 拟合 外伸=%.3f×%s%+.2f ｜ 残差 中位|.| %.1f p90 %.1f" % (
        label, pear, spear, k, label, b, float(np.median(np.abs(resid))),
        float(np.percentile(np.abs(resid), 90))))

# 逐案例内的「组内相对差」：同一案例里 (元件,边) 之间的差异有多大？
print("\n案例内 (元件,边) 组的离散（相对该案例中位）：")
for split in ("10GT", "200_train_cases"):
    sub = [r for r in rows if r["split"] == split]
    if not sub:
        continue
    devs = []
    for r in sub:
        med = statistics.median(r["gmed"])
        devs += [g - med for g in r["gmed"]]
    if devs:
        print("  %-16s 组内中位偏离：中位 %.1f p90 %.1f max %.1f（n=%d 组）" % (
            split, statistics.median(devs), sorted(devs)[int(0.9 * (len(devs) - 1))],
            max(devs), len(devs)))
