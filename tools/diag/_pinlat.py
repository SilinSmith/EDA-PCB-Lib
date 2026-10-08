# -*- coding: utf-8 -*-
"""GT 的 IC 引脚排布有多「规则」？—— 决定「按库补缺失引脚」这条路值不值得走。

背景（`_libopp.py` 量出来的）：我方交付产物在双方都匹配上的 box/IC 上，120 个 GT 引脚里
55 个**根本没有输出**（不是名字错、也不是号错：多余号 = 0）。所以唯一能靠元器件库补的，
是「库告诉我们这颗 IC 该有哪些引脚号 ⇒ 我们把缺的号补出来」。补号容易，**补坐标**才是门槛
（PinF1 要求 pinname 相等 **且** 点距 ≤ `pin_tol`=5 px）。

本脚本只量几何：
  1) 引脚的 x 分布（几列）、每列 y 间距是否均匀；
  2) 引脚号是否为**数值单调**沿周边排列（相邻号的点在空间上是否相邻）；
  3) 决定性指标：**用「号数轴上相邻的已知引脚线性插值」预测缺失引脚坐标**的误差分布。

用法：
    python -X utf8 work\\tools\\diag\\_pinlat.py --pred work\\_fnew --cases 1,2,3,4,5,7,8,9,10
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.metrics import EvalConfig, assign_components, canon_type, rewrite_keys  # noqa: E402
from schematic_parser.data import IC_TYPES, pin_key_to_number  # noqa: E402

GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"


def num_of(s: str):
    return int(s) if re.fullmatch(r"\d+", s or "") else None


def point_of(rec):
    p = (rec or {}).get("point") or {}
    return (float(p.get("x", 0.0)), float(p.get("y", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--gt", default=str(GT9))
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    cfg = EvalConfig()
    gtroot, predroot = Path(a.gt), Path(a.pred)
    err = Counter()
    monoto = Counter()
    colstat = []
    n_miss = n_interp = 0

    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        gtf, pf = gtroot / c / ("%s.json" % c), predroot / ("%s.json" % c)
        if not (gtf.exists() and pf.exists()):
            continue
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        pred = json.loads(pf.read_text(encoding="utf-8"))
        _f1, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)
        gpin, ppin = gt.get("pins", {}), pred2.get("pins", {})

        for k, rec in gt.get("components", {}).items():
            if canon_type(rec.get("type")) not in IC_TYPES or k not in gpin:
                continue
            if k not in pred2.get("components", {}):
                continue
            g = {pin_key_to_number(pk): point_of(pv) for pk, pv in gpin[k].items()}
            p = {pin_key_to_number(pk) for pk in (ppin.get(k) or {})}
            if len(g) < 4:
                continue
            # --- 1) 列结构与间距均匀性
            xs = Counter(round(x, 1) for x, _ in g.values())
            cols = [x for x, n in xs.items() if n >= 3]
            for cx in cols:
                ys = sorted(y for x, y in g.values() if abs(x - cx) < 0.6)
                if len(ys) >= 3:
                    gaps = [round(b - aa, 2) for aa, b in zip(ys, ys[1:])]
                    gaps_s = sorted(gaps)
                    med = gaps_s[len(gaps_s) // 2]
                    dev = max(abs(g2 - med) for g2 in gaps)
                    colstat.append((c, k, cx, len(ys), med, dev))
            # --- 2) 号数单调性（相邻号空间相邻）
            nums = sorted(n for n in (num_of(x) for x in g) if n is not None)
            for n1, n2 in zip(nums, nums[1:]):
                if n2 - n1 == 1:
                    d = math.dist(g[str(n1)] if str(n1) in g else g[n1], g[str(n2)] if str(n2) in g else g[n2])
                    monoto["adj"] += 1
                    if d <= 25.0:
                        monoto["adj_near"] += 1
            # --- 3) 缺失脚的插值可行性
            known = sorted((num_of(x), x) for x in g if num_of(x) is not None and x in p)
            known = [(n, s) for n, s in known if n is not None]
            for nstr, (x, y) in g.items():
                if nstr in p:
                    continue
                n_miss += 1
                n = num_of(nstr)
                if n is None or len(known) < 2:
                    err["no_number_or_too_few"] += 1
                    continue
                lo = [t for t in known if t[0] < n]
                hi = [t for t in known if t[0] > n]
                if not lo or not hi:
                    err["no_bracket"] += 1
                    continue
                n1, s1 = max(lo, key=lambda t: t[0])
                n2, s2 = max(hi, key=lambda t: -t[0])
                if n2 == n1:
                    err["no_bracket"] += 1
                    continue
                w = (n - n1) / float(n2 - n1)
                px = g[s1][0] + w * (g[s2][0] - g[s1][0])
                py = g[s1][1] + w * (g[s2][1] - g[s1][1])
                d = math.dist((px, py), (x, y))
                n_interp += 1
                for thr in (5.0, 10.0, 20.0, 50.0):
                    if d <= thr:
                        err["<=%g" % thr] += 1
                if a.verbose and d > 5.0:
                    print("    [interp-miss] %s %s pin %s  预测 (%.1f,%.1f) 真值 (%.1f,%.1f) 误差 %.1f px"
                          % (c, k, nstr, px, py, x, y, d))

    print("===== 列的间距均匀性（每列 ≥3 个引脚）=====")
    if colstat:
        devs = [d for *_x, m, d in [(c, k, cx, n, m, d) for c, k, cx, n, m, d in colstat]]
        meds = [m for _c, _k, _x, _n, m, _d in colstat]
        print("  列数 %d｜中位间距 中位 %.2f px｜间距最大偏差的中位 %.2f px｜偏差>2px 的列 %d/%d"
              % (len(colstat), sorted(meds)[len(meds) // 2], sorted(devs)[len(devs) // 2],
                 sum(1 for d in devs if d > 2.0), len(devs)))
    print("===== 号数单调性 =====")
    if monoto["adj"]:
        print("  相邻号对 %d 个，其中空间距离 ≤25 px 的占 %.1f%%"
              % (monoto["adj"], 100.0 * monoto["adj_near"] / monoto["adj"]))
    print("===== 缺失引脚用「号数轴线性插值」预测坐标（%d 个缺失脚，%d 个可插值）=====" % (n_miss, n_interp))
    if n_interp:
        for thr in (5.0, 10.0, 20.0, 50.0):
            print("  误差 ≤%4g px : %4d  %5.1f%%" % (thr, err["<=%g" % thr], 100.0 * err["<=%g" % thr] / n_interp))
        print("  无法插值：无号 %d、号不在中间 %d" % (err["no_number_or_too_few"], err["no_bracket"]))


if __name__ == "__main__":
    main()
