# -*- coding: utf-8 -*-
"""_leadprobe.py — 「GT 把导线切在焊盘/引线边界，而我们切在引脚中心」这条假设取证。

对每条 GT 导线端点 p（结构类别 blank：既不挨引脚/别的导线/元件框）：
  找最近的 GT 引脚 q（<=40px），报告
    d        |p-q|
    dt[p]    p 处墨的 distance-transform（≈导线半宽，1~2px）
    dt[mid]  p 与 q 中点处
    dt[q]    q（引脚/焊盘）处
    run      沿 p→q 方向，dt 第一次 >= THICK 的位置离 p 的距离（找不到记 -1）
判据：若 dt[p] 细、dt[q] 粗，且 run ≈ d ⇒ GT 的切点就是「细墨变粗墨」的边界，
      可以在提取器里用「沿导线走回粗墨边界」实现；若整条都细（run=-1）⇒ 另一回事。
用法：python -X utf8 tools\\diag\\_leadprobe.py --cases 6,10
"""
from __future__ import print_function

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                       # noqa: E402
from scipy import ndimage                                                # noqa: E402
from pipeline import pcore, prun                                         # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import normalise_doc                        # noqa: E402
import _nlerr as NL                                                       # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--thick", type=float, default=3.0)
    ap.add_argument("--max-d", type=float, default=40.0)
    a = ap.parse_args(argv)

    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)

    tot = {"n": 0, "thin_end": 0, "thick_pin": 0, "run_hit": 0}
    ds, runs = [], []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        if case is None:
            continue
        pc = pcore.find_case(cid)
        inp = prun.inputs_of(pc)
        res = prun.skeleton_of(inp, "adaptive")
        ink = np.asarray(res.ink).astype(bool)
        dt = ndimage.distance_transform_edt(ink)
        H = ink.shape[0]
        gt = normalise_doc(load_json(case.json_path))
        pins = NL._pins_of(gt)
        boxes = NL._boxes_of(gt)
        rows = []
        for _nk, net in gt["nets"].items():
            es = NL._edges_of(net)
            for gi, e in enumerate(es):
                for p in e:
                    if NL._end_class(p, pins, es, gi, boxes, 3.0) != "blank":
                        continue
                    q, d = None, 1e18
                    for c2 in pins:
                        dd = math.hypot(c2[0] - p[0], c2[1] - p[1])
                        if dd < d:
                            q, d = c2, dd
                    if q is None or d > a.max_d or d < 3.0:
                        continue

                    def val(pt):
                        x, y = int(round(pt[0])), int(round(H - pt[1]))
                        if 0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]:
                            return float(dt[y, x])
                        return -1.0

                    ux, uy = (q[0] - p[0]) / d, (q[1] - p[1]) / d
                    run = -1.0
                    t = 0.0
                    while t <= d:
                        if val((p[0] + ux * t, p[1] + uy * t)) >= a.thick:
                            run = t
                            break
                        t += 0.5
                    dp, dm, dq = val(p), val(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)), val(q)
                    tot["n"] += 1
                    if dp <= 2.5:
                        tot["thin_end"] += 1
                    if dq >= a.thick:
                        tot["thick_pin"] += 1
                    if run >= 0:
                        tot["run_hit"] += 1
                    ds.append(d)
                    runs.append(run if run >= 0 else d)
                    rows.append((p, d, dp, dm, dq, run))
        rows.sort(key=lambda r: -r[1])
        print("case %s: blank 且近引脚的 GT 端点 %d 个" % (cid, len(rows)))
        for p, d, dp, dm, dq, run in rows[:6]:
            print("   p=(%7.1f,%7.1f) d=%5.1f  dt[p]=%.1f dt[mid]=%.1f dt[pin]=%.1f "
                  "run=%s" % (p[0], p[1], d, dp, dm, dq,
                              ("%.1f" % run) if run >= 0 else "none"))
    if tot["n"]:
        ds.sort()
        runs.sort()
        n = tot["n"]
        print("\n== 合计 %d 个 blank 端点（%s）==" % (n, a.cases))
        print("   d 到最近 GT 引脚：中位 %.1f  p90 %.1f  最大 %.1f"
              % (ds[n // 2], ds[int(n * 0.9)], ds[-1]))
        print("   dt[p]<=2.5（末端是细线）：%d/%d = %.1f%%" % (tot["thin_end"], n, 100.0 * tot["thin_end"] / n))
        print("   dt[pin]>=%.1f（引脚处是粗块）：%d/%d = %.1f%%" % (a.thick, tot["thick_pin"], n, 100.0 * tot["thick_pin"] / n))
        print("   沿 p→q 能走到粗墨：%d/%d = %.1f%%  run 中位 %.1f" % (tot["run_hit"], n, 100.0 * tot["run_hit"] / n, runs[n // 2]))


if __name__ == "__main__":
    main()
