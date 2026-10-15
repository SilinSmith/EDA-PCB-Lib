# -*- coding: utf-8 -*-
"""_nlpinoracle.py —— 「自由端沿轴回缩/吸附到最近引脚锚点」的**离线神谕上界**。

在**已交付产物**上模拟 `stage2.assemble.snap_ends_to_pins` 的新语义（允许回缩：
锚点落在端点身后 backoff px 内则把端点沿轴拉到锚点），用 `_wirecut.score` 重算
`net_line_f1`，从而在**不跑管线**的情况下一分钟扫完参数网格，判定该规则是否值得做。

用法：
  python -X utf8 tools/diag/_nlpinoracle.py --dir work/_fast_out_delivery_default \
      --cases 1,2,3,4,5,6,7,8,9,10
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, load_json, DEFAULT_DATA_ROOT   # noqa: E402
from schematic_parser.metrics import EvalConfig                # noqa: E402
from tools.diag._nlerr import _edges_of, _pins_of              # noqa: E402
from tools.diag._nlpair2 import _ends, _d                      # noqa: E402
from tools.diag import _wirecut                                # noqa: E402


def _pe(seg):
    """交付/GT 结构里的边 = [{'x','y'}, ...]；取前两点。"""
    return ((float(seg[0]["x"]), float(seg[0]["y"])),
            (float(seg[1]["x"]), float(seg[1]["y"])))


def _pin_pts(doc):
    return list(_pins_of(doc) or [])


def snap_doc(doc, ppt, radius, perp, backoff):
    """原地改 doc 的 nets[*].edges 端点，返回移动端点数。语义与
    `stage2.assemble.snap_ends_to_pins` 的 backoff 版一致（只动自由端）。"""
    moved = 0
    for _nk, net in (doc.get("nets") or {}).items():
        es = _edges_of(net)
        if isinstance(net, dict) and "edges" in net:
            edges = net["edges"]
        else:
            continue
        eo = edges if isinstance(edges, dict) else None
        if not eo:
            continue
        keys = [k for k, v in eo.items()
                if isinstance(v, list) and len(v) >= 2
                and isinstance(v[0], dict) and "x" in v[0]]
        segs = [(k, _pe(eo[k])) for k in keys]
        verts = [p for _k, e in segs for p in e]
        for k, ((ax, ay), (bx, by)) in segs:
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            for idx, (px, py) in ((0, (ax, ay)), (1, (bx, by))):
                if any(math.hypot(px - qx, py - qy) <= 3.0
                       for (qx, qy) in verts if (qx, qy) != (px, py)):
                    continue
                sx, sy = (ux, uy) if idx == 1 else (-ux, -uy)
                bt, best = 0.0, None
                for (qx, qy) in ppt:
                    vx, vy = qx - px, qy - py
                    t = vx * sx + vy * sy
                    if t <= -backoff or t > radius:
                        continue
                    if abs(vx * (-sy) + vy * sx) > perp:
                        continue
                    if best is None or abs(t) < abs(bt):
                        bt, best = t, (qx, qy)
                if best is None:
                    continue
                eo[k][idx]["x"] = round(best[0], 2)
                eo[k][idx]["y"] = round(best[1], 2)
                moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/_fast_out_delivery_default")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--data", default=DEFAULT_DATA_ROOT)
    a = ap.parse_args()
    cases = [x.strip() for x in a.cases.split(",") if x.strip()]
    index = {}
    for c in discover(a.data):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()

    grid = []
    for r in (10.0, 15.0, 20.0, 30.0, 45.0):
        for p in (3.0, 5.0):
            for b in (0.0, 8.0, 16.0, 25.0):
                if b == 0.0 and (r, p) != (10.0, 3.0) and (r, p) not in ((45.0, 5.0),):
                    pass
                grid.append((r, p, b))

    print("%-26s %8s %8s %8s | %s" % ("arm R/PERP/BACK", "baseF1", "F1", "delta", "逐案 delta"))
    for (r, p, b) in grid:
        dirs = {}
        base_all, new_all = [], []
        per = []
        mv = 0
        for cid in cases:
            case = index.get(cid)
            fp = os.path.join(a.dir, "final_results", "%s_Pred.json" % cid)
            if case is None or not os.path.exists(fp):
                continue
            gt = load_json(case.json_path)
            pred = load_json(fp)
            kb = float(_wirecut.score(gt, pred, cfg)[0].f1)
            ppt = _pin_pts(pred)
            m = snap_doc(pred, ppt, r, p, b)
            mv += m
            kn = float(_wirecut.score(gt, pred, cfg)[0].f1)
            base_all.append(kb)
            new_all.append(kn)
            per.append((cid, round(kn - kb, 4)))
        if not base_all:
            continue
        bb = sum(base_all) / len(base_all)
        nn = sum(new_all) / len(new_all)
        print("%-26s %8.4f %8.4f %8.4f | mv=%-5d %s"
              % ("R%.0f P%.0f B%.0f" % (r, p, b), bb, nn, nn - bb, mv, per))


if __name__ == "__main__":
    main()
