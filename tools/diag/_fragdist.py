# -*- coding: utf-8 -*-
"""③「拓扑丢」的**可修性度量**：这颗脚离「它 GT 同伴聚集的那条网」的边有多远？

`work/_netwhy.py` 把 ③ 分成三类（没有网 / 过碎 / 并进别的网），但「能不能用一条几何规则
把它接回去」取决于一个数：**这颗脚的坐标，离那条正确网的边（或正确网里同伴的脚）多远**。

本探针按下面的口径逐条打印：
  * 目标网 = 我们产物里**包含该脚 GT 同伴最多**的那条网（≥1 个同伴才算有目标）；
  * `d_edge` = 该脚坐标到目标网所有边线段的最短距离；
  * `d_pin`  = 该脚坐标到目标网所有同伴脚坐标的最短距离；
  * `solo`   = 该脚当前所在网的成员数（0 = 任何网都没有它）。
距离小（≤ 3~6px）⇒ 「按距离吸附」这条规则能救；距离大（>15px）⇒ 中间缺的是**线**（检测问题）。

用法：
    python -X utf8 work/_fragdist.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
DSROOT = EDA / "yolo_part" / "Dataset(waitforpreprocess)"
from schematic_parser import metrics as M                       # noqa: E402

CASES = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
DIR = ROOT / "work" / "_fast_out_cd10on"


def gt_of(case):
    for p in DSROOT.rglob(case + ".json"):
        if p.parent.name == case:
            return json.load(open(p, encoding="utf-8"))
    return None


def hg_members(hg):
    return [x.strip() for x in str(hg or "").strip().strip("()").split(",") if x.strip()]


def seg_dist(px, py, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    t = 0.0 if dd <= 1e-9 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / dd))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def main() -> int:
    cfg = M.EvalConfig()
    rows = []
    for case in CASES:
        g = gt_of(case)
        pf = DIR / "final_results" / (case + "_Pred.json")
        if g is None or not pf.exists():
            continue
        pred = json.load(open(pf, encoding="utf-8"))
        _c, keymap = M.assign_components(g, pred, cfg, None)
        p2 = M.rewrite_keys(pred, keymap)
        part = {}
        for _n, s in M._net_sets(g.get("nets") or {}):
            mems = [m for m in s if isinstance(m, tuple) and len(m) == 2]
            for m in mems:
                part[(str(m[0]), str(m[1]))] = set("%s.%s" % (x, y) for x, y in mems)
        ours = []
        for nid, net in (p2.get("nets") or {}).items():
            mems = set("%s.%s" % (r[0], r[1]) for r in (net.get("_refs") or [])
                       if isinstance(r, (list, tuple)) and len(r) == 2) or set(hg_members(net.get("hyperGraph")))
            edges = []
            for sg in (net.get("edges") or {}).values():
                try:
                    edges.append(((float(sg[0]["x"]), float(sg[0]["y"])),
                                  (float(sg[1]["x"]), float(sg[1]["y"]))))
                except Exception:                                # noqa: BLE001
                    pass
            ours.append((nid, mems, edges))
        where = defaultdict(set)
        for nid, mems, _e in ours:
            for m in mems:
                where[m].add(nid)
        # 每颗脚坐标
        pts = {}
        for ck, pv in (p2.get("pins") or {}).items():
            for pk, v in (pv or {}).items():
                pt = (v or {}).get("point")
                if pt:
                    pts["%s.%s" % (ck, str(pk).replace("pin_", ""))] = (float(pt["x"]), float(pt["y"]))
        # GT 成员里 ③ 的（键号都在我们 pins 里、却不在正确网）
        for (k, n), partners in part.items():
            name = "%s.%s" % (k, n)
            if name not in pts or not partners:
                continue
            # 已拿对？（出现在与 GT 网配对的网里）
            if where.get(name):
                # 粗判：只要它出现在与它同伴重叠最多的那条网里就算拿对
                best = max((len(partners & mems) for _nid, mems, _e in ours), default=0)
                cur = max((len(partners & mems) for nid, mems, _e in ours if nid in where[name]), default=0)
                if cur and cur >= best:
                    continue
            bestj, bestov = None, -1
            for nid, mems, edges in ours:
                ov = len(partners & mems)
                if ov > bestov:
                    bestj, bestov = (nid, mems, edges), ov
            if bestj is None or bestov < 1:
                continue
            nid, mems, edges = bestj
            px, py = pts[name]
            de = min((seg_dist(px, py, a, b) for a, b in edges), default=1e9)
            dp = min((math.hypot(px - q[0], py - q[1]) for m2, q in pts.items() if m2 in mems),
                     default=1e9)
            solo = min((len(m) for n2, m, _e in ours if n2 in where.get(name, ())), default=0)
            rows.append((case, name, solo, len(partners), bestov, de, dp, nid))
    print("%-5s %-12s %5s %6s %6s %8s %8s  %s" %
          ("case", "成员", "现网", "GT同伴", "目标网∩", "d_edge", "d_pin", "目标网"))
    for r in sorted(rows, key=lambda x: min(x[5], x[6])):
        print("%-5s %-12s %5d %6d %6d %8.1f %8.1f  %s" % r)
    print()
    for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)):
        n = [r for r in rows if lo <= min(r[5], r[6]) < hi]
        print("  最近距离 ∈ [%4g,%4g)：%2d 条" % (lo, hi, len(n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
