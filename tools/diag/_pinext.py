# -*- coding: utf-8 -*-
"""§5.112 取证：**GT 的导线端点在引脚点的外侧多少 px？**

motif（case 2，x=576，R2 的下端）：GT 引脚 R2.pin_2 = (575.0,672.0)，GT 的导线边端 = (576,684)
⇒ **沿「离开元件」方向超出引脚点 12px**；我方引脚 (576.4,674.2)、我方导线端 (576,676)（≈ 就在引脚点上）。
若这个 +12 是**符号库约定**（引脚长度为 20/其它），它就能解释导线通道 `near` 类（226 条）的
「差 6~12px」——也能解释为什么「把端点到引脚点」的三种改法全亏（方向相反）。

本探针统计：每条 GT 边的每个端点，找最近的 GT 引脚点 p（≤ --maxd），取元件 bbox 中心到 p 的
单位向量 u 为「向外」方向，统计 **s = (端 − p)·u** 的分布（>0 = 在引脚点外侧）。
同时对**我方**产物做同样的统计（对照：我方 s 应该 ≈ 0，因为我方端 = 我方引脚点）。

用法（cwd = Y:\\workspace\\EDA\\eda-pcb）：
    python -X utf8 ..\\work\\_pinext.py --run work/_fast_out_gsd9
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--maxd", type=float, default=40.0)
ap.add_argument("--show", type=int, default=12)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()


def bbox_of(comp):
    for key in ('bbox', 'bbox_img', 'box', 'rect'):
        v = comp.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            b = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
    return None


def gather(doc, tag):
    pins = []
    for comp, group in (doc.get('pins') or {}).items():
        if not isinstance(group, dict):
            continue
        bb = bbox_of((doc.get('components') or {}).get(comp) or {})
        for pk, v in group.items():
            p = M._point((v or {}).get('point'))
            if p:
                pins.append((str(comp), p, bb))
    hist = Counter()
    vals = []
    ex = []
    for nn, net in (doc.get('nets') or {}).items():
        if not isinstance(net, dict):
            continue
        for ek, ev in (net.get('edges') or {}).items():
            if not isinstance(ev, list) or len(ev) < 2:
                continue
            a, b = M._point(ev[0]), M._point(ev[1])
            if not a or not b:
                continue
            for end, other in ((a, b), (b, a)):
                best = None
                for comp, p, bb in pins:
                    d = math.hypot(p[0] - end[0], p[1] - end[1])
                    if d <= args.maxd and (best is None or d < best[0]):
                        best = (d, comp, p, bb)
                if best is None:
                    continue
                d, comp, p, bb = best
                if bb is None:
                    cx, cy = p
                else:
                    cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
                ux, uy = p[0] - cx, p[1] - cy
                n = math.hypot(ux, uy)
                if n < 1e-6:
                    continue
                ux, uy = ux / n, uy / n
                s = (end[0] - p[0]) * ux + (end[1] - p[1]) * uy       # 沿向外方向
                t = -(end[0] - p[0]) * uy + (end[1] - p[1]) * ux      # 切向（侧偏）
                hist[int(round(s))] += 1
                vals.append(s)
                if len(ex) < args.show:
                    ex.append("%s %-10s %-8s pin(%.0f,%.0f) end(%.0f,%.0f) s=%+.1f t=%+.1f d=%.1f"
                              % (tag, nn, comp, p[0], p[1], end[0], end[1], s, t, d))
    vals.sort()
    print("== %s : n=%d  mean s=%+.2f  median %+.1f  p25 %+.1f p75 %+.1f"
          % (tag, len(vals), (sum(vals) / len(vals)) if vals else 0.0,
             vals[len(vals) // 2] if vals else 0.0,
             vals[len(vals) // 4] if vals else 0.0,
             vals[3 * len(vals) // 4] if vals else 0.0))
    print("   直方图 s:", "  ".join("%+d:%d" % (k, hist[k]) for k in sorted(hist)))
    for e in ex:
        print("     " + e)


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
for cid in sel:
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    _, km = M.assign_components(gt, pred, cfg, {})
    pred = M.rewrite_keys(pred, km)
    print("---- case %s" % cid)
    gather(gt, "GT  ")
    gather(pred, "PRED")
