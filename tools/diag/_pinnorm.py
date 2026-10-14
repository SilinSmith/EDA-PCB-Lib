# -*- coding: utf-8 -*-
"""引脚点的**法向约定**取证：GT 与预测的脚点各落在框边多远？

对每个 case，把 GT 与预测（`assign_components` + `rewrite_keys` 对齐后）里
type=box 元件的引脚点，按「最近框边」归边，量**带符号的法向外距**
（正 = 在框外，0 = 恰在框边），然后逐案/逐边比较：
  · GT 的法向外距分布 —— 揭示 GT 的约定（贴边？外推 12px？外推 22px？）
  · 预测的法向外距分布 —— 揭示我方的约定
  · 同键配对的 (pred - gt) 差值 —— 直接就是判分器第④关的超差量

用法：
    python -X utf8 work\\_pinnorm.py --run gsd9 [--cases 1,5,8] [--pins]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, rewrite_keys)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--pins", action="store_true", help="逐脚打印")
a = ap.parse_args()

CFG = EvalConfig()


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
    """(边名, 带符号外距)——取最近的一条边。"""
    x0, y0, x1, y1 = box
    cands = [("x-", x0 - p[0]), ("x+", p[0] - x1), ("y-", y0 - p[1]), ("y+", p[1] - y1)]
    name, signed = max(cands, key=lambda kv: kv[1])
    return name, signed


cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    if c.split == "10GT":
        cases[cid] = c

print("run = %s" % a.run)
hdr = ("%-4s | %-28s | %-28s | %s" % (
    "case", "GT 外距 中位/均值/n", "pred 外距 中位/均值/n", "逐边 (gt中位→pred中位 | 同键差中位,n)"))
print(hdr)
print("-" * len(hdr))
g_all, p_all, d_all = [], [], []
for cid in a.cases.split(","):
    if cid not in cases:
        continue
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    box = {k for k, v in gt["components"].items()
           if canon_type((v or {}).get("type")) in IC_TYPES}
    pp = ROOT / "work" / ("_fast_out_%s" % a.run) / "final_results" / ("%s_Pred.json" % cid)
    if not pp.exists():
        print("%-4s  （缺预测）" % cid)
        continue
    pred = normalise_doc(json.load(open(str(pp), encoding="utf-8")))
    _, keymap = assign_components(gt, pred, CFG, {})
    pred = rewrite_keys(pred, keymap)

    def collect(doc):
        out = {}
        for c, group in (doc.get("pins") or {}).items():
            if c not in box or not isinstance(group, dict):
                continue
            bb = box_of((doc.get("components") or {}).get(c))
            if not bb:
                continue
            for pk, info in group.items():
                p = _pt((info or {}).get("point"))
                if not p:
                    continue
                name, signed = edge_dist(p, bb)
                out[(c, pin_key_to_number(pk))] = (name, signed, p,
                                                   (info or {}).get("pinname"))
        return out

    gf, pf = collect(gt), collect(pred)
    gs = [v[1] for v in gf.values()]
    ps = [v[1] for v in pf.values()]
    g_all += gs
    p_all += ps
    by_edge = {}
    diffs = []
    for key, gv in gf.items():
        pv = pf.get(key)
        if not pv or pv[0] != gv[0]:
            continue
        by_edge.setdefault(gv[0], [[], []])
        by_edge[gv[0]][0].append(gv[1])
        by_edge[gv[0]][1].append(pv[1])
        diffs.append(pv[1] - gv[1])
        d_all.append(pv[1] - gv[1])
    # 逐对误差分解：法向（外距差） vs 切向（沿边方向的位置差）
    nt = []
    for key, gv in gf.items():
        pv = pf.get(key)
        if not pv:
            continue
        dx, dy = pv[2][0] - gv[2][0], pv[2][1] - gv[2][1]
        edist = (dx * dx + dy * dy) ** 0.5
        if gv[0].startswith("x"):
            dn, dt = dx, dy
        else:
            dn, dt = dy, dx
        nt.append((edist, dn, dt, key))
    bad = [x for x in nt if x[0] > CFG.pin_tol]
    fmt = (lambda v: "%.1f/%.1f/%d" % (statistics.median(v), statistics.mean(v), len(v))
           if v else "-")
    edges = " ".join("%s %.0f→%.0f" % (e, statistics.median(v[0]), statistics.median(v[1]))
                     for e, v in sorted(by_edge.items()))
    print("%-4s | %-28s | %-28s | %s | 同键差 中位 %s n=%d" % (
        cid, fmt(gs), fmt(ps), edges,
        "%.1f" % statistics.median(diffs) if diffs else "-", len(diffs)))
    if bad:
        print("     ↳ 超差 %d/%d：|误差|中位 %.1f；法向|分量|中位 %.1f（>5 的 %d）· "
              "切向|分量|中位 %.1f（>5 的 %d）" % (
                  len(bad), len(nt), statistics.median([x[0] for x in bad]),
                  statistics.median([abs(x[1]) for x in bad]),
                  sum(1 for x in bad if abs(x[1]) > 5),
                  statistics.median([abs(x[2]) for x in bad]),
                  sum(1 for x in bad if abs(x[2]) > 5)))
    if a.pins:
        for key in sorted(gf):
            gv, pv = gf[key], pf.get(key)
            print("      %s.%-7s gt=(%s,%s) %s gt外%s %.1f  pred=%s" % (
                key[0], key[1], "%.0f" % gv[2][0], "%.0f" % gv[2][1], gv[0], gv[0], gv[1],
                ("(%s,%s) %s 外%s %.1f" % ("%.0f" % pv[2][0], "%.0f" % pv[2][1], pv[0], pv[0], pv[1]))
                if pv else "MISS"))
print("-" * len(hdr))
print("汇总 GT 外距 中位 %.1f 均值 %.1f n=%d | pred 外距 中位 %.1f 均值 %.1f n=%d"
      % (statistics.median(g_all), statistics.mean(g_all), len(g_all),
         statistics.median(p_all), statistics.mean(p_all), len(p_all)))
if d_all:
    print("同键 (pred-gt) 外距差：中位 %.2f 均值 %.2f | |差|<=5 的占 %.1f%% | 分布 %s"
          % (statistics.median(d_all), statistics.mean(d_all),
             100.0 * sum(1 for x in d_all if abs(x) <= 5) / len(d_all),
             sorted(round(x) for x in d_all)))
