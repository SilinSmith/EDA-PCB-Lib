# -*- coding: utf-8 -*-
"""锚点可行性普查：GT 的「脚点外距」有多少带标注、能不能按（元件,边）预测？

三问：
  A) 有标注的规模 —— 池子（200_train_cases）里 GT 的 box 引脚有多少条？
     这是「用池子 GT 训练逐边外距预测器」的可行性前提。
  B) 结构 —— GT 的外距是不是按（元件, 边）成组恒定？组内极差分布如何？
     若成组恒定，预测问题就从「逐脚」降成「逐元件逐边」（≈4 个标量/元件）。
  C) 奖金 —— 离线改产物做三把神谕，量化「知道外距」值多少分：
       常数（逐案中位） / 逐（元件,边）中位 / 逐脚精确；
     只挪脚点，不改键、不改名、不动网表；用官方判分器重评（秒级）。

用法：
    python -X utf8 work\\_anchorfit.py                       # A + B（10GT + 池子）
    python -X utf8 work\\_anchorfit.py --oracle --run sjndef  # C（官方 10 张）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, evaluate_case, rewrite_keys)

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="10GT,200_train_cases")
ap.add_argument("--run", default="sjndef")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--oracle", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="池子只取前 N 张（0=全取）")
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
    """(边名, 带符号外距) —— 取最近的一条边；正 = 框外。"""
    x0, y0, x1, y1 = box
    cands = [("x-", x0 - p[0]), ("x+", p[0] - x1), ("y-", y0 - p[1]), ("y+", p[1] - y1)]
    return max(cands, key=lambda kv: kv[1])


def groups_of(doc):
    """{(comp, side): [(offset, key)]} —— 只取 box 元件。"""
    out = defaultdict(list)
    for c, group in (doc.get("pins") or {}).items():
        if canon_type(((doc.get("components") or {}).get(c) or {}).get("type")) not in IC_TYPES:
            continue
        if not isinstance(group, dict):
            continue
        bb = box_of((doc.get("components") or {}).get(c))
        if not bb:
            continue
        for pk, info in group.items():
            p = _pt((info or {}).get("point"))
            if not p:
                continue
            side, off = edge_dist(p, bb)
            out[(c, side)].append((off, (c, pin_key_to_number(pk)), p, bb))
    return out


# ------------------------------------------------------------------ A + B
def survey():
    want = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    all_cases = discover(DEFAULT_DATA_ROOT)
    per_split = defaultdict(list)
    for c in all_cases:
        if c.split in want:
            per_split[c.split].append(c)
    print("数据源 %s" % DEFAULT_DATA_ROOT)
    for split in want:
        cases = per_split.get(split) or []
        if a.limit and split != "10GT":
            cases = cases[:a.limit]
        n_case_pins, n_comp, offsets, spreads, modal = 0, 0, [], [], Counter()
        n_grp = 0
        for c in cases:
            if not c.json_path:
                continue
            gt = normalise_doc(json.load(open(str(c.json_path), encoding="utf-8")))
            g = groups_of(gt)
            if not g:
                continue
            comps = {k[0] for k in g}
            n_comp += len(comps)
            for (comp, side), items in g.items():
                n_grp += 1
                n_case_pins += len(items)
                offs = [it[0] for it in items]
                offsets += offs
                spreads.append(max(offs) - min(offs))
                modal[round(statistics.median(offs))] += 1
        if not offsets:
            print("%-16s 无 box 引脚" % split)
            continue
        hist = Counter(round(o) for o in offsets)
        const = sum(1 for s in spreads if s <= 1.0)
        print("%-16s 案例 %d ｜ box 元件 %d ｜ box 引脚 %d ｜ (元件,边) 组 %d" % (
            split, len(cases), n_comp, n_case_pins, n_grp))
        print("%-16s 外距 直方 %s" % ("", " ".join("%+d:%d" % kv for kv in sorted(hist.items()))))
        print("%-16s 外距 中位 %.1f 均值 %.1f ｜ 组内极差≤1px 的组 %d/%d (%.1f%%) ｜ 组中位众数 %s" % (
            "", statistics.median(offsets), statistics.mean(offsets),
            const, len(spreads), 100.0 * const / len(spreads),
            " ".join("%d:%d" % kv for kv in modal.most_common(6))))
        print("%-16s 组内极差 中位 %.1f p90 %.1f max %.1f" % (
            "", statistics.median(spreads),
            sorted(spreads)[int(0.9 * (len(spreads) - 1))], max(spreads)))


# ------------------------------------------------------------------ C
def oracle():
    """只动脚点几何，**键映射固定**（用 baseline 的 keymap），直接调 pin_f1 重评。

    上一版用 evaluate_case 重评会在改写后的 doc 上**再跑一次元件匹配**，
    键被二次改写 ⇒ 数字全废（case 3 逐边掉到 0.5）。这里显式只用 pin_f1。
    """
    from schematic_parser.metrics import pin_f1 as PIN_F1
    by_cid = {}
    for c in discover(DEFAULT_DATA_ROOT):
        if c.split == "10GT":
            by_cid[str(c.case_id)] = c
    print("run = %s（只改 pred 脚点，键映射固定，≈总分增量 = Δ引脚F1均值 × 25）" % a.run)
    hdr = "%-4s | %-24s | %-16s | %-16s | %s" % (
        "case", "基线 pinF1 (tp/fp/fn)", "逐案常数", "逐(元件,边)", "逐脚(仅配对)")
    print(hdr)
    print("-" * len(hdr))
    agg = defaultdict(list)
    for cid in a.cases.split(","):
        c = by_cid.get(cid)
        if not c:
            continue
        pp = ROOT / "work" / ("_fast_out_%s" % a.run) / "final_results" / ("%s_Pred.json" % cid)
        if not pp.exists():
            print("%-4s （缺预测 %s）" % (cid, pp))
            continue
        gt = normalise_doc(json.load(open(str(c.json_path), encoding="utf-8")))
        pred0 = normalise_doc(json.load(open(str(pp), encoding="utf-8")))
        _, keymap = assign_components(gt, pred0, CFG, {})
        pred = rewrite_keys(pred0, keymap)
        base = PIN_F1(gt, pred, CFG, {})

        ggrp = groups_of(gt)
        pgrp = groups_of(pred)
        gt_pt = {}
        for comp, group in (gt.get("pins") or {}).items():
            for pk, info in (group or {}).items():
                p = _pt((info or {}).get("point"))
                if p:
                    gt_pt[(comp, pin_key_to_number(pk))] = p
        case_off = None
        allg = [it[0] for items in ggrp.values() for it in items]
        if allg:
            case_off = statistics.median(allg)

        def moved(mode):
            doc = json.loads(json.dumps(pred))
            for (comp, side), items in pgrp.items():
                gitems = ggrp.get((comp, side)) or []
                tgt = None
                if mode == "case":
                    tgt = case_off
                elif mode == "side" and gitems:
                    tgt = statistics.median([it[0] for it in gitems])
                for off, key, p, bb in items:
                    if mode == "pin":
                        newp = gt_pt.get((comp, key[1]))
                        if not newp:
                            continue
                    else:
                        if tgt is None:
                            continue
                        # 该脚是否落在这条边上（避免把角上的脚推错轴）
                        side_now, off_now = edge_dist(p, bb)
                        if side_now != side:
                            continue
                        # 约定：offset = 框边到点的**朝外**距离。
                        # x-/y-（左/上）边：offset = x0 - p ⇒ 外距变大点坐标变小 ⇒ 反向。
                        sgn = -1.0 if side.endswith("-") else 1.0
                        d = (tgt - off) * sgn
                        newp = (p[0] + d, p[1]) if side.startswith("x") else (p[0], p[1] + d)
                    for pk, info in (doc.get("pins") or {}).get(comp, {}).items():
                        if pin_key_to_number(pk) == key[1]:
                            info["point"] = {"x": newp[0], "y": newp[1]}
                            break
            return PIN_F1(gt, doc, CFG, {})

        res = {"base": base}
        for mode in ("case", "side", "pin"):
            res[mode] = moved(mode)
        for tag, r in res.items():
            agg[tag].append(r)
        print("%-4s | %.4f (%d/%d/%d)%s | %.4f (%+.4f) | %.4f (%+.4f) | %.4f (%+.4f)" % (
            cid, base.f1, base.tp, base.fp, base.fn, " " * 3,
            res["case"].f1, res["case"].f1 - base.f1,
            res["side"].f1, res["side"].f1 - base.f1,
            res["pin"].f1, res["pin"].f1 - base.f1))
    print("-" * len(hdr))
    for tag, label in (("base", "基线"), ("case", "逐案常数"), ("side", "逐(元件,边)"), ("pin", "逐脚(配对)")):
        rr = agg[tag]
        if not rr:
            continue
        m = statistics.mean([r.f1 for r in rr])
        print("%-10s 引脚 F1 均值 %.4f ｜ tp=%d fp=%d fn=%d ｜ 隐含总分增量 %+.2f" % (
            label, m, sum(r.tp for r in rr), sum(r.fp for r in rr), sum(r.fn for r in rr),
            (m - statistics.mean([r.f1 for r in agg["base"]])) * 25.0))


survey()
if a.oracle:
    print()
    oracle()
