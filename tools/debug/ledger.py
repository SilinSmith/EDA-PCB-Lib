# -*- coding: utf-8 -*-
"""逐条对账：把本地重建评分器 `metrics.py` 的打分过程**一条一条摊开**。

本地重建评分器的 `assign_components` / `pin_f1` / `net_hypergraph_f1` / `net_line_f1`
只返回 `tp/fp/fn` 三个计数，看不到「到底是哪一条错了、为什么错」。
本模块把这些函数的判定逻辑**原样重写一遍**，但每判一条就记一条明细，
最后拿本地 `evaluate_case` 的计数做**自校验**（不一致会在报告里报警）。

用法：
    from ledger import build_ledger
    led = build_ledger(gt_doc, pred_doc, "0001")

`led` 是一个普通 dict（可直接 `json.dumps`）。
"""
from __future__ import annotations

import math
from dataclasses import asdict
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from schematic_parser.metrics import (  # noqa: E402
    IC_TYPES, VALUE_TYPES, WEIGHTS, EvalConfig, F1,
    _assign, _box, _centre, _edge_points, _point, _seg_distance,
    bbox_ok, canon_type, component_text_ok, component_pairs, evaluate_case,
    flat_scored_pins, _net_sets, net_hypergraph_f1,
    match_nets, net_pin_set, normalise_doc, pin_key_to_number,
)

# 四个指标的中文名与「这一项为什么会丢分」的一句话解释
METRIC_META = {
    "component": ("元件", 0.30, "键 / 类型 / 外框 / 文字 四道关，任何一道不过就 **fp+1 且 fn+1**（一笔亏三处）"),
    "pin":       ("引脚", 0.25, "只算 GT 里 type=box 的元件；引脚号必须存在、`pinname` 全等、点距不超过配置的引脚容差（默认20px）"),
    "net_hypergraph": ("网络（超图）", 0.35, "普通元件按元件、IC按引脚比较；默认按配对网络集合的软F1计分"),
    "net_line":  ("网络导线", 0.10, "同一网络内逐条导线段配几何，端点 unordered 距离不超过配置的端点容差（默认20px） 算一对"),
}


def _f1_of(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _gain(weight, tp, fp, fn, dtp, dfp, dfn):
    """把某类错误全部修好能涨多少分（百分制）。"""
    base = _f1_of(tp, fp, fn)
    new = _f1_of(tp + dtp, fp + dfp, fn + dfn)
    return 100.0 * weight * (new - base)


def _dist(a, b):
    try:
        return round(math.hypot(a[0] - b[0], a[1] - b[1]), 2)
    except Exception:
        return None


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return int(f) if f == int(f) else round(f, 2)


# ---------------------------------------------------------------------------
# 元件


def ledger_components(gt, pred, cfg):
    """镜像 `assign_components`，但每一步都留痕。"""
    gc, pc = gt["components"], pred["components"]
    rows = []

    def ctype_of(rec):
        t = (rec or {}).get("type")
        return canon_type(t) if cfg.canon_types else t

    pred_used, gt_used = set(), set()
    pairs = []

    pairs = component_pairs(gt, pred, cfg)
    gt_used = {g for g, _, _ in pairs}
    pred_used = {p for _, p, _ in pairs}
    for key in sorted(set(gc) - gt_used):
        r = gc[key] or {}
        rows.append({"verdict": "missing", "stage": "元件匹配", "gt_key": key,
                     "gt_type": ctype_of(r), "gt_value": r.get("value"),
                     "gt_name": r.get("Name"), "gt_bbox": _box(r.get("bbox")),
                     "pred_key": None, "pred_type": None, "pred_value": None,
                     "pred_name": None, "pred_bbox": None,
                     "reason": "未找到满足身份与几何条件的元件；电源/地还须类别和值一致"})
    for key in sorted(set(pc) - pred_used):
        r = pc[key] or {}
        rows.append({"verdict": "extra", "stage": "元件匹配", "gt_key": None,
                     "gt_type": None, "gt_value": None, "gt_name": None, "gt_bbox": None,
                     "pred_key": key, "pred_type": ctype_of(r), "pred_value": r.get("value"),
                     "pred_name": r.get("Name"), "pred_bbox": _box(r.get("bbox")),
                     "reason": "预测元件没有满足条件的GT对应；不以电源自动编号判定身份"})
    n_missing_key = sum(r["verdict"] == "missing" for r in rows)
    n_extra_key = sum(r["verdict"] == "extra" for r in rows)
    coord_matched = sum(how != "key" for _, _, how in pairs)

    # ---- 3) 逐对判定
    for gk, pk, how in pairs:
        g, p = gc.get(gk), pc.get(pk)
        if g is None or p is None:
            continue
        tg, tp_ = ctype_of(g), ctype_of(p)
        rec = {"stage": "逐对判定（%s）" % ({"key": "键匹配", "coord": "坐标匹配", "power": "电源类别/值/几何匹配"}[how]),
               "gt_key": gk, "pred_key": pk, "gt_type": tg, "pred_type": tp_,
               "gt_value": g.get("value"), "pred_value": p.get("value"),
               "gt_name": g.get("Name"), "pred_name": p.get("Name"),
               "gt_bbox": _box(g.get("bbox")), "pred_bbox": _box(p.get("bbox"))}
        if tg != tp_:
            rec.update(verdict="type_mismatch",
                       reason="类型不一致：GT 是 `%s`，我们报成了 `%s`" % (tg, tp_))
        elif bbox_ok(g.get("bbox"), p.get("bbox"), cfg) is not True:
            d = _dist(_centre(_box(g.get("bbox"))), _centre(_box(p.get("bbox"))))
            rec.update(verdict="bbox_mismatch",
                       reason="外框不合格（`bbox_mode=%s`，容差 %.0fpx）；中心距 %s px"
                              % (cfg.bbox_mode, cfg.bbox_tol, d), centre_dist=d)
        elif not component_text_ok(g, p, cfg):
            if tg in IC_TYPES:
                rec.update(verdict="text_mismatch",
                           reason="**器件型号**不一致（`box` 类比 `Name`）：GT `%s`，我们 `%s`"
                                  % (g.get("Name"), p.get("Name")))
            elif tg in VALUE_TYPES:
                rec.update(verdict="text_mismatch",
                           reason="**参数值**不一致（`r/c` 类比 `value`）：GT `%s`，我们 `%s`"
                                  % (g.get("value"), p.get("value")))
            else:
                rec.update(verdict="text_mismatch", reason="文字不一致")
        else:
            rec.update(verdict="tp", reason="键 / 类型 / 外框 / 文字 全过")
            d = _dist(_centre(_box(g.get("bbox"))), _centre(_box(p.get("bbox"))))
            rec["centre_dist"] = d
        rows.append(rec)

    tp = sum(1 for r in rows if r["verdict"] == "tp")
    fp = sum(1 for r in rows if r["verdict"] in ("extra", "type_mismatch",
                                                 "bbox_mismatch", "text_mismatch"))
    fn = sum(1 for r in rows if r["verdict"] in ("missing", "type_mismatch",
                                                 "bbox_mismatch", "text_mismatch"))
    buckets = {}
    for r in rows:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
    stats = {"tp": tp, "fp": fp, "fn": fn, "buckets": buckets,
             "n_gt": len(gc), "n_pred": len(pc), "n_pairs": len(pairs),
             "coord_matched": coord_matched,
             "n_pure_key_missing": n_missing_key, "n_pure_key_extra": n_extra_key,
             "revived_by_coord": coord_matched}
    return rows, stats


# ---------------------------------------------------------------------------
# 引脚


def ledger_pins(gt, pred, cfg):
    box_comps = None
    if cfg.pin_box_only:
        box_comps = {k for k, v in gt["components"].items()
                     if canon_type((v or {}).get("type")) in IC_TYPES}
    # 记录被 pin_box_only 排除掉的 GT 引脚数（解释「为什么分母这么小」）
    skipped = 0
    for comp, group in gt["pins"].items():
        if box_comps is not None and comp not in box_comps and isinstance(group, dict):
            skipped += len(group)

    gf = {k: ("pin_" + k[1], v) for k, v in flat_scored_pins(gt, gt, cfg).items()}
    pf = {k: ("pin_" + k[1], v) for k, v in flat_scored_pins(pred, gt, cfg).items()}

    rows = []
    shared = sorted(set(gf) & set(pf), key=lambda k: (str(k[0]), str(k[1])))
    for key in shared:
        gpk, g = gf[key]
        ppk, p = pf[key]
        comp, num = key
        rec = {"comp": comp, "num": num, "gt_pin_key": gpk, "pred_pin_key": ppk,
               "gt_name": (g or {}).get("pinname"), "pred_name": (p or {}).get("pinname"),
               "gt_point": _point((g or {}).get("point")),
               "pred_point": _point((p or {}).get("point"))}
        if not isinstance(p, dict) or not isinstance(g, dict):
            rec.update(verdict="malformed", reason="结构不是字典")
        elif p.get("pinname") != g.get("pinname"):
            rec.update(verdict="name_mismatch",
                       reason="引脚名不同：GT `%s`，我们 `%s`（`pinname` 要求**严格相等**）"
                              % (g.get("pinname"), p.get("pinname")))
        else:
            a, b = _point(g.get("point")), _point(p.get("point"))
            if a is None or b is None:
                rec.update(verdict="no_point", reason="`point` 缺失")
            else:
                dd = math.hypot(a[0] - b[0], a[1] - b[1])
                rec["dist"] = round(dd, 2)
                if dd > cfg.pin_tol:
                    rec.update(verdict="position_mismatch",
                               reason="引脚名对了，但点距 %.1fpx > 容差 %.0fpx" % (dd, cfg.pin_tol))
                else:
                    rec.update(verdict="tp", reason="引脚名对 + 点距 %.1fpx ≤ %.0fpx" % (dd, cfg.pin_tol))
        rows.append(rec)
    for key in sorted(set(gf) - set(pf), key=lambda k: (str(k[0]), str(k[1]))):
        gpk, g = gf[key]
        rows.append({"comp": key[0], "num": key[1], "gt_pin_key": gpk, "pred_pin_key": None,
                     "gt_name": (g or {}).get("pinname"), "pred_name": None,
                     "gt_point": _point((g or {}).get("point")), "pred_point": None,
                     "verdict": "missing",
                     "reason": "GT 有这颗 IC 的这个引脚号，我们的输出里**没有这个号**"})
    for key in sorted(set(pf) - set(gf), key=lambda k: (str(k[0]), str(k[1]))):
        ppk, p = pf[key]
        rows.append({"comp": key[0], "num": key[1], "gt_pin_key": None, "pred_pin_key": ppk,
                     "gt_name": None, "pred_name": (p or {}).get("pinname"),
                     "gt_point": None, "pred_point": _point((p or {}).get("point")),
                     "verdict": "extra",
                     "reason": "我们输出了这个引脚号，GT 的这颗 IC 里没有"})

    tp = sum(1 for r in rows if r["verdict"] == "tp")
    fp = sum(1 for r in rows if r["verdict"] in ("extra", "malformed", "name_mismatch",
                                                 "no_point", "position_mismatch"))
    fn = sum(1 for r in rows if r["verdict"] in ("missing", "malformed", "name_mismatch",
                                                 "no_point", "position_mismatch"))
    buckets = {}
    for r in rows:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
    return rows, {"tp": tp, "fp": fp, "fn": fn, "buckets": buckets,
                  "n_gt": len(gf), "n_pred": len(pf),
                  "box_comps": sorted(box_comps or []),
                  "gt_pins_outside_box": skipped,
                  "gt_components": sorted({k[0] for k in gf}),
                  "pred_components": sorted({k[0] for k in pf})}


# ---------------------------------------------------------------------------
# 网络


def ledger_nets(gt, pred, cfg):
    g_nets = _net_sets(gt["nets"], gt["components"], cfg.net_pin_collapse_nonbox)
    types = pred["components"] if cfg.legacy_identity_rules else {**pred["components"], **gt["components"]}
    p_nets = _net_sets(pred["nets"], types, cfg.net_pin_collapse_nonbox)
    pairs = match_nets(g_nets, p_nets, soft=cfg.net_soft)
    rows = []
    for i, j in pairs:
        pn, ps = p_nets[i]
        gn, gs = g_nets[j]
        inter = ps & gs
        rec = {"gt_net": gn, "pred_net": pn, "gt_pins": sorted(gs), "pred_pins": sorted(ps),
               "inter": sorted(inter), "n_gt": len(gs), "n_pred": len(ps),
               "n_inter": len(inter)}
        if ps == gs:
            rec.update(verdict="tp", reason="引脚集合**完全相同**（%d 个引脚）" % len(gs))
        elif inter:
            rec.update(verdict="partial",
                       reason="配对上了，但集合不同：交集 %d ｜ 少 %d ｜ 多 %d"
                              % (len(inter), len(gs - ps), len(ps - gs)),
                       missing_pins=sorted(gs - ps), extra_pins=sorted(ps - gs))
        else:
            rec.update(verdict="partial",
                       reason="被匈牙利算法硬配上，但**交集为 0**（两边都没得分）",
                       missing_pins=sorted(gs), extra_pins=sorted(ps))
        rows.append(rec)
    matched_g = {j for _, j in pairs}
    matched_p = {i for i, _ in pairs}
    for j, (gn, gs) in enumerate(g_nets):
        if j not in matched_g:
            rows.append({"gt_net": gn, "pred_net": None, "gt_pins": sorted(gs), "pred_pins": [],
                         "inter": [], "n_gt": len(gs), "n_pred": 0, "n_inter": 0,
                         "verdict": "missing",
                         "reason": "GT 这个网络在我们的输出里**找不到任何对应**（%d 个引脚全丢）" % len(gs)})
    for i, (pn, ps) in enumerate(p_nets):
        if i not in matched_p:
            rows.append({"gt_net": None, "pred_net": pn, "gt_pins": [], "pred_pins": sorted(ps),
                         "inter": [], "n_gt": 0, "n_pred": len(ps), "n_inter": 0,
                         "verdict": "extra",
                         "reason": "我们多出来的网络（%d 个引脚），GT 里没有对应" % len(ps)})

    total_pred = sum(len(s) for _, s in p_nets)
    total_gt = sum(len(s) for _, s in g_nets)
    score, _ = net_hypergraph_f1(gt, pred, cfg)
    tp, fp, fn = score.tp, score.fp, score.fn
    buckets = {}
    for r in rows:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
    return rows, {"tp": tp, "fp": fp, "fn": fn, "buckets": buckets,
                  "n_gt_nets": len(g_nets), "n_pred_nets": len(p_nets),
                  "gt_pins": total_gt, "pred_pins": total_pred,
                  "singleton_pred_nets": sum(1 for _, s in p_nets if len(s) <= 1),
                  "singleton_gt_nets": sum(1 for _, s in g_nets if len(s) <= 1),
                  "matched": len(pairs)}


def ledger_lines(gt, pred, cfg, pairs):
    """`pairs` 用 `ledger_nets` 现算一遍（本地 `net_line_f1` 收的是同一个）。"""
    g_nets = list(gt["nets"].items())
    p_nets = list(pred["nets"].items())
    rows = []
    INF = float("inf")
    matched_p, matched_g = set(), set()
    tp = fp = fn = 0
    for i, j in pairs:
        matched_p.add(i); matched_g.add(j)
        p_edges = [_edge_points(v) for v in (p_nets[i][1].get("edges") or {}).values()]
        g_edges = [_edge_points(v) for v in (g_nets[j][1].get("edges") or {}).values()]
        p_edges = [e for e in p_edges if e is not None]
        g_edges = [e for e in g_edges if e is not None]
        if not p_edges or not g_edges:
            fp += len(p_edges); fn += len(g_edges)
            rows.append({"gt_net": g_nets[j][0], "pred_net": p_nets[i][0],
                         "verdict": "partial",
                         "reason": "一边有导线段、另一边没有：我们 %d 段 / GT %d 段"
                                   % (len(p_edges), len(g_edges))})
            continue
        cost = []
        for pe in p_edges:
            row = []
            for ge in g_edges:
                if len(pe) < 2 or len(ge) < 2:
                    row.append(INF); continue
                d = _seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                row.append(d if d <= cfg.line_tol else INF)
            cost.append(row)
        got = _assign(cost)
        n_tp = len(got); n_fp = len(p_edges) - n_tp; n_fn = len(g_edges) - n_tp
        tp += n_tp; fp += n_fp; fn += n_fn
        rows.append({"gt_net": g_nets[j][0], "pred_net": p_nets[i][0],
                     "verdict": "tp" if n_fp == 0 and n_fn == 0 else "partial",
                     "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
                     "reason": "配对网络（%d/%d 段：%d 对上 ｜ 我们多 %d ｜ 少了 %d）"
                               % (n_tp, len(g_edges), n_tp, n_fp, n_fn)})
    for i, (pn, net) in enumerate(p_nets):
        if i not in matched_p:
            n = len([v for v in (net.get("edges") or {}).values() if _edge_points(v) is not None])
            fp += n
            rows.append({"gt_net": None, "pred_net": pn, "verdict": "extra", "n_fp": n, "n_fn": 0,
                         "reason": "这个网络没配上，%d 段导线全算误检" % n})
    for j, (gn, net) in enumerate(g_nets):
        if j not in matched_g:
            n = len([v for v in (net.get("edges") or {}).values() if _edge_points(v) is not None])
            fn += n
            rows.append({"gt_net": gn, "pred_net": None, "verdict": "missing", "n_fp": 0, "n_fn": n,
                         "reason": "GT 这个网络的 %d 段导线我们一段都没找到对应" % n})
    buckets = {}
    for r in rows:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
    return rows, {"tp": tp, "fp": fp, "fn": fn, "buckets": buckets,
                  "n_gt_edges": sum(len([v for v in (n.get("edges") or {}).values()
                                         if _edge_points(v) is not None])
                                    for _, n in g_nets),
                  "n_pred_edges": sum(len([v for v in (n.get("edges") or {}).values()
                                           if _edge_points(v) is not None])
                                      for _, n in p_nets)}


# ---------------------------------------------------------------------------


def build_ledger(gt_doc, pred_doc, case_id="", cfg=None, split=""):
    cfg = cfg or EvalConfig()
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)

    comp_rows, comp_s = ledger_components(gt, pred, cfg)

    # 网络要在 rewrite_keys 之后的预测上算（和评分器一致）
    from schematic_parser.metrics import rewrite_keys
    # 重新走一遍键匹配拿 keymap（纯函数，无副作用）
    _, keymap = ledger_components_keymap(gt, pred, cfg)
    pred2 = rewrite_keys(pred, keymap)
    pin_rows, pin_s = ledger_pins(gt, pred2, cfg)
    net_rows, net_s = ledger_nets(gt, pred2, cfg)
    _, pairs = net_hypergraph_f1(gt, pred2, cfg)
    line_rows, line_s = ledger_lines(gt, pred2, cfg, pairs)

    parts = []
    for name, stats in (("component", comp_s), ("pin", pin_s),
                        ("net_hypergraph", net_s), ("net_line", line_s)):
        cn, w, why = METRIC_META[name]
        if name == 'pin':
            why = ('只统计box引脚' if cfg.pin_box_only else '历史口径：统计全部引脚') + '；名称、编号严格比较，点距≤%gpx' % cfg.pin_tol
        elif name == 'net_hypergraph':
            why = ('普通元件按整体、IC按pin' if cfg.net_pin_collapse_nonbox else '历史口径：全部按pin') + ('；配对网络软F1' if cfg.net_soft else '；配对网络交集')
        elif name == 'net_line':
            why = '配对网络内比较导线端点，容差%gpx' % cfg.line_tol
        f1 = _f1_of(stats["tp"], stats["fp"], stats["fn"])
        parts.append({
            "key": name, "cn": cn, "weight": w, "why": why,
            "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"],
            "p": round(stats["tp"] / (stats["tp"] + stats["fp"]), 6) if (stats["tp"] + stats["fp"]) else 1.0,
            "r": round(stats["tp"] / (stats["tp"] + stats["fn"]), 6) if (stats["tp"] + stats["fn"]) else 1.0,
            "f1": round(f1, 6), "pts": round(100.0 * w * f1, 4),
            "max_pts": round(100.0 * w, 4),
            "lost": round(100.0 * w * (1.0 - f1), 4),
            "stats": stats,
        })

    # ---- 自校验：和本地 evaluate_case 比
    official = evaluate_case(gt_doc, pred_doc, case_id, cfg)
    check = []
    for p in parts:
        o = getattr(official, p["key"] if p["key"] != "net_hypergraph" else "net_hypergraph")
        ok = all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(
            (o.tp, o.fp, o.fn), (p["tp"], p["fp"], p["fn"])))
        check.append({"metric": p["key"], "ok": ok,
                      "official": [o.tp, o.fp, o.fn], "ledger": [p["tp"], p["fp"], p["fn"]]})

    return {
        "case": case_id, "split": split,
        "score": {"total": round(official.score, 4),
                  "official": round(official.score, 4),
                  "parts": parts},
        "components": comp_rows,
        "pins": pin_rows,
        "nets": net_rows,
        "lines": line_rows,
        "selfcheck": check,
        "config": asdict(cfg),
    }


def ledger_components_keymap(gt, pred, cfg):
    """只取本地 `assign_components` 的 keymap（给 `rewrite_keys` 用）。"""
    from schematic_parser.metrics import assign_components
    return assign_components(gt, pred, cfg)


# ---------------------------------------------------------------------------
# 「修好某类错误能涨多少分」——这是回答「为什么只有这么点分」的核心


def headroom(led):
    """对每个指标算：把某类错误全修好，能涨多少分。"""
    out = []
    for p in led["score"]["parts"]:
        w = p["weight"]
        tp, fp, fn = p["tp"], p["fp"], p["fn"]
        b = p["stats"]["buckets"]
        items = []
        # 一笔亏三处：fp+fn 各减 1，tp+1
        for verdict, cn in (("text_mismatch", "文字不符"), ("bbox_mismatch", "外框不符"),
                            ("type_mismatch", "类型不符")):
            n = b.get(verdict, 0)
            if n:
                items.append({"what": "%s（%d 条）" % (cn, n),
                              "gain": round(_gain(w, tp, fp, fn, n, -n, -n), 3),
                              "note": "fp+fn 一起减、tp 加，一笔亏三处"})
        if b.get("missing", 0):
            n = b["missing"]
            items.append({"what": "补上 %d 个漏掉的（纯 fn）" % n,
                          "gain": round(_gain(w, tp, fp, fn, n, 0, -n), 3),
                          "note": "只减 fn、tp 加"})
        if b.get("extra", 0):
            n = b["extra"]
            items.append({"what": "删掉 %d 个多余的（纯 fp）" % n,
                          "gain": round(_gain(w, tp, fp, fn, 0, -n, 0), 3),
                          "note": "只减 fp"})
        if b.get("position_mismatch", 0):
            n = b["position_mismatch"]
            items.append({"what": "把 %d 个位置偏的引脚挪正" % n,
                          "gain": round(_gain(w, tp, fp, fn, n, -n, -n), 3),
                          "note": "fp+fn 一起减、tp 加"})
        if b.get("name_mismatch", 0):
            n = b["name_mismatch"]
            items.append({"what": "把 %d 个引脚名改对" % n,
                          "gain": round(_gain(w, tp, fp, fn, n, -n, -n), 3),
                          "note": "fp+fn 一起减、tp 加"})
        if b.get("partial", 0) and p["key"] in ("net_hypergraph", "net_line"):
            n = b["partial"]
            items.append({"what": "把 %d 个部分对上的网络补成完全一致" % n,
                          "gain": None, "note": "取决于差几个引脚/几段线，看逐条明细"})
        out.append({"key": p["key"], "cn": p["cn"], "f1": p["f1"], "lost": p["lost"],
                    "items": items})
    return out
