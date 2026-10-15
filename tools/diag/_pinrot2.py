#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""旋转引脚文本的**可读性**普查 + 神谕上界（§5.132 第 0 步）。只测量，不改逻辑。

数据来源（交付/确定性任意一个 run 的中间产物）：
  runs/stage1/<cid>.stage1.json          components[*].pin_evidence[comp_id] = 引脚槽
  runs/stage2/_assigned/<cid>.json       comps[comp_id].pins[str(slot)] = {number, name}
  final_results/<cid>_Pred.json          交付产物
GT 只用于**评估**（§5.132 的纪律：判据必须 GT-blind）。
用法：
  python -X utf8 tools/diag/_pinrot2.py [--run-dir ...] [--cases 1,2,...] [--show]
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.dirname(os.path.dirname(HERE))
if EDA not in sys.path:
    sys.path.insert(0, EDA)

from schematic_parser.data import DEFAULT_DATA_ROOT, discover          # noqa: E402
from schematic_parser.metrics import (                                  # noqa: E402
    EvalConfig, assign_components, rewrite_keys, pin_f1, net_hypergraph_f1,
)

CASES10 = [str(i) for i in range(1, 11)]
DEFAULT_RUN = "Y:/workspace/EDA/work/_fast_out_delivery_default"
HIT = 5.0          # 槽锚点到 GT 脚的「可救」半径（= pin_tol）
NEAR_NET = 12.0    # 槽点到我方网边的「加入该网」半径


def _gk(d):
    """designator 文本 -> 文档里的元件键。"""
    if not isinstance(d, dict):
        return None
    for k in ("canon", "text"):
        v = (d.get(k) or "").strip()
        if v:
            return v
    return None


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _gt_pins(gt, key):
    out = []
    for pk, v in (gt.get("pins") or {}).get(key, {}).items():
        p = v.get("point") or {}
        out.append((str(pk).split("_")[-1], (v.get("pinname") or ""),
                    (float(p.get("x", 0.0)), float(p.get("y", 0.0)))))
    return out


def _our_pins(doc, key):
    out = []
    for pk, v in (doc.get("pins") or {}).get(key, {}).items():
        p = v.get("point") or {}
        out.append((str(pk).split("_")[-1], (v.get("pinname") or ""),
                    (float(p.get("x", 0.0)), float(p.get("y", 0.0)))))
    return out


def _nearest_net(doc, pt, maxd):
    best = (None, None)
    for nk, net in (doc.get("nets") or {}).items():
        for ek, e in (net.get("edges") or {}).items():
            for q in e:
                qq = (float(q.get("x", 0.0)), float(q.get("y", 0.0)))
                d = _dist(pt, qq)
                if best[0] is None or d < best[0]:
                    best = (d, nk)
    if best[0] is not None and best[0] <= maxd:
        return best[1]
    return None


def analyse_case(cid, case, run_dir, cfg, show=False):
    pp = os.path.join(run_dir, "final_results", "%s_Pred.json" % cid)
    s1p = os.path.join(run_dir, "runs", "stage1", "%s.stage1.json" % cid)
    asp = os.path.join(run_dir, "runs", "stage2", "_assigned", "%s.json" % cid)
    for p in (pp, s1p, asp):
        if not os.path.exists(p):
            print("!! missing %s" % p)
            return None
    gt = json.load(io.open(case.json_path, encoding="utf-8"))
    pred = json.load(io.open(pp, encoding="utf-8"))
    s1 = json.load(io.open(s1p, encoding="utf-8"))
    asg = json.load(io.open(asp, encoding="utf-8"))
    _, km = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, km)

    base_pin = pin_f1(gt, pred2, cfg, {}).f1
    base_net = net_hypergraph_f1(gt, pred2, cfg, {})[0].f1

    pe = s1.get("pin_evidence") or {}
    slots = []
    for comp in s1.get("components") or []:
        cid1 = comp.get("id")
        if cid1 not in pe:
            continue
        key = _gk(((asg.get("comps") or {}).get(cid1) or {}).get("designator"))
        if not key or key not in (gt.get("pins") or {}):
            continue
        gp = _gt_pins(gt, key)
        op = _our_pins(pred2, key)
        for i, slot in enumerate(pe[cid1]):
            rec = ((asg.get("comps") or {}).get(cid1, {}).get("pins") or {}).get(str(i)) or {}
            num = (rec.get("number") or {}).get("text")
            nam = (rec.get("name") or {}).get("text")
            nsrc = (rec.get("name") or {}).get("src")
            pt = slot.get("point_img") or [0.0, 0.0]
            pt = (float(pt[0]), float(pt[1]))
            # 图像坐标 -> doc 坐标
            h = s1["image"]["h"] if isinstance(s1.get("image"), dict) else None
            doc_pt = (pt[0], (h - pt[1]) if h is not None else pt[1])
            # GT 真值：同元件内最近脚
            gnear = None
            if gp:
                d, cand = min(((_dist(doc_pt, g[2]), g) for g in gp), key=lambda t: t[0])
                if d <= HIT:
                    gnear = (d, cand)
            # 我方已发的脚：该槽位附近
            onear = None
            if op:
                d, cand = min(((_dist(doc_pt, o[2]), o) for o in op), key=lambda t: t[0])
                if d <= 3.0:
                    onear = (d, cand)
            slots.append(dict(slot=i, key=key, edge=slot.get("edge"),
                              trust=slot.get("trust"), conf=slot.get("confidence"),
                              reason=slot.get("reason"),
                              num=num, nam=nam, nsrc=nsrc, doc_pt=doc_pt,
                              gnear=gnear, onear=onear))
    if show:
        for s in slots:
            g = s["gnear"]
            print("  %-5s slot%-2d %-6s trust=%-9s conf=%.2f num=%-4r name=%-8r src=%-5s | GT %s"
                  % (s["key"], s["slot"], s["edge"], s["trust"], s["conf"] or 0.0,
                     s["num"], s["nam"], s["nsrc"],
                     ("none" if not g else "%s/%s d=%.1f" % (g[1][0], g[1][1], g[0]))))
    return dict(case=cid, slots=slots, gt=gt, base_doc=pred2,
                base_pin=base_pin, base_net=base_net, cfg=cfg)


def _apply_oracle(base_doc, pending):
    """把「可救」的槽按 GT 的号+名发到**槽自身坐标**，并把成员挂到槽附近的我方网上。"""
    doc = json.loads(json.dumps(base_doc))
    pins = doc.setdefault("pins", {})
    nets = doc.get("nets") or {}
    for (key, gnum, gname, dpt, _edge) in pending:
        pins.setdefault(key, {})
        for pk in list(pins.get(key, {}).keys()):
            q = (pins[key][pk].get("point") or {})
            if _dist((float(q.get("x", 0.0)), float(q.get("y", 0.0))), dpt) <= 3.0:
                if str(pk).split("_")[-1] != str(gnum):
                    pins[key].pop(pk, None)
        pins[key]["pin_%s" % gnum] = {
            "pinname": gname,
            "point": {"x": dpt[0], "y": dpt[1]},
        }
        nk = _nearest_net(doc, dpt, NEAR_NET)
        if nk and nk in nets:
            hg = nets[nk].get("hyperGraph") or ""
            m = "%s.%s" % (key, gnum)
            toks = [t for t in hg.strip().strip("()").split(",") if t.strip()]
            if m not in toks:
                toks.append(m)
            nets[nk]["hyperGraph"] = "(%s)" % ",".join(toks)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cases", default=",".join(CASES10))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    index = {str(c.case_id): c for c in discover(a.data_root)}
    cfg = EvalConfig()

    print("== 引脚槽普查：文本读出情况（按边） ==")
    print("case  边    槽数 号空 号错 名空 名错 名src=rot* | 槽距GT<=5 可救")
    tot = dict(n=0, num_missing=0, num_wrong=0, name_missing=0, name_wrong=0, rot=0,
               hit=0, hit_bad=0)
    by_edge = {}
    o_pin_sum = o_net_sum = b_pin_sum = b_net_sum = 0.0
    t_pin_sum = t_net_sum = 0.0
    tb_share_n = 0
    rows = []
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case = index.get(cid)
        if case is None:
            continue
        r = analyse_case(cid, case, a.run_dir, cfg, a.show)
        if r is None:
            continue
        agg = {}
        pending = []
        for s in r["slots"]:
            e = s["edge"] or "?"
            d = agg.setdefault(e, dict(n=0, nm=0, nw=0, mm=0, mw=0, rot=0, hit=0, hb=0))
            d["n"] += 1
            tot["n"] += 1
            g = s["gnear"]
            num_bad = (not s["num"])
            name_bad = (not s["nam"])
            if num_bad:
                d["nm"] += 1
                tot["num_missing"] += 1
            elif g and str(s["num"]).strip() != str(g[1][0]):
                d["nw"] += 1
                tot["num_wrong"] += 1
            if name_bad:
                d["mm"] += 1
                tot["name_missing"] += 1
            elif g and str(s["nam"]).strip().upper() != str(g[1][1]).strip().upper():
                d["mw"] += 1
                tot["name_wrong"] += 1
            if s["nsrc"] and str(s["nsrc"]).startswith("rot"):
                d["rot"] += 1
                tot["rot"] += 1
            if g:
                d["hit"] += 1
                tot["hit"] += 1
                bad = num_bad or name_bad or (
                    g and (str(s["num"]).strip() != str(g[1][0])
                           or str(s["nam"]).strip().upper() != str(g[1][1]).strip().upper()))
                if bad:
                    d["hb"] += 1
                    tot["hit_bad"] += 1
                    pending.append((s["key"], g[1][0], g[1][1], s["doc_pt"], e))
        for e in sorted(agg.keys()):
            d = agg[e]
            print("%-5s %-4s %3d %4d %4d %4d %4d %5d | %6d %6d"
                  % (cid, e, d["n"], d["nm"], d["nw"], d["mm"], d["mw"], d["rot"],
                     d["hit"], d["hb"]))
            by_edge.setdefault(e, dict(n=0, nm=0, nw=0, mm=0, mw=0, rot=0, hit=0, hb=0))
            for k in ("n", "nm", "nw", "mm", "mw", "rot", "hit", "hb"):
                by_edge[e][k] += d[k]
        doc_all = _apply_oracle(r["base_doc"], pending)
        pending_tb = [p for p in pending if p[4] in ("top", "bottom")]
        doc_tb = _apply_oracle(r["base_doc"], pending_tb)
        op = pin_f1(r["gt"], doc_all, cfg, {}).f1
        on = net_hypergraph_f1(r["gt"], doc_all, cfg, {})[0].f1
        tbp = pin_f1(r["gt"], doc_tb, cfg, {}).f1
        tbn = net_hypergraph_f1(r["gt"], doc_tb, cfg, {})[0].f1
        rows.append((cid, r["base_pin"], op, r["base_net"], on, tbp, tbn))
        b_pin_sum += r["base_pin"]
        o_pin_sum += op
        b_net_sum += r["base_net"]
        o_net_sum += on
        t_pin_sum += tbp
        t_net_sum += tbn
        tb_share_n += len(pending_tb)
    print()
    print("== 合计（按边） ==")
    print("边    槽数  号空  号错  名空  名错  rot重读胜 | 槽距GT<=5  可救")
    for e in sorted(by_edge.keys()):
        d = by_edge[e]
        print("%-5s %4d %5d %5d %5d %5d %9d | %8d %5d"
              % (e, d["n"], d["nm"], d["nw"], d["mm"], d["mw"], d["rot"], d["hit"], d["hb"]))
    print("总计  %4d %5d %5d %5d %5d %9d | %8d %5d"
          % (tot["n"], tot["num_missing"], tot["num_wrong"], tot["name_missing"],
             tot["name_wrong"], tot["rot"], tot["hit"], tot["hit_bad"]))
    print()
    print("== 神谕：把「可救」槽按 GT 的号+名发到槽自身坐标（锚点不变） ==")
    print("case   pin base    pin all     dPin      pin T/B     dPinTB    | "
          "netH base   netH all    dNetH     netH T/B    dNetHTB")
    dp = dn = dtp = dtn = 0.0
    for (cid, bp, op, bn, on, tp, tnx) in rows:
        dp += op - bp
        dn += on - bn
        dtp += tp - bp
        dtn += tnx - bn
        print("%-5s %8.4f  %8.4f  %+8.4f  %8.4f  %+8.4f | %8.4f  %8.4f  %+8.4f  %8.4f  %+8.4f"
              % (cid, bp, op, op - bp, tp, tp - bp, bn, on, on - bn, tnx, tnx - bn))
    n = max(1, len(rows))
    print("均值  %8.4f  %8.4f  %+8.4f  %8.4f  %+8.4f | %8.4f  %8.4f  %+8.4f  %8.4f  %+8.4f"
          % (b_pin_sum / n, o_pin_sum / n, dp / n, t_pin_sum / n, dtp / n,
             b_net_sum / n, o_net_sum / n, dn / n, t_net_sum / n, dtn / n))
    print("分数[全部可救]  dPin = %+.2f 分 (x0.25)   dNetH = %+.2f 分 (x0.35)   合计 %+.2f 分"
          % (25.0 * dp / n, 35.0 * dn / n, 25.0 * dp / n + 35.0 * dn / n))
    print("分数[仅T/B旋转]  dPin = %+.2f 分 (x0.25)   dNetH = %+.2f 分 (x0.35)   合计 %+.2f 分"
          % (25.0 * dtp / n, 35.0 * dtn / n, 25.0 * dtp / n + 35.0 * dtn / n))
    print("可救槽中位于 top/bottom（旋转边）的条数 = %d / %d" % (tb_share_n, tot["hit_bad"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
