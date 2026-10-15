# -*- coding: utf-8 -*-
"""_nlvis.py — 把「配不上的 GT 段 vs 它最近的 pred 段」画成裁剪图，肉眼看病根。

输出：work\\_nlvis\\<case>_<idx>.png，2x 放大，裁剪窗口以 GT 段为中心。
颜色：绿=GT 所有边  红=我方所有边  黄(粗)=目标 GT 段  青(粗)=它最近的 pred 段
      品红点=GT 引脚  橙点=我方引脚  蓝圈=目标 GT 段端点
用法：python -X utf8 tools\\diag\\_nlvis.py --cases 10 --max 1
"""
from __future__ import print_function

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw                                         # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, _assign, _edge_points, _seg_distance, assign_components,
    net_hypergraph_f1, normalise_doc, rewrite_keys,
)
import _nlerr as NL                                                       # noqa: E402

GREEN = (0, 170, 0)
RED = (220, 30, 30)
YEL = (255, 200, 0)
CYAN = (0, 200, 255)
MAG = (255, 0, 255)
ORG = (255, 130, 0)
BLU = (0, 60, 255)


def records(gt_doc, pred_doc, cfg, tol=3.0, far=30.0):
    gt = normalise_doc(gt_doc)
    pred = normalise_doc(pred_doc)
    _c, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)
    _n, pairs = net_hypergraph_f1(gt, pred2, cfg, {})
    g_nets = list(gt["nets"].items())
    p_nets = list(pred2["nets"].items())
    g_pins = NL._pins_of(gt)
    p_pins = NL._pins_of(pred2)
    g_boxes = NL._boxes_of(gt)
    out = []
    for _pi, _gi in pairs:
        i, j = _gi, _pi
        if i >= len(g_nets) or j >= len(p_nets):
            continue
        ge = NL._edges_of(g_nets[i][1])
        pe = NL._edges_of(p_nets[j][1])
        if not ge or not pe:
            continue
        raw = [[_seg_distance(p, g, cfg.line_unordered) for g in ge] for p in pe]
        cost = [[(v if v <= cfg.line_tol else float("inf")) for v in row] for row in raw]
        got = set(_assign(cost))
        mg = set(gj for _p, gj in got)
        for gi, g in enumerate(ge):
            if gi in mg:
                continue
            bi = min(range(len(pe)), key=lambda k: raw[k][gi])
            best = raw[bi][gi]
            kind = "far"
            if best <= far:
                perp = 0.0
                along = 0.0
                for e in g:
                    pp, oo = NL._perp_along(e, pe[bi][0], pe[bi][1])
                    perp = max(perp, pp)
                    along = max(along, oo)
                if perp > cfg.line_tol:
                    kind = "lateral"
                elif along > cfg.line_tol:
                    kind = "underspan"
                else:
                    kind = "overspan"
            ends = "%s+%s" % (NL._end_class(g[0], g_pins, ge, gi, g_boxes, tol),
                              NL._end_class(g[1], g_pins, ge, gi, g_boxes, tol))
            out.append({"gt": g, "pred": pe[bi], "best": best, "kind": kind,
                        "ends": ends, "net": g_nets[i][0]})
    return out, g_pins, p_pins, gt, pred2


def draw_case(cid, recs, gpins, ppins, gt, pred, h, out_dir, want, maxn, half):
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for n, r in enumerate(recs):
        if want and r["ends"] not in want and r["kind"] not in want:
            continue
        if len(made) >= maxn:
            break
        cx = (r["gt"][0][0] + r["gt"][1][0]) / 2.0
        cy = (r["gt"][0][1] + r["gt"][1][1]) / 2.0
        x0, x1 = int(cx - half), int(cx + half)
        y0, y1 = int(cy - half), int(cy + half)
        S = 2
        im = Image.new("RGB", ((x1 - x0) * S, (y1 - y0) * S), (255, 255, 255))
        dr = ImageDraw.Draw(im)

        def T(p):
            return ((p[0] - x0) * S, (h - p[1] - y0) * S)

        for _k, net in gt["nets"].items():
            for e in NL._edges_of(net):
                dr.line([T(e[0]), T(e[1])], fill=GREEN, width=2)
        for _k, net in pred["nets"].items():
            for e in NL._edges_of(net):
                dr.line([T(e[0]), T(e[1])], fill=RED, width=2)
        for q in gpins:
            a = T(q)
            dr.ellipse([a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3], fill=MAG)
        for q in ppins:
            a = T(q)
            dr.ellipse([a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3], fill=ORG)
        dr.line([T(r["gt"][0]), T(r["gt"][1])], fill=YEL, width=5)
        dr.line([T(r["pred"][0]), T(r["pred"][1])], fill=CYAN, width=3)
        for e in r["gt"]:
            a = T(e)
            dr.ellipse([a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5], outline=BLU, width=2)
        fp = os.path.join(out_dir, "%s_%02d_%s_%s.png"
                          % (cid, n, r["kind"], r["ends"].replace("+", "-")))
        im.save(fp)
        made.append((fp, r["kind"], r["ends"], round(r["best"], 1),
                     [round(v) for e in r["gt"] for v in e],
                     [round(v) for e in r["pred"] for v in e]))
    return made


def dump_at(gt, pred, gpins, ppins, g_boxes, x, y, r):
    def near(p):
        return math.hypot(p[0] - x, p[1] - y) <= r
    print("  -- GT 边（窗口内）--")
    for k, net in gt["nets"].items():
        es = NL._edges_of(net)
        for gi, e in enumerate(es):
            if near(e[0]) or near(e[1]):
                print("     %-10s len %6.1f  (%7.1f,%7.1f)-(%7.1f,%7.1f)  ends=%s+%s"
                      % (k, math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]),
                         e[0][0], e[0][1], e[1][0], e[1][1],
                         NL._end_class(e[0], gpins, es, gi, g_boxes, 3.0),
                         NL._end_class(e[1], gpins, es, gi, g_boxes, 3.0)))
    print("  -- 我方边（窗口内）--")
    for k, net in pred["nets"].items():
        es = NL._edges_of(net)
        for e in es:
            if near(e[0]) or near(e[1]):
                print("     %-10s len %6.1f  (%7.1f,%7.1f)-(%7.1f,%7.1f)"
                      % (k, math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]),
                         e[0][0], e[0][1], e[1][0], e[1][1]))
    print("  -- GT 引脚（窗口内）--")
    for q in gpins:
        if near(q):
            print("     GT  pin (%7.1f,%7.1f)  到窗口心 %.1f"
                  % (q[0], q[1], math.hypot(q[0] - x, q[1] - y)))
    print("  -- 我方引脚（窗口内）--")
    for q in ppins:
        if near(q):
            print("     我们的 pin (%7.1f,%7.1f)  到窗口心 %.1f"
                  % (q[0], q[1], math.hypot(q[0] - x, q[1] - y)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_delivery_default")
    ap.add_argument("--cases", default="10")
    ap.add_argument("--want", default="blank+join,join+blank")
    ap.add_argument("--max", type=int, default=3)
    ap.add_argument("--half", type=int, default=150)
    ap.add_argument("--dump-at", default=None, help="x,y,r：只打印该窗口内的边与引脚")
    ap.add_argument("--out", default=r"Y:\workspace\EDA\work\_nlvis")
    a = ap.parse_args(argv)

    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()
    d = os.path.join(a.run, "final_results")
    want = [w for w in a.want.split(",") if w]
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺" % cid)
            continue
        gt_doc = load_json(case.json_path)
        pred_doc = load_json(fp)
        recs, gp, pp, gt, pred = records(gt_doc, pred_doc, cfg)
        if a.dump_at:
            x, y, r = [float(v) for v in a.dump_at.split(",")]
            print("case %s 窗口 (%.0f,%.0f) r=%.0f" % (cid, x, y, r))
            dump_at(gt, pred, gp, pp, NL._boxes_of(gt), x, y, r)
            continue
        h = int(gt_doc.get("image_height") or gt_doc.get("height") or 0)
        if not h:
            from PIL import Image as I
            h = I.open(case.png_path).size[1] if case.png_path else 1000
        made = draw_case(cid, recs, gp, pp, gt, pred, h, a.out, want, a.max, a.half)
        print("case %s: fn %d 条，画出 %d 张" % (cid, len(recs), len(made)))
        for m in made:
            print("   %s  kind=%s ends=%s best=%.1f  gt=%s pred=%s"
                  % (os.path.basename(m[0]), m[1], m[2], m[3], m[4], m[5]))


if __name__ == "__main__":
    main()
