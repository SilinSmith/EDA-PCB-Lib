# -*- coding: utf-8 -*-
"""离线 A/B：在交付文档的导线上「补切点」能不能拿回 §5.120 量到的 19.1%。

`work/_wiresplit.py` 量到：639 条 GT 导线里 **122 条（19.1%）** 我方段**跨过**了 GT 的切点
（`overspan` 12 + `cut1` 110）——线画到了、只是没在那里断开。本探针不动流水线，
只在**已交付文档**上做后处理再评分：

  --mode tjoin   切点来源 = 我方自己的预测：其它交付段的端点落在本段内部（T 接）+ 我方脚点
  --mode oracle  切点来源 = GT：把 GT 同名网的端点投到本段内部（**上界**，不可交付）
  --mode both    两种都算，便于看「自己的 T 接」离上界有多远

评分口径与判分器一致（`net_hypergraph_f1` 给出 pairs，再算 `net_line_f1`），
只改 `nets[*].edges`，所以键映射不变。
"""
from __future__ import print_function

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb\tools\eval")
from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover       # noqa: E402
from schematic_parser.metrics import (EvalConfig, _edge_points, assign_components,
                                      net_hypergraph_f1, net_line_f1,
                                      normalise_doc, rewrite_keys)              # noqa: E402


def flip(pts, h):
    return [(float(p[0]), h - float(p[1])) for p in pts]


def all_edges(doc):
    out = []
    for nkey, net in (doc.get("nets") or {}).items():
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge) or []
            if len(pts) >= 2:
                out.append((nkey, ekey, pts))
    return out


def pin_points(doc):
    pts = []
    for comp in (doc.get("components") or {}).values():
        pins = comp.get("pins")
        if isinstance(pins, dict):
            for p in pins.values():
                pt = (p or {}).get("point") if isinstance(p, dict) else p
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append((float(pt[0]), float(pt[1])))
    for p in (doc.get("pins") or {}).values():
        pt = (p or {}).get("point") if isinstance(p, dict) else p
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((float(pt[0]), float(pt[1])))
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def t_param(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 1e-12:
        return None, None
    t = float((p - a) @ ab) / L2
    proj = a + ab * t
    return t, float(np.hypot(*(proj - p)))


def split_nets(doc, cuts, h, tol=2.0):
    """按 cuts（图像坐标点集）把每条边在 t∈(0.02,0.98) 且垂距 ≤tol 处切断。"""
    n_split = 0
    n_added = 0
    for net in (doc.get("nets") or {}).values():
        edges = net.get("edges") or {}
        new = {}
        for ekey, edge in edges.items():
            pts = _edge_points(edge) or []
            if len(pts) < 2 or len(cuts) == 0:
                new[ekey] = edge
                continue
            # 交付边都是 2 点；多点时按折线逐段切
            cur = [np.asarray(q, dtype=float) for q in pts]
            pieces = []
            for i in range(len(cur) - 1):
                a, b = cur[i], cur[i + 1]
                ts = []
                for c in cuts:
                    t, d = t_param(c, a, b)
                    if t is None or d is None or d > tol:
                        continue
                    if 0.02 < t < 0.98:
                        ts.append(t)
                if not ts:
                    pieces.append((a, b))
                    continue
                ts.sort()
                prev = a
                for t in ts:
                    q = a + (b - a) * t
                    if float(np.hypot(*(q - prev))) > 1.0:
                        pieces.append((prev, q))
                        prev = q
                if float(np.hypot(*(b - prev))) > 1.0:
                    pieces.append((prev, b))
            if len(pieces) == 1:
                new[ekey] = edge
                continue
            n_split += 1
            for k, (u, v) in enumerate(pieces):
                # 交付文档坐标是 y 向上 ⇒ 翻回 doc 坐标写回
                new["%s#%d" % (ekey, k)] = [[float(u[0]), h - float(u[1])],
                                            [float(v[0]), h - float(v[1])]]
            n_added += len(pieces) - 1
        net["edges"] = new
    return n_split, n_added


def score(gt_doc, pred_doc, cfg):
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    detail = {}
    _, keymap = assign_components(gt, pred, cfg, detail)
    pred2 = rewrite_keys(pred, keymap)
    _net, pairs = net_hypergraph_f1(gt, pred2, cfg, detail)
    line = net_line_f1(gt, pred2, cfg, pairs, detail)
    return line, pred2, pairs


def net_endpoints(pred2, pairs, gt_norm):
    """返回 {pred_net_key: [cut points(image)]}：同网切点。

    `own`   = 该网内**我方**其它段的端点（真 T 接/T 形分叉）；
    `gt`    = 该网**匹配到的 GT 网**的段端点（上界）。
    """
    p_nets = list(pred2["nets"].items())
    g_nets = list(gt_norm["nets"].items())
    own, gtc = {}, {}
    for i, j in pairs:
        pk = p_nets[i][0]
        pts = []
        for _e, edge in (p_nets[i][1].get("edges") or {}).items():
            pl = _edge_points(edge) or []
            if len(pl) >= 2:
                pts.append(np.asarray(pl[0], dtype=float))
                pts.append(np.asarray(pl[-1], dtype=float))
        own[pk] = pts
        g = []
        for _e, edge in (g_nets[j][1].get("edges") or {}).items():
            pl = _edge_points(edge) or []
            if len(pl) >= 2:
                g.append(np.asarray(pl[0], dtype=float))
                g.append(np.asarray(pl[-1], dtype=float))
        gtc[pk] = g
    return own, gtc


def split_nets_pernet(doc, cuts_by_net, h, tol=2.0):
    """按「每个网自己的切点表」切段。"""
    n_split = n_add = 0
    for net in (doc.get("nets") or {}).values():
        nkey = None
        for k, v in (doc.get("nets") or {}).items():
            if v is net:
                nkey = k
                break
        cuts = cuts_by_net.get(nkey)
        if not cuts:
            continue
        arr = np.asarray([[float(q[0]), h - float(q[1])] for q in cuts], dtype=float) \
            if cuts else np.zeros((0, 2))
        new = {}
        for ekey, edge in (net.get("edges") or {}).items():
            pts = _edge_points(edge) or []
            if len(pts) < 2 or len(arr) == 0:
                new[ekey] = edge
                continue
            cur = [np.asarray(q, dtype=float) for q in pts]
            pieces = []
            for i in range(len(cur) - 1):
                a, b = cur[i], cur[i + 1]
                ts = []
                for c in arr:
                    t, dd = t_param(c, a, b)
                    if t is None or dd is None or dd > tol:
                        continue
                    if 0.02 < t < 0.98:
                        ts.append(t)
                if not ts:
                    pieces.append((a, b))
                    continue
                ts.sort()
                prev = a
                for t in ts:
                    q = a + (b - a) * t
                    if float(np.hypot(*(q - prev))) > 1.0:
                        pieces.append((prev, q))
                        prev = q
                if float(np.hypot(*(b - prev))) > 1.0:
                    pieces.append((prev, b))
            if len(pieces) == 1:
                new[ekey] = edge
                continue
            n_split += 1
            for k, (u, v) in enumerate(pieces):
                new["%s#%d" % (ekey, k)] = [[float(u[0]), h - float(u[1])],
                                            [float(v[0]), h - float(v[1])]]
            n_add += len(pieces) - 1
        net["edges"] = new
    return n_split, n_add


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--mode", default="both", choices=["tjoin", "gtnet", "both"])
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args(argv)

    from PIL import Image  # noqa: E402
    cfg = EvalConfig(line_tol=a.tol)
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    rows = []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        h = Image.open(case.png_path).size[1]
        gt_doc = load_json(case.json_path)
        pred_doc = load_json(fp)
        gt_norm = normalise_doc(gt_doc)
        base, pred2, pairs = score(gt_doc, pred_doc, cfg)
        own, gtc = net_endpoints(pred2, pairs, gt_norm)
        out = {"case": cid, "base": base.f1}
        det = {}
        if a.mode in ("tjoin", "both"):
            pd = copy.deepcopy(pred2)
            ns, na = split_nets_pernet(pd, own, h)
            sc = net_line_f1(gt_norm, pd, cfg, pairs, det)
            out["tjoin"], out["tjoin_split"], out["tjoin_added"] = sc.f1, ns, na
        if a.mode in ("gtnet", "both"):
            pd = copy.deepcopy(pred2)
            ns, na = split_nets_pernet(pd, gtc, h)
            sc = net_line_f1(gt_norm, pd, cfg, pairs, det)
            out["gtnet"], out["gtnet_split"] = sc.f1, ns
        rows.append(out)
        line = "case %-3s 基线 %.4f" % (cid, out["base"])
        if "tjoin" in out:
            line += "  同网 T 接 %.4f (%+.4f) [切 %d 段→+%d]" % (
                out["tjoin"], out["tjoin"] - out["base"], out["tjoin_split"], out["tjoin_added"])
        if "gtnet" in out:
            line += "  同网 GT 切点 %.4f (%+.4f) [切 %d]" % (
                out["gtnet"], out["gtnet"] - out["base"], out["gtnet_split"])
        print(line)
    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张）--" % len(rows))
        print("   基线                  %.4f" % mb)
        if any("tjoin" in r for r in rows):
            mt = sum(r["tjoin"] for r in rows) / n
            print("   同网 T 接切点（可交付）%.4f  (%+.4f)  ⇒ 加权 %+.2f 分"
                  % (mt, mt - mb, (mt - mb) * 10.0))
        if any("gtnet" in r for r in rows):
            mo = sum(r["gtnet"] for r in rows) / n
            print("   同网 GT 切点（上界）   %.4f  (%+.4f)  ⇒ 加权 %+.2f 分"
                  % (mo, mo - mb, (mo - mb) * 10.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
