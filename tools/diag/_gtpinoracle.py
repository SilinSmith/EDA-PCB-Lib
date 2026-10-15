# -*- coding: utf-8 -*-
"""神谕：如果**我们的引脚位置是对的**（把导线端点搬到同名 GT 引脚），导线通道能拿多少？

动机（§5.121 续）：case 1 排针扇出区量到——
  GT   U1.pin_48 = (1334.0, 204.0)，GT 导线边 (1336,203)->(1432,203)，左端距该 GT 引脚 2.2px；
  我方 U1.pin_48 = (1326.0, 202.8)，我方导线边 (1429,202)->(1328,202.8)，左端距**我方**引脚 2px。
⇒ 我方导线端点**挂在我方（错的）引脚上**，偏 8~10px；netL 的亏损可能主要是**引脚位置误差的投影**。

本脚本量两件事（官方 10 张）：
  1) 同名引脚的位置误差分布（我方 vs GT）；
  2) 神谕分数：把「贴在我方引脚 3px 内的导线端点」搬到**同名 GT 引脚**后重算 net_line_f1。
     （这是「引脚位置满分、其余不变」的上界，不是可交付规则。）
"""
import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _wireendtrim import apply_moves, edge_ends_img, score   # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, net_line_f1, normalise_doc  # noqa: E402


def pins_img(doc, h):
    out = {}
    for ck, v in (doc.get("pins") or {}).items():
        if not isinstance(v, dict):
            continue
        for pk, p in v.items():
            pt = p.get("point") if isinstance(p, dict) else None
            if isinstance(pt, dict) and "x" in pt:
                out[(ck, pk)] = (float(pt["x"]), h - float(pt["y"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--attach", type=float, default=3.0)
    a = ap.parse_args(argv)
    cfg = EvalConfig()
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
        gt_doc = load_json(case.json_path)
        pred_doc = load_json(fp)
        base, pred2, pairs = score(gt_doc, pred_doc, cfg)
        h = None
        # 从 pcore 拿图高（y 翻转用）
        from pipeline import pcore, prun
        inp = prun.inputs_of(pcore.find_case(cid))
        h = inp.h
        gpin = pins_img(normalise_doc(gt_doc), h)
        ppin = pins_img(pred2, h)
        # 1) 同名引脚误差
        ds = [float(np.hypot(gpin[k][0] - ppin[k][0], gpin[k][1] - ppin[k][1]))
              for k in gpin if k in ppin]
        ds = np.asarray(ds)
        # 2) 神谕：导线端点贴在我方引脚上 ⇒ 搬到同名 GT 引脚
        ends = edge_ends_img(pred2, h)
        pp_arr = list(ppin.items())
        moves = {}
        n_att = 0
        for p, _q, hd in ends:
            best = None
            for k, q in pp_arr:
                dd = float(np.hypot(p[0] - q[0], p[1] - q[1]))
                if best is None or dd < best[0]:
                    best = (dd, k)
            if best is None or best[0] > a.attach:
                continue
            n_att += 1
            g = gpin.get(best[1])
            if g is not None:
                moves[hd] = g
        pd = copy.deepcopy(pred2)
        n_moved = apply_moves(pd, moves, h)
        sc = net_line_f1(normalise_doc(gt_doc), pd, cfg, pairs, {})
        rows.append({"case": cid, "base": base.f1, "gt": sc.f1, "n_att": n_att,
                     "n_moved": n_moved, "n_pin": len(gpin), "n_same": int(ds.size),
                     "med": float(np.median(ds)) if ds.size else float("nan"),
                     "p90": float(np.percentile(ds, 90)) if ds.size else float("nan"),
                     "le5": int((ds <= 5).sum()) if ds.size else 0})
        print("case %-3s 基线 %.4f  引脚 %3d(同名 %3d, 中位偏 %.1f p90 %.1f, <=5px %d)  "
              "端点贴我方引脚 %3d  神谕(搬到同名 GT 引脚) %.4f (%+.4f, mv%d)"
              % (cid, base.f1, len(gpin), ds.size, rows[-1]["med"], rows[-1]["p90"],
                 rows[-1]["le5"], n_att, sc.f1, sc.f1 - base.f1, n_moved))
    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        mg = sum(r["gt"] for r in rows) / n
        same = sum(r["n_same"] for r in rows)
        le5 = sum(r["le5"] for r in rows)
        print("\n== 均值（%d 张）==" % len(rows))
        print("   基线导线 F1                     %.4f" % mb)
        print("   神谕：引脚满分（搬到同名 GT 引脚） %.4f  (%+.4f)  ⇒ 加权 %+.2f 分"
              % (mg, mg - mb, (mg - mb) * 10.0))
        print("   同名引脚总 %d 个，<=5px %d (%.1f%%)，>5px %d"
              % (same, le5, 100.0 * le5 / max(same, 1), same - le5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
