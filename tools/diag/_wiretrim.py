# -*- coding: utf-8 -*-
"""离线 A/B：把「越过引脚的线头」往回收（trim back），而不是向前吸（snap forward）。

`_linenet.py` 把 case 1 最差那个网摊开后，病根写得很清楚（这是一个排针扇出区）：
  GT  (1336,291)->(1432,291) len  96      ← GT 线端落在引脚处（GT 端离引脚 ~2px）
  P08 (1329,262)->(1433,262) len 104      ← 我方线端**越过引脚 5~8px**（端1 离最近脚点 5.4px）
⇒ `_seg_distance` 取两端最大距离、`line_tol = 5.0`（`schematic_parser/metrics.py:88`）
⇒ 一端偏 8px 就整条判 fp+fn。case 1 这种「越界 5~12px」的横向走线，一个网里就有 6 条。

关键：现有 `stage2/assemble.py:1023 snap_ends_to_pins` **只能向前搜**（`t <= 0` 直接跳过），
所以「引脚落在端点**后面**」的越界情形它永远救不了；而它被记成判负（RESULTS §5.76：
`STAGE2_PIN_SNAP` 默认 0，ps45p8 官方 10 张 −0.34）用的正是 R=45 的**向前**搜。

本探针只动已交付文档，用**我方自己的引脚点**（不是 GT 引脚）做三种规则：
  --mode fwd   复刻现实现（t>0，向前）—— 用来校验离线口径（应复现 ≈ −0.34）
  --mode back  只允许「引脚在端点后方 0.5~R px」（越过 → 往回收）
  --mode both  两侧都允许
并扫 R × PERP。只改 `nets[*].edges` 端点，网配对沿用基线。

用法：
    python -X utf8 tools\\diag\\_wiretrim.py --mode both --rs 6,8,10,12 --perps 3,5,8
"""
from __future__ import print_function

import argparse
import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover   # noqa: E402
from schematic_parser.metrics import (EvalConfig, _edge_points, net_line_f1,
                                      normalise_doc)                       # noqa: E402
from _dotsnap import score                                                 # noqa: E402


def pred_pin_points(doc):
    """交付文档里的引脚点（doc 坐标）。结构：`pins[comp][pin] = {"point": {"x","y"}}`
    —— 注意 `_wirecut.pin_points` 对这个结构取不到点（它按 `pins[k].point` 取）。"""
    pts = []
    for _ck, pkv in (doc.get("pins") or {}).items():
        if not isinstance(pkv, dict):
            continue
        for _pk, pv in pkv.items():
            pt = (pv or {}).get("point") if isinstance(pv, dict) else None
            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                pts.append((float(pt["x"]), float(pt["y"])))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts.append((float(pt[0]), float(pt[1])))
    for comp in (doc.get("components") or {}).values():
        pkv = comp.get("pins")
        if isinstance(pkv, dict):
            for pv in pkv.values():
                pt = (pv or {}).get("point") if isinstance(pv, dict) else pv
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    pts.append((float(pt["x"]), float(pt["y"])))
    return pts


def snap_variant(pred2, pins_img, R, P, mode, h):
    """返回 (改过的 doc, 计数)。pins_img/pred 端点都在图像坐标下算，写回 doc 坐标。"""
    doc = copy.deepcopy(pred2)
    moved = {"fwd": 0, "back": 0}
    for _nkey, net in (doc.get("nets") or {}).items():
        edges = net.get("edges") or {}
        segs = []
        for k, v in edges.items():
            pts = _edge_points(v) or []
            if len(pts) < 2:
                continue
            segs.append((k, [(float(p[0]), h - float(p[1])) for p in pts]))
        if not segs:
            continue
        verts = [(p[0], p[1]) for _k, img in segs for p in (img[0], img[-1])]
        for k, img in segs:
            a = np.asarray(img[0], dtype=float)
            b = np.asarray(img[-1], dtype=float)
            d = b - a
            L = float(np.hypot(d[0], d[1]))
            if L < 1e-6:
                continue
            u = d / L
            for idx, p in ((0, a), (1, b)):
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) <= 3.0
                       for q in verts if (q[0], q[1]) != (p[0], p[1])):
                    continue                      # 结点，不动（复刻现实现）
                s = u if idx == 1 else -u
                best = None
                for q in pins_img:
                    v = q - p
                    t = float(v[0] * s[0] + v[1] * s[1])
                    if mode == "fwd" and not (0.0 < t <= R):
                        continue
                    if mode == "back" and not (-R <= t <= -0.5):
                        continue
                    if mode == "both" and abs(t) > R:
                        continue
                    if abs(v[0] * (-s[1]) + v[1] * s[0]) > P:
                        continue
                    if best is None or abs(t) < abs(best[0]):
                        best = (t, q)
                if best is None:
                    continue
                pt = best[1]
                new = [[float(img[j][0]), float(img[j][1])] for j in range(len(img))]
                new[idx] = [float(pt[0]), float(pt[1])]
                edges[k] = [[float(q[0]), h - float(q[1])] for q in new]
                moved["back" if best[0] < 0 else "fwd"] += 1
    return doc, moved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--mode", default="both", choices=["fwd", "back", "both"])
    ap.add_argument("--rs", default="6,8,10,12,45")
    ap.add_argument("--perps", default="3,5,8")
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    rs = [float(x) for x in a.rs.split(",") if x.strip()]
    ps = [float(x) for x in a.perps.split(",") if x.strip()]
    from PIL import Image  # noqa: E402
    rows = []
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        case = index.get(cid)
        fp = os.path.join(d, "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            print("case %s 缺 GT 或产物" % cid)
            continue
        h = Image.open(case.png_path).size[1]
        gt_doc = load_json(case.json_path)
        gt_norm = normalise_doc(gt_doc)
        pred_doc = load_json(fp)
        base, pred2, pairs = score(gt_doc, pred_doc, cfg)
        pins = pred_pin_points(pred2)
        pins_img = np.asarray([[p[0], h - p[1]] for p in pins], dtype=float) \
            if len(pins) else np.zeros((0, 2))
        rec = {"case": cid, "base": base.f1, "npin": len(pins), "v": {}}
        for R in rs:
            for P in ps:
                doc, mv = snap_variant(pred2, pins_img, R, P, a.mode, h)
                sc = net_line_f1(gt_norm, doc, cfg, pairs, {})
                rec["v"][(R, P)] = (sc.f1, mv)
        rows.append(rec)
        line = "case %-3s 基线 %.4f  我方引脚 %3d" % (cid, rec["base"], rec["npin"])
        for R in rs:
            for P in ps:
                f1, mv = rec["v"][(R, P)]
                line += "  R%-3gP%-2g %.4f(%+.4f;前%d后%d)" % (
                    R, P, f1, f1 - rec["base"], mv["fwd"], mv["back"])
        print(line)
    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张；mode=%s）--" % (len(rows), a.mode))
        print("   基线                            %.4f" % mb)
        for R in rs:
            for P in ps:
                m = sum(r["v"][(R, P)][0] for r in rows) / n
                mvf = sum(r["v"][(R, P)][1]["fwd"] for r in rows)
                mvb = sum(r["v"][(R, P)][1]["back"] for r in rows)
                print("   R<=%-4g PERP<=%-3g              %.4f  (%+.4f)  ⇒ 加权 %+.2f 分"
                      "  [前移 %d / 回收 %d]" % (R, P, m, m - mb, (m - mb) * 10.0, mvf, mvb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
