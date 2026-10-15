# -*- coding: utf-8 -*-
"""离线 A/B：自由端沿自身轴**延伸到元件框边界**（框是准的，引脚点不是）。

三条本地点锚点已被量死：吸结点圆点 +0.24（3 张，case 1 为 0）、端点聚类 +0.24（同样 case 1 为 0）、
引脚吸附/回收 **负**（本目录 `_wiretrim.py`，正与 RESULTS §5.76 的 −0.34 同向）。
但 `_linenet.py` 的坐标摊开显示 case 1 的排针扇出区是另一种病：

  GT  (1336,291)->(1432,291) len  96   ← GT 端正好在**元件框边/焊盘**上
  P08 (1433,262)->(1329,262) len 104   ← 我方线头停在框边**外 7px**（最近 GT 脚点 5.4，圆点 103.8）
  P04/P07/P12/P13 同病：相差 7~12px

关键差别：**元件框几乎是对的**（官方 10 张 component_f1 = 1.0000、空产物也能拿 25 分那几个
case 之外全中），而**引脚点是自己算的**（pin_f1 0.8166，`_pinsnap.py` 证明位置常偏）
⇒ 用「框」当锚点，而不是用「引脚点」。

规则：自由端（同网别的边端点不在 3px 内）沿自身轴向前 R px 内若碰到**任一元件框边界**，
就把端点搬到该交点。只改 `nets[*].edges`，网配对沿用基线。

用法：
    python -X utf8 tools\\diag\\_wirebox.py --rs 6,8,10,12,16,20
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


def boxes_img(doc, h):
    """元件框（图像坐标）：[(x0,y0,x1,y1)]"""
    out = []
    for comp in (doc.get("components") or {}).values():
        b = comp.get("bbox")
        if not b or len(b) < 4:
            continue
        v = [float(x) for x in b[:4]]
        x0, x1 = sorted((v[0], v[2]))
        y0, y1 = sorted((v[1], v[3]))
        out.append((x0, h - y1, x1, h - y0))
    return out


def ray_box_t(p, s, box):
    """从 p 沿 s 的射线与轴对齐框边界的最短正交点距离；无交点返回 None。"""
    x0, y0, x1, y1 = box
    cand = []
    for val, axis in ((x0, 0), (x1, 0), (y0, 1), (y1, 1)):
        if abs(s[axis]) < 1e-9:
            continue
        t = (val - p[axis]) / s[axis]
        if t <= 0:
            continue
        q = (p[0] + s[0] * t, p[1] + s[1] * t)
        if axis == 0:
            if y0 - 0.5 <= q[1] <= y1 + 0.5:
                cand.append((t, q))
        else:
            if x0 - 0.5 <= q[0] <= x1 + 0.5:
                cand.append((t, q))
    if not cand:
        return None
    cand.sort(key=lambda r: r[0])
    return cand[0]


def extend_variant(pred2, boxes, R, h):
    doc = copy.deepcopy(pred2)
    moved = 0
    for _nkey, net in (doc.get("nets") or {}).items():
        edges = net.get("edges") or {}
        segs = []
        for k, v in edges.items():
            pts = _edge_points(v) or []
            if len(pts) < 2:
                continue
            segs.append((k, [(float(p[0]), float(p[1])) for p in pts]))
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
            hit_any = False
            new = [[float(img[j][0]), float(img[j][1])] for j in range(len(img))]
            for idx, p in ((0, a), (1, b)):
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) <= 3.0
                       for q in verts if (q[0], q[1]) != (p[0], p[1])):
                    continue                       # 结点，不动
                s = u if idx == 1 else -u
                best = None
                for box in boxes:
                    r = ray_box_t(p, s, box)
                    if r is None or r[0] > R:
                        continue
                    if best is None or r[0] < best[0]:
                        best = r
                if best is None:
                    continue
                new[idx] = [float(best[1][0]), float(best[1][1])]
                hit_any = True
                moved += 1
            if hit_any:
                edges[k] = [[float(q[0]), h - float(q[1])] for q in new]
    return doc, moved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_sjndef")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--rs", default="6,8,10,12,16,20")
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    d = os.path.join(a.run, "final_results")
    rs = [float(x) for x in a.rs.split(",") if x.strip()]
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
        boxes = boxes_img(pred2, h)
        rec = {"case": cid, "base": base.f1, "nbox": len(boxes), "v": {}}
        for R in rs:
            doc, mv = extend_variant(pred2, boxes, R, h)
            sc = net_line_f1(gt_norm, doc, cfg, pairs, {})
            rec["v"][R] = (sc.f1, mv)
        rows.append(rec)
        line = "case %-3s 基线 %.4f  框 %3d" % (cid, rec["base"], rec["nbox"])
        for R in rs:
            f1, mv = rec["v"][R]
            line += "  R%-3g %.4f(%+.4f,mv%d)" % (R, f1, f1 - rec["base"], mv)
        print(line)
    if rows:
        n = float(len(rows))
        mb = sum(r["base"] for r in rows) / n
        print("\n-- 均值（导线 F1，%d 张）--" % len(rows))
        print("   基线                        %.4f" % mb)
        for R in rs:
            m = sum(r["v"][R][0] for r in rows) / n
            mv = sum(r["v"][R][1] for r in rows)
            print("   延伸到框 R<=%-4g (可交付)   %.4f  (%+.4f)  ⇒ 加权 %+.2f 分  [移动 %d 端点]"
                  % (R, m, m - mb, (m - mb) * 10.0, mv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
