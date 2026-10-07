"""A/B：把「占据整张图的大连通分量」当成图纸边框/标题栏表格线删掉，值不值？

动机：stage1_wire_attrib.py 显示假阳性里最大的一坨是 frame（占总长 26.5%，到最近 GT 线
中位距离 190px）—— 就是图纸外框 + 标题栏表格。它之所以躲过 min_extent，是因为它本身
就是一个横跨整图的大分量；它之所以躲过 reconcile 的 anchor 规则，是因为标题栏里**有文字**
（"Schematic53"/"嘉立创EDA"…），于是被判成"有锚点"。

所以判据不能靠锚点，要靠**几何跨度**：分量的包围盒几乎盖住整张图 ⇒ 它是图框，不是导线。

用法: python work\\_frame_ab.py [stage1产出目录]
"""
import collections
import glob
import json
import os
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))

from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline  # noqa: E402
from stage1.reconcile import wire_graph  # noqa: E402

ARMS = [("base", (9.9, 9.9)), ("span>=0.50", (0.50, 0.50)),
        ("span>=0.60", (0.60, 0.60)), ("span>=0.70", (0.70, 0.70)),
        ("span>=0.80", (0.80, 0.80)), ("span>=0.60&len>=200", (0.60, 0.60))]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage1_all9"
    tol = 5.0
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    agg = {a[0]: np.zeros(3) for a in ARMS}       # tp, fp, fn
    n = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        cid = d["case"]
        case = cases.get(cid)
        if case is None:
            for k, v in cases.items():
                if k.lstrip("0") == cid.lstrip("0"):
                    case, cid = v, k
                    break
        if case is None:
            continue
        n += 1
        H = float(d["image"]["h"]); W = float(d["image"]["w"])
        gt = json.load(open(case.json_path, encoding="utf-8"))
        gsegs = []
        for net in gt["nets"].values():
            for pts in (net.get("edges") or {}).values():
                for i in range(len(pts) - 1):
                    gsegs.append([[float(pts[i]["x"]), H - float(pts[i]["y"])],
                                  [float(pts[i + 1]["x"]), H - float(pts[i + 1]["y"])]])
        # 产物里的 segments 是 polyline 列表，先摊平成 N×4 的 (x1,y1,x2,y2)
        ALL = seg_arrays(d["wire_detail"]["segments"])
        flat = [[[float(ALL[i, 0]), float(ALL[i, 1])], [float(ALL[i, 2]), float(ALL[i, 3])]]
                for i in range(ALL.shape[0])]
        groups, seg_of = wire_graph(flat, tol=2.0)
        # 每个分量的包围盒
        bb = {}
        for gi, g in enumerate(groups):
            xs, ys = [], []
            for si in g["segs"]:
                s = ALL[si]
                xs += [s[0], s[2]]; ys += [s[1], s[3]]
            bb[gi] = (max(xs) - min(xs), max(ys) - min(ys))
        GA = seg_arrays(gsegs)
        for name, (fw, fh) in ARMS:
            keep = [si for si in range(ALL.shape[0])
                    if not (bb[seg_of[si]][0] >= fw * W and bb[seg_of[si]][1] >= fh * H)]
            PA = ALL[keep] if keep else np.zeros((0, 4))
            if PA.shape[0]:
                pp = np.concatenate([sample_polyline(PA[i, :2], PA[i, 2:], 16) for i in range(PA.shape[0])])
                po = np.concatenate([np.full(16, i) for i in range(PA.shape[0])])
                pd = dist_pts_to_segs(pp, GA)
                pw = np.full(PA.shape[0], 0.0)
                np.maximum.at(pw, po, pd)
            else:
                pw = np.array([])
            tp = int((pw <= tol).sum()); fp = int(PA.shape[0]) - tp
            if GA.shape[0]:
                gp = np.concatenate([sample_polyline(GA[i, :2], GA[i, 2:], 16) for i in range(GA.shape[0])])
                go = np.concatenate([np.full(16, i) for i in range(GA.shape[0])])
                gd = dist_pts_to_segs(gp, PA) if PA.shape[0] else np.full(gp.shape[0], 1e9)
                gw = np.full(GA.shape[0], 0.0)
                np.maximum.at(gw, go, gd)
                fn = int((gw > tol).sum())
            else:
                fn = 0
            agg[name] += np.array([tp, fp, fn], dtype=float)

    print(f"stage1 产物 {n} 份   tol={tol}px")
    print(f'\n  {"臂":18s} {"TP":>7s} {"FP":>7s} {"FN":>7s} {"R":>7s} {"P":>7s} {"F1":>7s}')
    for name, _ in ARMS:
        tp, fp, fn = agg[name]
        R = tp / (tp + fn) if tp + fn else 0
        P = tp / (tp + fp) if tp + fp else 0
        F = 2 * P * R / (P + R) if P + R else 0
        print(f'  {name:18s} {tp:7.0f} {fp:7.0f} {fn:7.0f} {R:7.4f} {P:7.4f} {F:7.4f}')


if __name__ == "__main__":
    main()
