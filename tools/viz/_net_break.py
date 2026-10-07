"""把一个「同一 GT net 的引脚在我的图里被拆到多个分量」的网络画出来。

动机：连通 R 只有 0.175，而覆盖 R 有 0.683 —— 大部分 GT 走线边上确实有我的段，
可分量却连不起来。数字已经问不出更多了（距离门槛 tol 1.5->16 能把 R 抬到 0.534，
但精度从 0.532 塌到 0.121；方向性桥接只值 +0.015）。所以直接看图。

每张图：灰底原图 / 绿=我提取的段（只画该 net 涉及的分量）/ 红=GT 该 net 的走线 /
蓝圈=GT 引脚点 / 品红=各引脚绑定到的那条段。

用法：
  python Y:\workspace\EDA\work\_net_break.py [--cases 20] [--per-case 2] [--tol 1.5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))

import cv2  # noqa: E402

from schematic_parser.data import (DEFAULT_DATA_ROOT, discover,  # noqa: E402
                                   parse_hypergraph, pin_key_to_number)
from stage1.labels import _as_segs, wire_groups  # noqa: E402

OUT = r"Y:\workspace\EDA\work\preview"
S1 = r"Y:\workspace\EDA\work\stage1_all13"
GREEN = (80, 220, 80)
RED = (60, 60, 240)
BLUE = (240, 160, 40)
MAG = (230, 80, 230)


def gt_edges_img(doc, H):
    out = []
    for nid, net in (doc.get("nets") or {}).items():
        for e in (net.get("edges") or {}).values():
            pts = [(float(p["x"]), H - float(p["y"])) for p in e]
            for i in range(len(pts) - 1):
                out.append((nid, pts[i], pts[i + 1]))
    return out


def gt_pins_img(doc, H):
    out = []
    for comp, plist in (doc.get("pins") or {}).items():
        for pkey, rec in (plist or {}).items():
            pt = rec.get("point") or {}
            if pt.get("x") is None:
                continue
            out.append(((comp, pin_key_to_number(pkey)),
                        (float(pt["x"]), H - float(pt["y"]))))
    return out


def nearest_seg(segs, pts):
    """(M,2) -> 最近段下标与距离（全矩阵，不用网格索引）。"""
    if not len(segs) or not len(pts):
        return np.zeros(len(pts), int), np.full(len(pts), 1e18)
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = x2 - x1, y2 - y1
    L2 = np.where(dx * dx + dy * dy < 1e-12, 1e-12, dx * dx + dy * dy)
    t = ((pts[:, 0][:, None] - x1) * dx + (pts[:, 1][:, None] - y1) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    d = np.hypot(pts[:, 0][:, None] - (x1 + t * dx), pts[:, 1][:, None] - (y1 + t * dy))
    j = np.argmin(d, axis=1)
    return j, d[np.arange(len(pts)), j]


def render(case_id, nid, crop, H, pin_pts, gt_segs, mine, bound, path, png=None):
    (x0, y0, x1, y1) = crop
    W = int(x1 - x0)
    Hh = int(y1 - y0)
    scale = min(1.0, 900.0 / max(W, Hh, 1))
    sw, sh = max(2, int(W * scale)), max(2, int(Hh * scale))

    def T(p):
        return (int((p[0] - x0) * scale), int((p[1] - y0) * scale))

    # 底图必须是**原图**：只画白底的话看不出「那段墨迹到底在不在」，而
    # 「在但没提出来」和「本来就没画」是两种完全不同的病。
    if png and os.path.exists(png):
        src = cv2.imread(png, cv2.IMREAD_COLOR)
        ih, iw = src.shape[:2]
        cx0, cy0 = max(0, int(x0)), max(0, int(y0))
        cx1, cy1 = min(iw, int(np.ceil(x1))), min(ih, int(np.ceil(y1)))
        sub = src[cy0:cy1, cx0:cx1]
        # 缩放到与画布同一尺度，缺的部分补白
        if sub.size:
            bh = max(1, int(round((cy1 - cy0) * scale)))
            bw = max(1, int(round((cx1 - cx0) * scale)))
            sub = cv2.resize(sub, (bw, bh), interpolation=cv2.INTER_AREA)
            img = np.full((sh, sw, 3), 255, np.uint8)
            ox, oy = int((cx0 - x0) * scale), int((cy0 - y0) * scale)
            img[oy:oy + bh, ox:ox + bw] = sub
        else:
            img = np.full((sh, sw, 3), 255, np.uint8)
    else:
        img = np.full((sh, sw, 3), 255, np.uint8)
    for (a, b) in mine:
        cv2.line(img, T(a), T(b), GREEN, 2, cv2.LINE_AA)
    for (a, b) in bound:
        cv2.line(img, T(a), T(b), MAG, 2, cv2.LINE_AA)
    for (a, b) in gt_segs:
        cv2.line(img, T(a), T(b), RED, 1, cv2.LINE_AA)
    for p in pin_pts:
        cv2.circle(img, T(p), 6, BLUE, 2, cv2.LINE_AA)
    cv2.putText(img, "%s net=%s" % (case_id, nid),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return sw, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--per-case", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--pin-tol", type=float, default=6.0)
    a = ap.parse_args()

    cases = sorted(discover(DEFAULT_DATA_ROOT), key=lambda c: c.case_id)
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for case in cases[:a.cases]:
        sp = os.path.join(S1, case.case_id + ".stage1.json")
        if not os.path.exists(sp):
            continue
        sd = json.load(open(sp, encoding="utf-8"))
        H = float(sd["image"]["h"])
        gdoc = json.load(open(case.json_path, encoding="utf-8"))
        segs = _as_segs(sd["wire_detail"]["segments"])
        if not len(segs):
            continue
        groups = wire_groups(segs, tol=a.tol)
        pins = gt_pins_img(gdoc, H)
        if not pins:
            continue
        pkeys = [k for k, _ in pins]
        ppts = np.asarray([p for _, p in pins], np.float64)
        jmin, dmin = nearest_seg(segs, ppts)
        key2grp = {}
        for k, j, d in zip(pkeys, jmin, dmin):
            if float(d) <= a.pin_tol:
                key2grp[k] = groups[int(j)]
        key2seg = {k: int(j) for k, j, d in zip(pkeys, jmin, dmin) if float(d) <= a.pin_tol}

        all_edges = gt_edges_img(gdoc, H)
        by_net = {}
        for (nid, a1, b1) in all_edges:
            by_net.setdefault(nid, []).append((a1, b1))

        cand = []
        for nid, net in (gdoc.get("nets") or {}).items():
            keys = [tuple(k) for k in parse_hypergraph(net.get("hyperGraph") or "")]
            ks = [k for k in keys if k in key2grp]
            if len(ks) < 2:
                continue
            gs = set(key2grp[k] for k in ks)
            if len(gs) < 2:
                continue
            cand.append((len(gs) - len(ks), len(gs), nid, ks, gs))
        cand.sort()
        for (_rank, ngs, nid, ks, gs) in cand[:a.per_case]:
            mine = [(segs[i][0:2], segs[i][2:4]) for i in range(len(segs))
                    if groups[i] in gs]
            bound = [(segs[key2seg[k]][0:2], segs[key2seg[k]][2:4])
                     for k in ks if k in key2seg]
            gsegs = by_net.get(nid, [])
            pts = [p for k, p in pins if k in set(ks)]
            xs = [p[0] for p in pts] + [q[0] for s in gsegs for q in s]
            ys = [p[1] for p in pts] + [q[1] for s in gsegs for q in s]
            if not xs:
                continue
            m = 60.0
            crop = (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)
            fn = os.path.join(OUT, "netbreak_%s_%s.png" % (case.case_id, nid))
            sw, sh = render(case.case_id, nid, crop, H, pts, gsegs, mine, bound, fn,
                            png=case.png_path)
            made += 1
            print("  %-28s %-10s 分量%d 引脚%d GT段%d  段%d -> %s (%dx%d)"
                  % (case.case_id, nid, ngs, len(ks), len(gsegs), len(mine),
                     os.path.basename(fn), sw, sh))
    print("共 %d 张，输出目录 %s" % (made, OUT))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
