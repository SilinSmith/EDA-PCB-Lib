# -*- coding: utf-8 -*-
"""_wiresnap.py —— 5px 到底是「差一点」还是「结构性错位」：
   * 对 GT 边找最近预测段，打印两端点距离对（看是均匀 5~20px 还是随坐标放大 ⇒ 缩放错）
   * 试「端点吸附」：把预测端点吸附到原图墨迹像素后重算（不改源码，纯离线验证）
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import _edge_points  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)

CASES = ["1", "4", "8", "9", "0019", "0024"]
W = ROOT / "work"


def segd(p, q):
    f = max(math.hypot(p[0][0] - q[0][0], p[0][1] - q[0][1]),
            math.hypot(p[1][0] - q[1][0], p[1][1] - q[1][1]))
    r = max(math.hypot(p[0][0] - q[1][0], p[0][1] - q[1][1]),
            math.hypot(p[1][0] - q[0][0], p[1][1] - q[0][1]))
    return min(f, r)


for cid in CASES:
    sp = W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % cid)
    if not sp.exists() or cid not in exact:
        print("skip", cid)
        continue
    s1 = json.loads(sp.read_text(encoding="utf-8"))
    H = float((s1.get("image") or {}).get("h") or 0)
    img = cv2.imread((s1.get("image") or {}).get("path", ""), cv2.IMREAD_GRAYSCALE)
    segs = [((float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1])))
            for s in ((s1.get("wire_detail") or {}).get("segments") or [])]
    gt = load_json(exact[cid].json_path)
    ge = []
    for k, n in (gt.get("nets") or {}).items():
        for ek, e in (n.get("edges") or {}).items():
            p = _edge_points(e)
            if p is None:
                continue
            ge.append((k, ek, (p[0][0], H - p[0][1]), (p[-1][0], H - p[-1][1])))
    print("=== %s  H=%.0f  img=%s  segs=%d  gtE=%d"
          % (cid, H, None if img is None else img.shape, len(segs), len(ge)))
    if img is None:
        continue
    # 端点吸附：在 5x5 窗口里找最近的墨迹像素（阈值 128）
    ink = (img < 128).astype(np.uint8)
    ys, xs = np.nonzero(ink)

    def snap(p):
        x0, y0 = int(round(p[0])), int(round(p[1]))
        x1, x2 = max(0, x0 - 4), min(img.shape[1] - 1, x0 + 4)
        y1, y2 = max(0, y0 - 4), min(img.shape[0] - 1, y0 + 4)
        m = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        if not m.any():
            return p, None
        d = (xs[m] - p[0]) ** 2 + (ys[m] - p[1]) ** 2
        i = int(np.argmin(d))
        return (float(xs[m][i]), float(ys[m][i])), float(math.sqrt(d[i]))

    raw = {"5": 0, "10": 0, "20": 0}
    snapped = {"5": 0, "10": 0, "20": 0}
    dists = []
    for _k, _e, a, b in ge:
        best, bs = 1e18, None
        for s in segs:
            d = segd((a, b), s)
            if d < best:
                best, bs = d, s
        for tol in raw:
            if best <= float(tol):
                raw[tol] += 1
        sa, da = snap(bs[0])
        sb, db = snap(bs[1])
        sd = segd((a, b), (sa, sb))
        for tol in snapped:
            if sd <= float(tol):
                snapped[tol] += 1
        if best > 5:
            dists.append((best, a, b, bs, sd))
    n = len(ge)
    print("   原始    tol5 %5.1f%%  tol10 %5.1f%%  tol20 %5.1f%%"
          % (100.0 * raw["5"] / n, 100.0 * raw["10"] / n, 100.0 * raw["20"] / n))
    print("   端点吸附后 tol5 %5.1f%%  tol10 %5.1f%%  tol20 %5.1f%%"
          % (100.0 * snapped["5"] / n, 100.0 * snapped["10"] / n,
             100.0 * snapped["20"] / n))
    dists.sort(key=lambda t: t[0])
    print("   最接近但差一点的 4 条（GT 边 → 最近预测段，d 原始→吸附）：")
    for d, a, b, s, sd in dists[:4]:
        print("     d=%6.2f→%6.2f  GT(%s,%s)→(%s,%s)  pred(%s,%s)→(%s,%s)"
              % (d, sd, round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1),
                 round(s[0][0], 2), round(s[0][1], 2), round(s[1][0], 2), round(s[1][1], 2)))
