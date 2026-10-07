# -*- coding: utf-8 -*-
"""把 GT 引脚点周围的邻域裁出来，看「一个引脚」在图上到底长什么样。

用于设计「逐引脚问 VLM」的裁片尺寸与网格批量。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_pincrop.py --case 1 --key U1 --half 40 --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402


def crop_at(bgr, pt, half):
    h, w = bgr.shape[:2]
    x0 = max(0, int(pt[0] - half)); y0 = max(0, int(pt[1] - half))
    x1 = min(w, int(pt[0] + half)); y1 = min(h, int(pt[1] + half))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


def crop_dir(bgr, pt, center, outer=30.0, inner=78.0, cross=30.0):
    """以引脚点为锚，沿「远离器件中心」方向外扩 `outer`、朝中心方向内伸 `inner`。

    引脚**编号**印在器件外框之外（靠外），引脚**名称**印在器件内部（靠内），
    所以必须用这种不对称窗口才能一裁片同时拿到两者。
    """
    h, w = bgr.shape[:2]
    px, py = float(pt[0]), float(pt[1])
    dx, dy = float(center[0]) - px, float(center[1]) - py
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-3:
        dx, dy, n = 1.0, 0.0, 1.0
    ux, uy = dx / n, dy / n
    ax0, ay0 = px - ux * outer, py - uy * outer          # 外侧角
    ax1, ay1 = px + ux * inner, py + uy * inner          # 内侧角
    x0 = int(min(ax0, ax1) - cross); x1 = int(max(ax0, ax1) + cross)
    y0 = int(min(ay0, ay1) - cross); y1 = int(max(ay0, ay1) + cross)
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


def tile(crops, cols=4, side=140, pad=6, label_h=22, scale=2):
    if not crops:
        return None
    rows = (len(crops) + cols - 1) // cols
    cw = side + pad * 2
    ch = side + pad * 2 + label_h
    out = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for i, c in enumerate(crops):
        r, q = divmod(i, cols)
        if c is None:
            continue
        hh, ww = c.shape[:2]
        s = min(side / max(1, ww), side / max(1, hh)) * scale
        im = cv2.resize(c, (max(1, int(ww * s)), max(1, int(hh * s))),
                        interpolation=cv2.INTER_CUBIC)
        ih, iw = im.shape[:2]
        y = r * ch + label_h + pad + max(0, (side - ih) // 2)
        x = q * cw + pad + max(0, (side - iw) // 2)
        out[y:y + ih, x:x + iw] = im
        cv2.putText(out, "#%d" % i, (q * cw + pad, r * ch + label_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.rectangle(out, (q * cw, r * ch), (q * cw + cw - 1, r * ch + ch - 1),
                      (190, 190, 190), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--key", default="U1")
    ap.add_argument("--half", type=int, default=40)
    ap.add_argument("--dir", action="store_true", help="用「外30/内78」的不对称窗口")
    ap.add_argument("--outer", type=float, default=30.0)
    ap.add_argument("--inner", type=float, default=78.0)
    ap.add_argument("--cross", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    for c in discover():
        if c.case_id != a.case or not c.json_path:
            continue
        gt = load_json(c.json_path)
        gp = (gt.get("pins") or {}).get(a.key) or {}
        cv = (gt.get("components") or {}).get(a.key) or {}
        bgr = imread_unicode(c.png_path)
        H = bgr.shape[0]
        bb = cv.get("bbox") or [0, 0, 1, 1]
        ctr = ((bb[0] + bb[2]) / 2.0, H - (bb[1] + bb[3]) / 2.0)
        items = sorted(gp.items(), key=lambda kv: (kv[1]["point"]["y"], kv[1]["point"]["x"]))
        crops, tags = [], []
        for pk, info in items:
            pt = info["point"]
            q = (pt["x"], H - pt["y"])
            im = (crop_dir(bgr, q, ctr, a.outer, a.inner, a.cross) if a.dir
                  else crop_at(bgr, q, a.half))
            crops.append(im)
            tags.append("%s=%s" % (pk, info.get("pinname")))
        print("图 %s  GT 引脚 %d 个，裁片 %dx%d"
              % (c.png_path.split("\\")[-1], len(items), 2 * a.half, 2 * a.half))
        for t in tags[:a.n]:
            print("  ", t)
        sub = crops[:a.n]
        g = tile(sub, cols=a.cols, side=2 * a.half, scale=2 if not a.dir else 1)
        out = a.out or str(ROOT / "work" / "preview" / ("_pincrop_%s_%s.png" % (a.case, a.key)))
        cv2.imwrite(out, g)
        print("saved:", out, g.shape)
        return 0
    print("没找到"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
