# -*- coding: utf-8 -*-
"""把指定 case 的若干 GT 引脚邻域裁成一张拼图（带十字与标签），用来看「图上到底画了什么」。

    python work/tools/diag/_pincropgt.py --case 1 --key U1 --pins 7,113,115,39,48,68 --half 45 --scale 3
    python work/tools/diag/_pincropgt.py --case 1 --key U1 --n 12        # 不指定就取前 n 个
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M                                  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from stage2.text import imread_unicode                                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--pins", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--half", type=int, default=45)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    c = {x.case_id: x for x in discover(DEFAULT_DATA_ROOT)}[a.case]
    gt = M.normalise_doc(load_json(Path(c.json_path)))
    img = imread_unicode(c.png_path)
    H, W = img.shape[:2]
    print("图 %s  %dx%d" % (Path(c.png_path).name, W, H))
    g = (gt.get("pins") or {}).get(a.key) or {}
    items = []
    want = [s.strip() for s in a.pins.split(",") if s.strip()]
    for pk, info in g.items():
        num = M.pin_key_to_number(pk)
        if want and str(num) not in want:
            continue
        p = (info or {}).get("point") or {}
        items.append((pk, (info or {}).get("pinname"),
                      float(p.get("x", 0)), float(p.get("y", 0))))
    items.sort(key=lambda t: -t[3])
    if not want:
        items = items[:a.n]
    if not items:
        print("没有匹配的引脚")
        return 1
    h = a.half
    s = a.scale
    side = 2 * h * s
    cols = min(a.cols, len(items))
    rows = (len(items) + cols - 1) // cols
    lab = 30
    out = np.full((rows * (side + lab), cols * side, 3), 245, np.uint8)
    for i, (pk, nm, x, y) in enumerate(items):
        y_img = H - y
        x0, x1 = int(max(0, x - h)), int(min(W, x + h))
        y0, y1 = int(max(0, y_img - h)), int(min(H, y_img + h))
        sub = img[y0:y1, x0:x1].copy()
        sub = cv2.resize(sub, (side, side), interpolation=cv2.INTER_NEAREST)
        cx = int((x - x0) * s)
        cy = int((y_img - y0) * s)
        cv2.line(sub, (cx - 14, cy), (cx + 14, cy), (0, 0, 255), 2)
        cv2.line(sub, (cx, cy - 14), (cx, cy + 14), (0, 0, 255), 2)
        r, cc = divmod(i, cols)
        oy, ox = r * (side + lab), cc * side
        out[oy + lab:oy + lab + side, ox:ox + side] = sub
        cv2.putText(out, "%s n=%s" % (pk, nm),
                    (ox + 3, oy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        print("   %-9s %-10s gt=(%.0f,%.0f) 裁 (x %d..%d, y_img %d..%d)"
              % (pk, nm, x, y, x0, x1, y0, y1))
    outdir = ROOT / "work" / "preview"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = Path(a.out) if a.out else outdir / ("_pincropgt_%s_%s.png" % (a.case, a.key))
    cv2.imencode(".png", out)[1].tofile(str(dst))
    print("→ %s" % dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
