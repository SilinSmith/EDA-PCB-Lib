"""把 GT 某个元件的框（含四周留白）裁出来放大，看**位号到底画没画在图上**。

坐标：GT JSON 是 **y 向上**，图是 y 向下 ⇒ `y_img = H - y_json`。

用法：
  python work\\tools\\diag\\_boxcrop.py --case 0019 --key u1 --pad 0.35 --scale 2
  python work\\tools\\diag\\_boxcrop.py --case 0019 --all --pad 0.35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import discover, load_json          # noqa: E402
from stage2.text import imread_unicode                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--all", action="store_true", help="该 case 的所有元件")
    ap.add_argument("--pad", type=float, default=0.35, help="四周留白（比例于边长）")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    case = None
    for c in discover(__import__("schematic_parser.data", fromlist=["d"]).DEFAULT_DATA_ROOT):
        if c.case_id == a.case and c.json_path:
            case = c
            break
    if case is None:
        print("找不到 case %s" % a.case)
        return
    bgr = imread_unicode(case.png_path)
    H, W = bgr.shape[:2]
    doc = load_json(case.json_path)
    comps = doc.get("components") or {}

    want = list(comps)
    if a.key:
        want = [k for k in comps if k.lower() == a.key.lower()] or [a.key]
    if not a.all and a.key:
        want = want[:1]

    tiles, labels = [], []
    for k in want:
        v = comps.get(k) or {}
        bb = v.get("bbox")
        if not bb or len(bb) < 4:
            continue
        x1j, y1j, x2j, y2j = [float(t) for t in bb[:4]]
        # JSON -> 图像（y 翻）
        xa, xb = sorted((x1j, x2j))
        ya, yb = sorted((H - y1j, H - y2j))
        pw = (xb - xa) * a.pad
        ph = (yb - ya) * a.pad
        x0 = max(0, int(xa - pw)); y0 = max(0, int(ya - ph))
        x1 = min(W, int(xb + pw)); y1 = min(H, int(yb + ph))
        if x1 <= x0 or y1 <= y0:
            continue
        im = bgr[y0:y1, x0:x1].copy()
        # 画出 GT 框
        cv2.rectangle(im, (int(xa - x0), int(ya - y0)), (int(xb - x0), int(yb - y0)),
                      (0, 0, 220), 2)
        im = cv2.resize(im, None, fx=a.scale, fy=a.scale,
                        interpolation=cv2.INTER_NEAREST)
        tiles.append(im)
        labels.append("%s  %s  %.0fx%.0f" % (a.case, k, xb - xa, yb - ya))
        print("  %-12s type=%-10s bbox(JSON)=%s  -> 图像 (%d,%d)-(%d,%d)  %dx%d"
              % (k, v.get("type"), bb[:4], x0, y0, x1, y1, x1 - x0, y1 - y0))

    if not tiles:
        print("没有可裁的元件")
        return
    gap, lab = 10, 26
    tw = max(t.shape[1] for t in tiles) + gap * 2
    th = sum(t.shape[0] + lab + gap for t in tiles) + gap
    out = np.full((th, tw, 3), 255, np.uint8)
    y = gap
    for t, s in zip(tiles, labels):
        cv2.putText(out, s, (gap, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 200), 2, cv2.LINE_AA)
        y += lab
        h, w = t.shape[:2]
        out[y:y + h, gap:gap + w] = t
        y += h + gap
    op = a.out or str(ROOT / "work" / "preview" / ("_boxcrop_%s%s.png"
                                                   % (a.case, ("_" + a.key) if a.key else "")))
    cv2.imencode(".png", out)[1].tofile(op)
    print("已写 %s  (%dx%d)" % (op, out.shape[1], out.shape[0]))


if __name__ == "__main__":
    main()
