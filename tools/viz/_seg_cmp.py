# -*- coding: utf-8 -*-
"""文字框切分对照图：真值（CVJsonStd）绿 / 旧 textbox 红 / 新 textseg 蓝。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_seg_cmp.py 0059 0101 0001
    产物 -> Y:\\workspace\\EDA\\work\\preview\\segcmp_<case>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402

_YP = ROOT / "eda-pcb" / "yolo_part"
_DS = [p for p in _YP.iterdir() if p.is_dir() and p.name.startswith("Dataset300")]
D300 = _DS[0] if _DS else _YP / "Dataset300"
OUT = ROOT / "work" / "preview"

LEG = ("绿=真值(CVJsonStd location)   红=旧 textbox   蓝=新 textseg   "
       "黄=两者都有")


def find_case(cid):
    for d in sorted(D300.iterdir()):
        if d.is_dir() and d.name.startswith(cid):
            png = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            js = sorted(d.glob("*_CVJsonStd.json"))
            if png and js:
                return png[0], js[0]
    return None, None


def draw(img, boxes, color, thick=1, tag=None):
    vis = img.copy()
    for b in boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thick)
    return vis


def band(w, h, text, color=(0, 0, 0)):
    b = np.full((22, w, 3), 255, np.uint8)
    cv2.putText(b, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return b


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    crop = None
    zoom = 1
    ids = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--crop":
            crop = tuple(float(v) for v in args[i + 1].split(","))
            i += 2
            continue
        if a == "--zoom":
            zoom = int(args[i + 1])
            i += 2
            continue
        if not a.startswith("-"):
            ids.append(a)
        i += 1
    ids = ids or ["0059", "0101", "0001"]
    OUT.mkdir(parents=True, exist_ok=True)
    for cid in ids:
        png, js = find_case(cid)
        if png is None:
            print(f"  {cid}: 找不到")
            continue
        bgr = imread_unicode(png)
        d = json.loads(Path(js).read_text(encoding="utf-8"))
        gt = []
        for t in (d.get("texts") or {}).values():
            loc = t.get("location") or {}
            try:
                gt.append((float(loc["x1"]), float(loc["y1"]),
                           float(loc["x2"]), float(loc["y2"])))
            except (TypeError, KeyError):
                pass
        ob = [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
        nb = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr)]
        H, W = bgr.shape[:2]
        half = cv2.resize(bgr, (W // 2, H // 2))
        scale = 0.5

        def sc(boxes):
            return [(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale)
                    for b in boxes]

        if crop:
            x1, y1, x2, y2 = (int(v) for v in crop)
            half = bgr[y1:y2, x1:x2]
            if zoom != 1:
                half = cv2.resize(half, None, fx=zoom, fy=zoom,
                                  interpolation=cv2.INTER_NEAREST)
            scale = 1.0 * zoom

            def sc(boxes):
                return [((b[0] - x1) * scale, (b[1] - y1) * scale,
                         (b[2] - x1) * scale, (b[3] - y1) * scale)
                        for b in boxes
                        if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2]
        HH, WW = half.shape[:2]
        panels = [
            band(WW, 22, f"{cid}  GT {len(gt)} 框", (0, 140, 0)),
            draw(half, sc(gt), (0, 200, 0)),
            band(WW, 22, f"旧 textbox {len(ob)} 框", (0, 0, 200)),
            draw(half, sc(ob), (0, 0, 220)),
            band(WW, 22, f"新 textseg {len(nb)} 框", (200, 0, 0)),
            draw(half, sc(nb), (220, 0, 0)),
            band(WW, 22, "叠加", (0, 0, 0)),
            draw(draw(half, sc(gt), (0, 200, 0)), sc(nb), (220, 0, 0)),
        ]
        hdr = band(WW, 22, LEG, (60, 60, 60))
        out = np.vstack([hdr] + panels)
        p = OUT / f"segcmp_{cid}{'_crop' if crop else ''}.png"
        if crop:
            p = OUT / (f"segcmp_{cid}_crop_"
                       f"{int(crop[0])}_{int(crop[1])}.png")
        cv2.imwrite(str(p), out)
        print(f"  {cid}: GT {len(gt)} / 旧 {len(ob)} / 新 {len(nb)} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
