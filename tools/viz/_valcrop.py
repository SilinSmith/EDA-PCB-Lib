# -*- coding: utf-8 -*-
"""把「位号读对了、但值没写出来」的元件裁出来拼成对照图，肉眼看值到底在不在图上。

用法：python work\tools\viz\_valcrop.py --case 0001 --pad 90 --cols 5 ^
        --out work\preview\_valmiss_0001.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402

FONTS = cv2.FONT_HERSHEY_SIMPLEX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v13"))
    ap.add_argument("--pad", type=int, default=90)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from _crop_one import find_png
    png = find_png(a.case)
    if png is None:
        print("找不到图"); return
    im = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_COLOR)
    H = im.shape[0]

    gt = load_json(next(Path(DEFAULT_DATA_ROOT).rglob(
        "%s*_target_named.json" % a.case)))
    pred = json.loads((Path(a.pred) / ("%s.json" % a.case)).read_text(encoding="utf-8"))
    gc, pc = gt.get("components") or {}, pred.get("components") or {}
    want = []
    for k in gc:
        if k not in pc:
            continue
        g, p = gc[k], pc[k]
        if (g.get("type") or "") not in ("c", "r", "l", "d"):
            continue
        gv = g.get("value")
        if not gv:
            continue
        if p.get("value"):
            continue
        want.append((k, gv, g.get("bbox")))
    print("该图「位号对、值没写」的 c/r/l/d：%d  %s" % (len(want), [w[0] for w in want[:30]]))

    cells = []
    for k, gv, gb in want[:a.max]:
        x1, y1, x2, y2 = gb
        # GT 左下角 → 图像坐标
        iy1, iy2 = int(H - y2) - a.pad, int(H - y1) + a.pad
        ix1, ix2 = int(x1) - a.pad, int(x2) + a.pad
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(im.shape[1], ix2), min(H, iy2)
        crop = im[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        s = min(a.cell / max(1, w), a.cell / max(1, h))
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA)
        cell = np.full((a.cell + 26, a.cell, 3), 255, np.uint8)
        ch, cw = crop.shape[:2]
        cell[26:26 + ch, :cw] = crop
        cv2.putText(cell, "%s=%s" % (k, gv), (3, 18), FONTS, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        cells.append(cell)
    if not cells:
        print("没有可裁的"); return
    rows = []
    for i in range(0, len(cells), a.cols):
        row = cells[i:i + a.cols]
        while len(row) < a.cols:
            row.append(np.full_like(cells[0], 255))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", sheet)[1].tofile(a.out)
    print("写出", a.out, sheet.shape)


if __name__ == "__main__":
    main()
