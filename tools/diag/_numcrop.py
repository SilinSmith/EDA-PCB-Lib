# -*- coding: utf-8 -*-
"""把 stage1 检出的「引脚号码框」裁出来堆成接触表，用来判断 CRNN 读错是
「框太紧把字切掉」还是「模型自己认错」。

用法：
    python work\\tools\\diag\\_numcrop.py 1 -o work\\preview\\_numcrop_1.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT))

from schematic_parser import data as D  # noqa: E402
from stage2.text import TextEngine, pad_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--margin", type=int, default=14)
    ap.add_argument("--texts", default=None, help="stage1 json 目录或文件")
    ap.add_argument("--min-w", type=float, default=6.0)
    ap.add_argument("--min-h", type=float, default=6.0)
    ap.add_argument("--bands", default=None, help="如 '880-960,1320-1400'")
    a = ap.parse_args()

    cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)}
    cid = a.case
    c = cases[cid]
    img = cv2.imread(str(c.png_path))
    H, W = img.shape[:2]

    src = Path(a.texts) if a.texts else (ROOT / "work" / "_submit_out220r32" / "runs" / "stage1")
    if src.is_dir():
        src = src / ("%s.stage1.json" % cid)
    j = json.loads(src.read_text(encoding="utf-8"))
    texts = j.get("texts") or []

    boxes = []
    for t in texts:
        x0, y0, x1, y1 = t["bbox_img"]
        if x1 - x0 < a.min_w or y1 - y0 < a.min_h:
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if a.bands:
            ok = False
            for band in a.bands.split(","):
                xa, xb = [float(v) for v in band.split("-")]
                if xa <= cx <= xb:
                    ok = True
            if not ok:
                continue
        boxes.append((cx, cy, t["bbox_img"], t.get("id")))
    boxes.sort(key=lambda b: (b[0] > W / 2, b[1]))

    eng = TextEngine()
    cells = []
    for cx, cy, bb, tid in boxes:
        x0, y0, x1, y1 = bb
        pb = pad_box(bb, eng.pad_min, eng.pad_frac)
        gx0 = max(0, int(x0) - a.margin)
        gy0 = max(0, int(y0) - a.margin)
        gx1 = min(W, int(np.ceil(x1)) + a.margin)
        gy1 = min(H, int(np.ceil(y1)) + a.margin)
        wide = img[gy0:gy1, gx0:gx1].copy()
        if wide.size == 0:
            continue
        sc = a.scale
        wide = cv2.resize(wide, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        for (bx0, by0, bx1, by1), col in (
                ((x0, y0, x1, y1), (0, 0, 255)),
                (pb, (0, 200, 0))):
            p1 = (int(round((bx0 - gx0) * sc)), int(round((by0 - gy0) * sc)))
            p2 = (int(round((bx1 - gx0) * sc)), int(round((by1 - gy0) * sc)))
            cv2.rectangle(wide, p1, p2, col, 1)
        reads = eng.read(img, [bb], ["pin_number"])
        got = "-"
        if reads:
            r = reads[0].get("reads", {}).get("pin_number")
            if r:
                got = str(r[0])
        cells.append((tid or "", got, wide))

    if not cells:
        print("no boxes")
        return
    pad_top = 26
    hmax = max(c[2].shape[0] for c in cells) + pad_top
    wmax = max(c[2].shape[1] for c in cells) + 8
    cols = 5
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * (hmax + 10), cols * wmax, 3), 255, np.uint8)
    for i, (tid, got, im) in enumerate(cells):
        r, cc = divmod(i, cols)
        y = r * (hmax + 10) + pad_top
        x = cc * wmax
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(sheet, "%s=>%s" % (tid, got), (x + 2, r * (hmax + 10) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1, cv2.LINE_AA)
    out = Path(a.out) if a.out else (ROOT / "work" / "preview" / ("_numcrop_%s.png" % cid))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print("wrote %s  cells=%d  sheet=%s" % (out, len(cells), sheet.shape))


if __name__ == "__main__":
    main()
