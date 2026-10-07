# -*- coding: utf-8 -*-
"""对照：同一个框，左＝现行裁剪（框 + PAD_X/Y_FRAC），右＝墨迹归一后的裁剪。

渲染的是**模型真正吃到的 32px 输入**（放大 6×），第三列画出探测带里被判定为
「字」的行（`_ink_band` 找到的墨迹范围）叠在原图上，用来判断归一是对的还是错的。

    python work\\tools\\diag\\_inkvis.py --n 14
    TEXTREC_INK_YNORM=1 TEXTREC_INK_FRAME=2.5 python work\\tools\\diag\\_inkvis.py --n 14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from textrec import config as C                                   # noqa: E402
from textrec.dataset import crop_gray, _ink_band                  # noqa: E402

S = 6            # 输入放大倍数
GAP = 10


def to_img(crop):
    a = np.asarray(crop, dtype="uint8")
    im = Image.fromarray(a).convert("RGB").resize(
        (a.shape[1] * S, a.shape[0] * S), Image.NEAREST)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "work" / "ocr_real" / "manifest.jsonl"))
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--role", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--out", default=str(ROOT / "work" / "preview" / "_inkvis.png"))
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    val = [r for r in recs if r.get("split") == "val"]
    if a.role:
        val = [r for r in val if r.get("role") == a.role]
    if a.case:
        val = [r for r in val if r.get("case") == a.case]
    if not val:
        print("没有条目")
        return
    step = max(1, len(val) // a.n)
    val = val[::step][:a.n]

    tiles = []
    for r in val:
        b = r["bbox"]
        H = max(1.0, float(b[3]) - float(b[1]))
        im = Image.open(r["image"]).convert("L")
        W, Hi = im.size
        # 左：现行裁剪
        left = to_img(crop_gray(im, b, C.IMG_H, C.MAX_W))
        # 右：强制开启归一
        C.INK_YNORM = True
        right = to_img(crop_gray(im, b, C.IMG_H, C.MAX_W))
        C.INK_YNORM = False
        # 第三列：原图上画探测带 / 墨迹带 / 归一后的裁剪
        x1 = int(max(0, min(W - 1, round(b[0])))); x2 = int(max(x1 + 1, min(W, round(b[2]))))
        y1 = int(max(0, min(Hi - 1, round(b[1])))); y2 = int(max(y1 + 1, min(Hi, round(b[3]))))
        my = C.PAD_Y_FRAC * H
        cy1 = int(max(0, min(Hi - 1, round(b[1] - my))))
        cy2 = int(max(cy1 + 1, min(Hi, round(b[3] + my))))
        px1 = max(0, x1 - 6); px2 = min(W, x2 + 6)
        py1 = max(0, cy1 - 8); py2 = min(Hi, cy2 + 8)
        got = _ink_band(im, cy1, cy2, x1, x2, W, Hi, False, None)
        box = im.crop((px1, py1, px2, py2)).convert("RGB")
        sc = max(1, int(round(60.0 / max(1, box.size[1]))))
        box = box.resize((box.size[0] * sc, box.size[1] * sc), Image.NEAREST)
        d = ImageDraw.Draw(box)
        def hline(yv, col):
            yy = (yv - py1) * sc
            d.line([(0, yy), (box.size[0], yy)], fill=col, width=1)
        hline(cy1, (60, 60, 255))          # 现行裁剪上边
        hline(cy2, (60, 60, 255))          # 现行裁剪下边
        if got is not None:
            hline(got[0], (0, 190, 0))     # 归一后裁剪上边
            hline(got[1], (0, 190, 0))     # 归一后裁剪下边
        lab = "%s %s H=%.0f g=%s %r" % (
            r.get("case"), r.get("role"), H, ("%.0f" % got[2]) if got else "None",
            r.get("text"))
        w = max(left.size[0] + right.size[0] + box.size[0] + 2 * GAP, 8 * len(lab))
        row = Image.new("RGB", (w, max(left.size[1], right.size[1], box.size[1]) + 14),
                        (255, 255, 255))
        row.paste(left, (0, 14))
        row.paste(right, (left.size[0] + GAP, 14))
        row.paste(box, (left.size[0] + right.size[0] + 2 * GAP, 14))
        ImageDraw.Draw(row).text((2, 2), lab, fill=(200, 0, 0))
        tiles.append(row)

    W = max(t.size[0] for t in tiles)
    Ht = sum(t.size[1] for t in tiles)
    out = Image.new("RGB", (W, Ht), (255, 255, 255))
    y = 0
    for t in tiles:
        out.paste(t, (0, y)); y += t.size[1]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print("写了 %s  (%dx%d)  左=现行  右=归一  右三=原图(蓝=现行裁剪边界 绿=归一后边界)"
          % (a.out, out.size[0], out.size[1]))


if __name__ == "__main__":
    main()
