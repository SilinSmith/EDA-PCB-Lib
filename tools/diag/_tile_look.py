# -*- coding: utf-8 -*-
"""把某个切片的标签画出来，人眼核对 (用户 m07249「是不是训练集本身就有问题」)。

    & 'K:\python\311\python.exe' _tile_look.py 200_train_cases-0050__t00_01
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(r"Y:\workspace\EDA\trainset_v2")
CLASSES = [c.strip() for c in (ROOT / "classes.txt").read_text(
    encoding="utf-8", errors="ignore").splitlines() if c.strip()]
COL = {"r": (255, 60, 60), "c": (60, 200, 255), "gnd": (80, 255, 120),
       "v": (255, 200, 40), "testpoint": (255, 80, 255), "box": (255, 140, 0),
       "net_input": (0, 255, 200), "l": (160, 120, 255), "pin": (200, 200, 200)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    stem = sys.argv[1]
    up = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    p = ROOT / "images" / "val" / (stem + ".png")
    if not p.exists():
        p = ROOT / "images" / "train" / (stem + ".png")
    im = Image.open(str(p)).convert("RGB")
    W, H = im.size
    print("图像 %s  %dx%d" % (p.name, W, H))
    d = ImageDraw.Draw(im)
    txt = (ROOT / "labels" / ("val" if "val" in str(p.parent) else "train")
           / (stem + ".txt"))
    lines = [l.split() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = {}
    for q in lines:
        cid = int(float(q[0])); cx, cy, bw, bh = (float(v) for v in q[1:5])
        x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
        nm = CLASSES[cid] if cid < len(CLASSES) else str(cid)
        n[nm] = n.get(nm, 0) + 1
        col = COL.get(nm, (255, 255, 255))
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        d.text((x1, max(0, y1 - 11)), "%.0fx%.0f" % (x2 - x1, y2 - y1), fill=col)
    print("标签 %d 个：%s" % (len(lines), n))
    if up > 1:
        im = im.resize((W * up, H * up), Image.NEAREST)
    out = Path(r"Y:\workspace\EDA\work\preview") / ("tilelook_%s.png" % stem)
    im.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    main()
