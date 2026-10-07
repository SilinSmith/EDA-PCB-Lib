# -*- coding: utf-8 -*-
"""渲染任意图的任意区域，用于肉眼判定图形到底是什么。"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA\trainset_v3\images\val")
OUT = Path(r"Y:\workspace\EDA\work\preview")

CASES = [
    # (stem, x1, y1, x2, y2, 缩放)
    ("200_train_cases-0050__full", 560, 2440, 1780, 2650, 1.0),
    ("200_train_cases-0050__t03_00", 0, 0, 1024, 1024, 0.6),
    ("200_train_cases-0190__full", 250, 250, 560, 420, 3.0),
    ("200_train_cases-0050__full", 150, 2500, 900, 2620, 1.6),
]


def main():
    for stem, x1, y1, x2, y2, sc in CASES:
        p = ROOT / (stem + ".png")
        if not p.exists():
            print("missing", p)
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            c = im.crop((x1, y1, x2, y2))
        if sc != 1.0:
            c = c.resize((int(c.width * sc), int(c.height * sc)), Image.LANCZOS)
        name = OUT / ("region_%s_%d_%d.png" % (stem[:26], x1, y1))
        c.save(name)
        print("[saved] %s  %dx%d  region=(%d,%d)-(%d,%d) scale=%.2f"
              % (name.name, c.width, c.height, x1, y1, x2, y2, sc))


if __name__ == "__main__":
    raise SystemExit(main())
