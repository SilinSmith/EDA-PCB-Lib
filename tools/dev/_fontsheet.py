# -*- coding: utf-8 -*-
"""把真实裁片和候选字体的渲染并排画出来，肉眼看哪个像（`_fontfit.py` 的数字靠这个兜底）。

用法：
  python -X utf8 work\tools\dev\_fontsheet.py --role value --variant jlc --n 14 --out work\_fontsheet.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORK = Path(r"Y:\workspace\EDA\work")
MPL = Path(r"Y:\workspace\EDA\eda-pcb\eda_yolo\Lib\site-packages\matplotlib\mpl-data\fonts\ttf")
FONTS = [
    ("arial", Path(r"C:\Windows\Fonts\arial.ttf")),
    ("calibri", Path(r"C:\Windows\Fonts\calibri.ttf")),
    ("segoeui", Path(r"C:\Windows\Fonts\segoeui.ttf")),
    ("consola", Path(r"C:\Windows\Fonts\consola.ttf")),
    ("courbd", Path(r"C:\Windows\Fonts\courbd.ttf")),
    ("DejaVuMono", MPL / "DejaVuSansMono.ttf"),
]
SCALE = 6          # 放大倍数（官方字太小，肉眼看必须放大）
ROW_H = 46


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(WORK / "official_text" / "manifest.jsonl"))
    ap.add_argument("--role", default="value")
    ap.add_argument("--variant", default="jlc")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--out", default=str(WORK / "_fontsheet.png"))
    a = ap.parse_args()

    rows = []
    for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("role") != a.role or r.get("dist", 0) != 0:
            continue
        if len(r.get("text") or "") < 3:
            continue
        if a.variant and a.variant.lower() not in Path(r["image"]).name.lower():
            continue
        rows.append(r)
        if len(rows) >= a.n:
            break

    W = 760
    img = Image.new("RGB", (W, ROW_H * len(rows)), "white")
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        y = i * ROW_H
        p = Path(r["image"])
        sheet = Image.open(p).convert("RGB")
        x1, y1, x2, y2 = [int(round(float(v))) for v in r["bbox"]]
        crop = sheet.crop((x1 - 2, y1 - 2, x2 + 2, y2 + 2))
        h = crop.height
        crop = crop.resize((crop.width * SCALE, h * SCALE), Image.NEAREST)
        img.paste(crop, (150, y + 2))
        d.text((4, y + 14), "%s" % r["text"], fill="black")
        x = 150 + crop.width + 16
        for name, path in FONTS:
            if not path.is_file():
                continue
            f = ImageFont.truetype(str(path), max(6, h) * SCALE)
            w = int(d.textlength(r["text"], font=f))
            d.text((x, y + 2), r["text"], font=f, fill=(0, 0, 0))
            d.text((x, y + 2 + h * SCALE + 2), name, fill=(120, 120, 120))
            x += max(w, 40) + 16
    img.save(a.out)
    print("写好", a.out, img.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
