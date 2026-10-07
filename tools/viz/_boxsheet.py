# -*- coding: utf-8 -*-
"""把一张图里所有 GT box(IC) 元件周围的「环带」裁出来拼成一张对照图，
用来肉眼判断：官方 GT 的位号到底印没印在图上、印在哪。

用法：
    python work\tools\viz\_boxsheet.py --case 0003 --pad 70 \
        --out work\preview\_boxsheet_0003.png
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))

from PIL import Image, ImageDraw  # noqa: E402

TRAIN = (REPO / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
         / "200_train_cases")
GT10 = (REPO / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT")


def find(case):
    for d in glob.glob(str(TRAIN / (case + "*"))) + glob.glob(str(GT10 / "*")):
        p = Path(d)
        if p.is_dir() and p.name.split("-")[0] == case:
            png = next(p.glob("*.png"), None)
            js = next((x for x in p.glob("*.json")), None)
            if png and js:
                return png, js
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pad", type=float, default=70.0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cell", type=int, default=420)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    png, js = find(a.case)
    if not png:
        raise SystemExit("找不到 case %s" % a.case)
    gt = json.loads(Path(js).read_text(encoding="utf-8"))
    im = Image.open(png).convert("RGB")
    H = im.height
    boxes = [(k, v) for k, v in (gt.get("components") or {}).items()
             if (v.get("type") or "") == "box"]
    print("GT box %d 个（图 %dx%d）：%s" % (len(boxes), im.width, H,
                                          [k for k, _ in boxes]))
    tiles = []
    for k, v in boxes:
        b = v["bbox"]
        # GT 是左下角原点 → 转成图像坐标
        iy0, iy1 = H - b[3], H - b[1]
        crop = im.crop((max(0, int(b[0] - a.pad)), max(0, int(iy0 - a.pad)),
                        min(im.width, int(b[2] + a.pad)),
                        min(im.height, int(iy1 + a.pad))))
        crop.thumbnail((a.cell, a.cell))
        t = Image.new("RGB", (a.cell, a.cell + 26), (255, 255, 255))
        t.paste(crop, ((a.cell - crop.width) // 2,
                       (a.cell - crop.height) // 2))
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, a.cell - 1, a.cell - 1], outline=(255, 0, 0))
        d.text((6, a.cell + 6), "%s  Name=%s" % (k, v.get("Name")),
               fill=(0, 0, 0))
        tiles.append(t)
    cols = a.cols
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * a.cell, rows * (a.cell + 26)), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * a.cell, (i // cols) * (a.cell + 26)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("-> %s  %s" % (out, sheet.size))


if __name__ == "__main__":
    main()
