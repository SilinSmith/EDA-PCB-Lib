# -*- coding: utf-8 -*-
"""把某个 stage1 元件周围裁出来看一眼（判断位号到底印没印、印在哪）。

用法：
    python work\tools\viz\_crop_one.py --case 0003 --comp c0001 --margin 90 \
        --out work\preview\_crop_0003_c0001.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))

from PIL import Image, ImageDraw  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, load_json  # noqa: E402


def find_png(case):
    for sp in SPLITS:
        d = Path(DEFAULT_DATA_ROOT) / sp
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            if sub.name == case or sub.name.split("-")[0] == case:
                for p in sub.glob("*.png"):
                    return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--stage1", default=str(REPO / "work" / "stage1_v6"))
    ap.add_argument("--margin", type=float, default=90.0)
    ap.add_argument("--texts", action="store_true", help="把文字框画上")
    ap.add_argument("--gt", action="store_true", help="把 GT bbox 也画上")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s1 = json.loads((Path(a.stage1) / ("%s.stage1.json" % a.case)).read_text(encoding="utf-8"))
    H = float(s1["image"]["h"])
    comp = None
    for c in s1["components"]:
        if c["id"] == a.comp:
            comp = c
            break
    if comp is None:
        raise SystemExit("没有这个元件 %s（可选：%s）"
                         % (a.comp, [c["id"] for c in s1["components"]]))
    png = find_png(a.case)
    im = Image.open(png).convert("RGB")
    b = comp.get("bbox_img") or comp.get("bbox_json") or [0, 0, 0, 0]
    x0 = max(0, int(b[0] - a.margin))
    y0 = max(0, int(b[1] - a.margin))
    x1 = min(im.width, int(b[2] + a.margin))
    y1 = min(im.height, int(b[3] + a.margin))
    crop = im.crop((x0, y0, x1, y1)).copy()
    d = ImageDraw.Draw(crop)
    d.rectangle([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], outline=(255, 0, 0), width=3)
    if a.texts:
        for t in s1.get("texts") or []:
            tb = t.get("bbox_img")
            if not tb:
                continue
            d.rectangle([tb[0] - x0, tb[1] - y0, tb[2] - x0, tb[3] - y0],
                        outline=(0, 140, 255), width=2)
    if a.gt:
        for k, v in (load_json(case_json(a.case))["components"] if False else {}).items():
            pass
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out)
    print("元件 %s type=%s bbox_img=%s  裁剪 (%d,%d)-(%d,%d) -> %s"
          % (a.comp, comp.get("type"), [round(v, 1) for v in b], x0, y0, x1, y1, out))


def case_json(case):
    return case


if __name__ == "__main__":
    main()
