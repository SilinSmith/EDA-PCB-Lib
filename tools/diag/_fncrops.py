# -*- coding: utf-8 -*-
"""把 FN/FP 的具体框从原图裁出来拼成证据图，用于肉眼确认「标注错 vs 模型错」。

用法:
  K:\\python\\311\\python.exe work\\tools\\diag\\_fncrops.py --json work\\tools\\diag\\_gtfaith_v3val.json
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA\trainset_v3")
OUT = Path(r"Y:\workspace\EDA\work\preview")
PAD = 26           # 框外留多少像素上下文
CELL = 300         # 每格渲染尺寸


def crop(img, box, pad=PAD):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = max((x2 - x1), (y2 - y1)) / 2 + pad
    a = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    return img.crop(a), a


def sheet(items, path, title):
    if not items:
        return 0
    cols = 5
    rows = (len(items) + cols - 1) // cols
    W, H = cols * CELL, rows * (CELL + 26)
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), title, fill=(0, 0, 0))
    for i, (p, box, label) in enumerate(items):
        with Image.open(p) as im:
            im = im.convert("RGB")
            c, (ax1, ay1, ax2, ay2) = crop(im, box)
        sc = min(CELL / max(1, c.width), CELL / max(1, c.height))
        c = c.resize((max(1, int(c.width * sc)), max(1, int(c.height * sc))),
                     Image.LANCZOS)
        cx0 = (i % cols) * CELL
        cy0 = (i // cols) * (CELL + 26) + 22
        canvas.paste(c, (cx0 + 2, cy0 + 2))
        # 画红框（原框在裁剪坐标系中的位置）
        rx1 = (box[0] - ax1) * sc + 2
        ry1 = (box[1] - ay1) * sc + 2
        rx2 = (box[2] - ax1) * sc + 2
        ry2 = (box[3] - ay1) * sc + 2
        d.rectangle([cx0 + rx1, cy0 + ry1, cx0 + rx2, cy0 + ry2],
                    outline=(255, 0, 0), width=2)
        # 把框放大 6 倍后贴在右上角做放大镜
        zx1, zy1 = max(0, int(box[0]) - 3), max(0, int(box[1]) - 3)
        zoom = im.crop((zx1, zy1, int(box[2]) + 3, int(box[3]) + 3))
        if zoom.width and zoom.height:
            zsc = min(9, max(1, 90 // max(zoom.width, zoom.height)))
            zoom = zoom.resize((zoom.width * zsc, zoom.height * zsc), Image.NEAREST)
            canvas.paste(zoom, (cx0 + 2, cy0 + CELL - zoom.height - 2))
        d.text((cx0 + 4, cy0 - 18), label[:46], fill=(0, 0, 120))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print("[saved] %s  (%d cells)" % (path, len(items)))
    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(Path(__file__).with_name("_gtfaith_v3val.json")))
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dump = json.loads(Path(a.json).read_text(encoding="utf-8"))
    fn, fp = dump["fn"], dump["fp"]

    def it(rows, kindf, label):
        out = []
        for s in rows:
            if not kindf(s):
                continue
            st = Path(s["img"]).stem
            p = ROOT / "images" / a.split / (st + ".png")
            if not p.exists():
                p = ROOT / "images" / a.split / (st + ".jpg")
            out.append((p, s["box"], label(s)))
        return out

    n = 0
    for cls, tag in (("r", "FN"), ("net_input", "FN"), ("testpoint", "FN"),
                     ("v", "FN"), ("box", "FN")):
        rows = fn.get(cls, [])
        for kind in ("DEADGT", "CLASSCONF", "DUP", "NEAROTHER", "TRUEMISS"):
            sub = [s for s in rows if s["kind"] == kind]
            if not sub:
                continue
            n += sheet([(ROOT / "images" / a.split / (Path(s["img"]).stem + ".png"),
                         s["box"],
                         "%s %s ink=%.2f %s" % (cls, kind, s["ink"],
                                                Path(s["img"]).stem[:22]))
                        for s in sub[:20]],
                       OUT / ("ev_%s_%s_%s.png" % (tag, cls, kind)),
                       "%s %s / %s  (%d) 红框=GT 框" % (tag, cls, kind, len(sub)))

    for cls in ("r", "net_input", "box", "gnd", "c"):
        rows = fp.get(cls, [])
        for kind in ("REALFP", "NEARMISS"):
            sub = [s for s in rows if s["kind"] == kind]
            if not sub:
                continue
            n += sheet([(ROOT / "images" / a.split / (Path(s["img"]).stem + ".png"),
                         s["box"],
                         "FP %s %s conf=%.2f %s" % (cls, kind, s["score"],
                                                    Path(s["img"]).stem[:20]))
                        for s in sorted(sub, key=lambda x: -x["score"])[:20]],
                       OUT / ("ev_FP_%s_%s.png" % (cls, kind)),
                       "FP %s / %s (%d) 红框=预测框" % (cls, kind, len(sub)))
    print("total sheets/cells:", n)


if __name__ == "__main__":
    raise SystemExit(main())
