"""看得见的第一步：把官方 GT 的文字真值与检测到的文字框对上，裁出来放大看。

目的：判断「本地手段能不能识别」这件事，先看清楚墨迹长什么样。

跑法（CWD = Y:\\workspace\\EDA\\eda-pcb）：
    $env:PYTHONIOENCODING='utf-8'
    python Y:\\workspace\\EDA\\work\\_ocr_contact.py 0001 0059 0025 0101
产物：Y:\\workspace\\EDA\\work\\preview\\ocr_crops_<case>.png
"""
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import cv2
import numpy as np
from PIL import Image, ImageDraw

from schematic_parser import data as gtdata
from cvparts import textbox as tb

sys.stdout.reconfigure(encoding="utf-8")

OUT = r"Y:\workspace\EDA\work\preview"
SCALE = 6
ROW_H = 30
PAD = 2


def nearest_box(boxes, x, y, r=None):
    """返回 (dist, box)。x,y 是图像坐标点；r 是矩形 (x1,y1,x2,y2) 时算点到矩形的距离。"""
    best = None
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if r is None:
            d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
        else:
            dx = max(r[0] - cx, 0, cx - r[2])
            dy = max(r[1] - cy, 0, cy - r[3])
            d = (dx * dx + dy * dy) ** 0.5
        if best is None or d < best[0]:
            best = (d, b)
    return best


def crop_upscale(img, box, scale=SCALE, pad=PAD):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    c = img[y1:y2, x1:x2]
    return cv2.resize(c, (max(1, (x2 - x1) * scale), max(1, (y2 - y1) * scale)),
                      interpolation=cv2.INTER_NEAREST)


def build(case_id):
    c = gtdata.find_case(case_id, gtdata.DEFAULT_DATA_ROOT)
    if c is None:
        print(f"{case_id}: not found")
        return
    doc = gtdata.load_json(c.json_path)
    img = cv2.imread(c.png_path)
    H = img.shape[0]
    boxes = tb.detect_boxes(img)
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}

    items = []  # (role, text, crop)
    # designators / values：锚 = 元件框
    for cid, comp in comps.items():
        bx = comp["bbox"]
        r = (bx[0], H - bx[3], bx[2], H - bx[1])
        got = nearest_box(boxes, 0, 0, r=r)
        if got is None:
            continue
        nm = comp.get("Name")
        if nm:
            cr = crop_upscale(img, got[1])
            if cr is not None:
                items.append(("desig", nm, cr, got[0]))
        val = comp.get("value")
        if val:
            cr = crop_upscale(img, got[1])
            if cr is not None:
                items.append(("value", val, cr, got[0]))
    # pin names / numbers：锚 = 引脚点（图像坐标）
    for cid, pp in pins.items():
        for pk, pv in pp.items():
            pt = pv["point"]
            px, py = pt["x"], H - pt["y"]
            got = nearest_box(boxes, px, py)
            if got is None:
                continue
            nm = pv.get("pinname")
            if nm:
                cr = crop_upscale(img, got[1])
                if cr is not None:
                    items.append(("pinname", nm, cr, got[0]))
            num = pk.split("_", 1)[1] if "_" in pk else pk
            cr = crop_upscale(img, got[1])
            if cr is not None:
                items.append(("pinnum", num, cr, got[0]))

    if not items:
        print(f"{case_id}: no items")
        return

    maxw = max(it[2].shape[1] for it in items)
    canvas_w = maxw + 380
    canvas_h = len(items) * (ROW_H + 4) + 30
    canv = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    d = ImageDraw.Draw(canv)
    d.text((4, 6), f"{case_id}  {c.split}  {img.shape[1]}x{H}  boxes={len(boxes)}  items={len(items)}",
           fill=(0, 0, 0))
    y = 26
    for role, text, cr, dist in items:
        pil = Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
        if pil.height > ROW_H:
            pil = pil.resize((max(1, int(pil.width * ROW_H / pil.height)), ROW_H),
                             Image.NEAREST)
        canv.paste(pil, (4, y))
        d.text((maxw + 12, y + 8), f"{role:<8} {text[:40]!r:<44} d={dist:5.1f} {cr.shape[1]//SCALE}x{cr.shape[0]//SCALE}",
               fill=(0, 0, 0))
        y += ROW_H + 4
    p = os.path.join(OUT, f"ocr_crops_{case_id}.png")
    canv.save(p)
    print(f"{case_id}: {len(items)} items -> {p}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        build(a)
