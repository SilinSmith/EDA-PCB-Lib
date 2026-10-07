"""检查弱监督标注的质量：随机抽 manifest 里的行，裁图 + 显示「声称的真值」。

跑法（CWD = Y:\\workspace\\EDA\\eda-pcb）：
    $env:PYTHONIOENCODING='utf-8'
    python Y:\\workspace\\EDA\\work\\_ocr_verify.py --manifest <path> --per-role 25 --out <png>
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import cv2
from PIL import Image, ImageDraw

from schematic_parser import data as gtdata

sys.stdout.reconfigure(encoding="utf-8")

SCALE = 5
ROW_H = 34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--per-role", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-ncand", type=int, default=0, help="只取 n_cand<= 这个值的行，0=不筛")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    if a.only_ncand:
        rows = [r for r in rows if r["n_cand"] <= a.only_ncand]
    by = defaultdict(list)
    for r in rows:
        by[r["role"]].append(r)

    cache = {}
    picked = []
    for role in ("designator", "value", "pin_number", "pin_name"):
        rs = by.get(role, [])
        rnd.shuffle(rs)
        picked.extend(rs[: a.per_role])

    items = []
    for r in picked:
        key = (r["case"], r["split"])
        if key not in cache:
            c = gtdata.find_case(r["case"], gtdata.DEFAULT_DATA_ROOT)
            cache[key] = cv2.imread(c.png_path)
        img = cache[key]
        x1, y1, x2, y2 = [int(round(v)) for v in r["bbox"]]
        x1 = max(0, x1 - 2)
        y1 = max(0, y1 - 2)
        x2 = min(img.shape[1], x2 + 2)
        y2 = min(img.shape[0], y2 + 2)
        if x2 <= x1 or y2 <= y1:
            continue
        cr = cv2.resize(img[y1:y2, x1:x2],
                        ((x2 - x1) * SCALE, (y2 - y1) * SCALE), interpolation=cv2.INTER_NEAREST)
        items.append((r, cr))

    maxw = max(cr.shape[1] for _, cr in items)
    W = maxw + 470
    canv = Image.new("RGB", (W, len(items) * (ROW_H + 4) + 26), (255, 255, 255))
    d = ImageDraw.Draw(canv)
    d.text((4, 5), f"{os.path.basename(a.manifest)}  {len(items)} rows  (每 role {a.per_role})",
           fill=(0, 0, 0))
    y = 24
    for r, cr in items:
        pil = Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
        if pil.height > ROW_H:
            pil = pil.resize((max(1, int(pil.width * ROW_H / pil.height)), ROW_H), Image.NEAREST)
        canv.paste(pil, (4, y))
        d.text((maxw + 12, y + 4),
               f"{r['role']:<11}{r['case']:<6}{r['text'][:22]!r:<26}"
               f"d={r['dist']:5.1f} nc={r['n_cand']:<3}{r['w']:.0f}x{r['h']:.0f}",
               fill=(0, 0, 0))
        y += ROW_H + 4
    canv.save(a.out)
    print(f"{len(items)} rows -> {a.out}")


if __name__ == "__main__":
    main()
