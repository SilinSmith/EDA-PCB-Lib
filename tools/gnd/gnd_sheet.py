"""Contact sheet of real gnd crops, per source family -- is gnd ONE shape?"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

TILE = 40
PAD = 4
PER_SRC = 14
OUT = Path(r"Y:\workspace\EDA\work\preview\gnd_glyphs.png")
SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def crop(g, box, h):
    x1, y1, x2, y2 = box
    iy1, iy2 = h - y2, h - y1
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    iy1, iy2 = sorted((int(round(iy1)), int(round(iy2))))
    if x2 - x1 < 3 or iy2 - iy1 < 3:
        return None
    sub = g[max(0, iy1):min(g.shape[0], iy2), max(0, x1):min(g.shape[1], x2)]
    if sub.size == 0:
        return None
    return Image.fromarray(sub).convert("L").resize((TILE, TILE), Image.BILINEAR)


def main() -> None:
    got: dict[str, list[Image.Image]] = defaultdict(list)
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        src = source_of(c.json_path)
        if len(got[src]) >= PER_SRC:
            continue
        try:
            doc = S.load_json(c.json_path)
            if not (doc.get("components") or {}):
                continue
        except Exception:
            continue
        g = None
        for comp in (doc.get("components") or {}).values():
            if str(comp.get("type", "")).strip() != "gnd":
                continue
            box = comp.get("bbox")
            if not box or len(box) != 4:
                continue
            if g is None:
                try:
                    g = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
                except Exception:
                    break
            im = crop(g, [float(v) for v in box], g.shape[0])
            if im is not None and len(got[src]) < PER_SRC:
                got[src].append(im)

    rows = [s for s in SRC_ORDER if got.get(s)]
    if not rows:
        print("no gnd crops")
        return
    cols = max(len(got[s]) for s in rows)
    W = cols * (TILE + PAD) + PAD
    H = len(rows) * (TILE + PAD + 16) + PAD
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for r, s in enumerate(rows):
        y = PAD + r * (TILE + PAD + 16) + 14
        d.text((PAD, y - 13), f"{s}  (n={len(got[s])})", fill=(0, 0, 0))
        for cix, im in enumerate(got[s]):
            sheet.paste(im.convert("RGB"), (PAD + cix * (TILE + PAD), y))
    sheet.save(OUT)
    print(f"wrote {OUT}  {sheet.size[0]}x{sheet.size[1]}")
    for s in rows:
        print(f"  {s:18s} {len(got[s])} crops")


if __name__ == "__main__":
    main()
