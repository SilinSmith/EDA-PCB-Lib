"""Verify Y:\\workspace\\EDA\\PCB训练集 before it is handed to the team's trainer.

Checks (any failure is fatal):
  1. every image has a label and vice versa
  2. every label line is `cls cx cy w h`, cls within range, all four values in (0, 1]
  3. every box, denormalised, lies inside its image
  4. no source sheet contributes to both train and val
Also renders a contact sheet of random boxes so a human can eyeball the labels.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(r"Y:\workspace\EDA\PCB训练集")
PREVIEW = Path(r"Y:\workspace\EDA\work\preview")
STEM_SPLIT = "__t"


def main() -> int:
    names = [ln.strip() for ln in (OUT / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"classes         : {len(names)}")
    cfg = (OUT / "data.yaml").read_text(encoding="utf-8").splitlines()
    nc_line = next(l for l in cfg if l.startswith("nc:"))
    print(f"{nc_line}  (classes.txt is {'consistent' if int(nc_line.split(':')[1]) == len(names) else 'INCONSISTENT'})")

    bad = Counter()
    per_class = Counter()
    boxes_total = 0
    source_owner: dict[str, str] = {}
    leak = 0
    preview_pool: list[tuple] = []

    for split in ("train", "val"):
        idir, ldir = OUT / "images" / split, OUT / "labels" / split
        imgs = {p.stem: p for p in idir.iterdir() if p.is_file()}
        lbls = {p.stem: p for p in ldir.iterdir() if p.is_file()}
        if set(imgs) != set(lbls):
            bad["image/label mismatch"] += len(set(imgs) ^ set(lbls))
        print(f"{split:6} images={len(imgs):4} labels={len(lbls):4}")
        for stem, ip in imgs.items():
            source = stem.split(STEM_SPLIT)[0]
            if source_owner.setdefault(source, split) != split:
                leak += 1
                bad[f"leak:{source}"] += 1
            with Image.open(ip) as im:
                w, h = im.size
            lines = [l for l in lbls[stem].read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                bad["empty label"] += 1
                continue
            drawn = []
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    bad["field count"] += 1
                    continue
                try:
                    c = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                except ValueError:
                    bad["non-numeric"] += 1
                    continue
                if not (0 <= c < len(names)):
                    bad["class out of range"] += 1
                    continue
                if not (0 < cx <= 1 and 0 < cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    bad["normalised out of range"] += 1
                    continue
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                if x1 < -0.5 or y1 < -0.5 or x2 > w + 0.5 or y2 > h + 0.5:
                    bad["box outside image"] += 1
                    continue
                per_class[names[c]] += 1
                boxes_total += 1
                drawn.append((names[c], x1, y1, x2, y2))
            preview_pool.append((ip, drawn, split, stem))

    print(f"boxes           : {boxes_total}")
    print(f"train/val leaks : {leak}")
    print("=== failures ===")
    print("  none" if not bad else "\n".join(f"  {v:6}  {k}" for k, v in bad.most_common(20)))

    print("=== per class (top 20) ===")
    for k, v in per_class.most_common(20):
        print(f"  {v:6}  {k}")
    missing = [n for n in names if n not in per_class]
    print(f"classes with 0 boxes in this set: {missing}")

    # contact sheet
    PREVIEW.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    picks = rng.sample(preview_pool, 6)
    tiles = []
    for ip, drawn, split, stem in picks:
        with Image.open(ip) as im:
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            for cls, x1, y1, x2, y2 in drawn:
                d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                d.text((x1 + 2, y1 + 2), cls, fill=(0, 90, 255))
            im.thumbnail((640, 640))
            tiles.append((stem, im))
    W = max(t.width for _, t in tiles)
    H = sum(t.height for _, t in tiles)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for _, t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    p = PREVIEW / "label_contact_sheet.png"
    sheet.save(p)
    print(f"contact sheet   : {p}  ({sheet.width}x{sheet.height})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
