"""Regenerate `eda-pcb/synth/profiles.py` from the real 213-sheet corpus.

Rationale: width, height and component density are *not* independent in the real data. A
5100x3296 datasheet scan carries ~66 components (3.9 per megapixel) because most of the page is
a figure; a 1190x845 CAD sheet carries ~56 (55 per megapixel). Rolling them separately produces
canvases the detector will never be tested on, so we sample a measured triple instead.

    $env:PYTHONIOENCODING='utf-8'
    K:\\python\\311\\python.exe work\\make_sheet_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

REAL = Path(r"Y:\workspace\EDA\PCB训练集")
OUT = Path(r"Y:\workspace\EDA\eda-pcb\synth\profiles.py")
# Supersample 3 of this is a 26-megapixel canvas. The cap is deliberately just above the
# 2222x1199..2390x1190 band, because those sizes are what the competition test sheets look like.
MPX_CAP = 2_900_000


def main() -> int:
    rows: list[tuple[int, int, int]] = []
    for split in ("train", "val"):
        for p in (REAL / "images" / split).iterdir():
            if "__t" in p.stem:  # tiles are crops, not whole sheets
                continue
            with Image.open(p) as im:
                w, h = im.size
            n = len([l for l in (REAL / "labels" / split / f"{p.stem}.txt")
                     .read_text(encoding="utf-8").splitlines() if l.strip()])
            rows.append((w, h, n))
    rows.sort()

    kept = [r for r in rows if r[0] * r[1] <= MPX_CAP and r[2] >= 4]
    dropped = [r for r in rows if r not in kept]
    if not kept:
        raise SystemExit("nothing kept -- corpus path wrong?")

    print(f"total {len(rows)}  kept {len(kept)}  dropped {len(dropped)}")
    print(f"kept   : comps/sheet {sum(r[2] for r in kept) / len(kept):.1f}"
          f"  comps/Mpx {sum(r[2] / (r[0] * r[1] / 1e6) for r in kept) / len(kept):.1f}"
          f"  mean canvas {sum(r[0] * r[1] for r in kept) / len(kept) / 1e6:.3f} Mpx")
    print(f"dropped: {[(w, h, n) for w, h, n in dropped]}")

    doc = '''"""Empirical sheet profiles, one per distinct real schematic in the 213-sheet corpus.

Each entry is (width, height, n_components) of an actual labelled sheet. Sampling a whole triple
rather than rolling width, height and density separately keeps the *joint* distribution honest:
real pages are not uniformly dense. A 5100x3296 datasheet scan carries only ~66 components
(3.9 per megapixel) because most of the page is a figure, while a 1190x845 CAD sheet carries
~56 (55 per megapixel). Independent sampling gets that wrong in both directions, and the
detector then sees a canvas distribution it will never be tested on.

Sheets above the render cap (2.6 Mpx -- {dropcount} of the {total}) are excluded: at
supersample 3 they would need a 150-megapixel canvas. Their source style is still covered by the
palettes, and the largest profile kept is {maxw}x{maxh}.

Regenerate with `work/make_sheet_profiles.py`; do not hand-edit.
"""

SHEET_PROFILES: list[tuple[int, int, int]] = [
'''.format(dropcount=len(dropped), total=len(rows), maxw=max(r[0] for r in kept),
           maxh=max(r[1] for r in kept))
    body = "\n".join(f"    ({w}, {h}, {n})," for w, h, n in kept)
    OUT.write_text(doc + body + "\n]\n", encoding="utf-8")
    print(f"wrote {OUT}  ({len(kept)} profiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
