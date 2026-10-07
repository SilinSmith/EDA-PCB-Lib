"""Generate the A+C demo set: synthetic schematics with free, perfect labels.

    python synth/make_demo.py --n 24 --out "Y:\\workspace\\EDA\\合成原理图demo"

Route A: `gt/<case>.json` is competition-format ground truth, so the whole scoring pipeline can
be exercised without ever having a labelled real sheet.
Route C: `labels_text/<case>.txt` + `texts.jsonl` supervise a designator/text detector, which is
where the points actually are (a missed designator costs its component, its pins and its nets).

The component label ids come from `PCB训练集/classes.txt` when it exists, so the synthetic set
and the 763 real images share one label space and can simply be concatenated.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import emit as emit_mod  # noqa: E402
from synth.scene import GenConfig, generate  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REAL_CLASSES = Path(r"Y:\workspace\EDA\PCB训练集\classes.txt")
DEFAULT_OUT = Path(r"Y:\workspace\EDA\合成原理图demo")

PALETTE_CYCLE = ["kicad", "altium", "jlc", "datasheet"]


def load_classes() -> list[str]:
    """Reuse the real training set's class order when present, so ids 0..42 mean the same
    thing in both datasets."""
    if REAL_CLASSES.is_file():
        names = [l.strip() for l in REAL_CLASSES.read_text(encoding="utf-8").splitlines() if l.strip()]
        if names:
            return names
    print("[warn] PCB训练集/classes.txt not found -> deriving class order from TYPE_MIX")
    return [t for t, _ in __import__("synth.scene", fromlist=["x"]).TYPE_MIX]


def montage(images: list[Path], out: Path, cols: int = 4, cell: int = 420) -> Path | None:
    from PIL import Image, ImageDraw
    if not images:
        return None
    tiles = []
    for p in images:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((cell, cell))
            tiles.append((p.stem, im))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 14)), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    for i, (name, t) in enumerate(tiles):
        cx, cy = (i % cols) * cell, (i // cols) * (cell + 14)
        sheet.paste(t, (cx + (cell - t.width) // 2, cy + 14))
        d.text((cx + 4, cy + 2), name, fill=(0, 0, 0))
    sheet.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="how many sheets to generate")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--no-image", action="store_true", help="skip the PNGs (JSON/labels only)")
    ap.add_argument("--ss", type=int, default=3, help="supersampling factor")
    args = ap.parse_args()

    class_names = load_classes()
    class_id = {n: i for i, n in enumerate(class_names)}
    args.out.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "gt", "labels_comp", "labels_text"):
        (args.out / sub).mkdir(exist_ok=True)
    (args.out / "texts.jsonl").write_text("", encoding="utf-8")
    emit_mod.write_class_lists(args.out, class_names)

    rng = random.Random(args.seed)
    types: Counter = Counter()
    roles: Counter = Counter()
    misses: Counter = Counter()
    made: list[Path] = []
    rows = []

    for i in range(args.n):
        palette = PALETTE_CYCLE[i % len(PALETTE_CYCLE)]
        name = f"synth_{i + 1:04d}"
        # No width/height: `generate` draws a measured (w, h, n_components) profile from
        # `profiles.py`, which is the only way the size/density pairing stays honest.
        cfg = GenConfig(palette=palette, seed=rng.randrange(1 << 30), ss=args.ss)
        scene = generate(cfg)
        w, h = scene.w, scene.h
        em = emit_mod.emit(scene, name, class_id)
        emit_mod.write(em, args.out, save_image=not args.no_image)
        if not args.no_image:
            made.append(args.out / "images" / f"{name}.png")

        types.update(c.ctype for c in scene.comps)
        roles.update(t.role for t in scene.texts if t.content)
        dropped = len(scene.comps) - len(em.comp_labels)
        if dropped:
            misses["comp dropped by normalisation"] += dropped
        rows.append({
            "case": name, "palette": palette, "size": [w, h],
            "components": len(scene.comps), "nets": len(scene.nets),
            "edges": sum(len(n.edges) for n in scene.nets),
            "texts": len(em.texts),
            "pins_on_nets": sum(len(n.pins) for n in scene.nets),
        })
        print(f"  {name}  {palette:9} {w}x{h}  comps {len(scene.comps):3}  "
              f"nets {len(scene.nets):3}  texts {len(em.texts):4}")

    (args.out / "summary.json").write_text(
        json.dumps({"n": args.n, "seed": args.seed, "classes": class_names, "cases": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    n_comp = sum(r["components"] for r in rows)
    n_net = sum(r["nets"] for r in rows)
    n_edge = sum(r["edges"] for r in rows)
    n_text = sum(r["texts"] for r in rows)
    print("\n=== generated ===")
    print(f"sheets            : {len(rows)}")
    print(f"components        : {n_comp}   ({n_comp / len(rows):.1f} per sheet)")
    print(f"nets / edges      : {n_net} / {n_edge}   ({n_edge / n_net:.2f} edges per net)")
    print(f"text items        : {n_text}   ({n_text / len(rows):.1f} per sheet)")
    print(f"component types   : {len(types)} distinct")
    print("  " + "  ".join(f"{k}:{v}" for k, v in types.most_common(10)))
    print(f"text roles        : " + "  ".join(f"{k}:{v}" for k, v in roles.most_common()))
    if misses:
        print("  losses:", dict(misses))
    print(f"out               : {args.out}")
    if made:
        m = montage(made[:16], args.out / "montage.png")
        print(f"montage           : {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
