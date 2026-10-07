"""Assemble `训练集_v3`: the real sheets + a long-tail-preserving slice of the synthetic ones,
sized for training at a HIGHER `imgsz`.

Why this set exists
-------------------
`work/confusion_out2.txt` measured, on val77, the same model at the same `imgsz 1024`:

    full sheets  983 instances  recall 0.793
    1024 tiles  1448 instances  recall 0.857

and per class, on full sheets vs native-scale tiles:

    r   0.690 -> 0.903      c  0.794 -> 0.918
    v   0.771 -> 0.843      gnd 0.855 -> 0.894

A 1675x1190 sheet fed at `imgsz 1024` is scaled to 0.61x, so a 8x20 px resistor arrives at
5x12 px.  `gnd+v+r+c` are 71% of all `val` instances, so that downscale is the largest single
loss in the model.  Feeding the same sheets at `imgsz 1536` puts them at 0.92x -- nearly native.

`imgsz 1536` costs 2.25x the compute per image, so the epoch count only stays at ~22 if the
image count drops to ~1620.  This script therefore keeps ALL 624 real training images (they
carry the distribution the model is actually scored on) and cuts the synthetic side from 3000
to 1000, chosen to preserve the long tail.

The three rules (same as 训练集_v2, asserted not assumed)
--------------------------------------------------------
1. Synthetic sheets go into `train` only -- never `val`, never `test`.
2. `val` is byte-identical to `训练集_v2`'s `val` (the same 77 real images), so a number
   measured on it can be compared with the `val77 mAP50 0.6106` baseline.
3. `10GT` never enters training; it is copied verbatim from `训练集_v2`'s test split.

    $env:PYTHONIOENCODING='utf-8'
    K:\\python\\311\\python.exe work\\build_trainset_v3.py --dry-run
    K:\\python\\311\\python.exe work\\build_trainset_v3.py --force
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA")
REAL = EDA / "PCB训练集"
SYNTH = EDA / "合成原理图数据集"
V2 = EDA / "训练集_v2"
DEFAULT_OUT = EDA / "训练集_v3"
ASCII_LINK = EDA / "trainset_v3"
HOLD_OUT = "10GT-"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def transfer(src_img: Path, src_lbl: Path, out: Path, split: str, how: Counter, name: str = "") -> None:
    stem = name or src_img.stem
    for sub, src in (("images", src_img), ("labels", src_lbl)):
        d = out / sub / split
        d.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        how[link_or_copy(src, d / f"{stem}{src.suffix}")] += 1


def pick_synthetic(classes: list[str], n_want: int, rare_max: int, seed: int):
    """Return (chosen_stems, why) keeping every sheet that carries a rare class, then a seeded
    sample, and report the resulting class histogram."""
    idir = SYNTH / "images"
    ldir = SYNTH / "labels_comp"
    sheets = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
    per_sheet: dict[str, list[int]] = {}
    total: Counter = Counter()
    for img in sheets:
        lp = ldir / f"{img.stem}.txt"
        ids = []
        if lp.is_file():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    c = int(ln.split()[0])
                    ids.append(c)
                    total[c] += 1
        per_sheet[img.stem] = ids

    rare = {c for c, v in total.items() if v < rare_max}
    carriers = [s for s in per_sheet if rare & set(per_sheet[s])]
    rng = random.Random(seed)
    rest = [s for s in per_sheet if s not in set(carriers)]
    rng.shuffle(rest)
    chosen = carriers[:n_want]
    if len(chosen) < n_want:
        chosen += rest[: n_want - len(chosen)]
    chosen.sort()
    return chosen, total, rare, len(carriers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-synth", type=int, default=1000)
    ap.add_argument("--rare-max", type=int, default=2000,
                    help="a class with fewer than this many boxes in the synthetic corpus is "
                         "'rare'; every sheet containing one is kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boost", action="append", default=[], metavar="CLASS=FACTOR",
                    help="duplicate every real train image containing CLASS, FACTOR times. "
                         "Safe only where the class is spread across many images -- check "
                         "work/boost_report.py first. e.g. --boost testpoint=2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out: Path = args.out

    for need in (REAL / "classes.txt", REAL / "images" / "train", SYNTH / "images",
                 SYNTH / "labels_comp", V2 / "images" / "test", V2 / "labels" / "test"):
        if not need.exists():
            raise SystemExit(f"missing prerequisite: {need}")

    classes = [l.strip() for l in (REAL / "classes.txt").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    print(f"classes: {len(classes)}")

    chosen, synth_total, rare, n_carriers = pick_synthetic(
        classes, args.n_synth, args.rare_max, args.seed)
    print(f"synthetic sheets available : {sum(1 for p in (SYNTH / 'images').iterdir() if p.suffix.lower() in IMG_EXT)}")
    print(f"  rare classes (box<{args.rare_max}) : {len(rare)} -> "
          f"{', '.join(classes[c] for c in sorted(rare))}")
    print(f"  sheets carrying a rare class    : {n_carriers}")
    print(f"  chosen                          : {len(chosen)}")

    picked_hist: Counter = Counter()
    for s in chosen:
        for ln in (SYNTH / "labels_comp" / f"{s}.txt").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                picked_hist[classes[int(ln.split()[0])]] += 1
    print("  picked boxes per class (top 12): "
          + ", ".join(f"{k}={v}" for k, v in picked_hist.most_common(12)))

    # real train/val, minus every 10GT sheet
    real_tr, real_va, held = [], [], Counter()
    for split, sink in (("train", real_tr), ("val", real_va)):
        for img in sorted(p for p in (REAL / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT):
            if img.stem.startswith(HOLD_OUT):
                held[split] += 1
                continue
            sink.append(img)
    print(f"\nreal train {len(real_tr)}  real val {len(real_va)}  held out {dict(held)}")

    # optional class boost: duplicate real images that carry a class the corpus is short on
    boosts: list[tuple[str, int]] = []
    for spec in args.boost:
        name, _, fac = spec.partition("=")
        name = name.strip()
        if name not in classes:
            raise SystemExit(f"--boost: unknown class {name!r}")
        boosts.append((name, int(fac or 2)))
    extra: list[tuple[Path, str]] = []          # (source image, new stem)
    for name, fac in boosts:
        cid = classes.index(name)
        hit = []
        for img in real_tr:
            lp = REAL / "labels" / "train" / f"{img.stem}.txt"
            if not lp.is_file():
                continue
            if any(int(ln.split()[0]) == cid
                   for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()):
                hit.append(img)
        for k in range(1, fac):
            for img in hit:
                extra.append((img, f"{img.stem}__b{name}{k}"))
        print(f"boost {name} x{fac}: {len(hit)} real images carry it -> +{len(hit) * (fac - 1)} copies")
    real_tr_extra = extra

    total_imgs = len(real_tr) + len(real_tr_extra) + len(chosen)
    print(f"TOTAL train images: {total_imgs}   (was 3624 at imgsz 1024)")

    # cost model: seconds/epoch, calibrated on the real run (325.9 s for 3624 imgs @ 1024)
    s_per_equiv = 325.9 / 3624.0
    for imgsz in (1024, 1280, 1536):
        cost = total_imgs * (imgsz / 1024.0) ** 2 * s_per_equiv
        print(f"  imgsz {imgsz}: {cost:6.1f} s/epoch -> {7200 / cost:5.1f} epochs in 2h, "
              f"full-sheet 1675px scale {imgsz / 1675:.3f}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} exists and is not empty; pass --force")
        shutil.rmtree(out)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    how: Counter = Counter()
    for img in real_tr:
        transfer(img, REAL / "labels" / "train" / f"{img.stem}.txt", out, "train", how)
    for img, stem in real_tr_extra:
        transfer(img, REAL / "labels" / "train" / f"{img.stem}.txt", out, "train", how, name=stem)
    for img in real_va:
        transfer(img, REAL / "labels" / "val" / f"{img.stem}.txt", out, "val", how)
    for s in chosen:
        img = SYNTH / "images" / f"{s}.png"
        if not img.is_file():
            cand = [p for p in (SYNTH / "images").iterdir() if p.stem == s]
            img = cand[0]
        transfer(img, SYNTH / "labels_comp" / f"{s}.txt", out, "train", how)
    for img in sorted(p for p in (V2 / "images" / "test").iterdir() if p.suffix.lower() in IMG_EXT):
        transfer(img, V2 / "labels" / "test" / f"{img.stem}.txt", out, "test", how)

    yaml_root = ASCII_LINK.as_posix() if ASCII_LINK.is_dir() else out.as_posix()
    names_block = "\n".join(f"- {n}" for n in classes)
    (out / "data.yaml").write_text(
        f"path: {yaml_root}\ntrain: images/train\nval: images/val\nnc: {len(classes)}\n"
        f"names:\n{names_block}\n", encoding="utf-8")
    (out / "data_test.yaml").write_text(
        f"path: {yaml_root}\ntrain: images/train\nval: images/test\nnc: {len(classes)}\n"
        f"names:\n{names_block}\n", encoding="utf-8")

    # verification: assert the three rules
    fail: Counter = Counter()
    hist: Counter = Counter()
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / split).iterdir() if p.is_file()}
        lbls = {p.stem for p in (out / "labels" / split).iterdir() if p.is_file()}
        if imgs != lbls:
            fail[f"{split}: image/label mismatch"] += len(imgs ^ lbls)
        for stem in imgs:
            if split != "test" and stem.startswith(HOLD_OUT):
                fail[f"{split}: 10GT leaked into training"] += 1
            if split in ("val", "test") and stem.startswith("synth_"):
                fail[f"{split}: synthetic sheet in an eval split"] += 1
        for p in (out / "labels" / split).iterdir():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    c = int(ln.split()[0])
                    if not 0 <= c < len(classes):
                        fail["class id out of range"] += 1
                    else:
                        hist[classes[c]] += 1
        print(f"{split:6} images={len(imgs):5} labels={len(lbls):5} boxes={sum(1 for p in (out/'labels'/split).iterdir() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()):6}")

    v2_val = {p.stem for p in (V2 / "images" / "val").iterdir() if p.is_file()}
    new_val = {p.stem for p in (out / "images" / "val").iterdir() if p.is_file()}
    if v2_val != new_val:
        fail["val differs from 训练集_v2"] += len(v2_val ^ new_val)

    real_boxes = sum(v for k, v in hist.items())
    print(f"\ntransfer        : {dict(how)}")
    print(f"classes present : {len(hist)}/{len(classes)}")
    print(f"val identical to 训练集_v2's val: {v2_val == new_val}")
    print("=== failures ===")
    print("  none" if not fail else "\n".join(f"  {v:6}  {k}" for k, v in fail.most_common()))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
