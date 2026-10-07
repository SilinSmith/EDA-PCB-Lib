"""Assemble `训练集_v2`: real sheets + 3000 synthetic sheets, with an honest held-out test.

Layout
------
    images/train  = real train  (minus every 10GT sheet)  +  all synthetic sheets
    images/val    = real val    (minus every 10GT sheet)          <- same 91-ish sheets as
                                                                     the previous run, so the
                                                                     new mAP is comparable
    images/test   = the 10 `10GT` sheets, converted from their competition GT

The three rules this script exists to enforce
---------------------------------------------
1. **Synthetic sheets go into `train` only.** Never `val`, never `test`. A generator that is
   also the examiner proves nothing.
2. **`val` must not change between runs.** Otherwise the new mAP cannot be compared with the
   `mAP50 0.6049` run that already exists.
3. **`10GT` is held out of training entirely.** It is the only real set that a model trained on
   this manifest has never seen -- `10_GTcase` duplicates `200_train_cases`, and md5 proved the
   10 `10GT` sheets are not duplicates of anything else (`work/10gt_leak_check.json`).

Hardlinks are used where the filesystem allows it: same volume, so it costs no space and the
files are immutable once written. Anything that cannot be linked is copied.

    $env:PYTHONIOENCODING='utf-8'
    K:\\python\\311\\python.exe work\\build_trainset_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA")
REAL = EDA / "PCB训练集"
SYNTH = EDA / "合成原理图数据集"
GT_ROOT = EDA / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
DEFAULT_OUT = EDA / "训练集_v2"
ASCII_LINK = EDA / "trainset_v2"  # so data.yaml can stay pure ASCII
HOLD_OUT = "10GT-"                # every real sample whose case key starts with this is held out
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


def transfer(src_img: Path, src_lbl: Path, out: Path, split: str, how: Counter) -> None:
    for sub, src in (("images", src_img), ("labels", src_lbl)):
        d = out / sub / split
        d.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        how[link_or_copy(src, d / src.name)] += 1


def gt_to_yolo(doc: dict, w: int, h: int, class_id: dict[str, int]) -> tuple[list[str], int]:
    """Competition GT -> YOLO rows. `bbox` is [x1,y1,x2,y2] in the LEFT-BOTTOM origin, so the
    top edge in image coordinates is `h - y2`."""
    lines = []
    skipped = 0
    for comp in doc.get("components", {}).values():
        ctype = comp.get("type")
        if ctype not in class_id:
            skipped += 1
            continue
        box = comp.get("bbox")
        if not box or len(box) != 4:
            skipped += 1
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            skipped += 1
            continue
        cx = (x1 + x2) / 2 / w
        cy = (h - (y1 + y2) / 2) / h
        lines.append(f"{class_id[ctype]} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out: Path = args.out

    for need in (REAL / "classes.txt", REAL / "images" / "train", SYNTH / "images", GT_ROOT):
        if not need.exists():
            raise SystemExit(f"missing prerequisite: {need}")
    n_synth_expected = len([p for p in (SYNTH / "images").iterdir() if p.suffix.lower() in IMG_EXT])
    if n_synth_expected == 0:
        raise SystemExit(f"no synthetic sheets in {SYNTH / 'images'} -- run gen_synth_batch.py first")

    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} exists and is not empty; pass --force")
        shutil.rmtree(out)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    classes = [l.strip() for l in (REAL / "classes.txt").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    class_id = {n: i for i, n in enumerate(classes)}
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    how: Counter = Counter()
    held_real = Counter()
    real_counts = Counter()

    # ---- real train/val, minus every 10GT sheet -------------------------------------------
    for split in ("train", "val"):
        idir = REAL / "images" / split
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
            if img.stem.startswith(HOLD_OUT):
                held_real[split] += 1
                continue
            transfer(img, REAL / "labels" / split / f"{img.stem}.txt", out, split, how)
            real_counts[split] += 1

    # ---- synthetic, train only ------------------------------------------------------------
    synth_counts = Counter()
    for img in sorted(p for p in (SYNTH / "images").iterdir() if p.suffix.lower() in IMG_EXT):
        transfer(img, SYNTH / "labels_comp" / f"{img.stem}.txt", out, "train", how)
        synth_counts["train"] += 1

    # ---- the held-out test set, converted from the competition GT -------------------------
    test_counts = 0
    skipped_types: Counter = Counter()
    for case in sorted(GT_ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not case.is_dir():
            continue
        imgs = [p for p in case.iterdir() if p.suffix.lower() in IMG_EXT]
        jsons = [p for p in case.iterdir() if p.suffix.lower() == ".json"]
        if len(imgs) != 1 or len(jsons) != 1:
            raise SystemExit(f"{case}: expected exactly one image and one json, got {imgs} {jsons}")
        img, js = imgs[0], jsons[0]
        from PIL import Image

        with Image.open(img) as im:
            w, h = im.size
        doc = json.loads(js.read_text(encoding="utf-8"))
        lines, skipped = gt_to_yolo(doc, w, h, class_id)
        for comp in doc.get("components", {}).values():
            if comp.get("type") not in class_id:
                skipped_types[comp.get("type")] += 1
        if skipped:
            skipped_types["(degenerate/other)"] += skipped
        name = f"10GT-{case.name}"
        dst_img = out / "images" / "test" / f"{name}{img.suffix.lower()}"
        link_or_copy(img, dst_img)
        (out / "labels" / "test" / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        test_counts += 1
        print(f"  test {name:10} {w}x{h}  {len(lines)} boxes ({len(doc.get('components', {}))} comps)")

    # ---- manifests --------------------------------------------------------------------------
    yaml_root = ASCII_LINK.as_posix() if ASCII_LINK.is_dir() else out.as_posix()
    names_block = "\n".join(f"- {n}" for n in classes)
    (out / "data.yaml").write_text(
        f"path: {yaml_root}\ntrain: images/train\nval: images/val\nnc: {len(classes)}\n"
        f"names:\n{names_block}\n", encoding="utf-8")
    (out / "data_test.yaml").write_text(
        f"path: {yaml_root}\ntrain: images/train\nval: images/test\nnc: {len(classes)}\n"
        f"names:\n{names_block}\n", encoding="utf-8")

    # ---- verification: the three rules, asserted not assumed --------------------------------
    fail: Counter = Counter()
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
        print(f"{split:6} images={len(imgs):5} labels={len(lbls):5}")

    hist: Counter = Counter()
    for split in ("train", "val", "test"):
        for p in (out / "labels" / split).iterdir():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    c = int(ln.split()[0])
                    if not 0 <= c < len(classes):
                        fail["class id out of range"] += 1
                    else:
                        hist[classes[c]] += 1

    print(f"\nsynthetic sheets : {synth_counts['train']} (expected {n_synth_expected})")
    print(f"real train       : {real_counts['train']}")
    print(f"real val         : {real_counts['val']}")
    print(f"test (10GT)      : {test_counts}")
    print(f"held out of real : {dict(held_real)}")
    print(f"transfer         : {dict(how)}")
    print(f"classes present  : {len(hist)}/{len(classes)}")
    print("=== failures ===")
    print("  none" if not fail else "\n".join(f"  {v:6}  {k}" for k, v in fail.most_common()))
    if skipped_types:
        print(f"test GT dropped  : {dict(skipped_types)}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
