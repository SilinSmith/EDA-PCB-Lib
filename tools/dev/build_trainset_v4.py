# -*- coding: utf-8 -*-
"""Assemble `work/trainset_v4` = `trainset_v3` (optionally label-repaired) + `新补充训练集`.

Why
---
Owner authorised a retrain and supplied a new hand-made corpus at
`Y:\\workspace\\EDA\\新补充训练集` (58 sheets, 4720 components) that must be merged into the
existing training set.  The two corpora do **not** use the same bbox convention, so a naive
copy would teach the detector two different box shapes for the same class:

    official GT (`yolo_part/Dataset(waitforpreprocess/**/*.json`)
        bbox = [x1, y1, x2, y2]  ABSOLUTE px, origin BOTTOM-LEFT, and shape is a per-class
        FIXED stamp (`r` 20x8, `c` 20x16, `gnd` 20x19, `l` 34x6 ... — see
        `schematic_parser/stamp.py:STAMP_WH`, 100% unique per class over 220 cases)

    新补充训练集 (`新补充训练集/*/*.json`)
        bbox = [cx, cy, w, h]  NORMALISED 0..1, origin TOP-LEFT  (measured, see
        `work/probe_newbbox2.py`: ink-in 0.2115 vs 0.1158, centroid drift 0.059 vs 0.163,
        cover 0.223 vs 0.130 for the competing [x,y,w,h] reading; cross-checked by hand on
        crops in `work/_probe_bbox/`), shape HUGS the drawn symbol and therefore varies with
        sheet resolution (sheets are 1241..4698 px wide).

So the conversion is: absolute -> `[cx, cy, w, h]` absolute -> **snap (long, short) to
`STAMP_WH[cls]`**, keeping the centre and taking the axis from the measured aspect.  That is
exactly what stage 1 does to every prediction at inference time (`stage1/detect.py:9-12`
"贴规范图章"), so the new targets are expressed in the same language as the old ones.

Label repair (owner: "上一次 YOLO 训练的时候有大量的训练集中的问题 ...错误的标号之类的")
----------------------------------------------------------------------------------------
`--fix-ink {off,recenter,drop}` handles the L1 defect measured in
`eda-pcb/docs/reports/漏检错检_归因.md:179` (4.9% of GT boxes sit on a stroke gap: ink
inside the box == 0 but ink within 25 px of the centre != 0).  `recenter` moves the box
centre to the ink centroid of the blob nearest the box centre; `drop` removes the box.

`--snap {off,new,new+train}` controls template snapping.  `new` (default) touches only the
new corpus — `trainset_v3/val` stays byte-identical so any val77 mAP50 is still comparable
with the recorded 0.6208 baseline.  `new+train` also rewrites the old train labels, which
is a *train-target shaping experiment*: it cannot be compared on val77 unless val is snapped
too, so it is off by default.

`--merge-classes pin,testpoint` emits a second variant whose labels collapse those classes
into one (`pin_testpoint`), for the stage-1 class-merge experiment.  **This variant needs a
code change before it can score**: `stage1/detect.py:97` defaults
`STAGE1_DROP_TYPES=pin,testpoint,mosfet_npn`, and `cvparts/testpoint.py:88` /
`stage2/assign.py:227` compare against the literal names `pin` / `testpoint`.

    $env:PYTHONIOENCODING='utf-8'
    cd eda-pcb
    python tools\\dev\\build_trainset_v4.py --dry-run
    python tools\\dev\\build_trainset_v4.py --force --fix-ink recenter
    python tools\\dev\\build_trainset_v4.py --force --merge-classes pin,testpoint --out ..\\work\\trainset_v4_merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
EDA_PCB = HERE.parents[2]              # ...\eda-pcb
EDA = HERE.parents[3]                  # ...\EDA
sys.path.insert(0, str(EDA_PCB))

from schematic_parser.stamp import STAMP_WH          # noqa: E402

V3 = EDA / "trainset_v3"               # ASCII junction -> 训练集_v3
V3_REAL = EDA / "训练集_v3"
NEW = EDA / "新补充训练集"
DEFAULT_OUT = EDA / "work" / "trainset_v4"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INK = 200                              # grey < INK counts as "ink" (same threshold family as
                                       # `docs/reports/漏检错检_归因.md` uses for zero-ink boxes)


# --------------------------------------------------------------------------- io helpers
def imread_gray(path: Path) -> np.ndarray:
    """cv2.imread cannot open non-ASCII paths on Windows; go through imdecode."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    return img


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# --------------------------------------------------------------------------- conversion
def read_new_sheet(json_path: Path) -> tuple[list[tuple[str, float, float, float, float]], int]:
    """Return [(cls, cx, cy, w, h) absolute px] + n_invalid for one 新补充 sheet."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = []
    bad = 0
    for name, comp in data.get("components", {}).items():
        cls = (comp.get("type") or "").strip()
        bb = comp.get("bbox")
        if not cls or not isinstance(bb, (list, tuple)) or len(bb) != 4:
            bad += 1
            continue
        x, y, w, h = (float(v) for v in bb)
        if not (0.0 < w < 1.0 and 0.0 < h < 1.0):
            bad += 1
            continue
        out.append((cls, x, y, w, h))
    return out, bad


def snap(cls: str, cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Replace the measured (w, h) with the class's official stamp, keeping axis + centre."""
    tpl = STAMP_WH.get(cls)
    if tpl is None:
        return cx, cy, w, h
    long_side, short_side = tpl
    if h >= w:                       # measured box is upright -> long side vertical
        bw, bh = float(short_side), float(long_side)
    else:
        bw, bh = float(long_side), float(short_side)
    return cx, cy, bw, bh


def tiles_for(w: int, h: int, tile: int, stride: int):
    if w <= tile and h <= tile:
        yield 0, 0, w, h
        return
    xs = list(range(0, max(1, w - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, h - tile + 1), stride)) or [0]
    if w > tile and xs[-1] != w - tile:
        xs.append(w - tile)
    if h > tile and ys[-1] != h - tile:
        ys.append(h - tile)
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            yield x0, y0, min(x0 + tile, w), min(y0 + tile, h)


def yolo_line(cid: int, cx: float, cy: float, w: float, h: float, W: int, H: int) -> str:
    return (f"{cid} {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}")


# --------------------------------------------------------------------------- ink repair
def ink_recenter(gray: np.ndarray, cx: float, cy: float, w: float, h: float,
                 radius: float = 25.0):
    """L1 repair: if the box interior has no ink, move the centre onto the nearest blob.

    Returns (cx, cy, status) with status in {'ok','noink_noblob','moved'}.
    """
    H, W = gray.shape
    x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x2, y2 = int(round(cx + w / 2)), int(round(cy + h / 2))
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    if x2c <= x1c or y2c <= y1c:
        return cx, cy, "noink_noblob"
    inside = gray[y1c:y2c, x1c:x2c]
    frac = float((inside < INK).mean())
    if frac >= 0.01:
        return cx, cy, "ok"
    # ring search: ink within `radius` of the box centre
    r = int(round(radius))
    X1, Y1 = max(0, int(cx) - r), max(0, int(cy) - r)
    X2, Y2 = min(W, int(cx) + r + 1), min(H, int(cy) + r + 1)
    patch = gray[Y1:Y2, X1:X2]
    mask = (patch < INK).astype(np.uint8)
    if mask.sum() < 4:
        return cx, cy, "noink_noblob"
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, bestd = None, None
    for i in range(1, n):
        mx, my = cent[i]
        d = (mx + X1 - cx) ** 2 + (my + Y1 - cy) ** 2
        if bestd is None or d < bestd:
            best, bestd = (mx + X1, my + Y1), d
    if best is None:
        return cx, cy, "noink_noblob"
    return float(best[0]), float(best[1]), "moved"


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base", type=Path, default=V3, help="existing YOLO dataset to start from")
    ap.add_argument("--new", type=Path, default=NEW, help="新补充训练集 root")
    ap.add_argument("--tile", type=int, default=1536,
                    help="tile edge for oversize new sheets (0 = never tile)")
    ap.add_argument("--max-scale", type=float, default=1.35,
                    help="tile a sheet when max(W,H) > max-scale * tile")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--snap", choices=("off", "new", "new+train"), default="new")
    ap.add_argument("--fix-ink", choices=("off", "recenter", "drop"), default="off")
    ap.add_argument("--merge-classes", default="",
                    help="comma list of classes to collapse into one, e.g. pin,testpoint")
    ap.add_argument("--merged-name", default="",
                    help="name of the merged class (default: the classes joined by '_')")
    ap.add_argument("--new-split", choices=("train", "val"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.stride = max(1, int(args.tile * (1 - args.overlap))) if args.tile else 0

    base: Path = args.base
    classes = [l.strip() for l in (base / "classes.txt").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    merged = [c.strip() for c in args.merge_classes.split(",") if c.strip()]
    for c in merged:
        if c not in classes:
            raise SystemExit(f"--merge-classes: unknown class {c!r}")
    if merged:
        name = args.merged_name or "_".join(merged)
        # The merged class takes the slot of the **first** merged class and the rest are dropped, so
        # every class id below that slot keeps its meaning.  (Appending `tp` at index 0 instead would
        # renumber all 43 classes, which throws away the per-class alignment of the pretrained head
        # we fine-tune from.)
        first = min(classes.index(c) for c in merged)
        classes_out: list[str] = []
        remap: dict[int, int] = {}
        for i, c in enumerate(classes):
            if c in merged:
                if i == first:
                    remap[i] = len(classes_out)
                    classes_out.append(name)
                else:
                    remap[i] = remap[first]
            else:
                remap[i] = len(classes_out)
                classes_out.append(c)
    else:
        classes_out = list(classes)
        remap = {i: i for i in range(len(classes))}
    print(f"base classes {len(classes)} -> out classes {len(classes_out)}"
          + (f"  (merged {merged} -> {classes_out[remap[first]]!r} at index {remap[first]})"
             if merged else ""))

    stats: Counter = Counter()
    per_class: Counter = Counter()
    new_pairs: list[tuple[Path, Path, str]] = []      # (sheet png, label txt to write later, stem)

    # Clean the target **now**, not at write time: the scan phase below writes tiled crops of the
    # new corpus into `<out>/_tmp/`, so a later `rmtree(out)` would delete the very files the
    # write phase is about to link (that was a real bug: `--force` + tiling died with
    # `FileNotFoundError` on `_tmp/new01_…_t0000_0000.png`).
    if not args.dry_run and args.out.exists() and any(args.out.iterdir()):
        if not args.force:
            raise SystemExit(f"{args.out} exists and is not empty; pass --force")
        shutil.rmtree(args.out)

    # ---- 1. new corpus -----------------------------------------------------------------
    sheets = sorted(p for p in args.new.glob("*/*.json"))
    print(f"\n新补充训练集: {len(sheets)} sheets")
    for js in sheets:
        png = js.with_suffix(".png")
        if not png.is_file():
            stats["new: missing png"] += 1
            continue
        gray = imread_gray(png)
        H, W = gray.shape
        comps, bad = read_new_sheet(js)
        stats["new: invalid bbox"] += bad
        boxes = []
        for cls, cx, cy, w, h in comps:
            if cls not in classes:
                stats[f"new: unknown class {cls}"] += 1
                continue
            cx, cy = cx * W, cy * H
            w, h = w * W, h * H
            if args.snap in ("new", "new+train"):
                before = (w, h)
                cx, cy, w, h = snap(cls, cx, cy, w, h)
                if (round(w), round(h)) != (round(before[0]), round(before[1])):
                    stats["new: size snapped to stamp"] += 1
            if args.fix_ink != "off":
                cx2, cy2, st = ink_recenter(gray, cx, cy, w, h)
                stats[f"new: ink {st}"] += 1
                if st == "moved":
                    if args.fix_ink == "drop":
                        continue
                    cx, cy = cx2, cy2
                elif st == "noink_noblob" and args.fix_ink == "drop":
                    continue
            if w < 2 or h < 2:
                stats["new: degenerate box"] += 1
                continue
            boxes.append((cls, cx, cy, w, h))
        # tile
        if args.tile and max(W, H) > args.max_scale * args.tile:
            stats["new: sheets tiled"] += 1
            for x0, y0, x1, y1 in tiles_for(W, H, args.tile, args.stride):
                keep = []
                for cls, cx, cy, w, h in boxes:
                    bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                    if bx1 >= x0 and by1 >= y0 and bx2 <= x1 and by2 <= y1:
                        keep.append((cls, cx - x0, cy - y0, w, h))
                if not keep:
                    stats["new: empty tiles dropped"] += 1
                    continue
                stem = f"new{js.parent.name}_{png.stem[:28]}_t{x0:04d}_{y0:04d}"
                sub = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_COLOR)[y0:y1, x0:x1]
                tmp = args.out / "_tmp" / f"{stem}.png"
                if not args.dry_run:
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imencode(".png", sub)[1].tofile(str(tmp))
                lines = [yolo_line(remap[classes.index(c)], a, b, cw, ch, x1 - x0, y1 - y0)
                         for c, a, b, cw, ch in keep]
                new_pairs.append((tmp, lines, stem, x1 - x0, y1 - y0))
                for c, *_ in keep:
                    per_class[c] += 1
                    stats["new: boxes kept"] += 1
        else:
            stem = f"new{js.parent.name}_{png.stem[:28]}"
            lines = [yolo_line(remap[classes.index(c)], cx, cy, w, h, W, H)
                     for c, cx, cy, w, h in boxes]
            new_pairs.append((png, lines, stem, W, H))
            for c, *_ in boxes:
                per_class[c] += 1
                stats["new: boxes kept"] += 1
        stats[f"new: sheets ok ({'tile' if (args.tile and max(W, H) > args.max_scale * args.tile) else 'whole'})"] += 1

    # ---- 2. base dataset ---------------------------------------------------------------
    base_pairs: list[tuple[Path, Path, str, str]] = []
    for split in ("train", "val", "test"):
        idir, ldir = base / "images" / split, base / "labels" / split
        if not idir.is_dir():
            continue
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
            lbl = ldir / f"{img.stem}.txt"
            if not lbl.is_file():
                stats[f"base: {split} label missing"] += 1
                continue
            base_pairs.append((img, lbl, split, img.stem))
    print(f"\nbase: train={sum(1 for p in base_pairs if p[2] == 'train')} "
          f"val={sum(1 for p in base_pairs if p[2] == 'val')} "
          f"test={sum(1 for p in base_pairs if p[2] == 'test')}")

    needs_remap = any(remap[i] != i for i in range(len(classes)))
    fixed_lines: dict[Path, list[str]] = {}
    if args.fix_ink != "off" or args.snap == "new+train" or needs_remap:
        for img, lbl, split, stem in base_pairs:
            # `--fix-ink` / `--snap new+train` are train-only repairs, but a **class remap** applies
            # to every split: linking a val/test label whose ids were renumbered would silently point
            # those boxes at the wrong classes.
            if split != "train" and not needs_remap:
                continue
            pixel_work = (split == "train"
                          and (args.fix_ink != "off" or args.snap == "new+train"))
            raw = [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not raw:
                continue
            W = H = None
            if pixel_work:
                gray = imread_gray(img)
                H, W = gray.shape
            out_lines = []
            for parts in raw:
                cid = int(parts[0])
                if not pixel_work:
                    # Class-id-only rewrite: the stored numbers are already normalised, so carrying
                    # them through a pixel round-trip would only add rounding.
                    out_lines.append(f"{remap[cid]} " +
                                     " ".join(f"{float(v):.6f}" for v in parts[1:]))
                    continue
                cx, cy, w, h = (float(v) * (W if i % 2 == 0 else H) for i, v in enumerate(parts[1:]))
                cls = classes[cid]
                if args.snap == "new+train":
                    cx, cy, w, h = snap(cls, cx, cy, w, h)
                if args.fix_ink != "off":
                    cx2, cy2, st = ink_recenter(gray, cx, cy, w, h)
                    stats[f"base: ink {st}"] += 1
                    if st == "moved":
                        if args.fix_ink == "drop":
                            stats["base: box dropped"] += 1
                            continue
                        cx, cy = cx2, cy2
                    elif st == "noink_noblob" and args.fix_ink == "drop":
                        stats["base: box dropped"] += 1
                        continue
                out_lines.append(yolo_line(remap[cid], cx, cy, w, h, W, H))
            fixed_lines[lbl] = out_lines
            stats["base: labels rewritten"] += 1

    if args.dry_run:
        print("\n=== box counts per class (v4 = base + new) ===")
        base_hist: Counter = Counter()
        new_hist: Counter = Counter()
        for img, lbl, split, stem in base_pairs:
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    base_hist[classes_out[remap[int(ln.split()[0])]]] += 1
        for c, n in per_class.items():
            new_hist[classes_out[remap[classes.index(c)]]] += n
        print(f"{'class':16s} {'v3':>7s} {'+new':>7s} {'v4':>7s}")
        for c in classes_out:
            print(f"{c:16s} {base_hist[c]:7d} {new_hist[c]:7d} {base_hist[c] + new_hist[c]:7d}")
        print("\n=== stats ===")
        for k, v in sorted(stats.items()):
            print(f"  {v:7d}  {k}")
        n_new = len(new_pairs)
        n_tiled = stats["new: sheets tiled"]
        print(f"\nnew sheets -> {n_new} images ({n_tiled} sheets tiled)")
        tot_new_boxes = sum(per_class.values())
        print(f"new boxes kept {tot_new_boxes}  (of 4720 decoded)")
        n_train = sum(1 for p in base_pairs if p[2] == "train") + n_new
        n_val = sum(1 for p in base_pairs if p[2] == "val")
        print(f"\nTOTAL train {n_train}  val {n_val}")
        # cost model off the real runs (work/_full220 + yolo_part/runs/detect/*/results.csv)
        for imgsz, s_per_img in ((1536, 89.8 / 1624.0), (1024, 119.2 / 3624.0)):
            cost = n_train * s_per_img * (imgsz / 1536.0 if imgsz == 1536 else 1.0) ** 0
            print(f"  imgsz {imgsz}: {cost:6.1f} s/epoch -> {7200 / cost:5.1f} epoch in 2h, "
                  f"{3600 / cost:5.1f} epoch/h")
        print("\n[dry-run] nothing written")
        return 0

    # ---- 3. write ----------------------------------------------------------------------
    out: Path = args.out
    # (the `--force` / rmtree guard already ran at the top of main, before `_tmp` was populated)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(classes_out) + "\n", encoding="utf-8")
    how: Counter = Counter()
    for img, lbl, split, stem in base_pairs:
        dst_img = out / "images" / split / f"{stem}{img.suffix}"
        dst_lbl = out / "labels" / split / f"{stem}.txt"
        how[link_or_copy(img, dst_img)] += 1
        if lbl in fixed_lines:
            dst_lbl.write_text("\n".join(fixed_lines[lbl]) + "\n", encoding="utf-8")
            how["label rewritten"] += 1
        else:
            how[link_or_copy(lbl, dst_lbl)] += 1
    for src, lines, stem, W, H in new_pairs:
        split = args.new_split
        dst_img = out / "images" / split / f"{stem}.png"
        how[link_or_copy(src, dst_img)] += 1
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        how["new label written"] += 1
    tmp = out / "_tmp"
    if tmp.is_dir():
        shutil.rmtree(tmp)

    names_block = "\n".join(f"- {n}" for n in classes_out)
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnc: {len(classes_out)}\n"
        f"names:\n{names_block}\n", encoding="utf-8")
    print(f"\nwrote {out}\n  transfer {dict(how)}")
    hist: Counter = Counter()
    for split in ("train", "val", "test"):
        for p in (out / "labels" / split).iterdir():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    cid = int(ln.split()[0])
                    if cid >= len(classes_out):
                        raise SystemExit(f"class id {cid} out of range in {p}")
                    hist[classes_out[cid]] += 1
    print(f"classes present {len(hist)}/{len(classes_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
