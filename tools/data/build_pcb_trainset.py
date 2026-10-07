"""Build Y:\\workspace\\EDA\\PCB训练集 -- a drop-in YOLO dataset for the team's component detector.

Why this shape
--------------
The task's own JSON uses a BOTTOM-LEFT origin (y_json = H - y_img) because that is what the
EasyEDA/Altium exporters emit.  YOLO does not: it wants normalised (cx, cy, w, h) measured
from the TOP-LEFT of the image.  This builder deliberately emits the YOLO convention and
flips nothing except the task's own bottom-left bboxes on the way in.

Sources (every one of them carries real, hand-checked labels)
-------------------------------------------------------------
  10GT              10 cases   official GT  components[*].bbox   (bottom-left origin)
  10_GTcase         10 cases   official GT  components[*].bbox   (bottom-left origin)
  200_train_cases  200 cases   official GT  components[*].bbox   (bottom-left origin)
  Dataset300        32 cases   CVJsonStd    components[*].realBBox (top-left origin, already)

The team's own prepare_yolo_dataset.py only ever scanned 200_train_cases, so 10GT and
10_GTcase (20 cases, and the two types missing from classes.txt) have never been trained on.

Augmentation
------------
Large sheets are cut into overlapping tiles at native resolution.  This matters because the
sheets run up to 5100x3296 while the detector is trained at 1536/640 -- without tiling a
0402-sized symbol survives the downscale as a handful of pixels.  Labels are recomputed per
tile and a tile is kept only when it fully contains at least one box.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA")
DATA_ROOT = EDA / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
D300_ROOT = EDA / "eda-pcb" / "yolo_part" / "Dataset300我不知道干嘛用的先放置"
TEAM_CLASSES = EDA / "eda-pcb" / "yolo_part" / "training_data" / "classes.txt"
OUT_ROOT = EDA / "PCB训练集"

SPLITS = ("10GT", "10_GTcase", "200_train_cases")

# The team's classes.txt order is kept byte-for-byte so their existing labels stay valid.
# The two types present in the data but missing from it are appended at the end, which is the
# only position that does not renumber anything already on disk.
EXTRA_CLASSES = ["bjt_npn", "mosfet_npn"]

# CVJsonStd `category` -> competition `type`.  Proven for Box/R/C/Battary/MOSFET by matching
# case 0059 against 10_GTcase/0059 (13/13 components, median centre distance 0.5 px).
D300_MAP = {
    "box": "box", "r": "r", "c": "c", "gnd": "gnd", "v": "v", "switch": "switch",
    "d": "d", "net_short": "net_short", "net_input": "net_input", "led": "led",
    "other": "other", "bjt": "bjt", "mosfet": "mosfet", "l": "l",
    "net_output": "net_output", "m": "motor", "crystal": "crystal", "battary": "battary",
    "fuse": "fuse", "pin": "pin", "dc": "dc", "ant": "ant", "buzzer": "buzzer",
    "amp": "amp", "jumper": "jumper", "esd": "esd", "transfomer": "transfomer",
    "opt": "opt", "block": "block",
}


def load_classes() -> list[str]:
    names = [ln.strip() for ln in TEAM_CLASSES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for extra in EXTRA_CLASSES:
        if extra not in names:
            names.append(extra)
    return names


# --------------------------------------------------------------------------- collection

def _clip(box, w, h):
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def collect_official(classes: list[str], warn: Counter):
    """Yield (split, case_id, png_path, [ (cls, x1,y1,x2,y2) ] in TOP-LEFT image coords)."""
    for split in SPLITS:
        sdir = DATA_ROOT / split
        for case_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            pngs = sorted(case_dir.glob("*.png")) + sorted(case_dir.glob("*.jpg"))
            jsons = sorted(case_dir.glob("*.json"))
            if not pngs or not jsons:
                continue
            png, js = pngs[0], jsons[0]
            doc = json.loads(js.read_text(encoding="utf-8"))
            with Image.open(png) as im:
                w, h = im.size
            boxes = []
            for key, comp in doc.get("components", {}).items():
                ctype = comp.get("type")
                if ctype not in classes:
                    warn[f"official:unknown_type:{ctype}"] += 1
                    continue
                b = comp.get("bbox")
                if not b or len(b) != 4:
                    warn["official:no_bbox"] += 1
                    continue
                # bbox is [x1, y1, x2, y2] with a BOTTOM-LEFT origin -> flip to image coords
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                box = _clip((x1, h - y2, x2, h - y1), w, h)
                if box is None:
                    warn["official:degenerate"] += 1
                    continue
                boxes.append((ctype, *box))
            if boxes:
                yield f"{split}", case_dir.name, png, boxes, (w, h)


def collect_dataset300(classes: list[str], warn: Counter):
    if not D300_ROOT.is_dir():
        return
    for case_dir in sorted(p for p in D300_ROOT.iterdir() if p.is_dir()):
        pngs = [p for p in case_dir.iterdir() if p.suffix.lower() in (".png", ".jpg")]
        jsons = list(case_dir.glob("*_CVJsonStd.json"))
        if not pngs or not jsons:
            continue
        png, js = pngs[0], jsons[0]
        doc = json.loads(js.read_text(encoding="utf-8"))
        canvas = doc.get("canvas") or {}
        with Image.open(png) as im:
            w, h = im.size
        if canvas.get("width") and (canvas["width"], canvas["height"]) != (w, h):
            warn["d300:canvas_mismatch"] += 1
        boxes = []
        comps = doc.get("components", {})
        for comp in (comps.values() if isinstance(comps, dict) else comps):
            cat = (comp.get("category") or "").strip()
            cls = D300_MAP.get(cat.lower())
            if cls is None or cls not in classes:
                warn[f"d300:unmapped_category:{cat}"] += 1
                continue
            bb = comp.get("realBBox") or comp.get("location") or {}
            try:
                box = _clip((float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])), w, h)
            except (KeyError, TypeError, ValueError):
                warn["d300:bad_bbox"] += 1
                continue
            if box is None:
                warn["d300:degenerate"] += 1
                continue
            boxes.append((cls, *box))
        if boxes:
            yield "Dataset300", case_dir.name, png, boxes, (w, h)


# --------------------------------------------------------------------------- emit

def tiles_for(w, h, tile, stride):
    """Yield (col, row, x0, y0, x1, y1) covering the image, edge-aligned."""
    if w <= tile and h <= tile:
        return
    xs = list(range(0, max(1, w - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, h - tile + 1), stride)) or [0]
    if w > tile and xs[-1] != w - tile:
        xs.append(w - tile)
    if h > tile and ys[-1] != h - tile:
        ys.append(h - tile)
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            yield c, r, x0, y0, min(x0 + tile, w), min(y0 + tile, h)


def write_label(path: Path, boxes, w, h):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{CLASS_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def emit(out: Path, samples, args, stats: Counter):
    for split, case_key, png, boxes, (w, h) in samples:
        tag = split_ok(case_key, args)
        with Image.open(png) as im:
            im = im.convert("RGB")
            stem = f"{case_key}__full"
            if not args.dry_run:
                im.save(out / "images" / tag / f"{stem}.png", optimize=True)
                write_label(out / "labels" / tag / f"{stem}.txt", boxes, w, h)
            stats[f"{tag}:full"] += 1
            stats["boxes"] += len(boxes)
            if args.tile and (w > args.tile or h > args.tile):
                nt = 0
                for c, r, x0, y0, x1, y1 in tiles_for(w, h, args.tile, args.stride):
                    kept = [
                        (cls, bx1 - x0, by1 - y0, bx2 - x0, by2 - y0)
                        for cls, bx1, by1, bx2, by2 in boxes
                        if bx1 >= x0 and by1 >= y0 and bx2 <= x1 and by2 <= y1
                    ]
                    if len(kept) < args.min_boxes:
                        continue
                    nt += 1
                    tstem = f"{case_key}__t{r:02d}_{c:02d}"
                    if not args.dry_run:
                        im.crop((x0, y0, x1, y1)).save(out / "images" / tag / f"{tstem}.png", optimize=True)
                        write_label(out / "labels" / tag / f"{tstem}.txt", kept, x1 - x0, y1 - y0)
                    stats[f"{tag}:tile"] += 1
                    stats["boxes"] += len(kept)
                stats["tiled_cases"] += 1 if nt else 0


def split_ok(case_key: str, args) -> str | None:
    return "val" if case_key in VAL_KEYS else "train"


CASE_NUM = re.compile(r"(\d+)")
# Golden cases carry hand-corrected annotations; Dataset300 is a re-annotation of the same
# sheets in a different schema.  When the same numeric case id shows up twice, the higher
# priority wins -- but only if the two spellings really are the same image (same pixel size).
PRIORITY = {"10_GTcase": 3, "10GT": 3, "200_train_cases": 2, "Dataset300": 1}


def case_num(dir_name: str) -> str:
    m = CASE_NUM.match(dir_name)
    return m.group(1) if m else dir_name


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--tile", type=int, default=1024, help="tile edge in px (0 disables tiling)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-boxes", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.stride = max(1, int(args.tile * (1 - args.overlap))) if args.tile else args.tile

    global CLASS_ID, VAL_KEYS
    classes = load_classes()
    CLASS_ID = {n: i for i, n in enumerate(classes)}

    warn = Counter()
    raw = list(collect_official(classes, warn)) + list(collect_dataset300(classes, warn))

    # Deduplicate by numeric case id: 10_GTcase spells the same sheet "0059-Altium Designer"
    # that 200_train_cases spells "0059", and Dataset300 re-annotates the same sheets again.
    # Golden cases win, but only when the pixel size proves it is actually the same image.
    best: dict[str, tuple] = {}
    for split, case, png, boxes, size in raw:
        num = case_num(case)
        prev = best.get(num)
        if prev is None:
            best[num] = (PRIORITY.get(split, 0), split, case, png, boxes, size)
            continue
        p_pri, p_split, p_case, p_png, p_boxes, p_size = prev
        pri = PRIORITY.get(split, 0)
        if p_size == size:
            warn[f"same_image:{num}:{p_split}+{split}"] += 1
            if pri > p_pri:
                best[num] = (pri, split, case, png, boxes, size)
        else:
            best[f"{num}@{split}"] = (pri, split, case, png, boxes, size)
            warn[f"same_id_diff_image:{num}:{p_split}{p_size}+{split}{size}"] += 1
    samples = [(s, f"{s}-{c}", p, b, z) for _, s, c, p, b, z in best.values()]

    # split by case, not by tile, so a sheet never leaks across train/val
    keys = sorted(k for _, k, *_ in samples)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_frac))
    VAL_KEYS = set(keys[:n_val])

    out = Path(args.out)
    if not args.dry_run:
        for sub in ("train", "val"):
            (out / "images" / sub).mkdir(parents=True, exist_ok=True)
            (out / "labels" / sub).mkdir(parents=True, exist_ok=True)
    stats = Counter()
    emit(out, samples, args, stats)

    print("=== sources ===")
    print(f"cases collected : {len(raw)}  (unique {len(samples)})")
    print(f"by split        : {Counter(s for s, *_ in samples)}")
    print(f"val cases ({len(VAL_KEYS)}): {sorted(VAL_KEYS)[:20]}")
    print("=== images written ===")
    for k in sorted(stats):
        print(f"  {k:16} {stats[k]}")
    if warn:
        print("=== warnings ===")
        for k, v in warn.most_common(25):
            print(f"  {v:6}  {k}")

    if not args.dry_run:
        (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        # The folder is Chinese by request, but a Chinese absolute path inside data.yaml is a
        # needless failure mode for the trainer.  An ASCII junction to the same folder is used
        # when one exists, so `path:` stays pure ASCII.
        ascii_link = Path(r"Y:\workspace\EDA\PCB_trainset")
        yaml_root = ascii_link.as_posix() if ascii_link.is_dir() else out.as_posix()
        (out / "data.yaml").write_text(
            f"path: {yaml_root}\ntrain: images/train\nval: images/val\nnc: {len(classes)}\nnames:\n"
            + "\n".join(f"- {n}" for n in classes) + "\n", encoding="utf-8")
        counts = {s: len(list((out / "images" / s).iterdir())) for s in ("train", "val")}
        print(f"images on disk : {counts}")
    return 0


CLASS_ID: dict[str, int] = {}
VAL_KEYS: set[str] = set()

if __name__ == "__main__":
    raise SystemExit(main())
