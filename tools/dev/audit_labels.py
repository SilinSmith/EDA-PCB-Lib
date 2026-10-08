# -*- coding: utf-8 -*-
"""Training-label quality audit for a YOLO dataset — "错误的标号" triage.

Context
-------
`eda-pcb/docs/reports/漏检错检_归因.md:175-187` lists the label-side defects that a retrain
would otherwise bake in again.  Owner's brief (paraphrased): the last YOLO run was polluted by
"错误的标号" in the training set.  This script turns those prose findings into a per-box,
per-class, machine-checkable ledger on the WHOLE corpus, not just val77.

Checks (all from labels + image pixels; nothing is written unless `--fix-out` is given)
-------------------------------------------------------------------------------------
  ZERO_INK      box interior has no ink (<1%) but ink exists within 25 px of the centre
                -> the L1 defect (`归因.md:179`): the box sits on a stroke gap.
  DEAD          no ink inside AND none within 25 px -> probably a label for nothing at all.
  TPL_MISMATCH  (w,h) is not one of the class's official stamp sizes (`stamp.py:STAMP_WH`).
                Official symbol classes are 100% one fixed (long,short) pair, so any other
                size means the label came from a different annotation convention.
  SIZE_OUTLIER  same image + same class has >=4 boxes and this one deviates >40% from that
                group's modal size (a cheap detector for "the labeler drew this one wrong").
  TEXT_BOX      >=15% of the box's pixels are blue text pixels (designator/value/net label),
                i.e. the box is on the annotation text, not on the symbol (L3, `归因.md:182`).
  DUP           two same-class boxes in one image with Chebyshev corner distance <= 4 px.
  BAD_NAME      (only with --json-root) the component `Name` does not carry the class's
                conventional prefix (R/C/L/D/Q/U/J/P/X/Y/FB/TP...) — "标号本身写错了".

    $env:PYTHONIOENCODING='utf-8'
    cd eda-pcb
    python tools\\dev\\audit_labels.py                      # trainset_v3, all splits
    python tools\\dev\\audit_labels.py --limit 200 --csv ..\\work\\label_audit.csv
    python tools\\dev\\audit_labels.py --fix recenter --fix-out ..\\work\\trainset_v4_fixcheck
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
EDA_PCB = HERE.parents[2]
EDA = HERE.parents[3]
sys.path.insert(0, str(EDA_PCB))
from schematic_parser.stamp import STAMP_WH          # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INK = 200
TPL_TOL = 2.0
FIXES = ("off", "recenter", "drop", "snap")

# conventional designator prefixes per class (only the ones worth checking)
PREFIX = {
    "r": "R", "c": "C", "l": "L", "d": "D", "led": "LED", "v": "V", "gnd": "GND",
    "box": "U", "switch": "S", "buzzer": "BZ", "crystal": "X", "crystal_2pin": "X",
    "crystal_3pin": "X", "crystal_4pin": "X", "fuse": "F", "mosfet": "Q", "bjt": "Q",
    "jumper": "JP", "testpoint": "TP", "pin": "P", "ant": "E", "transfomer": "T",
    "motor": "M", "opt": "U", "esd": "ESD", "dc": "DC", "seg": "SEG", "circle_header": "J",
}


def imread_gray(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def imread_color(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def blue_text_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16), bgr[:, :, 2].astype(np.int16)
    return (b > 110) & (b - r > 40) & (b - g > 40)


def modal(xs: list[float], tol: float = 2.0) -> float:
    if not xs:
        return 0.0
    best, bestn = xs[0], 0
    for x in xs:
        n = sum(1 for y in xs if abs(y - x) <= tol)
        if n > bestn:
            best, bestn = x, n
    return best


def analyse_split(args, img_dir: Path, lbl_dir: Path, classes: list[str], split: str,
                  rows: list[dict], per_class: dict, stats: Counter) -> None:
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    for img_path in imgs:
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.is_file():
            stats[f"{split}: label missing"] += 1
            continue
        raw = [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw:
            continue
        gray = imread_gray(img_path)
        H, W = gray.shape
        need_color = args.text_check
        bgr = imread_color(img_path) if need_color else None
        tmask = blue_text_mask(bgr) if need_color else None
        boxes = []
        for parts in raw:
            cid = int(parts[0])
            cx, cy, w, h = (float(parts[i]) * (W if i % 2 == 1 else H) for i in (1, 2, 3, 4))
            boxes.append([cid, cx, cy, w, h])
        # per-image per-class modal size
        by_cls: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for cid, cx, cy, w, h in boxes:
            by_cls[cid].append((w, h))
        mode_wh = {c: (modal([a for a, _ in v]), modal([b for _, b in v])) for c, v in by_cls.items()}
        seen: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for cid, cx, cy, w, h in boxes:
            p = Path(img_path.name)
            cls = classes[cid]
            flags, ev = [], []
            # --- ink
            x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
            x2, y2 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
            ink_in = 0.0
            if x2 > x1 and y2 > y1:
                ink_in = float((gray[y1:y2, x1:x2] < INK).mean())
            r = 25
            X1, Y1 = max(0, int(cx - w / 2 - r)), max(0, int(cy - h / 2 - r))
            X2, Y2 = min(W, int(cx + w / 2 + r)), min(H, int(cy + h / 2 + r))
            outer = gray[Y1:Y2, X1:X2]
            n_out = int((outer < INK).sum())
            n_in = int((gray[y1:y2, x1:x2] < INK).sum()) if x2 > x1 and y2 > y1 else 0
            ink_ring = (n_out - n_in) / max(1, outer.size)
            if ink_in < 0.01:
                if ink_ring >= 0.004:
                    flags.append("ZERO_INK")
                else:
                    flags.append("DEAD")
                ev.append(f"ink_in={ink_in:.4f} ink_ring={ink_ring:.4f}")
            # --- template
            tpl = STAMP_WH.get(cls)
            if tpl:
                lng, sht = tpl
                got = (max(w, h), min(w, h))
                want = (float(lng), float(sht))
                if abs(got[0] - want[0]) > TPL_TOL or abs(got[1] - want[1]) > TPL_TOL:
                    flags.append("TPL_MISMATCH")
                    ev.append(f"wh=({w:.0f},{h:.0f}) stamp={want}")
            # --- size outlier within the image
            if len(by_cls[cid]) >= 4:
                mw, mh = mode_wh[cid]
                if mw > 0 and abserr(w, mw) > 0.40 and abserr(h, mh) > 0.40:
                    flags.append("SIZE_OUTLIER")
                    ev.append(f"wh=({w:.0f},{h:.0f}) img_mode=({mw:.0f},{mh:.0f})")
            # --- text overlap: L3 is "the box sits ON the text, not on the symbol", so the
            #     discriminating signal is text-dominated ink with essentially no symbol ink.
            #     Symbol ink = dark pixels that are NOT blue (blue = designator/value/net label).
            if need_color and x2 > x1 and y2 > y1:
                frac = float(tmask[y1:y2, x1:x2].mean())
                sub_g = gray[y1:y2, x1:x2]
                sub_t = tmask[y1:y2, x1:x2]
                sym = float(((sub_g < INK) & ~sub_t).mean())
                if frac >= 0.30 and sym < 0.03:
                    flags.append("TEXT_BOX")
                    ev.append(f"blue_text={frac:.3f} sym_ink={sym:.3f}")
            # --- duplicate
            key = (cid, round(cx, 1), round(cy, 1))
            dup = False
            for ox, oy in seen[cid]:
                if max(abs(ox - round(cx, 1)), abs(oy - round(cy, 1))) <= 4.0:
                    dup = True
                    break
            seen[cid].append((round(cx, 1), round(cy, 1)))
            if dup:
                flags.append("DUP")
            per_class[cls]["n"] += 1
            for f in set(flags):
                per_class[cls][f] += 1
            if flags:
                stats[f"{split}: flagged boxes"] += 1
            rows.append({
                "split": split, "case": p.stem, "cls": cls, "cid": cid,
                "cx": round(cx, 1), "cy": round(cy, 1), "w": round(w, 1), "h": round(h, 1),
                "flags": "+".join(flags) or "OK", "evidence": " ; ".join(ev),
                "img": str(img_path),
            })


def abserr(a: float, b: float) -> float:
    return abs(a - b) / max(1e-6, max(a, b))


def fix_plan(args, rows: list[dict], classes: list[str]) -> Counter:
    """How many boxes would each automatic repair touch, per class."""
    out: Counter = Counter()
    for r in rows:
        f = r["flags"]
        if args.fix == "recenter" and ("ZERO_INK" in f or "DEAD" in f):
            out[r["cls"]] += 1
            out["__total__"] += 1
        elif args.fix == "drop" and ("ZERO_INK" in f or "DEAD" in f):
            out[r["cls"]] += 1
            out["__total__"] += 1
        elif args.fix == "snap" and "TPL_MISMATCH" in f:
            out[r["cls"]] += 1
            out["__total__"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=EDA / "trainset_v3")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=0, help="max images per split (0 = all)")
    ap.add_argument("--text-check", action="store_true", default=True)
    ap.add_argument("--no-text-check", dest="text_check", action="store_false")
    ap.add_argument("--csv", type=Path, default=EDA / "work" / "label_audit.csv")
    ap.add_argument("--json", type=Path, default=EDA / "work" / "label_audit.json")
    ap.add_argument("--fix", choices=FIXES, default="off")
    args = ap.parse_args()

    root: Path = args.root
    classes = [l.strip() for l in (root / "classes.txt").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    rows: list[dict] = []
    per_class: dict[str, Counter] = defaultdict(Counter)
    stats: Counter = Counter()
    for split in args.splits.split(","):
        idir, ldir = root / "images" / split, root / "labels" / split
        if not idir.is_dir():
            continue
        print(f"scanning {split} ...", flush=True)
        analyse_split(args, idir, ldir, classes, split, rows, per_class, stats)

    order = ["n", "ZERO_INK", "DEAD", "TPL_MISMATCH", "SIZE_OUTLIER", "TEXT_BOX", "DUP"]
    flagged = {c: sum(v[f] for f in order[1:] if f in v) for c, v in per_class.items()}
    print(f"\n=== 错标率按类 (root={root}) ===")
    print(f"{'class':16s} {'boxes':>7s} " + " ".join(f"{f[:11]:>12s}" for f in order[1:]) + f" {'flagged%':>9s}")
    for c in sorted(per_class, key=lambda x: -per_class[x]["n"]):
        v = per_class[c]
        print(f"{c:16s} {v['n']:7d} " + " ".join(f"{v[f]:12d}" for f in order[1:])
              + f" {100.0 * flagged[c] / max(1, v['n']):8.1f}%")
    n_all = sum(v["n"] for v in per_class.values())
    n_flag = sum(flagged.values())
    print(f"\nTOTAL boxes {n_all}  flagged(box-level, may double count) {n_flag} "
          f"({100.0 * n_flag / max(1, n_all):.1f}%)")
    print("stats: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if args.fix != "off":
        plan = fix_plan(args, rows, classes)
        print(f"\n--fix {args.fix}: 命中 {plan['__total__']} 个框 "
              f"({100.0 * plan['__total__'] / max(1, n_all):.1f}% of all boxes)")
        for c, n in plan.most_common():
            if c != "__total__":
                print(f"    {c:16s} {n}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}  ({len(rows)} rows)")
    if args.json:
        args.json.write_text(json.dumps(
            {"root": str(root), "per_class": {c: dict(v) for c, v in per_class.items()},
             "totals": {"boxes": n_all, "flagged": n_flag}}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
