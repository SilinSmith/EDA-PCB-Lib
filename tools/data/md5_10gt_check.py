"""Is the 10GT hold-out really a hold-out?

`build_pcb_trainset.py` deduplicates the four source folders by *numeric case id* and keeps the
highest-priority spelling.  That means the sheet drawn in `10GT/3/3.png` may be the very same
file as `200_train_cases/0003/...` -- in which case the "hold-out" has already been trained on
(under whichever spelling won the dedup) and any test score computed on it is memorisation, not
generalisation.

Two-stage check, cheapest first:
  1. pixel size  -- a different canvas size proves the two files are different sheets outright
  2. md5         -- only run on size collisions, since md5 of a 20 MB PNG is not free

    $env:PYTHONIOENCODING='utf-8'
    K:\\python\\311\\python.exe work\\md5_10gt_check.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
GT10 = DATA / "10GT"
OUT = Path(r"Y:\workspace\EDA\work\10gt_leak_check.json")


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def index(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    targets = []
    for d in sorted(GT10.iterdir(), key=lambda p: (len(p.name), p.name)):
        if not d.is_dir():
            continue
        imgs = index(d)
        if not imgs:
            print(f"[warn] {d} has no image")
            continue
        targets.append((d.name, imgs[0]))
    print(f"10GT images: {len(targets)}")

    others = []
    for sub in ("200_train_cases", "10_GTcase"):
        for p in index(DATA / sub):
            others.append((sub, p))
    print(f"comparison pool: {len(others)} images in 200_train_cases + 10_GTcase")

    # pass 1: sizes
    print("\n=== pass 1: pixel size ===")
    t_meta = []
    for name, p in targets:
        with Image.open(p) as im:
            w, h = im.size
        t_meta.append({"case": name, "path": str(p), "size": [w, h], "bytes": p.stat().st_size})
        print(f"  10GT/{name:>3}  {w:>4}x{h:<4}  {p.stat().st_size / 1024:8.1f} KB  {p.name}")

    o_meta = []
    for sub, p in others:
        with Image.open(p) as im:
            w, h = im.size
        o_meta.append({"sub": sub, "path": p, "size": (w, h)})
    sizes = {tuple(m["size"]) for m in t_meta}
    cands = [m for m in o_meta if m["size"] in sizes]
    print(f"  size collisions: {len(cands)} candidate files "
          f"(of which {len({m['sub'] for m in cands})} source folders)")

    # pass 2: md5
    print("\n=== pass 2: md5 on size collisions ===")
    t_hash = {}
    for m in t_meta:
        t_hash.setdefault(tuple(m["size"]), []).append(m)
    for m in t_meta:
        m["md5"] = md5(Path(m["path"]))

    hits = []
    for c in cands:
        c["md5"] = md5(c["path"])
        for m in t_meta:
            if tuple(m["size"]) == c["size"] and m["md5"] == c["md5"]:
                hits.append({"gt10_case": m["case"], "gt10_path": m["path"],
                             "gt10_md5": m["md5"], "other_sub": c["sub"],
                             "other_path": str(c["path"]), "size": list(m["size"])})

    by_gt: dict[str, list] = {}
    for h in hits:
        by_gt.setdefault(h["gt10_case"], []).append(h)
    contaminated = sorted(by_gt, key=lambda s: (len(s), s))
    print(f"  identical files found: {len(hits)}")
    for case in sorted(contaminated, key=lambda s: int(s)):
        for h in by_gt[case]:
            print(f"    10GT/{case}  ==  {h['other_sub']}/{Path(h['other_path']).parent.name}"
                  f"/{Path(h['other_path']).name}   (md5 {h['gt10_md5']})")
    clean = [m["case"] for m in t_meta if m["case"] not in by_gt]
    print(f"  contaminated 10GT cases: {len(contaminated)} -> {contaminated}")
    print(f"  never-seen  10GT cases: {len(clean)} -> {clean}")

    # is the contaminated sheet present in the *training* split specifically?
    tr = Path(r"Y:\workspace\EDA\PCB训练集\images\train")
    va = Path(r"Y:\workspace\EDA\PCB训练集\images\val")
    split_of = {}
    for case in sorted(by_gt, key=lambda s: int(s)):
        split_of[case] = {
            "in_train": any(p.name.startswith(f"10GT-{case}__") for p in tr.iterdir()),
            "in_val": any(p.name.startswith(f"10GT-{case}__") for p in va.iterdir()),
        }
        print(f"    10GT-{case} present in PCB训练集: "
              f"train={split_of[case]['in_train']} val={split_of[case]['in_val']}")

    report = {
        "10gt": t_meta, "pool": len(others), "size_candidates": len(cands),
        "identical_hits": hits, "contaminated_cases": contaminated, "clean_cases": clean,
        "present_in": split_of,
        "verdict": ("10GT is NOT a clean hold-out: every one of its sheets is byte-identical to a "
                    "file in 200_train_cases or 10_GTcase, so all 10 were already trained on."
                    if len(contaminated) == len(targets) else
                    f"{len(contaminated)}/{len(targets)} 10GT sheets are duplicates of "
                    "already-trained files."),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nverdict: {report['verdict']}")
    print(f"report : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
