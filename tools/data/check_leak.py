"""Two leak questions behind "why is mAP50 this high after 1 epoch?"

Q1  what is actually inside yolo_part/weights/*.pt?
    A COCO checkpoint has 80 English class names; a team-trained one carries the
    41 component classes.  `Remapped 41/43 cls head rows from pretrained weights by
    class name` can only happen if the checkpoint already knows those 41 names.

Q2  does any validation image duplicate a training image?
    The split is by source sheet, so tiles cannot leak -- but the same schematic can
    appear twice under different case ids (and a different pixel size, which defeats
    the exact-size dedup used when the set was built).  Average-hash finds those.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
WEIGHTS = ROOT / "eda-pcb" / "yolo_part" / "weights"
DS = ROOT / "PCB训练集"


def q1() -> None:
    import torch

    for p in sorted(WEIGHTS.glob("*.pt")):
        print(f"\n--- {p.name}  ({p.stat().st_size / 1e6:.1f} MB) ---")
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"  load failed: {type(e).__name__}: {e}")
            continue
        if not isinstance(ck, dict):
            print(f"  not a checkpoint dict: {type(ck).__name__}")
            continue
        m = ck.get("model")
        nc = getattr(m, "nc", None)
        names = getattr(m, "names", None)
        print(f"  nc={nc}  epoch={ck.get('epoch')}  version={ck.get('version')}  date={ck.get('date')}")
        if isinstance(names, dict):
            vals = list(names.values())
            print(f"  names({len(vals)}): {vals[:8]}{' ...' if len(vals) > 8 else ''}")
        elif names is not None:
            print(f"  names: {names}")
        ta = ck.get("train_args") or {}
        if ta:
            keep = {k: ta.get(k) for k in ("data", "epochs", "imgsz", "model", "batch", "fraction") if k in ta}
            print(f"  train_args: {keep}")


def q2() -> None:
    def ahash(p: Path, s: int = 16) -> int:
        with Image.open(p) as im:
            px = list(im.convert("L").resize((s, s), Image.LANCZOS).getdata())
        avg = sum(px) / len(px)
        return sum(1 << i for i, v in enumerate(px) if v > avg)

    def bits(h: int) -> int:
        return bin(h).count("1")

    hashes: dict[str, list[tuple[str, str, int]]] = {}
    for split in ("train", "val"):
        for p in sorted((DS / "images" / split).iterdir()):
            hashes.setdefault(split, []).append((p.stem, p.name, ahash(p)))

    train, val = hashes["train"], hashes["val"]
    print(f"\n--- Q2: average-hash  train={len(train)} val={len(val)} ---")
    hits = Counter()
    worst = []
    for vstem, vname, vh in val:
        best = min(((bits(vh ^ th), tstem) for tstem, _, th in train), default=(999, None))
        hits[best[0]] += 1
        if best[0] <= 10:
            worst.append((best[0], vstem, best[1]))
    print("  distance from each val image to its nearest train image:")
    for d in sorted(hits):
        print(f"    d={d:3}  {hits[d]:4} val images")
    if worst:
        print(f"  NEAR-DUPLICATE pairs (d<=10): {len(worst)}")
        for d, vstem, tstem in sorted(worst)[:25]:
            print(f"    d={d:3}  {vstem:46} <-> {tstem}")
    else:
        print("  no val image is a near-duplicate of any train image (all d>10)")


if __name__ == "__main__":
    q1()
    q2()
