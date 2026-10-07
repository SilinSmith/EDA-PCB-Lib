# -*- coding: utf-8 -*-
"""For each rare class, how many real training images carry it, and how concentrated is it?

If one or two images hold most of a class's instances, duplicating them is memorisation; if the
instances are spread over many images, duplicating is a legitimate re-weighting.
"""
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\训练集_v3")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "train"
CLASSES = (Path(r"Y:\workspace\EDA\PCB训练集") / "classes.txt").read_text(
    encoding="utf-8").splitlines()
CLASSES = [c for c in CLASSES if c.strip()]

WATCH = ["testpoint", "net_input", "net_bidirection", "block", "crystal_2pin", "esd",
         "jumper", "pin", "bjt", "buzzer", "mosfet", "crystal_4pin", "l", "m3螺丝"]

per_img: dict[str, Counter] = {}
for p in sorted((ROOT / "labels" / SPLIT).iterdir()):
    c: Counter = Counter()
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            c[CLASSES[int(ln.split()[0])]] += 1
    if c:
        per_img[p.stem] = c

print(f"{SPLIT}: {len(per_img)} non-empty label files")
print("%-18s %6s %5s %7s   %s" % ("class", "boxes", "imgs", "top5%", "concentration"))
for name in WATCH:
    hits = {k: v[name] for k, v in per_img.items() if v[name]}
    tot = sum(hits.values())
    if not tot:
        print("%-18s %6d %5d %7s   -" % (name, 0, 0, "-"))
        continue
    top = sorted(hits.values(), reverse=True)[:5]
    print("%-18s %6d %5d %6.1f%%   top5=%s  (n>0 imgs, mean %.1f)"
          % (name, tot, len(hits), 100.0 * sum(top) / tot, top, tot / len(hits)))
