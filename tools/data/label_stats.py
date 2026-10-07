# -*- coding: utf-8 -*-
"""统计 YOLO 数据集中每个 split 的每类实例数。

用法: python label_stats.py <dataset_root> [--splits train,val] [--out x.json]
dataset_root 需含 classes.txt 与 labels/<split>/*.txt
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    classes = (root / "classes.txt").read_text(encoding="utf-8").splitlines()
    out = {}
    for split in args.splits.split(","):
        d = root / "labels" / split
        if not d.is_dir():
            continue
        c = Counter()
        files = 0
        for f in d.glob("*.txt"):
            files += 1
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    c[int(line.split()[0])] += 1
        out[split] = {"files": files, "boxes": sum(c.values()),
                      "per_class": {classes[i]: c.get(i, 0) for i in range(len(classes))}}
        print("=== %s: %d files, %d boxes ===" % (split, files, sum(c.values())))
        for i in sorted(range(len(classes)), key=lambda i: -c.get(i, 0)):
            print("  %-20s %6d %5.2f%%" % (classes[i], c.get(i, 0),
                                           100.0 * c.get(i, 0) / max(1, sum(c.values()))))
        print()
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
