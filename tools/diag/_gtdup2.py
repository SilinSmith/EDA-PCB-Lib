"""重复框对到底是「同一个框写了两遍」还是「两个挨得很近的符号」。

对每对（同类、角点距 <=20px）量 corner_dev，分桶统计。
用法：python work\\tools\\diag\\_gtdup2.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(r"Y:\workspace\EDA\trainset_v3")
NAMES = ['amp', 'amp_3pin', 'amp_5pin', 'ant', 'battary', 'bjt', 'bjt_pnp', 'block', 'box',
         'buzzer', 'c', 'circle_header', 'crystal', 'crystal_2pin', 'crystal_3pin',
         'crystal_4pin', 'd', 'dc', 'esd', 'fuse', 'gnd', 'jumper', 'l', 'led', 'm3螺丝',
         'mosfet', 'mosfet_pnp', 'motor', 'net_bidirection', 'net_input', 'net_output',
         'net_short', 'opt', 'other', 'pin', 'r', 'seg', 'switch', 'testpoint', 'transfomer',
         'v', 'bjt_npn', 'mosfet_npn']
BUCKETS = [("=0", 0.0, 0.0), ("0~2", 0.0, 2.0), ("2~5", 2.0, 5.0),
           ("5~10", 5.0, 10.0), ("10~20", 10.0, 20.001)]

for split in ("val", "train"):
    lab_dir, img_dir = ROOT / "labels" / split, ROOT / "images" / split
    hist = Counter()
    per_cls = defaultdict(list)
    exact = 0
    npair = 0
    for f in sorted(lab_dir.glob("*.txt")):
        img = img_dir / f"{f.stem}.png"
        if not img.exists():
            continue
        with Image.open(img) as im:
            W, H = im.size
        boxes = []
        for ln in f.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *(float(v) for v in p[1:5])
            boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] != b[0]:
                    continue
                ds = [abs(a[1 + k] - b[1 + k]) for k in range(4)]
                d = max(ds)
                if d > 20.0:
                    continue
                npair += 1
                exact += (d == 0.0)
                hist[next(lab for lab, lo, hi in BUCKETS if lo <= d < hi)] += 1
                per_cls[NAMES[a[0]]].append(d)
    print(f"\n[{split}] 重复框对 {npair}")
    print("  corner_dev 分桶: " + "  ".join(
        f"{lab}={hist[lab]}" for lab, _lo, _hi in BUCKETS))
    print(f"  完全重合(四个角点逐位相同) 的对数 = {exact}  ({exact/max(1,npair):.1%})")
    top = sorted(per_cls.items(), key=lambda kv: -len(kv[1]))[:8]
    for k, ds in top:
        ds.sort()
        print(f"    {k:<14} {len(ds):>4} 对  中位差 {ds[len(ds)//2]:6.1f}px  "
              f"最大 {ds[-1]:5.1f}px  完全重合 {sum(1 for x in ds if x == 0)} 对")
