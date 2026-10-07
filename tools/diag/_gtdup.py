"""GT 标注里有没有「同一颗器件被标了两遍」—— 重复检测是不是被**训练出来的**。

1) 逐 label 文件找「同类、中心距/角点距 <= 容差」的框对；角点距按归一化坐标换算成像素。
2) 数官方数据集目录下每个 case 有几份标注 json（多份来源合并 = 重复框的天然来源）。

用法：python work\\tools\\diag\\_gtdup.py [--tol=20]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(r"Y:\workspace\EDA\trainset_v3")
DATA = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
TOL = 20.0
for a in sys.argv[1:]:
    if a.startswith("--tol="):
        TOL = float(a.split("=", 1)[1])

names = []
for ln in (ROOT / "data.yaml").read_text(encoding="utf-8").splitlines():
    if ln.startswith("names:"):
        names = [s.strip().strip("'\"") for s in ln.split(":", 1)[1].strip("[]").split(",")]
print(f"类数 {len(names)}｜容差 {TOL}px")


def scan(split: str) -> None:
    lab_dir, img_dir = ROOT / "labels" / split, ROOT / "images" / split
    n_img = n_dup = n_box = n_pair = 0
    by_cls: Counter = Counter()
    worst = []
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
        n_img += 1
        n_box += len(boxes)
        local = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] != b[0]:
                    continue
                d = max(abs(a[1 + k] - b[1 + k]) for k in range(4))
                if d <= TOL:
                    local += 1
                    by_cls[names[a[0]] if a[0] < len(names) else a[0]] += 1
        if local:
            n_dup += 1
            n_pair += local
            worst.append((local, f.stem, len(boxes)))
    print(f"\n[{split}] {n_img} 图 / {n_box} 框；含重复框对的图 {n_dup} 张，重复框对 {n_pair} 对")
    print("  按类：" + (", ".join(f"{k}={v}" for k, v in by_cls.most_common(12)) or "无"))
    for local, stem, nb in sorted(worst, reverse=True)[:8]:
        print(f"    {local:>3} 对  {stem}  (共 {nb} 框)")


for sp in ("val", "train"):
    scan(sp)

print("\n官方数据集目录里每个 case 有几份标注 json：")
cases = sorted(DATA.glob("200_train_cases/*"))
multi = [(c, [j.name for j in c.glob("*.json")]) for c in cases if c.is_dir()]
cnt = Counter(len(v) for _, v in multi)
print(f"  {len(multi)} 个 case；含 0/1/2/3+ 份 json 的 case 数 = "
      f"{cnt.get(0,0)}/{cnt.get(1,0)}/{cnt.get(2,0)}/{sum(v for k,v in cnt.items() if k>=3)}")
for c, js in [(c, js) for c, js in multi if len(js) >= 2][:6]:
    print(f"    {c.name}: {js}")
