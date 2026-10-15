# -*- coding: utf-8 -*-
"""v3 vs v4：框是不是真的套在墨迹上（recenter 修复的定量体检）。

背景：v4 用 `--fix-ink recenter` 重建过，队列接下来要在 v4 上花 4.4 h 重训。
「中位尺寸没变」不能证明 recenter 没把框推歪，所以直接量：
  对每个框，取框内墨迹（灰度 < INK）的质心，算它相对框中心的偏移
  （按框短边归一化）。偏移越小 ⇒ 框越居中。
同时报框内墨迹占比（coverage）：接近 0 说明框套空了，接近 1 说明框里全是线。

用法：python -X utf8 work\_inkcheck.py [--sample 300] [--classes gnd,r,c,v]
"""
from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys

import numpy as np
import yaml
from PIL import Image

INK = 200  # 灰度低于它的算墨迹
V3 = pathlib.Path(r"Y:\workspace\EDA\trainset_v3")
V4 = pathlib.Path(r"Y:\workspace\EDA\work\trainset_v4")


def measure(root: pathlib.Path, sample: int, want: set[int], seed: int = 0) -> dict:
    names = list(yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"])
    idir, ldir = root / "images" / "train", root / "labels" / "train"
    imgs = [p for p in sorted(idir.iterdir()) if p.suffix.lower() == ".png" and (ldir / (p.stem + ".txt")).exists()]
    random.Random(seed).shuffle(imgs)
    offsets: list[float] = []
    cov: list[float] = []
    per_cls: dict[str, list[float]] = {}
    for img in imgs[:sample]:
        arr = np.asarray(Image.open(img).convert("L"), dtype=np.uint8)
        H, W = arr.shape
        for line in (ldir / (img.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c not in want:
                continue
            cx, cy, w, h = (float(v) for v in p[1:5])
            x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            sub = arr[y0:y1, x0:x1]
            mask = sub < INK
            n = int(mask.sum())
            cov.append(n / mask.size)
            if n == 0:
                offsets.append(1.0)  # 空框：按最坏记
                continue
            ys, xs = np.nonzero(mask)
            mx, my = xs.mean(), ys.mean()  # 相对框左上角的质心
            bcx, bcy = (x1 - x0 - 1) / 2.0, (y1 - y0 - 1) / 2.0
            off = float(np.hypot(mx - bcx, my - bcy)) / max(1.0, min(x1 - x0, y1 - y0))
            offsets.append(off)
            per_cls.setdefault(names[c], []).append(off)
    def q(v, f):
        return statistics.quantiles(v, n=100)[int(f * 100) - 1] if len(v) > 5 else (max(v) if v else 0.0)
    return {
        "n": len(offsets),
        "med": statistics.median(offsets) if offsets else 0.0,
        "p90": q(offsets, 0.9),
        "empty": sum(1 for o in offsets if o >= 1.0),
        "cov_med": statistics.median(cov) if cov else 0.0,
        "per_cls": {k: (len(v), statistics.median(v)) for k, v in sorted(per_cls.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--classes", default="gnd,r,c,v,box,d,led")
    ap.add_argument("--roots", default=f"v3={V3},v4={V4}",
                    help="tag=path,tag=path（默认 v3/v4）；逐类表比较第一个与最后一个")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    roots = []
    for part in a.roots.split(","):
        tag, _, path = part.partition("=")
        roots.append((tag.strip(), pathlib.Path(path.strip())))
    names = list(yaml.safe_load((roots[-1][1] / "data.yaml").read_text(encoding="utf-8"))["names"])
    want = {names.index(n) for n in a.classes.split(",") if n in names}
    print(f"抽样 {a.sample} 张 train 图；只看类 {a.classes}\n")
    res = {}
    for tag, root in roots:
        r = measure(root, a.sample, want, seed=a.seed)
        res[tag] = r
        print(f"{tag}: n={r['n']}  居中偏移 中位 {r['med']:.3f} / p90 {r['p90']:.3f}（按框短边归一化）"
              f"  空框 {r['empty']}  框内墨迹占比中位 {r['cov_med']:.3f}")
    t0, t1 = roots[0][0], roots[-1][0]
    print(f"\n{'类':<12}{t0 + ' n':>8}{t0 + ' 偏移':>10}{t1 + ' n':>8}{t1 + ' 偏移':>10}   Δ偏移")
    for name in a.classes.split(","):
        n3, m3 = res[t0]["per_cls"].get(name, (0, 0.0))
        n4, m4 = res[t1]["per_cls"].get(name, (0, 0.0))
        flag = f"  <-- {t1} 更歪" if m4 > m3 + 0.02 else ""
        print(f"{name:<12}{n3:>8}{m3:>10.3f}{n4:>8}{m4:>10.3f}  {m4 - m3:+.3f}{flag}")
    print("\n读法：偏移是「墨迹质心 vs 框中心」的距离 ÷ 框短边。0.05 以内都算套得正；"
          "空框（框里没有墨迹）是硬伤。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
