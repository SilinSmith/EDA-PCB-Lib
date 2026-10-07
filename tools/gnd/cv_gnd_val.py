"""把「纯 CV 找 GND」和「YOLO 找 GND」放在同一把尺子、同一批图上比。

判据统一为：检测框中心与某个 GT 图章中心的切比雪夫距离 <= 20px，一对一贪心。
（对边长 20 的图章，这个判据和官方 max|角点差| <= 20 完全等价。）

YOLO 一侧直接读 `fn_fp_official.py` 写出的 `preview/official/summary.json`（同一批 77 张 val）。
CV 一侧现场跑 `gnd_grammar.detect`，并测每张耗时。
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from gnd_grammar import detect, dedup, MIN_SIDE, MAX_SIDE  # noqa: E402

ROOT = Path(r"Y:\workspace\EDA\trainset_v2")
TOL = 20.0


def load_gt(lbl: Path, w: int, h: int):
    out = []
    for line in lbl.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
        out.append((c, ((cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h)))
    return out


def cen(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)


def match(pred_centers, gt_centers, tol=TOL):
    cand = []
    for i, p in enumerate(pred_centers):
        for j, g in enumerate(gt_centers):
            d = max(abs(p[0]-g[0]), abs(p[1]-g[1]))
            if d <= tol:
                cand.append((d, i, j))
    cand.sort(key=lambda t: t[0])
    up, ug = set(), set()
    for d, i, j in cand:
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j)
    return len(up), len(pred_centers) - len(up), len(gt_centers) - len(ug)


def main():
    names = (ROOT / "classes.txt").read_text().split()
    gid = names.index("gnd")
    img_dir = ROOT / "images" / "val"
    lab_dir = ROOT / "labels" / "val"

    n = 0
    cv_tp = cv_fp = cv_fn = 0
    cv_time = 0.0
    files = sorted(lab_dir.glob("*.txt"))
    for lp in files:
        ip = img_dir / (lp.stem + ".png")
        if not ip.exists():
            ip = img_dir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        gts = [b for c, b in load_gt(lp, W, H) if c == gid]
        t0 = time.time()
        det = dedup([b for b in detect(g)
                     if MIN_SIDE <= max(b[2]-b[0], b[3]-b[1]) <= MAX_SIDE])
        cv_time += time.time() - t0
        tp, fp, fn = match([cen(b) for b in det], [cen(b) for b in gts])
        cv_tp += tp; cv_fp += fp; cv_fn += fn
        n += 1

    def pr(tp, fp, fn):
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        return p, r, f

    summary = json.loads((Path(r"Y:\workspace\EDA\work\preview\official") / "summary.json")
                         .read_text(encoding="utf-8"))
    y = summary.get("gnd", {})
    y_gt, y_fn, y_fp = y.get("gt", 0), y.get("fn", 0), y.get("fp", 0)
    y_tp = y_gt - y_fn

    print(f"同一批 {n} 张 val 图，判据 = 中心切比雪夫距离 <= {TOL:.0f}px，一对一")
    print(f"\n{'方案':<34}{'P':>8}{'R':>8}{'F1':>8}{'TP':>7}{'FP':>7}{'FN':>7}{'FP/张':>9}")
    for nm, tp, fp, fn in (("YOLO 11m imgsz1024 + 贴图章（官方口径）", y_tp, y_fp, y_fn),
                           ("纯 CV 横线叠文法（四向 0/90/180/270）", cv_tp, cv_fp, cv_fn)):
        p, r, f = pr(tp, fp, fn)
        print(f"{nm:<34}{p:>8.3f}{r:>8.3f}{f:>8.3f}{tp:>7}{fp:>7}{fn:>7}{fp/n:>9.1f}")
    print(f"\n纯 CV 耗时 {cv_time:.2f} s / {n} 张 = {cv_time/max(1,n)*1000:.0f} ms 每张（CPU，单线程）")
    print("YOLO 耗时参考：训练前向+反向 112.8 s / 1624 张 ≈ 69 ms 每张（RTX 4080, batch 12, imgsz 1024）")


if __name__ == "__main__":
    main()
