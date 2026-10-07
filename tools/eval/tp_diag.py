"""诊断 CV testpoint 检测器的误报：TP 与 FP 的环特征分布差在哪里。

输出：
  1. TP / FP 的 r_min / r_max / ratio / n_rings / circ 分位数对照
  2. FP 接触表（按半径排序取样），看它们到底是什么东西
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "eda-pcb"))
sys.path.insert(0, str(HERE))
import schematic_parser as S  # noqa: E402
from cvparts.testpoint import detect_rings  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(r"Y:\workspace\EDA\work\preview")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 220


def gt_centers(doc, H):
    out = []
    for k, v in (doc.get("components") or {}).items():
        if str(v.get("type", "")).strip() != "testpoint":
            continue
        b = v.get("bbox")
        if not b:
            continue
        x1, y1, x2, y2 = [float(t) for t in b]
        out.append(((x1 + x2) / 2.0, H - (y1 + y2) / 2.0))
    return out


def main():
    tp_feat, fp_feat = [], []
    fp_crops = []
    ncase = 0
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        if ncase >= LIMIT:
            break
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        gts = gt_centers(doc, 0)
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        ncase += 1
        H = bgr.shape[0]
        gts = gt_centers(doc, H)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for g in detect_rings(gray):
            cx, cy = g["center"]
            f = (g["r_min"], g["r_max"], g["n_rings"], g["circ"])
            hit = any(max(abs(cx - gx), abs(cy - gy)) <= 20.0 for gx, gy in gts)
            (tp_feat if hit else fp_feat).append(f)
            if not hit and len(fp_crops) < 4000:
                hw = max(45, int(g["r_max"] * 3))
                x1c, y1c = max(0, int(cx - hw)), max(0, int(cy - hw))
                x2c, y2c = min(bgr.shape[1], int(cx + hw)), min(H, int(cy + hw))
                sub = bgr[y1c:y2c, x1c:x2c].copy()
                if sub.size:
                    fp_crops.append((sub, g["r_min"], g["r_max"], g["n_rings"], c.case_id))
    print(f"扫了 {ncase} 张")

    def show(name, arr, idx, label):
        v = np.array([a[idx] for a in arr], dtype=float)
        if v.size == 0:
            print(f"  {name} 空")
            return
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"  {name:<9}{label:<8} n {v.size:>6}  "
              f"p5 {q[0]:7.2f} p25 {q[1]:7.2f} p50 {q[2]:7.2f} "
              f"p75 {q[3]:7.2f} p95 {q[4]:7.2f}  mean {v.mean():7.2f}")

    print(f"\nTP {len(tp_feat)}   FP {len(fp_feat)}")
    for i, lab in ((0, "r_min"), (1, "r_max"), (2, "n_rings"), (3, "circ")):
        show("TP", tp_feat, i, lab)
        show("FP", fp_feat, i, lab)
        print()

    # 环数分布
    print("n_rings 直方图  TP:", dict(Counter(a[2] for a in tp_feat)))
    print("n_rings 直方图  FP:", dict(Counter(a[2] for a in fp_feat)).most_common(10))

    # FP 接触表：按 r_min 分几桶各取几个
    fp_crops.sort(key=lambda t: t[1])
    picks = []
    if fp_crops:
        step = max(1, len(fp_crops) // 40)
        picks = fp_crops[::step][:40]
    CELL, PER = 150, 8
    rows = (len(picks) + PER - 1) // PER
    sheet = Image.new("RGB", (CELL * PER, CELL * rows), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    for n, (sub, rmin, rmax, nr, cid) in enumerate(picks):
        im = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        sc = min(CELL / im.width, (CELL - 16) / im.height)
        im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
        px, py = (n % PER) * CELL, (n // PER) * CELL
        sheet.paste(im, (px, py))
        dr.text((px + 3, py + CELL - 15), f"r{rmin:.0f}/{rmax:.0f} n{nr} c{cid[-4:]}",
                fill=(230, 230, 230))
    out = OUT / "tp_fp.png"
    sheet.save(out)
    print(f"\nFP 接触表 -> {out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
