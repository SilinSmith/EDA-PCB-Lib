# -*- coding: utf-8 -*-
"""连通分量到底有多大？在「墨迹掩膜」上量，还是「骨架」上量？

用户的大尺度连通先验说的是**画出来的墨迹**：一条走线的墨迹是一个大连通体。
但骨架化（zhang_suen）会在细线处断连，我的遮挡又会把导线在元件处切断。
所以要先看清：jlc / kicad 的骨架分量是不是被切碎了。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import schematic_parser as S
from pipeline.pskel import WireExtractor

CASES = ["0101", "0107", "0059", "0050", "0001", "0011", "0002", "0102"]


def hist(mask, tag):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    diag = np.hypot(st[1:, 2].astype(float), st[1:, 3].astype(float))
    area = st[1:, 4].astype(float)
    big = int((diag >= 50).sum())
    print(f"    {tag:<10} 分量 {n-1:>6}   >=50px 的 {big:>5}   "
          f"最大 extent {diag.max() if len(diag) else 0:>7.0f}   "
          f"extent 分位 p50 {np.percentile(diag,50) if len(diag) else 0:>5.0f} "
          f"p90 {np.percentile(diag,90) if len(diag) else 0:>6.0f} "
          f"p99 {np.percentile(diag,99) if len(diag) else 0:>7.0f}")
    return diag, area


for cid in CASES:
    c = S.find_case(cid, S.DEFAULT_DATA_ROOT)
    if c is None:
        print(f"{cid}: 找不到")
        continue
    bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    ex = WireExtractor(ink_mode="adaptive", thin_w=1.6, use_colour=True,
                       rdp_eps=1.6, snap=True)
    thin, ink, chroma = ex.foreground(bgr)
    print(f"{cid}  {W}x{H}  chroma={chroma}")
    hist(ink, "墨迹掩膜")
    hist(thin, "骨架")
