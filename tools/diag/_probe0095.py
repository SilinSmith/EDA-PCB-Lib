# -*- coding: utf-8 -*-
"""0095 为什么全军覆没？"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _textseg_eval import D300, cases  # noqa: E402

for cid in ("0095", "0061", "0112", "0146"):
    hit = None
    for png, js in cases(0):
        if png.parent.name == cid:
            hit = (png, js)
    if hit is None:
        print(f"{cid}: 无"); continue
    png, js = hit
    bgr = imread_unicode(png)
    g = bgr.mean(axis=2)
    ink = (g < 170)
    ob = textbox.detect_boxes(bgr)
    nb = textseg.detect_boxes(bgr)
    st = {}
    gl = textseg.glyphs(bgr, stats_out=st)
    print(f"\n{cid}: {bgr.shape[1]}x{bgr.shape[0]}  灰度 mean {g.mean():.1f} "
          f"p05 {np.percentile(g,5):.0f} p50 {np.percentile(g,50):.0f} p95 {np.percentile(g,95):.0f}")
    print(f"  ink 占比 {ink.mean():.3f}   旧 {len(ob)} 框   新 {len(nb)} 框")
    print(f"  glyphs 统计 {st}")
    ncomp = 0
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    print(f"  墨迹 8 连通分量 {n-1}")
