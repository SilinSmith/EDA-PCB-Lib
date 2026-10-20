# -*- coding: utf-8 -*-
"""页尺度护栏：把「DPI 完全不同的输入」拉回本管线标定过的尺度。

**为什么需要**：管线里所有阈值都是**绝对 px**（`STAGE2_STEAL=80`、`STAGE2_EDGE_SNAP=10`、
`STAGE1_PIN_OUT_BOX=12`、判定器的 20/5 px …），没有任何缩放归一化。实测同一张 GT10 图
等比放大后重跑（`work/_scaletest.py`）：

    k=1.00 → 86.41        k=0.90 → 81.27
    k=1.15 → case1 39.33  k=1.40 → case1 19.18（comp 0.130）

**为什么默认关**：用「墨迹连通块高度中位数」当字号代理量了本仓库全部图，**原生**值就横跨
7–25 px（`work/_ccscale.py`，10GT case1 5…13、池 0003 = 25），说明字号是**风格**差异而不是
DPI 差异。所以只在明显超出该分布（<4 px 或 >30 px）时才动手，阈值留着余量。

用法（未接线，接线点见 `submission/run.py` 的 stage1 之前）：

    from .pagescale import factor, resize, unscale_doc
    k = factor(med_h)                    # 1.0 = 不动
    if k != 1.0: bgr2 = resize(bgr, k)   # 缩到基准尺度后再跑整条管线
    ...                                  # 产物坐标全部 /k 之后才是原图坐标
    doc = unscale_doc(doc, k)
"""
from __future__ import annotations

import cv2
import numpy as np

#: 本仓库数据里实测的字号分布（墨迹连通块高中位数）：7–25 px。
BAND = (4.0, 30.0)
#: 缩放到这个字号（10GT 的众数在 7–10）。
TARGET = 9.0


def estimate(bgr) -> tuple:
    """返回 (中位连通块高, 样本数)。只数「像字」的块：高 3..40、宽 1..40、宽高比 ≤3。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hs = [int(stats[i][3]) for i in range(1, n)
          if 3 <= stats[i][3] <= 40 and 1 <= stats[i][2] <= 40
          and stats[i][4] >= 6 and stats[i][2] <= 3 * stats[i][3]]
    if not hs:
        return 0.0, 0
    return float(np.median(hs)), len(hs)


def factor(med_h: float) -> float:
    """仅在字号明显跑出本仓库实测分布时才给出 ≠1 的缩放系数。"""
    if not med_h or BAND[0] <= med_h <= BAND[1]:
        return 1.0
    return max(0.5, min(2.0, TARGET / med_h))


def resize(bgr, k: float):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (max(1, int(round(w * k))), max(1, int(round(h * k)))),
                      interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)


def unscale_doc(doc: dict, k: float) -> dict:
    """把整份产物（components/pins/nets）的坐标除回原图帧。k=1 时原样返回。"""
    if k == 1.0 or not doc:
        return doc

    def _p(pt):
        if isinstance(pt, dict) and "x" in pt:
            pt["x"] = pt["x"] / k
            pt["y"] = pt["y"] / k

    for c in (doc.get("components") or {}).values():
        if c.get("bbox"):
            c["bbox"] = [v / k for v in c["bbox"]]
    for pins in (doc.get("pins") or {}).values():
        for p in pins.values():
            _p(p.get("point"))
    for net in (doc.get("nets") or {}).values():
        for e in (net.get("edges") or {}).values():
            for q in e:
                _p(q)
    return doc
