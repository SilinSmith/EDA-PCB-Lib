# -*- coding: utf-8 -*-
"""stage1/xnc.py —— 「未连接脚」的 X 标记当引脚点（`STAGE1_PIN_XMARK`，默认 0 = 关）。

动机（用户 m03625 指出 case8 的大错位是 X 符号干扰，本模块把它量化了）：
在这套渲染里，符号引脚腿画成**符号色**（暗红），导线画成**网络色**（绿色），
而「未连接（NC）」的脚在腿的外端画一个**网络色的 X**（两笔斜线交叉）。
`work/_xnc.py` 在官方 10 张上检出 3 个 X，其中 2 个**中心落在 GT 脚点 2.0px 内**
（case8 `U1 pin_1`、`U1 pin_6`）——而我们把这两根脚发在了 169.8px / 9.0px 外。
所以这个 X 是一个**可测的引脚端点证据**，不像腿长那样不可测。

实现：在墨迹掩膜上找「两条对角线都够黑」的小方块（9~20px），把它当候选脚点：
  ① `snap`：某个 box/ic 元件的已有引脚离 X 中心 ≤ `XMARK_SNAP`（默认 12px）⇒ 把该脚移到 X 中心；
  ② `add` ：X 落在某个 box/ic 的框外 `XMARK_ADD_LO..HI`（默认 3..26px）且附近没有别的脚
            ⇒ **新增**一个脚（`XMARK_ADD` 默认 0，因为多发一根脚要付 fp）。
每个 X 只用一次；命中/新增都写进 `stat` 账本，便于 A/B 时对账。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from . import labels as _labels  # noqa: F401  (保持与其它 stage1 模块一致的导入习惯)


def _env_on(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() not in ("", "0", "false", "off", "none")


XMARK_ON = _env_on("STAGE1_PIN_XMARK", "0")
XMARK_ADD = _env_on("STAGE1_PIN_XMARK_ADD", "0")
XMARK_SNAP = float(os.environ.get("STAGE1_PIN_XMARK_SNAP", "12") or 12)
XMARK_LO = float(os.environ.get("STAGE1_PIN_XMARK_LO", "3") or 3)
XMARK_HI = float(os.environ.get("STAGE1_PIN_XMARK_HI", "26") or 26)
XMARK_MIN, XMARK_MAX = 12, 24


def _mark_mask(src) -> np.ndarray:
    """彩色 X 的候选像素：**非红色的彩色**像素（给 BGR 图），或原样的二值掩膜。

    为什么是「非红」而不是「有彩色」：这套渲染把符号引线画成暗红、导线画成网络色
    （case8 绿），X 用网络色画在引线外端。用「任何有饱和度」会把暗红引线一起进来，
    X 与引线连成一块，尺寸过滤直接丢光（实测 0 命中）；排掉红色后 X 是孤立的
    ~18×18 方块（实测两对角覆盖 0.94~1.00）。
    """
    a = np.asarray(src)
    if a.ndim == 3:
        hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        return (((s >= 60) & (v >= 40) & (h > 12) & (h < 168))).astype(np.uint8)
    return (a > 0).astype(np.uint8)


def find_x_marks(ink) -> List[Tuple[float, float, int, int]]:
    """找 X 形小方块，返回 [(cx, cy, w, h)]。`ink` 可以是 BGR 图或二值掩膜。"""
    if ink is None:
        return []
    m = _mark_mask(ink)
    if m.ndim != 2 or m.sum() == 0:
        return []
    n, _lab, st, _ce = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (XMARK_MIN <= w <= XMARK_MAX and XMARK_MIN <= h <= XMARK_MAX):
            continue
        if abs(int(w) - int(h)) > 6:
            continue
        if area < 20 or area > 0.65 * w * h:
            continue
        sub = m[y:y + h, x:x + w]
        diag_a = [int(sub[min(int(kk * h / max(w, 1)), h - 1), kk]) for kk in range(w)]
        diag_b = [int(sub[min(int(kk * h / max(w, 1)), h - 1), w - 1 - kk]) for kk in range(w)]
        fa, fb = sum(diag_a) / float(w), sum(diag_b) / float(w)
        if fa < 0.80 or fb < 0.80:
            continue
        out.append((x + w / 2.0, y + h / 2.0, int(w), int(h)))
    return out


def _band(bb: Sequence[float], cx: float, cy: float) -> bool:
    """点在框外、且到框的欧氏距离落在 [XMARK_LO, XMARK_HI] 里。"""
    x0, y0, x1, y1 = [float(v) for v in bb]
    dx = max(x0 - cx, 0.0, cx - x1)
    dy = max(y0 - cy, 0.0, cy - y1)
    return XMARK_LO <= math.hypot(dx, dy) <= XMARK_HI


def snap_x_marks(comps, comp_pins: Dict[str, List[Tuple[float, float]]], ink,
                 add: bool = None):
    """把 X 标记并进引脚点。返回 (comp_pins, stat)。"""
    stat = {"marks": 0, "snap": 0, "add": 0}
    if not XMARK_ON:
        return comp_pins, stat
    marks = find_x_marks(ink)
    stat["marks"] = len(marks)
    if not marks:
        return comp_pins, stat
    if add is None:
        add = XMARK_ADD
    boxes = []
    for c in comps:
        bb = c.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        ct = str(c.get("type") or "").lower()
        if "box" not in ct and "bigic" not in ct:
            continue
        cid = c.get("id") or c.get("uid") or c.get("name")
        if cid is None or cid not in comp_pins:
            continue
        boxes.append((cid, bb))

    out = {k: list(v) for k, v in comp_pins.items()}
    used = set()
    # ① 先「移已有脚」：优先照最近的脚，抢最小距离的配对
    pairs = []
    for cid, bb in boxes:
        for pi, (px, py) in enumerate(out.get(cid) or []):
            for mi, (mx, my, _w, _h) in enumerate(marks):
                d = math.hypot(mx - px, my - py)
                if d <= XMARK_SNAP and _band(bb, mx, my):
                    pairs.append((d, cid, pi, mi))
    pairs.sort()
    taken_pin = set()
    for d, cid, pi, mi in pairs:
        if mi in used or (cid, pi) in taken_pin:
            continue
        mx, my, _w, _h = marks[mi]
        out[cid][pi] = (round(mx, 2), round(my, 2))
        used.add(mi)
        taken_pin.add((cid, pi))
        stat["snap"] += 1
    # ② 再「补脚」：X 在框外的腿长区间里、附近 8px 没有脚
    if add:
        for mi, (mx, my, _w, _h) in enumerate(marks):
            if mi in used:
                continue
            best = None
            for cid, bb in boxes:
                if not _band(bb, mx, my):
                    continue
                if any(math.hypot(mx - px, my - py) <= 8.0 for px, py in (out.get(cid) or [])):
                    continue
                d = min(abs(mx - float(bb[0])), abs(mx - float(bb[2])),
                        abs(my - float(bb[1])), abs(my - float(bb[3])))
                if best is None or d < best[0]:
                    best = (d, cid)
            if best is None:
                continue
            out[best[1]].append((round(mx, 2), round(my, 2)))
            used.add(mi)
            stat["add"] += 1
    return out, stat
