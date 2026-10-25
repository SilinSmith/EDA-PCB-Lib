# -*- coding: utf-8 -*-
"""stage1/pstub.py —— 引脚点必须有「引线桩」支撑（`STAGE1_PIN_XSTUB`，默认 1）。

动机（§5.397.6/§5.397.7）：case8 `U1` 左面出现相距 ~10px 的**重复行**——脚号数字贴着
元件面印、`NC` 的绿色 X 压在腿端，这两样都不是引线桩，却都在面上生成了候选行；多出来
一行之后，脚号/脚名整体错绑一行（5 条 GT 脚变「缺键」、6 条变「位置错」）。
「过近就丢」「等距格点拟合」「墨迹厚度分组」三种后处理判据都判否（§5.397.6/.7），
但**单点判据**是有效的：真引线桩是薄横条，从元件面**一直外伸** ≥10px，所以
「面外 6px 与 9px 处的垂直厚度都 < 阈值」的脚点根本没有桩。

实测（官方 10 张、`work/_fast_out_xdef10` 产物、离线 `work/_rowfilter.py`）：
丢 6 根（case5 `U1 pin_7` / `U3 pin_3`，case8 `U1 pin_1` / `pin_12` / `pin_2` / `pin_13`），
pin **0.8570 → 0.8595**、总分 **90.09 → 90.15**；阈值 (2,2)/(2,3)/(3,2) 丢的是同一批 6 根，
(3,3) 就开始丢真脚（46 根，pin 0.6606）⇒ 判据在很窄的一档有效，默认取 (2,2)。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import labels as _labels  # noqa: F401  (与其它 stage1 模块保持一致的导入习惯)


def _env_on(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() not in ("", "0", "false", "off", "none")


STUB_ON = _env_on("STAGE1_PIN_XSTUB", "1")
STUB_T6 = int(os.environ.get("STAGE1_PIN_XSTUB_T6", "2") or 2)
STUB_T9 = int(os.environ.get("STAGE1_PIN_XSTUB_T9", "2") or 2)
STUB_INK = int(os.environ.get("STAGE1_PIN_XSTUB_INK", "170") or 170)
# 页面级自校验：只有「绝大多数脚都能在面外找到桩」的页面，才相信这个探针。
# 异源页（池子 200_train_cases）有 ~26~38% 的脚在面外 4~14px 找不到墨迹（渲染/框口径不同），
# 无条件开门会把它们当成「无桩」误删（实测池 50 案 34.47 → 34.10）；官方 10 张 100% 通过。
STUB_PAGE = float(os.environ.get("STAGE1_PIN_XSTUB_PAGE", "0") or 0)
# 「重复行」守卫：只有在这根无桩脚的 NEAR 像素内**还有另一根脚**时才算重复行（`0` = 不设守卫）。
# 这直接对应 §5.397.6 观察到的坏例：多出来的一行紧贴真行 ~10px。
STUB_NEAR = float(os.environ.get("STAGE1_PIN_XSTUB_NEAR", "14") or 14)
_STUB_TYPES = ("box", "ic")


def _ink_mask(src) -> np.ndarray:
    a = np.asarray(src)
    if a.ndim == 3:
        gray = a[:, :, 0].astype(np.int32) * 0.114 + a[:, :, 1].astype(np.int32) * 0.587 \
            + a[:, :, 2].astype(np.int32) * 0.299
        return (gray < STUB_INK).astype(np.uint8)
    return (a > 0).astype(np.uint8)


def _thick(m: np.ndarray, x: float, y: float, axis: str) -> int:
    """(x,y) 处沿 axis 的连续墨迹厚度；该点本身不是墨迹则返回 0。"""
    H, W = m.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < W and 0 <= yi < H) or m[yi, xi] == 0:
        return 0
    n = 1
    for s in (-1, 1):
        i = 1
        while True:
            xx, yy = (xi, yi + s * i) if axis == "y" else (xi + s * i, yi)
            if not (0 <= xx < W and 0 <= yy < H) or m[yy, xx] == 0:
                break
            n += 1
            i += 1
    return n


def filter_stubs(comps, comp_pins: Dict[str, List[Tuple[float, float]]], ink):
    """丢掉「面外 6px 与 9px 处都没有墨迹」的脚点。返回 (comp_pins, stat)。"""
    stat = {"drop": 0, "keep": 0, "page": 1.0}
    if not STUB_ON or ink is None:
        return comp_pins, stat
    m = _ink_mask(ink)
    if m.ndim != 2 or m.sum() == 0:
        return comp_pins, stat
    # ① 先量全页「有桩率」，决定这一页要不要开门
    probes = []
    for c in comps:
        bb = c.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        ct = str(c.get("type") or "").lower()
        if not any(t in ct for t in _STUB_TYPES):
            continue
        cid = c.get("id") or c.get("uid") or c.get("name")
        pins = comp_pins.get(cid)
        if cid is None or not pins or len(pins) < 2:
            continue
        x0, y0, x1, y1 = [float(v) for v in bb]
        for px, py in pins:
            dx, dy, _d = min(((-1, 0, abs(px - x0)), (1, 0, abs(px - x1)),
                              (0, -1, abs(py - y0)), (0, 1, abs(py - y1))), key=lambda t: t[2])
            probes.append((cid, px, py, dx, dy, _d, x0, y0, x1, y1))
    if not probes:
        return comp_pins, stat
    ok = 0
    for cid, px, py, dx, dy, _d, x0, y0, x1, y1 in probes:
        if _d > 60:
            ok += 1
            continue
        fx = x0 if dx == -1 else x1 if dx == 1 else px
        fy = y0 if dy == -1 else y1 if dy == 1 else py
        axis = "y" if dy == 0 else "x"
        if _thick(m, fx + dx * 6, fy + dy * 6, axis) >= STUB_T6 or \
                _thick(m, fx + dx * 9, fy + dy * 9, axis) >= STUB_T9:
            ok += 1
    stat["page"] = ok / float(len(probes))
    if os.environ.get("STAGE1_PIN_XSTUB_DEBUG", "").strip() not in ("", "0"):
        import sys
        sys.stderr.write("[pstub] probes=%d page=%.3f\n" % (len(probes), stat["page"]))
    if STUB_PAGE > 0 and stat["page"] < STUB_PAGE:
        return comp_pins, stat
    by_comp = {}
    for cid, px, py, dx, dy, _d, x0, y0, x1, y1 in probes:
        by_comp.setdefault(cid, []).append((px, py))
    drop = set()
    for cid, px, py, dx, dy, _d, x0, y0, x1, y1 in probes:
        if _d > 60:
            continue
        fx = x0 if dx == -1 else x1 if dx == 1 else px
        fy = y0 if dy == -1 else y1 if dy == 1 else py
        axis = "y" if dy == 0 else "x"
        if _thick(m, fx + dx * 6, fy + dy * 6, axis) < STUB_T6 and \
                _thick(m, fx + dx * 9, fy + dy * 9, axis) < STUB_T9:
            if STUB_NEAR > 0:
                near = min([((qx - px) ** 2 + (qy - py) ** 2) ** 0.5
                            for qx, qy in by_comp.get(cid, []) if (qx, qy) != (px, py)] or [1e9])
                if near > STUB_NEAR:
                    continue
            drop.add((cid, px, py))
    out = {}
    for k, v in comp_pins.items():
        keep = [(px, py) for (px, py) in v if (k, px, py) not in drop]
        stat["drop"] += len(v) - len(keep)
        stat["keep"] += len(keep)
        out[k] = keep
    return out, stat
