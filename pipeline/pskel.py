# -*- coding: utf-8 -*-
"""Wire skeleton extraction: raster -> drawing graph -> axis-aligned polyline segments.

    ink -> thin-stroke mask (wire/pin strokes only) -> Zhang-Suen skeleton
        -> 8-connected graph, pixels collapsed into nodes (endpoint / T / X / corner)
        -> path per skeleton branch, Douglas-Peucker + 0/45/90 lattice snap
        -> junction resolution: pair branches "straight through" (crossing) vs. unpaired (T)
        -> trace half-edges through the pairings -> maximal polylines
        -> split into two-point edges and union-find them into nets

Everything is in image coordinates (top-left origin); the single y-flip to competition
JSON happens in ``config.img_to_json``.

The stroke-width gate is what makes this work on a greyscale datasheet where colour is
useless: wires are ~2 px strokes, glyphs and symbol bodies are thicker.
"""
from __future__ import annotations

import math
import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- morphology

def zhang_suen(img: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen thinning on a uint8 0/255 mask.

    8 shift-add planes and one boolean remove-mask per sub-iteration; returns as soon as a
    sub-iteration deletes nothing, which is the common case for schematic line work (a
    skeleton of a 1675x1190 sheet converges in ~20 sweeps).
    """
    I = (img > 0).astype(np.uint8)
    # cache the eight neighbours once per *pixel*; they only change where we delete
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = np.pad(I, 1)
            p2 = P[0:-2, 1:-1]; p3 = P[0:-2, 2:]; p4 = P[1:-1, 2:]
            p5 = P[2:, 2:];     p6 = P[2:, 1:-1]; p7 = P[2:, 0:-2]
            p8 = P[1:-1, 0:-2]; p9 = P[0:-2, 0:-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            A = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            A += ((p3 == 0) & (p4 == 1))
            A += ((p4 == 0) & (p5 == 1))
            A += ((p5 == 0) & (p6 == 1))
            A += ((p6 == 0) & (p7 == 1))
            A += ((p7 == 0) & (p8 == 1))
            A += ((p8 == 0) & (p9 == 1))
            A += ((p9 == 0) & (p2 == 1))
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            rm = (I == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rm.any():
                I[rm] = 0
                changed = True
        if not changed:
            break
    return (I * 255).astype(np.uint8)


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbours(pts: set, p) -> List[Tuple[int, int]]:
    x, y = p
    return [(x + dx, y + dy) for dx, dy in _NB if (x + dx, y + dy) in pts]


# ---------------------------------------------------------------- stroke masks

def estimate_bg(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    ring = np.concatenate([bgr[0:2].reshape(-1, 3), bgr[h - 2:h].reshape(-1, 3),
                           bgr[:, 0:2].reshape(-1, 3), bgr[:, w - 2:w].reshape(-1, 3)])
    return np.median(ring, axis=0).astype(np.float32)


def estimate_bg_local(bgr: np.ndarray, k: int = 51) -> np.ndarray:
    """**局部**背景（纸面）估计：中值模糊。

    `k` 必须远大于笔画宽度（导线 2~6px、文字 ~14px），中值才会把笔画整个滤掉、
    只留下纸面。白纸有渐晕/偏色/网格底纹，边框环形中值（`estimate_bg`）在纸中央
    会偏，局部估计不会。
    """
    h, w = bgr.shape[:2]
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(3, min(k, (min(h, w) // 2) * 2 + 1))
    return cv2.medianBlur(bgr, k)


def bg_distance(bgr: np.ndarray, bg: np.ndarray | None = None, space: str = 'lab',
                k: int = 51) -> np.ndarray:
    """**与背景的偏离量**（任意颜色方向）—— 前景判据的本来面目。

    用户 m40034：「背景是白色的，而导线的颜色不确定……只要从背景过渡到任意颜色，
    都有可能是导线」。所以判据既不是灰度、也不是某个颜色，而是**离背景多远**：
    亮度偏、彩度偏、色相偏、任意组合偏，都算前景。

    * ``space='lab'`` —— Lab 里的欧氏距离（ΔE，感知均匀：亮度与彩度同权，不会像
      灰度那样把「等亮度但不同色」的边界判成背景，也不会像逐通道最大差那样把
      单通道的大偏色放大）。
    * ``space='bgr'`` —— 旧的逐通道最大绝对差（`bgdiff` 语义，保留兼容）。

    ``bg=None`` ⇒ 用**局部**背景（`estimate_bg_local`）；给了 `bg` 就用它（例如边框
    环形中值）。

    ⚠ 陷阱：`cv2.cvtColor` 对 **CV_32F** 输入用另一套刻度（L∈0..100、a/b∈−127..127），
    把 0..255 的 float 白纸喂进去会得到 ΔE≈238 的假距离（整张图全成前景）。所以
    Lab 转换一律走 uint8，只在算距离时转 float。
    """
    if bg is None:
        bg = estimate_bg_local(bgr, k)
    bg = np.asarray(bg, dtype=np.float32)
    if bg.ndim == 1:
        bg = bg.reshape(1, 1, 3)
    if space == 'bgr':
        return np.abs(bgr.astype(np.float32) - bg).max(axis=2)
    bgu = np.clip(bg, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(bgu, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.sqrt(((lab - lab_bg) ** 2).sum(axis=2))


def ink_mask(bgr: np.ndarray, mode: str = 'adaptive', k: int = 31,
             diff_tol: int = 26, c: int = 8) -> np.ndarray:
    """Foreground = everything that is not the sheet background.

    ``adaptive`` cv2 adaptive mean threshold on the inverted grey level (handles the
                 vignetting / gradient backgrounds of the photographed sheets)
    ``otsu``     global Otsu on the inverted grey level
    ``bgdiff``   per-channel distance from the border-ring background colour
    ``bglab``    ★ Lab ΔE from the **local** background, Otsu on the distance map
                 （用户 m40034 的判据：从背景过渡到**任意颜色**都可能是导线。
                 灰度自适应只能看见亮度跳变；这一模式把亮度/彩度/色相的偏离
                 等权地当候选，再由下游的细笔画 + 交叉验证决定是不是导线。）
    """
    if mode == 'bglab':
        d = bg_distance(bgr, bg=None, space='lab')
        dn = np.clip(d * 4.0, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(float(diff_tol), float(t) / 4.0)
        return ((d >= thr) * 255).astype(np.uint8)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - g
    if mode == 'otsu':
        _, m = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m
    if mode == 'bgdiff':
        bg = estimate_bg(bgr)
        d = bg_distance(bgr, bg=bg, space='bgr')
        return ((d > diff_tol) * 255).astype(np.uint8)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, k, c)


# `know` 节点用：由 `stage1/wires.py` 问一次模型「导线是什么颜色」后写进来的强制色相。
HINT_HUE = None


def chroma(bgr: np.ndarray) -> np.ndarray:
    b = bgr.astype(np.int16)
    return (b.max(axis=2) - b.min(axis=2)).astype(np.float32)


def colour_edge(bgr: np.ndarray, blur: float = 1.2) -> np.ndarray:
    """**彩色梯度**幅值（Di Zenzo 结构张量）—— 不依赖亮度。

    灰度自适应阈值只能看见「亮度跳变」。等亮度但不同色的边界（浅色细走线压在浅色
    网格上、扫描偏色的印刷、低对比底纹上的彩线）在灰度里根本不存在，而在彩色梯度里
    是最强的一类边界 —— 这正是用户 m40034 说的「按像素颜色跳变判导线」。

    做法：对 B/G/R 三通道各求 Sobel，像素级 2x2 结构张量
    `[[Σgx², Σgxgy], [Σgxgy, Σgy²]]` 的最大特征值开方，即「任一通道变化的最大响应」。
    纯灰度边界（三通道同向变化）与纯色边界（等亮度反色）都会得到正响应。
    """
    f = bgr.astype(np.float32)
    if blur and blur > 0:
        f = cv2.GaussianBlur(f, (0, 0), float(blur))
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = (gx * gx).sum(axis=2)
    gyy = (gy * gy).sum(axis=2)
    gxy = (gx * gy).sum(axis=2)
    tmp = np.sqrt(np.maximum(0.0, (gxx - gyy) ** 2 + 4.0 * gxy * gxy))
    return np.sqrt(np.maximum(0.0, 0.5 * (gxx + gyy + tmp)))


def colour_edge_mask(bgr: np.ndarray, ink: np.ndarray, max_half_width: float = 2.6,
                     rel: float = 0.55, abs_min: float = 10.0,
                     close: bool = True):
    """彩色边缘 → **细笔画**掩膜（与灰度路径取并用，只加不减）。返回 `(mask, thr)`。

    阈值是**相对**的：`max(abs_min, rel * p90(幅度 | 墨迹))`。写死阈值在扫描偏色/低对比
    图上必然失效，而「以本图墨迹内的 p90 为尺」在两种极端下都能自适应（彩色图上边缘
    响应高 ⇒ 门槛随之抬高，不会把整张纸都判成边缘）。
    """
    mag = colour_edge(bgr)
    v = ink > 0
    if v.any():
        p = float(np.percentile(mag[v], 90))
    else:
        p = float(np.percentile(mag, 99)) if mag.size else 0.0
    thr = max(float(abs_min), float(rel) * p)
    ce = ((mag >= thr) * 255).astype(np.uint8)
    if close:
        ce = cv2.morphologyEx(ce, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    thin, _dt = thin_stroke_mask(bgr, ce, max_half_width)
    return thin, thr


def chroma_floor(bgr: np.ndarray, ink: np.ndarray, rel: float = 0.35,
                 abs_min: float = 12.0):
    """**相对**彩度门（替代写死的 `22.0`）。返回 `(阈值, 彩度图)`。

    写死的 22.0 在两种图上都是错的：低饱和扫描件里真导线只有 ~10，高饱和矢量图里
    底纹就有 ~40。改用「墨迹内彩度 p95」为尺，两者都能分开。
    """
    ch = chroma(bgr)
    v = ink > 0
    p = float(np.percentile(ch[v], 95)) if v.any() else 0.0
    return max(float(abs_min), float(rel) * p), ch


def thin_stroke_mask(bgr: np.ndarray, ink: np.ndarray, max_half_width: float = 1.6):
    """Keep only ink whose half stroke width (2*EDT) is <= *max_half_width*."""
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    thin = (ink > 0) & (dt <= max_half_width)
    return (thin.astype(np.uint8) * 255), dt


def _shift_mask(m: np.ndarray, dy: int, dx: int):
    """把布尔掩膜整体平移 (dy, dx)，越界部分按 0 补（不环绕）。"""
    h, w = m.shape
    out = np.zeros_like(m)
    ys = slice(max(0, -dy), h - max(0, dy))
    yd = slice(max(0, dy), h - max(0, -dy))
    xs = slice(max(0, -dx), w - max(0, dx))
    xd = slice(max(0, dx), w - max(0, -dx))
    out[yd, xd] = m[ys, xs]
    return out


def cross_core_fill(thin: np.ndarray, dt: np.ndarray, max_half_width: float = 2.6,
                    r: float = 8.0, min_axes: int = 2):
    """把「两条细线十字相交」处被 `thin_stroke_mask` 挖掉的**十字芯**补回来。

    病（§5.107 的图像级取证，现场 case 1 x≈843、y≈173）：`thin_stroke_mask` 按「到墨迹
    边界的距离 ≤ max_half_width」留像素，两条线相交处十字芯的 dt 最大（实测 4~6 > 2.6）
    ⇒ 整块被当粗线剔掉；骨架只好**绕着芯走**（旁移 5~6px 再回来），段提取把这段绕行丢掉
    ⇒ 出口边在每个交叉处留 ~11px 的洞（我方 case 1 x≈843 出 9 条 25~33px，GT 是 13 条
    29~30px 首尾相接）。

    药：只补「**两个轴向**上都夹着细笔画」的胖像素 —— 上下有细笔画（竖线穿过）且左右也有
    （横线穿过）才是十字芯。粗导线的芯只在法向两侧有细笔画（只有一个轴向），不被补；
    元件框/文字/照片底纹内部附近没有细笔画，也不被补（`STAGE1_WIRE_CLOSE_K` 那种无差别
    闭运算做不到这个区分，池子回归 −0.47）。`r` 是找细笔画的最大距离（十字芯的 dt 最大
    约 6，细笔画自己再占 2px ⇒ 8px 够）。返回新的 uint8(0/255) 掩膜，未命中时原样返回。
    """
    th = np.asarray(thin) > 0
    if not th.any():
        return thin
    thick = np.asarray(dt) > float(max_half_width)
    if not thick.any():
        return thin
    R = max(1, int(round(float(r))))
    dirs = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    ev = []
    for dy, dx in dirs:
        acc = th.copy()
        for k in range(1, R + 1):
            acc |= _shift_mask(th, dy * k, dx * k)
        ev.append(acc)
    both = np.zeros(th.shape, np.int16)
    for i in range(4):
        both += (ev[i] & ev[i + 4]).astype(np.int16)
    keep = thick & (both >= int(min_axes))
    if not keep.any():
        return thin
    return ((th | keep).astype(np.uint8) * 255)


def _palette_estimate(bgr: np.ndarray, ink: np.ndarray):
    """`pipeline/palette.py` 的懒加载入口（唯一颜色接口）；不可用时退化成「无彩」。"""
    try:
        from . import palette as _pal                 # 作为 package 导入
    except Exception:                                 # noqa: BLE001
        try:
            from pipeline import palette as _pal      # 作为顶层模块导入
        except Exception:                             # noqa: BLE001
            return {"has_chroma": False, "palette": []}
    try:
        return _pal.estimate(bgr, ink=(ink > 0))
    except Exception:                                 # noqa: BLE001
        return {"has_chroma": False, "palette": []}


def _hue_in_palette(h0: float, buckets, tol: float) -> bool:
    """调色板里有没有一串占比达标的色桶落在这个色相 ±tol 内。"""
    for b in buckets or []:
        d = abs(float(b.get("hue", 0.0)) - h0)
        d = min(d, 180.0 - d)
        if d <= tol:
            return True
    return False


def wire_colour_mask(bgr: np.ndarray, ink: np.ndarray, hue_tol: float = 42.0):
    """Pixels whose hue matches the dominant ink hue (the wire colour), or ``None``.

    ``None`` means "no usable chroma" (greyscale datasheet) -- the caller then keeps the
    stroke-width route only.

    模式由环境变量 ``STAGE1_WIRE_COLOUR`` 控制（**默认 ``confirm``**）：

    * ``hint`` —— 由 `stage1/wires.py` 在提线前先问模型一次「这张图的导线是什么
      颜色」（灰度图零成本跳过），把色相写进模块级 ``HINT_HUE``；**只在主色相是红系
      （``h0 < 25`` 或 ``h0 > 155`` = 图框/符号色）时才采纳**，否则维持 ``dominant``。
      19 张快测：24.87 vs ``dominant`` 24.81，导线 F1 0.0146→0.0201、导线 tp 29→52。
    * ``confirm`` / ``hint_confirm``（**默认**；用户 m56804/m56805：「颜色不是确定信息」）——
      **颜色假设必须在本图调色板里得到证实，否则这条颜色证据作废（返回 ``None`` ⇒ 一个
      像素都不删）**。判据来自 `pipeline/palette.py`（唯一颜色入口）：主色相必须落在
      调色板某个占比 ≥2% 的色桶 ±``hue_tol`` 内。要点是**失败方向是「不删」而不是
      「删光」**：旧 ``hint`` 路径在模型答错色相时仍会按那个色相过滤，主色相恰是图框色时
      整张图的导线会被抹掉（``dominant`` 在 case ``0001`` 上留过 6px 绿的历史事故）。
      **为什么翻成默认（2026-10-17 实测）**：`work/_cf_ab.ps1` 官方 10 张上 `confirm` 与
      `hint` **逐字节完全相同**（11 个文件里只有 `run.log` 不同，总分都是 79.14；
      四个尺子 protocol2/protocol/harness/judge 也全等），而派生 val 19 上 `confirm`
      **30.96 vs hint 30.83（+0.13）** ⇒ 更严的门在官方口径上零代价、在派生语料上还略赚，
      同时把「答错色相就删光」这个失败方向彻底关掉。
    * ``h<N>`` —— 实验用：强制指定导线色相（如 ``h55``）。
    * ``dominant``（旧默认）—— 取 ink 里**像素数最多**的那个色相，只留 ±``hue_tol``。
      **这个假设在多色原理图上会翻车**：KiCad 导出的图里图框与元件符号是暗红
      （色相 ~5°）、导线是绿（~85°）、网络标签是蓝（~125°），红色像素数最多 ⇒
      掩膜把**整张图的导线全删掉**。实测 case ``0001``：`thin` 掩膜剩 67042 px，
      其中绿色只剩 **6 px**（导线全灭）。
    * ``chroma`` —— 只丢掉**没有彩度**的墨迹（文字/灰度底图），保留所有有彩度的
      像素（导线、符号、标签都留）。实测 case ``0001`` 保住绿色 12248/13122 = 0.933。
    * ``auto`` —— **逐图**选导线色：在**图框以内**做 >=40px 的长直线（横/竖开运算），
      选长直线像素最多的那个候选色相。先决条件是「导线是图里唯一的长直线」，
      而图框本身的长直线全在 4.5% 的边带上，排除掉才不会让「框的颜色」赢。
      **实测失效**：它给 ``0001`` 选的还是红 h35（图框内的分隔线比导线更长更多）。
    * ``off`` —— 直接不做颜色过滤。
    """
    # 默认 `confirm`（2026-10-17 起）：仍然问一次模型「这张图的导线是什么颜色」，但**那个色相
    # 必须在本图调色板（`pipeline/palette.py`）里被证实**，否则整条颜色证据作废、一个像素都不删。
    # 取证：官方 10 张 `confirm` 与旧默认 `hint` 逐字节相同（79.14），派生 val 19 上 30.96 vs 30.83。
    mode = os.environ.get("STAGE1_WIRE_COLOUR", "confirm").strip().lower()
    hint_want = None
    confirm = mode in ("confirm", "hint_confirm", "hint+confirm", "verify")
    if confirm:
        mode = "hint"
    if mode == "hint":
        # `know` 节点：`stage1/wires.py` 先问一次模型「这张图的导线是什么颜色」，写进 `HINT_HUE`。
        # **只在「主色相是红系（图框/符号色）」时才有资格改写** —— 主色相本来就是绿系
        # （比如 `9` 的 h65），说明掩膜没删错东西，别去动它。
        hint_want = HINT_HUE
        mode = "dominant"
    if mode in ("off", "none", "0", "false"):
        return None
    if confirm:
        # ★ 用户 m56804/m56805「颜色并不是确定信息」：颜色只是一条**需要被证实的证据**。
        #   本图调色板（`pipeline/palette.py`，唯一颜色入口）里找不到这个色相 ⇒ 假设作废，
        #   返回 None = **不做任何颜色过滤**（宁可少删，不可删错）。
        pal = _palette_estimate(bgr, ink)
        if not pal.get("has_chroma"):
            return None
        if hint_want is None:
            return None
        if not _hue_in_palette(float(hint_want), pal["palette"], hue_tol):
            if os.environ.get("STAGE1_WIRE_DEBUG"):
                print("  [wire] confirm: hint=%s 未在调色板中证实 %s -> 不做颜色过滤"
                      % (hint_want, [b_["hue"] for b_ in pal["palette"][:4]]), flush=True)
            return None
    ch = chroma(bgr)
    v = (ink > 0)
    if mode in ("rel", "chroma_rel"):
        # ★ 相对彩度门（2026-10-10）：写死的 22.0 对低饱和扫描件太高、对高饱和矢量图太低。
        t, ch = chroma_floor(bgr, ink)
        if not v.any():
            return None
        return (((ch >= t) & v) * 255).astype(np.uint8)
    if not v.any() or float(np.percentile(ch[v], 90)) < 22.0:
        return None
    if mode == "chroma":
        return (((ch >= 22.0) & v) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    hist = np.bincount((h[v] // 10).astype(np.int32), minlength=18)
    if mode.startswith("h") and mode[1:].isdigit():
        # 实验用：强制指定导线色相，例如 STAGE1_WIRE_COLOUR=h55 / h85
        h0 = float(mode[1:])
        d = np.abs(h - h0)
        d = np.minimum(d, 180.0 - d)
        return (((d <= hue_tol) & v) * 255).astype(np.uint8)
    if mode == "auto":
        # 逐图选「导线色」：**图框以内**的长直线（>=40px 的横/竖开运算）最多的那个色相。
        # 两个要点：①只在**细笔画**上判（粗的实心填充/图框会淹没直方图）；
        # ②排除 4.5% 的图框边带，否则「框的颜色」永远赢。
        _t, _ = thin_stroke_mask(bgr, ink, 1.6)
        _t = _t > 0
        if not _t.any():
            return None
        h_all = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32)
        hist_t = np.bincount((h_all[_t] // 10).astype(np.int32), minlength=18)
        H_, W_ = bgr.shape[:2]
        ey, ex = int(H_ * 0.045), int(W_ * 0.045)
        inner = np.zeros((H_, W_), bool)
        inner[ey:H_ - ey, ex:W_ - ex] = True
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        best_h, best_n = None, -1
        cand = [b_ for b_ in range(18) if hist_t[b_] >= max(1, hist_t.max() * 0.02)]
        for b_ in cand:
            h0 = float(b_) * 10.0 + 5.0
            dd = np.abs(h_all - h0)
            dd = np.minimum(dd, 180.0 - dd)
            mm = (((dd <= hue_tol) & _t).astype(np.uint8)) * 255
            lr = (cv2.morphologyEx(mm, cv2.MORPH_OPEN, kh) > 0) | \
                 (cv2.morphologyEx(mm, cv2.MORPH_OPEN, kv) > 0)
            n = int((lr & inner).sum())
            if n > best_n:
                best_h, best_n = h0, n
        h0 = best_h if best_h is not None else float(np.argmax(hist)) * 10.0 + 5.0
    else:
        h0 = float(np.argmax(hist)) * 10.0 + 5.0
    if hint_want is not None:
        # 主色相是「红系」（图框暗红 h~5）才接受模型的答案；否则维持 dominant。
        if h0 < 25.0 or h0 > 155.0:
            h0 = float(hint_want)
        if os.environ.get("STAGE1_WIRE_DEBUG"):
            print("  [wire] hint=%s dominant=%.0f -> h0=%.0f"
                  % (hint_want, float(np.argmax(hist)) * 10.0 + 5.0, h0), flush=True)
    d = np.abs(h - h0)
    d = np.minimum(d, 180.0 - d)
    return (((d <= hue_tol) & v) * 255).astype(np.uint8)



# ---------------------------------------------------------------- graph

@dataclass
class Path:
    idx: int
    n0: int
    n1: int
    pts: List[Tuple[float, float]]                 # xy, from n0 to n1
    length: float = 0.0


@dataclass
class Node:
    idx: int
    x: float
    y: float
    slots: List[int] = field(default_factory=list)      # half-edge ids in this node
    dirs: List[Tuple[float, float]] = field(default_factory=list)
    pair: Dict[int, int] = field(default_factory=dict)  # slot pos -> slot pos
    kind: str = 'E'


@dataclass
class HalfEdge:
    path: int
    end: int                                        # 0 = at path.n0, 1 = at path.n1
    node: int
    partner: int = -1                               # the half-edge from the other end


def build_graph(skel: np.ndarray):
    pts = {(int(y), int(x)) for y, x in zip(*np.nonzero(skel))}
    deg = {}
    for p in pts:
        deg[p] = sum(1 for q in neighbours(pts, p) if q in pts)
    node_px = {p for p in pts if deg[p] != 2}

    used = set()
    raw: List[List[Tuple[int, int]]] = []
    for p in node_px:
        for q in neighbours(pts, p):
            if (p, q) in used:
                continue
            used.add((p, q))
            walk = [p, q]
            prev, cur = p, q
            while cur not in node_px:
                nxt = [r for r in neighbours(pts, cur) if r in pts and r != prev]
                if not nxt:
                    break
                n2 = nxt[0]
                used.add((cur, n2)); used.add((n2, cur))
                walk.append(n2)
                prev, cur = cur, n2
            used.add((cur, prev)); used.add((prev, cur))
            raw.append(walk)
    return pts, deg, node_px, raw


# ---------------------------------------------------------------- 定向接线（交叉处的洞）
#
# 病（§5.107 的图像级取证）：`thin_stroke_mask` 只留「到墨迹边界距离 ≤ thin_w」的像素，
# 两条 4px 线相交处的**十字芯** dt 最大（~5px）⇒ 整块被当粗线剔掉 ⇒ 骨架在**每一个交叉**
# 处断 5~10px ⇒ 我们的边在交叉处留洞（GT 是「线连续、恰在交叉处切开、首尾相接」，case 1
# x≈843 上 GT 有 14 条 29~30px 的边，我们只出 9 条）。
#
# 为什么不用全图形态学闭运算：`STAGE1_WIRE_CLOSE_K=9` 官方 10 张 +0.08，但池子 10 张 −0.47
# （确定性 `--no-vlm` 口径）—— 9x9 椭圆核在照片类图纸上会把邻近笔画焊在一起。
# 这里改成**定向**接线：只补「两个骨架端点相向 + 缺口处两侧都有横穿骨架」的位置，
# 后者是交叉的特征，单纯断线/文字残渣没有，所以池子上不会乱焊。


def _endpoint_dirs(pts, k: int = 7):
    """骨架端点 → 指向**缺口**那一侧的单位向量。

    坐标沿用 `build_graph` 的约定：集合元素是 `(row, col)`。端点 = 8 邻域里只有 1 个骨架
    像素的点；方向 = 端点减去「沿线朝内走 k 步的那个点」，因此朝线外。
    """
    out = {}
    for p in pts:
        nb = [q for q in neighbours(pts, p) if q in pts]
        if len(nb) != 1:
            continue
        walk = [p, nb[0]]
        prev, cur = p, nb[0]
        while len(walk) <= k:
            nxt = [r for r in neighbours(pts, cur) if r in pts and r != prev]
            if not nxt:
                break
            n2 = nxt[0]
            walk.append(n2)
            prev, cur = cur, n2
        p0 = np.asarray(walk[0], dtype=float)
        p1 = np.asarray(walk[-1], dtype=float)
        v = p0 - p1
        n = float(np.hypot(float(v[0]), float(v[1])))
        if n <= 0.0:
            continue
        out[p] = (float(v[0]) / n, float(v[1]) / n)
    return out


def _has_perp_evidence(pts, my: float, mx: float, u, r: float) -> bool:
    """缺口两侧是否都有**横穿**方向的骨架像素（= 这里是交叉，不是单纯断线）。"""
    rr = max(4.0, float(r))
    left = right = False
    for (y, x) in pts:
        dy, dx = float(y) - my, float(x) - mx
        d = float(np.hypot(dy, dx))
        if d <= 0.0 or d > rr:
            continue
        along = (dy * u[0] + dx * u[1]) / d
        if abs(along) > 0.45:          # 与桥线同向 ⇒ 那是桥自己的两端，不算横穿
            continue
        side = dy * (-u[1]) + dx * u[0]
        if side > 0:
            left = True
        elif side < 0:
            right = True
        if left and right:
            return True
    return False


def _draw_seg_px(img: np.ndarray, p0, p1) -> None:
    y0, x0 = int(p0[0]), int(p0[1])
    y1, x1 = int(p1[0]), int(p1[1])
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    np.clip(ys, 0, img.shape[0] - 1, out=ys)
    np.clip(xs, 0, img.shape[1] - 1, out=xs)
    img[ys, xs] = 255


def bridge_skeleton_gaps(skel: np.ndarray, max_gap: float = 12.0,
                         max_angle_deg: float = 35.0, require_cross: bool = True):
    """返回 `(new_skel, n_bridges, bridges)`；`bridges` 里是 `(端点1, 端点2, 缺口长度)`。"""
    pts = {(int(y), int(x)) for y, x in zip(*np.nonzero(skel))}
    dirs = _endpoint_dirs(pts)
    eps = list(dirs.keys())
    cos_lim = float(np.cos(np.radians(max_angle_deg)))
    cands = []
    for i in range(len(eps)):
        y1, x1 = eps[i]
        d1 = dirs[eps[i]]
        for j in range(i + 1, len(eps)):
            y2, x2 = eps[j]
            d2 = dirs[eps[j]]
            dy, dx = float(y2 - y1), float(x2 - x1)
            g = float(np.hypot(dy, dx))
            if g <= 0.0 or g > max_gap:
                continue
            u = (dy / g, dx / g)
            if d1[0] * u[0] + d1[1] * u[1] < cos_lim:        # 端 1 的外指方向要指向缺口
                continue
            if d2[0] * -u[0] + d2[1] * -u[1] < cos_lim:      # 端 2 同理
                continue
            if require_cross and not _has_perp_evidence(
                    pts, (y1 + y2) / 2.0, (x1 + x2) / 2.0, u, max(4.0, max_gap * 0.8)):
                continue
            cands.append((g, eps[i], eps[j]))
    cands.sort()
    used, bridges = set(), []
    for _g, e1, e2 in cands:
        if e1 in used or e2 in used:
            continue
        used.add(e1); used.add(e2)
        bridges.append((e1, e2, _g))
    if not bridges:
        return skel, 0, []
    out = skel.copy()
    for e1, e2, _g in bridges:
        _draw_seg_px(out, e1, e2)
    return out, len(bridges), bridges


def snap_text_gaps(segs, boxes, max_gap: float = 30.0, max_angle_deg: float = 25.0,
                   k: int = 4, pad: float = 2.0, cover: float = 0.6,
                   max_short: float = 0.0, long_gap: float = 0.0):
    """抹字断口对接（§5.110）。

    病（§5.109 之后的下一步取证）：`extract_wires(erase=texts)` 把**盖在导线上的文字连同
    导线像素一起抹掉**，于是导线在每个文本框处断成两截。10GT case 1 x≈1432 实证（同一个
    窗口、同一份 stage1 输入、只切 `erase`）：
      `erase=[]`        出 GT 对齐的 6 条 [174,202]/[202,232]/[232,261]/[261,291]/[291,321]/[321,379]；
      `erase=texts`     出 [195,223]/[242,253]/[271,283]/[301,311]/[330,389] —— 中间三条 10~12px
                        的碎片夹在缺口里，而缺口**中点** 232.5/262/292/320.5 与 GT 的切点
                        232/261/291/320 相差 ≤1.5px。
    （`occlude` 元件框不背这个锅：`erase=[]` + `occlude=comps` 与完全不涂逐段相同。）

    规则：两个**共线相向**的端点，若其缺口被**同一个被抹掉的文本框**覆盖（两端都贴住该框、
    缺口上多数采样点落在框内），就把**两端都挪到缺口中点**（两端都变长，不会缩短任何一段）。
    与 `snap_collinear_gaps` 的区别：不看长度比 —— 抹字断口两侧常常**一样短**，长度比判据
    会整批拒绝；这里改看**抹字证据**，也正是 `pipeline.build_nets(bridges=texts)`
    （`STAGE1_TEXT_BRIDGE`）已经在用的判据：「只认两个线端各贴住同一个被擦除文本框的缺口」。

    `boxes` = `erase` 传来的文本框 `(x1,y1,x2,y2)`（原图像素）。返回 `(new_segs, n_moved)`。
    """
    cos_lim = math.cos(math.radians(max_angle_deg))
    nb = []
    for bx in boxes or []:
        try:
            x1, y1, x2, y2 = (float(v) for v in bx[:4])
        except (TypeError, ValueError):
            continue
        nb.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    if not nb or len(segs) < 2:
        return segs, 0

    def _near(p, bx, pad_):
        return (bx[0] - pad_ <= p[0] <= bx[2] + pad_) and (bx[1] - pad_ <= p[1] <= bx[3] + pad_)

    work = [[[float(p[0]), float(p[1])] for p in s] for s in segs]
    ends = []
    for i, s in enumerate(work):
        if len(s) < 2:
            continue
        L = sum(math.hypot(s[j + 1][0] - s[j][0], s[j + 1][1] - s[j][1])
                for j in range(len(s) - 1))
        for e in (0, 1):
            p = s[0] if e == 0 else s[-1]
            j = min(k, len(s) - 1)
            q = s[j] if e == 0 else s[-1 - j]
            vy, vx = p[1] - q[1], p[0] - q[0]
            n = math.hypot(vx, vy)
            if n < 1e-6:
                continue
            ends.append((i, e, p, (vx / n, vy / n), L))

    cands = []
    for a in range(len(ends)):
        ia, ea, pa, da, la = ends[a]
        for b in range(a + 1, len(ends)):
            ib, eb, pb, db, lb = ends[b]
            if ia == ib:
                continue
            # 「至少一侧是碎片」（`max_short>0` 时）：抹字断口是**导线被挖掉一小段**，
            # 因而至少一侧剩下的是 10~15px 的碎片；两条都长的缺口多半是两条真导线
            # 各自终止在同一个文本框旁（case 4/6/9 的反例），不该焊在一起。
            if max_short > 0 and min(la, lb) > max_short:
                # 两侧都不是碎片：只有「缺口本身很短」才可能是**单个引脚号/文字**压在线上
                # 挖出的一个小口子（长-长、缺口 ≤ `long_gap`）；缺口一大就是两条真导线
                # 各自终止在同一个文本框旁（case 4/6/9 的反例），不该焊在一起。
                if not (long_gap > 0 and math.hypot(pb[0] - pa[0], pb[1] - pa[1]) <= long_gap):
                    continue
            vx, vy = pb[0] - pa[0], pb[1] - pa[1]
            d = math.hypot(vx, vy)
            if d < 1e-6 or d > max_gap:
                continue
            ux, uy = vx / d, vy / d
            if da[0] * ux + da[1] * uy < cos_lim:            # a 的外指方向要指向缺口
                continue
            if -db[0] * ux - db[1] * uy < cos_lim:           # b 同理（两者共线相向）
                continue
            mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
            # 缺口的**多数采样点**要落在「被抹掉的文本框」里（并集覆盖，允许缺口跨两个相邻
            # 文本框 —— 引脚号常常上下两行，中间只隔 2~3px 的缝）
            ts = (0.12, 0.3, 0.5, 0.7, 0.88)
            pts = [(pa[0] + vx * t, pa[1] + vy * t) for t in ts]
            hit = sum(1 for q in pts if any(_near(q, bx, pad) for bx in nb))
            if hit < cover * len(ts):
                continue
            # 端点不必贴在框上（骨架常在框边又缩 3~5px），只要「中点在框内、两端离该框不远」
            if not any(_near(mid, bx, pad + 1.0) and _near(pa, bx, pad + 8.0)
                       and _near(pb, bx, pad + 8.0) for bx in nb):
                continue
            cands.append((d, a, b, mid))
    cands.sort(key=lambda t: t[0])

    used, n = set(), 0
    for _d, a, b, mid in cands:
        if a in used or b in used:
            continue
        ia, ea = ends[a][0], ends[a][1]
        ib, eb = ends[b][0], ends[b][1]
        work[ia][0 if ea == 0 else -1] = [mid[0], mid[1]]
        work[ib][0 if eb == 0 else -1] = [mid[0], mid[1]]
        used.add(a); used.add(b); n += 1
    return work, n


def snap_collinear_gaps(segs, max_gap: float = 18.0, max_angle_deg: float = 25.0,
                        k: int = 4, mode: str = "short", min_ratio: float = 0.9):
    """把「同一条走线在交叉处被切碎」的碎片接回相邻的长段（§5.109）。

    病（§5.107 取证、§5.109 复核）：`thin_stroke_mask` 在两条线的交叉处挖掉**十字芯**
    （那里到墨迹边界的距离最大）⇒ 骨架在交叉处**绕行**（旁移 5~6px 再回来）或断开 ⇒
    `snap_polyline`/`split_segments` 把绕行段切掉、只留下一条 5~15px 的碎片夹在两条
    27~30px 的长段之间 ⇒ stage2 的 `MIN_EDGE_LEN=20` 再把碎片丢掉 ⇒ 我们的边在**每个
    交叉**处留一个 ~11px 的洞。GT 是「线连续、恰在交叉处切开、首尾相接」：case 1 x≈843
    上 GT 有 13 条 29~30px 的边（115→497 首尾相接），我们只出 9 条。

    规则：两条共线碎片**相向**的端点距离 ≤ `max_gap`，且较短那条的长度 ≤ `min_ratio` ×
    较长那条，就把**短的那条的端点挪到长的那条的端点上**（不动长段）。
    为什么不按「两端都挪到缺口中点」：缺口不是以交叉中心对称的（骨架在交叉下方多丢
    11px），中点会把已经很准的长段推歪 5.5px（实测反而把 tp 变成 fp）——不动长段、
    只把碎片「撑」到相邻长段的端点上，长段的位置与 GT 对齐这一事实就保住了。

    `segs` 是 `[[x, y], ...]` 的折线列表（图像帧、y 向下，见 `stage1/wires.py`）。
    返回 `(new_segs, n_snapped)`；`mode="mid"` 是「两端都挪到中点」的对照臂。
    """
    cos_lim = math.cos(math.radians(max_angle_deg))

    def _plen(s):
        return sum(math.hypot(s[i + 1][0] - s[i][0], s[i + 1][1] - s[i][1])
                   for i in range(len(s) - 1))

    work = [[[float(p[0]), float(p[1])] for p in s] for s in segs]
    ends = []          # (idx, end, point, outward_dir, seg_len)
    for i, s in enumerate(work):
        if len(s) < 2:
            continue
        L = _plen(s)
        for e in (0, 1):
            p = s[0] if e == 0 else s[-1]
            j = min(k, len(s) - 1)
            q = s[j] if e == 0 else s[-1 - j]
            vy, vx = p[1] - q[1], p[0] - q[0]
            n = math.hypot(vx, vy)
            if n < 1e-6:
                continue
            ends.append((i, e, p, (vx / n, vy / n), L))

    cands = []
    for a in range(len(ends)):
        ia, ea, pa, da, la = ends[a]
        for b in range(a + 1, len(ends)):
            ib, eb, pb, db, lb = ends[b]
            if ia == ib:
                continue
            vx, vy = pb[0] - pa[0], pb[1] - pa[1]
            d = math.hypot(vx, vy)
            if d < 1e-6 or d > max_gap:
                continue
            ux, uy = vx / d, vy / d
            if da[0] * ux + da[1] * uy < cos_lim:            # a 的外指方向要指向缺口
                continue
            if -db[0] * ux - db[1] * uy < cos_lim:           # b 同理
                continue
            cands.append((d, a, b, la, lb))
    cands.sort(key=lambda t: t[0])

    used, n = set(), 0
    for _d, a, b, la, lb in cands:
        if a in used or b in used:
            continue
        ia, ea, pa, _da, la = ends[a]
        ib, eb, pb, _db, lb = ends[b]
        if mode == "mid":
            mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
            work[ia][0 if ea == 0 else -1] = [mx, my]
            work[ib][0 if eb == 0 else -1] = [mx, my]
            used.add(a); used.add(b); n += 1
            continue
        short_is_a = la <= lb * min_ratio
        short_is_b = lb <= la * min_ratio
        if not (short_is_a or short_is_b):
            continue                                          # 长度相当 ⇒ 不动（可能是两条真边）
        if short_is_a:
            work[ia][0 if ea == 0 else -1] = [pb[0], pb[1]]
        else:
            work[ib][0 if eb == 0 else -1] = [pa[0], pa[1]]
        used.add(a); used.add(b); n += 1
    return work, n


def _dir_at(walk: Sequence[Tuple[int, int]], from_end: int, k: int = 7):
    """Unit direction *leaving* the path: from_end=0 -> leaving walk[0]."""
    w = walk if from_end == 0 else list(reversed(walk))
    if len(w) < 2:
        return (0.0, 0.0), None
    p0 = np.array(w[0], dtype=float)
    p1 = np.array(w[min(k, len(w) - 1)], dtype=float)
    v = p1 - p0
    n = float(np.linalg.norm(v))
    return ((float(v[0] / n), float(v[1] / n)) if n > 0 else (0.0, 0.0)), w


# ---------------------------------------------------------------- fitting

def _cross2(a, b) -> float:
    """二维叉积的 z 分量：a.x*b.y - a.y*b.x。

    ★ 不能用 `np.cross`：numpy 自 2.0 起对 2 元素向量弃用（DeprecationWarning），
    到评测容器里的 **numpy 2.5.3** 直接抛
      `ValueError: Both input arrays must be (arrays of) 3-dimensional vectors,
       but they are 2 and 2 dimensional instead.`
    第六次提交（第一次真正跑起来）就死在这一行上 —— `rdp` 是骨架拟合的必经之路，
    阶段一 10 张图全 FAIL、耗时 31.4 s、导出 0 份。手算与旧版 `np.cross` 完全等价。
    """
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def rdp(points: Sequence[Tuple[float, float]], eps: float = 1.6) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    p0 = np.asarray(points[0], dtype=float)
    p1 = np.asarray(points[-1], dtype=float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-9:
        dists = [float(np.hypot(*(np.asarray(p, dtype=float) - p0))) for p in points]
    else:
        dists = [abs(_cross2(d, np.asarray(p, dtype=float) - p0)) / L for p in points]
    i = int(np.argmax(dists))
    if dists[i] <= eps:
        return [points[0], points[-1]]
    return rdp(points[:i + 1], eps)[:-1] + rdp(points[i:], eps)


_SNAP_TARGETS = (0.0, 45.0, 90.0, 135.0)


def snap_polyline(poly: Sequence[Tuple[float, float]], tol_deg: float = 9.0):
    """Snap segments to the 0/45/90 lattice (97.1% of 8439 GT edges are exactly axis aligned)."""
    if len(poly) < 2:
        return [tuple(map(float, p)) for p in poly]
    out = [(float(poly[0][0]), float(poly[0][1]))]
    for b in poly[1:]:
        ax, ay = out[-1]
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        tgt = min(_SNAP_TARGETS, key=lambda t: min(abs(t - ang), 180 - abs(t - ang)))
        if min(abs(tgt - ang), 180 - abs(tgt - ang)) > tol_deg:
            out.append((bx, by)); continue
        if tgt == 0.0:
            out.append((bx, ay))
        elif tgt == 90.0:
            out.append((ax, by))
        elif tgt == 45.0:
            m = (abs(dx) + abs(dy)) / 2.0
            out.append((ax + math.copysign(m, dx), ay + math.copysign(m, dy)))
        else:
            m = (abs(dx) + abs(dy)) / 2.0
            out.append((ax + math.copysign(m, dx), ay - math.copysign(m, dy)))
    return out


def split_segments(poly, min_len: float = 2.0):
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
            out.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return out


# ---------------------------------------------------------------- extractor

@dataclass
class SkeletonResult:
    skel: np.ndarray
    thin: np.ndarray
    ink: np.ndarray
    polylines: List[List[Tuple[float, float]]] = field(default_factory=list)
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    endpoints: List[Tuple[float, float]] = field(default_factory=list)
    junctions: List[Tuple[float, float]] = field(default_factory=list)
    n_nodes: int = 0
    n_branch_paths: int = 0
    has_chroma: bool = False
    n_gap_bridges: int = 0


class WireExtractor:
    def __init__(self, ink_mode: str = 'adaptive', thin_w: float = 1.6,
                 use_colour: bool = True, rdp_eps: float = 1.6, snap: bool = True,
                 min_branch_len: float = 3.0, close_ink: bool = False,
                 close_thin: bool = False, dilate_to: int = 0, adap_c: int = 8,
                 adap_k: int = 31, scale: float = 1.0, max_thin_iter: int = 200,
                 colour_edge: bool | None = None):
        self.ink_mode = ink_mode
        self.thin_w = thin_w
        self.use_colour = use_colour
        # ★ 彩色边缘证据通道（env `STAGE1_WIRE_EDGE`，默认关；见 `foreground` 里的长注释）
        if colour_edge is None:
            colour_edge = os.environ.get("STAGE1_WIRE_EDGE", "0").strip().lower() in (
                "1", "true", "on", "yes")
        self.colour_edge = bool(colour_edge)
        self.last_edge_thr = 0.0
        self.last_edge_add = 0
        self.rdp_eps = rdp_eps
        self.snap = snap
        self.min_branch_len = min_branch_len
        self.close_ink = close_ink
        self.close_thin = close_thin
        self.dilate_to = dilate_to
        self.adap_c = adap_c
        self.adap_k = adap_k
        self.scale = scale
        self.max_thin_iter = max_thin_iter
        self.last_colour_drop = 0
        self.last_colour_keep = 0
        # ★ 定向接线（`STAGE1_WIRE_GAPBRIDGE`，默认 0 = 关）。见 `bridge_skeleton_gaps`
        #   上面的长注释：补交叉处的洞，但只补「端点相向 + 缺口处有横穿骨架」的位置。
        _on = ("1", "true", "on", "yes")
        self.gap_bridge = os.environ.get("STAGE1_WIRE_GAPBRIDGE", "0").strip().lower() in _on
        try:
            self.gap_bridge_max = float(os.environ.get("STAGE1_WIRE_GAPBRIDGE_MAX", "12.0") or 12.0)
        except ValueError:
            self.gap_bridge_max = 12.0
        try:
            self.gap_bridge_ang = float(os.environ.get("STAGE1_WIRE_GAPBRIDGE_ANG", "35.0") or 35.0)
        except ValueError:
            self.gap_bridge_ang = 35.0
        self.gap_bridge_cross = os.environ.get(
            "STAGE1_WIRE_GAPBRIDGE_CROSS", "1").strip().lower() in _on
        self.last_gap_bridges = 0
        # ★ 十字芯定向补全（`STAGE1_WIRE_CROSS_FILL`，**默认 1 = 开**，2026-10-11 §5.108）。
        #   见 `cross_core_fill` 的 docstring：只补「两个轴向都夹着细笔画」的胖像素（真十字
        #   芯），补回后交叉处重新成为真结点。交付口径官方 10 张 84.13 → **84.38（+0.25）**
        #   （元件/引脚两通道逐位不变），池子 10 张 18.69 → 18.82（+0.13）；`_R` 默认 8.0
        #   （r=6 与基线打平 ⇒ 阈在 6~8 之间，诚实记录）。
        self.cross_fill = os.environ.get("STAGE1_WIRE_CROSS_FILL", "1").strip().lower() in _on
        try:
            self.cross_fill_r = float(os.environ.get("STAGE1_WIRE_CROSS_FILL_R", "8.0") or 8.0)
        except ValueError:
            self.cross_fill_r = 8.0
        try:
            self.cross_fill_axes = int(os.environ.get("STAGE1_WIRE_CROSS_FILL_AXES", "2") or 2)
        except ValueError:
            self.cross_fill_axes = 2

    def foreground(self, bgr: np.ndarray):
        if self.scale and self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_CUBIC)
        ink = ink_mask(bgr, self.ink_mode, k=self.adap_k, c=self.adap_c)
        if self.close_ink:
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        thin, dt = thin_stroke_mask(bgr, ink, self.thin_w)
        has_chroma = False
        if self.use_colour:
            cm = wire_colour_mask(bgr, ink)
            if cm is not None:
                has_chroma = True
                # ★ 2026-10-10（用户 m40034：「你还要用其他手段交叉验证才能得出这是导线」）：
                #   色相门以前是**一票否决**（`thin &= cm`）——主色相猜错就整张图的导线全灭
                #   （见 `wire_colour_mask` docstring 里 case 0001 的 6px 事故）。颜色只是
                #   **一条证据**，不该有否决权。`STAGE1_WIRE_UNCONF` 提供三种口径：
                #     veto（默认，旧行为）/ keep（颜色不定，全部留）/ near（只留离已确认
                #     像素 ≤`STAGE1_WIRE_UNCONF_NEAR`(3)px 的未确认像素 —— 靠**连通性**交叉验证）
                pol = os.environ.get("STAGE1_WIRE_UNCONF", "veto").strip().lower()
                if pol not in ("veto", "keep", "near"):
                    pol = "veto"
                conf = cv2.bitwise_and(thin, cm)
                unc = cv2.bitwise_and(thin, cv2.bitwise_not(cm))
                n_unc = int((unc > 0).sum())
                if pol == "veto":
                    self.last_colour_drop = n_unc
                    thin = conf
                else:
                    if pol == "keep":
                        keep = unc
                    else:
                        try:
                            _unr = int(os.environ.get("STAGE1_WIRE_UNCONF_NEAR", "3"))
                        except ValueError:
                            _unr = 3
                        keep = unc if _unr <= 0 else cv2.bitwise_and(
                            unc, cv2.dilate(conf, np.ones((2 * _unr + 1,) * 2, np.uint8)))
                    n_keep = int((keep > 0).sum())
                    self.last_colour_drop = n_unc - n_keep
                    self.last_colour_keep = n_keep
                    thin = cv2.bitwise_or(conf, keep)
        # ★ 2026-10-10（用户 m40034：「你不能指望灰度……我一直以为你是通过检测像素的颜色
        #   跳变来判断的」）：加一条**彩色边缘证据**通道 —— 灰度自适应阈值看不见「等亮度但
        #   不同色」的边界，彩色梯度（Di Zenzo）看得见。它是**并集**（只加不减），所以
        #   灰度路径已经找到的像素一个都不会丢；门槛是相对本图的（阈值随图自适应）。
        #   与 `use_colour` 独立：`use_colour=0` 时这条通道仍然可用（它不依赖色相假设）。
        if self.colour_edge:
            try:
                _er = float(os.environ.get("STAGE1_WIRE_EDGE_REL", "0.55"))
            except ValueError:
                _er = 0.55
            try:
                _em = float(os.environ.get("STAGE1_WIRE_EDGE_MIN", "10.0"))
            except ValueError:
                _em = 10.0
            ce, self.last_edge_thr = colour_edge_mask(
                bgr, ink, self.thin_w, rel=_er, abs_min=_em)
            # ★ 关键约束（2026-10-10 首测教训）：彩色边缘**只能补「灰度墨迹附近的像素」**。
            #   首测（`work\tools\dev\_wireedge.py`，官方 case 1/2）把整张图的彩色边缘都并进来，
            #   段覆盖 R 从 0.9225 掉到 0.7887 —— 原因不是「多了真导线」，而是 `occlude_boxes`
            #   用四角中位色涂元件框，**涂色与局部背景有差** ⇒ 框边界本身成了强彩色边缘，
            #   凭空长出骨架分支，把真导线的分段打碎（判据要求整条 GT 边的采样点落在同一段上）。
            #   所以默认 `STAGE1_WIRE_EDGE_NEAR=2`：只保留离灰度墨迹 ≤2px 的彩色边缘像素，
            #   「补空心/连断点」而不「发明结构」。0 = 关闭该限制（首测那种裸并集）。
            try:
                _nr = int(os.environ.get("STAGE1_WIRE_EDGE_NEAR", "2"))
            except ValueError:
                _nr = 2
            if _nr > 0:
                _k = np.ones((2 * _nr + 1, 2 * _nr + 1), np.uint8)
                ce = cv2.bitwise_and(ce, cv2.dilate(ink, _k))
            n_add = int(((ce > 0) & (thin == 0)).sum())
            self.last_edge_add = n_add
            thin = cv2.bitwise_or(thin, ce)
        if self.close_thin:
            # repair the 1px gaps thinning leaves at stroke crossings --
            # a closing keeps a pixel only when its whole 3x3 neighbourhood is set
            thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # `STAGE1_WIRE_CLOSE_K`：3x3 只补得动 1px 缝，而**交叉处的洞是 5~10px** ——
        # `thin_stroke_mask` 在十字芯处 dt 大（那里墨迹最宽），整块被当粗线剔掉，
        # 于是每条线在交叉处断开、骨架碎成 5~10px 的片（现场 case 1 x≈843：
        # GT 是 13 条 29~30px 首尾相接的边，我方只出 9 条、缺 [115,144]/[173,203]/
        # [380,410]/[438,468]）。用可调核把交叉芯补回来，十字复原成真结点，
        # `STAGE1_SPLIT_JUNCTIONS`/`_JUNC_NODES` 才有结点可切。默认 0 = 不动（= 旧行为）。
        try:
            _ck = int(os.environ.get("STAGE1_WIRE_CLOSE_K", "3" if self.close_thin else "0") or 0)
        except ValueError:
            _ck = 0
        if _ck >= 3:
            _ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ck, _ck))
            thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, _ker)
        # `STAGE1_WIRE_CROSS_FILL`：闭运算是无差别膨胀-腐蚀（粗线芯、元件框边也会被补，
        # 池子回归 −0.47），这里改成**有判据的**十字芯补全：只有当某胖像素在**两个轴向**
        # 上都被细笔画夹住（= 两条细线真的相交）才补。
        if self.cross_fill:
            thin = cross_core_fill(thin, dt, self.thin_w, r=self.cross_fill_r,
                                   min_axes=self.cross_fill_axes)
        if self.dilate_to:
            # conservative "shell" tolerance: a pixel that has a thin-stroke pixel nearby is
            # kept as a wire pixel too, which also repairs the gaps
            near = cv2.dilate(thin, np.ones((self.dilate_to * 2 + 1,) * 2, np.uint8))
            thin = cv2.bitwise_and(near, ink)
        return thin, ink, has_chroma

    def extract(self, bgr: np.ndarray) -> SkeletonResult:
        thin, ink, has_chroma = self.foreground(bgr)
        skel = zhang_suen(thin, max_iter=self.max_thin_iter)
        res = SkeletonResult(skel=skel, thin=thin, ink=ink, has_chroma=has_chroma)
        if self.gap_bridge:
            skel, _nb, _brs = bridge_skeleton_gaps(
                skel, max_gap=self.gap_bridge_max, max_angle_deg=self.gap_bridge_ang,
                require_cross=self.gap_bridge_cross)
            res.skel = skel
            res.n_gap_bridges = int(_nb)
            self.last_gap_bridges = int(_nb)
        self._trace(skel, res)
        return res

    # ---------------------------------------------------------- internals
    def _trace(self, skel: np.ndarray, res: SkeletonResult):
        pts, deg, node_px, raw = build_graph(skel)

        # --- collapse adjacent non-degree-2 pixels into one node
        node_list = sorted(node_px)
        pos = {p: i for i, p in enumerate(node_list)}
        parent = list(range(len(node_list)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        # 注意：node_list 在这个循环里不变，集合必须提到循环外。
        # 原先写成 neighbours(set(node_list), p) —— 每次迭代都重建一个 |node_list|
        # 元素的集合，整体代价 O(n^2)。在 0107 上（39,605 个结点像素）这一行独占
        # 92.9 s / 104 s，提前后降到 0.14 s，输出完全一致。
        nset = set(node_list)
        for p in node_list:
            for q in neighbours(nset, p):
                if q in pos:
                    ra, rb = find(pos[p]), find(pos[q])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        clusters: Dict[int, List[Tuple[int, int]]] = {}
        for p in node_list:
            clusters.setdefault(find(pos[p]), []).append(p)
        node_of_px, nodes = {}, []
        for members in clusters.values():
            n = Node(idx=len(nodes), x=float(np.mean([m[1] for m in members])),
                     y=float(np.mean([m[0] for m in members])))
            nodes.append(n)
            for m in members:
                node_of_px[m] = n.idx

        # --- paths
        paths: List[Path] = []
        half: List[HalfEdge] = []
        for walk in raw:
            if len(walk) < 2 or walk[0] not in node_of_px or walk[-1] not in node_of_px:
                continue
            xy = [(float(c), float(r)) for r, c in walk]
            L = math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
            p = Path(idx=len(paths), n0=node_of_px[walk[0]], n1=node_of_px[walk[-1]],
                     pts=xy, length=L)
            paths.append(p)
        res.n_branch_paths = len(paths)

        keep = [p for p in paths if p.length >= self.min_branch_len]
        for p in keep:
            he0 = HalfEdge(path=p.idx, end=0, node=p.n0)
            he1 = HalfEdge(path=p.idx, end=1, node=p.n1)
            he0.partner, he1.partner = len(half) + 1, len(half)
            half.extend([he0, he1])

        # --- node slot directions
        node_slots: Dict[int, List[int]] = {}
        for he in half:
            node_slots.setdefault(he.node, []).append(len([h for h in half[:0]]))
        node_slots = {}
        for hi, he in enumerate(half):
            node_slots.setdefault(he.node, []).append(hi)
        for nid, hs in node_slots.items():
            n = nodes[nid]
            for hi in hs:
                he = half[hi]
                walk = [(int(y), int(x)) for x, y in paths[he.path].pts]
                d, _ = _dir_at(walk, he.end)
                n.slots.append(hi)
                n.dirs.append(d)

        # --- pair branches at every node by "straight through" preference
        for nid, hs in node_slots.items():
            n = nodes[nid]
            k = len(hs)
            if k == 0:
                n.kind = 'E'
                continue
            if k == 1:
                n.kind = 'E'
                continue
            if k == 2:
                n.kind = 'C'
                n.pair[0] = 1
                n.pair[1] = 0
                continue
            cost = np.full((k, k), 1e6, np.float64)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    vi, vj = n.dirs[i], n.dirs[j]
                    cosv = -(vi[0] * vj[0] + vi[1] * vj[1])
                    cost[i, j] = (1.0 - cosv) * 100.0
            if k % 2 == 0:
                try:
                    from scipy.optimize import linear_sum_assignment
                    r, c = linear_sum_assignment(cost)
                except Exception:
                    r, c = np.arange(k), np.arange(k)
                for i, j in zip(r, c):
                    n.pair[int(i)] = int(j)
            else:
                big = np.full((k, k + 1), 1e6, np.float64)
                big[:, :k] = cost
                for i in range(k):
                    big[i, k] = 56.0            # cost of leaving branch i unpaired
                try:
                    from scipy.optimize import linear_sum_assignment
                    r, c = linear_sum_assignment(big)
                except Exception:
                    r, c = np.arange(k), np.arange(k)
                for i, j in zip(r, c):
                    if j < k:
                        n.pair[int(i)] = int(j)
            n.kind = 'T' if k == 3 else 'X'

        # --- trace maximal polylines by jumping across the pairings
        consumed = set()
        polylines: List[List[Tuple[float, float]]] = []
        for he in half:
            if he.partner in consumed:
                continue
            # start at `he`, i.e. enter the path at its `he.end` side
            chain: List[Tuple[float, float]] = []
            cur = he
            guard = 0
            while guard < 10000:
                guard += 1
                if cur.partner in consumed:
                    break
                consumed.add(cur.partner)
                consumed.add(half[cur.partner].partner if half[cur.partner].partner >= 0 else -1)
                p = paths[cur.path]
                # `cur.end` 是我们进入这条 path 的那一端（见 _dir_at 的约定：
                # from_end=0 -> 从 walk[0] 出发）。所以 end==0 时要按原始顺序走。
                # 2026-02 修：这里原先写成 `if cur.end == 1`，方向正好反了 ——
                # 每进入一条 path 都会先瞬移到它的**另一端**再倒着走回来，于是
                # 每条 chain 在每个结点处都产生一个等于该 path 长度的跳变
                # （0101 单张 >30px 的跳变 131 次）。轴对齐的跳变表现为凭空多出
                # 的图纸边框长线（总长比 2.5×），斜向的跳变又被 snap_polyline
                # 吸附成精确 45°，伪装成"大尺度斜线"并躲过 max_axis_dev 过滤。
                seg = p.pts if cur.end == 0 else list(reversed(p.pts))
                chain.extend(seg if not chain else seg[1:])
                exit_he = half[cur.partner]
                nid = exit_he.node
                n = nodes[nid]
                # which slot position does exit_he occupy at this node?
                slot_pos = None
                for si, hi in enumerate(n.slots):
                    if hi == cur.partner:
                        slot_pos = si
                        break
                if slot_pos is None:
                    break
                nxt_pos = n.pair.get(slot_pos)
                if nxt_pos is None:
                    break
                nxt_hi = n.slots[nxt_pos]
                if half[nxt_hi] is cur:
                    break
                if consumed and nxt_hi in consumed:
                    break
                cur = half[nxt_hi]
                if cur.partner in consumed:
                    break
                # continue
                consumed.add(nxt_hi)
            if len(chain) >= 2:
                polylines.append(chain)

        res.polylines = polylines
        for chain in polylines:
            poly = rdp(chain, self.rdp_eps)
            if self.snap:
                poly = snap_polyline(poly)
            res.segments.extend(split_segments(poly))
        # ★ 2026-07 审计项⑤：`half` 是**半边表** —— 一个 k 度结点挂着 k 条半边，
        #   所以原来那两条按半边列表推导的写法会把同一个结点坐标在 `junctions`
        #   里**重复 k 次**（`endpoints` 因为条件是 `len(slots) <= 1`，最多命中一次，
        #   不受影响）。目前 `stage1\wires.py:447` 是用 `_rebuild_nodes(segments)`
        #   **重算** endpoints/junctions 的，`pskel` 这两个字段并没有被下游读，
        #   所以这个重复不影响得分；但它会误导任何直接看 `pskel` 输出的人，按结点去重。
        _ep, _jn = [], []
        _seen_ep, _seen_jn = set(), set()
        for hi in half:
            nd = nodes[hi.node]
            if len(nd.slots) <= 1:
                if hi.node not in _seen_ep:
                    _seen_ep.add(hi.node)
                    _ep.append((nd.x, nd.y))
            elif len(nd.slots) >= 3 and hi.node not in _seen_jn:
                _seen_jn.add(hi.node)
                _jn.append((nd.x, nd.y))
        res.endpoints = _ep
        res.junctions = _jn
        res.n_nodes = len(nodes)
        return res


# ---------------------------------------------------------------- net grouping

def group_segments(segs, agg: float = 2.6):
    """Union-find two-point segments into connected wires by endpoint adjacency."""
    n = len(segs)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ends = [(np.array(a, float), np.array(b, float)) for a, b in segs]
    cell: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(ends):
        for p in (a, b):
            k = (int(p[0] // agg), int(p[1] // agg))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cell.get((k[0] + dx, k[1] + dy), []):
                        if min(float(np.hypot(*(ends[j][0] - p))),
                               float(np.hypot(*(ends[j][1] - p)))) <= agg:
                            ra, rb = find(i), find(j)
                            if ra != rb:
                                parent[max(ra, rb)] = min(ra, rb)
            cell.setdefault(k, []).append(i)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[segs[i] for i in g] for g in groups.values()]
