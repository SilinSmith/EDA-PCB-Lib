# -*- coding: utf-8 -*-
"""第二阶段 · **沿阅读方向再框几个字**（用户 m06734）。

用户 m06734 的原话：「不可分就默认是长的，让他尝试往字的自然阅读方向看看能不能再框字
进去，但是截止条件定的高一点。」

要解决的问题：`work\\reports\\错例集_导线与元件.md` 里最大的那类错（7645 条
`component.text_mismatch`）全是**长器件名被切碎**——GT `RP2350B` 读成 `GP/2`、
`APS6404L-3SQR-ZR` 读成 `C1H+`。切分器给出的框只盖住了名字中间的一小段，
`expect.merge_runs` 那种「拼相邻的**检测框**」救不了它（因为旁边根本没有第二个框），
必须**直接看像素**：沿同一行往左右把相邻的笔画一并框进来，再让 CRNN 重读。

三个「高」截止条件（缺一不可，宁可不长也不要把别的图形框进来）：

1. **空列预算** `gap_mult × 行高` —— 超过就停（不许跨过词间空白去够远处的东西）。
2. **带外有墨就停** —— 这一列在「本行行带」之外还有墨，说明那是一根导线/一个图形符号，
   绝不允许被框进来。这是最重要的一条：`_ocr_pad.py` 里那个朴素的 `grow` 臂只有
   0.0753 的 exact，就是因为它顺着墨一路长到符号和邻行上去了。
3. **生长上限** `max_grow_frac × 原框宽`。

都不依赖 OCR。用法：`grow_reading(bgr, box) -> Optional[box]`。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 行带上下各削掉一点（防止反锯齿/相邻行的边缘被算成「本行有墨」）
BAND_FRAC = 0.10
# 空列预算 = gap_mult × 行高
GAP_MULT = 0.35
# 生长上限 = max_grow_frac × 原框宽
MAX_GROW_FRAC = 1.6
# 带外允许的零星墨点（噪声/反锯齿），超过就认定是结构线
OUTSIDE_TOL = 2
# 一根竖列的墨量超过 `OUTSIDE_MULT × 行高` 就认定是导线/符号的竖边（一行字的笔画远达不到）
OUTSIDE_MULT = 2.2
# 至少长这么多像素才算「长进去了」
MIN_NEW = 1.5
# 绝对上限，防止超宽框失控
MAX_PX = 420.0


def ink_mask(bgr: np.ndarray, thr: Optional[int] = None) -> np.ndarray:
    """墨 = 灰度 < 阈值。阈值默认 Otsu 并夹到 [110, 235]（与 `cvparts.textseg` 同口径）。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    if thr is None:
        t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(max(110, min(int(t), 235)))
    return g < thr


def _walk(ink: np.ndarray, band: Tuple[int, int], start: int, step: int,
          limit: int, gap: float, max_col_ink: float) -> int:
    """从 `start` 沿 `step` 方向走，返回**最后一个有墨的列**（不含方向那侧）。

    `start` 是框的边界列（向右走传 x2，向左走传 x1-1）。返回的是半开区间端点。

    停下的两个条件：整列墨量超过 `max_col_ink`（那是一根导线/一个图形符号的竖边，贯穿
    的高度远大于一行字），或者空列预算 `gap` 用完。
    """
    y1, y2 = band
    last = start
    x = start
    W = ink.shape[1]
    while (x - start) * step <= limit and 0 <= x < W:
        if int(ink[:, x].sum()) > max_col_ink:
            break                          # 贯穿式的竖线：导线/符号边框，停
        if int(ink[y1:y2, x].sum()):
            last = x + (1 if step > 0 else 0)
            x += step
            continue
        if abs(x - last) > gap:
            break                          # 空列预算用完
        x += step
    return last


def grow_reading(bgr: np.ndarray, box: Sequence[float], *,
                 thr: Optional[int] = None, band_frac: float = BAND_FRAC,
                 gap_mult: float = GAP_MULT, max_grow_frac: float = MAX_GROW_FRAC,
                 outside_mult: float = OUTSIDE_MULT, min_new: float = MIN_NEW,
                 max_px: float = MAX_PX) -> Optional[List[float]]:
    """把 `box` 沿同一行向左右扩到相邻笔画上；没长进就返回 `None`。

    返回的是 `[x1, y1, x2, y2]`（**只动 x，不动 y**）—— 因为横向的阅读方向是确定的，
    纵向扩会把上下两行也吃进来（`_ocr_pad.py` 的 `grow` 臂就是这么坏掉的）。
    """
    if bgr is None:
        return None
    ink = ink_mask(bgr, thr)
    H, W = ink.shape[:2]
    x1f, y1f, x2f, y2f = (float(v) for v in box[:4])
    x1 = max(0, min(W - 1, int(round(x1f))))
    x2 = max(x1 + 1, min(W, int(round(x2f))))
    y1 = max(0, min(H - 1, int(round(y1f))))
    y2 = max(y1 + 1, min(H, int(round(y2f))))
    h = y2 - y1
    pad = int(round(h * band_frac))
    band = (min(y2, y1 + pad), max(y1 + 1, y2 - pad))
    if band[1] <= band[0]:
        band = (y1, y2)

    gap = max(1.5, gap_mult * h)
    limit = int(min(max_px, max_grow_frac * (x2 - x1)))
    max_col_ink = outside_mult * h

    right = _walk(ink, band, x2, +1, limit, gap, max_col_ink)
    left = _walk(ink, band, x1 - 1, -1, limit, gap, max_col_ink)
    left = max(0, left)
    if (right - x2) < min_new and (x1 - left) < min_new:
        return None
    if right <= left:
        return None
    return [float(left), y1f, float(right), y2f]


# ---------------------------------------------------------------------------
def _canvas(w: int = 240, h: int = 60) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _digits(img, x0, y0, h, widths, gap=2) -> int:
    """在图上画一排「数字状」竖条，返回右边界。"""
    x = x0
    for w in widths:
        cv2.rectangle(img, (x, y0), (x + w, y0 + h), (0, 0, 0), -1)
        x += w + gap
    return x - gap


def _selftest() -> int:
    fails: List[str] = []

    def chk(c, m):
        if not c:
            fails.append(m)

    # ① 一行 5 个笔画，框只盖住前 2 个 -> 应该长到盖住 5 个
    img = _canvas()
    end = _digits(img, 40, 20, 20, [6, 4, 6, 6, 8])
    got = grow_reading(img, [38, 18, 62, 42])
    chk(got is not None and got[2] >= end - 1,
        "① 右侧应长到 %d，实际 %r" % (end, got))
    chk(got is not None and abs(got[0] - 38) < 6, "① 左侧不该乱动: %r" % (got,))

    # ② 带外有墨（一根竖导线）-> 不许跨过去
    img2 = _canvas()
    _digits(img2, 40, 20, 20, [6, 4])
    cv2.line(img2, (80, 0), (80, 59), (0, 0, 0), 2)          # 贯穿全高的导线
    _digits(img2, 100, 20, 20, [6, 6])                        # 导线另一侧还有字
    got2 = grow_reading(img2, [38, 18, 52, 42])
    chk(got2 is None or got2[2] <= 80, "② 不许跨过导线: %r" % (got2,))

    # ③ 远处还有字但间隙过大 -> 不许去够
    img3 = _canvas()
    _digits(img3, 40, 20, 20, [6, 4])
    _digits(img3, 140, 20, 20, [6, 6])
    got3 = grow_reading(img3, [38, 18, 52, 42])
    chk(got3 is None or got3[2] < 60, "③ 间隙 %dpx 不该去够: %r" % (140 - 52, got3))

    # ④ 右边什么都没有 -> None
    img4 = _canvas()
    _digits(img4, 40, 20, 20, [6, 4])
    chk(grow_reading(img4, [38, 18, 52, 42]) is None, "④ 没东西可长就该返回 None")

    # ⑤ 上下相邻行有字 -> y 不许动、也不许把邻行算进来
    img5 = _canvas(h=90)
    _digits(img5, 40, 20, 18, [6, 4, 6, 6])
    _digits(img5, 40, 52, 18, [6, 6, 6])
    got5 = grow_reading(img5, [38, 18, 52, 40])
    chk(got5 is not None and got5[1] == 18 and got5[3] == 40, "⑤ y 不许动: %r" % (got5,))
    chk(got5 is not None and got5[2] > 52, "⑤ 同一行应继续长: %r" % (got5,))

    # ⑥ 左侧补字（框只盖住后半截）
    img6 = _canvas()
    _digits(img6, 40, 20, 20, [6, 6, 6, 6])
    got6 = grow_reading(img6, [60, 18, 90, 42])
    chk(got6 is not None and got6[0] <= 42, "⑥ 左侧应补到 40: %r" % (got6,))

    print("自检 %d 项：%s" % (6, "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
