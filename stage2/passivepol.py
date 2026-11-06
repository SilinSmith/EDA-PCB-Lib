"""电容（`c`）两脚 1/2 号朝向：靠**引线不对称**读图定向（`submission/run.py` 导出后整形）。

背景（`docs/RESULTS.md` §5.600）。官方 13 页 GT 里 `r/c/l` 的 `pin_1` 多数落在
「坐标小的一侧」（`r` 54/61、`c` 59/80），`stage2/assemble.py` 的 `PIN_STAMP` −SPAN
规则已经吃满这个多数口径（我方一致率 `r` 52/59、`c` 59/80、`l` 10/10）。剩下 21 件
翻面电容在图上**没有极性标记**（两块平行板旋转 180° 同形），但有一个**几像素级的手性**：
把元件框内的暗红笔画沿长轴做行剖面，两条极板到框上/下边的距离并不相等 ——
翻面件的下引线平均比上引线长约 3 px。

判据：`d = 下引线 - 上引线`（沿长轴，像素），`d > 1.0` ⇒ 该件翻面（GT 的 `pin_1`
在 +SPAN 一侧）⇒ 把 `pin_1`/`pin_2` 的点位与网表引用一起对调。

为什么放在导出后而不是 `stamp_pins` 里：判据要量的是**整根引线**的长度（GT 的 `c`
框 34×42 就是「极板 + 两段引线」的外接框），而 `stamp_pins` 拿到的贴章框只有 16×20
（只盖住极板区），那一刻量出来的 d 恒在 ±1 内、没有判别力；导出时的 `bbox` 才是
与 GT 同一套外接框。

读数（`work/_caprule_sim.py`，13 页、判题机那把尺子 = 不折叠非 box 脚号）：
  * 阈值 0.5~2.0 是平台期，翻转 8 件 ⇒ **84.344 → 84.903（+0.560）**：
    case1 +2.81（netH .552→.621）、case9 +5.14（.617→.695）、case6 −0.66；
  * 同一规则套到 `r` 上负收益（+0.478 < +0.560，多翻那件是错的）⇒ 只对 `c` 生效；
  * 本地折叠口径（`work/_cv.py` 的 `CFG`）恒为 +0.000 —— 折叠抹平非 box 脚号，
    看不见这笔账，**必须看 `judge mean`**；
  * 不打判据、整页全翻是 21 改对 / 59 改错（净 −38 件）⇒ 只能逐件看图。
"""
from typing import Optional, Sequence

import numpy as np


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """元件笔画的暗红掩膜（与 `work/_caprule_sim.py` 同一套阈值）。"""
    return (rgb[:, :, 0] > 120) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 90)


def lead_diff(mask: np.ndarray, bbox_img: Sequence[float], tall: bool = True,
              frac: float = 0.62) -> Optional[float]:
    """`bbox_img` 是图像坐标（y 向下）的外接框；返回「下引线 - 上引线」。"""
    if mask is None or len(bbox_img) != 4:
        return None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_img]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    blk = mask[y1:y2 + 1, x1:x2 + 1]
    if blk.size == 0:
        return None
    prof = blk.sum(axis=1) if tall else blk.sum(axis=0)
    if len(prof) < 10 or prof.max() == 0:
        return None
    plate = np.where(prof >= frac * prof.max())[0]
    if len(plate) == 0:
        return None
    lead_lo = float(plate[0])
    lead_hi = float(len(prof) - 1 - plate[-1])
    return lead_hi - lead_lo


def flip_cap(mask: np.ndarray, bbox_img: Sequence[float], tall: bool,
             min_d: float = 1.0) -> bool:
    """翻面电容（`pin_1` 在 +SPAN 一侧）返回 True。"""
    d = lead_diff(mask, bbox_img, tall)
    return d is not None and d > min_d
