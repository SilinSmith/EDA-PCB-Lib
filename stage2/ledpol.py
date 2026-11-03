"""LED/二极管（`d`/`led`）的**极性**读数：图上横杠在哪一端 = 哪一端是 K（pin_1）。

背景（`work/_ledpool.py` + `work/_ledrun.py` 的普查）：官方 GT 里 d/led 的 `pin_1`
脚名恒为 `"K"`、`pin_2` 恒为 `"A"`，而**画法四种朝向都有**（220 池：竖放 pin_1 在
下端 121 / 在上端 102，横放 pin_1 在右端 109 / 在左端 51）。`assemble.stamp_pins`
原来按「`d`/`led` 的 pin_1 恒在 +SPAN 一侧」发号（`PIN_STAMP_POS`），于是竖放时有一
半的件号是反的；「按轴选符号」的多数票规则在官方 10 张上
是 **80.88 → 80.81（−0.07）**（3/D1 判对、7/LED1 与 10/LED1 判反）⇒ 必须**看图**。

判据（v3，本模块）：元件笔画是暗红（`red_mask`），LED 符号 = 三角形 + 横杠（K 端）
+ 两根斜向发光箭头。按「沿轴每一行的**最长连续红墨游程**」取剖面：
  * 横笔（三角形底边 ~24px、横杠 ~34px）→ 两个长游程峰；斜箭头/斜边只有 5~10px；
  * 取这两条横笔，再在**它们之间**（那里没有箭头）量「斜边跨度」：靠底边宽、
    靠顶点(横杠)窄 ⇒ **窄的那一侧是 K**；要求 `L≥8`、峰游程 `≥15`、峰间距 `≥6`、
    两侧跨度差 `≥2`，否则返回 `None`（不判）。

读数（`work/_ledrun.py`、`work/_ledpoolsim.py` 与离线 oracle `work/_ledoracle.py`）：
  * 官方 10 张：**判对 7 / 判错 0 / 不判 1**（case 6 LED2）；
  * 220 池 383 件（`work/_ledpoolsim.py`，输入就是流水线自己的两个图章端点、
    真值取 GT 的 K 脚点）：现行几何默认只对 **211/383 = 55.1%**；本判据判出 179 件、
    换向 80 次（**真该换 63 / 误换 17**，精度 78.8%）⇒ **池上净收益 +46 件**；
  * 离线重排（把 pin_1/pin_2 换到横杠侧，含网成员号同步）官方 10 张总分
    **80.88 → 80.98（+0.10）**：case 3 `D1` netH 0.6788→0.7074（+2 成员），
    case 10 `LED1` 与 case 7 `LED1` 的**两脚本来就没进任何网**（换向无变化）。
  * 试过「第二信号闸门」（用「两条横笔里游程更长的那条是横杠」做交叉校验，两信号
    一致就判不出）：换向精度升到 23/24 = 95.8%，但判出件数腰斩 ⇒ 池上净收益只剩
    **+22 件**（不到一半）⇒ **不采用**，本模块只留跨度信号。
本模块常开（见 `assemble.build_doc`）。
"""
from typing import Optional, Sequence, Tuple

import numpy as np


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """暗红笔画（元件体/引线）掩膜：亮红（导线）与蓝（文字）都不算。"""
    b, g, r = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    m = (r > g + 25) & (r > b + 25)
    m |= (r > 90) & (r > g + 12) & (r > b + 12)
    return m


def _runs(row) -> int:
    """一行布尔序列里最长的连续 True 游程。"""
    best = cur = 0
    for v in row:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _extent(mask: np.ndarray, cx: float, cy: float,
            ux: float, uy: float, band: float) -> float:
    """在 (cx,cy) 处沿 u 采样、垂直方向 ±band 内红墨的跨度（像素）。"""
    vx, vy = -uy, ux
    lo = hi = None
    n = int(band)
    h, w = mask.shape[:2]
    for k in range(-n, n + 1):
        px, py = int(round(cx + vx * k)), int(round(cy + vy * k))
        if 0 <= py < h and 0 <= px < w and mask[py, px]:
            if lo is None:
                lo = k
            hi = k
    if lo is None:
        return 0.0
    return float(hi - lo + 1)


def profile(mask: np.ndarray, pk: Sequence[float], pa: Sequence[float], H: float,
            n: int = 100, band: float = 34.0):
    """沿 K→A 轴采样，返回 `(L, [(t, 行内最长红墨游程)])`；`pk`/`pa` 是 json 坐标。"""
    a = (float(pk[0]), float(H) - float(pk[1]))
    b = (float(pa[0]), float(H) - float(pa[1]))
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L < 8.0:
        return None
    ux, uy = dx / L, dy / L
    vx, vy = -uy, ux
    h, w = mask.shape[:2]
    out = []
    nb = int(band)
    for i in range(n + 1):
        t = (i / n) * L
        cx, cy = a[0] + ux * t, a[1] + uy * t
        vals = []
        for k in range(-nb, nb + 1):
            px, py = int(round(cx + vx * k)), int(round(cy + vy * k))
            if 0 <= py < h and 0 <= px < w:
                vals.append(bool(mask[py, px]))
        out.append((t, _runs(vals)))
    return L, out


def k_end(mask: np.ndarray, pk: Sequence[float], pa: Sequence[float], H: float,
          n: int = 100, band: float = 34.0,
          verbose: bool = False) -> Tuple[Optional[str], object]:
    """`pk`/`pa` = 两个候选端点的 json 坐标（`pk` 是当前认为 K 的那端）。

    返回 `('K' | 'A' | None, profile)`：`'A'` 表示横杠在 `pa` 那端（应当换向）。
    """
    r = profile(mask, pk, pa, H, n=n, band=band)
    if r is None:
        return None, None
    L, prof = r
    mx = max(e for _, e in prof)
    if mx < 15.0:
        return None, prof
    wide = [(t, e) for t, e in prof if e >= 0.55 * mx]
    groups, cur = [], [wide[0]]
    for t, e in wide[1:]:
        if t - cur[-1][0] <= 4.0:
            cur.append((t, e))
        else:
            groups.append(cur)
            cur = [(t, e)]
    groups.append(cur)
    peaks = sorted([max(gp, key=lambda z: z[1]) for gp in groups], key=lambda z: -z[1])
    if len(peaks) < 2:
        return None, prof
    (t_a, e_a), (t_b, e_b) = sorted(peaks[:2], key=lambda z: z[0])
    dd = t_b - t_a
    if dd < 6.0:
        return None, prof
    bnd = 0.75 * max(e_a, e_b) + 6.0
    a = (float(pk[0]), float(H) - float(pk[1]))
    ux, uy = (float(pa[0]) - a[0]) / L, ((float(H) - float(pa[1])) - a[1]) / L
    e1 = _extent(mask, a[0] + ux * (t_a + 0.3 * dd), a[1] + uy * (t_a + 0.3 * dd), ux, uy, bnd)
    e2 = _extent(mask, a[0] + ux * (t_b - 0.3 * dd), a[1] + uy * (t_b - 0.3 * dd), ux, uy, bnd)
    if verbose:
        print("      [ledpol] 横笔 %s 跨度 靠a=%s 靠b=%s 带宽%.0f L=%.0f"
              % ([(round(t, 1), int(e)) for t, e in ((t_a, e_a), (t_b, e_b))], e1, e2, bnd, L))
    if abs(e1 - e2) < 2.0:
        return None, prof
    # 靠底边宽、靠顶点(横杠)窄 ⇒ 窄的那一侧是 K。
    return ("K" if e1 < e2 else "A"), prof
