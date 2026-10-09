"""第一阶段：引脚列枚举（pin column enumeration）—— 把 IC 的**全部**引脚点出来。

## 为什么必须有这个模块（owner m25661 之后的新证据）

官方 `pin_f1` 的键是 `(元件键, 引脚号)`，占 **25%** 的权重（`score =
100*(0.30*comp + 0.25*pin + 0.35*netH + 0.10*netL)`，权重由 220 张 `_score.json`
无截距最小二乘精确解出、回代 case `0001` 逐位吻合）。而 220 张实测：

    pin tp 206 / fp 1649 / fn 9150  →  P 0.1111 R 0.0220 F1 0.0367  →  只拿到 0.92/25 分

拆开看（`work\\tools\\diag\\_pinloss.py`）：GT box 引脚 9356 个，
- **4779 个（51.1%）挂在「元件键没对上」的元件上**（键不对，官方连 fp 都不记）；
- 另 4577 个在键匹配上的元件里：tp 206 / **号不对 3556** / 名字错 620 / 位置错 195；
- 匹配上的 402 个 box 里，「一个引脚都没发」的是 **0** 个。

**⇒ 瓶颈不是「发不出引脚」，是「发得远不够多」。** 实例 `0143` 的 `U1`
（STM32F205RCT6，GT **64** 个脚）我们只发 **8** 个点：`19,22,28,40,43,69,77,85`。

## 而号和名其实画在图上（实证）

`work\\preview\\_jlcbox.png`（`0143` 的 U1，4× 放大）看得清清楚楚：
**框外贴着边印 `1..64`**（左侧 1→32 从上往下、右侧 64→33 从下往上），
**框内印引脚名**（`VBAT` / `PC13/TRC_AF1` / …）。

`pins.pins_from_touch` 的主判据是「**导线**触到框边」，所以**没接线的引脚一个都看不见**。

## ★ 关键实测：引脚是**等距格点**（`_pincolprobe.py`）

`0143` 的 `U1`：`bbox_img = (140.6, 45.3, 289.1, 375.1)`，
GT 的 64 个脚 **L 边 32 个 y=55,65,…,365、R 边 32 个 y=55,65,…,365** ——
**间距精确 10.0px，两端各留一个 pitch**（55 = y0+9.7、365 = y1−10.1）。
引脚点落在框外 **~10.6px**（GT x=130.0 / 300.0，框边 140.6 / 289.1）。

于是本模块不再「逐行找墨迹桩」（实测噪声大：L 边 21 个原始桩里混着
6.5/7.0/3.0 的假间距、R 边更差，是 4.5/5.5 交替的假谐波），而是：

1. 收集**两种**沿边证据 —— ①去掉文字后的墨迹桩（`_stub_positions`，判据是
   **从框边往外连续 `CONNECT` 像素都是墨**，✕ 这类孤立斑点自然排除）；
   ②**贴着框边的文字框**（`_num_candidates`，引脚号本身就是被检出的文字框，
   `0143` 的 L 边实测 `cx=130.0 6x8`、R 边 `cx=297.0 8x8`，正好压在 GT 的 x 上）；
2. **只用号码文字**拟合 pitch（`_fit_pitch`，内点数不足 `NUM_MIN_EL` 就整面作废；
   退到墨迹桩的旧行为由 `STAGE1_PINC_FALLBACK=1` 显式打开）；
3. 在**证据跨度内**生成格点（绝不对没有证据的地方外推 —— 框边常常比引脚列长得多），
   只保留 `0.5*pitch` 内有据的格点、以及**被两个有据格点夹住**的纯插值格点
   （跨度 > `MAX_GAP` 个 pitch 就不补）；有据格点占比低于 `COVER_MIN` 整面作废；
4. 每个格点给两个文字窗：**框外的号码窗**（`out_win`）与**框内的名字窗**（`in_win`）。

第二阶段（`stage2`）读这两个窗里的字，发出 `pin_<号>` + `pinname`。
本模块**只做几何与分派，不 OCR**。

## ★ 端到端实测（不是自检，是 `work\\tools\\eval\\fast.py --no-vlm` 19 张定点档）

| 配置 | 19 张均值 | 10GT(9) | netH | pin |
|---|---|---|---|---|
| 无 pincol（真基线 `pinc_off`） | 23.30 | 35.75 | 0.2040 | 0.0052 |
| pincol + 后来作废的宽松格点 | 21.67 | 33.42 | 0.1582 | 0.0022 |
| …加尺寸闸/号码闸 | 22.49 | — | — | — |
| …加「不外推 + 有据 + 覆盖率」三道闸（`pinc_v4`） | 23.16 | 35.49 | 0.2004 | 0.0051 |
| …再加 `assemble.BOX_PIN_NEED_NAME`（`pinc_v5`，**当前默认**） | **23.83** | **37.09** | **0.2162** | 0.0088 |

真实大 IC 上收益远大于这个均值 —— `0143` 的 `U1`（GT 64 脚）单张：
**27.84 → 39.06**（pin F1 0.0163 → **0.2857**、netH 0.2839 → **0.4088**、
GT 点 5px 命中 **3/64 → 63/64**，全部 `--no-vlm`）。

失败模式（已修）：早期版本对**小 IC**过度枚举 2~5 倍（case 3 的框 118x97 只有 6 个
GT 脚却报 19 个），根因是「没拟合出 pitch 就退化成 `lat = pos_all`」+「按整条边铺格点」。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ★ 「不删除，只削弱」的全局账本（owner m23945，`docs/reports/工程哲学.md:41`/`:94`）。
#   `pincol` 是**最容易静默丢东西**的地方：整颗 IC 因为「有号码文字的边不足 MIN_NUM」
#   被 `continue` 掉、引脚列并入触点时有 `extra` 被挪到末尾 —— 历史上这些只体现在
#   一个计数里，看不出**丢的是谁**。允许独立 import（探针/自检）。
try:
    from cvparts.hypo import DEMOTE_SOFT, LEDGER
except Exception:                       # pragma: no cover
    class _NullLedger:
        def demote(self, *a, **k):
            return None

        def record(self, *a, **k):
            return None

    LEDGER = _NullLedger()
    DEMOTE_SOFT = 0.15

# 框外搜索带宽度（像素）：引脚桩一般 8~12px，留到 22 够用且不会够到图纸别的笔画
BAND = float(os.environ.get("STAGE1_PINC_BAND", "22"))
# 一个「桩」在搜索带里至少要占几个像素
MIN_STUB = int(os.environ.get("STAGE1_PINC_MIN_STUB", "2"))
# 沿边方向至少连续几行才算一个桩
MIN_RUN = int(os.environ.get("STAGE1_PINC_MIN_RUN", "1"))
# 桩必须从框边往外**连续**这么多像素（把 ✕ 标记、孤立斑点排除掉）
CONNECT = int(os.environ.get("STAGE1_PINC_CONNECT", "3"))
# pitch 搜索范围
PITCH_MIN = float(os.environ.get("STAGE1_PINC_PITCH_MIN", "5.0"))
PITCH_MAX = float(os.environ.get("STAGE1_PINC_PITCH_MAX", "48.0"))
# 一个格点离最近证据超过这么多倍 pitch 就丢掉
KEEP_REACH = float(os.environ.get("STAGE1_PINC_KEEP_REACH", "2.5"))
# 一个格点算「有据」的半径（× pitch）：有桩或有号码文字落在这个半径内
SUP_TOL = float(os.environ.get("STAGE1_PINC_SUP_TOL", "0.5"))
# 纯插值补脚时，两侧有据格点最多隔着几个 pitch
MAX_GAP = int(os.environ.get("STAGE1_PINC_MAX_GAP", "3"))
# 整个面「有据格点」的最低占比，低于此整面作废（防噪声 pitch 造出 2~5 倍假脚）
COVER_MIN = float(os.environ.get("STAGE1_PINC_COVER", "0.5"))
# pitch 只允许由「号码文字」拟合（0=不许退到噪声很大的墨迹桩）
PITCH_FALLBACK = os.environ.get("STAGE1_PINC_FALLBACK", "0").strip().lower() not in (
    "", "0", "off", "no", "false", "none")
# 号码文字落在格点上的最少内点数，低于此认为 pitch 是假的
NUM_MIN_EL = int(os.environ.get("STAGE1_PINC_MIN_EL", "4"))
# 引脚点离框边多远（官方 GT 实测 ~10.6px；与 stage1 的 PIN_OUT_BOX=12 对齐）
OUT_DIST = float(os.environ.get("STAGE1_PINC_OUT", "12.0"))
# 号码窗从框边往外多深；名字窗从框边往内多深（还会被框宽的 0.45 倍卡住）
NUM_REACH = float(os.environ.get("STAGE1_PINC_NUM_REACH", "26"))
NAME_REACH = float(os.environ.get("STAGE1_PINC_NAME_REACH", "90"))
# 「看起来像引脚号」的文字框尺寸闸（引脚号是短小的数字串）
NUM_MAX_W = float(os.environ.get("STAGE1_PINC_NUM_MAXW", "42"))
NUM_MAX_H = float(os.environ.get("STAGE1_PINC_NUM_MAXH", "26"))
# ★★ `TIP_PAD` / `TIP_TOL`：**以引脚桩为参照**找号（默认 0 = 老行为）。
#
# 动机（官方 10GT 实测）：10GT 的 IC 把引脚号画得**离框边很远** ——
# case `1` 的 `U1`（LQFP-128，框 `x 961..1314`）左边号中心在 `x≈926`（离边 28~42px）、
# 右边在 `x≈1352`（离边 38~45px），而 `NUM_REACH=26` 是从**框边**量的 ⇒ **一个号都够不到**。
# 但直接把 `NUM_REACH` 抬到 45/60 会掉分（官方 9 张 53.41 → 52.21/52.37，case 1 从
# 39.32 崩到 22.82）：放宽的是**整条边的横向带**，会把邻列的号、网标、值全吞进来，
# 格点拟合（`_fit_pitch`/`_snap_anchor`）就被带偏了。
#
# 正确做法：只在「**沿边坐标与某个引脚桩对得上**」时才允许取到 `reach` 之外、
# `reach + TIP_PAD` 之内的号。桩是几何算出来的（`_stub_positions`），不是文字，
# 所以这个闸门不引入文字噪声。`TIP_PAD=0` 时与老行为逐位相同。
TIP_PAD = float(os.environ.get("STAGE1_PINC_TIP_PAD", "0"))
TIP_TOL = float(os.environ.get("STAGE1_PINC_TIP_TOL", "6.0"))
# 一条边上最多枚举多少脚（防毛刺炸开）
MAX_SIDE = int(os.environ.get("STAGE1_PINC_MAX_SIDE", "200"))
# ★ 只枚举「像 IC 的框」：短边下限（默认 60，与 `bigic.MIN_SIDE` 同值）。
MIN_SIDE = float(os.environ.get("STAGE1_PINC_MIN_SIDE", "60"))
# ★ 至少要检到几个**真号码候选**才认这次枚举（否则纯属几何外推）。
MIN_NUM = int(os.environ.get("STAGE1_PINC_MIN_NUM", "2"))
# 是否把「没有桩、只在格点上」的位置也发出来（owner m25661「等距对齐」）
FILL_MISSING = os.environ.get("STAGE1_PINC_FILL", "1").strip().lower() not in (
    "", "0", "false", "off", "none")
# 并入 `comp_pins` 时，已有的「导线触点」离格点多近就用触点（它是真实走线接点）
MERGE_SNAP = float(os.environ.get("STAGE1_PINC_SNAP", "5.0"))
MERGE_ON = os.environ.get("STAGE1_PINC_MERGE", "1").strip().lower() not in (
    "", "0", "false", "off", "none")
IC_TYPES = ("box", "bigic")


# --------------------------------------------------------------------- 基础
def text_free_ink(ink: np.ndarray, texts: Sequence, pad: float = 0.0) -> np.ndarray:
    """把文字框的位置涂掉，只留线条墨迹。

    **这是本模块成立的前提**：框外那一排引脚号码本身就是墨，不去掉的话
    `cnt >= MIN_STUB` 在每一个引脚行都成立、`_runs` 又会把相邻行并起来，
    整条边会变成一根「桩」（`0143` 的 U1 就会从 64 个脚退化成 1 个）。
    """
    m = ink.copy()
    h, w = m.shape[:2]
    for b in texts or ():
        try:
            x1, y1, x2, y2 = [float(v) for v in (b["bbox_img"] if isinstance(b, dict) else b)]
        except (TypeError, ValueError, KeyError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        ix1 = int(max(0, math.floor(x1 - pad)))
        iy1 = int(max(0, math.floor(y1 - pad)))
        ix2 = int(min(w, math.ceil(x2 + pad)))
        iy2 = int(min(h, math.ceil(y2 + pad)))
        if ix2 > ix1 and iy2 > iy1:
            m[iy1:iy2, ix1:ix2] = 0
    return m


def _runs(mask: np.ndarray, min_len: int, gap: int = 1) -> List[Tuple[int, int]]:
    """布尔一维数组里的连续 True 段 -> [(start, end_exclusive)]。"""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = list(np.nonzero(d == 1)[0])
    ends = list(np.nonzero(d == -1)[0])
    s2: List[int] = []
    e2: List[int] = []
    for s, e in zip(starts, ends):
        if s2 and s - e2[-1] <= gap:
            e2[-1] = e
        else:
            s2.append(s)
            e2.append(e)
    return [(int(s), int(e)) for s, e in zip(s2, e2) if e - s >= min_len]


# ------------------------------------------------------------------ 证据收集
def _band_slice(box: Sequence[float], side: str, band: float):
    x0, y0, x1, y1 = [float(v) for v in box]
    B = int(round(band))
    if side == "L":
        return ("y", int(max(0, x0 - B)), int(max(0, x0 - 2)), int(y0), int(y1), -1)
    if side == "R":
        return ("y", int(x1 + 2), int(x1 + B), int(y0), int(y1), +1)
    if side == "T":
        return ("x", int(max(0, y0 - B)), int(max(0, y0 - 2)), int(x0), int(x1), -1)
    return ("x", int(y1 + 2), int(y1 + B), int(x0), int(x1), +1)


def _stub_positions(mask: np.ndarray, box: Sequence[float], side: str,
                    band: float = BAND, min_stub: int = MIN_STUB,
                    min_run: int = MIN_RUN, connect: int = CONNECT) -> List[float]:
    """去掉文字后的墨迹里，沿边方向的「引脚桩」位置。

    桩的判据是**从框边往外连续 `connect` 像素都是墨**（横向的一小段线），
    这样 ✕ 未连接标记、结点圆点、孤立斑点都不会被当成桩。
    """
    H, W = mask.shape[:2]
    axis, a0, a1, b0, b1, outdir = _band_slice(box, side, band)
    a0 = max(0, a0); a1 = min(W if axis == "y" else H, a1)
    b0 = max(0, b0); b1 = min(H if axis == "y" else W, b1)
    if a1 - a0 < connect or b1 - b0 < 3:
        return []
    seg = mask[b0:b1, a0:a1] if axis == "y" else mask[a0:a1, b0:b1].T
    if seg.shape[1] < connect:
        return []
    # 朝外方向：L/T 时列号递减 = 越靠外，所以从最内侧那 `connect+2` 列里数
    if outdir < 0:
        inner = seg[:, -min(seg.shape[1], connect + 2):]
    else:
        inner = seg[:, :min(seg.shape[1], connect + 2)]
    cnt = inner.sum(axis=1)
    out = []
    for s, e in _runs(cnt >= min_stub, min_run):
        # 段的像素中心：覆盖 s..e-1，所以是 (s + e - 1) / 2（1 行时正好是 s）
        out.append(float((s + e - 1) / 2.0) + b0)
    return out


def _num_candidates(texts: Sequence, box: Sequence[float], side: str,
                    reach: float = NUM_REACH, max_w: float = NUM_MAX_W,
                    max_h: float = NUM_MAX_H, stubs: Sequence = None,
                    tip_pad: float = None, tip_tol: float = None):
    """贴着框边的**小文字框**（引脚号的候选）。返回 `[(沿边坐标, text_index)]`。

    `tip_pad > 0` 时启用「以引脚桩为参照」的放宽（见 `TIP_PAD` 的注释）：距边
    `reach` 之外、`reach + tip_pad` 之内的小文字框，**只有沿边坐标落在某个引脚桩
    `tip_tol` 内**才收。`stubs` 是 `_stub_positions` 的沿边坐标。
    """
    if tip_pad is None:
        tip_pad = TIP_PAD
    if tip_tol is None:
        tip_tol = TIP_TOL
    _st = sorted(float(v) for v in (stubs or ()))
    outer = reach + (tip_pad if tip_pad > 0 else 0.0)

    def _aligned(d_out: float, *cands: float) -> bool:
        """`d_out` 在 `reach` 内直接放行；否则要求**号的任一条边/中心**贴着某个桩。

        ★ 为什么用「任一条边」而不是中心：实测 10GT case `1` 的 `U1` 里，
        号画在引脚线的**正上方** —— 桩在 `y=114/144/173…`，而号 `t0002` 的
        `cy=102.5`、`by2=111` ⇒ **中心差 11.5px（超过任何合理容差），
        下边缘只差 3px**。按中心对齐 ⇒ 一个号都救不回来（实测 `再救=0`）；
        按边对齐 ⇒ 唯一的物理含义是「这个号就贴着这条引脚线」。
        """
        if d_out <= reach:
            return True
        if tip_pad <= 0 or not _st:
            return False
        return any(min(abs(v - s) for s in _st) <= tip_tol for v in cands)

    x0, y0, x1, y1 = [float(v) for v in box]
    out = []
    for i, t in enumerate(texts or ()):
        b = t.get("bbox_img") if isinstance(t, dict) else t
        if not b:
            continue
        try:
            bx1, by1, bx2, by2 = [float(v) for v in b]
        except (TypeError, ValueError):
            continue
        w, h = bx2 - bx1, by2 - by1
        if w <= 0 or h <= 0 or w > max_w or h > max_h:
            continue
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        if side == "L":
            if not (x0 - outer <= cx <= x0 - 1 and y0 - 3 <= cy <= y1 + 3):
                continue
            if not _aligned(x0 - cx, cy, by1, by2):
                continue
            out.append((cy, i))
        elif side == "R":
            if not (x1 + 1 <= cx <= x1 + outer and y0 - 3 <= cy <= y1 + 3):
                continue
            if not _aligned(cx - x1, cy, by1, by2):
                continue
            out.append((cy, i))
        elif side == "T":
            if not (y0 - outer <= cy <= y0 - 1 and x0 - 3 <= cx <= x1 + 3):
                continue
            if not _aligned(y0 - cy, cx, bx1, bx2):
                continue
            out.append((cx, i))
        else:
            if not (y1 + 1 <= cy <= y1 + outer and x0 - 3 <= cx <= x1 + 3):
                continue
            if not _aligned(cy - y1, cx, bx1, bx2):
                continue
            out.append((cx, i))
    return out


# ------------------------------------------------------------------ 格点拟合
def _fit_pitch(pos: Sequence[float], min_pitch: float = PITCH_MIN,
               max_pitch: float = PITCH_MAX) -> Optional[float]:
    """从一堆（含漏检、含噪声的）位置里稳健地拟合出**等距间距**。

    先在 `[min_pitch, max_pitch)` 上直接扫候选 pitch（0.1 步长），用
    「落在格点上的点数 − 0.25×空出来的格点数」打分（`on/lattice` 判据：
    既要求点多，又要求格点不要空出一片 —— 否则取到 pitch 的 2 倍会同样「点多」），
    再用**最小二乘**精修（`pitch = Σ k·d / Σ k²`，对理想格点能精确还原）。

    试过的弯路：只用「相邻差的中位数」当 pitch，在漏检/混入谐波时会
    取到 2 倍或 5px；只用「差的直方图众数」在位置带半像素抖动时会跑偏。
    """
    s = np.asarray(sorted(float(v) for v in pos))
    if s.size < 4:
        return None
    d = np.diff(s)
    d = d[d >= max(3.0, min_pitch * 0.5)]
    if d.size < 3:
        return None
    # ① 相邻差取众数（0.5px 桶）当基础 pitch：
    #    「逐个候选去扫全局」在含噪声时会偏向**大的** pitch（tol 随 pitch 变大 ⇒
    #    大 pitch 能「解释」更多噪声点，实测把 0143 的 10px 错拟成 12.5）。
    #    相邻差众数只看局部结构，对噪声鲁棒。
    bins = np.arange(0.0, max_pitch * 1.6 + 0.5, 0.5)
    hist, edges = np.histogram(d, bins=bins)
    if hist.max() == 0:
        return None
    k = int(np.argmax(hist))
    c0 = float((edges[k] + edges[k + 1]) / 2.0)
    near = d[(d >= c0 * 0.7) & (d <= c0 * 1.4)]
    p0 = float(np.median(near)) if near.size else c0

    def _on(pitch):
        tol = min(0.2 * pitch, 1.5)
        best = 0
        for a in s:
            kk = np.round((s - a) / pitch)
            best = max(best, int((np.abs((s - a) - kk * pitch) <= tol).sum()))
        return best

    # ② 次谐波守卫：`p0/2` 也会「看起来很整齐」；若 2 倍 pitch 的解释力不差就取大的
    for mult in (2.0, 3.0):
        cand = p0 * mult
        if cand < max_pitch and _on(cand) >= _on(p0) - 1:
            p0 = cand
    tol = min(0.2 * p0, 1.5)
    best = None
    for a in s:
        kk = np.round((s - a) / p0)
        on = int((np.abs((s - a) - kk * p0) <= tol).sum())
        if best is None or on > best[0]:
            best = (on, float(a))
    if best is None or best[0] < max(3, int(0.5 * s.size)):
        return None
    anchor = best[1]
    # ③ 最小二乘精修（对理想格点精确还原）
    pitch = p0
    for _ in range(4):
        kk = np.round((s - anchor) / pitch)
        m = np.abs((s - anchor) - kk * pitch) <= max(tol, 0.35 * pitch)
        if int(m.sum()) < 3:
            break
        kk2, vv = kk[m], (s - anchor)[m]
        den = float((kk2 * kk2).sum())
        if den <= 0:
            break
        pitch = float((kk2 * vv).sum() / den)
    if not (min_pitch <= pitch < max_pitch):
        return None
    kk = np.round((s - anchor) / pitch)
    on = int((np.abs((s - anchor) - kk * pitch) <= tol).sum())
    if on < max(3, int(0.5 * s.size)):
        return None
    return pitch


def _lattice(lo: float, hi: float, pitch: float, anchor: float) -> List[float]:
    """在 `[lo, hi]` 上按 `pitch` 与相位 `anchor` 生成格点。"""
    if pitch <= 0:
        return []
    k0 = int(math.ceil((lo - anchor) / pitch - 1e-6))
    k1 = int(math.floor((hi - anchor) / pitch + 1e-6))
    if k1 < k0:
        return []
    if k1 - k0 + 1 > MAX_SIDE * 4:
        return []
    return [round(anchor + k * pitch, 3) for k in range(k0, k1 + 1)]


def _snap_anchor(pos: Sequence[float], pitch: float) -> float:
    """找一个能把最多证据吸到格点上的相位。"""
    if not pos:
        return 0.0
    best = None
    for a in pos:                      # 每个证据都当一次锚，取最优相位
        kk = np.round((np.asarray(pos) - a) / pitch)
        resid = np.abs((np.asarray(pos) - a) - kk * pitch)
        on = int((resid <= 0.25 * pitch).sum())
        # 相位微调：把残差的中位数补回去
        shift = float(np.median((np.asarray(pos) - a) - kk * pitch))
        if best is None or on > best[0]:
            best = (on, a + shift)
    return best[1] if best else float(pos[0])


# ------------------------------------------------------------------ 单侧枚举
def _windows(box: Sequence[float], side: str, pos: float, pitch: Optional[float]):
    """给一个格点算 (引脚点, 号码窗, 名字窗)。窗是图像坐标的 `(x1,y1,x2,y2)`。"""
    x0, y0, x1, y1 = [float(v) for v in box]
    half = max(3.0, (pitch or 10.0) * 0.48)
    name = min(NAME_REACH, max(24.0, 0.45 * max(x1 - x0, y1 - y0)))
    if side == "L":
        pin = (x0 - OUT_DIST, pos)
        out = (x0 - NUM_REACH, pos - half, x0 - 1.0, pos + half)
        inn = (x0 + 1.0, pos - half, min(x1, x0 + name), pos + half)
    elif side == "R":
        pin = (x1 + OUT_DIST, pos)
        out = (x1 + 1.0, pos - half, x1 + NUM_REACH, pos + half)
        inn = (max(x0, x1 - name), pos - half, x1 - 1.0, pos + half)
    elif side == "T":
        pin = (pos, y0 - OUT_DIST)
        out = (pos - half, y0 - NUM_REACH, pos + half, y0 - 1.0)
        inn = (pos - half, y0 + 1.0, pos + half, min(y1, y0 + name))
    else:
        pin = (pos, y1 + OUT_DIST)
        out = (pos - half, y1 + 1.0, pos + half, y1 + NUM_REACH)
        inn = (pos - half, max(y0, y1 - name), pos + half, y1 - 1.0)
    return pin, out, inn


def _in_win(cx: float, cy: float, win) -> bool:
    return win[0] <= cx <= win[2] and win[1] <= cy <= win[3]


def _side_records(mask: np.ndarray, box: Sequence[float], side: str,
                  texts: Sequence, cents: Sequence, cid: str) -> List[dict]:
    stubs = _stub_positions(mask, box, side)
    nums = _num_candidates(texts, box, side, stubs=stubs)
    npos = sorted(set(round(float(v), 3) for v in nums_pos(nums)))
    spos = sorted(set(round(float(v), 3) for v in stubs))
    pos_all = sorted(set(spos) | set(npos))
    if not pos_all:
        return []
    # **引脚号文字是可靠证据；墨迹桩噪声很大**（实测 0143 的 L 边桩里混着 ✕ 标记、
    # 折线拐点、文字残墨，一大半不在真格点上）。所以先只用号码文字拟合 pitch，
    # 不够 4 个再退到全部证据。
    pitch = None
    _evs = [npos] + ([pos_all] if PITCH_FALLBACK else [])
    for ev in _evs:
        if len(ev) >= 4:
            pitch = _fit_pitch(ev)
            if pitch:
                break
    # 【号码文字的内点数要够】实测几个小 IC（case 3 的框 118x97 只有 6 个脚、case 7 的
    # 138x118 只有 8 个脚）边上只有零星几个号码文字，拟合出来的 pitch 是假的 ——
    # 一旦按它铺格点就会造出 2~5 倍的假脚，把 netH 拉下来。所以要求内点数
    # >= NUM_MIN_EL，且 pitch 必须来自号码文字（PITCH_FALLBACK=0 时不许退到墨迹桩）。
    if pitch is not None:
        _np = np.asarray(npos) if npos else None
        _on = 0
        if _np is not None and _np.size:
            _lat0 = _lattice(float(_np.min()), float(_np.max()),
                             pitch, _snap_anchor(npos, pitch))
            _on = int(sum(1 for v in _np
                          if min((abs(v - w) for w in _lat0), default=1e9) <= SUP_TOL * pitch))
        if _on < NUM_MIN_EL:
            pitch = None
            if not PITCH_FALLBACK:
                return []
    if pitch is None:
        # 【没拟合出 pitch 就一个脚都不发】早期版本在这里退化成 `lat = pos_all`，
        # 等于把「墨迹桩 + 号码文字」的原始位置直接当引脚 —— 小 IC 的框边全是导线
        # 拐点、✕ 标记、网络标签的残墨，发出去全是假脚（case 3 的框 118x97 因此
        # 报了 19 个脚、GT 只有 6 个）。宁可一个不发。
        return []
    else:
        anchor = _snap_anchor(npos if len(npos) >= 4 else pos_all, pitch)
        x0, y0, x1, y1 = [float(v) for v in box]
        # 格点铺满整条边（不是只铺证据之间）—— 实测（0143 的 U1）引脚精确落在
        # `[y0 + 1.0*pitch, y1 - 1.0*pitch]` 上，所以内缩半个 pitch 既不会漏掉
        # 两端的脚、也不会在框角上凭空补脚。
        pad = 0.5 * pitch
        if side in ("L", "R"):
            lo, hi = y0 + pad, y1 - pad
        else:
            lo, hi = x0 + pad, x1 - pad
        # 【绝不对没有证据的地方外推】格点只在「证据跨度」内铺。框边常常比引脚列
        # 长得多：实测 case 3 的框 118x97 只有 6 个脚、case 7 的 138x118 只有 8 个脚，
        # 按整条边铺格点会多造 2~5 倍的脚（这正是 19 张 A/B 里 netH 掉分的根源）。
        # 最外两个证据各外扩 0.6 个 pitch 就够。
        ev = np.asarray(pos_all)
        lo = max(lo, float(ev.min()) - 0.6 * pitch)
        hi = min(hi, float(ev.max()) + 0.6 * pitch)
        lat = _lattice(lo, hi, pitch, anchor)
        # 【只留「有据」的格点】0.5*pitch 内有桩或号码文字才算有据；无据的格点只有
        # 被两个有据格点夹住（纯插值）且跨度 <= MAX_GAP 时才补（FILL_MISSING 关则一律不补）。
        stub_a0 = np.asarray(stubs) if stubs else None
        sup = []
        for v in lat:
            ok = float(np.abs(ev - v).min()) <= SUP_TOL * pitch
            if not ok and stub_a0 is not None and stub_a0.size:
                ok = float(np.abs(stub_a0 - v).min()) <= SUP_TOL * pitch
            sup.append(bool(ok))
        keep = []
        for i, v in enumerate(lat):
            if sup[i]:
                keep.append(v)
                continue
            if not FILL_MISSING:
                continue
            li = i - 1
            while li >= 0 and not sup[li]:
                li -= 1
            ri = i + 1
            while ri < len(lat) and not sup[ri]:
                ri += 1
            if li >= 0 and ri < len(lat) and (ri - li) <= MAX_GAP:
                keep.append(v)
        lat = keep
        # 【补出来的脚不能占多数】一整个面若「0.5*pitch 内没有任何证据」的格点过半，
        # 说明这条边的证据其实是噪声、pitch 是假的（实测 case 3 R 15 个里只有 6 个有据、
        # case 7 L 19 个里 5 个、case 8 L 30 个里 2 个）——整面作废，宁可不发。
        if lat:
            n_ev = 0
            for v in lat:
                ok = float(np.abs(ev - v).min()) <= SUP_TOL * pitch
                if not ok and stub_a0 is not None and stub_a0.size:
                    ok = float(np.abs(stub_a0 - v).min()) <= SUP_TOL * pitch
                if ok:
                    n_ev += 1
            if n_ev < COVER_MIN * len(lat):
                return []
    if not FILL_MISSING:
        lat = [v for v in lat
               if min((abs(v - s) for s in stubs), default=1e9) <= 0.35 * (pitch or 10.0)]
    lat = lat[:MAX_SIDE]
    stub_a = np.asarray(stubs) if stubs else None
    _half = max(3.0, (pitch or 10.0) * 0.48)

    # ★★★ 引脚点要落在**墨迹桩**的格点上，而 `lat` 是「号码文字中心」的格点 ——
    # 两者差一个固定的「号写在引脚线上方」偏移。10GT 实测：case `10` 的 `U1`
    # 差 **14.5px**、case `8` 差 ~14px、case `1` 差 11.5px，而 `cfg.pin_tol = 5.0`。
    #
    # 不改会怎样（2026-10 实测）：`merge_into` 把这些偏离的点当成**新脚**
    # （case 10 `added 1 / extra 14`、case 8 `added 9 / extra 23`），既灌进一堆
    # 纯 fp 的假脚，又把 `pin.position_mismatch` 拉满 —— 官方 9 张的「位置错 18」。
    #
    # 偏移量取**每条边一个中位数**（不是逐点取最近桩）：逐点最近桩在偏移接近
    # 半个格距时会**认到隔壁脚的桩**（case 8 的 14px vs pitch 20px 就已经过线），
    # 而整条边的偏移是同一个物理量，取中位数既稳又能被多数点投票保护。
    # 只有「有一半以上的格点能在半格内找到桩」时才采信。
    # `lat` 本身**不动** —— 号码窗 / 名字窗仍然开在文字所在的位置。
    _off = 0.0
    if pitch and stub_a is not None and stub_a.size and len(lat) >= 3:
        _ds = []
        for _v in lat:
            _j = int(np.argmin(np.abs(stub_a - _v)))
            if abs(float(stub_a[_j]) - _v) <= 0.5 * pitch:
                _ds.append(float(stub_a[_j]) - _v)
        if len(_ds) >= max(2, int(0.5 * len(lat))):
            _m = float(np.median(_ds))
            if abs(_m) <= 0.5 * pitch:
                _off = _m
    recs = []
    for k, p in enumerate(lat):
        pin, out_win, in_win = _windows(box, side, p, pitch)
        # 引脚点 = 号码格点 + 该边的「号↔桩」偏移（见上）。`TIP_PAD>0` 时再做一次
        # 逐点微调（残余偏移 < 0.5*pitch，Voronoi 分界保证无歧义）。
        _pa = float(p) + _off
        if TIP_PAD > 0 and stub_a is not None and stub_a.size:
            _ok = np.abs(stub_a - _pa) <= 0.5 * (pitch or 10.0)
            if bool(_ok.any()):
                _pa = float(stub_a[_ok][int(np.argmin(np.abs(stub_a[_ok] - _pa)))])
        if side in ("L", "R"):
            pin = (pin[0], _pa)
        else:
            pin = (_pa, pin[1])
        out_ids, in_ids, seen_o = [], [], set()
        # ★★ 号码候选（`_num_candidates`，已按「与引脚桩对齐」筛过）也算「贴边号码」。
        #
        # 动机（10GT case `1` 的 `U1`，2026-10 实测）：号画在**离框边 28~45px** 处，
        # 而 `_windows` 的号码窗只到 `NUM_REACH=26` ⇒ 每个格点的 `out_texts` 全空
        # ⇒ `enumerate_pins` 的「至少 `MIN_NUM` 个脚有号码文字」闸门把**整颗 IC 丢掉**
        # （实测 `pincol.enumerate_pins` 对 case 1 返回 `{}`，`pincol` 账全是 0）。
        # `_num_candidates` 用的是同一套几何、但受 `TIP_PAD` 的桩对齐闸门控制，
        # 所以这里把它的结论并进来：**两条通道的并集**，`TIP_PAD=0` 时两者等价、
        # 行为与历史逐位相同。
        for _a, _ti in nums:
            if abs(float(_a) - p) <= _half and _ti not in seen_o:
                seen_o.add(_ti)
                out_ids.append(_ti)
        for ti, (cx, cy) in enumerate(cents):
            if cx < -1e8:
                continue
            if _in_win(cx, cy, out_win):
                if ti not in seen_o:
                    seen_o.add(ti)
                    out_ids.append(ti)
            elif _in_win(cx, cy, in_win):
                in_ids.append(ti)
        has_stub = bool(stub_a is not None
                        and np.abs(stub_a - _pa).min() <= 0.35 * (pitch or 10.0))
        recs.append({
            "cid": cid, "side": side, "idx": k,
            "x": round(float(pin[0]), 1), "y": round(float(pin[1]), 1),
            "axis": ("y" if side in ("L", "R") else "x"),
            "pitch": (round(float(pitch), 2) if pitch else None),
            "synth": (not has_stub),
            "out_win": [round(float(v), 1) for v in out_win],
            "in_win": [round(float(v), 1) for v in in_win],
            "out_texts": out_ids, "in_texts": in_ids,
        })
    return recs


def nums_pos(nums) -> List[float]:
    return [float(p) for p, _ in nums]


def enumerate_component(mask: np.ndarray, box: Sequence[float],
                        texts: Sequence = None, cid: str = "") -> List[dict]:
    """枚举**一个**框元件的四条边的引脚列。返回该元件的引脚记录列表。"""
    cents: List[Tuple[float, float]] = []
    for b in (texts or ()):
        bb = b["bbox_img"] if isinstance(b, dict) else b
        try:
            bx1, by1, bx2, by2 = [float(v) for v in bb]
        except (TypeError, ValueError):
            cents.append((-1e9, -1e9))
            continue
        cents.append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))
    recs: List[dict] = []
    for side in ("L", "R", "T", "B"):
        recs.extend(_side_records(mask, box, side, texts, cents, cid))
    return recs


def enumerate_pins(bgr: np.ndarray, comps: Sequence[dict], texts: Sequence = None,
                   ink: np.ndarray = None, ink_thresh: int = 170) -> Dict[str, List[dict]]:
    """对每个 `box`/`bigic` 元件枚举引脚列。返回 `{cid: [记录, ...]}`。

    `texts` 是 `cvparts.textbox.detect_boxes` 的输出（图像坐标 `[x1,y1,x2,y2]`，
    元素可以是 dict（带 `bbox_img`）也可以是裸列表）。
    """
    if ink is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ink = (gray < ink_thresh).astype(np.uint8)
    mask = text_free_ink(ink, texts or (), pad=0.0)
    out: Dict[str, List[dict]] = {}
    for c in comps or ():
        bb = c.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        ct = str(c.get("type") or "").lower()
        if not any(t in ct for t in IC_TYPES):
            continue
        x0, y0, x1, y1 = [float(v) for v in bb]
        if not (6 <= abs(x1 - x0) <= 1600 and 6 <= abs(y1 - y0) <= 1600):
            continue
        # ★★ **只对「像 IC 的框」枚举**（`STAGE1_PINC_MIN_SIDE`，默认 60，和
        # `bigic.MIN_SIDE` 同值）。
        #
        # 19 张定点档实测（`STAGE1_PINC=1 STAGE2_PINCOL=0` vs `=0`）：平均 21.65 vs
        # 23.30，**全部损失来自 netH**（0.2040→0.1582）。逐元件看，闯祸的是 0024 那
        # 16 个 **33×33 px** 的小方框 —— 每个都被枚举出 8~12 个脚、编号全是位置补的，
        # 而 GT 里那些元件本来就没对上（该 case `component` F1 只有 0.0357）。真正
        # 受益的是 0143 的 `U1`（148×330、GT 64 脚）：单张 27.84 → 36.49。
        # 33px 的框根本装不下 11 个 pitch≥5px 的脚，这种「枚举」只是把噪声当引脚。
        w, h = abs(x1 - x0), abs(y1 - y0)
        if min(w, h) < MIN_SIDE or max(w, h) < MIN_SIDE:
            continue
        recs = enumerate_component(mask, bb, texts, cid=str(c.get("id")))
        # 还要有**真号码候选**兜底：一条边的号一个都没检到，那条边就是纯几何外推。
        if recs and sum(1 for r in recs if r.get("out_texts")) < MIN_NUM:
            # ★★ 「不删除，只削弱」：这里是**整颗 IC 被丢掉**的地方，而它以前是
            #   **静默**的 —— `pincol` 里从此没有这颗元件，复核者分不清「没枚举出
            #   引脚列」和「枚举出来了但没过闸门」。现在把丢掉的是谁、枚举出几条边、
            #   每条边有几个号码候选、判据是什么，写进全局账本（值本身留在 `before`
            #   里，降权分取 `DEMOTE_SOFT > 0`，绝不取 0）。
            LEDGER.record("drop", f"pincol:{c.get('id')}",
                          before=[{"side": r.get("side"),
                                   "n_out": len(r.get("out_texts") or []),
                                   "x": round(float(r.get("x", 0.0)), 1),
                                   "y": round(float(r.get("y", 0.0)), 1)}
                                  for r in recs],
                          after=None,
                          reason=f"有号码文字的边 < MIN_NUM={MIN_NUM}，整颗 IC 的引脚列被丢弃",
                          by="pincol:enumerate_pins", at="stage1:pincol")
            LEDGER.demote(f"pincol:{c.get('id')}", c.get("type"),
                          float(len(recs)), DEMOTE_SOFT,
                          reason="引脚列不合格（证据留在 drop 事件的 before 里）",
                          by="pincol:enumerate_pins", at="stage1:pincol")
            continue
        if recs:
            out[str(c.get("id"))] = recs
    return out


# ------------------------------------------------------------------ 并入 comp_pins
def merge_into(comp_pins: Dict[str, list], cols: Dict[str, list],
               snap: float = MERGE_SNAP
               ) -> Tuple[Dict[str, list], Dict[str, List[int]], dict]:
    """把枚举出来的引脚列并进 `comp_pins`（= `nets.build_pins` 的输出）。

    规则（**不丢弃**）：
    - 枚举点附近 `snap` px 内已有触点 ⇒ 用**触点**的坐标（它是真实走线接点，
      网络装配靠它），触点被标记为已用；
    - 枚举点附近没有触点 ⇒ 用格点坐标（这就是「等距对齐」的补脚）；
    - 剩下没被任何枚举点认领的触点 ⇒ **原样保留**在末尾。

    返回 `(新 comp_pins, 每脚「是不是补出来的」标志, 统计)`。
    标志与 `新 comp_pins[cid]` **逐位对齐**：`1` = 这一脚是格点补出来的
    （框边外侧 3px 内没有墨 ⇒ **没有导线接上去**），`0` = 有真实触点。
    这个标志是「并行导线等距对齐」的关键输入：相邻脚间距只有 ~10px，而
    接线半径 `PIN_JOIN_DIST=12` 比它还大，补出来的悬空脚会**凑到隔壁脚的导线**
    上去（把两条网络并成一条）。有了标志，`build_nets` 就能对补脚用**紧半径**。

    统计键 `comps/enum/kept/added/extra`。
    """
    out = {cid: [tuple(map(float, p)) for p in (pl or [])]
           for cid, pl in (comp_pins or {}).items()}
    flags: Dict[str, List[int]] = {cid: [] for cid in out}
    stats = {"comps": 0, "enum": 0, "kept": 0, "added": 0, "extra": 0}
    for cid, recs in (cols or {}).items():
        if not recs:
            continue
        stats["comps"] += 1
        old = out.get(cid) or []
        used = [False] * len(old)
        merged: List[Tuple[float, float]] = []
        n_kept = n_added = 0
        for r in recs:
            rx, ry = float(r["x"]), float(r["y"])
            stats["enum"] += 1
            best, bj = 1e9, -1
            for j, (ox, oy) in enumerate(old):
                if used[j]:
                    continue
                d = math.hypot(rx - ox, ry - oy)
                if d < best:
                    best, bj = d, j
            if bj >= 0 and best <= snap:
                used[bj] = True
                merged.append(old[bj])
                flags[cid].append(0)
                stats["kept"] += 1
                n_kept += 1
            else:
                merged.append((rx, ry))
                # `synth`：格点位置在框边外侧没有墨 ⇒ 悬空补脚。即便枚举时
                # 有「号字」被当成证据，也不能当成导线接点。
                flags[cid].append(1 if r.get("synth") else 0)
                stats["added"] += 1
                n_added += 1
        for j, p in enumerate(old):
            if not used[j]:
                merged.append(p)
                flags[cid].append(0)
                stats["extra"] += 1
        out[cid] = merged
        # ★ merge 事件（owner 要求「每次 merge/split 必须写一条
        #   `{kind, before, after, reason}`」）：这里把「几何触点」与「枚举格点」
        #   两种来源合成一张引脚表，`kept` = 用触点坐标的那部分、`added` = 补出来的
        #   格点、`extra` = 没被认领但**原样保留**的触点。账目进全局账本。
        LEDGER.record("merge", f"comp:{cid}",
                      before=[{"n_old": len(old), "n_recs": len(recs)}],
                      after={"n_merged": len(merged), "kept": n_kept,
                             "added": n_added,
                             "extra": len(merged) - n_kept - n_added,
                             "n_synth": sum(flags[cid])},
                      reason="引脚列枚举并入几何触点（枚举点附近 snap 内有触点则用触点）",
                      by="pincol:merge_into", at="stage1:pincol")
    return out, flags, stats


# ---------------------------------------------------------------------------
# 把引脚列窗口变成**能直接送进 CRNN 的文字框**
#
# 动机：`cvparts.textbox` 对 6x8 px 的引脚号召回很低 —— 0143 的 U1（GT 64 脚）
# 只检出约 33 个号。而 pincol 已经知道每个号的窗口在哪，所以不必等检测器：
# 直接按窗口切、直接读。
#
# 外扩量写死不走比例：`work\reports\尺度归一_诊断.md` 实测识别 exact 对**定长**
# 外扩的曲线是 `+0 0.1915 · +2 0.6345 · +3 0.6544 · +4 0.6105 · +5 0.5076 ·
# +6 0.2554`（真值框原样 0.8430）⇒ 最优是**定长 +3px**，不是按比例的
# `max(2, 0.30*h)`（后者在 h≈19 时给到 5.7px，已越过峰值）。
# ---------------------------------------------------------------------------
TIGHT_PAD_X = float(os.environ.get("STAGE1_PINC_TIGHT_X", "2.0"))
TIGHT_PAD_Y = float(os.environ.get("STAGE1_PINC_TIGHT_Y", "3.0"))
TIGHT_MIN_H = int(os.environ.get("STAGE1_PINC_TIGHT_MINH", "3"))
# ★★ 只收「最靠近窗口中心那一行」的连通块。
#
# 这个参数是**必须的**，不是调优：0143 的 U1 相邻脚间距 10px、号码字高 8px，
# 窗口半高本来只有 ~0.48*pitch ≈ 4.8px，但只要把窗内**所有**高度 ≥3 的连通块取
# 并集，上下两根脚的号就会被并成一块 15px 高的裁剪（实测窗 `[124,50,142,65]`、
# 下一个 `[124,60,142,75]`，纵向重叠 5px）—— 一张裁剪里含着两根脚的号，
# CRNN 于是把每根脚都读成同一个 `3`（GT=1/2/8/9/14/… 全读成 `3`）。
CENTRE_FRAC = float(os.environ.get("STAGE1_PINC_CENTRE_FRAC", "0.30"))


def tighten(gray: np.ndarray, win, ink_thresh: int = 170,
            pad_x: float = None, pad_y: float = None,
            min_h: int = None, centre_frac: float = None) -> Tuple[float, float, float, float]:
    """把窗口收紧到里面**像字的墨迹连通块**上，再按定长外扩。

    丢掉高度 < `min_h` 的连通块（细横线、✕ 的碎块），并且**只留最靠近窗口中心
    那一行的块**（否则会把上下相邻脚的号并进同一张裁剪，见 `CENTRE_FRAC` 的注释）。
    一个合格块都没有时原样返回窗口（宁可读一个空框，也别读到别处的字）。
    """
    pad_x = TIGHT_PAD_X if pad_x is None else pad_x
    pad_y = TIGHT_PAD_Y if pad_y is None else pad_y
    min_h = TIGHT_MIN_H if min_h is None else min_h
    centre_frac = CENTRE_FRAC if centre_frac is None else centre_frac
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in win[:4])
    x0, y0 = max(0, min(x0, W - 1)), max(0, min(y0, H - 1))
    x1, y1 = max(x0 + 1, min(x1, W)), max(y0 + 1, min(y1, H))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return (float(win[0]), float(win[1]), float(win[2]), float(win[3]))
    sub = (gray[y0:y1, x0:x1] < ink_thresh).astype(np.uint8)
    n, _lab, stats, _cen = cv2.connectedComponentsWithStats(sub, 8)
    win_cy = (y1 - y0) / 2.0
    cand = []
    for i in range(1, n):
        xx, yy, ww, hh, area = (int(v) for v in stats[i])
        if hh < min_h or area < 3:
            continue
        cy = yy + hh / 2.0
        cand.append((abs(cy - win_cy), xx, yy, ww, hh))
    if not cand:
        return (float(win[0]), float(win[1]), float(win[2]), float(win[3]))
    cand.sort()
    lim = max(2.0, centre_frac * (y1 - y0))
    keep = [c for c in cand if c[0] <= lim] or [cand[0]]
    bx0 = min(c[1] for c in keep)
    by0 = min(c[2] for c in keep)
    bx1 = max(c[1] + c[3] for c in keep)
    by1 = max(c[2] + c[4] for c in keep)
    return (float(max(0, x0 + bx0 - pad_x)), float(max(0, y0 + by0 - pad_y)),
            float(min(W, x0 + bx1 + pad_x)), float(min(H, y0 + by1 + pad_y)))


def win_boxes(bgr: np.ndarray, cols: Dict[str, List[dict]],
              keys: Sequence[str] = ("out_win",), tight: bool = True,
              ink_thresh: int = 170) -> List[Tuple[Tuple[float, float, float, float],
                                                   str, str]]:
    """`cols`（`enumerate_pins` 的输出）→ `[(框, cid, 窗口名), ...]`。

    默认只取 `out_win`（框外的**引脚号**）—— 名字在框内、文字检测器本来就检得到，
    而号是 6x8 px 的数字，恰恰是漏检最狠的那一类。
    """
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for cid, recs in (cols or {}).items():
        for r in (recs or []):
            for k in keys:
                w = r.get(k)
                if not w or len(w) != 4:
                    continue
                b = tighten(gray, w, ink_thresh) if tight else tuple(float(v) for v in w)
                if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
                    out.append((b, cid, k))
    return out


def summarise(cols: Dict[str, List[dict]]) -> str:
    n = sum(len(v) for v in cols.values())
    n_synth = sum(1 for v in cols.values() for r in v if r["synth"])
    n_out = sum(1 for v in cols.values() for r in v if r["out_texts"])
    n_in = sum(1 for v in cols.values() for r in v if r["in_texts"])
    return ("引脚列：元件 %d / 引脚 %d（无桩格点 %d、框外有字 %d、框内有字 %d）"
            % (len(cols), n, n_synth, n_out, n_in))


# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    fails = []

    def chk(name, got, want):
        ok = got == want
        if not ok:
            fails.append(name)
        print("  [%s] %-46s got %s want %s" % ("OK" if ok else "FAIL", name, got, want))

    print("1) pitch 拟合：理想 10px 格点（含大间隔）")
    p = _fit_pitch([55 + 10 * i for i in range(32)])
    chk("pitch", round(p, 2), 10.0)

    print("2) pitch 拟合：漏掉几个点仍然拟合得出")
    p = _fit_pitch([55 + 10 * i for i in range(32) if i not in (3, 4, 17)])
    chk("pitch", round(p, 2), 10.0)

    print("3) pitch 拟合：混入 5px 谐波噪声仍取 10")
    p = _fit_pitch([55 + 10 * i for i in range(16)] + [60.5, 70.5, 80.5])
    chk("pitch", round(p, 2), 10.0)

    print("4) 桩必须「贴着框边往外连续」——✕ 标记这种孤立斑点不算桩")
    ink = np.zeros((220, 200), np.uint8)
    box = [100.0, 20.0, 160.0, 200.0]
    for i in range(12):                       # 孤立斑点（像 ✕）落在带里但不碰框边
        ink[30 + i * 10, 80:86] = 1
    chk("孤立斑点：0 根桩", len(_stub_positions(ink, box, "L")), 0)
    for i in range(12):                       # 真桩：贴着框边（内侧 5 列）往外
        ink[30 + i * 10, 86:100] = 1
    chk("真桩：12 根", len(_stub_positions(ink, box, "L")), 12)

    print("5) 号字压到框边内侧时会并成一根桩 —— 这是仍然要去字的原因")
    ink = np.zeros((220, 200), np.uint8)
    for i in range(12):                       # 只在框边内侧画号字（x 93..97）
        y = 30 + i * 10
        ink[y - 5:y + 5, 93:98] = 1
    boxes = [[93, 30 + i * 10 - 5, 98, 30 + i * 10 + 5] for i in range(12)]
    chk("脏墨迹：并成 1 根桩", len(_stub_positions(ink, box, "L")), 1)
    chk("去字后：0 根桩", len(_stub_positions(text_free_ink(ink, boxes), box, "L")), 0)

    print("6) 文字分派 + 格点补齐：桩只在偶数序号的脚上，格点应补出奇数序号的脚")
    ink2 = np.zeros((300, 400), np.uint8)
    box2 = [200.0, 40.0, 340.0, 170.0]      # 框高 130：现实里框是按引脚数撑出来的
    for i in range(0, 12, 2):                 # 只画偶数序号的桩
        y = 50 + i * 10
        ink2[y, 188:200] = 1
    texts = []
    for i in range(12):
        y = 50 + i * 10
        texts.append([192, y - 5, 198, y + 5])          # 框外号码（都在）
        texts.append([205, y - 5, 245, y + 5])          # 框内名字
    cols = enumerate_component(ink2, box2, texts, cid="u1")
    left = [r for r in cols if r["side"] == "L"]
    chk("12 个格点", len(left), 12)
    chk("其中 6 个「无桩」", sum(1 for r in left if r["synth"]), 6)
    chk("每个都拿到框外字", sum(1 for r in left if r["out_texts"]), 12)
    chk("每个都拿到框内字", sum(1 for r in left if r["in_texts"]), 12)

    print("7) merge_into：触点优先、格点补齐、多余触点保留")
    cp = {"c1": [(188.0, 50.0), (188.0, 450.0)]}        # 第 2 个离所有格点都远
    cols = {"c1": [{"x": 188.0, "y": 50.0, "synth": False},
                   {"x": 188.0, "y": 70.0, "synth": True}]}
    out, flags, st = merge_into(cp, cols)
    chk("保留触点 1 个", st["kept"], 1)
    chk("补格点 1 个", st["added"], 1)
    chk("多余触点 1 个", st["extra"], 1)
    chk("总数 3", len(out["c1"]), 3)
    # 标志逐位对齐：第 0 个留的是真实触点 ⇒ 0；第 1 个是格点补的 ⇒ 1；
    # 末尾那个没人认领的触点 ⇒ 0（它有导线，只是不在格点上）。
    chk("synth 标志", flags["c1"], [0, 1, 0])

    print("")
    if fails:
        print("**有 %d 个 FAIL，禁止把结果当真**：%s" % (len(fails), fails))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(_selftest())
