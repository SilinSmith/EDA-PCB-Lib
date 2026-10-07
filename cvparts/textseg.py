"""文字**切分**（给 OCR 用，与 `textbox.py` 分工不同）。

为什么要有第二个文字检测器
--------------------------
`textbox.py` 是为**第一阶段的导线提取**调的：它的产物是「文字行的框」，用来把文字从墨迹里
涂掉（`text_mask`），调优口径是「引脚名**覆盖**率 @20px = 0.9864」——只要框**挨到**引脚名
就算赢。那是**定位**口径，不是**裁字**口径。

拿它去给 CRNN 裁字，实测（220 张官方图 + 32 张 CVJsonStd 真值）：
    - 检出框对真值框中位 IoU 只有 **0.357**，`>=0.5` 只占 11.5%
    - 19.7% 的检出框落不到任何真值框上（**图形符号误检**：电阻锯齿、三极管、电池、地）
    - 结果：真值紧框读 exact **0.413**，检出框读 exact **0.027** —— **差 15 倍**
所以本模块重做切分，口径是「裁得准」，直接以 CRNN 的 exact 为验收指标。

核心判别（这是与 `textbox.py` 唯一实质的区别）
--------------------------------------------
`textbox.py` 只把**长直线像素**从墨迹里减掉（形态学开运算），留下的锯齿/符号照样成连通域，
于是被判成字形。但符号和文字的**拓扑**差别很清楚：

    符号（电阻锯齿、地、电池、三极管……）是**接线的一部分**，它和导线在原始墨迹里属于
    同一个 8 连通分量；文字是**独立的一坨**，和导线之间隔着一两个像素的空白。

⇒ **结构分量** = 「含有长直线像素」**且**「外接框最长边 >= `STRUCT_DIM`」的原始墨迹连通分量，
   整块丢掉（导线、图纸边框、IC 外框、挂在导线上的符号一起没）。
   加 `STRUCT_DIM` 这一条是必须的：一个 16px 高的 "1" 竖笔本身就是一条 16px 的长直线，
   只看「有没有长直线」会把高字母整块误杀。

⇒ **回收流（`recover=True`）**：被丢掉的结构分量里，减掉长直线像素后如果还剩**高个子紧凑块**
   （`h >= RESID_MIN_H`、短边 >= `MIN_DIM`、尺寸/填充率像字形），就捞回来，再把**上下被切断的
   同一个字**重新缝上（`_stitch`）。这是给「文字压在导线上 / 贴着图纸边框」用的。
   锯齿残渣是 4x8 的小块（h=8 < 10），会被 `RESID_MIN_H` 挡掉。

自检（`python eda-pcb\\cvparts\\textseg.py`，7 条合成用例）逐条覆盖上面每一条。

和 `textbox.py` 的关系：**textbox.py 不动**（stage1 的导线掩膜继续用它，那边是覆盖优先；
本模块只给 stage2 的 OCR 裁字用）。两套口径不同是刻意的，别合并。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

Box = Tuple[float, float, float, float]

INK = 170            # gray < INK 算墨（mode="fixed" 时的阈值）
INK_MODE = "max"     # "fixed" / "max" / "otsu" —— 见 ink_mask 的 docstring
INK_CEIL = 235       # 自适应阈值的上限（再高就把浅灰背景也当墨了）
LINE_LEN = 25        # 开运算结构元长度：>= 它的直线段算「长直线」。
#   取 25 而不是 15：字符自己的竖笔在 24px 高的字里能到 24px，15 会把「贴着导线的字」
#   整个吃掉（自检用例④ 就是这条）。导线/边框/IC 外框都远长于 25。
STRUCT_DIM = 36      # 「结构分量」的外接框最长边下限
RESID_MIN_H = 10     # 回收流的要求高度下限
RESID_MIN_DIM = 2    # 回收流的短边下限（可以比主流的 MIN_DIM 松，因为有高度门槛兜着）
STITCH_GAP = 16      # 缝合同一个被切断的字：上下间距上限
STITCH_OV = 0.5      # 缝合同一个字：水平重叠比例下限
MIN_DIM = 3          # 字形的短边下限（挡掉 1~2px 宽的导线残渣）
MIN_H, MAX_H = 5, 48     # 字形高度范围（真实图纸实测 8~36px，留余量）
MIN_W, MAX_W = 1, 80
MIN_AREA = 6
FILL_LO, FILL_HI = 0.08, 0.95
# 并成文字行（串）
GAP_FRAC = float(os.environ.get("TEXTSEG_GAP", "1.2"))
"""水平间距 <= GAP_FRAC * 较小字高 才并。`TEXTSEG_GAP` 环境变量可覆盖。

实测（10 张 / 1287 个真值框，加 `pad_box` 后比公共子集；此时 `STITCH_H_MULT=2.2` 已生效）：
  0.45 → 0.5985 · 0.8 → 0.6204 · **1.2 → 0.6456** · 1.8 → 0.6204
旧值 0.45 是**在还没有高度闸的时候**定的，那时放大间距会把上下相邻的串并成一个高框，
所以只能压得很小；有了 `STITCH_H_MULT` 这道闸之后，间距就该放宽到「一个字宽」的量级。
"""
V_OVERLAP = 0.40     # 垂直重叠比例下限
H_RATIO = 2.2        # 同行内高度比上限
MAX_LINE_W = 400.0   # 一行（串）的最大宽度，防止把整行表格并成一条
STITCH_H_MULT = float(os.environ.get("TEXTSEG_H_MULT", "2.2"))
"""缝出来的字号块 / 一行字，总高不得超过全图中位字高的这么多倍。

用全图的中位字高当尺子，而不是拿固定像素 —— 同一张图上字号是接近的，
跨图差异却很大。实测（10 张 / 1287 个真值框，加 `pad_box` 后比公共子集）：
  不开这个闸 0.4321 · 2.2 倍 **0.5985**（命中也从 1152 涨到 1253、IoU p50 0.303→0.359）。
`TEXTSEG_H_MULT` 环境变量可覆盖，便于扫描。
"""


# ------------------------------------------------------------------ 基础

def otsu_thr(gray: np.ndarray) -> int:
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(round(t))


def ink_mask(bgr: np.ndarray, ink: int = INK, mode: str = INK_MODE) -> np.ndarray:
    """墨迹掩膜。`mode`：

    * ``"fixed"`` —— 固定阈值 `ink`（原来只有这一种）。
    * ``"max"``   —— ``max(ink, Otsu)``，上限 `INK_CEIL`。
    * ``"otsu"``  —— 纯 Otsu。

    为什么需要自适应：**低对比度的图纸**（灰字，例如 Dataset300 的 0095，背景 252 而字
    只有 180~200）用固定 170 只能取到笔画最黑的核心，字被打碎成 1~2px 的渣，
    再被 `MIN_DIM` 挡掉 ⇒ 整张图几乎全灭。旧版 `textbox` 因为 `MIN_W=1` 还能捡回来
    一点，新版门槛更严就先崩了。
    """
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mode == "fixed":
        thr = ink
    elif mode == "otsu":
        thr = otsu_thr(gray)
    else:  # "max"
        thr = max(ink, otsu_thr(gray))
    thr = int(min(thr, INK_CEIL))
    return np.where(gray < thr, 255, 0).astype(np.uint8)


def line_mask(ink_img: np.ndarray, length: int = LINE_LEN) -> np.ndarray:
    """长直线掩膜（水平或垂直的开运算）。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    h = cv2.morphologyEx(ink_img, cv2.MORPH_OPEN, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    v = cv2.morphologyEx(ink_img, cv2.MORPH_OPEN, k)
    return cv2.bitwise_or(h, v)


def _glyph_ok(w: float, h: float, area: float, min_dim: float = MIN_DIM,
              min_h: float = MIN_H, max_h: float = MAX_H, min_w: float = MIN_W,
              max_w: float = MAX_W, min_area: float = MIN_AREA) -> bool:
    if h < min_h or h > max_h or w < min_w or w > max_w or area < min_area:
        return False
    if min(w, h) < min_dim:
        return False
    fill = area / float(max(1.0, w * h))
    return FILL_LO <= fill <= FILL_HI


def _stitch(boxes: Sequence[Box], gap: float = STITCH_GAP,
            ov: float = STITCH_OV, max_h: float = 0.0) -> List[Box]:
    """把「上下被切断的同一个字」重新缝成一个字号块。只对回收流用。

    `max_h > 0` 时给缝出来的块加**总高上限**：一个字号块不可能比图上
    中位字高的 `STITCH_H_MULT` 倍还高。没有这个闸，一条长导线残留下来的
    碎片会顺着列一路缝下去，缝出高比真值框大 6 倍的「框」（实测
    `高 det/gt` 的 p90 = 6.40）。
    """
    bs = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: List[list] = []
        for b in sorted(bs, key=lambda t: (t[1], t[0])):
            hit = -1
            for i, a in enumerate(out):
                wa, wb = a[2] - a[0], b[2] - b[0]
                if min(wa, wb) <= 0:
                    continue
                xo = min(a[2], b[2]) - max(a[0], b[0])
                if xo < ov * min(wa, wb):
                    continue
                vg = max(a[1], b[1]) - min(a[3], b[3])   # >0 = 上下分开
                if vg > gap:
                    continue
                if max_h > 0 and (max(a[3], b[3]) - min(a[1], b[1])) > max_h:
                    continue
                hit = i
                break
            if hit < 0:
                out.append(list(b))
            else:
                a = out[hit]
                a[0], a[1] = min(a[0], b[0]), min(a[1], b[1])
                a[2], a[3] = max(a[2], b[2]), max(a[3], b[3])
                changed = True
        bs = out
    return [tuple(b) for b in bs]


def _dedup(boxes: Sequence[Box], tol: float = 1.0) -> List[Box]:
    """去掉几乎重合的框。"""
    out: List[Box] = []
    for b in sorted(boxes, key=lambda t: (t[1], t[0])):
        dup = False
        for a in out:
            if (abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
                    and abs(a[2] - b[2]) <= tol and abs(a[3] - b[3]) <= tol):
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def glyphs(bgr: np.ndarray, ink: int = INK, line_len: int = LINE_LEN,
           struct_dim: float = STRUCT_DIM, recover: bool = True,
           resid_min_h: float = RESID_MIN_H, resid_min_dim: float = RESID_MIN_DIM,
           min_dim: float = MIN_DIM,
           min_h: float = MIN_H, max_h: float = MAX_H, min_w: float = MIN_W,
           max_w: float = MAX_W, min_area: float = MIN_AREA,
           ink_mode: str = INK_MODE,
           stats_out: Dict | None = None) -> List[Box]:
    """字形字号块（不是文字行）。独立分量流 + 结构分量的回收流。"""
    im = ink_mask(bgr, ink, ink_mode)
    lines = line_mask(im, line_len)

    n, lab, st, _ = cv2.connectedComponentsWithStats(im, connectivity=8)
    has_line = np.zeros(max(1, n), bool)
    if n > 1:
        hit = np.unique(lab[lines > 0])
        hit = hit[hit > 0]
        has_line[hit] = True

    keep: List[Box] = []
    n_struct = 0
    n_standalone = 0
    struct_mask = np.zeros_like(im)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if has_line[i] and max(w, h) >= struct_dim:
            n_struct += 1
            struct_mask[lab == i] = 255
            continue
        if _glyph_ok(w, h, area, min_dim, min_h, max_h, min_w, max_w, min_area):
            keep.append((float(x), float(y), float(x + w), float(y + h)))
            n_standalone += 1

    n_resid = 0
    if recover and n_struct:
        # 全图字号高度基准：独立流里所有合格块的中位高。用它给「缝合」和
        # 「成行」两个阶段设闸 —— 一个字号块/一行字不可能比它高很多。
        hs = sorted(b[3] - b[1] for b in keep)
        med_h = hs[len(hs) // 2] if hs else 0.0
        max_h_stitch = STITCH_H_MULT * med_h if med_h > 0 else 0.0
        resid = cv2.bitwise_and(struct_mask, cv2.bitwise_not(lines))
        n2, _l2, st2, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
        frags: List[Box] = []
        for i in range(1, n2):
            x, y, w, h, area = (int(v) for v in st2[i])
            # 回收流只卡高度/短边/面积，**不卡填充率上界** —— 碎片本来就可能是实心的一小块
            if h < resid_min_h or min(w, h) < resid_min_dim or area < min_area:
                continue
            if w > max_w or h > max_h:
                continue
            if area / float(max(1, w * h)) < FILL_LO:
                continue
            frags.append((float(x), float(y), float(x + w), float(y + h)))
        frags = _stitch(frags, max_h=max_h_stitch)
        n_resid = len(frags)
        keep.extend(frags)

    keep = _dedup(keep)
    if stats_out is not None:
        allh = sorted(b[3] - b[1] for b in keep)
        stats_out.update(n_comp=max(0, n - 1), n_struct=n_struct,
                         n_standalone=n_standalone, n_resid=n_resid,
                         n_glyph=len(keep),
                         med_h=(allh[len(allh) // 2] if allh else 0.0))
    return keep


def group_lines(glyphs_in: Sequence[Box], gap_frac: float = GAP_FRAC,
                v_overlap: float = V_OVERLAP, h_ratio: float = H_RATIO,
                max_w: float = MAX_LINE_W, max_h: float = 0.0) -> List[Box]:
    """把字形并成「串」（文字行）。左→右扫描，挂到间距最小的兼容组上。

    与 `textbox._merge_lines` 的三处不同：
      ① 间距门槛用 `gap_frac * min(h_i, h_j)`（旧版用 `0.9 * max`，把相邻两个串也并了）
      ② 高度比只比**字形本身**，不比累计组高
      ③ 组宽超过 `max_w` 不再吸收

    `max_h > 0` 时再卡一条：并出来的**组高**不得超过 `max_h`
    （= `STITCH_H_MULT * 全图中位字高`）。上面 ② 只挡住「高低悬殊的两个字形」，
    挡不住一串高度相近的字形顺着子列一路并下去拼出 6 倍高的框。
    """
    boxes = [list(b) for b in sorted(glyphs_in, key=lambda t: (t[1], t[0]))]
    groups: List[Dict] = []
    for b in boxes:
        hb = b[3] - b[1]
        best = -1
        best_gap = None
        for i, g in enumerate(groups):
            ha = g["h"]
            if min(hb, ha) <= 0 or max(hb, ha) / min(hb, ha) > h_ratio:
                continue
            ov = min(g["y2"], b[3]) - max(g["y1"], b[1])
            if ov < v_overlap * min(hb, ha):
                continue
            gap = max(g["x1"], b[0]) - min(g["x2"], b[2])
            if gap > gap_frac * min(hb, ha):
                continue
            if max(g["x2"], b[2]) - min(g["x1"], b[0]) > max_w:
                continue
            if max_h > 0 and (max(g["y2"], b[3]) - min(g["y1"], b[1])) > max_h:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
        if best < 0:
            groups.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "h": hb})
        else:
            g = groups[best]
            g["x1"] = min(g["x1"], b[0])
            g["y1"] = min(g["y1"], b[1])
            g["x2"] = max(g["x2"], b[2])
            g["y2"] = max(g["y2"], b[3])
            g["h"] = max(g["h"], hb)
    return [(g["x1"], g["y1"], g["x2"], g["y2"]) for g in groups]


GROUP_KEYS = ("gap_frac", "v_overlap", "h_ratio", "max_w", "max_h")


def detect_boxes(bgr: np.ndarray, **kw) -> List[Box]:
    """返回文字**串**的紧框，图像坐标（左上原点）。

    `**kw` 里属于 `group_lines` 的（`gap_frac` / `v_overlap` / `h_ratio` /
    `max_w` / `max_h`）被分出来，其余透传给 `glyphs`。
    没显式给 `max_h` 时，用 `glyphs` 报上来的全图中位字高乘 `STITCH_H_MULT`。
    """
    gkw = {k: kw.pop(k) for k in list(kw) if k in GROUP_KEYS}
    stat: Dict = {}
    g = glyphs(bgr, stats_out=stat, **kw)
    if "max_h" not in gkw:
        med = float(stat.get("med_h") or 0.0)
        gkw["max_h"] = STITCH_H_MULT * med if med > 0 else 0.0
    return group_lines(g, **gkw)


def debug(bgr: np.ndarray, **kw) -> Dict:
    """给诊断图用：返回各流的分段计数。"""
    stat: Dict = {}
    g = glyphs(bgr, stats_out=stat, **kw)
    stat["n_line"] = len(group_lines(g))
    return stat


# ------------------------------------------------------------------ 自检

def _render_rect(m: np.ndarray, x1, y1, x2, y2) -> None:
    m[int(y1):int(y2), int(x1):int(x2)] = 255


def _glyph(m: np.ndarray, x, y, w, h, wall: int = 2) -> None:
    """画一个「口」字形（2px 壁）。实心块的填充率是 1.0，会被 FILL_HI 当成符号挡掉。"""
    _render_rect(m, x, y, x + w, y + wall)
    _render_rect(m, x, y + h - wall, x + w, y + h)
    _render_rect(m, x, y, x + wall, y + h)
    _render_rect(m, x + w - wall, y, x + w, y + h)


def _selftest() -> int:
    """7 条合成用例。本项目已在「评测函数没自检」上栽过四次，这是第五个必须带的。"""
    fails: List[str] = []

    def case(name, img, want_boxes, **kw):
        got = detect_boxes(img, **kw)
        ok = len(got) == want_boxes
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: 期望 {want_boxes} 框，得到 {len(got)}"
              + ("" if ok else " -> " + str([tuple(round(v) for v in b) for b in got])))
        if not ok:
            fails.append(name)

    # ① 两个字形（"R" "1"）→ 1 框
    m = np.zeros((60, 120), np.uint8)
    _glyph(m, 20, 20, 6, 16)
    _glyph(m, 30, 20, 6, 16)
    case("① 两个字形并成一串", 255 - m, 1)

    # ② 长导线 + 挂在它上面的锯齿 → 0 框
    m = np.zeros((60, 200), np.uint8)
    _render_rect(m, 10, 30, 190, 32)
    for k in range(6):
        _render_rect(m, 60 + k * 6, 22 - (k % 2) * 8, 64 + k * 6, 30 - (k % 2) * 8)
    case("② 挂在导线上的锯齿不是文字", 255 - m, 0)

    # ③ 长导线旁边的独立文字 → 1 框
    m = np.zeros((80, 200), np.uint8)
    _render_rect(m, 10, 60, 190, 62)
    _glyph(m, 20, 20, 6, 16)
    _glyph(m, 30, 20, 6, 16)
    case("③ 导线旁边的独立文字", 255 - m, 1)

    # ④ 文字压在导线上（回收流 + 缝合应还原成一个字）→ 1 框
    m = np.zeros((80, 200), np.uint8)
    _render_rect(m, 10, 30, 190, 32)
    _glyph(m, 20, 20, 6, 24)
    _glyph(m, 34, 20, 6, 24)
    case("④ 压在导线上的文字", 255 - m, 1)

    # ⑤ 两个相隔很远的串 → 2 框
    m = np.zeros((60, 300), np.uint8)
    _glyph(m, 20, 20, 6, 16)
    _glyph(m, 30, 20, 6, 16)
    _glyph(m, 200, 20, 6, 16)
    _glyph(m, 210, 20, 6, 16)
    case("⑤ 两个相隔很远的串", 255 - m, 2)

    # ⑥ 高度差太大不并成一个串 → 2 框
    m = np.zeros((80, 200), np.uint8)
    _glyph(m, 20, 40, 8, 8)
    _glyph(m, 40, 10, 6, 30)
    case("⑥ 高度差大不并成一个串", 255 - m, 2)

    # ⑦ 图纸边框（四条长直线）→ 0 框
    m = np.zeros((120, 160), np.uint8)
    _render_rect(m, 0, 0, 160, 3)
    _render_rect(m, 0, 117, 160, 120)
    _render_rect(m, 0, 0, 3, 120)
    _render_rect(m, 157, 0, 160, 120)
    case("⑦ 图纸边框不算文字", 255 - m, 0)

    print("\n自检：7 项全部通过" if not fails else f"\n自检 {len(fails)} 项失败")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
