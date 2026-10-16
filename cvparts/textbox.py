"""纯 CV 文字**定位**（第一阶段的第二个专类检测器）。

用户 m03802 的第三条指令：文字的内容识别属第二阶段，但**文字的检测/定位属第一阶段**。
而且它不只是"顺便做做"——下面这条链子是本项目的主干：

    文字框 = 导线端点的第二类锚点
      → 有了锚点，"悬空导线"才判得出来（只有元件框一个锚点时，jlc 风格全是假悬空）
      → 悬空才变成可用的召回/去噪信号（用户 m03802 第二条指令）
      → 同时文字框可以反过来遮挡掉导线提取里最大的一坨假阳性（README 已记的 7 条限制之首）

做法（不训练、纯形态学，理由：文字笔画和导线一样是 1px 细线，连通域形状分不开，
必须先把"长直线"从墨迹里减掉）：
    1. ink = gray < INK
    2. 用 21px 的水平/垂直结构元做开运算取出长直线，从 ink 里减掉
    3. 对残余墨迹做连通域，按尺寸/填充率筛"字形"颗粒
    4. 按水平间距与高度重叠把颗粒并成"文字行"

自检口径（用官方 220 张的 GT，不依赖任何非官方文件）：
    GT 里每个 pin 都有 `pinname`（例如 "GPIO0"）和绘制点 `point`，
    引脚名一定是画在引脚点附近的文字。
    ⇒ 「引脚名覆盖率」= 引脚点附近 tol 内存在文字框的比例，是对文字检测的**官方数据召回下界**。
    只统计 type=='box' 的元件、且 pinname 不是 `P_<数字>` 的引脚
    （`P_31` 是 gnd/v 这类符号的内部占位名，纸上并没有这几个字）。

用法：
    python eda-pcb\\cvparts\\textbox.py            # 全量自检
    python eda-pcb\\cvparts\\textbox.py 0001 0011  # 只跑这几张，并渲染叠加图到 work\\preview
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
EDA = HERE.parent
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))

Box = Tuple[float, float, float, float]

def _env_int(name: str, default: int) -> int:
    """§5.207.5：让交付链路能整体 A/B 墨迹阈值（默认 170 = 历史行为）。"""
    try:
        return int(os.environ.get(name, "") or default)
    except Exception:
        return default


INK = _env_int("EDA_TEXTBOX_INK", 170)            # gray < INK 算墨
LINE_LEN = 21        # 开运算结构元长度：>= 它的连续笔画会被当"长直线"减掉
MIN_H, MAX_H = 4, 34     # 字形高度范围
MIN_W, MAX_W = 1, 70     # 字形宽度范围
MIN_AREA = 5
FILL_LO, FILL_HI = 0.06, 0.98
# 并成文字行
GAP_FRAC = 0.9       # 水平间距 <= GAP_FRAC * 行高 才并
V_OVERLAP = 0.45     # 垂直重叠比例下限
H_RATIO = 1.9        # 同行内高度比上限
PLACEHOLDER = re.compile(r"^P_\d+$")


# ------------------------------------------------------------------ 检测

def _long_lines(ink: np.ndarray, length: int = LINE_LEN) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    return cv2.bitwise_or(h, v)


def _glyphs(ink_wo: np.ndarray) -> List[Box]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink_wo, connectivity=8)
    out: List[Box] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (MIN_H <= h <= MAX_H and MIN_W <= w <= MAX_W and area >= MIN_AREA):
            continue
        fill = area / float(max(1, w * h))
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


def _merge_lines(g: Sequence[Box]) -> List[Box]:
    """把字形颗粒按 x 方向并成文字行（贪心，先按 y 再按 x）。"""
    boxes = [list(b) for b in g]
    merged = True
    while merged:
        merged = False
        boxes.sort(key=lambda b: (b[1], b[0]))
        out: List[list] = []
        for b in boxes:
            hit = -1
            for i, a in enumerate(out):
                hb, ha = b[3] - b[1], a[3] - a[1]
                if min(hb, ha) <= 0 or max(hb, ha) / min(hb, ha) > H_RATIO:
                    continue
                ov = min(a[3], b[3]) - max(a[1], b[1])
                if ov < V_OVERLAP * min(hb, ha):
                    continue
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if gap > GAP_FRAC * max(ha, hb):
                    continue
                hit = i
                break
            if hit < 0:
                out.append(list(b))
            else:
                a = out[hit]
                a[0], a[1] = min(a[0], b[0]), min(a[1], b[1])
                a[2], a[3] = max(a[2], b[2]), max(a[3], b[3])
                merged = True
        boxes = out
    return [tuple(b) for b in boxes]


def detect_boxes(bgr: np.ndarray, min_h: int = MIN_H, max_h: int = MAX_H) -> List[Box]:
    """返回文字**行**的框，图像坐标（左上原点）。"""
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = np.where(gray < INK, 255, 0).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    g = _glyphs(cv2.bitwise_and(ink, cv2.bitwise_not(_long_lines(ink))))
    keep = [b for b in g if min_h <= (b[3] - b[1]) <= max_h]
    return _merge_lines(keep)


def text_mask(shape, boxes: Sequence[Box], pad: float = 1.0) -> np.ndarray:
    """把文字框涂成掩膜（给导线提取当 occlude 用）。"""
    m = np.zeros(shape[:2], np.uint8)
    for b in boxes:
        cv2.rectangle(m, (int(b[0] - pad), int(b[1] - pad)), (int(b[2] + pad), int(b[3] + pad)), 255, -1)
    return m


# ------------------------------------------------------------------ 自检

def _main() -> None:
    from schematic_parser.data import discover, DEFAULT_DATA_ROOT

    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    cases = list(discover(DEFAULT_DATA_ROOT))
    if want:
        cases = [c for c in cases if c.case_id in want or c.case_id.lstrip("0") in
                 [w.lstrip("0") for w in want]]
    outdir = EDA.parent / "work" / "preview"
    outdir.mkdir(parents=True, exist_ok=True)

    tols = (12.0, 20.0, 30.0, 40.0)
    hit = {t: 0 for t in tols}
    n_pin = 0
    n_box = 0
    n_txt = 0
    t_tot = 0.0
    per_case = []
    for c in cases:
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        t0 = time.perf_counter()
        boxes = detect_boxes(bgr)
        dt = time.perf_counter() - t0
        t_tot += dt
        n_txt += len(boxes)
        H = bgr.shape[0]
        doc = json.load(open(c.json_path, encoding="utf-8"))
        # GT 引脚点 -> 图像坐标
        pts = []
        for key, gp in (doc.get("pins") or {}).items():
            comp = (doc.get("components") or {}).get(key)
            if not comp or comp.get("type") != "box":
                continue
            for p in gp.values():
                name = str(p.get("pinname") or "")
                if PLACEHOLDER.match(name) or not name:
                    continue
                pt = p.get("point") or {}
                pts.append((float(pt.get("x", 0)), H - float(pt.get("y", 0))))
        n_pin += len(pts)
        n_box += sum(1 for x in (doc.get("components") or {}).values() if x.get("type") == "box")
        arr = np.asarray(boxes, dtype=np.float64) if boxes else np.zeros((0, 4))
        ch = [0] * len(tols)
        if arr.shape[0]:
            cx = (arr[:, 0] + arr[:, 2]) / 2.0
            cy = (arr[:, 1] + arr[:, 3]) / 2.0
            for px, py in pts:
                dx = np.maximum(np.maximum(arr[:, 0] - px, px - arr[:, 2]), 0.0)
                dy = np.maximum(np.maximum(arr[:, 1] - py, py - arr[:, 3]), 0.0)
                d = np.hypot(dx, dy)
                d = np.minimum(d, np.hypot(cx - px, cy - py))
                for i, t in enumerate(tols):
                    if d.min() <= t:
                        ch[i] += 1
        for i, t in enumerate(tols):
            hit[t] += ch[i]
        per_case.append((c.case_id, len(boxes), len(pts), ch[1] / max(1, len(pts)), dt))

        if want:
            vis = bgr.copy()
            for b in boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            for px, py in pts:
                cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)
            p = outdir / f"textbox_{c.case_id}.png"
            cv2.imwrite(str(p), vis)
            print(f"  {c.case_id:8s} 文字框 {len(boxes):4d}  引脚 {len(pts):4d}  "
                  f"@20px {ch[1]/max(1,len(pts)):.3f}  {dt*1000:6.1f} ms -> {p}")

    print(f"\n=== 文字定位自检（{len(cases)} 张官方图）===")
    print(f"  文字行框合计 {n_txt}   平均 {n_txt/max(1,len(cases)):.1f} 框/张")
    print(f"  耗时合计 {t_tot:.1f} s   平均 {t_tot/max(1,len(cases))*1000:.1f} ms/张（CPU 单线程）")
    print(f"  GT 引脚名样本 {n_pin} 个（来自 {n_box} 个 box 元件，已排除 P_<n> 占位名）")
    for t in tols:
        print(f"    引脚名覆盖率 @{t:4.0f}px : {hit[t]/max(1,n_pin):.4f}")
    per_case.sort(key=lambda r: r[3])
    print("  最差 8 张（按 @20px 覆盖率）:")
    for cid, nb, npin, cov, dt in per_case[:8]:
        print(f"    {cid:10s} 文字框 {nb:4d}  引脚 {npin:4d}  覆盖 {cov:.3f}  {dt*1000:6.1f} ms")
    per_case.sort(key=lambda r: -r[3])
    print("  最好 5 张:")
    for cid, nb, npin, cov, dt in per_case[:5]:
        print(f"    {cid:10s} 文字框 {nb:4d}  引脚 {npin:4d}  覆盖 {cov:.3f}  {dt*1000:6.1f} ms")


if __name__ == "__main__":
    _main()
