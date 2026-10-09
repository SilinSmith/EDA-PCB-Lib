# -*- coding: utf-8 -*-
"""第二阶段 · 文字读入：文字框定位（复用 `cvparts.textbox`）＋ CRNN 识别。

角色条件读法：同一张裁片会在 5 个角色下各读一遍，交给 `assign.py` 按几何 + 字符串
模式挑最合适的那一次。原因是「先要知道角色才能读对」与「先要读出来才知道角色」
互相依赖——用「全都读一遍再选」把鸡生蛋问题绕过去；一张图几百个框，代价可忽略。

识别器缺席（没有权重）时不报错：返回空读法，管线仍能输出结构完整的 JSON，
只是所有 `key` 退化成占位符。这样评测器能直接量出「OCR 值多少分」。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

Box = Tuple[float, float, float, float]
ROLES: Tuple[str, ...] = ("designator", "value", "pin_name", "pin_number", "net_label")


def _default_ckpt() -> Path:
    """CRNN 权重：**不再写死 `Y:\\workspace\\...`**（进容器必炸）。

    顺序：`STAGE2_CKPT` 环境变量 -> `eda-pcb/weights/textrec_best.pt`（随仓库提交）
    -> `EDA/work/ocr_mix/runs/best.pt`（本机训练产物）。详见 `assets.py`。

    ★ 2026-10-11：兜底路径从 `ocr_real`(e51) 改成 `ocr_mix`(e57) —— `weights/textrec_best.pt`
    也一起换成了 e57（两把尺子都更好：官方 9 张 57.26→62.80，派生 80 张 23.81→26.14，
    见 `docs/DIAGNOSTICS.md`「交付 CRNN 权重陈旧」）。旧权重留在
    `work/weights_backup/textrec_best.real_e51.pt`（= `work/ocr_real/runs/best.pt`）。
    """
    try:
        from assets import textrec_ckpt
    except ImportError:                       # 以别的方式被导入时退回老路径
        return Path(os.environ.get(
            "STAGE2_CKPT", r"Y:\workspace\EDA\work\ocr_mix\runs\best.pt"))
    return Path(textrec_ckpt(must_exist=False))


# 训练产物原本落在工作目录（`python -m textrec.train --out ...`），不在包内；
# 已复制一份到 `weights/textrec_best.pt` 随仓库提交。可用 `STAGE2_CKPT` 覆盖。
DEFAULT_CKPT = _default_ckpt()
CHUNK = 256

# 文字框是**按阈值墨迹核心**取出来的，系统性比真值框小（中位 IoU 0.36）。
# 实测（`work/_boxgeom.py`，10 张 CVJsonStd，1103 个配对，配置 `TEXTSEG_GAP=1.2`）：
#   +0 0.1915 · +2 0.6345 · **+3 0.6544** · +4 0.6105 · +5 0.5076 · +6 0.2554 · +8 0.0527
# 而「直接拿真值框原样读」是 **0.8430**。外扩只用于裁剪，几何归属仍用原框（见 `read`）。
# `pad_frac` 那条在 h≈19 时会给到 5.7px —— **已经越过 +3 的峰**，所以必须加上限。
#
# ★ 端到端扫出来的最优（2026，`work\tools\eval\fast.py --no-vlm`，`STAGE1_WIRE_COLOUR=off`）：
#     H = +2 定长（不给比例）   V = max(3, 0.20h)
#   10GT(9 张) 49.3516 vs 老默认 48.5981（**+0.75**）；19 张 29.4111 vs 28.9338（**+0.48**）。
#   ⇒ 两套独立样本同号，才敢把默认从 `2.0/0.30 · 2.0/0.30` 换过来。
#   老默认的垂直项在 h≈19 时给 5.7px，越过了「+3 最优」的峰；水平比例项在现有框尺寸下
#   完全不起作用（`2/0.10` 与 `2/0` 同分 49.3516）。
#   ⚠️ 不是全面变好：10GT(9) 与 19 张逐图都出现单张倒退（19 张里 case 1 −4.56，被 case 7
#   +7.28 抹平）。9 张上的 +0.75、19 张上的 +0.48 都偏小，**别当成精确最优点**，
#   它只是这一段平台里量到最高的那个；换默认前请复核 220 张。
#   网格（9 张）：`padmin3f0` 47.61｜`pad13`(1,3) 48.09｜`pad23`(2,3) 48.96｜**`pad23y20` 49.35**｜
#                `pad23y25` 49.05｜`pad253`(2.5,3) 48.57｜`pad24`(2,4) 47.20｜`pad32`(3,2) 44.12。
PAD_MIN = float(os.environ.get("STAGE2_PAD_MIN", "2.0"))   # 定长外扩下限（水平）
PAD_FRAC = float(os.environ.get("STAGE2_PAD_FRAC", "0.0"))  # 按框高比例外扩（水平）
PAD_MAX = float(os.environ.get("STAGE2_PAD_MAX", "99.0"))  # 外扩上限（99 = 不设上限）
# ★ **垂直**外扩单独一个旋钮。
# 动机（10GT case 10 `U1` 实测）：把检出框「各向外扩 3px」再读，同一块 `IS` 从
# `'IS'`（conf 0.816）变成 `'V5'`，而把上下也扩出去会把**邻行的字**拉进来
# （`GND` 框 +6px 读出 `12K` —— 那是它下面那颗电阻的值）。水平外扩则只有好处：
# 检出框系统性比真值小（中位 IoU 0.36），而**丢的是首/末字符**（`IS`→`S`、`FB`→`F`、
# `GND`→`D`）。所以水平与垂直应当分开扫。
# 垂直不能低于 3（`pad32` 只给 2 ⇒ 44.12，掉 5 分）。
PAD_FRAC_Y = float(os.environ.get("STAGE2_PAD_FRAC_Y", "0.20"))  # 垂直比例
PAD_MIN_Y = float(os.environ.get("STAGE2_PAD_MIN_Y", "3.0"))     # 垂直下限

# `merge_boxes_both` 里「多大的重合算同一个框、不必再加一份」的阈值。
# **默认 0.65 是扫出来的、且不是单调的**（220 张实测 23.2632/0.35 · 23.3815/0.50 ·
# 23.5009/0.65 · 23.3294/0.80 · 23.2735/0.95）。整个区间只有 0.24 的起伏，
# 其中有相当一部分是噪声，所以**别把这个 0.65 当成精确的最优点**：
# 它只是「0.5~0.8 这一段平台」里量到最高的那个。详见 `merge_boxes_both`。
BOTH_TOL = float(os.environ.get("STAGE2_BOTH_TOL", "0.65"))

# `merge_boxes_multi` 里「外扩版」的外扩量：左右 = 框宽 × MULTI_W，上下 = 框高 × MULTI_H。
# 取自 `work\tools\diag\_padsweep.py` 的网格扫描（10 张 / 1287 个真值框，公共口径）：
# 检出框原样 0.1915 → (0.15w, 0.43h) 0.6281 → (0.20w, 0.50h) 0.6121 → 再大就掉。
# **不追求精确最优**：网格整个平台只有 0.03 的起伏，而**两份都留着让 assign 挑**
# 比任何单一取值都稳（见 `merge_boxes_multi`）。
MULTI_W = float(os.environ.get("STAGE2_MULTI_W", "0.15"))
MULTI_H = float(os.environ.get("STAGE2_MULTI_H", "0.43"))

# ★ 引脚号专用：读之前先把**有色的描边**从框里切掉。
#
# 病灶（官方 10GT case 10 的 `U1`，肉眼核对过）：引脚 4 (`VIA`) 是**未连接脚**，图上画着一个
# **绿色的 no-connect 叉**；文字检测给出的框是 `[468,462,497,480]`（宽 29px），把叉和数字 `4`
# 圈在同一个框里 ⇒ CRNN 把叉读成 `K`、交出 **`K4`**（GT 是 `4`）。case 8 更重：`14→L14`、
# `10→L10`、`2→X1`、`6→K8`、`1→1L` …——「框宽 25~29px 却只该有一个数字」的错**全是这一条**。
#
# 判据是**颜色**，不是几何：这些图里绿色 = 导线与 no-connect 叉、暗红 = 元件框、蓝 = 位号/型号，
# 而**引脚号是黑字**。所以「低饱和（max-min 通道差 < SAT）且暗（灰度 < DARK）」的像素就是字。
# 取这些像素的紧致外接框，再照常 `pad_box`（**外扩是承重的，不能省**：不 pad 时 exact 只有
# 0.44）。实测（`work\tools\diag\_numread.py`，官方 9 张 **134 个能对上 GT 脚点的引脚号读数**）：
#     基线 0.6866（1字 0.7700 / 2字 0.4333 / 3字+ 0.5000）
#     切描边 0.7090（0.7700 / **0.5333** / 0.5000）      ⇒ **+0.0224，且 2 字桶 +0.10**
#   阈值扫描：`(60,160)` 0.7090 · `(50,140)` 0.7015 · `(80,170)` 0.6940 ⇒ 取 (60,160)。
#   ⚠️ 只对 **`pin_number`** 生效，且垂直比例项归零（`PAD_FRAC_Y=0.20` 在 h≈19 时给到 3.8px
#   之外还越过了「+3 峰」；同一个网格里 `(2,0,3,0)` 0.6866 > `cur(2,0,3,.20)` 0.6716）。
#
# ★ **端到端 A/B（官方 10GT 9 张 / 派生 19 张，`FUSION_LLM=0`）**
#   基线（不切描边）         off9 **56.7090**  pin 0.3548        d19 **35.3940**  pin 0.1811
#   只切 `pin_number`        off9  56.8493   pin 0.3632        d19  35.4898   pin 0.1850
#   切 `pin_number,pin_name` off9 **57.5417** pin **0.3920**   d19 **35.7504** pin **0.1987**  ★ 采纳
#   切全部 5 个角色           off9  55.6317   pin 0.3802（comp 0.9403 → **0.9025**）⇒ **不许**：
#     位号/型号是**蓝字**，颜色假设不成立，切完反而把字切坏；而引脚名是黑字，跟着引脚号一起赚。
#   加 `net_label` 与不加逐位相同（这些图里没有网络标签的文字）。
#   关掉开关时逐位复现基线（`work\_fast_base\off9_ptight0.json` == `off9_numd.json`）。
PINNUM_TIGHT = float(os.environ.get("STAGE2_PINNUM_TIGHT", "1")) > 0
PINNUM_TIGHT_SAT = float(os.environ.get("STAGE2_PINNUM_TIGHT_SAT", "60"))
PINNUM_TIGHT_DARK = float(os.environ.get("STAGE2_PINNUM_TIGHT_DARK", "160"))
PINNUM_PAD_FRAC_Y = float(os.environ.get("STAGE2_PINNUM_PAD_FRAC_Y", "0.0"))
# 哪些**角色**改用「切掉有色描边」的裁片。默认 `pin_number,pin_name`（量过、两档都赚）。
# ⚠️ 别把 `designator` / `value` 加进来：它们是蓝字，切完 component f1 掉 3.8 个点。
TIGHT_ROLES = tuple(r.strip() for r in
                    os.environ.get("STAGE2_TIGHT_ROLES", "pin_number,pin_name").split(",")
                    if r.strip()) if PINNUM_TIGHT else ()


def tight_ink_box(bgr: np.ndarray, box: Box,
                  sat: float = None, dark: float = None) -> Optional[Box]:
    """框内**低饱和暗墨**的紧致外接框；框里没有这类像素时返回 `None`。

    纯几何用途：把「同框里混进来的有色描边」排除后再裁。**不改框本身**（`read` 返回的
    `bbox` 仍是原框），只改喂给识别器的那一张裁片。灰度图输入直接返回 `None`。
    """
    if bgr is None or getattr(bgr, "ndim", 2) != 3:
        return None
    sat = PINNUM_TIGHT_SAT if sat is None else sat
    dark = PINNUM_TIGHT_DARK if dark is None else dark
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(int(bgr.shape[1]), x2); y2 = min(int(bgr.shape[0]), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    mx = sub.max(2)
    mn = sub.min(2)
    g = sub.mean(2)
    m = ((mx - mn) < sat) & (g < dark)
    on = np.nonzero(m.sum(0) > 0)[0]
    if len(on) == 0:
        return None
    oy = np.nonzero(m[:, int(on[0]):int(on[-1]) + 1].sum(1) > 0)[0]
    if len(oy) == 0:
        return None
    return (float(x1 + on[0] - 1), float(y1 + oy[0] - 1),
            float(x1 + on[-1] + 2), float(y1 + oy[-1] + 2))


def pad_box(box: Box, pad_min: float = PAD_MIN, pad_frac: float = PAD_FRAC,
            pad_max: float = PAD_MAX,
            pad_min_y: float = None, pad_frac_y: float = None) -> Box:
    """按框高比例外扩，**水平与垂直可以分开**（`*_y` 不给则与水平相同）。"""
    h = max(1.0, float(box[3]) - float(box[1]))
    px = min(pad_max, max(pad_min, pad_frac * h))
    if pad_min_y is None and pad_frac_y is None:
        py = px
    else:
        py = min(pad_max, max(pad_min if pad_min_y is None else pad_min_y,
                              (pad_frac if pad_frac_y is None else pad_frac_y) * h))
    px = max(0.0, px)
    py = max(0.0, py)
    return (float(box[0]) - px, float(box[1]) - py,
            float(box[2]) + px, float(box[3]) + py)


def imread_unicode(path) -> np.ndarray:
    """`cv2.imread` 读不了非 ASCII 路径（Dataset300 目录名带中文），必须走 imdecode。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("读不出图像: %s" % path)
    return img


def read_stage1_texts(stage1_doc: dict) -> List[Box]:
    out: List[Box] = []
    for t in stage1_doc.get("texts") or []:
        b = t.get("bbox_img") if isinstance(t, dict) else t
        if b and len(b) == 4:
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    return out


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def merge_boxes(base: List[Box], det: List[Box], min_iou: float = 0.25,
                max_grow: float = 1.25) -> List[Box]:
    """**保底再收紧**：以 `base`（stage1 `textbox` 的框，召回高但偏小）为准，
    每个 base 框如果能在 `det`（`textseg` 的框，紧贴墨迹、裁字准）里找到一个
    「IoU >= `min_iou` 或中心落在 base 内」且**面积不超过 `max_grow` 倍**的对应框，
    就用 det 那个；找不到就**原样保留 base**。

    动机（220 张端到端实测）：`textseg` 单独用比 `textbox` **差**
    （component F1 0.0762 vs 0.0846），因为它把「连通分量里带长直线」的整块
    当结构分量丢掉，**少切了约 8% 的文字框**（命中 1152/1287 vs 1248/1287）；
    丢一个位号 = 该元件 key 对不上 = 同时吃一个 fp 和一个 fn，代价远大于
    「框紧一点、OCR 准一点」的收益。所以要把两者的长处拼起来。
    """
    out: List[Box] = []
    used = set()
    for b in base:
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        best, bj = 0.0, -1
        for j, d in enumerate(det):
            if j in used or _area(d) > max_grow * _area(b) + 1e-6:
                continue
            inside = (b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
                      and d[0] <= cx <= d[2] and d[1] <= cy <= d[3])
            v = _iou(b, d)
            if (inside or v >= min_iou) and (v > best or (inside and bj < 0)):
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            out.append(det[bj])
        else:
            out.append(b)
    return out


def merge_boxes_both(base: List[Box], det: List[Box]) -> List[Box]:
    """`merge_boxes` 的结果 **再加上一部分被它换掉的 base 框**。

    动机（2025-10-04，`STAGE2_DBG_CID` 实测 `0001 c0052`）：`union` 是**替换** ——
    某个位置只留 `textseg` 那一版。但 `textseg` 的切法**在这一版 CRNN 上并不总是
    读得更好**：`[603,856,620,864]`（stage1 框）的 `pin_name` 角色读出 `TP1`（真值），
    而 `[602,855,619,863]`（textseg 框，只差 1px，IoU 0.82）五个角色全是
    `11`/`O`/`C1` 这种垃圾。框只差一个像素、读数天差地别。

    「加多少」是**扫出来的**，不是拍的（220 张，官方口径管线，`--stage1 stage1_v6`）：
    | `BOTH_TOL` | 加了哪些 base 框 | 总分 |
    |---|---|---:|
    | `union` 本身 | 一个都不加 | 22.9068 |
    | 0.35 | 只有 IoU 在 [0.25, 0.35) 的 | 23.2632 |
    | 0.50 | IoU 在 [0.25, 0.50) 的 | 23.3815 |
    | **0.65（默认）** | IoU 在 [0.25, 0.65) 的 | **23.5009** |
    | 0.80 | IoU 在 [0.25, 0.80) 的 | 23.3294 |
    | 0.95（全加） | 所有被换掉的 | 23.2735 |
    **不单调，峰值在 0.65 附近**，整个区间只起伏 0.24 —— 其中相当一部分是噪声
    （220 张里 0.1 分约等于一颗元件），所以别把 0.65 当精确最优点，它是那段平台里
    量到最高的一个。机制上讲得通：IoU 很大的两个框是**同一裁片的两种近似版本，
    OCR 结果高度相关**（一起对或一起错），多出来那份只会多给 `assign` 一个抢分机会
    （`_best_read` 还允许串角色借读，`pref=0.85`），净添噪声；IoU 适中（0.25~0.65）
    的那些是**位置确有差异的裁片**，能互补。
    """
    tol = BOTH_TOL
    out = list(merge_boxes(base, det))
    for b in base:
        if not any(_iou(b, o) >= tol for o in out):
            out.append(b)
    return out


def grow_box(b: Box, fx: float = None, fy: float = None) -> Box:
    """按**自身宽/高**的比例往四边外扩。fx 用框宽、fy 用框高。"""
    fx = MULTI_W if fx is None else fx
    fy = MULTI_H if fy is None else fy
    w = max(1.0, b[2] - b[0])
    h = max(1.0, b[3] - b[1])
    return (b[0] - fx * w, b[1] - fy * h, b[2] + fx * w, b[3] + fy * h)


def merge_boxes_multi(base: List[Box], det: List[Box]) -> List[Box]:
    """`merge_boxes_both` 的结果 **再加上每个检出框的「外扩版」**。

    动机（2025-10-05 实测，`work\\tools\\diag\\_padsweep.py`，10 张 / 1287 个真值框）：

    | 裁片 | OCR exact |
    |---|---:|
    | 真值紧框原样 | **0.8430** |
    | 真值紧框 + 任何外扩 | 0.09~0.20（密集排布下外扩把邻行拉进来，CTC 多读 ⇒ 归零） |
    | `textseg` 检出框原样 | 0.1915 |
    | `textseg` 检出框 + 0.15w/0.43h | **0.6281**（全网格最高 0.6544） |

    ⇒ **两个版本各有各的赢面，没有哪个统一更好**：检出框的几何比例并不稳定
    （高比中位 0.54、宽比中位 0.77，p10~p90 跨 0.47~0.78），所以「外扩多少」对一个框
    过、对另一个就毁。既然如此就别在 `text.py` 里二选一 —— **两份都留着**，让
    `assign._score`（`fit×(0.4+0.6·conf) − 0.035·d`）去挑：两个框中心相同 ⇒ 距离项
    相同 ⇒ 选择完全由「读数是否像位号/值」和「置信度」决定，正是我们要的信号。

    **⚠ 220 张端到端实测：这条路是负的**（`work\\tools\\eval\\stage2_eval.py`）：
    `--texts multi` **23.0447** vs `--texts both` **23.5009**，墙钟 355.8 s vs 189.4 s。
    拆开看是**框数一多、噪声就赢**：`text_mismatch 1029 → 1190`、`fp 4458 → 4623`。
    机制上讲得通：`assign` 的选择不只发生在「同一个位置的两个框之间」，还有**全局
    位号去重**和跨元件的抢读 —— 多出来的那份裁片会去跟别的位置的候选抢，赢下来就是
    一个错。所以「让打分器挑」只在**候选池本身干净**时成立，这里不成立。
    **默认仍是 `both`**，本函数留着只作对照（`--texts multi`）。
    """
    out = list(merge_boxes_both(base, det))
    for b in det:
        g = grow_box(b)
        if not any(_iou(g, o) >= 0.92 for o in out):
            out.append(g)
    return out


def detect_texts(bgr: np.ndarray, min_h: int = 4, max_h: int = 34,
                 engine: str = "textseg") -> List[Box]:
    """文字串定位。默认走 `cvparts.textseg`（为「精确裁字」调的）。

    `engine="textbox"` 可退回旧检测器（它当初是为「引脚名覆盖率」调的，
    框系统性偏小、且会把元件图形符号当文字）。实测公共子集 OCR exact：
    `textbox` + pad 0.40 vs `textseg` + pad **0.65**。
    """
    if engine == "textbox":
        from cvparts import textbox
        return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                for b in textbox.detect_boxes(bgr, min_h=min_h, max_h=max_h)]
    from cvparts import textseg
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b in textseg.detect_boxes(bgr)]


class TextEngine:
    """懒加载的 CRNN 包装。`ckpt=None` 或权重不存在时退化为「只定位不识别」。"""

    def __init__(self, ckpt=DEFAULT_CKPT, device: str = "auto",
                 chunk: int = CHUNK, allow_missing: bool = True,
                 pad_min: float = PAD_MIN, pad_frac: float = PAD_FRAC,
                 pad_min_y: float = PAD_MIN_Y, pad_frac_y: float = PAD_FRAC_Y):
        self.ckpt = Path(ckpt) if ckpt else None
        self.device = device
        self.chunk = chunk
        self.allow_missing = allow_missing
        self.pad_min = float(pad_min)
        self.pad_frac = float(pad_frac)
        self.pad_min_y = float(pad_min_y)
        self.pad_frac_y = float(pad_frac_y)
        self._rec = None
        self._tried = False
        self.note = ""

    # ------------------------------------------------------------------
    def recognizer(self):
        if self._tried:
            return self._rec
        self._tried = True
        if self.ckpt is None:
            self.note = "未指定权重，只做定位"
            return None
        if not self.ckpt.is_file():
            self.note = "权重不存在: %s" % self.ckpt
            if not self.allow_missing:
                raise FileNotFoundError(self.note)
            return None
        try:
            # ★ 先设显存配额（LLM 在同卡常驻），再加载 CRNN。
            try:
                from gpu_mem import cap_vram
                cap_vram()
            except Exception:                                       # noqa: BLE001
                pass
            from textrec.infer import Recognizer
            self._rec = Recognizer.load(self.ckpt, device=self.device)
            self.note = self._rec.debug()
        except Exception as exc:                       # 没装 torch 之类的环境问题
            self.note = "加载识别器失败: %r" % (exc,)
            if not self.allow_missing:
                raise
        return self._rec

    # ------------------------------------------------------------------
    def read(self, bgr: np.ndarray, boxes: Sequence[Box],
             roles: Sequence[str] = ROLES) -> List[Dict]:
        """-> [{"bbox": Box, "reads": {role: (text, conf)}}, ...]"""
        boxes = [tuple(float(v) for v in b) for b in boxes]
        rec = self.recognizer()
        if rec is None or not boxes:
            return [{"bbox": b, "reads": {}} for b in boxes]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        reads: List[Dict[str, Tuple[str, float]]] = [dict() for _ in boxes]
        items, keys = [], []
        for i, b in enumerate(boxes):
            # 只对**裁剪**外扩；`reads[i]["bbox"]` 保持原框，`assign` 的几何归属不受影响。
            cb = pad_box(b, self.pad_min, self.pad_frac,
                         pad_min_y=self.pad_min_y, pad_frac_y=self.pad_frac_y)
            # 引脚号那一遍单独换裁片：切掉有色描边（见 `tight_ink_box` 的长注释）。
            cb_t = cb_ty = None
            if TIGHT_ROLES and any(r in TIGHT_ROLES for r in roles):
                tb = tight_ink_box(bgr, b)
                if tb is not None:
                    cb_t = pad_box(tb, self.pad_min, self.pad_frac,
                                   pad_min_y=self.pad_min_y, pad_frac_y=self.pad_frac_y)
                    cb_ty = pad_box(tb, self.pad_min, self.pad_frac,
                                    pad_min_y=self.pad_min_y, pad_frac_y=PINNUM_PAD_FRAC_Y)
            for r in roles:
                if cb_t is not None and r in TIGHT_ROLES:
                    items.append((cb_ty if r == "pin_number" else cb_t, r))
                else:
                    items.append((cb, r))
                keys.append((i, r))
        for s in range(0, len(items), self.chunk):
            part = items[s:s + self.chunk]
            got = rec.read_many(gray, part)
            for (i, r), (txt, conf) in zip(keys[s:s + self.chunk], got):
                reads[i][r] = (txt, float(conf))
        return [{"bbox": boxes[i], "reads": reads[i]} for i in range(len(boxes))]

    # ------------------------------------------------------------------
    def describe(self) -> str:
        self.recognizer()
        return "TextEngine(ckpt=%s, %s)" % (self.ckpt, self.note or "ok")
