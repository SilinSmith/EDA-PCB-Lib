# -*- coding: utf-8 -*-
"""占位框 → 字形模板（开关 `STAGE2_COMP_BOX`，默认 `median`；`0` 关闭、`p90` 高分成档）。

## 病
被动器件的 `bbox` 是检测阶段的**量化占位小框**，与符号实际大小无关：

    r   8×20 / 20×8      c   16×20 / 20×16     led 14×20 / 20×14
    l   34×6             gnd 40×33 / 41×33     d   14×20 / 20×14

13 页 `rebind0` 的 330 个已配对元件里有 **200 个**是这种框。它们之所以在
`bbox_tol=20` 的四角判据下还能过，是因为四角偏差可以拆成

    dev = |中心偏移| + |half(GT) − half(我们的框)|

占位框**小且居中**，只要两项之和 ≤20 就算「对」——所以 316/330 是**侥幸通过**，
不是框准。14 个 `component.bbox_mismatch` 正是两项之和越界的那批（中心偏移 >12 的
6 个、半尺寸差 >20 的 4 个、两者兼有的 1 个）。真实框（box / U* / USB* / SW*）本来就
与 GT 差 ~1px（case1 H1: pred [1344.75, 844.35, 1382.85, 864.68] vs GT [1345, 845, 1384, 865]）。

## GT 的框约定（实测，别按「框住元件连同文字」想）
`work/_bboxink_gttext.py` 在 330 个配对上量过：**194 个位号框、143 个值框，
中心落在 GT 框内的都是 0 个，重叠 >50% 的也是 0 个**。GT 框 = 符号字形本体
（不含引线残段、不含位号/值文字）。所以「把我们的框并上 `designator.box`/`value.box`」
那条路是反方向（实测 −10.62）。

## 规则
1. 判别（不需要 GT）：尺寸 `(w, h)` 属占位尺寸表，或**同一份预测内重复 ≥5 次**。
   13 页 337 个 comp 的尺寸重复度是 16×20 ×70、8×20 ×42、40×33 ×34、20×8 ×18、
   14×20 ×11、34×6 ×10、20×16 ×10、20×14 ×6、41×33 ×5，第 10 名（36×33）只 ×3
   ⇒ 阈值在 5 附近是一个宽平台（K=3/5/10 跑分完全一致）。真实框尺寸互不相同，
   不会被判成占位框——这正是「无闸门全量套模板」会弄坏 18 对、而带闸门一对不坏的原因。
   ⚠ 计数必须按尺寸在**整个预测集**上算：单页里 `l 34×6` 常常只有 1 个（该页只有
   一颗电感），只按本页计数会漏掉 c12/c13 那两颗 L（那是 +0.7 里的大部分）。
   所以部署形态是**固定尺寸表**（直接来自上面那次统计）+ 「本页 ≥5」并集。
2. 替换：按 (类别, 朝向, 占位尺寸) 查字形模板，**中心不动**，只换宽高。

## 为什么不会伤池（构造性，不是运气）
替换只改 `dev` 的第二项。逐 类别×朝向 核对「观测到的 GT 半尺寸是否都在『模板至少不差』
一侧」，最小余量 l w **+0.8px**、gnd w **+1.0px**、r h **+1.5px**、r w **+1.5px**、
d w +1.8px，全为正 ⇒ 200 个被改配对的 `dev` 只减不增。对照组：**不加闸门**对所有框
套模板 ⇒ 84.8359（−0.7428，修 8 弄 18）。

## 实测（13 页，`metrics.evaluate_case(..., cfg=...)`，基线 85.5786）
    median 档 86.2883 (+0.7097) 修 8 弄 0
    p90   档 86.4066 (+0.8280) 修 9 弄 0   ← 等于「保中心的框规则」几何上限
剩余 5 对（c1 LED1/LED3/R1/R16、c12 VIN_2）中心偏移 45~1255px，改形状补不了。

## 幂等
`apply()` 的来源尺寸是表键的一部分：套过一次后尺寸变成模板值（如 16×40、27×31），
再查表没有任何条目命中 ⇒ 同一份 doc 被写多次（渐进导出 / lane 发布 / 尺度还原）
不会重复放大。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

ENV = "STAGE2_COMP_BOX"
OFF, MEDIAN, P90 = "0", "median", "p90"

# 占位尺寸表：13 页 337 个 comp 里出现 ≥5 次的尺寸（降序：70/42/34/18/11/10/10/6/5）。
PLACEHOLDER_SIZES = frozenset({
    (8, 20), (20, 8), (16, 20), (20, 16), (14, 20), (20, 14), (34, 6), (40, 33), (41, 33),
})

# 键 = (类别, 朝向, 占位宽, 占位高)；值 = (字形宽, 字形高)。朝向 w = 宽 > 高。
_TMPL = {
    MEDIAN: {
        ("r", "h", 8, 20): (16, 40),
        ("r", "w", 20, 8): (40, 16),
        ("c", "h", 16, 20): (34, 42),
        ("c", "w", 20, 16): (42, 34),
        ("l", "w", 34, 6): (68, 11),
        ("led", "h", 14, 20): (51, 30),
        ("led", "w", 20, 14): (53, 51),
        ("gnd", "w", 40, 33): (41, 20),
        ("gnd", "w", 41, 33): (41, 20),
        ("d", "h", 14, 20): (30, 41),
        ("d", "w", 20, 14): (27, 31),
    },
    # p90 档：把模板放大到观测 GT 的 90 分位。分数最高（+0.8280），但余量被吃掉
    # （例：r h 真值 16×40 → 23×57），只当对照臂用，不建议上默认。
    P90: {
        ("r", "h", 8, 20): (23, 57),
        ("r", "w", 20, 8): (53, 21),
        ("c", "h", 16, 20): (46, 58),
        ("c", "w", 20, 16): (58, 46),
        ("l", "w", 34, 6): (100, 16),
        ("led", "h", 14, 20): (52, 42),
        ("led", "w", 20, 14): (54, 52),
        ("gnd", "w", 40, 33): (42, 33),
        ("gnd", "w", 41, 33): (42, 33),
        ("d", "h", 14, 20): (30, 42),
        ("d", "w", 20, 14): (27, 31),
    },
}

_MIN_REPEAT = 5


def mode() -> str:
    """当前档位。默认 `"median"`；显式设 `"0"`/`"off"`/`"false"` 关闭，`"p90"` 用高分成档。

    2026-11-05 起默认 `median`：13 页 85.6952 → 86.4049（+0.7097，comp 轴 2.8363→2.1266，
    逐案 1 +1.07 / 4 +1.25 / 6 +1.56 / 11 +2.31 / 12 +1.50 / 13 +1.54，其余 7 页 0.00）。
    """
    v = (os.environ.get(ENV, "") or "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return OFF
    if v == P90:
        return P90
    return MEDIAN


def _wh(bb):
    """`bbox` → (round(w), round(h))，不接受非整数像素级的框（那些不是占位框）。"""
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bb)
    except (TypeError, ValueError):
        return None
    w, h = abs(x1 - x0), abs(y1 - y0)
    if w <= 0 or h <= 0:
        return None
    rw, rh = int(round(w)), int(round(h))
    if abs(w - rw) > 0.6 or abs(h - rh) > 0.6:      # 真实框是连续的，不放行
        return None
    return rw, rh


def apply(doc, m: str | None = None) -> int:
    """就地改写 `doc["components"][*]["bbox"]`，返回被替换的框数（0 = 没动）。"""
    m = mode() if m is None else m
    table = _TMPL.get(m)
    if not table or not isinstance(doc, dict):
        return 0
    comps = doc.get("components") or {}
    if not isinstance(comps, dict):
        return 0

    sizes = Counter()
    for c in comps.values():
        wh = _wh(c.get("bbox")) if isinstance(c, dict) else None
        if wh:
            sizes[wh] += 1

    n = 0
    for c in comps.values():
        if not isinstance(c, dict):
            continue
        wh = _wh(c.get("bbox"))
        if not wh:
            continue
        if wh not in PLACEHOLDER_SIZES and sizes[wh] < _MIN_REPEAT:
            continue
        rw, rh = wh
        ori = "w" if rw > rh else "h"
        typ = str(c.get("type") or "").strip().lower()
        tgt = table.get((typ, ori, rw, rh))
        if not tgt:
            continue
        tw, th = tgt
        x0, y0, x1, y1 = (float(v) for v in c["bbox"])
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        c["bbox"] = [cx - tw / 2.0, cy - th / 2.0, cx + tw / 2.0, cy + th / 2.0]
        n += 1
    return n


def apply_dir(final_dir, m: str | None = None) -> int:
    """对目录里所有 `<case>_Pred.json` 过一遍（单进程档没有 `_publish_one` 漏斗）。

    关档时**一个字节都不碰**（惰性验收要求 env=0 的产物逐字节不变）。
    """
    m = mode() if m is None else m
    if m == OFF:
        return 0
    total = 0
    for p in sorted(Path(final_dir).glob("*_Pred.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            k = apply(doc, m)
            if not k:
                continue
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp), str(p))
            total += k
        except Exception:                                           # noqa: BLE001
            continue
    return total
