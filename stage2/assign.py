# -*- coding: utf-8 -*-
"""第二阶段 · 角色分配：把「一团文字框 + 每个框的 5 种读法」变成
「哪个元件叫什么位号、值是多少、每个引脚叫什么/几号」。

几何依据（`work\\_text_geom.py` 在 32 张 CVJsonStd 上实测，1020 个 pinName 100% 落在
元件框**内**、1115 个 pinNumber 95% 落在框**外**且 p50 距离 5px、位号 p50 距离 6px、
网络标签离最近元件 p50 达 34px）——所以「框内 = 引脚名、框外 16px 内 = 位号/参数/
引脚号、更远 = 网络标签」是一条有实测支撑的分线，不是拍脑袋。

字符串模式提供第二重约束；类型→位号前缀先验（`work\\_text_geom.py` 在 9609 个官方
元件上统计，例如 `r`→R 1617/1626、`box`→U 900/996、`d`→D 201/201）提供第三重。
"""
from __future__ import annotations

try:                                    # 作为包导入（submission\run.py 把 eda-pcb\ 加进 sys.path）
    from envflag import env_on      # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage2\assign.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on
import io
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import structmatch
from . import keyform as _keyform
from .expect import expect as expect_kind, fragment_score, score_expect

# ---------------------------------------------------------------------------
# ★ 来自队友分支 `origin/codex/improve-pin-network-parsing`（2026-10-15 并入）：
#   「引脚图像证据」通道在 assign 侧的**入口校验**。队友那五个引脚读法模块
#   （`stage2/pin_lane_reading.py` / `mono_pin_reading.py` / `pin_number_reading.py` /
#   `pin_name_reading.py`）会 `from .assign import assign, _target_index`，并在内部
#   用 `assign(..., pincol=..., pin_evidence=...)` 反查「现有绑定」⇒ 这里给出所需符号。
#
#   `_target_index(text, cid, pins, evidence)`：某个裁片带 `pin_target`
#   （`{component_id, pin_index, edge, point_img, anchor_img}`）时，核验它指向的那根脚
#   是否真有**新鲜的**实测接触证据。返回 `None`（没有 target）/ `-1`（不可信）/ 下标。
#   `_trusted_text_contact`：彩色接触要求 `frame_confirmed`；灰度则走
#   `structmatch.trusted_frame_pin_evidence`（实测帧边/短桩，绝不提升猜测）。
# ---------------------------------------------------------------------------
_COLOR_TEXT_SOURCES = frozenset(("red_green_transition", "red_stub_endpoint",
                                 "red_green_transition_uncertain"))
_MONO_TEXT_SOURCES = frozenset(("mono_pin_name_ppocr", "mono_pin_number_ppocr"))
# 最近一次 `assign` 收到的逐脚图像证据（`{cid: [record, ...]}`）；给队友那五个读法
# 模块的 `_target_index` 用，也方便离线探针复查。不参与返回签名。
LAST_PIN_EVIDENCE: Dict[str, List] = {}

# ★ 队友分支 `codex/improve-pin-network-parsing` 的「同框多角色一致票」读法
#   （`stage2/assign.py:573`）。他们默认 1；我们**先默认 0**（要 A/B 后才谈默认值，
#   用户 m40034 的规矩：改了识别逻辑必须 A/B）。
#   语义：5 个识别角色（CRNN 的多路读法）里 **>=3 个**读出同一个数字串才叫一致；
#   只有「原读法置信度 < 0.6」这种**不定**情形才允许用一致票改写，且要求位数相同
#   （绝不靠「图纸顺序」推断数字）。作用域只在**引脚号**这一路。
PINNUM_CONSENSUS = env_on("STAGE2_PINNUM_CONSENSUS", "0")


def _best_number_read(reads: Dict[str, Tuple[str, float]], fresh_color: bool = False):
    """Use stronger same-box agreement only for an uncertain numeric glyph.

    ★ 逐字并入队友实现（`codex/improve-pin-network-parsing:stage2/assign.py:573`），
    `fresh_color` 参数保留但我们的默认路径不传（那一路要彩色读数语料）。
    """
    original = _best_read(reads, "pin_number", fit_pinnum)
    if not PINNUM_CONSENSUS or original is None:
        return original
    groups = {}
    for txt, conf in reads.values():
        if txt.isdigit() and fit_pinnum(txt) > 0:
            groups.setdefault(txt, []).append(float(conf))
    # Three agreeing role reads are a majority of the five recognizer roles.
    supported = [(max(cs), txt) for txt, cs in groups.items() if len(cs) >= 3]
    if not supported:
        return original
    conf, txt = max(supported)
    if original[2] >= 0.6:
        # Role probabilities are not correctness guarantees. Revisit a strong
        # short read only when three strong reads of this same glyph agree on
        # its unshortened digits. Never infer a number from drawing order.
        cs = groups[txt]
        if not (fresh_color and len(txt) > len(original[1])
                and len([v for v in cs if v >= 0.85]) >= 3
                and 1 <= int(txt) <= 999):
            return original
        fit = fit_pinnum(txt)
        return fit * (0.3 + 0.7 * conf), txt, conf, fit
    if txt == original[1] or conf <= original[2] or len(txt) != len(original[1]):
        return original
    fit = fit_pinnum(txt)
    score = fit * (0.3 + 0.7 * max(0.0, min(1.0, conf)))
    return score, txt, conf, fit


def _same_image_point(value, expected):
    try:
        values = tuple(float(v) for v in value)
        other = tuple(float(v) for v in expected)
        return (len(values) == len(other) == 2
                and all(math.isfinite(v) for v in values + other)
                and math.dist(values, other) <= 0.001)
    except (TypeError, ValueError, OverflowError):
        return False


def _trusted_text_contact(record, box=None, point=None):
    """Accept measured color contacts or independently validated frame strokes."""
    if not isinstance(record, dict) or record.get("trust") != "trusted":
        return False
    if record.get("source") in _COLOR_TEXT_SOURCES:
        return record.get("frame_confirmed") is True
    return structmatch.trusted_frame_pin_evidence(record, bbox=box, point=point)


def _target_index(text, cid, pins, evidence):
    """Validate a crop target against fresh measured contact/frame evidence."""
    target = text.get("pin_target")
    if target is None:
        return None
    if not isinstance(target, dict) or target.get("component_id") != cid:
        return -1
    index = target.get("pin_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(pins):
        return -1
    # Only freshness/point alignment is needed here; actual body/corridor
    # checks are repeated by the binding code with the component rectangle.
    checked = structmatch.normalize_pin_evidence(pins, evidence)
    if not checked:
        return -1
    ev = checked[index]
    mono = text.get("source") in _MONO_TEXT_SOURCES
    if (not _trusted_text_contact(ev, point=pins[index])
            or (ev.get("source") not in _COLOR_TEXT_SOURCES and not mono)
            or target.get("edge") != ev.get("edge")):
        return -1
    if mono and (target.get("point_img") is None or target.get("anchor_img") is None):
        return -1
    if (target.get("point_img") is not None
            and not _same_image_point(target["point_img"], pins[index])):
        return -1
    if (target.get("anchor_img") is not None
            and not _same_image_point(target["anchor_img"], ev.get("anchor_img"))):
        return -1
    return index


# 位号前缀策略（`STAGE2_PREFIX` 环境变量切臂）。**默认 `strict`，是实测的赢家**：
# 5 张官方图（0001/0025/0050/0059/0101）逐张权重得分之和 —— off 20.00 / **strict 26.67** / force 24.86。
# 三臂的位号赋值率正好相反（off 0.379 / strict 0.264 / force 0.353）：
# **赋值率不等于正确率**，`strict` 键更少但更准。
#   off    不做任何矫正，只用读出来的字母
#   strict 只信「字母已在类型预期前缀集合里」的读法（不凭空造键）
#   force  字母不在预期集合里也强行换成首选前缀（readability 最高，但会造键）
PREFIX_MODE = os.environ.get("STAGE2_PREFIX", "strict")

# ★ 前缀「首字母被 CRNN 吃掉」的补救（默认 **off**，`STAGE2_PREFIX_SKEL=1` 打开）。
# 实测（`work\_tpA_dbg`，0050 的 `v0010` = GT `TP21` @图内(3888,1864)）：
# 文字框 `[3888,940,3927,962]` 的位号读数是 **`P21`（conf 0.926）** —— 首字母 `T`
# 被吃掉。`canon_designator("P21","testpoint")` 在 `strict` 下三道全过不去
# （`P` / `P2` 都不在 `TYPE_PREFIX["testpoint"]=("TP",)` 里）⇒ 返回 `("",0.0)`
# ⇒ 0050 的 20 个 testpoint **全部 nameless** ⇒ key 落到 `pNNN`、与 GT 的
# `TPn` 键交集 = 0 ⇒ 每个检出的 testpoint 都是纯 fp。
# 打开后：读出的字母若是某个允许前缀的**后缀**（`P` ⊂ `TP`）就补回完整前缀。
PREFIX_SKEL = env_on("STAGE2_PREFIX_SKEL", "0")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NEAR = 16.0          # 距元件框多近算「贴在元件上」
FAR = 40.0           # 多远算「网络标签」（离所有元件都远）
# ★★ 归属判据的度量方式（`STAGE2_OWN_BBOX`，默认 **1** = 用「框到框」的间距）。
#
# 动机（10GT case `1` 的 `U1`，2026-10 实测）：`FAR=40` 原来是按**文字中心到元件框**
# 的距离定的，于是**归属随文字长度变化** —— 同一个位置、同一排引脚号：
#   1 位号 `7`  宽 12 ⇒ 中心距 961−933 = **28** ⇒ 归属 ✓
#   2 位号 `26` 宽 24 ⇒ 中心距 961−926 = **35** ⇒ 归属 ✓
#   3 位号 宽 34~36 ⇒ 中心距 961−919..921 = **40~42**、右侧 `1359−1314 = 45`
#                                 ⇒ **判成 `free`（网络标签），整排 3 位球号一个
#                                   都进不了引脚号候选池**。
# 实测（`work\tools\diag\_pinband.py`）：case 1 的 `U1` 框外 70px 内有 **26 个**小文字框
# （与 GT 的 26 个脚一一对应），但 `STAGE2_DBG_CID=b0000` 打出来的 `pin_number` 候选池
# 只有 **21** 个，缺的正是那 5 个**最宽的**（32~36px）—— 它们全被这条长度依赖踢出去了。
# ⇒ 「贴不贴在这颗元件上」只应取决于两框之间还有多少空白。`OWN_BBOX=0` 退回历史行为。
OWN_BBOX = env_on("STAGE2_OWN_BBOX", "1")
# ★★ 位号通道的**全局一对一**归属（`STAGE2_OWN_ASSIGN`，默认 **0 = 关**）。
#
# 动机：`cvparts/own.py` 的 `assign()` 是「整体最小代价的一对一」（直接复用官方
# `schematic_parser.metrics._assign`），但全仓库**从未被调用** —— 这正是 `own` 节点
# 描述里「一个主张者够到多个标的 → 整体解一次一对一」那一半。
# 现状（`OWN_ASSIGN=0`）是：每颗元件各自从候选池挑最优，再按**位号字符串**贪心去重。
#
# 实测（`work\tools\diag\_ownasgn.py`，官方 9 张的 `_assigned` 产物，GT 用
# 官方 `bbox_mode='corners'` 口径近邻匹配）：现状与这一臂**都拿到 114 个位号**，
# `dup_box`（同一块文字框被 >=2 颗元件选中）与 `dup_canon` **两臂都是 0** ——
# 因为 `owned` 本身就是一份**划分**（每块文字只归一颗元件），一对一不变式
# 在改造前就已经成立，这一臂只可能**重排**。重排发生在 4 个元件上，去问 GT：
#   官方 case 2 `c0009`（GT `R6`，文字被误读成 `R5`）与 `c0013`（GT `R5`）：
#     现状 = c0013 拿 `R5` **✓**、c0009 空；这一臂 = c0009 拿 `R5` **✗**、c0013 空。
#   官方 case 7 `c0003`（GT `C6`，误读成 `C5`）与 `c0009`（GT `C5`）：
#     现状 = c0009 拿 `C5` **✓**；这一臂 = c0003 拿 `C5` **✗**。
#   ⇒ 现状 **2 对 0 错**、这一臂 **0 对 2 错**：整体求解会把被误读的那块框
#     配给错的元件。**判负，保持 0**；机制留在代码里以便复现与后续复用。
OWN_ASSIGN = env_on("STAGE2_OWN_ASSIGN", "0")
# ★★ 位号通道的**全局一对一 + 显式弃权**（`STAGE2_OWN_MATCH`，默认 **0 = 关**）。
#
# 与上面 `OWN_ASSIGN`（以及 `STAGE2_GLOBAL_MATCH` 的 `_global_match`）的**唯一区别**：
# 那两条都是一张 `n × m` 的方阵，`linear_sum_assignment` 被迫配出 `min(n, m)` 对，
# 再拿 `thr` 事后过滤 —— **没有弃权位**，被挤掉的真实主人再也拿不回自己的位号。
# 这里给每颗元件加一列**专属弃权列**，权值 = 该元件自己当前最优分 × `OWN_MATCH_ABSTAIN`：
# 于是「配一块真实文字框」只有**严格优于弃权**才会被选中（最优解里不含负增益的对）。
# 列的唯一性直接给出两条不变式：**一块文字框最多配一颗元件、一颗元件最多拿一个位号**。
# 候选池原样复用 `desig_all`（= 每颗元件的 `owned + steal` 候选，不新造候选），
# 权值就是贪心路径用的同一个 `_score`（见 `_own_match_pick` 的 docstring）。
OWN_MATCH = env_on("STAGE2_OWN_MATCH", "0")
OWN_MATCH_ABSTAIN = 0.5   # 弃权权值 = 该元件候选里的最高分 × 这个常数
def _dbg_want(cid: str) -> bool:
    """`STAGE2_DBG_CID` 支持逗号分隔多个 cid（调试用；不影响任何行为）。"""
    v = os.environ.get("STAGE2_DBG_CID") or ""
    return bool(v) and cid in [x.strip() for x in v.split(",") if x.strip()]


MAX_BOX_W = 3.0      # 位号框宽度 / 字高 之比上限（挡掉「一排字」被当成一个框）
# **IC 引脚号必须离得更远也能收**（2025-11 实测，`case 1` 的 `U1`）。
# 大 IC 的引脚号印在**符号框外**、而且离框边 29~44 px（`U1` 框左沿 x=961，
# 号框右沿 x=937 ⇒ d=29；更远的 `101/122/15` 在 x=936 ⇒ d=42~43）。
# 旧的 `d > NEAR(16)` 闸门把这 26 个号**全部丢掉**，`b0000` 的
# `pins[i]['number']` 全是 `None`，于是 `U1` 的 34 个引脚只能按位置编
# `pin_1..pin_34`，与 GT 的 `7/16/26/.../122` 一个都对不上 —— 这就是
# `pin_f1` 在 10GT 上 `missing 26`、以及 `net_hypergraph` 里 `U1` 的
# 引脚全成 fp 的直接原因。引脚**名**（VDD/GND）在框内，不受这条影响。
PINNUM_NEAR = float(os.environ.get("STAGE2_PINNUM_NEAR", "60"))
# 引脚号的**距离惩罚系数**。`_score` 的默认 `dist_w=0.035` 会让 `-0.035*d`
# 在 d>28 时把满分读法（`fit*conf ≈ 1.0`）直接压到 ≤0 ⇒ 被 `if s <= 0` 丢掉。
# 引脚号天生比位号离得远（大 IC 印在框外），所以它对距离应该几乎不敏感。
PINNUM_DISTW = float(os.environ.get("STAGE2_PINNUM_DISTW", "0.005"))
# ★★ 2026-10：`score_of` 的**沿边距离衰减尺度**。原来写死 `/(1 + d/20)`，
# 衰减太慢 ⇒ 单调对齐「整排挪一格」只让每对掉一点分，却能多配上一对
# （`cover=PIN_COVER=0.15`），于是 DP 宁可错位也要多拿一对。
#
# 实测（10GT case 2 的 `U1`，框 img x628..688 / y297..396）：GT 只有 4 个脚
# （img y 316 / 335 / 356 / 376），我们却给出 7 个左边点
# （307 / 316 / 326 / 336 / 354 / 370 / 376）—— 多出来的 307 / 326 / 370 是
# **不是真脚的触点**。候选池里 `2`/`3`/`4` 各被文本检测**重复检出两次**
# （`2`@330 与 `2`@331 …）。于是对齐把 `2` 发给 316 和 326、把 `3` 顶到 336 ——
# 而 336 才是真正的 `pin_2` ⇒ 症状是「GT 脚 2 上写的是 3」。
# 把衰减加快（`d/20` → `d/PINNUM_DECAY`）能让错位的代价超过 `cover`，
# DP 就会选择「让那个多余的触点悬空」而不是「整排挪一格」。
PINNUM_DECAY = float(os.environ.get("STAGE2_PINNUM_DECAY", "10"))
# ★★ 候选池去重半径（**默认 0 = 关**，因为它实测是**净亏**）。
#
# 现象上「`2(624,331) | 2(624,330) | 3(624,351) | 3(624,350)` 6 个候选其实只有
# 3 块文字」确实是文本检测的重复检出，两个副本会被对齐到**两根不同的脚**上。
# 但实测把重复去掉反而更差：
#   官方 9 张（`--cases 1,2,3,4,5,7,8,9,10`，`FUSION_LLM=0`，基线 56.45）
#   `DUP=3 DECAY=20` 54.65 · `DUP=3 DECAY=10` 54.57 · `DUP=3 DECAY=6` 54.19
#   `DUP=0 DECAY=20` 56.45 · `DUP=0 DECAY=10` **56.71** · `DUP=0 DECAY=6` 56.71
# ⇒ 重复副本实际上在**多占坑**：同一块文字被绑到两根脚上，其中一根是对的
#   （`tp`）、另一根是错的（`fp`），而 `pin_f1` 里丢 `tp` 比多 `fp` 更贵。
# 所以这里保留旋钮但**默认关**，并把「去重」这个直觉记下来当作反例。
PINNUM_DUP = float(os.environ.get("STAGE2_PINNUM_DUP", "0"))
# ★★ 引脚号的「抢读」半径（`STAGE2_PINNUM_STEAL`，**默认 0 = 关**）。
#
# 病（10GT case 5 的 `U3`，2026-10 实测）：`stage1` 在 AMS1117 处同时给出
# `c0002`（YOLO，score 0.9657）与 `b0000`（`bigic` 从**一圈绿色导线**围出的伪矩形，
# score 0.9，见 `work/_b5crop.png`）。文字归属是「到元件框的硬 argmin」，
# 而 `b0000` 的框把 `1`/`2`/`4` 三块号框整个包在里面（`d=0`）⇒ 它们归 `b0000`；
# `b0000` 永远拿不到位号（`DROP_UNKEYED` 直接丢）⇒ **这三块文字被静默地浪费掉**，
# `c0002` 的 `pin_number` 候选池只剩 `3`（`[dbg] pool=2` 那是同一块文字的两次检出），
# GT 的 `pin_1/2/4` 全部 `fn`。同类损失见 `work/tools/dev/_pinwhy.py`。
#
# 规则：引脚号这一轮，除了自己名下的框，还允许读**别人名下 / 无主、但离自己
# <= `PINNUM_STEAL` 的框** —— 只抢「主人**没有位号**（= 注定被丢）或**无主**」的框，
# 已挂键的邻居的框一根都不碰（那才是抢别人东西）。位号（`STEAL_R`）与值
# （`VALUE_STEAL_R`）早就有这条通道，引脚号是最后一个没有的。
PINNUM_STEAL = float(os.environ.get("STAGE2_PINNUM_STEAL", "0"))
# ★★ 「纯数字文本框 = 引脚号，不是引脚名」（`STAGE2_PINNUM_INSIDE` + `STAGE2_PINNAME_NODIGIT`，
#    **默认都 0 = 关**；两者**必须成对打开**，只开一个几乎不起作用，见下）。
#
# 病（10GT case 2 的 `U3`/`U4`，2026-10 实测，`STAGE2_DBG_CID=c0001`）：
#   真脚 `pin_1`（GT `GND`）的号码 `1` 被文字检测**过检成 21x13 的大框**
#   （同一排别的号框只有 6~8px 宽，见 `_pinnum.py` 的槽位表），框心于是落进符号框内
#   ⇒ `t["inside"] = True`：
#     * `pin_number` 轮要求 `not inside` ⇒ 这块文字**根本进不了号候选池**；
#     * `pin_name` 轮要求 `inside` ⇒ 它反而进了名候选池，靠 `fit_pinname("1") = 0.15`
#       加 `_score` 的 `inside` 奖励 0.15 拿到 `s = 0.299`，被 `structmatch` 绑给
#       **体外假槽** `pin[0]`（该槽本来会被丢），并进 `taken_box` 被永久占用。
#   ⇒ 真脚既没号（`fn`）、假槽白得一个「名」。`fit_pinnum` 本来就把纯数字判为
#   `1.0`、`fit_pinname` 只给 `0.15` —— 这条判据只是没有贯穿到 `inside` 闸门上。
#
# 规则：① `STAGE2_PINNUM_INSIDE=1` ⇒ 号候选池接受「框心在体内、但**最佳号读法是纯
#   数字**」的框（`d=0` 照旧参与打分，`MAX_BOX_W` 等闸门照旧）；②
#   `STAGE2_PINNAME_NODIGIT=1` ⇒ 名候选池不再收「最佳号读法是纯数字」的框。
#   成对使用：只开 ① 时那块框已被名轮 `taken_box` 吃掉，号轮仍拿不到它。
PINNUM_INSIDE = env_on("STAGE2_PINNUM_INSIDE", "0")
PINNAME_NODIGIT = env_on("STAGE2_PINNAME_NODIGIT", "0")


def _pure_digit_read(t: dict):
    """框的**最佳引脚号读法**；纯数字（`PINNUM_RE` 且 `isdigit`）返回该串，否则 `None`。

    `None` = 这块框在任何角色下都不像一个「号」（读不出、或读出来不像数字）。
    """
    try:
        got = _best_read(t.get("reads") or {}, "pin_number", fit_pinnum)
    except Exception:                                                       # noqa: BLE001
        return None
    if not got:
        return None
    txt = str(got[1] or "").strip()
    return txt if txt.isdigit() else None


# ★★ 幻影脚点槽合并半径（`STAGE2_PIN_STUB_MERGE`，**默认 0 = 关**，单位 px）。
#
# 病（10GT case 5 的 `U1`，2026-10 实测，见 `work/tools/dev/_slotdbg.py`）：
# `stage1` 的 `pintrace` 在同一根真脚上给出**两个相距 ~5px 的触点** ——
# `pins[c0001]` 的 slot0 `(654.6,266.0)`（`pincol` 记录 idx=0，号 `3`）与
# slot5 `(654.6,271.0)`（名 `FB`）；slot1 `(654.6,305.5)`（号 `7`）与
# slot6 `(654.6,311.0)`（名 `P`）。两个角色轮**各绑一半**：号落在 `pincol`
# 那个触点上、名落在另一个触点上。于是两槽都凑不齐「号 + 名」，被
# `assemble.build_doc` 的 `STAGE2_BOX_PIN_NEED_NAME`（late 模式）**一起丢掉**：
# GT 的 `pin_3 FB`（正解就在我们手上）直接变成 `fn`。
#
# 这两个触点其实在 stage1 里**已经属于同一个网**（`net_8 = {U1[0],U1[4],U1[5]}`、
# `net_18 = {U1[1],U1[6]}`）⇒ 合并**不会改变任何网的成员**，只让那根脚能交出去
# （`assemble.py:1144` 只把有 `num_of` 的引脚放进 `hyperGraph`）。
#
# 规则：两槽**互补半满**（一个只有号、一个只有名）且两点距离 <= 该半径时，
# 把号搬进**有名字**的那一槽（名字框在符号内、几何更紧，其槽点也更贴 GT 的
# 引脚点），号槽清空。**只搬已有读数、不新造读数**，且被搬走号的槽本来就被丢，
# 所以对现有输出的影响面只有「多出几根脚」。
#
# ★ 2026-10 A/B（官方 10 张 `--no-vlm --engine official`）：关 = 59.9903（= 历史默认口径），
#   `=12` = 59.9977（+0.0074/张；只有 case 5 变：51.9380 -> 52.0122，pin tp 22 -> 23）。
#   与 `STAGE2_PIN_ANCHOR=12` 合用（`anb1`）= **60.5854（+0.5951/张）**，逐案
#   case 4 +1.7946 / case 5 +1.1180 / case 8 +3.0387，**pin fp 与 netL fp 不增反降**。
#   ⇒ 与锚点是互补机制（锚点救「号被单调剪枝清掉」，这里救「号与名落在两个幻影触点上」），
#   默认转正为 `12`。
STUB_MERGE = float(os.environ.get("STAGE2_PIN_STUB_MERGE", "12"))
# 引脚**名**框的宽高比上限。`MAX_BOX_W=3.0` 会把 `VDDA`(64x18=3.56)、
# `GNDA`(67x18=3.72) 这类长名判成 `wide` 直接踢出候选池 —— 而它们恰恰
# 是 `U1` 的合法引脚名。IC 引脚名本来就比位号长，这里单独放宽。
PINNAME_MAXW = float(os.environ.get("STAGE2_PINNAME_MAXW", "6.0"))
# ★★ 引脚**名**候选的「**有色墨迹**」闸门（`STAGE2_PINNAME_CFRAC`，单位 px 无关，是占比）。
#
# 病（10GT case 5 的 `U3`，见 `docs/RESULTS.md` §5.91）：pin1 名字框那一行有两个候选
# —— 真名 `ADJ(GND)`（67x13）与**红点引脚标记**（8x8，CRNN 读成 `Q`、conf 0.864）。
# 名轮把标记绑给了 pin1、把 `ADJ(GND)` 挤到 pin2（`VOUT(TAB)` 再往下挤）⇒ 两条名字都错。
# **几何分不开**（合法名 `L` = 7x10，红点 = 8x8），**墨迹颜色分得开**：
# 真名框的 `cfrac`（`text.py::colored_frac`）= 0.000，红点 = 0.48~0.64。
# 于是名候选池里踢掉 `cfrac > PINNAME_CFRAC` 的框。**只压名轮**：号轮不动。
# 该属性只在 `STAGE2_TEXT_INK=1` 时才存在 —— 两开关要**一起**开（`cfrac` 缺席时本闸门不生效）。
PINNAME_CFRAC = float(os.environ.get("STAGE2_PINNAME_CFRAC", "0.20"))
# ★ 名轮压「像位号的读法」（`STAGE2_PINNAME_KEYLIKE`）。
#
# 病（10GT case 6 的 `U2`/`U4`，SOT23-5）：GT 名序列自上而下 `OUT,GND,FAULT,EN,IN`，
# 我们读到 `Q1,OUT,GND,EN,IN` —— pin1 被旁边一颗三极管位号 `Q1` 占走，其余名字整体下移
# 一格、`FAULT` 被挤掉。`fit_pinname("Q1") = 1.0`，几何上它又最近，所以纯评分救不回来。
# 规则：名轮里「大写字母 1~3 个 + 数字」的读法只给 0.2 分（真引脚名极少长这样，
# 而位号全是这个形状）；号轮不受影响。
PINNAME_KEYLIKE = float(os.environ.get("STAGE2_PINNAME_KEYLIKE", "0.2"))
_PINNAME_KEY_RE = re.compile(r"^[QRCULDKJFY]{1,2}\d{1,4}$")
# ★ 号轮的「每字符像素宽」闸门（`STAGE2_PINNUM_PXC`，默认 0 = 关）：
#   引脚号是 1~3 个字符，真号码框 ≈ 10~11 px/字符；而合并了上下两行、或抢到名字
#   宽度的框会到 14+ px/字符（case 8 的 U1：`[967,701,994,711]` 27x10 读成 `26`
#   = 13.5、`[967,470,996,485]` 29x15 读成 `11` = 14.5 ⇒ 后者的号码是错的）。
#   只对「读出的号码含数字」的框生效，宽度取框自身 bbox。
PINNUM_PXC = float(os.environ.get("STAGE2_PINNUM_PXC", "0"))
# ★★ 「器件名框放行回名轮」的高度闸门（`STAGE2_PARTNAME_FREE_H`，px；见 §5.92）。
# `take()` 原先无条件把器件名框赶出名轮候选池，而 `vlm/enrich.py` 改写器件名**文字**时
# 保留旧框，于是像 case 5 的 U3 那样「框其实是 `VOUT(TAB)` 名框」的记录会把真名框也赶走。
# 只放行**高得像一行引脚名**（≤18px）的框：10GT 引脚名行高 10~18px，而 case 4 的 U2
# 器件名块是 53x22（两行大字块），放行会被读成 `15IO` 抢走 pin2（80.00 → 78.22）。
# 实测三条该放行的框：case 1 U1 89x18、case 2 U3 26x10、case 5 U3 75x13。
PARTNAME_FREE_H = float(os.environ.get("STAGE2_PARTNAME_FREE_H", "18.0"))
# ★★ 「名候选必须靠近它的**最近那根脚**」与「同一块墨迹的重叠副本只留最宽的一份」
#   （`STAGE2_PINNAME_ROWNEAR` / `STAGE2_PINNAME_DUPOV`；见 `docs/RESULTS.md` §5.93）。
#
# 病（10GT case 6 的 `U4`，`STMPS2141STR`）：stage1 给出 **10 个脚点**（1 上 / 3 右 / 4 下 / 2 左），
# 而 `pincol` 那套窗口引导在这张图上**根本没产出**（`pincol_stat.comps = 0`）⇒ 名轮退回
# `structmatch.rebind` 的**沿边**距离打分。沿边距离对「上边脚」量的是 x 方向：
# 左侧名框 `IN`（中心 x1647）与上边脚点（x1643.8）沿边只差 3px ⇒ 它被对齐到**上边脚**上；
# 而左侧那两根脚（行 y=208 / y=247.5）只剩下**同一块墨迹的窄副本**（8x10，CRNN 读 `N`，
# conf 0.99 反而比 12x10 的 `IN` 0.987 高）⇒ 最终 `pin4`/`pin5` 都读成 `N`（每根 −1.67 分）。
#
# 两条零调用几何规则（互补，缺一条都不行）：
#   ① 名候选只允许归到「2D 距离最近的脚」（允许 `×PINNAME_ROWTIE` 的并列）——
#      挡住跨行/跨侧的抢读（上边脚抢左侧名框）；
#   ② 同一行（|Δcy| ≤ `PINNAME_DUPGAP`）且 x 重叠 ≥ `PINNAME_DUPOV`（占较窄者宽度）的两块框，
#      只留**更宽**的那份 —— 窄副本是同一块墨迹的截断裁片（§5.86 记过 `IS→S`/`EN→N`）。
PINNAME_ROWNEAR = int(os.environ.get("STAGE2_PINNAME_ROWNEAR", "1") or 0)
PINNAME_ROWTIE = float(os.environ.get("STAGE2_PINNAME_ROWTIE", "1.5"))
PINNAME_DUPOV = float(os.environ.get("STAGE2_PINNAME_DUPOV", "0.60"))
PINNAME_DUPGAP = float(os.environ.get("STAGE2_PINNAME_DUPGAP", "6.0"))
# ★★ 2026-10-08：**框外补引脚名**的距离上限（px，`0` = 关闭）。
# 语义 = 「引脚号那一轮跑完之后，只往**还空着**的名字槽里补框外的名字」，
# 也就是 late pass。取值是「文字中心到元件框」的距离上限，走 `t["d"]` 那套量。
# 实现在 `assign()` 里角色循环**之后**的那一段「★★ 框外补名」，`if PINNAME_OUTSIDE > 0:` 起。
#
# **为什么必须试这个**：`assign.py` 对 `pin_name` 角色有一条
#     `if want_inside and not t["inside"]: continue`
# 而 GT 的引脚 `point` 实测 **2220 根里 100% 落在元件 bbox 之外**
# （到框边界 中位 9px / p75 14px / p90 35px / p99 67px），
# 且「GT 引脚点 6px 内我们检出了文字」的那 1035 根里 **89.5% 的文本中心也在框外**
# （`work\tools\diag\_pinlabel.py --n 60`）。结果就是产出的引脚里
# **78.2% 的 `pinname` 回退成了引脚号本身**（`assemble.py:761 nm = r["nm"] or num`），
# 而 GT 的 IC 引脚名 **86.5% 是真信号名**（`CS / D1/DO / GPI04 / IOVDD / VCLK`…），
# 纯数字名只占 13.5%（`_gtpinname_stat.py`）。⇒ 引脚名这一项**结构性地对不上**。
#
# ★ **第一版（放宽 `pin_name`、仍在引脚号之前跑）实测判负**：19 张（官方 10GT 9 张 +
#   训练集最差 10 张，`fast.py --no-vlm` + `FUSION_LLM=0`）`OUTSIDE=0` 均分 **30.81**
#   （引脚 0.1987、网络 0.2612、tp 引脚 44 / 网络 214）→ `OUTSIDE=40` 均分 **30.09**
#   （引脚 **0.1756**、网络 0.2574、tp 引脚 **42** / 网络 217）；10GT 9 张单独看引脚
#   0.3920 → **0.3432**。机制：`pin_name` 角色在循环里**排在 `pin_number` 之前**
#   （`:979-981`），放宽之后它先抢走框外那块文字，而 `taken_box`（`:978`）规定
#   「同一块文字只能当一次引脚名或引脚号」⇒ **引脚号被饿死**、元件键配不上，
#   丢掉的 `tp` 比补来的名字更贵。⇒ 所以改成 late pass，**对引脚号那一轮零影响**。
PINNAME_OUTSIDE = float(os.environ.get("STAGE2_PINNAME_OUTSIDE", "0"))
# 位号框的宽高比上限。**同一个坑的第三次踩**：`MAX_BOX_W=3.0` 会把 3 位数字的位号
# 整排踢出候选池 —— 位号最长 7 字符（`USB1234`），字高 18px 时 `C152` 宽 56px
# ⇒ 宽高比 3.11 > 3.0 ⇒ `wide=True` ⇒ `if t["wide"]: continue` 直接丢掉。
# 实测对应关系（`case 1`，GT `C151..C156` 在 x=66/183/301/419/566/714）：
#   `t0027`(x=105, 2.94) 不 wide ⇒ `C151` ✅ ｜ `t0024`(x=754, 2.86) 不 wide ⇒ `C156` ✅
#   `t0028`(x=223, 3.11) `t0029`(x=341, 3.06) `t0030`(x=458, 3.17) `t0031`(x=606, 3.06)
#   **全 wide ⇒ 这四颗全退化成 `pNNN`**（键配不上 ⇒ 引脚 + 网络引用一起丢）。
# 真正的位号过滤器是 `canon_designator`（只认 `[A-Z]{1,3}\d{1,4}`），宽高比只用来
# 挡「一排字」。6.0 覆盖到 ~7.7 字符，足够；`STAGE2_DESIG_MAXW=3.0` 可关回旧行为。
DESIG_MAXW = float(os.environ.get("STAGE2_DESIG_MAXW", "6.0"))
#: 位号候选的**距离硬上限**（px，0=关）：只挡「让位到邻居标签」的候选，宁可退回 pNNN。
DESIG_MAXD = float(os.environ.get("STAGE2_DESIG_MAXD", "0"))
# `STAGE2_PIN_ALONG`：引脚名/号绑定用**沿边距离**（左右边比 |Δy|、上下边比 |Δx|）
# 代替 2D 中心距。文字框朝外方向的偏移随文字长度变（VDD/VDDIO/VDDREG 一起右对齐），
# 那份噪声在 2D 距离里比沿边信号还大，实测会把整排引脚名上移一格。
# **默认已翻成开**（A/B：10GT 9 张 44.28→44.70、pin 0.1712→0.1881；
# 19 张 26.09→26.46、pin 0.0789→0.0935；component/netH/net_line 逐位不变）。
# 设 `STAGE2_PIN_ALONG=0` 可关回旧行为。
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★ 原来手写了 5 个词（缺 `no`/`disable`/`disabled`）—— 「差不多全」正是 drift 的
#   温床。词表只有一处权威：`eda-pcb\envflag.py:FALSE_WORDS`。
PIN_ALONG = env_on("STAGE2_PIN_ALONG", "1")
MAX_ALTS = 6         # 每个元件最多保留几个位号候选（交给阶段三仲裁）
# 结构化重绑定时「每配上一对」的固定加分（`stage2\\structmatch.py` 的 `cover`）：
# 它让「一条自洽、把整排引脚都配上、每对都稍远」的方案压过「只配最近的那几对」。
PIN_COVER = float(os.environ.get("STAGE2_PIN_COVER", "0.15"))
# ★★ 引脚号**残差回填**（`STAGE2_PIN_RESIDUAL`，默认 `0` = 关；>0 时是最大沿边距离，px）。
# 动机与实测见 `stage2\structmatch.py` 里 `rebind` 的那段长注释：同一排引脚号会**分组重启**
# （10GT case 1 的 `U1` 左列 `7 16 … 122` 之后又从 `8 67 15 9 66` 重新开始，y 544~750），
# `monotone_keep` 把重启组当成「破坏严格单调的离群」并清零它们的收益 ⇒ 这些号明明就在
# 候选池里、分数也不低（`8` .70 / `13` .77 / `14` .80 / `10` .81）却永远绑不上。
# 代价（`work\tools\dev\_numcover.py`，官方 9 张 120 个 GT `box` 引脚）：**24 个（20%）**
# 是「号读对了、脚也枚举出来了，就是没绑上」。只对 `pin_number` 角色开，`pin_name` 不动。
PIN_RESIDUAL = float(os.environ.get("STAGE2_PIN_RESIDUAL", "0"))
PIN_RESIDUAL_MIN = float(os.environ.get("STAGE2_PIN_RESIDUAL_MIN", "0"))
# ★★ 引脚列窗口引导（`stage1/pincol.py`）。默认 **on**。
# 开了之后，「第 i 个脚的号码 / 名字」直接由 pincol 算好的矩形（`out_win`/`in_win`）
# 做包含判定归位，不再靠 `structmatch` 的整排单调对齐去猜 —— 后者在引脚数太少时
# （改前 stage1 对 0143 的 U1 只给 8 个点、GT 要 64 个）会退化成「谁近归谁」。
# 只有命中率 ≥ `PINCOL_WIN_COVER`（相对两侧较短的那个长度）时才采信，否则整颗
# 退回老路径 —— 不让新机制把已经对的元件弄坏。
# ★★ 2026-10 决定：**默认改成 off**。
#
# 这条通道当年是为「stage1 对 0143 的 U1 只给 8 个点、GT 要 64 个」加的，它让
# 「第 i 个脚的号/名」直接由 `pincol` 的矩形归位，不再靠 `structmatch` 整排单调对齐。
# 但 `stage1\pincol.py` 修掉「引脚点落在号字格点而不是墨迹桩上」之后（见
# `work\reports\引脚号回归与桩偏移.md` 第 3 节），pincol 的窗口引导反而**压住了**
# `structmatch.rebind` 的正确绑定。
#
# 官方 10GT 9 张实测（同一基线，`work\tools\eval\fast.py`）：
#   默认（`PINCOL_GUIDE=1`）      54.09   pin F1 0.3232  tp 28   netH 0.4856
#   `STAGE2_PINCOL=0`            **55.07**  pin F1 0.3446  tp 31   netH 0.4974
#   ⇒ **+0.98**，其中 case `8` 从 50.42 → **58.40**（pin F1 0.0000 → 0.2000，
#     不再爆 0），case `5` +0.39；派生 10 张 16.50 vs 17.02（跑间噪声内）。
#
# 老路径（`structmatch`）在引脚点正确之后是更好的默认。
PINCOL_GUIDE = env_on("STAGE2_PINCOL", "0")
PINCOL_WIN_COVER = float(os.environ.get("STAGE2_PINCOL_COVER", "0.5"))
# 「抢读」半径（用户 m06973：字要挂到元件上，用贪心解决）。
# 原来每个文字框**只归最近的元件**，于是「C8 的位号框离邻居更近」时，
# 真正的主人连候选都看不到（实测 468 个「没给位号」里 22% 属于这种，
# 其中框在 80px 内的有 48 个）。`STAGE2_STEAL > 0` 时，位号候选额外允许
# 从「别人名下、但离自己 <= STEAL_R」的框里抢，抢的冲突交给后面的
# 全局去重（同一位号只能给一个元件，分高者得）。
# 只影响**位号**候选；参数 / 器件名 / 引脚仍然只用自己名下的框。
# 220 张实测（`work\tools\eval\stage2_eval.py`）：`STAGE2_STEAL=0` → 7.0495，
# `=80` → **7.2441**（component F1 0.0849→0.0875，tp 823→847），所以 80 是默认。
STEAL_R = float(os.environ.get("STAGE2_STEAL", "80"))
# 自由文字也进抢读池（见下面那段长注释）。默认开；`STAGE2_FREE_STEAL=0` 关。
FREE_STEAL = env_on("STAGE2_FREE_STEAL", "1")

# **值兜底**。和位号那条 STEAL 同一个病：阻容的「值」字离**邻居**的元件框更近时，
# 归属判定把它划给邻居，这颗元件连候选都没有。`STAGE2_VALUE_STEAL > 0` 时，
# 在自己名下找不到像值的框，就去 `d <= VALUE_STEAL_R` 的**任意**文字框里兜一个。
# 与位号不同：值不唯一（一片 0.1uF），所以抢错也不会比空着更差 ——
# 空着在官方口径里是 `text_mismatch`，**同时计 fp 和 fn**，一笔亏两处。
# 默认 40：220 张实测 `STAGE2_VALUE_STEAL=0` → 15.4801，`=40` → **15.5567**
# （component F1 0.1426→0.1451，tp 971→988，text_mismatch 948→931）。
VALUE_STEAL_R = float(os.environ.get("STAGE2_VALUE_STEAL", "40"))

# **器件名兜底**（`STAGE2_PART_STEAL`，半径；0 = 关）。与上面两条 STEAL 同一个病，
# 但更狠：IC 的型号字往往印在符号**外**、离邻居更近，于是被邻居的 slots 吃掉，这颗 IC
# 的器件名只能拿旁边一个垃圾串。10GT case 6 的 `U3`：全页明明读出了 `78M12`（conf 1.0），
# 却因为那块框属于邻居而落到 `7SWM12`。这里允许在自己名下没有「像器件名」的候选时，
# 去 `d <= STAGE2_PART_STEAL` 的抢读池里取一个（要求 `fit_partname == 1.0`、
# conf ≥ `STAGE2_PART_STEAL_CONF`、长度 ≥ 5），并扣 `STAGE2_PART_STEAL_PEN` 分。
PART_STEAL = float(os.environ.get("STAGE2_PART_STEAL", "0"))
PART_STEAL_CONF = float(os.environ.get("STAGE2_PART_STEAL_CONF", "0.55"))
PART_STEAL_PEN = float(os.environ.get("STAGE2_PART_STEAL_PEN", "0.35"))


# ---- §5.222 贴中罚（GT-blind；**已按 §5.222.10 的 A/B 转默认开**） -----------------
# 依据（`docs/RESULTS.md` §5.222.3）：位号形态的已认串按「离最近 GT 元件多远」量，
# 真位号 `d_raw`（中心距 / span）均值 **1.029**、非位号（引脚名/脚号/网名/值）**0.667**，
# 即「几乎压在元件中心的那条串」**更不像**位号；AUC 0.753（方向与现役 `near` 相反）。
# 但绝对半径会误伤小件：0402 的位号本来就贴着自己的符号（`d_raw≈0.3`），
# 故只在 `span >= NEAR_REV_MIN_SPAN` 的元件上开火（§5.222.7 分档后两带损失同时变小）。
# 剂量-反应（`work/_gen220_m24bind.py` v47–v56，读集 `m9_texts_sc_ab1.json`）：
#   `1.5,0.9,20` 时失败带 失3/得17（Δ+0.60pp，p=2.6e-03），matched 带 −7 ⇒ 净 +7/3565。
# 过闸（§5.222.10，210 案 × `--engine protocol2` 配对，产物 `work/_fast_out_nr220b|n`）：
#   均分 **30.1801 → 30.2863**，Δ=**+0.1063**（噪声底 ±0.02 的 5 倍），
#   配对 t **p=0.0441**、逐案 胜24/负12/平174（符号检验 p=0.0652，偏弱）⇒ 判据 Δ≥+0.05 且
#   p<0.05 **成立**，默认值改为 `1.5,0.9,20`。要退回旧行为设 `STAGE2_NEAR_REV=0`。
# ⚠ 命中方式是**同一批点位上换归属**（如 0132 的 SW2/SW3/SW4 轮转、0135 的 U1 在两颗间挪），
#   不是新救回位号；收益靠大引脚件（键对了引脚/网络跟着对）兑现，故逐案是 2:1 而非全胜。
def _near_rev_cfg() -> Tuple[float, float, float]:
    raw = os.environ.get("STAGE2_NEAR_REV", "").strip()
    if not raw:
        return 1.5, 0.9, 20.0
    try:
        parts = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return 1.5, 0.9, 20.0
    w = parts[0] if parts else 1.5
    r = parts[1] if len(parts) > 1 else 0.9
    m = parts[2] if len(parts) > 2 else 20.0
    return max(0.0, w), max(1e-6, r), max(0.0, m)


NEAR_REV_W, NEAR_REV_R, NEAR_REV_MIN_SPAN = _near_rev_cfg()


def _name_shape_pen() -> float:
    """器件名「字形闸」的罚分（`STAGE2_NAME_SHAPE`，**默认 0 = 关**）。

    §5.248 的离线审计（blk1 34 案、配上的 IC 37 颗）：我们的 `Name` 对 11 / 错 26，
    错的名字里 65% 是**根本不是型号**的串——`LDO`/`OUT`/`AGND`/`3.3V`/`PA11`，也有
    直接拿了**自己的位号**（`U7`/`U1`）。这条给出一个只依赖字符串形状的惩罚项：

      * 一个数字都没有（`LDO`、`OUT`、`AGND`、`VREFA+`）；
      * 形如「数字+电压/单位」（`3.3V`、`5V`）；
      * 与这颗元件已选定的**位号串完全相同**（`U1`、`U7`）。

    硬拒绝会把对的名字一起杀掉（审计里误杀 5/11），所以只做**排序惩罚**，让它退到
    第二位候选去。默认 0.0 ⇒ 逐位不变。
    """
    raw = (os.environ.get("STAGE2_NAME_SHAPE") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


NAME_SHAPE_PEN = _name_shape_pen()

# 「值」必须带单位/字母（纯数字不要）。理由见 `fit_value` 的 docstring。
#
# ★ 默认 **1（开）**。本条在 2025-10-04 曾被判「有害」并关掉（15.2514 vs 22.9068），
#   那是 **`PIN_STAMP`/`PIN_ALONG`/`STAGE1_DROP_TYPES` 之前的旧管线**；在 2025-11
#   的管线上重新 A/B，方向翻正了：
#   220 张（`work\_submit_out220r27` 默认 25.1643 → `work\_submit_out220r29` **25.2389**，
#     +0.0746；component F1 0.5339→0.5364，tp 5318→5351，text_mismatch 948→915）
#   10GT(9) 47.67 → **48.15**（component 0.7624→0.7786）
#   19 张   28.35 → **28.60**（component 0.5239→0.5322）
#   `pin`/`netH`/`net_line` 三个分项一个数没动（只动 component）。
#   机制（`_valread.py` + case `0001` 的 `C20`）：框内一个纯数字（往往就是引脚号）
#   靠 `fit_value("1")=1.0` + `inside` 加成 +0.15 + 距离 0 抢走「值」槽，
#   随后 `assemble.py` 的 `VALUE_LIKE` 又把它清成 null ⇒ 真正的值（`0.1uF`）
#   白丢。把它评 0 就把槽让回给真值。
#   ⚠️ `STAGE2_VALUE_TRIM=1` 叠加会吃掉这份收益：220 张 `UNIT=1 TRIM=1` → 25.1265
#   （−0.038 vs 默认），所以 TRIM 仍保持 0（见下）。
VALUE_UNIT_ON = env_on("STAGE2_VALUE_UNIT", "1")
_UNIT_RE = re.compile(r"[A-Za-z\u03a9\u2126\u00b5\u03bc]")

# 值串尾部削垃圾（`0.1uFLK` -> `0.1uF`）。理由见 `canon_value`。
# **实测略负，默认关**（2025-10-04，官方口径 22.9068 的管线上四臂对照）：
#   UNIT=0 TRIM=0 → **22.9068**（tp 5077，text_mismatch 950）
#   UNIT=0 TRIM=1 → 22.8757（tp 5066，text_mismatch 961）
#   UNIT=1 TRIM=0 → 22.6932（tp 5029，text_mismatch 998）
#   UNIT=1 TRIM=1 → 22.6037（tp 4998，text_mismatch 1029）
# 拆开看 TRIM=1 的净效果：`V 我们没写值(None)` 347→337（-10，有用），
# 但 `V 写了但写错` 266→275、`V GT 本来就是空、我们多写` 105→117（+21，有害）。
# 结论：**「值读不出来」不是后处理能修的**，别在这条路上再花时间。
VALUE_TRIM_ON = env_on("STAGE2_VALUE_TRIM", "0")

# 「网络名当值」那一支（`fit_value` 里的 `NETNAME_RE` ⇒ 0.6）。
# **默认 0（关）**，2025-10-04 官方口径 2×2 实测（`work\stage2_v15` 同一条 stage1）：
#   UNIT=0 NETNAME=1 → 23.6392  tp 5224  text_mismatch 969
#   UNIT=0 NETNAME=0 → **23.6682** tp **5232** text_mismatch **961**   <- 最好
#   UNIT=1 NETNAME=1 → 23.4097  tp 5176  text_mismatch 1017
#   UNIT=1 NETNAME=0 → 23.4617  tp 5189  text_mismatch 1004
# 收益不大（+0.029，8 颗元件）但方向干净、机理清楚：`_valnone.py` 显示
# 「GT 有值 / 我们 None」的 433 条里 **385 条是读到了却被 `VALUE_LIKE` 清成 null**，
# 其中 213 条的 text 就是引脚号 `1`。关掉这一支后网络名不再顶替值，
# 那 8 颗从「写出非空错值」回到「写 null」（GT 值为空时 null 是对的）。
# 注意 `UNIT=1` 单独开是有害的（见 `fit_value`），两者不能想当然地配对。
VALUE_NETNAME_ON = env_on("STAGE2_VALUE_NETNAME", "0")

# 「值 / 器件名」这一槽的距离权重（`_score` 的 `dist_w`，全局默认 0.035）。
# 动机（2025-11，`_valread.py` + `_submit_out220r30` 的 value 失配逐条）：1835 个
# r/c/l/d 配对里 `diff 479` 全是「两边都是值形串但挑了别人那条」—— 例如 `0001/C15`
# GT `0.1uF` 我们写 `1.5V`、`C20` GT `0.1uF` 我们写 `10K`（邻居电阻的值）。
# 当时怀疑是 `-0.035 * min(d, FAR=40)` 这个距离项太软（最多扣 1.4 分）。
# ⛔ **实测否证：加大它没用，还略负。** 19 张快测（2025-11，其余旋钮为当前默认）
#   0.035 → **28.60**（component 0.5322）｜0.10 / 0.25 / 0.50 / 1.00 → 28.58（0.5316），
#   四档**逐位相同**且都低于基线 ⇒ 候选集在 `owned` 分组那一步就已经被定死了，
#   距离项再大也改不了归属。**别在这条路上再花时间。** 开关留着只为记录这个负结果。
VALUE_DISTW = float(os.environ.get("STAGE2_VALUE_DISTW", "0.035"))

DESIGNATOR_RE = re.compile(r"^[A-Za-z]{1,4}\d{1,4}[A-Za-z]?$")
PINNUM_RE = re.compile(r"^[A-Za-z]{0,2}\d{1,3}[A-Za-z]?$")
VALUE_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)\s*[pnu\u00b5mkKMGR]?[FHV\u03a9AWRJ]?$")
NETNAME_RE = re.compile(r"^[+\-]?[A-Za-z_][A-Za-z0-9_+\-\.]{0,14}$")
PARTNAME_RE = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9_\-\./+]{3,30}$")
# **电压型网络名**（`3.3V` / `5V` / `1.8V`）：`NETNAME_RE` 要求首字符是字母，
# 所以电源网名**永远进不了**网络标签集 —— 而这恰恰是最该按名字合并的一类。
RAILNAME_RE = re.compile(r"^[+\-]?\d+(\.\d+)?\s*(V|mV|kV|v)$")

# ★ 网络标签候选的**尺寸闸门**（`STAGE2_NETLABEL_MIN_W/H`）。
# 病灶实测（`work\_submit_out220w\runs\stage2\_assigned\1.json`）：case `1` 的 202 个
# 文字框里 107 个「自由框」，其中 **94 个**被当成网络标签，而绝大多数是 **4x4 ~ 12x5 px
# 的「点」**（焊盘/结点被 `cvparts.textbox` 误框），CRNN 把它们读成 `4` / `R` / `V` / `1`。
# 致命之处不是「读错了」，而是**同一个垃圾读法出现几十次**（`4` 出现约 30 次），
# `assemble.merge_by_label` 按文本合并 ⇒ **几十条互不相干的网络被并成一条**。
# 真标签（`GND` 约 18x30 px）远大于这些点，尺寸闸门能干净地切掉它们。
NETLABEL_MIN_W = float(os.environ.get("STAGE2_NETLABEL_MIN_W", "12"))
NETLABEL_MIN_H = float(os.environ.get("STAGE2_NETLABEL_MIN_H", "9"))


def is_netname(s: str) -> float:
    """网络标签名字判定（替代原来 `NETNAME_RE or VALUE_RE` 那个口子极大的谓词）。

    原谓词的两个漏洞：`VALUE_RE` **匹配裸数字**（`4`/`1`/`2`），`NETNAME_RE` **匹配单个字母**
    （`V`/`R`/`K`）⇒ 被误框成文字的焊盘结点（CRNN 读成 `4`）能进标签集，而 `4` 重复出现
    几十次 ⇒ `merge_by_label` 把几十条无关网络并成一条。
    现在：长度必须 ≥2、不能是纯数字，且要么是名字型（`NETNAME_RE`）要么是**电压型**
    （`RAILNAME_RE`，即 `3.3V` 这类原来永远进不来的电源网名）。
    """
    s = (s or "").strip()
    if len(s) < 2 or s.isdigit():
        return 0.0
    if RAILNAME_RE.match(s):
        return 1.0
    if NETNAME_RE.match(s) and re.search(r"[A-Za-z]", s):
        return 1.0
    return 0.0

# type -> 允许的位号前缀（按官方 9609 个元件统计）
# ★ `"tp"` 是 `STAGE1_MERGE_TP=1` 时 `stage1/detect.py` 出的**内部合并类**
#   （`pin` + `testpoint`）。**必须有这一条**，否则 `canon_designator` 走到
#   `:402 allow = TYPE_PREFIX.get(ctype, ())` 为空的兜底支（保真度 0.5、且绕过
#   `PREFIX_MODE=strict`），位号绑不上 ⇒ 合并类拿不到 TP 号 ⇒ 会被出口闸门全丢。
#   两者本来就共用 `TP` 前缀（见上一行的 `pin` / 下一行的 `testpoint`）。
#   默认 off 时 `tp` 这个 ctype 根本不会出现 ⇒ 现有结果逐位不变。
TYPE_PREFIX: Dict[str, Tuple[str, ...]] = {
    "c": ("C",), "r": ("R",), "box": ("U", "USB", "u"), "d": ("D",),
    "led": ("D", "LED"), "pin": ("TP",), "switch": ("SW", "S"), "l": ("L",),
    "mosfet": ("Q",), "bjt": ("Q",), "crystal_2pin": ("X", "Y", "C"),
    "crystal_3pin": ("X", "Y"), "crystal_4pin": ("X", "Y"), "crystal": ("X",),
    "testpoint": ("TP",), "tp": ("TP",), "fuse": ("F",), "amp": ("U",),
    "amp_3pin": ("U",),
    "amp_5pin": ("U",), "bjt_npn": ("Q",), "bjt_pnp": ("Q",),
    "mosfet_npn": ("Q",), "mosfet_pnp": ("Q",), "v": ("TP",),
}
# 这些 type 的 key 是伪造号（∅N），纸上画的是网络名/参数，没有位号可言
NO_DESIGNATOR = frozenset({"gnd", "v", "net_input", "net_short", "net_output",
                           "other", "net_bidirection", "m3\u87ba\u4e1d"})

# ---------------------------------------------------------------------------
# ★ 内涵键回收（`STAGE2_CONFORM_KEY`）：位号前缀是元件类型的**内涵**，两者必须互证。
#
# 官方 10 张实测（`STAGE2_DBG_CID` 逐条打印 + `work/_fast_out_gen2v`）：
# **「未挂键 ⇒ 丢弃」这条闸门丢掉的那 4 颗 LED，全部是「读得没错、判型/取证错了」**：
#   · case 6 `c0016`/`c0017`：YOLO 判成 `d`，而印字五个角色全部读作 `LED2`/`LED1`
#     （conf 0.98~0.99，干净读法）。但 `TYPE_PREFIX["d"] = ("D",)` 且
#     `PREFIX_MODE=strict` ⇒ `canon_designator("LED2", "d")` 返回 `""` ⇒ 无候选
#     ⇒ `[dbg-alts] cid=c0016 -> []` ⇒ 未挂键 ⇒ 被丢。
#   · case 7 `c0016` / case 10 `c0018`：**只信 `designator` 角色**。标签框的
#     `designator` 读作 `YXEP1`(0.399)/`Y7EF1`(0.458) 这种垃圾，而**同一块框**的
#     `pin_name`/`pin_number` 读作 `7LED1`(0.698)/`N7ED1`(0.735) —— 真值就在框里，
#     只是被角色条件的读法差异挡在门外（`_best_read` 本来就允许串角色借读，
#     但读串「开头粘了胶连符号」，`7LED1` 过不了 `^([A-Z]{1,3})(\d{1,4})$`）。
#
# 两条回收路径都**只对「本来就要丢的元件」生效**（strict 路径失败的元件今天一颗
# 都不输出）⇒ 天然是零回归的加法；但仍要过 A/B 才能转正。
#
# 依据（前缀归属）：`LED` 这个前缀在 `TYPE_PREFIX` 里**只有 `led` 拥有**
# （`D` 是 `d` 与 `led` 共有 ⇒ 不足以翻类型）—— 这就是「基于维度内涵的校验」：
# 读出的前缀反过来判定元件类型，而不是靠 YOLO 的类别。
#
# ★ **已上线（默认 1）**：官方 10 张 `-vlm-pin off` A/B（`work/_fast_out_pzoff`
#   =0 与 `work/_fast_out_ckey` =1，同一批输入、逐案对比）
#   **80.98 → 81.14（+0.16）**；逐案**只有 case 6 变了**，其余 9 张总分与四个分项
#   逐位相同。变的就是这两颗 LED（`type` `d` → `led`，`bbox`/`pins` 完全不动）⇒
#   `6` 的 comp **0.9474 → 1.0000**（评测口径 `protocol2` 的 `component_text_ok`：
#   `d`/`led` **不比对 `Name`**，只比 `type` + bbox ⇒ 类型一改就 tp，与「型号」无关）。
#   这也是为什么「多余检测」里那条「`d`/`led` 的 `Name` 槽发出去必然判错」的担心
#   在 `protocol2` 下并不成立（它只在判题机复刻尺 `harness` 下成立）。
#   置 `STAGE2_CONFORM_KEY=0` 可回到旧行为（那时官方 10 张 80.98）。
CONFORM_KEY = env_on("STAGE2_CONFORM_KEY", "1")
# 内涵回收路径要求的最低置信度（strict 路径不受影响）。官方 case 6 的 `LED2`
# 是 0.992、case 7 的 `7LED1` 是 0.698 ⇒ 地板上限设在 0.5 一带。
CONFORM_MIN_CONF = float(os.environ.get("STAGE2_CONFORM_CONF", "0.50"))
# 削掉开头胶连字符（`7LED1` -> `LED1`）的保真度折扣、翻类型（`d` -> `led`）的折扣。
CONFORM_GLUE_PEN = float(os.environ.get("STAGE2_CONFORM_GLUE_PEN", "0.85"))
CONFORM_TYPE_PEN = float(os.environ.get("STAGE2_CONFORM_TYPE_PEN", "0.85"))
# 允许互相纠正的类型。**必须是显式小表**：只登记「图形符号本身几乎无法区分、
# 只能靠印字前缀区分」的对子（`d` 与 `led` 的差别只是两条出射箭头）。
CONFORM_SIBLINGS: Dict[str, Tuple[str, ...]] = {"d": ("led",), "led": ("d",)}
# ★ **恒开的「遗漏检测」（`STAGE2_CONFORM_AUDIT`，默认 1）**：不管回收开关开不开，
#   都把「本该挂上、却因为判型/取证不对而没挂上」的元件记进 `LAST_CONFORM`，
#   由 `stage2/run.py` 写进 `_assigned/<case>.json` 的 `no_delete.conform_audit`。
#   **只写旁路报告，绝不改输出** —— 这正是用户 m48732 要的「基于维度内涵的识别
#   与校验，用来检索遗漏/多余的元素」：与「不删除只削弱」的账本同一条通道。
CONFORM_AUDIT = env_on("STAGE2_CONFORM_AUDIT", "1")
CONFORM_AUDIT_MAX = int(os.environ.get("STAGE2_CONFORM_AUDIT_MAX", "400"))
LAST_CONFORM: List[dict] = []
# ★ 同一通道的**「多余检测」**：我方输出里那些「内涵上不可能判对」的元件 ——
#   官方 `10GT` 的命名约定是 `c/r/l` 的 `Name` = 位号、`d`/`led`/`box` 的 `Name`
#   = **型号**；而 `stage2/assemble.py` 只对 `box` 取器件名，其余一律 `name = key`。
#   于是**每发出一颗 `d`/`led`，它就是一个必然判错的「多余元素」**（官方按
#   key+type+Name+value+bbox 全对才算 tp）—— 这正是用户 m48732 要的另一半：
#   检索「多余的元素」。记录里同时带上器件名读数，便于换约定时直接用。
#   **只写旁路报告**，见 `stage2/run.py` 的 `no_delete.conform_extra`。
LAST_CONFORM_EXTRA: List[dict] = []
# ★ 逐框读法全量导出（`STAGE2_DUMP_READS=1`，默认关）：把每个文字框的**所有角色读法**
#   落到 `_assigned/<case>.json` 的 `no_delete.text_reads`。用于回答「位号读不出来」
#   到底是检测没出框、还是出了框但读错 —— 只写旁路，交付物字节不变。
DUMP_READS = env_on("STAGE2_DUMP_READS", "0")
LAST_TEXT_READS: List[dict] = []
# ★ 远处位号认领（见 `assign()` 里同名注释块）：半径像素，0=关。
_FAR_KEY = float(os.environ.get("STAGE2_FAR_KEY", "0") or 0.0)
_FAR_CONF = float(os.environ.get("STAGE2_FAR_CONF", "0.50") or 0.50)
_FAR_TYPES = tuple(x.strip() for x in
                   os.environ.get("STAGE2_FAR_TYPES", "box").split(",") if x.strip())
# `STAGE2_FAR_TYPES=all` ⇒ 除 `NO_DESIGNATOR` 之外全类型（电阻/电容/晶振也要能认领：
# `work/_elsewhere.py` 的 B 组 111 颗里有 **53 颗是「键为空、但该串整页读到了」**，
# 绝大多数是被动件，只放开 `box` 就会把它们全漏掉）。
if any(x.lower() == "all" for x in _FAR_TYPES):
    _FAR_TYPES = None
# 只有「现键文字框离元件 > 这个像素」的元件才允许被远处认领**替换**
_FAR_REPL = float(os.environ.get("STAGE2_FAR_REPL", "40") or 40.0)
# 官方约定下 `Name` 应当放**型号**（而非位号）的类型；`box` 已经这么做了。
EXTRA_MODELNAME_TYPES = ("d", "led")


def clear_conform() -> None:
    LAST_CONFORM.clear()
    LAST_CONFORM_EXTRA.clear()


# ---------------------------------------------------------------------------
def dist_to_rect(px: float, py: float, r: Sequence[float]) -> float:
    dx = max(r[0] - px, 0.0, px - r[2])
    dy = max(r[1] - py, 0.0, py - r[3])
    return (dx * dx + dy * dy) ** 0.5


def dist_rect_to_rect(a: Sequence[float], b: Sequence[float]) -> float:
    """两个矩形之间的**间距**（重叠/相切时为 0）。

    与 `dist_to_rect`（点→矩形）的区别是它**与矩形自身的大小无关**：
    「一块文字贴不贴在这颗元件上」只应取决于两者之间还有多少空白，
    而不该取决于这块文字有多宽。见 `OWN_BBOX` 处的实测动机。
    """
    dx = max(b[0] - a[2], 0.0, a[0] - b[2])
    dy = max(b[1] - a[3], 0.0, a[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


def _too_wide(t: dict, lim: float) -> bool:
    """文字框是否宽到不可能是一个位号（`lim` = 宽高比上限）。

    ⚠️ **不要用布尔量 `t["wide"]`（= 宽高比 > `MAX_BOX_W`(3.0)）当位号的闸门**：
    3 位数字的位号（`C152`/`R104`）在 18px 字高下宽 ≈56px ⇒ 宽高比 ≈3.1，
    会被整排判成 `wide` 丢掉。见 `DESIG_MAXW` 处的实测对应关系。
    引脚名 / 器件名通道早就各自放宽过（`PINNAME_MAXW`、`assign.py` 里
    「IC 的器件名本来就长」那段注释），位号通道是最后一个。
    """
    return (t.get("wratio") or 0.0) > lim


def inside(px: float, py: float, r: Sequence[float]) -> bool:
    return r[0] <= px <= r[2] and r[1] <= py <= r[3]


def _letter(text: str) -> str:
    return "".join(ch for ch in text if ch.isalpha())


# CRNN 在**数字位**上最常犯的形近错（`R14` -> `Rl4` / `R1O` / `RI4`）。
# 只在「字母前缀已经对上类型」时才允许把尾部这些字符读成数字，见 `canon_designator`。
_FIX_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "U": "0",
                            "I": "1", "L": "1", "T": "7",
                            "S": "5", "B": "8", "Z": "2", "G": "6"})


def _digitize(tail: str) -> str:
    """把尾部里的形近字母换成数字；只要还剩非数字就返回 `""`（整段不可信）。"""
    d = tail.translate(_FIX_DIGIT)
    return d if d.isdigit() else ""


def canon_designator(text: str, ctype: str):
    """把读出的串规范成位号，返回 `(位号, 前缀保真度)`；认不出返回 `("", 0.0)`。

    CRNN 在位号上最常见的错就是**首字母**（`R14` -> `Rl4` / `R?4` / `Bl4`），
    而位号的首字母由元件类型决定（`r` 一定是 `R`）——所以「前缀矫正」是无损的：
    只要尾部是 1~4 位数字，就把字母部分换成该类型的首选前缀。实测（`work/_des_ceiling.py`，
    20 张官方图 / 258 个有位号的 GT 元件）识别天花板从 **0.500 抬到 0.725**。

    要求字母部分**非空**：纯数字（引脚号「14」）不能当位号，否则每个引脚号都会变成候选。

    **第二类错是「数字位被读成形近字母」**（`R14` -> `Rl4`，`Q1` -> `O1`）。
    原实现整个走 `^([A-Z]{1,3})(\\d{1,4})$`，`Rl4` 会被切成前缀 `RL` + 数字 `4`，
    前缀对不上 ⇒ `PREFIX_MODE="strict"` 直接返回 `""`（这就是那个「真 bug」）。
    现在加一档：前缀**贪心对不上时**，退一步用 `_digitize` 把尾部当数字重解
    （`Rl4` -> `R`+`14`），置信度给 0.9（是纠错，比干净读法低一档）。
    """
    u = re.sub(r"[^0-9A-Za-z]", "", text or "").upper()
    if not u:
        return "", 0.0
    allow = [a.upper() for a in TYPE_PREFIX.get(ctype, ())]
    if not allow or PREFIX_MODE == "off":
        m = re.match(r"^([A-Z]{1,3})(\d{1,4})$", u)
        return (m.group(1) + m.group(2), 0.5) if m else ("", 0.0)

    # ① 干净读法：字母前缀 + 纯数字（原有行为，优先级最高）
    m = re.match(r"^([A-Z]{1,3})(\d{1,4})$", u)
    if m:
        got, digits = m.group(1), m.group(2)
        if got in allow:
            return got + digits, 1.0

    # ② 前缀对、尾部含形近数字：`Rl4` -> `R14`
    for nlet in (1, 2, 3):
        if len(u) <= nlet or nlet > 3:
            continue
        head, tail = u[:nlet], u[nlet:]
        if head not in allow:
            continue
        d = _digitize(tail)
        if d and len(d) <= 4:
            return head + d, 0.9

    # ②.5 前缀**首字母被吃掉**（`TP21` -> `P21`）：读出的字母是某个允许前缀的后缀
    #      就补回完整前缀。只在 `STAGE2_PREFIX_SKEL=1` 时生效（默认关，见上方常量）。
    if PREFIX_SKEL and m:
        cand = [a for a in allow if len(a) > len(m.group(1)) and a.endswith(m.group(1))]
        if cand:
            # 取最短的候选前缀（`P` -> `TP` 而不是 `USBTP` 那种多义）
            return min(cand, key=len) + m.group(2), 0.7

    if PREFIX_MODE == "strict":
        # 只信「前缀已经在预期集合里」的读法。不做强制改名 —— 强制改名会把
        # 引脚名（`P_31` -> `U31`）也变成本来不存在的位号，凭空造键比漏键更伤。
        return "", 0.0

    # ③ force：前缀不属于该类型，强制改成首选前缀
    if m:
        return allow[0] + m.group(2), 0.45
    for nlet in (1, 2, 3):
        if len(u) <= nlet:
            continue
        d = _digitize(u[nlet:])
        if d and len(d) <= 4:
            return allow[0] + d, 0.35
    return "", 0.0


def fit_designator(text: str, ctype: str) -> float:
    return canon_designator(text, ctype)[1]


def prefix_owners(prefix: str) -> List[str]:
    """哪些 type 的 `TYPE_PREFIX` 里有这个前缀（用于「前缀是否唯一确定类型」）。"""
    p = (prefix or "").upper()
    if not p:
        return []
    return [t for t, ps in TYPE_PREFIX.items()
            if p in tuple(str(x).upper() for x in ps)]


def canon_designator_meaning(text: str, ctype: str,
                             rescue: Optional[bool] = None):
    """内涵互校验版的 `canon_designator`：`(位号, 保真度, 类型改写, 依据)`。

    `依据` ∈ `{"strict", "glue", "type", ""}`：
      · `strict` —— 原路径（`canon_designator` 直接认得），**行为与今天逐位相同**；
      · `glue`   —— 削掉读串开头粘上的非字母垃圾后认得（`7LED1` -> `LED1`）；
      · `type`   —— 该串在**另一种类型**的 `TYPE_PREFIX` 下才是干净位号，且所用
                    前缀**只被那一种类型拥有** ⇒ 采纳该键并**把类型改成它**
                    （`LED2` 在 `d` 上认不出、在 `led` 上是干净位号 ⇒ 类型 `d` 判错）。

    `CONFORM_KEY=0`（默认）时，非 strict 的两条路径一律不生效 ⇒ 逐位不变。
    `rescue=True` 是给**审计**用的：不管开关，都把两条救援路径跑一遍（只写报告，
    用来检索「读得没错、却因为判型/取证不对而没挂上键」的遗漏元件）。
    """
    can, fid = canon_designator(text, ctype)
    if can:
        return can, fid, None, "strict"
    if not (CONFORM_KEY if rescue is None else rescue):
        return "", 0.0, None, ""
    u = re.sub(r"[^0-9A-Za-z]", "", text or "").upper()
    if not u:
        return "", 0.0, None, ""
    # (a) 开头胶连：`7LED1` -> `LED1`（只削**开头**的非字母；`N7ED1` 开头是字母，
    #     削不动 ⇒ 老老实实认不出，见下方自检里的反例断言）。
    u2 = re.sub(r"^[^A-Z]+", "", u)
    if u2 and u2 != u:
        can, fid = canon_designator(u2, ctype)
        if can:
            return can, fid * CONFORM_GLUE_PEN, None, "glue"
    # (b) 前缀唯一归属另一种类型 ⇒ 翻类型（`LED` 只属于 `led`）。
    for ty in CONFORM_SIBLINGS.get(str(ctype).lower(), ()):
        for s in ((u2,) if u2 else ()) + (u,):
            can, fid = canon_designator(s, ty)
            if not can or fid < 0.9:
                continue
            m = re.match(r"^([A-Z]{1,3})", can)
            ow = prefix_owners(m.group(1) if m else "")
            if len(ow) != 1 or ow[0] != ty:
                continue
            return can, fid * CONFORM_TYPE_PEN, ty, "type"
    return "", 0.0, None, ""


def canon_value(text: str) -> str:
    """把值串**尾部被 OCR 粘上的垃圾**削掉：`0.1uFLK` -> `0.1uF`。

    220 张实测（`work/tools/diag/_textmiss.py`）：键对上的 2280 个元件里
    `text_mismatch` 950 条，其中 **c/r 的「值」占 718（76%）**，而值失配里
    `我们没写值(None)` **347** 条 + `写了但写错` 266 条。
    逐案看图（`work/tools/viz/_valcrop.py`，`_valmiss_0001.png`）**值字明明印在
    图上、框也被检出了**：`C12` 的值框 `[332,947,375,959]` 读出
    `value='0.1uFLK'(0.68)` —— CRNN 把右侧相邻笔画（`%`->`K` 之类）粘在值后面，
    `VALUE_RE` 因尾随垃圾整体失配 ⇒ `fit_value` 给 0 分 ⇒ **这个框连候选都不是**，
    于是 `assign` 只能拿更远的「耐压行」（`16V` 那行读成 `1.51`）顶上，
    而 `assemble.VALUE_LIKE` 又嫌 `1.51` 没有字母、把它丢成 `null` —— 两头落空。

    做法：取**能完整匹配 `VALUE_RE` 的最长前缀**，削掉的尾巴不超过 3 个字符。
    """
    t = (text or "").strip()
    if VALUE_RE.match(t):
        return t
    for cut in range(1, 4):
        if len(t) - cut < 2:
            break
        p = t[:len(t) - cut].rstrip()
        if VALUE_RE.match(p):
            return p
    return ""


def fit_value(text: str) -> float:
    """r/c 的「值」适配度。

    **纯数字一律不要**（`STAGE2_VALUE_UNIT=1` 时）。GT 里 3294 个 r/c 值中
    一个纯数字都没有（`has_letter 2327` + `None 966` + `10μ` 1），而引脚号
    `1`/`2`/`4` 会被 `VALUE_RE` 满分匹配上、在引脚号框离得更近时抢走「值」
    这个角色 —— `_val_why.py` 的头号失配就是 `c: 100nF -> 1`(36)、`r: None -> 1`(18)。
    """
    if VALUE_UNIT_ON and not _UNIT_RE.search(text):
        return 0.0
    if VALUE_RE.match(text):
        return 1.0
    if VALUE_TRIM_ON and canon_value(text):
        # 削过尾巴：比干净匹配略低，但**足以赢过远处那行耐压**。
        # 距离项是 `-0.035*min(d,40)`，值框通常贴在符号边（d≈1~5），
        # 而耐压行 / 容差行在 d≈13~25，差值 0.3~0.8 分，0.1 的让步吃得下。
        return 0.9
    # 位号形状（`C5` / `R14` / `U2`）**不能当值**：`NETNAME_RE` 会放行它们
    # （字母开头 + 含数字 ⇒ 0.6），而位号框通常紧贴符号、距离项又是主导项，
    # 于是「值」这个角色被自己的位号抢走 —— 冒烟时 `C5` 的值一度变成 `20K`，
    # 反过来也有 `100nF -> 1` 这种（引脚号抢值）。位号由 `canon_designator`
    # 那条通道负责，这里直接拒掉。
    if DESIGNATOR_RE.match(text):
        return 0.0
    # 网络名当值（`VCC` / `Vout` / `SDA1`）。这一支存在的意义是给「值不像标准
    # 单位串」的少数器件兜底，但它同时会让**网络名顶掉真正的值**。
    # `work\tools\diag\_valnone.py` 量过：把纯数字（引脚号 `1`）从值候选里剔掉
    # 而**不**关这一支，`text_mismatch` 反而 969→1017、tp 5224→5176 —— 因为被剔掉的
    # 引脚号原本「占着坑」，`VALUE_LIKE` 再把它清成 null（GT 值为空的 104 条因此正确），
    # 剔掉后网络名补位、写出非空错值。两者必须一起关。默认 `1`（保持历史行为）。
    if VALUE_NETNAME_ON and len(text) <= 16 and NETNAME_RE.match(text) \
            and any(c.isdigit() for c in text):
        return 0.6
    return 0.0


# ★ 值轮的「单位族」闸门（`STAGE2_VALUE_FAMILY`，默认 0 = 关）。
#
# 病（200 池前 10 案的值失配 top）：`16V|0.1uF` ×7、`6.3V|4.7uF` ×2、`50V|15pF` ×2 ——
# 电容符号旁边的文字是**一列三行**（位号 / 容值 / 耐压 / 容差），几何上耐压行往往比容值行
# 离符号更近，于是「值」槽抓到了 `16V`。类型给了可判据：c 的值带 `F`、l 带 `H`、
# r 带 `Ω/k/M/R/%`、晶振带 `HM`、二极管带 `V/A`。族不符的读法打 0.55 折，让同族候选胜出；
# 若同族一个候选都没有，打折后它仍然会赢（不打成 0，避免整条值丢掉）。
VALUE_FAMILY_ON = env_on("STAGE2_VALUE_FAMILY", "0")
VALUE_UNITX_ON = env_on("STAGE2_VALUE_UNITX", "0")
VALUE_FAMILY = {"c": "Ff", "l": "Hh", "r": "\u03a9\u2126kKmMR%",
                "crystal": "HM", "crystal_2pin": "HM", "crystal_3pin": "HM",
                "crystal_4pin": "HM", "fuse": "AF", "d": "VA"}


def fit_value_typed(text: str, ctype: str) -> float:
    s = fit_value(text)
    if not VALUE_FAMILY_ON or s <= 0.0:
        return s
    fam = VALUE_FAMILY.get(str(ctype or ""))
    if not fam:
        return s
    if any(ch in fam for ch in str(text or "")):
        return s
    return s * 0.55


#: 单位 → 它属于哪一类器件的「值」（`STAGE2_VALUE_UNITX`，默认 0 = 关）。
#: 池子实测（§5.345）阻容值失配 92/179 里，头号错法是**拿耐压行当值**：
#: `4.7uF -> 6.3V` / `0.1uF -> 16V` / `15pF -> 50V` / 晶振 `12MHz -> 5p`。
#: 这些串的「像值」分数与真值一样是满分，胜负只看距离，而耐压行常常更近。
#: `VALUE_FAMILY` 的 0.55 只降权不否决，实测压不住；这里按**单位语义**直接否决。
UNIT_CLASS = {"F": "cap", "H": "ind", "Hz": "xtal", "\u03a9": "res", "R": "res",
              "V": "volt", "A": "amp", "W": "watt", "%": "pct"}
FOREIGN_UNIT = {"c": {"volt", "amp", "watt", "pct", "ind", "xtal", "res"},
                "l": {"volt", "amp", "watt", "pct", "cap", "xtal", "res"},
                "r": {"volt", "amp", "watt", "pct", "cap", "ind", "xtal"},
                "crystal": {"volt", "amp", "watt", "pct", "cap", "res"},
                "crystal_2pin": {"volt", "amp", "watt", "pct", "cap", "res"},
                "crystal_3pin": {"volt", "amp", "watt", "pct", "cap", "res"},
                "crystal_4pin": {"volt", "amp", "watt", "pct", "cap", "res"}}
_UNIT_SUF = re.compile(r"(Hz|F|H|\u03a9|R|V|A|W|%)\s*$")


def is_foreign_value(text: str, ctype: str) -> bool:
    """读数是不是「别类器件的值」（`6.3V` 之于电容）。`VALUE_UNITX` 与
    `VALUE_FILL` 共用这一个判定。"""
    bad = FOREIGN_UNIT.get(str(ctype or ""))
    if not bad:
        return False
    m = _UNIT_SUF.search(str(text or "").strip())
    return bool(m and UNIT_CLASS.get(m.group(1)) in bad)


def fit_value_unitx(text: str, ctype: str) -> float:
    s = fit_value_typed(text, ctype)
    if not VALUE_UNITX_ON or s <= 0.0:
        return s
    if is_foreign_value(text, ctype):
        return 0.0
    return s


#: 值补候选半径（`STAGE2_VALUE_FILL`，默认 0 = 关）。`VALUE_UNITX` 只会**否决**
#: 耐压行，否决完那颗元件就空了；真正的容值框被归属判给了邻居，所以要按本类
#: 单位在半径内再捞一次。收益尚未过噪，故不进默认（§5.345）。
VALUE_FILL_R = float(os.environ.get("STAGE2_VALUE_FILL", "0"))
VALUE_FILL_GAIN = float(os.environ.get("STAGE2_VALUE_FILL_GAIN", "0"))


def fit_partname(text: str) -> float:
    """器件名适配度。**「期望」（`expect.py`）在这里第一次真正起作用**。

    原来的版本开头就是 `if DESIGNATOR_RE.match(text): return 0.0` —— 而
    `DESIGNATOR_RE = ^[A-Za-z]{1,4}\\d{1,4}[A-Za-z]?$` **正好也匹配 `RP2350B`**
    （`RP` + `2350` + `B`），于是用户亲手举的那个例子被一刀切掉。现在改成先问
    「按前缀该不该是个长名字」，再用 `fragment_score` 压掉「一看就是被切下来的一块」
    的候选（`P1C1` / `GP/2` / `C1H+` / `1.0KF`）。
    """
    if len(text) < 4:
        return 0.0
    lo, _hi, kind = expect_kind(text)
    if kind == "designator" and DESIGNATOR_RE.match(text):
        return 0.0                      # 形态与期望一致地指向「位号」
    base = 1.0 if PARTNAME_RE.match(text) else 0.3
    return base * (1.0 - 0.6 * fragment_score(text))


def fit_pinnum(text: str) -> float:
    if not PINNUM_RE.match(text):
        return 0.0
    return 1.0 if text.isdigit() else 0.7


def fit_pinname(text: str) -> float:
    if not text or len(text) > 20:
        return 0.0
    if PINNUM_RE.match(text) and text.isdigit():
        return 0.15
    if PINNAME_KEYLIKE and _PINNAME_KEY_RE.match(text):
        return PINNAME_KEYLIKE
    return 1.0


# ---------------------------------------------------------------------------
def _score(fit: float, conf: float, d: float, inside_flag: bool,
           dist_w: float = 0.035, conf_w: float = 0.6,
           dist_clamp: float = FAR) -> float:
    """`dist_clamp` 是距离项的**截断上限**。

    历史 bug（2026-07 审计）：这里写死 `min(d, FAR)`，而 `FAR = 40`。可位号通道的
    候选池除了 `owned`（`d <= FAR`）还有 `steal`/自由文字（半径 `STEAL_R = 80`），
    于是**凡是 `d >= 40` 的候选，距离项一律被截成同一个常数**，评分完全失去分辨力，
    `assign()` 末尾那句 `sorted(desig_cands, key=lambda x: -x[2])` 的全局贪心退化成
    「按 `components` 的列表顺序先到先得」。

    实测（10GT case 1 的电容排）：

      | 候选 | 归属者 | 真实 d | 修复前分 | 修复后分 |
      |---|---|---:|---:|---:|
      | `C151` 的印刷标签 `[104,147,157,165]` | `c0008`（=GT `C151`） | 44.5 | **-0.469** | **-0.627** |
      | 同上 | `c0001`（=GT `C152`） | 52.7 | **-0.469** | -0.914 |

    两个分数一模一样 ⇒ `c0001` 因为下标更小把标签抢走，`C151` 这颗电容退化成 `p008`，
    紧接着 `C152`/`C153`/`C154`/`C155` 整排连锁错位（键配不上 ⇒ 引脚与网络引用一起丢）。

    **这个改动是严格安全的**：`g(d) = 常数 - dist_w * min(d, dist_clamp)` 只是把原来
    在 `d >= FAR` 处压平的函数**单调延长**到 `dist_clamp`，所以对任意 `d1 < d2`
    都有 `g(d1) >= g(d2)`，**绝不会翻转任何原有的严格次序**，只把原本并列的候选
    按真实远近重新排开。`dist_clamp` 默认仍是 `FAR`，其余通道逐位不变。
    """
    return fit * (1.0 - conf_w + conf_w * max(0.0, min(1.0, conf))) \
        - dist_w * min(d, dist_clamp) + (0.15 if inside_flag else 0.0)


_BORROW_PREF = 0.0 if (os.environ.get("STAGE2_NO_BORROW") or "").strip() not in ("", "0") else 0.85


def _best_read(reads: Dict[str, Tuple[str, float]], role: str,
               fit) -> Optional[Tuple[str, float, float]]:
    """在 5 种读法里取该角色适配度最高的一次。也允许串角色借读
    （例如 `pin_name` 框在 `designator` 角色下读出来更像），因为角色条件只是弱条件。"""
    best = None
    for r, (txt, conf) in reads.items():
        if not txt:
            continue
        f = fit(txt)
        if f <= 0:
            continue
        pref = 1.0 if r == role else _BORROW_PREF
        s = f * pref * (0.3 + 0.7 * max(0.0, min(1.0, conf)))
        if best is None or s > best[0]:
            best = (s, txt, float(conf), f)
    return best


def _box_key(info: dict):
    """文字框的**物理身份**：有 `box_i`（跨元件唯一的文字框序号）就用它，
    没有（`-1`）才退回 `canon` 字符串。读不出 canon 也没有 box_i ⇒ 返回 `None`。

    ★ 去重必须认**身份**而不是只认**读法**：同一块文字框被两颗元件读到两个不同
    的 canon 时，按 canon 去重会把这块**同一个物理标签**同时发给两颗元件
    （官方 case 2 的 `c0009`/`c0013` 就是这种「一块框、两种读法」的邻居）。
    """
    try:
        bi = int(info.get("box_i", -1))
    except Exception:
        bi = -1
    if bi >= 0:
        return ("box", bi)
    can = info.get("canon")
    return ("canon", can) if can else None


def _global_match(cands: Sequence[Tuple[str, str, float, dict]],
                  comp_ids: Sequence[str], thr: float = 0.02) -> List[Tuple[str, dict]]:
    """（元件 × 文字框）一对一最大权匹配，返回 `[(cid, 证据)]`。

    为什么不逐元件取最近框：**「读得出正确位号的元件里，最近框正好是它的只有 0.55**」
    （`work/_des_ceiling.py`）——图形符号常被 `textbox` 当文字框检出而且正好压在元件中心，
    逐元件独立取最近必然踩它。而位号在一张图里天然唯一，做成全局一对一能把
    「同一个框被两个元件抢」和「一个好框被符号挤掉」同时治掉。
    """
    if not cands:
        return []
    cid_i = {c: i for i, c in enumerate(comp_ids)}
    box_ids = sorted({int(x[3].get("box_i", -1)) for x in cands})
    bi = {b: j for j, b in enumerate(box_ids)}
    m = np.full((len(comp_ids), len(box_ids)), -1e9)
    for cid, _val, sc, info in cands:
        i, j = cid_i.get(cid), bi.get(int(info.get("box_i", -1)))
        if i is None or j is None:
            continue
        if sc > m[i, j]:
            m[i, j] = sc
    pairs: List[Tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-m)
        pairs = [(int(a), int(b)) for a, b in zip(r, c)]
    except Exception:                                   # 没有 scipy 就贪心兜底
        taken_r, taken_c = set(), set()
        for idx in np.argsort(-m, axis=None):
            i, j = int(idx // m.shape[1]), int(idx % m.shape[1])
            if i in taken_r or j in taken_c or m[i, j] <= -1e8:
                continue
            taken_r.add(i)
            taken_c.add(j)
            pairs.append((i, j))
    best: Dict[str, Tuple[float, dict]] = {}
    for i, j in pairs:
        if i >= len(comp_ids) or m[i, j] < thr:
            continue
        cid = comp_ids[i]
        jj = box_ids[j]
        for c2, _val, sc, info in cands:
            if c2 == cid and int(info.get("box_i", -1)) == jj:
                if cid not in best or sc > best[cid][0]:
                    best[cid] = (sc, info)
                break
    return [(cid, v[1]) for cid, v in best.items()]


def _own_assign_pick(desig_all: Dict[str, List[dict]]) -> List[Tuple[str, dict]]:
    """★ `STAGE2_OWN_ASSIGN`：把「元件 × 位号文字框」整体解一次**一对一**。

    `desig_all[cid]` 是每颗元件的**全部**位号候选（`stage2` 自己算好的，阶段三仲裁也吃它），
    每条带 `box_i`（文字框序号，跨元件唯一）/ `box` / `canon` / `score`。

    与现状的区别只有一步：现状是「每颗元件各自挑最优 + 按 `canon` 字符串贪心去重」，
    这里是「一块文字框只能配给一颗元件，由 `cvparts.own.assign`（复用官方
    `schematic_parser.metrics._assign`）求整体最小代价」。`canon` 唯一这道去重**保持不变**，
    因为真图上位号确实唯一。

    实测结论见 `OWN_ASSIGN` 常量处的注释：**判负**（现状 2 对 0 错，这一臂 0 对 2 错）。
    """
    from cvparts import own as own_mod

    def _key(info: dict):
        bi = int(info.get("box_i", -1))
        if bi >= 0:
            return bi
        bx = info.get("box")
        return tuple(round(float(v), 2) for v in bx) if bx else None

    score: Dict[Tuple[str, object], float] = {}
    info_of: Dict[object, dict] = {}
    items: List = []
    seen: set = set()
    for cid, alts in desig_all.items():
        for al in alts or []:
            k = _key(al)
            if k is None:
                continue
            score[(cid, k)] = float(al.get("score") or 0.0)
            if k not in seen:
                seen.add(k)
                info_of[k] = al
                bx = al.get("box") or (0.0, 0.0, 0.0, 0.0)
                items.append(own_mod.Party("tb|%s" % (k,), tuple(float(v) for v in bx), "text"))
    if not items:
        return []

    def _kk(pid: str):
        s = pid[3:]
        try:
            return eval(s)  # 只吃我们自己写出来的 int / (..) 字面量
        except Exception:
            return None

    cl = [own_mod.Party(cid, (0.0, 0.0, 0.0, 0.0), "comp")
          for cid in desig_all if any(_key(al) is not None for al in (desig_all.get(cid) or []))]
    got, _led = own_mod.assign(
        cl, items,
        gate=lambda c, i: (c.id, _kk(i.id)) in score,
        cost=lambda c, i: -score[(c.id, _kk(i.id))])

    first: List[Tuple[float, str, dict]] = []
    for cid, iid in got.items():
        al = info_of.get(_kk(iid))
        if al:
            first.append((float(al.get("score") or 0.0), cid, al))
    first.sort(key=lambda x: -x[0])
    picked: List[Tuple[str, dict]] = []
    seen_canon: set = set()
    for _sc, cid, al in first:
        can = al.get("canon")
        if not can or can in seen_canon:
            continue
        seen_canon.add(can)
        picked.append((cid, al))
    return picked


def _own_match_pick(desig_all: Dict[str, List[dict]]) -> List[Tuple[str, dict]]:
    """★ `STAGE2_OWN_MATCH`：带**显式弃权**的（元件 × 位号文字框）最大权一对一匹配。

    候选池 = `desig_all`（每颗元件的 `desig_all` 条目，就是 `owned + steal` 那批
    已经算好的候选，**不新造任何候选生成**）。权值 = 条目上的 `score`，而它正是
    贪心路径用的同一个 `_score(fit, conf, d, inside, dist_clamp=max(FAR, STEAL_R))`
    （`read_score` 的 `fit * (1-confit + confit*conf)` 项 × 距离衰减 × inside 加成），
    所以这一臂与现状**只差「谁来选」**，不差打分。

    矩阵：`n` 行（元件）× `m + n` 列（`m` 块文字框 + **每颗元件一列专属弃权列**）。
      · `mat[i, j] = score(元件 i, 文字框 j)`（没候选 ⇒ `-1e9`）
      · `mat[i, m+i] = OWN_MATCH_ABSTAIN * max_j score(i, j)`（弃权分）
      · 别人那列弃权 = `-1e9` ⇒ 每颗元件只能弃自己的权
    `linear_sum_assignment` 求全局最大权；列唯一 ⇒ 文字框最多用一次、元件最多拿一个
    位号；弃权列的权值 ⇒ 只有**严格优于弃权**的候选才会被吃掉。
    （等价写法：方形矩阵再补 `m` 行「未配对的框」，权值 0。）
    """
    cids = list(desig_all)
    if not cids:
        return []
    bidx: Dict[object, int] = {}
    boxes: List[object] = []
    for cid in cids:
        for al in desig_all.get(cid) or []:
            k = _box_key(al)
            if k is not None and k not in bidx:
                bidx[k] = len(boxes)
                boxes.append(k)
    n, m = len(cids), len(boxes)
    if m == 0:
        return []
    NEG = -1e9
    mat = np.full((n, m + n), NEG)
    info_of: Dict[Tuple[int, int], dict] = {}
    abstain: List[float] = []
    for i, cid in enumerate(cids):
        alts = desig_all.get(cid) or []
        best = 0.0
        for al in alts:
            k = _box_key(al)
            if k is None:
                continue
            j = bidx[k]
            w = float(al.get("score") or 0.0)
            if (i, j) not in info_of or w > mat[i, j]:
                mat[i, j] = w
                info_of[(i, j)] = al
            if w > best:
                best = w
        abstain.append(OWN_MATCH_ABSTAIN * max(0.0, best))
        mat[i, m + i] = abstain[-1]

    pairs: List[Tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-mat)
        pairs = [(int(a), int(b)) for a, b in zip(r.tolist(), c.tolist())]
    except Exception:                                   # 没有 scipy 就贪心兜底
        taken: set = set()
        order = sorted(info_of, key=lambda ij: -mat[ij[0], ij[1]])
        for i, j in order:
            if i in taken or j in taken:
                continue
            if mat[i, j] <= abstain[i]:                  # 不比弃权高就不吃
                continue
            taken.add(i)
            taken.add(j)
            pairs.append((i, j))

    got: List[Tuple[float, str, dict]] = []
    for i, j in pairs:
        if j >= m or mat[i, j] <= NEG / 2.0:
            continue                                         # 弃权列
        if (i, j) not in info_of:
            continue
        got.append((float(mat[i, j]), cids[i], info_of[(i, j)]))
    got.sort(key=lambda x: -x[0])
    return [(cid, al) for _sc, cid, al in got]


def assign(components: Sequence[dict], pins: Dict[str, List], texts: Sequence[dict],
           near: float = NEAR, far: float = FAR,
           global_match: bool = None, pincol: Dict[str, List] = None,
           pin_evidence: Dict[str, List] = None) -> dict:
    """components: stage1 的元件表（要有 id/type/bbox_img）
    pins:       stage1 的 {cid: [[x,y], ...]}（图像坐标）
    texts:      `stage2.text.TextEngine.read` 的输出
    pincol:     stage1 的引脚列记录 {cid: [record, ...]}（`stage1/pincol.py`），
                用来把引脚号/名直接按**窗口**归位（见下面 `PINCOL_GUIDE` 那段）。
    pin_evidence: stage1 的**逐脚图像证据** {cid: [record, ...]}（队友分支的
                `stage1/pin_evidence.py` / `stage1/pincolor.py` 产出，字段
                `trust/source/edge/frame_confirmed/anchor_img/point_img` + 六项测量）。
                ★ 目前只**接收并挂到模块状态** `LAST_PIN_EVIDENCE`，供队友那五个读法
                模块（经 `_target_index` / `_trusted_text_contact`）做保守校验；本函数
                自己的绑定流程**尚未**消费它（队友那套 `STAGE2_PIN_TRUST_BIND` /
                `STAGE2_PIN_DIGIT_ENTITY` 识别逻辑要单独 A/B 后才会接进来）。

    global_match 默认取环境变量 `STAGE2_GLOBAL_MATCH`（默认关）。
    220 张实测：见 `stage2/README.md`（这一项一直是关的，从未在当前管线上量过）。
    """
    global LAST_PIN_EVIDENCE
    LAST_PIN_EVIDENCE = pin_evidence or {}
    if global_match is None:
        global_match = env_on("STAGE2_GLOBAL_MATCH", "0")
    # ★ 「不删除只削弱」：一个 case 的降权/保留池从零开始累计（`structmatch.HIST_*`），
    #   产物 `_assigned/<case>.json` 的 `no_delete` 块读它。
    try:
        structmatch.clear_hist()
    except Exception:
        pass
    comps = []
    for c in components:
        b = c.get("bbox_img")
        if not b or len(b) != 4:
            continue
        comps.append({"id": c["id"], "type": c.get("type") or "other",
                      "bbox": [float(v) for v in b]})

    # ---- 每个文字框归属到最近的元件 ----
    owned: Dict[str, List[dict]] = {c["id"]: [] for c in comps}
    free: List[dict] = []
    for _i, t in enumerate(texts):
        b = t["bbox"]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        h = max(1.0, b[3] - b[1])
        wide = (b[2] - b[0]) / h > MAX_BOX_W
        wratio = (b[2] - b[0]) / h
        if not comps:
            free.append(dict(t, d=1e9, inside=False, wide=wide, wratio=wratio, _i=_i))
            continue
        ds = [dist_to_rect(cx, cy, c["bbox"]) for c in comps]
        j = min(range(len(ds)), key=lambda i: ds[i])
        d = ds[j]
        ins = inside(cx, cy, comps[j]["bbox"])
        # ★ 归属（owned / free）改用**与文字长度无关**的框到框间距，见 `OWN_BBOX`。
        # `t["d"]`（中心距）原样保留 —— 它是评分与 `NEAR` 判据用的，语义不能动。
        d_own = dist_rect_to_rect(b, comps[j]["bbox"]) if OWN_BBOX else d
        rec = dict(t, d=d, own=d_own, inside=ins, wide=wide, wratio=wratio,
                   center=(cx, cy), _i=_i)
        if d_own <= far:
            owned[comps[j]["id"]].append(rec)
        else:
            free.append(rec)

    # ---- 「抢读」名单：别人名下、但离自己 <= STEAL_R 的框（只给位号用） ----
    steal: Dict[str, List[dict]] = {c["id"]: [] for c in comps}
    if STEAL_R > 0:
        for owner_id, rs in owned.items():
            for rec in rs:
                if _too_wide(rec, DESIG_MAXW):
                    continue
                cx, cy = rec["center"]
                for c in comps:
                    if c["id"] == owner_id:
                        continue
                    d = dist_to_rect(cx, cy, c["bbox"])
                    if d <= STEAL_R:
                        steal[c["id"]].append(dict(
                            rec, d=d, inside=inside(cx, cy, c["bbox"]),
                            stolen_from=owner_id))

    # ---- 自由文字的「抢读」--------------------------------------------------
    # 原来 `d > FAR` 的框**谁都不要**（没进任何元件的 `owned`），于是大 IC 的位号
    # 经常掉进这个空洞：实测 `0003` 的 `U1` 位号印在 IC **右上角外侧**，文字框中心
    # 到 IC 框边 48.3px > `FAR=40`，被判成自由文字 —— `c0001` 连一个候选都没有，
    # 整颗 15 个引脚的 IC 被 `DROP_UNKEYED` 丢掉。
    # 位号本质是「离符号最近的几个短文字之一」，不该被归属半径一刀切死，
    # 所以自由文字也进抢读池（半径同 `STEAL_R`，冲突仍交给后面的全局去重裁决）。
    # 自由候选的 `d` 更大、`inside=False`，打分天然低于自家名下的框，
    # 只在「名下什么都没有」时才会胜出 —— 这一支纯补空，不改已有结果。
    if FREE_STEAL and STEAL_R > 0:
        for rec in free:
            if _too_wide(rec, DESIG_MAXW) or not rec.get("center"):
                continue
            cx, cy = rec["center"]
            for c in comps:
                d = dist_to_rect(cx, cy, c["bbox"])
                if d <= STEAL_R:
                    steal[c["id"]].append(dict(
                        rec, d=d, inside=inside(cx, cy, c["bbox"]),
                        stolen_from=None))

    # ---- 逐元件定角色 ----
    out: Dict[str, dict] = {}
    desig_cands: List[Tuple[str, str, float, dict]] = []   # (cid, 位号, 分数, 证据)
    desig_all: Dict[str, List[dict]] = {}                 # cid -> 全部候选（阶段三用）
    # ★ 内涵「遗漏审计」暂存（`STAGE2_CONFORM_AUDIT`，恒开）：cid -> 本该挂上却
    #   在当前规则下过不了 canon 的读法。只写报告，见下面 `LAST_CONFORM`。
    audit_of: Dict[str, dict] = {}

    # ---- ★ 零候选补位（`STAGE2_SHAPE_FILL`，默认 0 = 关）------------------
    # 实测（§5.235）：blk1 34 案里 89 颗「框像素级正确、位号错」的元件，**53 颗（59.6%）
    # 的 `desig_slots` 是空的** —— 连被抢的资格都没有。这一带确有印字也进了 `dcands`，
    # 但 `_best_read(... "designator")` 一条都读不出「位号形态 + 前缀互证」，`alts` 为空
    # ⇒ 元件退化成 `pNNN` ⇒ 官方 keymap 不给有名元件坐标兜底（`metrics.py:347` 只对
    # `is_unkeyed` 开放）⇒ missing + extra 各记一次。
    # 本支只在**一个合法候选都没有**时补一条「形状是位号形」的兜底，分数压到 0 以下
    # （`STAGE2_SHAPE_PEN`），永远抢不过正常候选 ⇒ 默认关时逐位不变。
    _SHAPE_FILL = env_on("STAGE2_SHAPE_FILL", "0")
    _SHAPE_PEN = float(os.environ.get("STAGE2_SHAPE_PEN", "0.60") or 0.60)
    _SHAPE_MINCONF = float(os.environ.get("STAGE2_SHAPE_MINCONF", "0.90") or 0.90)
    _SHAPE_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")

    def _shape_canon(t):
        """从候选框的所有读法里挑一条**形状**像位号的串（不做前缀/类型互证）。

        返回 ``(串, 该串的读置信度)``；置信度取该读法自己的，用于 `STAGE2_SHAPE_MINCONF`
        这道闸——实测（§5.236）低置信的「像位号的错串」会把 `pNNN` 换掉，而具名元件
        拿不到官方的坐标兜底（`metrics.py:347`），反而把引脚/网络一起害死。
        """
        seen = []
        for v in (t.get("reads") or {}).values():
            if isinstance(v, (tuple, list)) and v and v[0]:
                seen.append((str(v[0]), float(v[1] or 0.0)))
        for key in ("text", "txt", "read"):
            if t.get(key):
                seen.append((str(t[key]), 0.0))
        for s, cf in seen:
            s2 = re.sub(r"\s+", "", s).upper()
            if _SHAPE_RE.match(s2):
                return s2, cf
        return "", 0.0

    for c in comps:
        cid, ctype = c["id"], c["type"]
        cands = owned.get(cid, [])
        rec = {"designator": None, "value": None, "partname": None, "pins": {}}
        out[cid] = rec

        # 位号。**前缀矫正**是无损的（位号首字母由类型决定），实测把识别天花板
        # 从 0.500 抬到 0.725（`work/_des_ceiling.py`）。
        # 这里保留每个元件的**全部候选**（不只最优）—— 阶段三的 LLM 仲裁要吃它们，
        # 只留最优的话「槽位」就是空的、第三阶段没有东西可裁。
        alts = []
        # 2026-07 审计修复：`NO_DESIGNATOR`（:194-196）定义了却**全仓库从未被引用**，
        # 于是「电源/地/网络符号没有位号可言」这道闸门根本没接上。这些 type 在
        # `TYPE_PREFIX` 里没有条目 ⇒ `canon_designator` 走 `allow` 为空的兜底支，
        # **绕过 `PREFIX_MODE=strict`**，任何 `[A-Z]{1,3}\d{1,4}` 读法都以 0.5 的保真度
        # 变成候选；再加上 +0.15 的 inside 加成，一个紧挨着电阻值的 `GND` 符号能以
        # 0.62 抢走真电阻的位号，真电阻则退化成 `pNNN` ⇒ 键配不上 ⇒ 它的引脚、网络
        # 引用跟着一起丢（还会叠加下面 `STAGE2_NET_DROP_UNKEYED` 把整条网剪空）。
        # 实测依据（220 个 GT 全量清点，key 是否形如 `[A-Za-z]{1,3}\d`）：
        #   gnd 2229/2229 非位号形、v 1151/1152、net_input 339/339、net_short 163/163、
        #   net_output 76/76、other 69/69、net_bidirection 13/13 —— 只有 1 个 `v` 例外。
        # 故对这些 type 直接不给位号候选（`desig_slots` 也一并清空，阶段三没有可仲裁的东西）。
        dcands = [] if ctype in NO_DESIGNATOR else (
            cands if STEAL_R <= 0 else list(cands) + steal.get(cid, []))
        if _dbg_want(cid):
            print("[dbg-D] cid=%s type=%s bbox=%s owned=%d steal=%d"
                  % (cid, ctype, c.get("bbox"), len(cands), len(steal.get(cid, []))))
            for t in dcands:
                print("        tb=%-30s d=%6.1f wide=%s reads=%s"
                      % ([round(x, 1) for x in t["bbox"]], t["d"], t.get("wide"),
                         {k: v for k, v in (t["reads"] or {}).items() if v[0]}))
        for t in dcands:
            if _too_wide(t, DESIG_MAXW):
                continue
            if DESIG_MAXD > 0 and t["d"] > DESIG_MAXD:
                continue          # 只挡「距离离谱」的让位候选（默认关）
            got = _best_read(t["reads"], "designator",
                             lambda s, ty=ctype: canon_designator_meaning(s, ty, rescue=True)[1])
            if not got:
                continue
            _s, txt, conf, _f = got
            can, fid, tfix, how = canon_designator_meaning(txt, ctype)
            if not can:
                # ★ 内涵「遗漏检测」（`STAGE2_CONFORM_AUDIT`，恒开、**只写报告**）：
                #   这条读法在当前开关下过不了 canon，但按内涵规则（前缀 ⇄ 类型互证、
                #   削开头胶连）能不能反推回一个合法位号？能 ⇒ 记一笔，由
                #   `stage2/run.py` 落到 `_assigned/<case>.json` 的
                #   `no_delete.conform_audit`。这就是「检索遗漏的元素」：
                #   官方 10 张上它恰好点出那 4 颗被丢掉的 LED。
                #   ⚠ 只写旁路报告，`alts` 一个都不加 ⇒ 输出逐位不变。
                if CONFORM_AUDIT and conf >= CONFORM_MIN_CONF and cid not in audit_of:
                    _c2, _f2, _tf2, _h2 = canon_designator_meaning(txt, ctype, rescue=True)
                    if _c2 and _h2:
                        audit_of[cid] = {
                            "read": txt, "canon": _c2, "how": _h2,
                            "type_fix": _tf2, "type_now": ctype,
                            "conf": round(float(conf), 4),
                            "d": round(float(t["d"]), 1),
                            "role": next((r for r, v in (t["reads"] or {}).items()
                                          if v and v[0] == txt), ""),
                            "box": [round(float(x), 1) for x in t["bbox"]],
                        }
                continue
            if how:
                # 内涵回收路径的两道额外闸门（`how == "strict"` 时一概不查，
                # 保证默认臂逐位不变）：
                #  · 置信度地板 —— 回收的前提是「读法本身可信」；
                #  · `type` 那条要求文字框在 `FAR` 之内 —— 翻类型等于改判元件类型，
                #    只允许用**贴着这颗元件**的印字当证据（抢读来的远端框不算）。
                if conf < CONFORM_MIN_CONF:
                    continue
                if how == "type" and t["d"] > FAR:
                    continue
            # ★ 距离项必须放宽到 `STEAL_R`（见 `_score` 的 docstring）：位号候选池里
            # 有一大半是 `d > FAR` 的抢读候选，用 `FAR` 截断会让它们全部同分。
            sc = _score(fid, conf, t["d"], t["inside"],
                        dist_clamp=max(FAR, STEAL_R))
            _dr = -1.0
            if NEAR_REV_W > 0.0:                                  # §5.222 贴中罚（默认开，§5.222.10）
                _cb = c.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                _span = 0.5 * (abs(_cb[2] - _cb[0]) + abs(_cb[3] - _cb[1])) + 1.0
                if _span >= NEAR_REV_MIN_SPAN:
                    _tcx = 0.5 * (t["bbox"][0] + t["bbox"][2])
                    _tcy = 0.5 * (t["bbox"][1] + t["bbox"][3])
                    _ccx = 0.5 * (_cb[0] + _cb[2])
                    _ccy = 0.5 * (_cb[1] + _cb[3])
                    _dr = (((_tcx - _ccx) ** 2 + (_tcy - _ccy) ** 2) ** 0.5) / _span
                    sc -= NEAR_REV_W * (max(0.0, NEAR_REV_R - _dr) / NEAR_REV_R)
            alts.append((sc, {"text": txt, "canon": can, "conf": conf,
                              "box": list(t["bbox"]), "d": t["d"],
                              "fit": fid, "box_i": t.get("_i", -1),
                              "type_fix": tfix}))
        alts.sort(key=lambda x: -x[0])
        if not alts and _SHAPE_FILL:
            # 零候选补位：只在「一个合法候选都没有」时启用；分数 = 常规项 − 罚
            _fb = []
            for t in dcands:
                if _too_wide(t, DESIG_MAXW):
                    continue
                _sh, _cf = _shape_canon(t)
                if not _sh or _cf < _SHAPE_MINCONF:
                    continue
                _sc = _score(t.get("fit") or 0.0, _cf, t["d"], t["inside"],
                             dist_clamp=max(FAR, STEAL_R)) - _SHAPE_PEN
                _fb.append((_sc, {"text": _sh, "canon": _sh, "conf": _cf,
                                  "box": list(t["bbox"]), "d": t["d"],
                                  "fit": t.get("fit") or 0.0,
                                  "box_i": t.get("_i", -1), "type_fix": None,
                                  "shape_only": True}))
            _fb.sort(key=lambda x: -x[0])
            alts = _fb
            if _dbg_want(cid):
                print("[dbg-shape] cid=%s 零候选补位 -> %s"
                      % (cid, [(round(s, 3), i["canon"]) for s, i in alts[:3]]))
        if _dbg_want(cid):
            print("[dbg-alts] cid=%s -> %s" % (cid, [
                (round(s, 3), i["canon"], round(i["conf"], 3), round(i["d"], 1))
                for s, i in alts[:4]]))
        if alts:
            desig_all[cid] = [dict(info, score=float(sc))
                              for sc, info in alts[:MAX_ALTS]]
            sc, info = alts[0]
            desig_cands.append((cid, info["canon"], sc, info))

    # ★ 键形成层（`STAGE2_KEY_CHANNEL`，默认 0 = 关；规则与依据见 `stage2/keyform.py`）：
    #   把「键是谁的」与「这个框读出什么」拆开——键的候选只来自自己名下的框，
    #   谁更像这颗的印字交给几何（体上方 + 横向重叠），一对一裁决，撞框/撞号作废
    #   退回 `pNNN`。上面那条「分数线里混距离 + 串角色借读」的老路径原样保留，
    #   默认臂（channel 关）逐位不变。
    _KEY_CHANNEL = env_on("STAGE2_KEY_CHANNEL", "0")
    if _KEY_CHANNEL:
        _kcomps = [c for c in comps if c["type"] not in NO_DESIGNATOR]
        picked = _keyform.pick(texts, _kcomps, canon_designator_meaning)
        if env_on("STAGE2_KEY_DBG", "0"):
            print("[dbg-key] %s" % (_keyform.LAST,))
    elif global_match:
        picked = _global_match(desig_cands, [c["id"] for c in comps])
    elif OWN_ASSIGN:
        picked = _own_assign_pick(desig_all)
    elif OWN_MATCH:
        picked = _own_match_pick(desig_all)
    else:
        # 位号去重：同一张图里位号唯一，全局按分数贪心，撞号的低分者让位。
        # ★ 撞号判据加一条**文字框身份**（`_box_key`：`box_i` 优先、缺了退回 canon）。
        #   两块不同的框读成同一个位号 ⇒ 仍按 canon 去重（原行为，位号确实唯一）；
        #   同一块框被两颗元件读到两个不同 canon（罕见）⇒ 原来两票全收、同一块文字
        #   成了两颗元件的位号，现在低分者让位。
        #   撞框那一票作废时**照样占住 canon**（否则这块框的误读会被顶给第三个元件，
        #   凭空造出一条今天不存在的归属）⇒ `picked` 恒为今天那份列表的**子集**。
        picked = []
        seen_txt: set = set()
        seen_box: set = set()

        def _rank(x):
            """撞号时的排序键：**救援性候选让位**，同组内再比分数。

            ★ `CONFORM_KEY` 的「翻类型」候选（`type_fix` 非空）是**救援**性质的 ——
              实测官方 case 4：同一颗 LED 被检出两遍（`d` 与 `led` 两块框指着
              同一个 `box_i`），翻开 `d` 那块后它的距离更近（18.1 vs 23.7），
              分数 0.2155 > 0.1677 就把 `LED1` 从本来就同类型的那颗手里抢走，
              引脚 tp 25→23、该案 −1.29。本该同类型的候选必须先拿号。
              默认关时 `type_fix` 恒为 `None` ⇒ 与今天的 `-sc` 排序逐位相同。
            """
            return ((x[3] or {}).get("type_fix") is not None, -x[2])

        for cid, can, sc, info in sorted(desig_cands, key=_rank):
            bi = _box_key(info)
            if can in seen_txt or (bi is not None and bi in seen_box):
                seen_txt.add(can)
                continue
            seen_txt.add(can)
            if bi is not None:
                seen_box.add(bi)
            picked.append((cid, info))

        # ★ 落选者补位（`STAGE2_ALT_FILL`，默认 0 = 关）：撞号/撞框时高分的拿走，
        #   低分的**直接掉回 `pNNN`**，哪怕它自己还有第二个空着的候选框可用 ——
        #   池子 0101 的「邻位串号」（`X1←C11`/`C11←C5`）正是这条链的产物：
        #   `C11` 那块框被水晶振子抢走，真电容没去拿自己的次优框。这里让落选者
        #   按自己的候选顺序再挑一次，只拿**没被任何人占过的框/串**。
        if env_on("STAGE2_ALT_FILL", "0"):
            _got = {cid for cid, _ in picked}
            for cid in [c["id"] for c in comps]:
                if cid in _got:
                    continue
                for info in (desig_all.get(cid) or [])[1:]:
                    can = (info or {}).get("canon")
                    bi = _box_key(info)
                    if not can or can in seen_txt or (bi is not None and bi in seen_box):
                        continue
                    seen_txt.add(can)
                    if bi is not None:
                        seen_box.add(bi)
                    picked.append((cid, info))
                    break

    # `desig_slots` 给的是**每个元件的全部候选**（阶段三 LLM 仲裁的原料），
    # 不是「最优候选」的列表 —— 后者会让第个三阶段没有任何东西可裁。
    desig_slots: Dict[str, List[dict]] = desig_all
    # ★ 内涵键回收（`CONFORM_KEY`）翻了类型的元件（官方 case 6 的 `d` -> `led`）：
    #   类型既是输出的一部分（官方按精确串比 `type`），又决定引脚图章与脚名别名
    #   （`assemble.PIN_STAMP_TYPES` / `PIN_NAME_ALIAS`），所以必须传下去 ——
    #   ① 记进 per-cid 记录（键 `ctype_fix`），给 `assemble.py` 读；
    #   ② 本文件后面的参数/器件名/引脚各 pass 直接用改写后的类型。
    ctype_of: Dict[str, str] = {c["id"]: c["type"] for c in comps}
    # ---- ★ 低置信具名键降级成 `pNNN`（`STAGE2_DEMOTE_CONF`，默认 0 = 关）----
    # 依据（§5.236/§5.237，blk1 34 案 324 颗具名元件，按我们自己的读置信度分桶，
    # GT 只当尺子）：cf≥0.95 正确率 **91.6%**（225 颗）、0.80–0.95 **72.2%**（72 颗）、
    # 0.60–0.80 **50.0%**（16 颗）、<0.60 **45.5%**（11 颗）。
    # 官方 keymap 的坐标兜底只对**无名**元件开放（`metrics.py:347`），所以一个「像位号
    # 但读错」的具名键会把原本可以被坐标救回的元件彻底判死，连它的引脚/网络一起
    # （§5.236 实测：补位臂元件 tp +7，网络 −0.006）。反过来，低置信时**退回 `pNNN`**
    # 是几乎免费的：框通常是对的，坐标兜底照样能配上。
    _DEMOTE = float(os.environ.get("STAGE2_DEMOTE_CONF", "0") or 0.0)
    if _DEMOTE > 0.0:
        _keep = []
        for cid, info in picked:
            _cf = float((info or {}).get("conf") or 0.0)
            if _cf < _DEMOTE:
                if _dbg_want(cid):
                    print("[dbg-demote] cid=%s %s conf=%.3f < %.2f → pNNN"
                          % (cid, (info or {}).get("canon"), _cf, _DEMOTE))
                continue
            _keep.append((cid, info))
        picked = _keep

    # ★ 键归属倒查视图（`STAGE2_DUMP_KEYPROV=1`，默认关）：上面 `LAST_TEXT_READS` 只记
    #   「每块框能读出什么」，答不了「这颗元件的键是从哪块框、多远、多少置信度来的」——
    #   池子取证时正是被这一点卡住（同名字串全在百米以外，无法判断是漏检还是错配）。
    #   每颗拿到键的元件落一��：cid / canon / 框 / 距离 / 置信度 / 分数。
    if env_on("STAGE2_DUMP_KEYPROV", "0"):
        try:
            _p = os.environ.get("STAGE2_DUMP_PATH") or os.path.join("work", "_reads_dump.jsonl")
            with io.open(_p, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "case": os.environ.get("STAGE2_DUMP_CASE", ""),
                    "picked": [{"cid": cid,
                                "key": (info or {}).get("canon"),
                                "box": (info or {}).get("box"),
                                "d": (info or {}).get("d"),
                                "conf": (info or {}).get("conf"),
                                "score": (info or {}).get("score"),
                                "chan": bool(_KEY_CHANNEL)}
                               for cid, info in picked]}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    for cid, info in picked:
        out[cid]["designator"] = info
        _tf = (info or {}).get("type_fix")
        if _tf and _tf != ctype_of.get(cid):
            _old = ctype_of.get(cid)
            ctype_of[cid] = _tf
            out[cid]["ctype_fix"] = _tf
            if _dbg_want(cid):
                print("[dbg-typefix] cid=%s %s -> %s (canon=%s)"
                      % (cid, _old, _tf, info.get("canon")))

    # ★ 远处位号认领（`STAGE2_FAR_KEY` = 半径像素，默认 0=关）-------------------------
    # 依据（`work/_readcheck.py` + `work/_farclaim.py`，50 评测案逐框全量读法）：
    #   非键对的 138 颗 IC 里 **29 颗的 GT 位号串确实被读到了**，只是在页面上离元件
    #   很远（>120px），进不了 `owned/steal` 这两个按距离划的口袋 ⇒ 元件退化成 `pNNN`。
    # 规则（**不看 GT**）：元件当前无位号 + 类型在 `_FAR_TYPES` 内；候选读法必须
    #   ①按**这颗元件自己的类型**能 canon 出位号（前缀 ⇄ 类型互证 ⇒ 别的类型的位号串
    #   不会被抢）；②该 canon 尚未被任何元件占用；③框-框距离 ≤ 半径；④一对一。
    # 由近到远贪心。分数只用于记账，不参与阶段三仲裁。
    if _FAR_KEY > 0.0:
        _byid = {c["id"]: c for c in comps}
        _cur = {}
        for c in comps:
            _i = (out.get(c["id"]) or {}).get("designator")
            if _i:
                _cur[c["id"]] = (str(_i.get("canon") or "").upper(), float(_i.get("d") or 0.0))
        _reassign = []
        for c in comps:
            cid = c["id"]
            if c["type"] in NO_DESIGNATOR or (_FAR_TYPES is not None
                                              and c["type"] not in _FAR_TYPES):
                continue
            if cid not in _cur or _cur[cid][1] > _FAR_REPL:
                _reassign.append(cid)
        _used = set()
        for cid, (_cc, _dd) in _cur.items():
            if _cc and cid not in set(_reassign):
                _used.add(_cc)
        _cand = []
        for cid in _reassign:
            c = _byid[cid]
            _cc, _cd = _cur.get(cid, ("", 0.0))
            for t in texts:
                if _too_wide(t, DESIG_MAXW):
                    continue
                _d = dist_rect_to_rect(t["bbox"], c["bbox"])
                if _d > _FAR_KEY or (_cc and _d >= _cd):
                    continue
                for _role, _rv in (t.get("reads") or {}).items():
                    if not (isinstance(_rv, (tuple, list)) and _rv and _rv[0]):
                        continue
                    _cf = float(_rv[1] or 0.0)
                    if _cf < _FAR_CONF:
                        continue
                    can, fid, tfix, _how = canon_designator_meaning(str(_rv[0]), c["type"])
                    if not can or can.upper() in _used or can.upper() == _cc:
                        continue
                    _cand.append((_d, cid, can, str(_rv[0]), _cf, list(t["bbox"]), tfix))
        _cand.sort(key=lambda x: x[0])
        if os.environ.get("STAGE2_FAR_DBG", "0") not in ("", "0", "false", "off"):
            print("[dbg-far-sum] reassign=%d cand=%d used=%d"
                  % (len(_reassign), len(_cand), len(_used)))
        _done = set()
        for _d, cid, can, _txt, _cf, _box, _tf in _cand:
            if cid in _done or can.upper() in _used:
                continue
            out[cid]["designator"] = {"text": _txt, "canon": can, "conf": _cf,
                                      "box": _box, "d": _d, "fit": 1.0,
                                      "box_i": -1, "type_fix": _tf,
                                      "key_src": "far"}
            _used.add(can.upper())
            _done.add(cid)
            if _dbg_want(cid):
                print("[dbg-far] cid=%s canon=%s read=%s d=%.0f conf=%.3f"
                      % (cid, can, _txt, _d, _cf))

    # ★ 内涵「遗漏检测」定稿（`STAGE2_CONFORM_AUDIT`，恒开、**只写报告**）：
    #   在**所有归属通道**（贪心/匈牙利/own.*）都跑完之后看一遍 —— 哪些元件
    #   最终没有位号，而内涵规则（前缀 ⇄ 类型互证、削开头胶连）本来能给它一个键？
    #   这些就是「被遗漏的元素」，由 `stage2/run.py` 落到
    #   `_assigned/<case>.json` 的 `no_delete.conform_audit`。
    #   与「不删除只削弱」账本同一条通道：只追加记录，`pred` 一个字节都不动。
    if CONFORM_AUDIT:
        clear_conform()
        for _cid in sorted(audit_of):
            if (out.get(_cid) or {}).get("designator"):
                continue      # 别的通道把它救回来了 ⇒ 不算遗漏
            rec = dict(audit_of[_cid])
            rec["cid"] = _cid
            LAST_CONFORM.append(rec)
        del LAST_CONFORM[CONFORM_AUDIT_MAX:]
        # ★ 同一通道的另一半：**「多余检测」**。内涵（官方命名约定）说
        #   `d`/`led` 的 `Name` 槽必须是**型号**，而我方 `assemble` 对非 `box`
        #   一律 `name = key` ⇒ 凡是我们发出去的 `d`/`led`，在官方口径下必然
        #   判错（既不是 tp、还占一个 fp）。把它们连同器件名读数一起报出来，
        #   换约定时可直接用；同样**只写报告**。
        for _cid, _info in picked:
            _ct = ctype_of.get(_cid)
            if _ct not in EXTRA_MODELNAME_TYPES:
                continue
            _pn, _pnc = "", 0.0
            # 器件名可能落在 `steal`（80px 内但没被硬划分给它）那一侧，一起看。
            for _t in list(owned.get(_cid, [])) + list(steal.get(_cid, [])):
                _r = (_t["reads"] or {}).get("partname")
                if _r and _r[1] > _pnc:
                    _pn, _pnc = _r[0], _r[1]
            LAST_CONFORM_EXTRA.append({
                "cid": _cid, "canon": _info.get("canon"), "type": _ct,
                "name_emitted": _info.get("canon"), "partname_read": _pn,
                "partname_conf": round(float(_pnc), 4),
                "why": "%s 的 Name 槽在官方 10GT 里是型号（我方 name=key）" % _ct,
                "box": [round(float(x), 1) for x in _info.get("box") or []],
            })
        del LAST_CONFORM_EXTRA[CONFORM_AUDIT_MAX:]
    # 已被选中的位号框：值兜底时必须绕开（`fit_value` 对 `C5` 这种会经 NETNAME 支给 0.6）。
    _desig_boxes = {tuple(info["box"]) for _cid, info in picked}

    # 结构化重绑定留下的「两种方案打架」的引脚（用户 m06470 要的**阶段三 LLM 接口**）。
    # 每条 = {"cid","role","pin","pins","cand","alt"}，`cand`/`alt` 都是候选文字。
    struct_slots: List[dict] = []

    # 参数 / 器件名 / 引脚号 / 引脚名
    for c in comps:
        # ★ `ctype_of` 取内涵回收改写后的类型（`d` -> `led`）：`led` 的参数槽是
        #   「器件名」而不是「值」，引脚通道的期望角色与图章也随类型变。
        cid, ctype = c["id"], ctype_of.get(c["id"], c["type"])
        cands = owned.get(cid, [])
        rec = out[cid]

        # 参数（r/c/l/...）或器件名（box 等）
        want = "value" if ctype in ("r", "c", "l", "fuse", "crystal_2pin",
                                    "crystal_3pin", "crystal_4pin", "crystal") else "partname"
        fit = (lambda _t: fit_value_unitx(_t, ctype)) if want == "value" else fit_partname
        # 注意：这里**不能**用 `wide` 过滤 —— IC 的器件名本来就长
        # （实测 `MSP432E401YTPDT` 这种），按宽高比筛会把器件名整条筛掉。
        best = None
        _src = list(cands)
        if want == "partname" and PART_STEAL > 0:
            for _t in steal.get(cid, []):
                if float(_t.get("d", 1e9)) > PART_STEAL:
                    continue
                _got = _best_read(_t["reads"], "partname", fit_partname)
                if not _got:
                    continue
                _s, _txt, _cf, _f = _got
                if _f < 1.0 or _cf < PART_STEAL_CONF or len(str(_txt or "")) < 5:
                    continue
                _t = dict(_t)
                _t["_part_pen"] = 1.0
                _src.append(_t)
        for t in _src:
            if rec["designator"] and list(t["bbox"]) == rec["designator"]["box"]:
                continue
            got = _best_read(t["reads"], want, fit)
            if not got:
                continue
            s, txt, conf, f = got
            sc = _score(s, conf, t["d"], t["inside"],
                        dist_w=(VALUE_DISTW if want == "value" else 0.035))
            if t.get("_part_pen"):
                sc -= PART_STEAL_PEN
            if want == "partname" and NAME_SHAPE_PEN > 0.0:      # §5.248 字形闸（默认关）
                _bad = 0.0
                _tx = str(txt or "").strip()
                if not any(ch.isdigit() for ch in _tx):
                    _bad = 1.0                       # `LDO`/`OUT`/`AGND`
                elif re.match(r"^\d+(?:\.\d+)?\s*[VvMm]$", _tx):
                    _bad = 1.0                       # `3.3V`/`5V`
                if rec["designator"] and _tx == str(rec["designator"]["text"]).strip():
                    _bad = 1.0                       # 拿了自己的位号当型号（`U7`）
                sc -= NAME_SHAPE_PEN * _bad
            if best is None or sc > best[0]:
                best = (sc, txt, conf, t)
        if best is None and want == "value" and VALUE_STEAL_R > 0:
            # **值兜底**（`STAGE2_VALUE_STEAL`）。阻容的「值」字在符号下方/右侧，
            # 但它离**邻居**的元件框往往更近，于是归属判定把它给了邻居，
            # 这颗元件就连候选都没有。实测 `work/stage2_v9` 上「键对上了、值空着」
            # 有 **371 条**（占 r/c 值失配的 54.3%）—— 与位号那条 STEAL 是同一个病。
            #
            # 两道闸门（缺了会立刻出错，实测）：①不许拿别人已经选中的位号框
            # （`fit_value("C5")` 会经 `NETNAME_RE` 那支拿到 0.6，把位号当值）；
            # ②`f` 必须 == 1.0，即真的长成 `VALUE_RE` 的样子，不走 0.6 的兜底支。
            bx = c["bbox"]
            ccx, ccy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
            for t in texts:
                tb = t.get("bbox")
                if not tb or len(tb) != 4:
                    continue
                if tuple(tb) in _desig_boxes:
                    continue
                d = dist_to_rect(ccx, ccy, tb)
                if d > VALUE_STEAL_R:
                    continue
                got = _best_read(t["reads"], "value",
                                 lambda _t: fit_value_unitx(_t, ctype))
                if not got or got[3] < 1.0:
                    continue
                s, txt, conf, f = got
                # 2026-07 审计修复：原来是 `inside(ccx, ccy, tb)`，两个实参对调了 ——
                # 拿**元件中心**当点、拿**文字框**当矩形，判的是「文字框盖住元件中心」，
                # 于是「文字落在元件框内」的 +0.15 加成给反了。同文件另外三处（:494 主路径、
                # :516 邻居、:537 自由候选）统一是 `inside(文字中心, 元件框)`，这里跟着统一。
                _tcx, _tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
                sc = _score(s, conf, d, inside(_tcx, _tcy, c["bbox"]))
                if best is None or sc > best[0]:
                    best = (sc, txt, conf, dict(t, d=d))
        if want == "value" and VALUE_FILL_R > 0 and (
                best is None or is_foreign_value(best[1], ctype)):
            # **值补候选**：只在「当前是空值/异类单位」时出手，不与已正确的值竞争。
            bx = c["bbox"]
            ccx, ccy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
            _fb = None
            for t in texts:
                tb = t.get("bbox")
                if not tb or len(tb) != 4:
                    continue
                if tuple(tb) in _desig_boxes:
                    continue
                d = dist_to_rect(ccx, ccy, tb)
                if d > VALUE_FILL_R:
                    continue
                got = _best_read(t["reads"], "value",
                                 lambda _t: fit_value_unitx(_t, ctype))
                if not got or got[3] < 1.0:
                    continue
                s, txt, conf, f = got
                if is_foreign_value(txt, ctype):
                    continue
                _tcx, _tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
                sc = _score(s, conf, d, inside(_tcx, _tcy, c["bbox"]))
                if _fb is None or sc > _fb[0]:
                    _fb = (sc, txt, conf, dict(t, d=d))
            if _fb is not None and (best is None or _fb[0] > best[0] + VALUE_FILL_GAIN):
                best = _fb
        if best is not None:
            _txt = best[1]
            if want == "value" and VALUE_TRIM_ON:
                _txt = canon_value(_txt) or _txt
            rec[want] = {"text": _txt, "conf": best[2],
                         "box": list(best[3]["bbox"]), "d": best[3]["d"]}
        if _dbg_want(cid):
            print("[dbg] cid=%s type=%s want=%s bbox=%s" % (cid, ctype, want, c.get("bbox")))
            print("      owned cands=%d" % len(cands))
            for t in cands:
                print("        tb=%-30s d=%6.1f in=%s reads=%s"
                      % ([round(x, 1) for x in t["bbox"]], t["d"], t.get("inside"),
                         {k: v for k, v in (t["reads"] or {}).items() if v[0]}))
            print("      -> best=%s" % (best if best is None else (best[1], round(best[0], 3)),))

        # 引脚名 / 引脚号
        plist = list(pins.get(cid) or [])
        if not plist:
            continue
        # ★★ 引脚列记录（`stage1/pincol.py`）：`plist[i]` 与 `_pcr[i]` 一一对应
        # （`pincol.merge_into` 就是按记录顺序写进去的，未被认领的老触点补在末尾）。
        _pcr = (pincol or {}).get(cid) or []
        pinfo = {i: {"number": None, "name": None} for i in range(len(plist))}
        rec["pins"] = pinfo
        # ★ 名轮不再排除「器件名（`partname`）框」（RESULTS §5.92）。
        #   原写法把 `rec[want]` 一并排除，而 IC 的 `want` 就是 `partname`：
        #   `vlm/enrich.py` 把 U3 的器件名**文字**改写成 `AMS1117-3.3` 时**保留了旧框**，
        #   那块框其实是 `VOUT(TAB)`（`[559,754,634,767]`）⇒ 真名框进不了名轮候选池，
        #   pin1 被旁边的红点标记抢走（读成 `Q`）、pin2 空着，两条名字一起错。
        #   但**不能无条件放行**：自检 c2 就是「框内器件名 `STM32F4` 别被名字抢走」，
        #   所以只在**记录与框脱节**（这块框自己读不出记录里的器件名）时才放行 ——
        #   那正是上面这种「文字被改写过、框还是旧的」的形态。
        def take(t) -> bool:
            if t.get("bbox") is None:
                return False
            b = list(t["bbox"])
            if rec["designator"] and b == rec["designator"]["box"]:
                return False
            if rec["value"] and b == rec["value"]["box"]:
                return False
            if rec["partname"] and b == rec["partname"]["box"]:
                _pn = (t.get("reads") or {}).get("partname")
                _h = b[3] - b[1]
                # 放行要同时满足两条（缺一条都有实测的坑）：
                #   ① 记录与框**脱节** —— 这块框自己读不出记录里的器件名；
                #      自检 c2「框内器件名 `STM32F4` 别被名字抢走」正是靠它挡住。
                #   ② 框**高得像一行引脚名**（≤ `PARTNAME_FREE_H`，默认 18px）。
                #      10GT 上引脚名行高 10~18px，而器件名块可能是 53x22 的两行大字块：
                #      case 4 的 U2 放行后它被读成 `15IO` 抢走 pin2（80.00 → 78.22）。
                #      三条**该放行**的框是 case 1 U1 89x18（`VREFA+`）、case 2 U3 26x10
                #      （`SCL`）、case 5 U3 75x13（`VOUT(TAB)`）⇒ 高度这一条能同时满足
                #      「该放的三条全放、该挡的一条挡住」（宽/高比做不到：case 2 是 2.6）。
                if _pn and _pn[0] == rec["partname"]["text"]:
                    return False
                if _h > PARTNAME_FREE_H:
                    return False
            return True
        taken_box: set = set()      # 同一块文字只能当一次引脚名或引脚号
        for role, want_inside, fit_fn, slot in (
                ("pin_name", True, fit_pinname, "name"),
                ("pin_number", False, fit_pinnum, "number")):
            pool = []
            # 引脚名可以更长（`VDDA`/`GNDA`），引脚号可以离框更远（大 IC 印在框外 29~44px）。
            _wlim = PINNAME_MAXW if slot == "name" else MAX_BOX_W
            _dlim = PINNUM_NEAR if slot == "number" else NEAR
            _dw = PINNUM_DISTW if slot == "number" else 0.035
            # ★ 引脚号抢读（`STAGE2_PINNUM_STEAL`）：见常量的长注释。只抢
            #   「主人没位号（注定被 `DROP_UNKEYED` 丢）」或「无主」的框。
            _src = list(cands)
            if slot == "number" and PINNUM_STEAL > 0:
                for _t in steal.get(cid, []):
                    _own_id = _t.get("stolen_from")
                    if _own_id and ((out.get(_own_id) or {}).get("designator")):
                        continue
                    if float(_t.get("d", 1e9)) > PINNUM_STEAL:
                        continue
                    _t = dict(_t)
                    _t["_steal"] = True
                    _src.append(_t)
            for t in _src:
                if not take(t) or id(t) in taken_box:
                    continue
                if t.get("wide") and (t.get("wratio") or 0.0) > _wlim:
                    continue
                # ★ 名候选的有色墨迹闸门（`STAGE2_PINNAME_CFRAC`）：挡掉红点/极性标记
                if slot == "name" and t.get("cfrac", 0.0) > PINNAME_CFRAC:
                    continue
                # ★ 号轮的「每字符像素宽」闸门（`STAGE2_PINNUM_PXC`，默认 0 = 关）：见常量注释
                if slot == "number" and PINNUM_PXC > 0:
                    _nr = (t.get("reads") or {}).get("pin_number")
                    _txt = str(_nr[0]) if _nr else ""
                    _dig = "".join(ch for ch in _txt if ch.isdigit())
                    _bb = t.get("bbox") or []
                    if _dig and len(_bb) == 4:
                        _wpx = float(_bb[2]) - float(_bb[0])
                        if _wpx / float(len(_dig)) > PINNUM_PXC:
                            continue
                # ★ 「纯数字 = 号」的两个开关（见 `PINNUM_INSIDE` 的长注释）：
                #   号轮放行「体内但读成纯数字」的框，名轮不再收这类框。
                _pdig = (_pure_digit_read(t) if (PINNUM_INSIDE or PINNAME_NODIGIT)
                         else None)
                if want_inside and not t["inside"]:
                    continue
                if (not want_inside) and t["inside"] \
                        and not (PINNUM_INSIDE and _pdig):
                    continue
                if (not want_inside) and t["d"] > _dlim:
                    continue
                got = (_best_number_read(t["reads"]) if (slot == "number" and PINNUM_CONSENSUS)
                       else _best_read(t["reads"], role, fit_fn))
                if not got:
                    continue
                if slot == "name" and PINNAME_NODIGIT and _pdig:
                    continue
                s, txt, conf, f = got
                s = _score(s, conf, t["d"], t["inside"], dist_w=_dw)
                if s <= 0:
                    continue
                pool.append({"center": t["center"], "text": txt, "score": s,
                             "conf": conf, "rec": t, "steal": bool(t.get("_steal"))})
            if not pool:
                continue
            # ★ 候选池去重：同一块文字被重复检出两次时，两个副本会被对齐到两根
            # 不同的脚上（其中一根必然是错的）。见 `PINNUM_DUP` 的注释。
            if PINNUM_DUP > 0 and len(pool) > 1:
                _keep = []
                for cc in pool:
                    if any(k["text"] == cc["text"]
                           and abs(k["center"][0] - cc["center"][0]) <= PINNUM_DUP
                           and abs(k["center"][1] - cc["center"][1]) <= PINNUM_DUP
                           for k in _keep):
                        continue
                    _keep.append(cc)
                pool = _keep
            # ★ 同一块墨迹的**窄副本**只留最宽的一份（`STAGE2_PINNAME_DUPOV`，见 §5.93）。
            #   窄副本是同一个字被截断的裁片（`IN`→`N`、`EN`→`N`），而它的 conf 反而可能更高，
            #   于是「谁分高谁上」会把完整的那份挤掉。
            if slot == "name" and PINNAME_DUPOV > 0 and len(pool) > 1:
                _keep = []
                for cc in sorted(pool, key=lambda x: -(x["rec"]["bbox"][2] - x["rec"]["bbox"][0])):
                    _cb = cc["rec"]["bbox"]
                    _cw = _cb[2] - _cb[0]
                    _dup = False
                    for k in _keep:
                        _kb = k["rec"]["bbox"]
                        _ov = min(_cb[2], _kb[2]) - max(_cb[0], _kb[0])
                        if _ov <= 0:
                            continue
                        if (_ov / max(1.0, min(_cw, _kb[2] - _kb[0])) >= PINNAME_DUPOV
                                and abs(cc["center"][1] - k["center"][1]) <= PINNAME_DUPGAP):
                            _dup = True
                            break
                    if not _dup:
                        _keep.append(cc)
                pool = _keep
            pool.sort(key=lambda x: -x["score"])

            # ★★ **引脚列窗口引导**（`stage1/pincol.py`，默认开）。
            #
            # pincol 已经把「第 i 个脚的号码 / 名字必然落在哪个矩形里」算好了
            # （`out_win` = 框外号码窗、`in_win` = 框内名字窗，窗口半高只有
            # 约 0.48 个 pitch）。于是「哪块文字属于哪根脚」变成一个**矩形包含**
            # 判定，不必再靠整排单调对齐去猜——后者在引脚数只有 8 个（改前
            # stage1 对 0143 的 U1 只给 8 个点）时退化成「谁近归谁」。
            #
            # 只在命中率够高时才采信（否则整颗退回老路径，不让新机制把旧行为弄坏）。
            if PINCOL_GUIDE and _pcr:
                _wk = "out_win" if slot == "number" else "in_win"
                picks = []
                for _i, _r in enumerate(_pcr):
                    if _i >= len(plist) or pinfo[_i][slot] is not None:
                        continue
                    w = _r.get(_wk)
                    if not w or len(w) != 4:
                        continue
                    bb = None
                    for cc in pool:
                        cxx, cyy = cc["center"]
                        if w[0] <= cxx <= w[2] and w[1] <= cyy <= w[3]:
                            if bb is None or cc["score"] > bb[0]:
                                bb = (cc["score"], cc["text"], cc["conf"], cc["rec"])
                    if bb is not None:
                        picks.append((_i, bb))
                _n_win = min(len(plist), len(_pcr))
                if len(picks) >= max(2, int(PINCOL_WIN_COVER * _n_win)):
                    for _i, bb in picks:
                        pinfo[_i][slot] = {"text": bb[1], "conf": bb[2],
                                           "box": list(bb[3]["bbox"]), "d": 0.0}
                        taken_box.add(id(bb[3]))
                    if _dbg_want(cid):
                        print("[dbg] cid=%s pin_col %s: 窗口直接归位 %d/%d"
                              % (cid, slot, len(picks), _n_win))
                    continue

            # **结构化批量重绑定**（`structmatch.py`）：不再「谁离引脚最近就归谁」，
            # 而是把整排候选与整排引脚做单调对齐（两侧都能悬空）、并用数值单调性
            # 剔掉离群候选、必要时整排按数值阶梯重排。引脚号比引脚名更信数值，所以
            # `cover` 给得更足。
            _cbb = c["bbox"]

            def score_of(c, pi, _pl=plist, _bb=_cbb,
                         _dec=PINNUM_DECAY if slot == "number" else 20.0):
                if PINNAME_ROWNEAR and slot == "name" and len(_pl) > 1:
                    # ★ 名候选只允许归到「2D 距离最近的那根脚」（并列允许 ×PINNAME_ROWTIE）。
                    #   见常量处注释：沿边距离对**上边脚**量的是 x 方向，左侧名框会被它抢走。
                    _dm = c.get("_dmin")
                    if _dm is None:
                        _dm = min(((p[0] - c["center"][0]) ** 2
                                   + (p[1] - c["center"][1]) ** 2) ** 0.5 for p in _pl)
                        c["_dmin"] = _dm
                    _dp = ((_pl[pi][0] - c["center"][0]) ** 2
                           + (_pl[pi][1] - c["center"][1]) ** 2) ** 0.5
                    if _dp > _dm * PINNAME_ROWTIE + 1e-6:
                        return 0.0
                if PIN_ALONG:
                    # ★ 用**沿边**距离而不是 2D 距离。
                    # 同一排的文字框在垂直方向（沿边）上各自贴着它的引脚，而在
                    # 朝外方向上的 x 偏移**随文字长度变化**（`VDD` 29px / `VDDIO` 44px
                    # / `VDDREG` 59px，一齐右对齐）。2D 距离把这份长度噪声当成信号，
                    # 实测在 10GT case 8 的 `U1` 上把 `VDD`/`VDDIO` 整体上移一格
                    # （正解 `pin[1](y494)<-VDDIO 沿边距3`、`pin[2](y512)<-VDDREG 距5`
                    # 被算成 `VDD(0.933)->494`、`VDDIO(0.882)->512`，A=1.337 vs
                    # B=1.245）；换成沿边距离后 A=2.849 vs B=3.224，正解胜。
                    e = structmatch.edge_of_point(_bb, _pl[pi][0], _pl[pi][1])
                    d = abs(structmatch.along(_bb, e, _pl[pi][0], _pl[pi][1])
                            - structmatch.along(_bb, e, c["center"][0], c["center"][1]))
                else:
                    dx = _pl[pi][0] - c["center"][0]
                    dy = _pl[pi][1] - c["center"][1]
                    d = (dx * dx + dy * dy) ** 0.5
                return c["score"] / (1.0 + d / _dec)

            cov = PIN_COVER if slot == "number" else PIN_COVER * 0.5
            bound, amb = structmatch.rebind(
                c["bbox"], plist, pool, lambda cc: cc["text"], score_of, cover=cov,
                resid=PIN_RESIDUAL if slot == "number" else 0.0,
                resid_min=PIN_RESIDUAL_MIN)
            if _dbg_want(cid):
                print("      [pin:%s] nplist=%d pool=%d cov=%.2f bound=%d amb=%d"
                      % (role, len(plist), len(pool), cov, len(bound), len(amb)))
                for rs in structmatch.LAST_RESIDUAL:
                    print("        [resid] pin[%2d] <- %s  along_d=%.1f edge=%s"
                          % (rs["pin"], pool[rs["cand"]]["text"], rs["d"], rs["edge"]))
                for bi, (pi, ci) in enumerate(sorted(bound.items())):
                    cc = pool[ci]
                    print("        pin[%2d] <- %-8s s=%.3f c=(%.0f,%.0f)  pt=(%.0f,%.0f)"
                          % (pi, cc["text"], cc["score"], cc["center"][0], cc["center"][1],
                             plist[pi][0], plist[pi][1]))
                unbound = [i for i in range(len(plist)) if i not in bound]
                print("        unbound pins=%s" % unbound)
                print("        pool: " + " | ".join("%s(%.2f)c(%.0f,%.0f)"
                      % (cc["text"], cc["score"], cc["center"][0], cc["center"][1])
                      for cc in pool))
            for a in amb:
                a = dict(a)
                a["cid"] = cid
                a["role"] = role
                a["pins"] = len(plist)
                struct_slots.append(a)
            for pi, ci in bound.items():
                cc = pool[ci]
                if pinfo[pi][slot] is not None or id(cc["rec"]) in taken_box:
                    continue
                pinfo[pi][slot] = {"text": cc["text"], "conf": cc["conf"],
                                   "box": list(cc["rec"]["bbox"])}
                taken_box.add(id(cc["rec"]))

        # ★★ **框外补名**（`STAGE2_PINNAME_OUTSIDE`，默认 0 = 关闭）。见该常量处的长注释。
        #
        # 三条规则同时成立才落地，目的是对上一轮（引脚号）**零影响**：
        #   ① 只写 `pinfo[i]["name"] is None` 的槽 —— 引脚号读出来的东西一个不动；
        #   ② 只吃**框外**的文字（框内的已由 `pin_name` 轮处理）；
        #   ③ 只吃**像名字**的读法：`fit_pinname` 对纯数字只给 0.15，这里要求 `f >= 0.5`。
        #      这条最关键 —— GT 的 IC 引脚名有 **13.5% 本来就是纯数字**，而
        #      `assemble.py:761` 的兜底 `nm = r["nm"] or num` 已经能把这种填对；
        #      拿一块纯数字的框外文字去覆盖它，只会把对的分变成错的。
        #
        # 分配用「空槽 × 候选」全配对的**全局贪心**（按最终分降序，槽与候选各用一次），
        # 不做单调对齐：框外剩下的名字往往只有零星几个，`structmatch` 的单调性假设不成立。
        if PINNAME_OUTSIDE > 0 and plist:
            _pool = []
            for t in cands:
                if not take(t) or id(t) in taken_box or t["inside"]:
                    continue
                if t["d"] > PINNAME_OUTSIDE:
                    continue
                if t.get("wide") and (t.get("wratio") or 0.0) > PINNAME_MAXW:
                    continue
                if t.get("cfrac", 0.0) > PINNAME_CFRAC:
                    continue
                got = _best_read(t["reads"], "pin_name", fit_pinname)
                if not got:
                    continue
                s0, txt, conf, f = got
                # 闸门只压在「适配度 × 置信度」上（`_best_read` 的原始分），**不压距离**：
                # 框外的名字天然比框内远，`_score` 里那 `-0.035*d` 加 `inside` 才有
                # `+0.15` 的算法会让 30px 外的合法名字直接算成负分（合成用例实测）。
                # 距离只进下面的排序项，作排序用。
                if f < 0.5 or s0 < 0.5:
                    continue
                s = _score(s0, conf, t["d"], False)
                _pool.append({"text": txt, "score": s, "conf": conf,
                              "rec": t, "center": t["center"]})
            if _pool:
                _pairs = []
                for cc in _pool:
                    for pi in range(len(plist)):
                        if pinfo[pi]["name"] is not None:
                            continue
                        dx = plist[pi][0] - cc["center"][0]
                        dy = plist[pi][1] - cc["center"][1]
                        dp = (dx * dx + dy * dy) ** 0.5
                        if dp > PINNUM_NEAR:
                            continue
                        # 排序 = 分数 **减** 距离惩罚（和 `_score` 同一套量纲）。
                        # ★ 不能用 `score / (1 + dp/20)`：当 score 为负（框外候选的常态）时，
                        # 除法会让**越远的分越高**，实测把 `VDDIO` 与 `VSSA` 两个名字对调。
                        _pairs.append((cc["score"] - 0.035 * dp, pi, id(cc["rec"]), cc))
                _pairs.sort(key=lambda x: -x[0])
                _used_pin, _used_rec, _n_late = set(), set(), 0
                for _s2, pi, rid, cc in _pairs:
                    if pi in _used_pin or rid in _used_rec:
                        continue
                    pinfo[pi]["name"] = {"text": cc["text"], "conf": cc["conf"],
                                         "box": list(cc["rec"]["bbox"])}
                    taken_box.add(rid)
                    _used_pin.add(pi)
                    _used_rec.add(rid)
                    _n_late += 1
                if _n_late and _dbg_want(cid):
                    print("[dbg] cid=%s pin_name late: 框外补名 %d 个（池 %d，空槽 %d）"
                          % (cid, _n_late, len(_pool),
                             sum(1 for i in range(len(plist)) if pinfo[i]["name"] is None)))

        # ★★ 幻影脚点槽合并（`STAGE2_PIN_STUB_MERGE`，默认关）：见常量处的长注释。
        # 两槽「一槽只有号、另一槽只有名」且两点近到像同一根脚时，把号搬进有名字
        # 的那一槽（被搬空的那一槽本来就会被 `BOX_PIN_NEED_NAME` 丢掉 ⇒ 只增不减）。
        if STUB_MERGE > 0 and len(plist) > 1:
            _used: set = set()
            for _i in range(len(plist)):
                if _i in _used:
                    continue
                for _j in range(_i + 1, len(plist)):
                    if _j in _used:
                        continue
                    _a, _b = pinfo[_i], pinfo[_j]
                    _a_num, _a_nm = bool(_a["number"]), bool(_a["name"])
                    _b_num, _b_nm = bool(_b["number"]), bool(_b["name"])
                    if not (((_a_num and not _a_nm) and (_b_nm and not _b_num))
                            or ((_b_num and not _b_nm) and (_a_nm and not _a_num))):
                        continue
                    _dx = plist[_i][0] - plist[_j][0]
                    _dy = plist[_i][1] - plist[_j][1]
                    _dd = (_dx * _dx + _dy * _dy) ** 0.5
                    if _dd > STUB_MERGE:
                        continue
                    if _a_nm:                       # i 有名字、j 有号 ⇒ 号搬进 i
                        _a["number"] = _b["number"]
                        _b["number"] = None
                        _a["merged_from"] = _j
                        _keep, _drop = _i, _j
                    else:                           # i 有号、j 有名字 ⇒ 号搬进 j
                        _b["number"] = _a["number"]
                        _a["number"] = None
                        _b["merged_from"] = _i
                        _keep, _drop = _j, _i
                    _used.add(_drop)
                    if _dbg_want(cid):
                        print("[dbg] cid=%s stub merge: pin[%d] <- 号 %s（来自 pin[%d]，d=%.1f）"
                              % (cid, _keep, pinfo[_keep]["number"]["text"], _drop, _dd))

    # ---- 网络标签：离所有元件都远、且不在元件框里的文字 ----
    net_labels = []
    for t in free:
        if t.get("wide"):
            continue
        bx = t["bbox"]
        if float(bx[2]) - float(bx[0]) < NETLABEL_MIN_W or \
           float(bx[3]) - float(bx[1]) < NETLABEL_MIN_H:
            continue                      # ★ 「点」不是文字，见 NETLABEL_MIN_W 处注释
        got = _best_read(t["reads"], "net_label", is_netname)
        if not got:
            continue
        s, txt, conf, f = got
        if conf < 0.3:
            continue
        net_labels.append({"text": txt, "conf": conf, "box": list(t["bbox"]),
                           "d": t["d"]})

    # ---- 位号补绑（`STAGE2_DESIG_RESCUE` = 关联半径上限，0=关） ----
    # 证据（台账 §5.270）：blk1 里 298 个没配上的有名元件中，**29.5% 的位号串
    # 整页已经读到过**，但 92.6% 从未成为任何元件的候选（`desig_slots` 里没有它）。
    # 所以这里不再「读」，只做**指派**：拿页面里已有、尚未被占用的位号形状文本，
    # 按「类型前缀必须相符 + 几何距离」做一次全局一对一补绑。
    _rescue_r = float(os.environ.get("STAGE2_DESIG_RESCUE", "0") or 0.0)
    if _rescue_r > 0.0:
        _ctype = {str(c.get("id")): str(c.get("type") or "") for c in components}
        _used_canon = set()
        _cur = {}
        for _cid, _rec in out.items():
            _d = _rec.get("designator") or {}
            _ct = _ctype.get(str(_cid), "")
            _c0 = canon_designator(str(_d.get("text") or ""), _ct)[0]
            if _c0:
                _used_canon.add(_c0)
                _cur[str(_cid)] = (float(_d.get("d") or 999.0), _c0)
        # 候选用**全部文本**（含已经被别的元件「占用」的框）：实测（§5.270 旁证）
        # 已读到的位号串几乎都落在**邻件**的owned 集合里，只看自由文本一个候选都
        # 拿不到（0011：456 个自由文本，最近的可读位号距离 131px）。占用只影响
        # 候选来源，不构成约束——真正的约束是「一对一」和「不许变差」。
        _pool = []
        for _t in texts:
            if _t.get("wide"):
                continue
            if not _t.get("bbox"):
                continue
            _pool.append(_t)
        _slots = []
        for _c in components:
            _cid = str(_c.get("id"))
            _cb = _c.get("bbox_img") or _c.get("bbox")
            if not _cb:
                continue
            _ty = _ctype.get(_cid, "")
            if not TYPE_PREFIX.get(_ty):
                continue
            if _ty in NO_DESIGNATOR:
                # `v`（电源符号）在 `TYPE_PREFIX` 里挂着 `TP`，但它是**没有位号**的
                # 类型；放它进来会把 `TP1` 这种串吸到电源符号上（`0019 ∅498`
                # 实测就是这样换名的），所以这里按 NO_DESIGNATOR 挡掉。
                continue
            _cd = _cur.get(_cid, (None, None))[0]
            for _t in _pool:
                _tb = _t.get("bbox")
                _got = _best_read(_t.get("reads") or {}, "designator",
                                  lambda s, _ty=_ty: canon_designator(s, _ty)[0] != "")
                if not _got:
                    continue
                _can, _fid = canon_designator(str(_got[1]), _ty)
                if not _can or _can in _used_canon:
                    continue
                _dx = max(float(_cb[0]) - float(_tb[2]), 0.0, float(_tb[0]) - float(_cb[2]))
                _dy = max(float(_cb[1]) - float(_tb[3]), 0.0, float(_tb[1]) - float(_cb[3]))
                _dist = (_dx * _dx + _dy * _dy) ** 0.5
                if _dist > _rescue_r:
                    continue
                # 已有位号的元件只允许「明显更近」时改绑（余量 10px），否则不动。
                if _cd is not None and _dist + 10.0 >= _cd:
                    continue
                _slots.append((_dist - 12.0 * _fid - 6.0 * float(_got[2]),
                               _can, _got[1], _got[2], list(_tb), _dist, _fid, _cid))
        _slots.sort(key=lambda x: x[0])
        _used_comp = set()
        _n_rescue = 0
        if os.environ.get("STAGE2_DESIG_DBG"):
            _probe = []
            for _c in components[:400]:
                _cb = _c.get("bbox_img") or _c.get("bbox")
                if not _cb:
                    continue
                _ty = _ctype.get(str(_c.get("id")), "")
                _best = (1e9, "", "")
                for _t in _pool:
                    _tb = _t.get("bbox")
                    _got = _best_read(_t.get("reads") or {}, "designator",
                                      lambda s, _ty=_ty: canon_designator(s, _ty)[0] != "")
                    if not _got:
                        continue
                    _dx = max(float(_cb[0]) - float(_tb[2]), 0.0, float(_tb[0]) - float(_cb[2]))
                    _dy = max(float(_cb[1]) - float(_tb[3]), 0.0, float(_tb[1]) - float(_cb[3]))
                    _dd = (_dx * _dx + _dy * _dy) ** 0.5
                    if _dd < _best[0]:
                        _best = (_dd, str(_got[1]), _ty)
                if _best[0] < 1e9:
                    _probe.append(_best)
            _probe.sort()
            print("[dbg rescue] free_desig_texts=%d slots=%d r=%.1f bound=%d nearest=%s"
                  % (len(_pool), len(_slots), _rescue_r, len(_used_canon),
                     [("%.0f" % p[0], p[1], p[2]) for p in _probe[:6]]))
        for _s in _slots:
            _cid = _s[7]
            if _cid in _used_comp or _s[1] in _used_canon:
                continue
            _used_comp.add(_cid)
            _used_canon.add(_s[1])
            out[_cid]["designator"] = {"text": _s[2], "canon": _s[1], "conf": float(_s[3]),
                                       "box": _s[4], "d": float(_s[5]), "fit": _s[6],
                                       "box_i": -1, "type_fix": None, "rescued": True}
            _n_rescue += 1
            if _dbg_want(cid):
                print("[dbg] rescue %s type=%s <- %s (canon=%s, d=%.1f)"
                      % (_cid, _ctype.get(_cid, ""), _s[2], _s[1], _s[5]))
        if _n_rescue and _dbg_want(cid):
            print("[dbg] rescue total %d" % _n_rescue)

    # 候选池保洁：`canon` 为空的串（`0603`/`U60` 这类封装或乱读）留着只会污染
    # 阶段三的仲裁与诊断视图（台账 §5.283 里能看到它们混在黄框里），直接丢掉。
    if os.environ.get("STAGE2_SLOT_CLEAN", "1") not in ("0", "off", "false"):
        desig_slots = {k: [s for s in v if (s or {}).get("canon")]
                       for k, v in desig_slots.items()}

    if DUMP_READS:
        LAST_TEXT_READS[:] = [
            {"box": [round(float(x), 1) for x in t["bbox"]],
             "d": round(float(t.get("d") or 0.0), 1),
             "inside": bool(t.get("inside")),
             "wide": bool(t.get("wide")),
             "reads": {r: [str(v[0]), round(float(v[1] or 0.0), 3)]
                       for r, v in (t.get("reads") or {}).items()
                       if isinstance(v, (tuple, list)) and v and v[0]}}
            for t in texts]
        # ★ 落盘（`STAGE2_DUMP_READS=1` 时）：`LAST_TEXT_READS` 原来只活在进程内，取不了证。
        # 写 JSONL 而不是 print —— `Tee-Object` 会吞掉子进程的 stdout。
        try:
            _p = os.environ.get("STAGE2_DUMP_PATH") or os.path.join("work", "_reads_dump.jsonl")
            with io.open(_p, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"case": os.environ.get("STAGE2_DUMP_CASE", ""),
                                     "n": len(LAST_TEXT_READS),
                                     "reads": LAST_TEXT_READS},
                                    ensure_ascii=False) + "\n")
        except Exception:
            pass

    return {"comps": out, "net_labels": net_labels, "desig_slots": desig_slots,
            "struct_slots": struct_slots,
            "n_texts": len(texts), "n_owned": sum(len(v) for v in owned.values()),
            "n_free": len(free)}


# ---------------------------------------------------------------------------
def _selftest() -> int:
    """合成用例（README 已知限制：任何新写的判定函数都要带合成自检）。"""
    comps = [
        {"id": "c0", "type": "r", "bbox_img": [100, 100, 140, 110]},   # 两脚电阻
        {"id": "c1", "type": "box", "bbox_img": [300, 100, 400, 300]},  # IC
        # ★ `STAGE2_PINNAME_OUTSIDE`（框外补名）专用：这颗 IC 的**名字与号码都在框外**。
        {"id": "c2", "type": "box", "bbox_img": [600, 100, 700, 200]},
    ]
    pins = {"c0": [[90, 105], [150, 105]],
            "c1": [[290, 150], [290, 250], [410, 150], [410, 250]],   # 引脚画在框外 10px
            "c2": [[700, 130], [700, 170]]}
    def T(box, **reads):
        return {"bbox": tuple(box), "reads": {k: (v, 0.9) for k, v in reads.items()}}
    texts = [
        T((100, 82, 118, 94), designator="R7"),            # 电阻位号（框上方）
        T((104, 114, 130, 126), value="10K"),              # 电阻值（框下方）
        T((330, 120, 352, 134), designator="U1"),          # IC 位号（框内上侧）
        T((320, 140, 372, 154), partname="MSP432E4"),      # IC 器件名
        T((306, 144, 318, 158), pin_name="VDD"),           # 框内引脚名
        T((306, 244, 318, 258), pin_name="VSS"),
        T((282, 144, 298, 158), pin_number="12"),          # 框外引脚号
        T((402, 244, 418, 258), pin_number="13"),
        T((500, 400, 540, 414), net_label="VBUS"),         # 远处网络标签
        T((702, 124, 716, 136), pin_number="7"),           # c2：框外号码（贴着脚）
        T((702, 164, 716, 176), pin_number="8"),
        T((714, 124, 752, 136), pin_name="VDDIO"),         # c2：框外**名字**，中心距框 33px
        T((714, 164, 744, 176), pin_name="VSSA"),
        T((640, 140, 692, 154), partname="STM32F4"),       # c2：框内器件名（别被名字抢走）
    ]
    res = assign(comps, pins, texts)
    fails = []
    n_chk = [0]

    def chk(cond, msg):
        n_chk[0] += 1
        if not cond:
            fails.append(msg)

    r = res["comps"]["c0"]
    chk(r["designator"] and r["designator"]["text"] == "R7", "c0 位号应为 R7: %r" % (r["designator"],))
    chk(r["value"] and r["value"]["text"] == "10K", "c0 参数应为 10K: %r" % (r["value"],))
    b = res["comps"]["c1"]
    chk(b["designator"] and b["designator"]["text"] == "U1", "c1 位号应为 U1: %r" % (b["designator"],))
    chk(b["partname"] and b["partname"]["text"] == "MSP432E4", "c1 器件名: %r" % (b["partname"],))
    names = sorted(v["name"]["text"] for v in b["pins"].values() if v["name"])
    nums = sorted(v["number"]["text"] for v in b["pins"].values() if v["number"])
    chk(names == ["VDD", "VSS"], "c1 引脚名: %r" % (names,))
    chk(nums == ["12", "13"], "c1 引脚号: %r" % (nums,))
    chk(any(n["text"] == "VBUS" for n in res["net_labels"]),
        "网络标签应包含 VBUS: %r" % (res["net_labels"],))

    # ★ 框外补名（`STAGE2_PINNAME_OUTSIDE`）：c2 的名字与号码都在框外。
    # 关键断言与开关状态无关：**引脚号必须拿到 7/8**（late pass 不许饿死引脚号）；
    # 名字槽在 `OUTSIDE=0` 时必须是空的（旧行为），开着时必须是 VDDIO/VSSA。
    d = res["comps"]["c2"]
    dnames = sorted(v["name"]["text"] for v in d["pins"].values() if v["name"])
    dnums = sorted(v["number"]["text"] for v in d["pins"].values() if v["number"])
    chk(dnums == ["7", "8"], "c2 引脚号应为 7/8（框外号码不被名字抢走）: %r" % (dnums,))
    if PINNAME_OUTSIDE > 0:
        chk(dnames == ["VDDIO", "VSSA"], "c2 框外引脚名应为 VDDIO/VSSA: %r" % (dnames,))
    else:
        chk(dnames == [], "c2 关掉框外补名时不应有引脚名: %r" % (dnames,))

    # 位号去重：两个元件抢同一个 R7
    dup = assign([{"id": "a", "type": "r", "bbox_img": [0, 0, 20, 10]},
                  {"id": "b", "type": "r", "bbox_img": [200, 0, 220, 10]}],
                 {}, [T((0, -14, 18, -2), designator="R7"),
                      T((200, -14, 218, -2), designator="R7")])
    got = [v["designator"] for v in dup["comps"].values()]
    chk(sum(1 for g in got if g) == 1, "重复位号应只留一个: %r" % (got,))

    # ★ 内涵互校验（`STAGE2_CONFORM_KEY`）：位号前缀是元件类型的**内涵** ——
    #   读出的前缀能反过来判定类型（`LED` 只属于 `led`），而同一块文字框在
    #   不同角色下的读法要一起看（`designator` 读垃圾、`pin_name` 读得对）。
    #   默认关时两条回收路径都必须**完全不生效**（逐位不变）。
    _m = canon_designator_meaning
    chk(_m("R14", "r")[:2] == ("R14", 1.0), "strict 路径应原样: %r" % (_m("R14", "r"),))
    if not CONFORM_KEY:
        chk(_m("LED2", "d")[0] == "", "默认关时不许翻类型: %r" % (_m("LED2", "d"),))
        chk(_m("7LED1", "led")[0] == "", "默认关时不许削胶连: %r" % (_m("7LED1", "led"),))
    else:
        # `d` 的允许前缀只有 `D`，而 `LED2` 在 `led` 下是干净位号 ⇒ 类型判错，改判 `led`
        chk(_m("LED2", "d")[:3] == ("LED2", CONFORM_TYPE_PEN, "led"),
            "d 上的 LED2 应翻成 led: %r" % (_m("LED2", "d"),))
        # 开头胶连（`7`）可以削；**中间**胶连（`N7ED1`）不许猜 —— 猜出来会是 `D1`
        chk(_m("7LED1", "led")[:3] == ("LED1", CONFORM_GLUE_PEN, None),
            "开头胶连应削掉: %r" % (_m("7LED1", "led"),))
        chk(_m("N7ED1", "led")[0] == "", "中间胶连不许猜: %r" % (_m("N7ED1", "led"),))
        # `D` 是 `d`/`led` 共有前缀 ⇒ 不足以翻类型（不许把二极管改判成 LED）
        chk(_m("D3", "d")[:2] == ("D3", 1.0), "共有前缀不得翻类型: %r" % (_m("D3", "d"),))

    # ★ 内涵「遗漏检测」审计（`STAGE2_CONFORM_AUDIT` 默认开，**只写报告**）：
    #   合成两颗「本该挂上键却挂不上」的元件 —— ① 类型判错（`d` 上的 `LED2`）；
    #   ② 只信 `designator` 角色（同一块框的 `pin_name` 读作 `7LED1`）。
    #   开关关时两颗都拿不到位号，但审计必须**把它们点出来**（这就是「检索遗漏」）。
    au_comps = [{"id": "d0", "type": "d", "bbox_img": [200, 200, 220, 220]},
                {"id": "d1", "type": "led", "bbox_img": [400, 200, 420, 220]}]
    au_pins = {"d0": [[210, 200], [210, 220]], "d1": [[410, 200], [410, 220]]}
    au_texts = [T((200, 224, 240, 238), designator="LED2"),
                T((400, 224, 440, 238), designator="YXEP1", pin_name="7LED1")]
    au_res = assign(au_comps, au_pins, au_texts)
    _au = {r["cid"]: r for r in LAST_CONFORM}
    if not CONFORM_KEY:
        chk(au_res["comps"]["d0"]["designator"] is None, "关时 d0 不应挂上位号")
        if CONFORM_AUDIT:
            chk(_au.get("d0", {}).get("canon") == "LED2" and _au["d0"]["how"] == "type",
                "审计应点出 d0 的遗漏(LED2/type): %r" % (_au.get("d0"),))
            chk(_au.get("d0", {}).get("type_fix") == "led",
                "审计应给出 d0 的类型改判: %r" % (_au.get("d0"),))
            chk(_au.get("d1", {}).get("canon") == "LED1" and _au["d1"]["how"] == "glue",
                "审计应借串角色点出 d1 的遗漏(LED1/glue): %r" % (_au.get("d1"),))
            chk(_au.get("d1", {}).get("role") == "pin_name",
                "审计应记下借用的角色: %r" % (_au.get("d1"),))
    else:
        chk(bool(au_res["comps"]["d0"]["designator"]), "开时 d0 应挂上 LED2")
        chk(not _au, "开时已被回收的元件不该再进遗漏审计: %r" % (_au,))
        # 同一通道的「多余检测」：`led` 的 Name 槽在官方 10GT 里是型号，
        # 我方 `name = key` ⇒ 这颗必然判错，必须被点出来（带器件名读数位）。
        _ex = {r["cid"]: r for r in LAST_CONFORM_EXTRA}
        if CONFORM_AUDIT:
            chk("d0" in _ex and _ex["d0"]["type"] == "led" and "型号" in _ex["d0"]["why"],
                "多余检测应点出 d0（led 的 Name 槽口径不符）: %r" % (_ex.get("d0"),))

    print("自检 %d 项：%s" % (n_chk[0], "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
