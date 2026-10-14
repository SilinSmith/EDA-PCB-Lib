# -*- coding: utf-8 -*-
"""第二阶段 · 组装：把 stage1 的几何产物 + 第二阶段的文字，拼成官方提交格式。

坐标系在**这一层**翻转（`y_json = H - y_img`，左下角原点）。stage1 全程用图像
坐标（左上角原点），OCR 用的也是图像坐标的框 —— 只在出口处翻一次，避免来回翻。

网络标签的逻辑合并也在这里：几何只连得出一部分网络（实测拿 GT 自己的走线去连
同一 net 的引脚，召回也只有 0.658），剩下 34% 靠纸上同名的网络标签相连。做法是
把标签绑到最近的那条走线的**自由端**，然后把同名标签的网络并起来。
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:                                    # 作为包导入（submission\run.py 把 eda-pcb\ 加进 sys.path）
    from envflag import env_on        # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage2\assemble.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on

from stage2 import structmatch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VALUE_TYPES = ("r", "c", "l", "fuse", "crystal_2pin", "crystal_3pin",
               "crystal_4pin", "crystal")
LABEL_TOL = 16.0        # 网络标签离走线自由端多远还算「贴在这条线上」
FREE_TOL = 2.0          # 两个端点离多近算「接上了」
# 位号没读出来时**要不要**输出这颗元件（默认**输出**，键用 `pNNN` 兜底）。
#
# 旧口径（`schematic_parser/metrics.py` 第一版）下默认是「丢掉」，理由是：
# 官方 component_f1 遍历 `set(gt) | set(pred)` 的键并集，键一字符之差就是
# 「多一个 + 少一个」；GT 里 44% 的元件键是画面上根本不存在的 `∅NNN`
# （全局对象计数器，不可复现），我们永远给不出那些键，输出它们只会白送 FP。
# 当时实测：只丢兜底键元件 7.2861 → 8.2617，连 net 超图里的引用一起剪
# 8.2617 → 9.7805（合计 +2.49 分）。
#
# **2025-10 官方 Q&A 推翻了这条**（`docs/QA_2025.txt` 评测 Q5）：
# 「新的10个golden case仍有大量 component key 是无法根据图片得到的内部编号
# 如 ∅109，请问这些组件如何进行 componentF1 的判分？」→
# **「Vcc和gnd是∅，直接是坐标匹配」**；评测 Q6 又补「GND的序号名字不影响评分」。
# 也就是说**那 44% 的 `∅NNN` 根本不靠键匹配、靠坐标匹配**，弃掉它们等于把
# 一半的 GT 直接扔掉。实测（220 张、官方口径 `--unkeyed-match gt`）：
#   `STAGE2_DROP_UNKEYED=1` → **15.5084**（component F1 0.2610 tp 1747）
#   `STAGE2_DROP_UNKEYED=0` → **21.1165**（component F1 0.5077 tp 5077）
# 足足 **+5.61 分**（component 的 tp 翻了近三倍，`component.missing` 6594→3270）。
# netH / line 各掉一点点（pred 引脚变多、FP 变多），但远不抵 component 的收益。
#
# ⇒ 默认改成 **0（输出）**。`STAGE2_DROP_UNKEYED=1` 可退回旧行为做对照。
DROP_UNKEYED = env_on("STAGE2_DROP_UNKEYED", "0")

# `box`(IC) 的引脚号：OCR 读出的号**一颗 IC 里有一个读废就整颗退回位置号**，
# 还是**逐引脚**回退。
#
# 旧行为（=0）是全有全无：`all(n.isdigit())` 且互不相同且都在 1..999 才采信，
# 否则 `for k in range(len(raw)): raw[k]["num"] = str(k + 1)`。一颗 72 脚的 IC
# 只要有一个号读成空/读成 `l`/撞号，**其余 71 个正确的号也一起作废**。
# 而 `pinF1` 只在 `box`(IC) 上算分（官方评测 Q2：「非box类型元器件 pin name,
# pin number不参与评分」），所以这等于把唯一算分的引脚全扔掉 —— 实测读到率
# 只有 0.272，344 个键命中的 box 元件里 GT 4064 个引脚我们只对 1046 个（0.257）。
#
# 新行为（=1，默认）逐引脚采信：某个号是 1..999 的整数或经独立验证的字母数字编号、且没跟同一颗 IC 里别的
# 采信号撞，就用它。未读/无效/重复落选的号默认不发布，也不进入网络引用。
# 内部位置号回退仅供下述显式 legacy 模式使用，不是默认输出的真实编号。
BOX_PIN_PARTIAL = env_on("STAGE2_BOX_PIN_PARTIAL", "1")
BOX_PIN_DUPLICATE_BEST = env_on("STAGE2_BOX_PIN_DUPLICATE_BEST", "1")

# `STAGE2_BOX_PIN_DROP_UNREAD`：`box`(IC) 元件里**没读出号、靠位置补出来**的引脚桩
# 不再写进引脚表（也不进网络），而不是给它发明一个位置号（`8`/`9`/…）。
#
# 动机（10GT 实测）：我们在一颗 IC 上交出的引脚数**普遍多于封装真脚数** ——
# case 5 的 `U1`(`TPS61240DRVR`) GT 只有 `1..7`，我们因为桩有 12 个、读到的号是
# `1..7`，剩下 5 个被补成 `8..12`；`U2` 补出 `8..10`、`U3`(`AMS1117-3.3`) 补出 `7`。
# 这些「发明出来的号」只会变成 `pin.extra`（纯 fp），还会把假引脚塞进网络。
# 10GT 的 `pin.extra` = 69（占我们交出的 188 个引脚的三分之一），`pin.missing` 只有 17。
#
# 取值：
#   * `1`/`all`（**默认**）= 一律不发布没读到的 IC 号；全无读数也不补 1..N。
#   * `0`/`off`    = 显式 legacy 行为（位置号补满）；
#   * `mixed`      = 显式 legacy 行为：只有至少读懂一个号时才丢未知号，
#     全无读数时仍按位置补 1..N。这种兼容模式会猜号，不能作为默认真实读数。
#   * `cap`        = 显式 legacy 上限行为：允许猜出不超过最大已读号的缺号。
#     仅为旧实验/调用保留，不能把补出的缺号视为图上识别的编号。
#
# 历史 A/B（旧默认 mixed，不能代替当前新数据验证）：
#   10GT(9)：老 44.70(pin 0.1881/netH 0.4746) → `1` **47.67**(pin **0.2720**/netH **0.4971**)
#            → `mixed` **47.67**（与 `1` 逐位相同）
#   19 张  ：老 26.46(pin 0.0935) → `1` **28.31**(pin **0.1320**) → `mixed` **28.31**（同 `1`）
#   220 张 ：老 25.0109(pin 0.0619/netH 0.2078) → `1` 25.0035(pin 0.0657/netH 0.2048)
#            → **`mixed` 25.0117**(pin 0.0649/netH 0.2059) ⇒ 只有 `mixed` 端到端不降。
_DROP_MODE = os.environ.get("STAGE2_BOX_PIN_DROP_UNREAD", "all").strip().lower()
BOX_PIN_DROP_UNREAD = _DROP_MODE not in ("", "0", "false", "off", "none")
BOX_PIN_DROP_MIXED_ONLY = _DROP_MODE in ("mixed", "mix", "partial")
BOX_PIN_DROP_CAP = _DROP_MODE in ("cap", "max", "ceiling")

# ★ `STAGE2_BOX_PIN_NEED_NAME`（默认关）：box 引脚「没读出**名字**就一个都不发」。
# 依据（`schematic_parser/metrics.py:455-456` 官方口径 + 全库实测）：
#   * `pin_f1` 的键是 `(元件, pin_key_to_number(pk))`，命中之后还要求
#     `p.get('pinname') == g.get('pinname')`，不等则 **fp += 1 且 fn += 1**；
#   * 全库 `discover(DEFAULT_DATA_ROOT)` 实测 box 引脚 9356 个、**pinname 非空 9356、空 0**
#     ⇒ **没有一个 GT 引脚名是空的**，所以「拿引脚号当名字」(`nm = r["nm"] or num`)
#     必然 `pin.name_mismatch`；
#   * 于是「发了但没名字」= 1 fp + 1 fn，「干脆不发」= 1 fn ⇒ **对 F1 来说不发严格更优**
#     （F1 = 2tp/(2tp+fp+fn)，分母少 1 或 2，tp 不变）。
# 代价：这些桩会少进一个网络（net_hypergraph 按引脚键匹配）。
# ★ 19 张定点档 A/B（`work\tools\eval\fast.py --no-vlm --save …`，均为 pincol 打开时）：
#     关 23.16（netH 0.2004，10GT 35.49）→ **开 23.83**（netH **0.2162**，10GT **37.09**）
#     逐张：3 +2.20 / 4 +3.99 / 5 +3.51 / 9 +4.58 / 7 +1.01 / 8 +1.66 / 10 +1.07 / 1 +0.48
#           2 **−4.17**（netH 0.4107→0.2895）/ 0024 −1.34 / 0103 −0.82
#     ⇒ 8 张赚 3 张亏、净值 +0.67（相对无 pincol 的 23.30 基线）。**默认改为开**。
BOX_PIN_NEED_NAME = env_on("STAGE2_BOX_PIN_NEED_NAME", "1")

# 兜底键的形状（`p000`…），和 `schematic_parser/metrics.py` 的 `FKEY_RE` 一致。
FKEY_RE = re.compile(r"^p\d{3,4}$")
# **兜底键元件的引脚要不要写进 net 超图**。
# `net_hypergraph` 的 FP = `total_pred_pins - tp`，而兜底键元件的引脚号
# （位置序号或硬读的号）对上一颗 GT 里同样是 `∅NNN` 的元件几乎不可能对
# （官方 Q&A 数据 Q6：「编号是随机生产的」）⇒ 这些引用基本是纯 FP。
# 注意：component 那一项**仍然要输出这些元件**（它们靠坐标匹配拿 tp），
# 这里剪的只是 net 里的引用。
# 220 张实测（官方口径 `--unkeyed-match gt`）：`=0` → **21.1165**，
# `=1` → **22.9068**（**+1.79**）。收益全在 net 那条线上：
# `net_hypergraph` F1 0.1357 → 0.1862（P 0.1328→0.3254，pred 引脚 23743→9676），
# `net_line` 0.0172 → 0.0195；而 component / pin 两项**一个数都没动**。
# 剪掉的正好是 `DROP_UNKEYED=0` 换 component 时多出来的那 14000 个假引用。
NET_DROP_UNKEYED = env_on("STAGE2_NET_DROP_UNKEYED", "1")
# Retain anonymous power/ground members when the image establishes exactly
# one terminal. Arbitrary GT numbering must not remove a real graph member.
KEEP_CONFIRMED_POWER_PINS = env_on("STAGE2_KEEP_CONFIRMED_POWER_PINS", "1")

# ★ 电源/地符号（GT 里键为 `∅NNN`）**只用来合并网络，不写进 hyperGraph**。
#
# 事实（`work\tools\diag\_phibox.py`，10GT）：GT 的 **160 条网络里有 108 条**
# 恰好带一个 `∅NNN.k` 成员（全是 `gnd` / `v` 符号），而 `∅NNN` 的引脚号是
# **随机生成的内部号**（`∅202` 的脚是 `pin_202`、`∅200` 的是 `pin_198`、
# `∅121` 的是 `pin_119`，官方 Q&A「编号是随机生产的」）⇒ 这些成员**永远拿不到 tp**，
# 写进 hyperGraph 就是纯 FP（所以 `NET_DROP_UNKEYED` 剪掉它们是对的）。
#
# 但它们携带**拓扑信息**：一个 GND 符号在电气上就是**一个节点**，所有碰到它的
# 走线必须属于同一条网络。我们恰好把符号的每个接触点都留成了一条独立引脚
# （`p012` 有 16 个脚，分落在许多条碎网里）⇒ 正是**过分割**的来源。
# `_netoracle.py` 实测：把我们所有引脚引用当成一个池子去和 GT 求交，tp 上限是
# **299**，而实际只有 **190** ⇒ **109 个 tp 纯粹丢在分组上**（pred 304 条网 vs GT 160 条）。
#
# 所以这里：把 φ 符号当**并查集的连边**用，合并完之后再把它的引用从 hyperGraph 里去掉。
# `STAGE2_SYM_MERGE=0` 可退回旧行为做对照。
# ⚠️ 默认 **关**。10GT 实测 2×2（`work\_gt10_ns0/ns1/em0/em1`）：
#   SYM=0/EDGE=0 → 42.5191｜SYM=0/EDGE=1 → **42.5418**｜SYM=1/EDGE=0 → 42.4147｜SYM=1/EDGE=1 → 42.4373
# ⇒ 合并电源符号**净亏 −0.104**（`net_hypergraph 0.4444 → 0.4418`，tp 190→189）。原因见下面的
# `merge_by_symbol` 注释：我们的电源符号引脚本来就散落在多条碎网里，合并它们会把本可以
# 各自配对的碎网粘成一条，反而丢掉一对一指派的机会。代码留着，旋钮可开。
SYM_MERGE = env_on("STAGE2_SYM_MERGE", "0")
POWER_TYPES = frozenset((
    "gnd", "gnda", "gndd", "agnd", "dgnd", "pgnd", "gnd_ref", "earth",
    "v", "vcc", "vdd", "vbus_in", "net_input", "net_output", "power"))

# ★ `net_line` 边的最小长度闸门（`STAGE2_MIN_EDGE_LEN`，默认 20.0）。
# 官方 `metrics._seg_distance` 要求**两个端点都**在 `line_tol = 5.0` 内才算命中，
# 所以边必须几乎和 GT 逐点重合。实测 case `1`：GT **102 条边、最短 12px、p50 = 67px**，
# 我们 **363 条边、其中 87 条短于 10px、p50 只有 25px**，边总长 22648 vs GT 8389（2.7 倍）。
# 那些几像素的碎边全是**元件符号笔画与骨架化毛刺**，GT 里根本不存在 ⇒ 纯 `fp`。
# 离线扫（`work\_gt10_s2new` 逐臂改写 `nets[*].edges` 后重打分，`tp` 恒为 42）：
#   阈值 0 → `net_line` 0.0399（fp 1555）｜5 → 0.0410（1486）｜8 → 0.0442（1287）
#   12 → 0.0466（1158）｜20 → 0.0494（1030）｜**25 → 0.0514（968）**｜30 → 0.0453（tp 掉到 37）
# ⇒ 取 25 是峰值，取 20 更保守（GT 全部 639 条边的 1% 分位是 13px、5% 分位是 26px）。
MIN_EDGE_LEN = float(os.environ.get("STAGE2_MIN_EDGE_LEN", "20"))

# ★ 共线碎边焊接（`STAGE2_EDGE_MERGE`，**默认 0 = 关**，理由见本节 ⚠️ 那一段）。
#
# 为什么必须有：官方 `net_line_f1`（`schematic_parser/metrics.py:522-562`）先把
# 「配对上的那一对网络」的预测边与 GT 边做**一对一**指派
# （`got = _assign(cost)`，`cost` 用 `_seg_distance(pe[:2], ge[:2], cfg.line_unordered)`，
# 超过 `cfg.line_tol` 记 `INF`），`tp = len(got)`、`fp = len(p_edges) - tp`。
# **一条 GT 长边最多只能被我们的一条边命中。**
#
# 我们的走线追踪会把一条完整轨道切成许多首尾相接的短线（骨架化 + 交叉点断开），
# 而 GT 里那是一条 100+ px 的整边 ⇒ 只有 1 条能拿 `tp`，其余全部变 `fp`。
# 实测 case `1`：GT 102 条边、p50 = 67px；我们 p50 只有 25px、边总长是 GT 的 2.7 倍。
# 所以**把同一网络里「端点相邻 + 方向共线」的碎边焊成一条两点直线**是严格变好的操作
# —— 它不会改变网络的引脚集合（`net_hypergraph` 逐位不变），只砍 `net_line` 的 fp。
#
# ⚠️ 关键细节：官方只取边的**前两个点** `pe[:2]`。所以焊完必须输出**两点**直线
# （首端、末端），把中间的拐点丢掉；否则焊了也没用（第一个点还是旧的第一段起点）。
# 只焊共线的：夹角超过 `EDGE_ANG` 就不焊，免得把 L 形拐角拉成一条斜线。
# ⚠️ 默认 **关**。10GT 上是正的（`--stage1 work\_gt10_pb10\runs\stage1` 扫参：
#   JOIN 6 → 42.5418｜10 → 42.5317｜16 → 42.5580｜24 → 42.5651｜**40 → 42.5695**｜
#   60 → 42.4618（tp 48→37）｜100 → 42.2801；`EDGE_ANG` 6/15/30 几乎无差，取 6）
# 但 19 张快测上是**平的且多爆两个零**（`work\_fast_base\edgemerge19.json` vs
# `edgemerge_off19.json`：开 25.3642 / 导线 tp 39 / 导线 F1=0 共 **10** 张；
# 关 25.3549 / 导线 tp 46 / 导线 F1=0 共 **8** 张）。用户明确在意「某个小分爆 0」，
# 所以**默认关**、旋钮留着。见 `merge_collinear` 的完整推导。
EDGE_MERGE = env_on("STAGE2_EDGE_MERGE", "0")
EDGE_JOIN = float(os.environ.get("STAGE2_EDGE_JOIN", "40.0"))     # 端点相接容差（px）
EDGE_ANG = float(os.environ.get("STAGE2_EDGE_ANG", "6.0"))        # 共线夹角容差（度）


def _ang_deg(d1, d2) -> float:
    """两个方向向量的夹角（度）；任一为零向量时返回 180（不焊）。"""
    n1 = math.hypot(d1[0], d1[1]); n2 = math.hypot(d2[0], d2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = (d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def merge_collinear(runs: List[list], join_tol: float = None,
                    ang_tol: float = None) -> List[list]:
    """把若干两点线段按「端点相接 + 方向共线」焊成尽量长的两点线段。

    `runs` 每项是 `[p0, p1]`（`{"x","y"}` 字典）。返回同样形态的列表。
    """
    join_tol = EDGE_JOIN if join_tol is None else join_tol
    ang_tol = EDGE_ANG if ang_tol is None else ang_tol
    out = [[r[0], r[1]] for r in runs if len(r) >= 2]
    changed = True
    while changed:
        changed = False
        n = len(out)
        for i in range(n):
            a, b = out[i]
            for k in range(i + 1, n):
                c, d = out[k]
                # 四种端点配对：(a,c)(a,d)(b,c)(b,d)，各对应不同的「向外延伸」方向
                for (p, o, q, r2) in ((a, b, c, d), (a, b, d, c),
                                      (b, a, c, d), (b, a, d, c)):
                    if math.hypot(p["x"] - q["x"], p["y"] - q["y"]) > join_tol:
                        continue
                    d1 = (p["x"] - o["x"], p["y"] - o["y"])      # 在 p 处向外
                    d2 = (r2["x"] - q["x"], r2["y"] - q["y"])    # 从 q 沿 k 走
                    if _ang_deg(d1, d2) > ang_tol:
                        continue
                    out[i] = [o, r2]
                    del out[k]
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return out

# 阻容的 `value` 只写「含字母/Ω/μ」的串，否则写 null。
# 官方 `component_text_ok`（metrics.py:149-157）对 VALUE_TYPES 比 `value` **逐字符相等**，
# 而官方语料里 r/c 的 value 只有两种形态：`None`（220 张 3294 个里的 966 = 29.3%）或
# **含字母**的串（`4.99k` / `100nF` / `10kΩ`）——**一个纯数字都没有**。
# 我们却写出了 489 个单字符（`1` `4` `5` …）和 77 个纯数字（`work/tools/diag/_val_why.py`
# 的 Top 对里 `100nF -> 1` 出现 36 次、`None -> 1` 18 次 —— 那些数字是被当成值抓进去的
# **引脚号**）。这些串在任何情况下都不可能是对的：GT 是 `None` 时，我们写了个垃圾，
# 一个本来能对的元件变成错（同时吃一个 fp 和一个 fn）；GT 是值串时我们本来就错。
# 所以这条门控是**严格不亏**的纯赚。220 张实测 14.7568 → 15.4801（+0.7233），
# text_mismatch 1061 → 948，pin / net_hypergraph / net_line 一项都没动。
VALUE_LIKE = re.compile(r"[A-Za-z\u03A9\u2126\u00B5\u03BC]")

# ---- 两脚元件的官方引脚「图章」---------------------------------------------
# 官方 GT 的引脚点位**不是走线接点**，而是「元件 bbox 中心沿符号主轴 ±SPAN」的理想点 ——
# `schematic_parser/stamp.py` 把每个类固化成一个正方形小图章，两个引脚就贴在中心两侧。
# 220 张实测（`work/tools/diag/_pin_stamp.py`）：
#   * `r/c/l/fuse/crystal_2pin` 的 `pin_1` **恒在 −SPAN 一侧**、`pin_2` 恒在 +SPAN 一侧，
#     纯度 1.00（r 竖 853 / 横 715，c 1287/342，l 27/88）；
#   * `d`/`led` **反号**且只有 0.68~0.90 纯度（1 脚是阳/阴极取决于符号怎么画）；
#   * SPAN 基本恒为 20，只有 `fuse` 是 21；
#   * `crystal_2pin` 的引脚轴与 bbox 长短边**垂直**，其余类一致。
# 我们原来的 pin 点来自 stage1 的导线附着点，实测与 GT 点的距离 p50 = 39.9px（r/c）——
# 那不是噪声，是**系统性错了一个符号长**。照图章合成后 ≤5px 命中率从 0.058 升到 0.62
# （判轴用 raw 框的话更高）。
#
# ⚠️ 判轴**必须**用贴章前的 `bbox_raw_img`：图章一律是正方形，`bbox_json` 已经把这个
# 信息抹掉了（raw 判轴命中率 0.955，贴章后的框只有 0.630 —— 对 r/c/l 而言就是瞎猜）。
PIN_STAMP_TYPES = frozenset({"r", "c", "l", "fuse", "crystal_2pin", "d", "led"})
PIN_STAMP_POS = frozenset({"d", "led"})          # 这一类 pin_1 在 +SPAN 一侧
PIN_STAMP_PERP = frozenset({"crystal_2pin"})     # 这一类引脚轴与 bbox 长短边垂直
# `led` 在官方语料里轴向不同时符号也不同（竖着 89/99 在 −SPAN、横着 57/84 在 +SPAN），
# 但做成「按轴选符号」后 220 张端到端 14.7496，比统一取 + 的 14.7568 略低（差 3 个
# net 引脚，纯噪声）—— 所以保留这条更简单的规则。
PIN_STAMP_SPAN = {"fuse": 21.0}
PIN_STAMP_DEFAULT_SPAN = 20.0
# 关掉就退回「用 stage1 的附着点当引脚点位」（旧行为）
PIN_STAMP_ON = env_on("STAGE2_PIN_STAMP", "1")


def stamp_pins(c: dict, ctype: str, pts_img: Sequence,
               H: float, *, axis_y: Optional[bool] = None) -> Optional[List[Tuple[str, Tuple[float, float]]]]:
    """两脚贴章元件的官方引脚点位。

    返回 `[(引脚号, (x, y)), ...]`，顺序与 `pts_img` 一一对应（图像坐标入参，
    返回的是 **json 坐标**）；不适用时返回 None。

    引脚号按官方约定给：−SPAN 一侧是 `1`、+SPAN 一侧是 `2`（`d`/`led` 反过来）。
    我们自己的附着点落在哪一侧，就继承那个号 —— 这一步决定了 net 超图里
    `(元件键, 引脚号)` 的对不对得上。
    """
    if (not PIN_STAMP_ON or ctype not in PIN_STAMP_TYPES or not len(pts_img)
            or c.get('terminal_group') == 'rgb_led'):
        return None
    bbox = c.get("bbox_json") or []
    if len(bbox) != 4:
        return None
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    raw = c.get("bbox_raw_img") or []
    if len(raw) == 4 and (raw[2] - raw[0]) > 0 and (raw[3] - raw[1]) > 0:
        tall = (raw[3] - raw[1]) >= (raw[2] - raw[0])
    else:
        tall = (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0])
    if axis_y is None:
        axis_y = (not tall) if ctype in PIN_STAMP_PERP else tall
    span = PIN_STAMP_SPAN.get(ctype, PIN_STAMP_DEFAULT_SPAN)
    # pin_1 在轴上的投影符号：+1 表示在 +SPAN 一侧
    s1 = 1.0 if ctype in PIN_STAMP_POS else -1.0
    if axis_y:
        p1, p2 = (cx, cy + s1 * span), (cx, cy - s1 * span)
    else:
        p1, p2 = (cx + s1 * span, cy), (cx - s1 * span, cy)

    side = []
    for p in pts_img:
        proj = (float(H) - float(p[1]) - cy) if axis_y else (float(p[0]) - cx)
        side.append(True if proj * s1 > 0 else False)
    # Several detected contacts may belong to the same physical terminal.
    # Keep their observed side: assigning a duplicate to the opposite terminal
    # makes the network reference a pin which that wire never touches. The
    # caller deduplicates the published pin keys while retaining every raw
    # index -> number mapping, so same-side wire fragments can reconnect.
    # A lone contact does not invent an unseen contact on the opposite side.
    return [(("1", p1) if want1 else ("2", p2)) for want1 in side]


# ---------------------------------------------------------------------------
def _measured_passive_points(stage1, cid, ctype, points):
    """Current colour evidence overrides the legacy coordinate stamp, never ICs."""
    if ctype not in PIN_STAMP_TYPES:
        return {}
    record = (stage1.get("passive_pin_report") or {}).get(cid) or {}
    if record.get("type") != ctype or record.get("coordinate_rule") not in {
            "red_green_pixel_boundary", "image_color_terminal_rules"}:
        return {}
    after = record.get("after") or []
    sources = record.get("sources") or []
    try:
        if len(after) != len(points) or len(sources) != len(points):
            return {}
        for observed, current in zip(after, points):
            distance = math.dist(observed, current)
            if not math.isfinite(distance) or distance > 0.15:
                return {}
    except (TypeError, ValueError):
        return {}
    return {i: point for i, point in enumerate(points)
            if sources[i] in {"body_attached_red_green_transition", "shared_single_green_pixel", "first_red_junction_center", "mixed_colour_span_boundary",
                              "first_filled_junction_center", "lead_wire_colour_boundary",
                              "symmetric_opposed_terminal", "peer_shared_terminal",
                              "direct_pin_corner", "same_symbol_terminal_length"}}


def _confirmed_power_terminal(stage1, cid, ctype, points) -> bool:
    """Only current, single-terminal image evidence can exempt a fallback key.

    Multiple grayscale guesses, stale reports, and unconfirmed detections keep
    the established filtering policy. No external annotation IDs are invented.
    """
    if not KEEP_CONFIRMED_POWER_PINS or ctype not in {"gnd", "v"} or len(points) != 1:
        return False
    rec = (stage1.get("passive_pin_report") or {}).get(cid) or {}
    if rec.get("type") != ctype or rec.get("coordinate_rule") not in {
            "red_green_pixel_boundary", "image_color_terminal_rules"}:
        return False
    after, sources = rec.get("after") or [], rec.get("sources") or []
    if len(after) != 1 or len(sources) != 1 or sources[0] not in {
            "body_attached_red_green_transition", "shared_single_green_pixel",
            "first_red_junction_center", "mixed_colour_span_boundary",
            "first_filled_junction_center", "lead_wire_colour_boundary",
            "peer_shared_terminal", "direct_pin_corner", "same_symbol_terminal_length"}:
        return False
    try:
        distance = math.dist(after[0], points[0])
        return math.isfinite(distance) and distance <= 0.15
    except (TypeError, ValueError):
        return False


def _free_ends(segs: Sequence[Sequence[Sequence[float]]],
               tol: float = FREE_TOL) -> List[Tuple[float, float]]:
    """一段走线的「自由端」：该端点附近没有别的段的端点。"""
    pts = []
    for i, s in enumerate(segs):
        pts.append((i, 0, float(s[0][0]), float(s[0][1])))
        pts.append((i, 1, float(s[1][0]), float(s[1][1])))
    out = []
    for i, _, x, y in pts:
        hit = False
        for j, _, u, v in pts:
            if j == i:
                continue
            if (x - u) ** 2 + (y - v) ** 2 <= tol * tol:
                hit = True
                break
        if not hit:
            out.append((x, y))
    return out


def bind_labels(net_record: dict, segs: Sequence, labels: Sequence[dict],
                tol: float = LABEL_TOL) -> Optional[dict]:
    """给一条网络找它自己的网络标签（取最近的自由端 - 标签框距离）。"""
    ends = _free_ends(segs)
    if not ends:
        return None
    best = None
    for lb in labels:
        b = lb["box"]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        for (x, y) in ends:
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= tol and (best is None or d < best[0]):
                best = (d, lb["text"], float(lb.get("conf") or 0.0))
    if best is None:
        return None
    return {"text": best[1], "conf": best[2], "d": round(best[0], 2)}


def _merge_nets(nets: Dict[str, dict], groups: Dict[str, List[str]]) -> Dict[str, dict]:
    """按并查集分组把网络并起来（`_refs` / `edges` / `pins` 取并集）。"""
    out: Dict[str, dict] = {}
    for i, (root, members) in enumerate(groups.items(), 1):
        members.sort()
        base = dict(nets[root])
        pins, edges = [], {}
        refs: List[list] = []
        seen, seen_ref = set(), set()
        for m in members:
            for p in nets[m].get("pins") or []:
                t = tuple(p)
                if t not in seen:
                    seen.add(t)
                    pins.append(list(p))
            # ⚠️ `_refs` 必须**跨成员取并集**：原来只留了 root 的 `_refs`，
            # 于是「按同名网络标签并起来」的那些 net，除第一条以外成员的
            # (元件, 引脚) 引用全部从 hyperGraph 里消失 —— 而
            # net_hypergraph 的 tp 正是按 `|pred_pins ∩ gt_pins|` 加的，
            # 合并本该让 tp 变多，旧写法反而把它们丢了。
            for r in nets[m].get("_refs") or []:
                t = tuple(r)
                if t not in seen_ref:
                    seen_ref.add(t)
                    refs.append(t)
            for e in (nets[m].get("edges") or {}).values():
                edges["edge_%d" % (len(edges) + 1)] = e
        base["pins"] = pins
        base["_refs"] = refs
        base["edges"] = edges
        out["net_%d" % i] = base
    return out


def merge_by_symbol(nets: Dict[str, dict],
                    syms: Dict[str, set]) -> Dict[str, dict]:
    """**共享同一个电源/地符号的网络合并成一条**（并查集）。

    `syms[net_key]` = 这条网络碰到的电源/地**元件键**集合（`work\\tools\\diag\\_phibox.py`
    证明 GT 也是这么分的：一个 `∅NNN` 只出现在一条网络里）。合并完成后这些符号的
    引脚引用**不写进 hyperGraph**（它们的号是随机内部号，写进去就是 FP）。
    """
    if not SYM_MERGE or not nets:
        return nets
    parent: Dict[str, str] = {k: k for k in nets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_sym: Dict[str, str] = {}
    for k in nets:
        for s in sorted(syms.get(k) or ()):
            if s in by_sym:
                union(by_sym[s], k)
            else:
                by_sym[s] = k
    groups: Dict[str, List[str]] = {}
    for k in nets:
        groups.setdefault(find(k), []).append(k)
    if len(groups) == len(nets):
        return nets
    return _merge_nets(nets, groups)


def merge_by_label(nets: Dict[str, dict]) -> Dict[str, dict]:
    """同名网络标签的网络合并成一条（并查集）。"""
    parent: Dict[str, str] = {k: k for k in nets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_label: Dict[str, str] = {}
    for k, n in nets.items():
        lb = (n.get("label") or {}).get("text")
        if not lb:
            continue
        if lb in by_label:
            union(by_label[lb], k)
        else:
            by_label[lb] = k

    groups: Dict[str, List[str]] = {}
    for k in nets:
        groups.setdefault(find(k), []).append(k)

    out: Dict[str, dict] = {}
    for i, (root, members) in enumerate(groups.items(), 1):
        members.sort()
        base = dict(nets[root])
        pins, edges = [], {}
        refs: List[list] = []
        seen, seen_ref = set(), set()
        for m in members:
            for p in nets[m].get("pins") or []:
                t = tuple(p)
                if t not in seen:
                    seen.add(t)
                    pins.append(list(p))
            # ⚠️ `_refs` 必须**跨成员取并集**：原来只留了 root 的 `_refs`，
            # 于是「按同名网络标签并起来」的那些 net，除第一条以外成员的
            # (元件, 引脚) 引用全部从 hyperGraph 里消失 —— 而
            # net_hypergraph 的 tp 正是按 `|pred_pins ∩ gt_pins|` 加的，
            # 合并本该让 tp 变多，旧写法反而把它们丢了。
            for r in nets[m].get("_refs") or []:
                t = tuple(r)
                if t not in seen_ref:
                    seen_ref.add(t)
                    refs.append(t)
            for e in (nets[m].get("edges") or {}).values():
                edges["edge_%d" % (len(edges) + 1)] = e
        base["pins"] = pins
        base["_refs"] = refs
        base["edges"] = edges
        out["net_%d" % i] = base
    return out


# ---------------------------------------------------------------------------
def _pin_read_quality(rec: dict) -> Optional[float]:
    """Rank complete OCR reads; this product is not a calibrated probability."""
    values = []
    for role in ("number", "name"):
        observed = (rec.get("number_selection_name") if "number_selection_name" in rec and role == "name"
                    else rec.get(role))
        try:
            conf = float((observed or {})["conf"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            return None
        values.append(conf)
    return values[0] * values[1]


COLOR_PIN_SOURCES = frozenset(("red_green_transition", "red_stub_endpoint",
                               "red_green_transition_uncertain"))


def _pin_contact_rank(evidence: dict, point: Sequence[float]) -> Optional[int]:
    """Use confirmed, index-aligned frame evidence; absent/stale records are neutral."""
    if not isinstance(evidence, dict) or evidence.get("frame_confirmed") is not True:
        return None
    observed = evidence.get("point_img")
    if not isinstance(observed, (list, tuple)) or len(observed) != 2:
        return None
    try:
        delta = math.hypot(float(observed[0]) - float(point[0]),
                           float(observed[1]) - float(point[1]))
    except (IndexError, TypeError, ValueError, OverflowError):
        return None
    # pin_evidence serializes point_img to one decimal place. A record from a
    # different contact must not acquire that contact's number or network.
    if not math.isfinite(delta) or delta > .11:
        return None
    trust = evidence.get("trust")
    return ({"rejected": 0, "uncertain": 1, "trusted": 2}.get(trust)
            if isinstance(trust, str) else None)


def _box_number_winners(raw: Sequence[dict]) -> Dict[int | str, int]:
    """Choose one existing contact per read number without joining their nets.

    Check actual output eligibility before confidence ranking. A complete read
    from a measured color or clean grayscale frame contact can replace an
    unnamed first read; a frame-rejected incumbent keeps the prior escape hatch.
    For two complete reads, the frozen selection name remains the ranking
    evidence so later name-only OCR does not silently switch physical contacts.
    Numbers, names, points and nets stay on the selected contact.
    """
    from .pin_identifier import pin_identifier_key
    winners: Dict[int | str, int] = {}
    for i, r in enumerate(raw):
        n = r["num"]
        v = pin_identifier_key(n, allow_alphanumeric=r.get("identifier_validated") is True)
        if v is None:
            continue
        prev = winners.get(v)
        if prev is None:
            winners[v] = i
            continue
        old = raw[prev]
        quality, old_quality = r.get("quality"), old.get("quality")
        selection_name = r.get("selection_name", r["nm"])
        old_selection_name = old.get("selection_name", old["nm"])
        if not BOX_PIN_DUPLICATE_BEST:
            continue
        # A contact deleted by the required-name gate cannot be the winner
        # over a complete, measured contact. Judge the final name, not a frozen
        # selection label that a later reader may have removed or replaced.
        if BOX_PIN_NEED_NAME and not old.get("nm"):
            if (r.get("nm") and quality is not None
                    and (r.get("color_contact") is True or r.get("frame_contact") is True
                         or old.get("name_rejected_by_frame") is True)):
                winners[v] = i
            continue
        if BOX_PIN_NEED_NAME and not r.get("nm"):
            continue
        if not selection_name or quality is None:
            continue
        if not old_selection_name:
            if BOX_PIN_NEED_NAME and (r.get("color_contact") is True
                                     or r.get("frame_contact") is True
                                     or old.get("name_rejected_by_frame") is True):
                winners[v] = i
            continue
        if old_quality is None:
            continue
        contact_rank, old_rank = r.get("contact_rank"), old.get("contact_rank")
        if (BOX_PIN_NEED_NAME and (r.get("color_contact") is True
                                  or old.get("color_contact") is True
                                  or r.get("frame_contact") is True
                                  or old.get("frame_contact") is True)
                and contact_rank is not None and old_rank is not None
                and contact_rank != old_rank):
            if contact_rank > old_rank:
                winners[v] = i
        elif quality > old_quality:
            winners[v] = i
    return winners


def build_doc(stage1: dict, assigned: dict, case_id: str = "") -> dict:
    img = stage1.get("image") or {}
    H = float(img.get("h") or 0.0)
    comps = stage1.get("components") or []
    a = assigned.get("comps") or {}
    st1_pins = stage1.get("pins") or {}
    st1_evidence = stage1.get("pin_evidence") or {}

    components: Dict[str, dict] = {}
    pins: Dict[str, dict] = {}
    keymap: Dict[str, str] = {}
    num_of: Dict[Tuple[str, int], str] = {}
    # ★ 引脚**索引 → 交付文档里的那个键**（`pin_<num>`，撞名时带 `_N` 后缀）。
    # 融合层要改引脚就必须知道这个映射：`num` 会被下面的「结构期望」重写过
    # （`1..N` 不成立时整颗重排成位置号），所以**从 `_assigned` 里的 OCR 原文
    # 反推不出来**（`work\tools\diag\_pinmap.py` 实测：插入序 0.560 / 按点匹配
    # 0.462 / 按 OCR 原文 0.318）。这里把真值带出来，落到旁路文件
    # `<case>.pinindex.json`，交付文档本身保持逐字节不变。
    pin_pk: Dict[Tuple[str, int], str] = {}
    ct_by_cid: Dict[str, str] = {c["id"]: (c.get("type") or "other") for c in comps}
    confirmed_power = {c["id"] for c in comps
                       if _confirmed_power_terminal(stage1, c["id"], c.get("type"),
                                                    st1_pins.get(c["id"]) or [])}

    for i, c in enumerate(comps):
        cid = c["id"]
        info = a.get(cid) or {}
        # `canon` 是前缀矫正后的规范位号（`Rl4` -> `R14`）；没有就用原始读法。
        # 输出键必须用规范位号 —— 官方按 key 匹配元件，一个字符错整条链就丢。
        _d = info.get("designator") or {}
        des = _d.get("canon") or _d.get("text")
        key = des if des else None
        if not key or key in components:
            # 拿不到位号（或位号撞号）—— 兜底键 `pNNN` 在 GT 里不可能存在，
            # 输出它 = 白送一个 component.extra + 它名下所有引脚在 net 超图里
            # 也全是 FP。直接不输出这颗元件（见 `DROP_UNKEYED` 的长注释）。
            if DROP_UNKEYED:
                continue
            key = "p%03d" % i
        keymap[cid] = key
        ctype = c.get("type") or "other"
        bbox = c.get("bbox_json") or c.get("bbox_img") or [0, 0, 0, 0]

        name = key
        if ctype == "box":
            pn = (info.get("partname") or {}).get("text")
            if pn:
                name = pn
        val = None
        if ctype in VALUE_TYPES:
            vv = (info.get("value") or {}).get("text")
            if vv and VALUE_LIKE.search(vv):
                val = vv
        components[key] = {"Name": name, "type": ctype, "value": val,
                           "bbox": [round(float(x), 2) for x in bbox]}

        pts = st1_pins.get(cid) or []
        if not pts:
            continue
        # Keep the established 1/2 numbering, publishing measured colour
        # boundaries when available. Two aligned measured terminals also
        # resolve the axis of a near-square or offset detector box.
        observed = _measured_passive_points(stage1, cid, ctype, pts)
        measured_axis = None
        if len(pts) == len(observed) == 2:
            dx = abs(float(pts[1][0]) - float(pts[0][0]))
            dy = abs(float(pts[1][1]) - float(pts[0][1]))
            if min(dx, dy) <= 2.5 and max(dx, dy) >= 8:
                measured_axis = dy > dx
        _st = stamp_pins(c, ctype, pts, H, axis_y=measured_axis)
        if _st is not None:
            _g: Dict[str, dict] = {}
            for k, (num, xy) in enumerate(_st):
                if k in observed:
                    xy = (float(observed[k][0]), H - float(observed[k][1]))
                _g["pin_%s" % num] = {
                    "pinname": num,
                    "point": {"x": round(float(xy[0]), 2),
                              "y": round(float(xy[1]), 2)}}
                num_of[(cid, k)] = num
                pin_pk[(cid, k)] = "pin_%s" % num
            pins[key] = _g
            continue
        pinfo = info.get("pins") or {}
        group: Dict[str, dict] = {}
        # 先把每个引脚 OCR 出来的号/名读齐，再**整颗元件一起决定**信不信 OCR。
        # ⚠️ 索引键必须两种形态都试：`assigned` 在阶段二里是内存字典（键为 int），
        # 但阶段三是**从 JSON 读回来的**，JSON 一定把字典键变成字符串 —— 只写
        # `pinfo.get(k)` 就会全部落空，于是所有 OCR 出来的引脚号/引脚名被丢掉。
        raw: List[dict] = []
        contact_evidence = st1_evidence.get(cid) or []
        for k in range(len(pts)):
            rec = pinfo.get(k) or pinfo.get(str(k)) or {}
            num = ((rec.get("number") or {}).get("text") or "").strip()
            nm = ((rec.get("name") or {}).get("text") or "").strip()
            selection_name = (rec.get("number_selection_name") if "number_selection_name" in rec
                              else rec.get("name"))
            evidence = (contact_evidence[k] if isinstance(contact_evidence, (list, tuple))
                        and k < len(contact_evidence) else None)
            contact_rank = _pin_contact_rank(evidence, pts[k])
            frame_contact = structmatch.trusted_frame_pin_evidence(
                evidence, c.get("bbox_img"), pts[k])
            raw.append({"num": num, "nm": nm, "quality": _pin_read_quality(rec),
                        "identifier_validated": (rec.get("number_independently_read") is True
                                                 and (frame_contact or (contact_rank == 2
                                                      and isinstance(evidence, dict)
                                                      and evidence.get("source") in COLOR_PIN_SOURCES))),
                        "selection_name": ((selection_name or {}).get("text") or "").strip(),
                        "contact_rank": contact_rank,
                        "color_contact": (contact_rank == 2 and isinstance(evidence, dict)
                                          and isinstance(evidence.get("source"), str)
                                          and evidence.get("source") in COLOR_PIN_SOURCES),
                        "frame_contact": frame_contact,
                        "name_rejected_by_frame": rec.get("name_rejected_by_frame") is True})
        # **结构期望**（2025-10 按官方 Q&A 重定）。
        #
        # 旧口径下这里有一条「OCR 必须交出完整的 1..N 排列才采信」的门控，理由是
        # 220 张实测「纯位置序号 netHF1 0.1217 / 无脑信 OCR 0.1137 / 门控 0.1209」。
        # 但那个测量是在**全部元件**上做的，而官方 Q&A「评测 Q2」明确回答：
        # **「非 box 类型元器件 pin name, pin number 不参与评分」** ——
        # 也就是说旧口径下撑起 pinF1 的那 2800 个 tp（`r`/`c`/`l`/`gnd`/… 的图章
        # 引脚）**一分都不算**，唯一算分的是 `box`(IC)。
        #
        # 而 `box` 恰恰是唯一**不能用 1..N**的一类：GT 里 `0001 U1` 的引脚键实测是
        # `pin_1/pin_2/.../pin_59/pin_61/pin_62/pin_63/pin_64/pin_68/pin_69/
        # pin_391/pin_392/pin_393/...` —— 那是**图上印的真实引脚号**，压根不是排列。
        # 强行按位置编号会让 box 引脚大面积 key_miss（改前实测 box 引脚
        # `key_miss 1391 / num_miss 537 / name_bad 185 / ok 3`，n=2132）。
        #
        # 所以现在**按类型分开**：
        #   * `box`：保留互不相同的 1..999 整数；字母数字编号还须具备
        #     独立双视图读取及当前真实框接点的验证。重复号仍只保留一个物理行。
        #   * 其余：仍要求完整的 1..N 排列 —— 它们的引脚号不进 pinF1，但会进
        #     `net_hyperGraph`，保守的先验在那里仍然更稳。
        nums = [r["num"] for r in raw]
        if ctype == "box":
            from .pin_identifier import pin_identifier_key
            identities = [pin_identifier_key(r["num"], allow_alphanumeric=r["identifier_validated"])
                          for r in raw]
            ok = bool(identities) and all(v is not None for v in identities)
            ok = ok and len(set(identities)) == len(identities)
        else:
            ok = bool(nums) and all(n.isdigit() for n in nums)
            if ok:
                try:
                    ordered = sorted(int(n) for n in nums)
                except ValueError:
                    ok = False
                else:
                    ok = ordered == list(range(1, len(nums) + 1))
        if os.environ.get("STAGE2_DBG_PINS"):
            print("[dbg pins] cid=%s ctype=%s n_pts=%d n_raw=%d ok=%s partial=%s "
                  "drop=%s mixed=%s cap=%s nums=%s"
                  % (cid, ctype, len(pts), len(raw), ok, BOX_PIN_PARTIAL,
                     BOX_PIN_DROP_UNREAD, BOX_PIN_DROP_MIXED_ONLY,
                     BOX_PIN_DROP_CAP, [r["num"] for r in raw][:40]))
        _anykept = True
        if not ok:
            if ctype == "box" and BOX_PIN_PARTIAL:
                # Duplicate OCR numbers may come from different physical pins.
                # Select the best complete read; do not alias the discarded
                # contact into the winner's network.
                used, tmp = set(), list(nums)
                kept = [False] * len(tmp)
                winners = _box_number_winners(raw)
                for i, n in enumerate(tmp):
                    v = pin_identifier_key(n, allow_alphanumeric=raw[i]["identifier_validated"])
                    if v is not None and winners.get(v) == i:
                        used.add(v)
                        kept[i] = True
                    else:
                        tmp[i] = ""
                nxt = 1
                _maxread = max((int(n) for n in nums if n and n.isdigit()
                                and 1 <= int(n) <= 999), default=0)
                for i, n in enumerate(tmp):
                    if n:
                        continue
                    while nxt in used:
                        nxt += 1
                    used.add(nxt)
                    tmp[i] = str(nxt)
                    if BOX_PIN_DROP_CAP and _maxread:
                        kept[i] = (nxt <= _maxread)
                for k in range(len(raw)):
                    raw[k]["num"] = tmp[k]
                    raw[k]["kept"] = kept[k]
                _anykept = any(kept)
            else:
                for k in range(len(raw)):
                    raw[k]["num"] = str(k + 1)
        for k, p in enumerate(pts):
            r = raw[k]
            num = r["num"] or str(k + 1)
            # ★ `STAGE2_BOX_PIN_DROP_UNREAD`：把「没读出号、靠位置补出来的」引脚桩
            # **既不写进引脚表、也不进网络**。它只可能变成 `pin.extra`（纯 fp），
            # 而且会把假引脚塞进网络（10GT 实测 netH 0.4746 → 0.4971）。
            # 默认 `1`/`all` 一律丢，包括全无读数的 IC。`mixed` 是显式 legacy
            # 兼容：至少读懂一个号时才丢，全无读数仍补位置号；默认不采用它。
            if (not r.get("kept", True)) and BOX_PIN_DROP_UNREAD \
                    and (not BOX_PIN_DROP_MIXED_ONLY or _anykept):
                continue
            if ctype == "box" and BOX_PIN_NEED_NAME and not r.get("nm"):
                continue
            nm = r["nm"] or num
            pk = "pin_%s" % num
            base = pk
            while pk in group:
                pk = base + "_%d" % (len(group) + 1)
            group[pk] = {"pinname": nm,
                         "point": {"x": round(float(p[0]), 2),
                                   "y": round(H - float(p[1]), 2)}}
            num_of[(cid, k)] = num
            pin_pk[(cid, k)] = pk
        pins[key] = group

    # ---- 网络 ----
    segs_all = (stage1.get("wire_detail") or {}).get("segments") or []
    labels = assigned.get("net_labels") or []
    raw: Dict[str, dict] = {}
    syms: Dict[str, set] = {}
    for j, n in enumerate(stage1.get("nets") or [], 1):
        segs, edges = [], {}
        seg_pts = []
        for si in (n.get("seg_index") or []):
            if not isinstance(si, int) or not (0 <= si < len(segs_all)):
                continue
            s = segs_all[si]
            segs.append(s)
            pt = [{"x": round(float(p[0]), 2), "y": round(H - float(p[1]), 2)}
                  for p in s]
            if len(pt) >= 2:
                seg_pts.append(pt)
        # ★ 先把共线碎边焊成两点长边，再逐条过长度闸门（见 `merge_collinear` 的长注释）。
        if EDGE_MERGE and len(seg_pts) > 1:
            seg_pts = merge_collinear([[p[0], p[-1]] for p in seg_pts])
        for pt in seg_pts:
            if len(pt) < 2:
                continue
            if MIN_EDGE_LEN > 0:
                L = math.hypot(pt[1]["x"] - pt[0]["x"], pt[1]["y"] - pt[0]["y"])
                if L < MIN_EDGE_LEN:
                    continue
            edges["edge_%d" % (len(edges) + 1)] = pt
        refs, seen = [], set()
        syms_here = set()
        for cid, idx in (n.get("pins") or []):
            k = keymap.get(cid)
            if not k or (cid, idx) not in num_of:
                continue
            if ct_by_cid.get(cid) in POWER_TYPES:
                # ★ 电源/地符号：只当**并查集的连边**，引用本身不写进 hyperGraph
                # （它的引脚号是随机内部号，写进去必然 FP）。详见 `SYM_MERGE` 的长注释。
                if SYM_MERGE:
                    syms_here.add(k)
                    if cid not in confirmed_power:
                        continue
            if NET_DROP_UNKEYED and FKEY_RE.match(k) and cid not in confirmed_power:
                continue
            t = (k, num_of[(cid, idx)])
            if t not in seen:
                seen.add(t)
                refs.append(t)
        if syms_here:
            syms["net_%d" % j] = syms_here
        rec = {"hyperGraph": "(" + ",".join("%s.%s" % t for t in refs) + ")",
               "edges": edges, "_refs": refs, "_segs": segs}
        lb = bind_labels(n, segs, labels)
        if lb:
            rec["label"] = lb
        raw["net_%d" % j] = rec

    nets = merge_by_label(merge_by_symbol(raw, syms))
    out_nets: Dict[str, dict] = {}
    for v in nets.values():
        v.pop("_segs", None)
        refs = [r for r in v.pop("_refs", []) if not (len(r) == 2 and r[1] is None)]
        # 一个引脚都没连上的网络：既拿不到 tp，又可能在匈牙利匹配里把 GT 网络
        # 「抢走」，官方口径下不输出更干净（实测 net_hypergraph P 0.1273→0.3065）。
        #
        # 2026-07 审计提出「改成 `if not refs and not v.get("edges")`，别连 edges 一起丢」，
        # **实测否证、已回退**（10GT 无 VLM：48.35 → 46.88，netH tp −14/fp +14/fn +14、
        # net_line fp +124）。原因：审计只算了评测端（`match_nets` 对空 pin 集贡献 0，
        # 这一半是对的），**漏了后面两步** —— `attach_orphan_rails`（:720）与
        # `merge_shared_pins`（:721）都在这个丢弃循环**之后**跑，被保留下来的「有边无引用」
        # 网络会去抢轨引脚，把本该挂到正确网上的轨引脚抢走 ⇒ netH 反而掉。
        # 结论：这些网确实该丢；要救它们的边只能另想办法（顺序、或只救边不救网）。
        if not refs:
            continue
        v["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
        out_nets["net_%d" % (len(out_nets) + 1)] = v
    out_nets = attach_orphan_rails(out_nets, pins)
    out_nets = merge_shared_pins(out_nets)
    _pidx: Dict[str, Dict[str, str]] = {}
    for (_c, _k), _pk in pin_pk.items():
        _pidx.setdefault(_c, {})[str(_k)] = _pk
    return {"case": case_id, "components": components, "pins": pins,
            "nets": out_nets, "__pin_index": _pidx}


RAIL_NAMES = frozenset((
    "gnd", "gnda", "gndd", "agnd", "dgnd", "pgnd", "vss", "vssa",
    "vcc", "vdd", "vddd", "vdda", "vddio", "3v3", "5v", "12v",
    "vbat", "vbus", "vin", "vout"))
RAIL_ATTACH = env_on("STAGE2_RAIL_ATTACH", "1")


def attach_orphan_rails(nets: Dict[str, dict], pins: Dict[str, dict]) -> Dict[str, dict]:
    """把「有引脚、但没有任何网络引用它」的**电源/地轨**引脚，挂到同名的最大网络上去。

    依据是用户给的电路常识：**引脚名叫 GND，那它必然接在接地网络上**。
    但这条常识**不能**用来「合并网络」—— 离线实测把共享同名轨引脚的网络全部并起来是
    `10GT` 总分 `50.11 → 49.34 (−0.78)`、`netH 0.4064 → 0.3839`、`tp 148 → 139`
    （同名轨道被强行并到一起后，匈牙利把它们配到了错的 GT 网）。
    所以这里只动**本来就悬空**的引脚（`work\\tools\\diag\\_netwhy.py` 里的
    `C_有脚但不在网里`，10GT 上 50 个 / 0.119）。

    离线实测（`work\\tools\\diag\\_netrail.py --arm orphan`）：
    `10GT` 9 张 总分 `50.11 → 50.78 (+0.67)`、`netH 0.4064 → 0.4254`、
    `tp 148 → 162 / fp 184 → 192 / fn 271 → 257`；
    快测集 19 张 `29.76 → 30.06 (+0.31)`、`netH 0.2167 → 0.2255`、`tp 205 → 219`。
    """
    if not RAIL_ATTACH or not nets or not pins:
        return nets
    in_nets = set()
    for rec in nets.values():
        in_nets |= set(_hg_refs(rec.get("hyperGraph", "")))
    # 每个轨名 -> 已经引用它的那个**最大**网络
    per_name: Dict[str, tuple] = {}
    for name, rec in nets.items():
        refs = _hg_refs(rec.get("hyperGraph", ""))
        for nm in {_pinname(pins, c, p) for c, p in refs} & RAIL_NAMES:
            if nm not in per_name or len(refs) > per_name[nm][1]:
                per_name[nm] = (name, len(refs))
    added = 0
    for comp, group in (pins or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, pr in group.items():
            num = str(pk).replace("pin_", "")
            if (comp, num) in in_nets:
                continue
            nm = str((pr or {}).get("pinname") or "").strip().lower()
            hit = per_name.get(nm)
            if not hit:
                continue
            rec = nets[hit[0]]
            refs = _hg_refs(rec.get("hyperGraph", ""))
            refs.append((comp, num))
            rec["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
            in_nets.add((comp, num))
            added += 1
    if added:
        try:
            print("[assemble] 轨引脚归位：把 %d 个悬空的电源/地引脚挂回了同名的网络"
                  % added)
        except Exception:
            pass
    return nets


def _pinname(pins: Dict[str, dict], comp: str, num: str) -> str:
    return str((((pins or {}).get(comp) or {}).get("pin_%s" % num) or {})
               .get("pinname") or "").strip().lower()


def _hg_refs(hg: str):
    """把 `"(U1.45,U2.112)"` 拆成 `[("U1","45"),("U2","112")]`。
    按**最后一个点**切，这样位号里带点的（KiCad 的 `U1.1`）也不会被切错。"""
    out, seen = [], set()
    for tok in str(hg or "").strip().strip("()").split(","):
        tok = tok.strip()
        if "." not in tok:
            continue
        comp, _, pinnum = tok.rpartition(".")
        ref = (comp.strip(), pinnum.strip())
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def merge_shared_pins(nets: Dict[str, dict]) -> Dict[str, dict]:
    """把**共享同一个 (元件, 引脚)** 的网络做传递闭包合并。

    「一个引脚只能属于一个网络」是电路的基本事实，而现在的装配没有做全局闭包：
    实测 case `1` 的 `C15.2` 同时出现在 `net_1` / `net_3` / `net_4` 里，这种重复归属
    在官方 `net_hypergraph` 口径下全是 fp（`fp = total_pred_pins - tp`）。

    离线实测（`work\\tools\\diag\\_netmerge.py`，快测集 19 张，不动管线只改预测文档）：
    `netH 0.2015 → 0.2115`、总分 `+0.34`，最猛的是 case `10`（`53 → 33 net`，`+2.10`）。
    零风险纯后处理（不新增任何引脚，只把碎片并起来）。
    """
    names = list(nets)
    if len(names) < 2:
        return nets
    parent = {k: k for k in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: Dict[object, str] = {}
    for n in names:
        for ref in _hg_refs(nets[n].get("hyperGraph", "")):
            key = (ref[0], ref[1])
            if key in owner:
                ra, rb = find(owner[key]), find(n)
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[key] = n

    groups: Dict[str, list] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    if len(groups) == len(names):
        return nets

    out: Dict[str, dict] = {}
    for members in groups.values():
        rec = dict(nets[members[0]])
        edges: Dict[str, list] = {}
        refs = []
        seen = set()
        for n in members:
            for ref in _hg_refs(nets[n].get("hyperGraph", "")):
                key = (ref[0], ref[1])
                if key not in seen:
                    seen.add(key)
                    refs.append(ref)
            for ev in (nets[n].get("edges") or {}).values():
                edges["edge_%d" % (len(edges) + 1)] = ev
        rec["edges"] = edges
        rec["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
        out["net_%d" % (len(out) + 1)] = rec
    return out


# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    st1 = {
        "image": {"w": 200, "h": 100},
        "components": [
            {"id": "c0", "type": "r", "bbox_img": [10, 70, 30, 80],
             "bbox_json": [10, 20, 30, 30]},
            {"id": "c1", "type": "box", "bbox_img": [100, 40, 160, 90],
             "bbox_json": [100, 10, 160, 60]},
        ],
        "pins": {"c0": [[8, 75], [32, 75]], "c1": [[98, 50], [98, 80]]},
        "wire_detail": {"segments": [
            [[8, 75], [98, 75]], [[98, 75], [98, 50]], [[150, 10], [150, 100]],
        ]},
        "nets": [
            {"id": "net_1", "pins": [["c0", 0], ["c1", 0]], "seg_index": [0, 1]},
            {"id": "net_2", "pins": [["c0", 1]], "seg_index": [2]},
        ],
    }
    assigned = {"comps": {
        "c0": {"designator": {"text": "R1"}, "value": {"text": "10K"}, "pins": {
            0: {"number": {"text": "1"}, "name": None},
            1: {"number": {"text": "2"}, "name": None}}},
        "c1": {"designator": {"text": "U1"}, "partname": {"text": "MSP432E4"},
               "pins": {0: {"number": {"text": "1"}, "name": {"text": "VDD"}},
                        1: {"number": None, "name": None}}},
    }, "net_labels": [{"text": "VBUS", "conf": 0.9, "box": [152, 8, 170, 20]}]}

    doc = build_doc(st1, assigned, "t")
    chk(set(doc["components"]) == {"R1", "U1"}, "元件键: %r" % (list(doc["components"]),))
    chk(doc["components"]["R1"]["value"] == "10K", "R1 参数")
    chk(doc["components"]["U1"]["Name"] == "MSP432E4", "U1 器件名")
    chk(doc["components"]["U1"]["type"] == "box", "U1 类型")
    # y 翻转：图像 y=75 -> json y=25
    chk(doc["pins"]["R1"]["pin_1"]["point"]["y"] == 25.0, "y 翻转: %r"
        % (doc["pins"]["R1"]["pin_1"]["point"],))
    chk(doc["pins"]["U1"]["pin_1"]["pinname"] == "VDD", "引脚名")
    # Default assembly deliberately drops this unread IC pin. The historical
    # self-test assumed the old number/name fallback and failed even before
    # the monochrome changes. Test the selected output policy explicitly.
    if BOX_PIN_NEED_NAME or BOX_PIN_DROP_UNREAD:
        chk("pin_2" not in doc["pins"]["U1"], "不输出未读出名号的芯片脚")
    else:
        chk(doc["pins"]["U1"].get("pin_2", {}).get("pinname") == "2", "缺名回退引脚号")
    n1 = doc["nets"]["net_1"]
    chk(set(n1["hyperGraph"].strip("()").split(",")) == {"R1.1", "U1.1"},
        "hyperGraph: %r" % (n1["hyperGraph"],))
    chk(len(n1["edges"]) == 2, "net_1 边数: %d" % len(n1["edges"]))
    chk(n1["edges"]["edge_1"][0] == {"x": 8.0, "y": 25.0}, "边坐标: %r"
        % (n1["edges"]["edge_1"][0],))

    # 网络标签合并：两条 net 贴同名标签 -> 并成一条
    merged = merge_by_label({
        "net_1": {"pins": [["R1", "1"]], "edges": {"edge_1": []},
                  "label": {"text": "VBUS"}},
        "net_2": {"pins": [["U1", "1"]], "edges": {"edge_1": []},
                  "label": {"text": "VBUS"}},
        "net_3": {"pins": [["R1", "2"]], "edges": {}, "label": None},
    })
    chk(len(merged) == 2, "同名标签应并成 2 条: %r" % (list(merged),))
    big = [v for v in merged.values() if len(v["pins"]) == 2]
    chk(len(big) == 1, "合并后应有一条双引脚网络")

    print("自检 %d 项：%s" % (12, "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
