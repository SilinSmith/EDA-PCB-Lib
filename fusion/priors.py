# -*- coding: utf-8 -*-
"""常识与先验：**不读图**就能说出来的东西。

它回答四类问题，并且**会自己造候选**（这是它与「给现有候选打分」的本质区别）：

1. **位号连号** —— 同一张图上同一个前缀的位号，几乎总是 1..N 的**连续区间**。
   `case 1` 实测：`C151..C156` 六个元件，CRNN 给每个框都读成 `C51/C15/C5/C54/C5/C15`
   （**6 个框里一个都没读对**，`work\preview\_c1desig_crops.png` 里字形清晰无缺笔，
   是模型自己被垂直外扩压小了字形）。这时「连号」这个先验就直接说出
   **答案集合是 {C151..C156}**，剩下的只是「哪个框配哪个号」——
   这是纯文本侧的 CRNN 永远给不出的信息，也是 `stage2` 的逐元件贪心**装不下**的
   全局约束（它只知道「不许撞号」，不知道「不许断号」）。

2. **引脚号连号** —— 一颗 IC 的引脚号是 1..N 的连续区间（`structmatch.align_monotone`
   已经会用「沿列单调」，但没有用「不许断号」）。

3. **词表** —— 网络名（电源轨 / 地族）、常用器件值（E 系列 / 电容单位）、
   常见引脚名（`VDD`/`SCLK`/`MOSI`…）。用来判断一个读法**像不像**该槽位该有的东西。

4. **类型约束** —— 位号前缀由元件类型决定（`stage2.assign.TYPE_PREFIX`）；
   哪些类型没有位号可言（`NO_DESIGNATOR`）。

用法
----
    from fusion import priors
    P = priors.Priors.load()
    P.annotate(pool)          # 往池子里补 prior 候选（含自己造的）
    bad = P.check(pool)       # 列出违反常识的地方，给 plausible / conflict 用
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- 复用 stage2 的表
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:                                                    # 前缀表与位号规范化
    from stage2.assign import (TYPE_PREFIX, NO_DESIGNATOR,   # noqa: F401
                              canon_designator, is_netname)
except Exception:                                        # noqa: BLE001
    TYPE_PREFIX = {
        "c": ("C",), "r": ("R",), "box": ("U", "USB", "u"), "d": ("D",),
        "led": ("D", "LED"), "pin": ("TP",), "switch": ("SW", "S"), "l": ("L",),
        "mosfet": ("Q",), "bjt": ("Q",), "crystal_2pin": ("X", "Y", "C"),
        "crystal_3pin": ("X", "Y"), "crystal_4pin": ("X", "Y"), "crystal": ("X",),
        "testpoint": ("TP",), "fuse": ("F",), "amp": ("U",), "amp_3pin": ("U",),
        "amp_5pin": ("U",), "bjt_npn": ("Q",), "bjt_pnp": ("Q",),
        "mosfet_npn": ("Q",), "mosfet_pnp": ("Q",), "v": ("TP",),
    }
    NO_DESIGNATOR = frozenset({"gnd", "v", "net_input", "net_short", "net_output",
                               "other", "net_bidirection", "m3\u87ba\u4e1d"})

    def canon_designator(text, ctype):                    # 退化实现
        u = re.sub(r"[^0-9A-Za-z]", "", text or "").upper()
        m = re.match(r"^([A-Z]{1,3})(\d{1,4})$", u)
        if not m:
            return "", 0.0
        pre = TYPE_PREFIX.get(ctype)
        if pre and m.group(1) not in pre:
            return "", 0.0
        return (pre[0] if pre else m.group(1)) + m.group(2), 1.0

    def is_netname(s):                                    # 退化实现
        s = (s or "").strip()
        if len(s) < 2 or s.isdigit():
            return 0.0
        return 1.0 if re.search(r"[A-Za-z]", s) else 0.0

# ---------------------------------------------------------------- 正则
DESIG_RE = re.compile(r"^([A-Za-z]{1,3})(\d{1,4})$")
PINNUM_RE = re.compile(r"^\d{1,3}[A-Za-z]?$")             # 1 / 12 / 8B
PINNUM2_RE = re.compile(r"^[A-Za-z]{1,2}\d{1,3}$")        # BGA: A1 / AB12
PINNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+\-/]{0,15}$")
RAIL_RE = re.compile(r"^[+-]?\d+(\.\d+)?V(\d+)?$", re.I)  # 3.3V / 5V / 1V8 / +12V
NETWORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+\-/*]{1,23}$")
VAL_RE = re.compile(r"^(\d+(\.\d+)?)([pnu\u00b5mkKMGRT]?)(F|H|OHM|\u03a9|R)?$",
                    re.I)

# ---------------------------------------------------------------- 读数形状闸
# ★ 「**原样读出来的串**像不像一个位号」—— 必须在**原样文本**上判，不能在
#   `canon_designator` 的输出上判。差别是决定性的：`stage2\assign.py:243-251`
#   的 `_FIX_DIGIT`/`_digitize` 会把尾部里的形近字母**逐个换成数字**
#   （`S`->`5`、`B`->`8`、`D`->`0`、`L`->`1`、`O`->`0`…），于是一个**网络名**能被
#   规范成**一个看起来完全合法的位号**：
#
#       `USB_D+`  --去掉 `_`/`+`-->  `USBD`
#                 --前缀 `U` 对上 box 类 + `_digitize("SBD")` = `580`-->
#                 **`U580`**（置信度 0.9，合法形状，谁都拦不住）
#
#   实测（`work\tools\diag\_keyorigin.py` / `_u1probe.py`，220 张默认链
#   `work\_submit_out220r36`，`case 0001`）：
#
#   * 最终文档里的键是 **`U580`**，而它的 `_assigned` 位号原样文本是 `US8_D+`；
#   * 该元件的 `desig_slots` 是
#         `US8_D+ (score  0.825, conf 0.584)`
#         **`U1  (score -0.101, conf 0.983)`**   ← 正确读数就在这里，置信度高得多
#         `USB_D+ (-1.403, conf 0.791)` / `USB9D (-1.616, conf 0.655)`
#     —— 只有**几何分**让一个不可能的形状赢了；
#   * 这颗元件的 GT 键是 **`U1`，72 个引脚**。
#
#   为什么这件事值钱：220 张上**「整颗 GT IC 没配上」贡献了 63% 的引脚 fn**
#   （`work\tools\diag\_pinzero.py`：5210/8273），其中 **88% 是「GT 键是真位号、
#   我们没配上」**（`_icmisswhy.py`：579 颗 / 4587 引脚）。这批垃圾键正是主因。
#
#   ⇒ 见 `desig_shape_ok` 与 `claims.from_stage2` 的位号候选循环。
DESIG_RAW_BAD = re.compile(r"[^0-9A-Za-z]")
DESIG_RAW_LET = re.compile(r"[A-Za-z]")
# 位号原样文本的最大长度：`字母1~3 + 数字1~4 + 可选字母` = 8。
DESIG_RAW_MAX = 8
# 形状闸开关（默认开）。关掉 = 回到「`canon_designator` 说什么就是什么」。
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★ 手写短词表 ⇒ `FUSION_SHAPE_GATE=off` 被当成开。统一。
SHAPE_GATE = env_on("FUSION_SHAPE_GATE", "1")
# 形状闸命中时给的分（默认直接**丢掉候选**；置成数字则改成**压分**保留）。
SHAPE_SCORE = os.environ.get("FUSION_SHAPE_SCORE", "").strip()
# ★★ `FUSION_BOX_PIN_PRIOR`（默认 **0**）：**不给 IC(`box`) 补位置号 `1..N`**。
#
# 依据（官方 10GT 实测）：IC 的引脚号是图上**印出来的封装号**，天然稀疏 ——
# case `1` 的 `U1`(`MSP432E401YTPDT`) GT 就是
# `pin_7/8/9/10/16/17/26/28/39/47/48/51/52/58/101/113/115/122…`（26 个）。
# 位置先验把「读到的 15 个稀疏号 + 19 个空位」补成**完整的 `1..34`**，于是
# `stage2/assemble.py` 的 `ok` 闸门（要求「全是数字且互不相同」）**由 False 变 True**，
# 整颗 IC 的号从「稀疏但对」变成「稠密且全错」：
# 官方 10GT(9) 的 pin tp **49 → 3**、pinF1 **0.4693 → 0.0350**、总分 58.15 → 44.16。
# 非 box 的引脚号不进 `pin_f1`（官方评测 Q2），位置先验在那里仍然有用，故只跳过 IC。
BOX_PIN_PRIOR = env_on("FUSION_BOX_PIN_PRIOR", "0")
_PRIOR_IC_TYPES = ("box", "ic", "boxic", "bigic")

# ★★ `FUSION_SHAPE_W`（默认 **0.0 = 关**）：**让形状分真的压分**。
#
# 原来第 3 节「形状惩罚」只把 `f` 写进 `a.evidence["形状分"]` 并给 `st["shape_pen"]`
# 加一 —— **分数一动不动**，是个静默的 no-op：`pin_number_shape("A")=0.1` 与
# `pin_number_shape("7")=1.0` 在 `best()` 眼里一模一样。`value_shape` / `key_fits_type`
# 里那几十行判据相当于只写给人看。
#
# 语义按 owner 的两条原则定：**判据不二值、信息只降权不销毁**。
#   * 不是「f 低就删」—— 候选永远还在池子里，`overflow`/`history` 也都留着；
#   * 只是**乘一个系数**：`k = FLOOR + (1-FLOOR)·f`，`f=1 → k=1`（原样），
#     `f=0 → k=FLOOR=0.35`（明显降权，仍能被别的证据重新扶起来）。
#   * `SHAPE_W` 是「这个降权管多大用」的插值：`0` = 老行为（只标注不压分），
#     `1` = 完全生效。
#
# ★★ **实测判负，默认取 0**（口径：`--no-vlm` + `FUSION_LLM=0`，
#    `work\tools\eval\fast.py`；归档 `work\_fast_base\{sw0,sw35,sw100,sw0_19,sw100_19}.json`）：
#    * 官方 10GT 9 张：`W=0` / `W=0.35` / `W=1.0` **三者逐位相同**
#      （51.06，comp 0.7682 / pin 0.3769 / netH 0.4840 / netL 0.1648）⇒ 对 10GT 毫无影响；
#    * 19 张档：`W=1.0` **30.16**（pin 0.1841 / netH 0.2539）vs `W=0` **30.73**
#      （pin 0.1974 / netH 0.2599 / netL 0.0862），tp 引脚 36→42、网络 204→211
#      ⇒ **压分反而掉 0.57**（引脚号与网络成员一起掉，说明形状分低的候选
#      里有用的是「分数低但对」的那些 —— 降权把它们推下去了）。
#    * ⇒ 机制保留（`SHAPE_W>0` 可复现上面每一臂），**默认关**。
#
# ⚠️ 压分之后**必须重排 `s.alts`**：`Slot.best()` 是 `alts[0]`，而列表是按分降序维护的
# （`hypo.py` 的 `Slot.add` 就踩过同一个坑：抬了分不重排 ⇒ `best()` 返回分更低那个）。
SHAPE_W = float(os.environ.get("FUSION_SHAPE_W", "0.0"))
SHAPE_FLOOR = float(os.environ.get("FUSION_SHAPE_FLOOR", "0.35"))


def desig_shape_ok(raw: Any) -> bool:
    """**原样读出的串** `raw` 有没有可能是个位号？

    判据刻意保守（只砍**没有争议**的那一类）：位号在图纸上**只有字母数字**
    （口径同 `stage2\\expect.py:39` 的 `_DESIG_RE`），所以

        * 含 `_` `+` `-` `/` 空格等**非字母数字**字符  ->  False
          （`USB_D+` / `USB_D+` / `L4/4` / `C1 5` / `D+`：这是**网络名/型号片段**）
        * 长度超过 8（`字母≤3 + 数字≤4 + 字母≤1`）      ->  False
        * 一个字母都没有（纯数字，是引脚号不是位号）    ->  False

    保留 `Rl` / `CL` 这种「数字位被读成形近字母」的串 —— `canon_designator`
    的 `_digitize` 正是为它们写的（实测把位号识别天花板从 0.500 抬到 0.725），
    这里**不能**把它们也砍掉。
    """
    s = str(raw if raw is not None else "").strip()
    if not s or len(s) > DESIG_RAW_MAX:
        return False
    if DESIG_RAW_BAD.search(s):
        return False
    return bool(DESIG_RAW_LET.search(s))

# 地族：纸上画的名字不同（AGND/PGND/DGND…），官方**都要求输出 gnd**，
# 但它们在图上是**本质不同的模式**（模拟地/功率地/数字地走线习惯完全不同），
# 所以先验必须分开认、分开连，只在最后装配时折叠回 `gnd`。
GROUND_WORDS = ("GND", "AGND", "PGND", "DGND", "EGND", "SGND", "EARTH", "GNDA",
                "GNDD", "GNDP", "VSS", "VSSA", "VSSD", "0V", "GROUND")
POWER_WORDS = ("VCC", "VDD", "VBAT", "VIN", "VOUT", "VBUS", "VUSB", "VAA",
               "AVDD", "DVDD", "VREF", "VBG", "VDDIO", "VDDCORE", "VCORE",
               "VSYS", "VPWR", "VBK", "VPP", "VBUS")

# 常见引脚名（IC 侧）。**只用来判断「像不像」，不用来替换** ——
# 替换只能由 llm_judge（看图）或 solve（结构约束）做。
PIN_NAME_VOCAB = (
    "VDD", "VSS", "VCC", "GND", "AVDD", "DVDD", "VBAT", "VREF", "VDDIO",
    "SCL", "SCLK", "SDA", "SDI", "SDO", "MOSI", "MISO", "CS", "SS", "SCK",
    "CLK", "RST", "RESET", "NRST", "EN", "ENABLE", "INT", "IRQ", "WAKE",
    "TX", "RX", "TXD", "RXD", "DTR", "RTS", "CTS", "D+", "D-", "USB_DP", "USB_DM",
    "SWDIO", "SWCLK", "SWO", "TCK", "TMS", "TDI", "TDO", "JTAG",
    "ADC", "DAC", "AIN", "AOUT", "PWM", "COMP", "OPA", "GPIO", "IO", "PIO",
    "BOOT", "MODE", "SEL", "SET", "CLR", "OE", "WR", "RD", "ALE", "CE", "WE",
    "XTAL", "XTAL1", "XTAL2", "OSC", "OSCIN", "OSCOUT", "XIN", "XOUT",
    "V+", "V-", "OUT", "IN", "FB", "COMP1", "COMP2", "ILIM", "SS1", "EP", "PAD",
    "TEST", "TDI0", "NC", "DNC", "AGND", "PGND", "DGND", "VSSA", "VSSD",
)
PIN_NAME_SET = frozenset(PIN_NAME_VOCAB)
# 引脚名的最短合法长度：官方图里 1~2 字符的名字极少（`EN`/`CS`/`TX`/`RX` 是例外）
PIN_NAME_MIN = 2

# 位号数字位数的经验区间（官方 9609 个元件统计：绝大多数是 1~3 位）
DESIG_DIGITS = (1, 4)
# 连号判定：一个前缀至少要有这么多**互相靠得近**的元件，才认为它成「串」
RUN_MIN_HITS = 3
RUN_MAX_GAP = 6          # 串内允许的最大空洞（超过就认为不是同一串）
# 修复变体（插回被吞掉的 `1`）的分数折扣：只有在没别的候选能解释连号时才该赢
REPAIR_PENALTY = float(os.environ.get("FUSION_REPAIR_PENALTY", "0.85"))
# 连号窗口返回前几名（并列时交给 llm_judge 看图二选一，不硬选）
TOP_WINDOWS = int(os.environ.get("FUSION_TOP_WINDOWS", "3"))


def _f(v, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _is_read(src) -> bool:
    """这个候选是不是**真的读出来的**（相对于 `prior:`/`solve:` 造出来的）。

    与 `fusion/solve.py:_is_read`、`fusion/doc.py:_read_src` 同一判据。
    """
    return str(src or "").startswith(("ocr", "vlm"))


def _cap(pool, sid: str, want: str, sc: float) -> float:
    """先验给分：**不得盖过该槽位已有的真读数**。

    ★ 这条是实测逼出来的：`prior:run_inwindow` 原来用 `best + 0.02`，
    于是把已经读对的位号抢走了 —— `0001` 的 `C1→C11`、`0004` 的
    `C1→C11` / `R6→R16` / `R5→R15` 全是这么来的。原理与 `no_read`
    同一条：**读数是证据，连号只是形状** —— 形状可以帮读不出号的槽位
    补号，不能把已经读出来的号改掉。
    """
    b = pool.best(sid)
    if b is None:
        return sc
    if _is_read(getattr(b, "source", "")) and str(b.value) != str(want):
        return min(sc, _f(b.score) - 0.01)
    return sc


def split_desig(s: str) -> Tuple[str, Optional[int]]:
    """`C151` -> ("C", 151)；不成位号形返回 (规范化文本, None)。"""
    u = re.sub(r"[^0-9A-Za-z]", "", s or "").upper()
    m = DESIG_RE.match(u)
    if not m:
        return u, None
    return m.group(1), int(m.group(2))


# ---------------------------------------------------------------- 读数模型
# 「识别器读到 obs、真值是 cand」的概率量级。**不是精确概率**，只用它们的
# 相对大小排序假设。刻意把「吞掉一位」定得比「替换一位」高一个量级：CRNN 的
# 系统性失败模式是**漏字**（窄 `1` 被吞、框偏导致尾字出框），而把 `1` 读成 `5`
# 这种**替换**在训练好的模型上很少见。`case 1` 的判别全靠这条：真值 `C151..C156`
# 时六条读数都是「漏字」，真值 `C51..C56` 时有两条必须靠替换。
READ_EXACT = float(os.environ.get("FUSION_READ_EXACT", "1.00"))
READ_DROP_ONE = float(os.environ.get("FUSION_READ_DROP1", "0.40"))
READ_DROP_ONE_OTHER = float(os.environ.get("FUSION_READ_DROP1O", "0.12"))
READ_DROP_MANY = float(os.environ.get("FUSION_READ_DROPM", "0.12"))
READ_INSERT = float(os.environ.get("FUSION_READ_INS", "0.30"))
READ_SUB = float(os.environ.get("FUSION_READ_SUB", "0.05"))
READ_FAR = float(os.environ.get("FUSION_READ_FAR", "0.01"))


def _subseq_drops(obs: str, cand: str) -> Optional[List[str]]:
    """`obs` 是不是 `cand` 删掉若干字符后得到的？返回被删的字符，否则 None。"""
    i, dropped = 0, []
    for ch in cand:
        if i < len(obs) and obs[i] == ch:
            i += 1
        else:
            dropped.append(ch)
    return dropped if i == len(obs) else None


def read_score(obs: str, cand: str) -> float:
    """识别器读到 `obs`、真值是 `cand` 的似然（量级，不是精确概率）。"""
    a = re.sub(r"[^0-9A-Za-z]", "", obs or "").upper()
    b = re.sub(r"[^0-9A-Za-z]", "", cand or "").upper()
    if not a or not b:
        return READ_FAR
    if a == b:
        return READ_EXACT
    drops = _subseq_drops(a, b)
    if drops is not None and 1 <= len(drops) <= 4:
        if len(drops) == 1:
            return READ_DROP_ONE if drops[0] == "1" else READ_DROP_ONE_OTHER
        return READ_DROP_MANY * (0.85 ** (len(drops) - 2))
    ins = _subseq_drops(b, a)
    if ins is not None and 1 <= len(ins) <= 2:
        return READ_INSERT * (0.6 ** (len(ins) - 1))
    # 替换：允许 1~2 个位置不同（长度相同）
    if len(a) == len(b):
        d = sum(1 for x, y in zip(a, b) if x != y)
        if d == 1:
            return READ_SUB
        if d == 2:
            return READ_SUB * 0.25
    return READ_FAR


def fit_numbers(observations: Sequence[Tuple[str, str, float]],
                numbers: Sequence[int], prefix: str) -> List[Tuple[str, str, float]]:
    """把 `numbers` 一对一派给 `observations`（贪心，按 `read_score` 最优）。

    `observations` 是 `[(slot_id, 读数, 原分)]`。返回 `[(slot_id, 派到的位号, 似然)]`，
    没派到的槽位不出现。这就是「窗里没人认领的号该给谁」的**读数模型**答案 ——
    原来按原分排序是错的：`C5` 比 `C15` 更该配 `C155`（少一个字符）而不是 `C152`。
    """
    pairs = []
    for sid, obs, sc in observations:
        for n in numbers:
            cand = "%s%d" % (prefix, n)
            pairs.append((read_score(obs, cand), sid, cand))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_s, used_n, out = set(), set(), []
    for sc, sid, cand in pairs:
        if sid in used_s or cand in used_n:
            continue
        used_s.add(sid)
        used_n.add(cand)
        out.append((sid, cand, sc))
    return out


def is_ground(s: str) -> bool:
    u = re.sub(r"[^0-9A-Za-z]", "", (s or "")).upper()
    if not u:
        return False
    if u in GROUND_WORDS:
        return True
    return any(w in u for w in ("GND",)) or u.startswith("VSS")


def is_power(s: str) -> bool:
    u = re.sub(r"[^0-9A-Za-z]", "", (s or "")).upper()
    if not u or is_ground(u):
        return False
    if u in POWER_WORDS:
        return True
    return bool(RAIL_RE.match((s or "").strip()))


def net_class(s: str) -> str:
    """网络名分类：`ground` / `power` / `signal` / `empty`。"""
    s = (s or "").strip()
    if not s:
        return "empty"
    if is_ground(s):
        return "ground"
    if is_power(s):
        return "power"
    return "signal"


def value_shape(s: str, ctype: str = "") -> float:
    """一个参数值「像不像」该类型元件的值。0=完全不像，1=标准形。"""
    v = (s or "").strip()
    if not v:
        return 0.0
    up = v.upper().replace("\u03a9", "OHM").replace("\u2126", "OHM")
    ct = (ctype or "").lower()
    if ct in ("r",):
        if re.match(r"^\d+(\.\d+)?([MKGR])?(OHM|R)?$", up):
            return 1.0
        if re.match(r"^R?\d+$", up):
            return 0.8
        return 0.1
    if ct in ("c",):
        if re.match(r"^\d+(\.\d+)?(P|N|U|M|F)?F$", up):
            return 1.0
        if re.match(r"^\d+(\.\d+)?$", up):
            return 0.7                       # 画图上常常省略单位
        return 0.1
    if ct in ("l",):
        if re.match(r"^\d+(\.\d+)?(N|U|M|H)?H$", up):
            return 1.0
        return 0.2
    if ct in ("fuse",):
        return 1.0 if re.match(r"^\d+(\.\d+)?(A|MA)?$", up) else 0.3
    # 其余（box 的器件名等）：字母+数字混合才像型号
    if re.match(r"^[A-Za-z][A-Za-z0-9\-/]{3,}$", v):
        return 0.9
    return 0.3


def pin_number_shape(s: str) -> float:
    v = (s or "").strip()
    if not v:
        return 0.0
    if v.isdigit() and 1 <= int(v) <= 999:
        return 1.0
    if PINNUM_RE.match(v) or PINNUM2_RE.match(v):
        return 0.7
    return 0.1


def pin_name_shape(s: str) -> float:
    v = (s or "").strip()
    if len(v) < PIN_NAME_MIN:
        return 0.0
    if not PINNAME_RE.match(v):
        return 0.1
    if v.upper() in PIN_NAME_SET:
        return 1.0
    if is_ground(v) or is_power(v):
        return 1.0
    return 0.6


def label_shape(s: str) -> float:
    """网络标签「像不像」一个网络名。"""
    v = (s or "").strip()
    if not v:
        return 0.0
    if is_ground(v):
        return 1.0
    if is_power(v):
        return 1.0
    if NETWORD_RE.match(v) and len(v) >= 3:
        return 0.8
    if NETWORD_RE.match(v):
        return 0.5
    return 0.1


def key_fits_type(key: str, ctype: str) -> float:
    """位号前缀与该元件类型是否相容。"""
    pre = TYPE_PREFIX.get((ctype or "").lower())
    if not pre:
        return 0.6 if DESIG_RE.match(key or "") else 0.0
    m = DESIG_RE.match(key or "")
    if not m:
        return 0.0
    return 1.0 if m.group(1).upper() in {p.upper() for p in pre} else 0.0


class Priors:
    """先验表。`load()` 会额外读 `work\fusion_priors.json`（如果存在）用于扩展词表。"""

    def __init__(self, extra: Optional[dict] = None):
        self.extra = extra or {}
        self.pin_names = set(PIN_NAME_SET) | set(self.extra.get("pin_names") or [])
        self.networds = set(self.extra.get("net_words") or [])
        self.values: Dict[str, List[str]] = dict(self.extra.get("values") or {})

    @classmethod
    def load(cls, path=None) -> "Priors":
        p = Path(path) if path else (_HERE / "priors.json")
        if p.exists():
            try:
                return cls(json.loads(p.read_text(encoding="utf-8")))
            except Exception:                                  # noqa: BLE001
                pass
        return cls()

    # ------------------------------------------------------------ 连号
    @staticmethod
    def desig_runs(nums: Sequence[int]):
        """把一串数字切成「连号段」。返回 [(lo, hi, hits, span)]，按长度降序。"""
        if not nums:
            return []
        u = sorted(set(int(n) for n in nums))
        segs, cur = [], [u[0]]
        for n in u[1:]:
            if n - cur[-1] <= RUN_MAX_GAP:
                cur.append(n)
            else:
                segs.append(cur)
                cur = [n]
        segs.append(cur)
        out = []
        for sg in segs:
            lo, hi = sg[0], sg[-1]
            span = hi - lo + 1
            if len(sg) >= RUN_MIN_HITS or (span <= 4 and len(sg) >= 2):
                out.append((lo, hi, len(sg), span))
        out.sort(key=lambda t: -(t[2] / max(1, t[3])))
        return out

    def run_keys_for(self, observations: Sequence[Tuple[str, int, float]]):
        """从 (来源标签, 数字, 分数) 的观测里推出**应当存在的位号集合**。

        返回 `{前缀: (缺失数字列表, lo, hi)}`；只对「成串且不够满」的前缀给结论。
        """
        by_pre: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
        for lab, n, sc in observations:
            pre, num = split_desig(lab)
            if num is None:
                continue
            by_pre[pre].append((lab, num, sc))
        out = {}
        for pre, obs in by_pre.items():
            nums = [n for _l, n, _s in obs]
            runs = self.desig_runs(nums)
            if not runs:
                continue
            lo, hi, hits, span = runs[0]
            have = {n for n in nums if lo <= n <= hi}
            missing = [n for n in range(lo, hi + 1) if n not in have]
            # 空洞太多就认为不是一串（可能是两组编号混在一起）
            if not missing or len(missing) > max(2, span // 2):
                continue
            out[pre] = (missing, lo, hi)
        return out

    # ------------------------------------------------------------ 位号修复
    @staticmethod
    def repair_variants(value: str, ctype: str = "", max_out: int = 6):
        """给一个位号读法生成**修复变体**。

        实测过的最主要错误就是「吞掉窄 `1`」（`case 1` 的 `C151`→`C51`、`C154`→`C54`），
        所以最值得生成的变体是**在数字段的每个位置插回一个 `1`**。
        这些变体的分数要打折（`REPAIR_PENALTY`），只有在没别的候选能解释连号时才该赢。
        """
        pre, num = split_desig(value)
        if num is None:
            return []
        if ctype and key_fits_type(value, ctype) <= 0:
            return []
        d = str(num)
        out = []
        for i in range(len(d) + 1):
            v = "%s%s1%s" % (pre, d[:i], d[i:])
            if v != value and v not in out:
                out.append(v)
        return out[:max_out]

    def windows_for(self, prefix: str, slots: Sequence[str], pool,
                    allow_repair: bool = True, topk: int = TOP_WINDOWS):
        """给一个前缀的 n 个槽位找**最可能的连号窗口** `[lo, lo+n-1]`，返回前 `topk` 个。

        做法：把每个槽位的全部候选（含**修复变体**）折算成数字，然后滑窗，
        按「有候选落在窗内的槽位数」降序、同分按分数和降序排。

        ⚠️ **纯文本分不出前两名时不要硬选**。实测 `case 1`（真值 `C151..C156`，
        读成 `C51/C15/C5/C54/C5/C15`）：
          - 窗口 `[51,56]` 覆盖 4/6（`C51`、`C54` 直接命中，另 2 个靠修复）
          - 窗口 `[151,156]` 覆盖 4/6（4 个都靠「插回窄 `1`」命中）
        两者证据量**完全相等**，文字侧无法区分 —— 能区分的是**图**
        （那个框里到底有没有那个窄 `1`）。所以这里返回前几名，
        由 `llm_judge` 拿着图去二选一；这正是「先验收窄、LLM 决策」的分工。
        """
        n = len(slots)
        if n <= 0:
            return []
        cands: Dict[str, Dict[int, float]] = {}
        for sid in slots:
            d: Dict[int, float] = {}
            ctype = (pool.meta.get(sid) or {}).get("type") or ""
            for a in pool.alts(sid):
                p, k = split_desig(str(a.value))
                if p == prefix and k is not None:
                    d[k] = max(d.get(k, 0.0), _f(a.score))
                # ★ 修复变体**必须对每一个候选都生成**，不能只在直接解析失败时生成：
                #   `C5` 能直接解析成 5，但它的真值很可能是 `C51`/`C153`。
                if allow_repair:
                    for v in self.repair_variants(str(a.value), ctype):
                        p2, k2 = split_desig(v)
                        if p2 == prefix and k2 is not None:
                            d[k2] = max(d.get(k2, 0.0),
                                        _f(a.score) * REPAIR_PENALTY)
            if d:
                cands[sid] = d
        if not cands:
            return []
        all_nums = sorted({k for d in cands.values() for k in d})
        out = []
        for lo in all_nums:
            hi = lo + n - 1
            cover = {}
            for sid, d in cands.items():
                inw = {k: s for k, s in d.items() if lo <= k <= hi}
                if inw:
                    cover[sid] = max(inw, key=lambda k: inw[k])
            free = [k for k in range(lo, hi + 1) if k not in set(cover.values())]
            score = sum(cands[sid][cover[sid]] for sid in cover) + 0.5 * len(cover)
            out.append((len(cover), score, lo, hi, dict(cover), free))
        out.sort(key=lambda t: (-t[0], -t[1]))
        return [(lo, hi, cover, free, (cnt, sc))
                for cnt, sc, lo, hi, cover, free in out[:topk]]

    def window_for(self, prefix: str, slots: Sequence[str], pool,
                   allow_repair: bool = True):
        """只取最优窗口（兼容旧调用）。"""
        ws = self.windows_for(prefix, slots, pool, allow_repair, topk=1)
        if not ws:
            return None
        lo, hi, cover, free, _k = ws[0]
        return lo, hi, cover, free

    # ------------------------------------------------------------ 往池子里补
    def annotate(self, pool, run_only_gaps: bool = True) -> dict:
        """给池子补先验候选。返回统计 dict（给 decision_log 用）。"""
        st = {"run_added": 0, "run_slots": 0, "run_windows": 0,
              "run_alt_windows": 0, "pin_prior": 0, "shape_pen": 0}

        # ---- 1 位号连号：**滑窗 + 修复变体** ------------------------------
        by_pre: Dict[str, List[str]] = defaultdict(list)
        # 一个位号都读不出来的槽位（装配层给的是占位符 `pNNN`）。它们**参与数窗口宽度**
        # （否则窗口会窄掉、`free` 号会算错），但**绝不接受先验直接派的号**。
        #
        # ★ 这条闸是实测逼出来的：`0001` 有 38 颗元件的键从 `p046` 被改成了 `P146`
        # （`solve:gapfill` / `prior:run_free` 编的）。而 `metrics.is_unkeyed`
        # 承认的占位形式只有 `^∅\d+$` 与 `^p\d{3,4}$`，`P146` **两个都不是** ——
        # 既配不上真实 GT 键，**又丢掉了坐标匹配**（GT 的 `∅NNN` 键正是靠
        # 「预测键未被键匹配占用」才走匈牙利坐标匹配的）。200 训练集里 GT 键有
        # 4034/9171 是 `∅NNN` ⇒ 这是成片的双重损失。9 张 GT 上 `0001` 因此
        # `29.33→` 掉到更低、`0005` `13.14→11.44`。
        # 结论：**先验只能细化读数，不能发明身份。**
        no_read = set()
        for sid in pool.slot_ids():
            if not sid.startswith("comp.") or not sid.endswith(".key"):
                continue
            ctype = (pool.meta.get(sid) or {}).get("type") or ""
            if ctype in NO_DESIGNATOR:
                continue
            b = pool.best(sid)
            v = str(b.value) if b else ""
            pre, num = split_desig(v)
            if pre and num is not None:
                by_pre[pre].append(sid)
            else:
                # 完全读不出位号的元件：靠**类型**把它挂到某个前缀上（只为数窗口宽度）
                no_read.add(sid)
                for p in TYPE_PREFIX.get(ctype, ()):
                    by_pre[p.upper()].append(sid)
        for pre, slots in by_pre.items():
            slots = sorted(set(slots))
            if len(slots) < 2:
                continue
            ws = self.windows_for(pre, slots, pool)
            if not ws:
                continue
            lo, hi, cover, free, wkey = ws[0]
            st["run_windows"] += 1
            if len(cover) < max(2, int(0.5 * len(slots))):
                continue                     # 覆盖太差，窗口不可信
            used = set(cover.values())
            # (a) 窗内的候选：**只给没有读数的槽位加成**，有读数的槽位上打成
            #     「比读数低一点」的备选。★ 实测：`prior:run_inwindow` 用
            #     `best + 0.02` 把真读数抢走，是 `0001 29.33→` 与 `0004`
            #     掉分的直接原因（`C1→C11`、`R6→R16`、`R5→R15`）。与上面
            #     `no_read` 同一条原理：**读数是证据，连号只是形状。**
            for sid, num in cover.items():
                cand = "%s%d" % (pre, num)
                b = pool.best(sid)
                if b is not None and str(b.value) == cand:
                    continue                 # 已经是最优，不用重复加
                pool.add(sid, cand, score=_cap(pool, sid, cand,
                                               (_f(b.score) if b else 0.5) + 0.02),
                         source="prior:run_inwindow",
                         evidence={"连号窗口": [lo, hi],
                                   "窗内已有号": sorted(used)},
                         question="元件 %s 的位号是哪一个？" % sid.split(".")[1],
                         meta=dict(pool.meta.get(sid) or {}))
                st["run_added"] += 1
            # (b) **窗里没人认领的号**：按**读数模型**一对一派给最像的槽位 —— 这条是
            #     纯文本读数给不出的信息（`case 1` 的 `C152/C153/C156` 就这样补上）。
            #     ⚠ 原来是「按原分高低顺序 zip 过去」，那是错的：`C5` 更该配 `C155`
            #     （少一个字符）而不是 `C152`。
            outside = [s for s in slots if s not in cover and s not in no_read]
            obs = []
            for s in outside:
                b = pool.best(s)
                obs.append((s, str(b.value) if b else "", _f(b.score) if b else 0.0))
            for sid, cand, lik in fit_numbers(obs, free, pre):
                b = pool.best(sid)
                # ⚠ 这里看着像 bug：`b.score < 0` 时 `* 0.95` 是把分数**抬高**，
                # 于是「连号窗口造出来的号」会盖过「读数」。我按「读数是证据」的
                # 原则把它改成「正分打 95 折 / 非正分再低 0.01 / 统一过 `_cap`」，
                # 220 张实测**反而更差**：28.4200 → **28.3137**（−0.11），
                # case 0077 从 −4.51 掉到 −5.36，netH tp 3383→3356、fp 4492→4519。
                # ⇒ 这条先验在这里是**净帮忙**的：它给的那些「图上没有的号」在
                # `solve` 的连号竞争里替别的槽位守住了位置。**已回退到原式**。
                pool.add(sid, cand, score=(_f(b.score) if b else 0.5) * 0.95,
                         source="prior:run_free",
                         evidence={"连号窗口": [lo, hi], "窗内已占用": sorted(used),
                                   "原最优": str(b.value) if b else None,
                                   "读数似然": round(float(lik), 3),
                                   "窗口覆盖": "%d/%d" % (len(cover), len(slots))},
                         question="元件 %s 的位号是哪一个？" % sid.split(".")[1],
                         meta=dict(pool.meta.get(sid) or {}))
                st["run_added"] += 1
                st["run_slots"] += 1
            # (c) **并列窗口**：文字侧分不出胜负的那几个，把各自窗内的候选也塞进去
            #     （分数打折，不会抢走最优），并把窗口清单写进 context ——
            #     `llm_judge` 就拿着这几个窗口去问**二选一**的问题，
            #     而不是让模型凭空猜。
            alts = [[l, h] for l, h, _c, _f, _k in ws[1:]]
            if alts:
                st["run_alt_windows"] += len(alts)
            for sid in slots:
                s = pool.hy.slots.get(sid)
                if s is not None:
                    s.context["连号窗口候选"] = [[lo, hi]] + alts
            for l, h, cv, _fr, _k in ws[1:]:
                for sid, num in cv.items():
                    if sid not in slots or sid in no_read:
                        continue
                    cand = "%s%d" % (pre, num)
                    b = pool.best(sid)
                    # ★ 必须过 `_cap`（与 `run_inwindow` 同判据）：这里的读数分
                    # 常常是**负的**（置信度减去「离框」惩罚），而 `× 0.90` 对负
                    # 数是**把分数抬高** —— 于是「并列窗口」造出来的号会盖过真读数。
                    # 220 张实测：不 cap 时 `0037` 的 `C1→C11`、`0053` 的
                    # `D1→D11/R2→R12/R3→R13`、`0054` 的 `C1→C11` 全是这么来的，
                    # 单这三张就值 −27。
                    pool.add(sid, cand, score=_cap(
                        pool, sid, cand, (_f(b.score) if b else 0.5) * 0.90),
                             source="prior:run_altwindow",
                             evidence={"并列窗口": [l, h], "首选窗口": [lo, hi]},
                             question="元件 %s 的位号是哪一个？" % sid.split(".")[1],
                             meta=dict(pool.meta.get(sid) or {}))
                    st["run_added"] += 1
            pool.note("run_prior", prefix=pre, lo=lo, hi=hi,
                      covered=len(cover), slots=len(slots), free=free,
                      windows=[[l, h] for l, h, _c, _f, _k in ws])

        # ---- 2 引脚号：一颗元件内 1..N，认不出的号由空位推出 --------------
        pins_of = (pool.doc or {}).get("pins") or {}
        for cid, rec in ((pool.assigned or {}).get("comps") or {}).items():
            # ★ `BOX_PIN_PRIOR=0`（默认）：IC 的引脚号是**印出来的封装号**，不是
            # `1..N` 排列，补位置号会把 `assemble` 的 `ok` 闸门骗过去（见常量注释）。
            if not BOX_PIN_PRIOR:
                _ct = str((pool.meta.get("comp.%s.key" % cid) or {}).get("type")
                          or "").strip().lower()
                if _ct in _PRIOR_IC_TYPES:
                    st["pin_prior_skip_ic"] = st.get("pin_prior_skip_ic", 0) + 1
                    continue
            n_pins = len(pins_of.get(cid) or []) or len(
                (rec or {}).get("pins") or {})
            if n_pins <= 0:
                continue
            slots = ["pin.%s.%d.number" % (cid, k) for k in range(n_pins)]
            read: Dict[str, int] = {}
            for sid in slots:
                b = pool.best(sid)
                if b is not None and str(b.value).isdigit():
                    read[sid] = int(str(b.value))
            exp = list(range(1, n_pins + 1))
            cnt = Counter(read.values())
            taken = {n for n, c in cnt.items() if c == 1}      # 只被一个引脚占用的号
            free = [n for n in exp if n not in taken]
            # 需要补号的目标：**没有读数**的引脚，以及**读数撞号**的引脚
            targets = [s for s in slots if s not in read]
            targets += [s for s, n in read.items() if cnt[n] > 1]
            targets = sorted(set(targets), key=lambda s: slots.index(s))
            for sid, num in zip(targets, free):
                k = sid.split(".")[2]
                pool.add(sid, str(num), score=_cap(pool, sid, str(num), 0.5),
                         source="prior:pinrun",
                         evidence={"元件引脚数": n_pins,
                                   "已有号": sorted(taken),
                                   "空号": free},
                         question="元件 %s 第 %s 个引脚的号是？" % (cid, k),
                         meta=dict(pool.meta.get(sid) or {}))
                st["pin_prior"] += 1

        # ---- 3 形状惩罚：给「不像」的候选降权（来源仍是它自己） -----------
        #
        # ★ `SHAPE_W > 0` 时这里**真的改分**（原先是静默 no-op，见文件头 `SHAPE_W` 注释）。
        # 乘 `k = FLOOR + (1-FLOOR)·f` 再按 `SHAPE_W` 插值：`f=1` 时 `k=1`
        # （**只降不升** —— 形状永远不能凭空抬高一个候选）；`f=0` 时 `k=FLOOR`。
        # 改完**必须重排** `s.alts`，否则 `best()`（= `alts[0]`）还指着旧的第一名。
        for sid in pool.slot_ids():
            s = pool.hy.slots.get(sid)
            if s is None:
                continue
            kind = sid.rsplit(".", 1)[-1]
            touched = False
            for a in s.alts:
                v = str(a.value)
                if sid.startswith("comp.") and kind == "key":
                    f = key_fits_type(v, (pool.meta.get(sid) or {}).get("type") or "")
                elif kind == "number":
                    f = pin_number_shape(v)
                elif kind == "name":
                    f = pin_name_shape(v)
                elif kind == "value":
                    f = value_shape(v, (pool.meta.get(sid) or {}).get("type") or "")
                elif kind == "partname":
                    f = 0.9 if re.match(r"^[A-Za-z]", v) else 0.2
                else:
                    f = label_shape(v)
                ev = dict(a.evidence or {})
                ev["形状分"] = round(f, 2)
                a.evidence = ev
                if f <= 0.0:
                    st["shape_pen"] += 1
                if SHAPE_W > 0.0 and f < 1.0:
                    k = SHAPE_FLOOR + (1.0 - SHAPE_FLOOR) * f
                    a.score *= (1.0 - SHAPE_W) + SHAPE_W * k
                    touched = True
                    st["shape_demote"] = st.get("shape_demote", 0) + 1
            if touched:
                s.alts.sort(key=lambda a: -a.score)
        return st

    # ------------------------------------------------------------ 体检
    def check(self, pool) -> List[dict]:
        """列出违反常识的地方（给 `plausible` / `conflict` 用的原始材料）。"""
        bad: List[dict] = []

        # 位号撞号 / 断号 / 前缀与类型不符
        by_pre: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for sid in pool.slot_ids():
            if not sid.startswith("comp.") or not sid.endswith(".key"):
                continue
            v = pool.chosen(sid)
            if v is None:
                continue
            cid = sid.split(".")[1]
            ctype = (pool.meta.get(sid) or {}).get("type") or ""
            if key_fits_type(str(v), ctype) <= 0:
                bad.append(dict(kind="key_type_mismatch", slot=sid, value=str(v),
                                type=ctype,
                                why="位号前缀 %s 与类型 %s 不允许的前缀不符"
                                    % (str(v), ctype)))
            pre, num = split_desig(str(v))
            if num is not None:
                by_pre[pre].append((sid, cid))
        for pre, lst in by_pre.items():
            nums = []
            for sid, cid in lst:
                _p, n = split_desig(str(pool.chosen(sid)))
                if n is not None:
                    nums.append(n)
            dup = [n for n, c in Counter(nums).items() if c > 1]
            if dup:
                bad.append(dict(kind="key_duplicate", prefix=pre, numbers=dup,
                                why="同一张图里位号唯一，%s%s 被多颗元件占用"
                                    % (pre, dup)))
            for lo, hi, hits, span in self.desig_runs(nums):
                have = {n for n in nums if lo <= n <= hi}
                miss = [n for n in range(lo, hi + 1) if n not in have]
                if miss and len(miss) <= max(2, span // 2):
                    bad.append(dict(kind="key_gap", prefix=pre, lo=lo, hi=hi,
                                    missing=miss,
                                    why="%s%d..%s%d 成串但缺 %s"
                                        % (pre, lo, pre, hi, miss)))

        # 网络标签形状
        for sid in pool.slot_ids():
            if not (sid.startswith("netlabel.") or sid.startswith("net.")):
                continue
            v = pool.chosen(sid)
            if v is None:
                continue
            if label_shape(str(v)) <= 0.1:
                bad.append(dict(kind="label_shape", slot=sid, value=str(v),
                                why="`%s` 不像网络名" % v))

        # 引脚名/号形状
        for sid in pool.slot_ids():
            if not sid.startswith("pin."):
                continue
            v = pool.chosen(sid)
            if v is None:
                continue
            if sid.endswith(".name") and pin_name_shape(str(v)) <= 0.1:
                bad.append(dict(kind="pinname_shape", slot=sid, value=str(v),
                                why="`%s` 不像引脚名" % v))
            if sid.endswith(".number") and pin_number_shape(str(v)) <= 0.1:
                bad.append(dict(kind="pinnum_shape", slot=sid, value=str(v),
                                why="`%s` 不像引脚号" % v))
        return bad


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    from cvparts.hypo import CaseHypotheses

    class _P:                                    # 极简假池子
        def __init__(self, doc=None, assigned=None):
            self.hy = CaseHypotheses("t")
            self.meta = {}
            self.notes = []
            self.doc = doc or {}
            self.assigned = assigned or {}

        def slot_ids(self):
            return sorted(self.hy.slots)

        def best(self, sid):
            s = self.hy.slots.get(sid)
            return s.best() if s else None

        def alts(self, sid):
            s = self.hy.slots.get(sid)
            return list(s.alts) if s else []

        def chosen(self, sid):
            s = self.hy.slots.get(sid)
            return s.chosen if s and s.resolved() else (s.best().value if s
                                                        and s.best() else None)

        def add(self, sid, v, score=0.0, source="", evidence=None, question="",
                meta=None, **kw):
            self.hy.add(sid, v, score, source, evidence or {}, question)
            if meta:
                self.meta[sid] = meta
            return self.hy.slots[sid]

        def note(self, *a, **kw):
            self.notes.append((a, kw))

    ok = []
    # ★ case 1 的真实形态：六个框，读数全错（CRNN 吞掉了窄 `1` 与尾数）
    #   真值 C151..C156，读数 C51 / C15 / C5 / C54 / C5 / C15 ⇒ 0/6。
    reads = ["C51", "C15", "C5", "C54", "C5", "C15"]
    p = _P()
    for i, r in enumerate(reads):
        p.add("comp.c%d.key" % i, r, score=1.0 - 0.01 * i,
              source="ocr:designator", meta={"type": "c", "cid": "c%d" % i})
    P = Priors()
    st = P.annotate(p)
    made = sorted({str(a.value) for sid in p.slot_ids() for a in p.alts(sid)
                   if str(a.source or "").startswith("prior:run")})
    inwin = [v for v in made if re.match(r"^C15\d$", v)]
    ok.append(("连号先验产出了候选", st["run_added"] > 0))
    ok.append(("修复变体覆盖到 C15x", len(inwin) >= 2))
    wctx = (p.hy.slots["comp.c0.key"].context or {}).get("连号窗口候选") or []
    ok.append(("真值窗口 151..156 出现在并列窗口里", [151, 156] in [list(w) for w in wctx]))
    ok.append(("首选窗口是 51..56（文字侧的贝叶斯最优）",
               [51, 56] in [list(w) for w in wctx]))
    # 干净情形：C151..C154 齐 + 一个错的 C15 ⇒ 缺口 C15x 应当被派出去
    p2 = _P()
    for i, r in enumerate(["C151", "C152", "C154", "C155", "C15"]):
        p2.add("comp.d%d.key" % i, r, score=1.0, source="ocr:designator",
               meta={"type": "c", "cid": "d%d" % i})
    P.annotate(p2)
    got2 = sorted({str(a.value) for a in p2.alts("comp.d4.key")})
    ok.append(("d4 拿到窗内号", any(re.match(r"^C15[0-9]$", v) for v in got2)))
    # 引脚号：5 个引脚只读出 1/2/4/5 ⇒ 第 5 个应当被补上 3
    doc = {"pins": {"u0": [[0, 0]] * 5}}
    assigned = {"comps": {"u0": {"pins": {}}}}
    p3 = _P(doc, assigned)
    for k, num in enumerate(["1", "2", "4", "5"]):
        p3.add("pin.u0.%d.number" % k, num, score=1.0, source="ocr:pin_number",
               meta={"type": "box", "cid": "u0", "k": k})
    P.annotate(p3)
    ok.append(("引脚 3 被补到空位",
               any(str(a.value) == "3" for a in p3.alts("pin.u0.4.number"))))
    # 形状
    ok.append(("10K 是像样的电阻值", value_shape("10K", "r") == 1.0))
    ok.append(("22T10uH4 不是电容值", value_shape("22T10uH4", "c") <= 0.2))
    ok.append(("AGND 归类为地", net_class("AGND") == "ground"))
    ok.append(("3.3V 归类为电源", net_class("3.3V") == "power"))
    ok.append(("SCLK 在引脚名词表里", pin_name_shape("SCLK") == 1.0))
    ok.append(("cLA 不像标准引脚名", pin_name_shape("cLA") < 1.0))
    bad = [n for n, c in ok if not c]
    print("priors 自检 %d 项：%s" % (len(ok), "全部通过" if not bad else "失败"))
    for n in bad:
        print("  FAIL", n)
    print("  连号统计：%s" % st)
    print("  case1 造出的候选：%s" % made)
    print("  d4 候选：%s" % got2)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_demo())
