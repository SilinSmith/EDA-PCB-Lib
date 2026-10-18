# -*- coding: utf-8 -*-
"""第二阶段 · 组装：把 stage1 的几何产物 + 第二阶段的文字，拼成官方提交格式。

坐标系在**这一层**翻转（`y_json = H - y_img`，左下角原点）。stage1 全程用图像
坐标（左上角原点），OCR 用的也是图像坐标的框 —— 只在出口处翻一次，避免来回翻。

网络标签的逻辑合并也在这里：几何只连得出一部分网络（实测拿 GT 自己的走线去连
同一 net 的引脚，召回也只有 0.658），剩下 34% 靠纸上同名的网络标签相连。做法是
把标签绑到最近的那条走线的**自由端**，然后把同名标签的网络并起来。
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:                                    # 作为包导入（submission\run.py 把 eda-pcb\ 加进 sys.path）
    from envflag import env_on, env_str  # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage2\assemble.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on, env_str

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VALUE_TYPES = ("r", "c", "l", "fuse", "crystal_2pin", "crystal_3pin",
               "crystal_4pin", "crystal")
LABEL_TOL = 16.0        # 网络标签离走线自由端多远还算「贴在这条线上」
FREE_TOL = 2.0          # 两个端点离多近算「接上了」
# 位号没读出来时**要不要**输出这颗元件（兜底键 `pNNN`）。默认 `0` = **输出**。
#
# ★★ 2026-10-14 翻转（完整依据见 `docs/RESULTS.md` §5.20）：官方**书面口径**（题面 +
#   `docs/QA_2025.txt` 官方问答 + 业主 2026-10 转述）与 2026-10-09 那台判题机
#   **有三处冲突**，而业主已明确说「之前的判题机写错了」（bbox 那处）：
#     ① bbox 容差：题面 `problem_statement.txt:179` **20px** ｜ 判题机 5px（业主已认错）；
#     ② 无名元（`∅NNN`）：官方问答 评测 Q5「Vcc 和 gnd 是 ∅，**直接是坐标匹配**」
#        ｜ 判题机对 `∅NNN` 也走**键精确**（case 1 的 8 个 `∅` 全记 `missing`）；
#     ③ 引脚分母：官方问答 评测 Q2「非 box 类型元器件 pin name, pin number 不参与评分」
#        ｜ 判题机把非 box 的脚也计进 `pin_f1`（`{"pin_id": "[\"C1513\",\"pin_1\"]", "error": "extra"}`）。
#   三处都是「判题机不照书面口径做」，① 既然修了，② ③ 按同一逻辑应当也修 ⇒ 按
#   **书面口径**定默认。
#
#   实测（官方 10 张；同一批 `*_Pred.json` 用两种尺子各打一遍）：
#     · 书面口径尺（`tools/eval/stage2_eval.py`，= `EvalConfig()` 默认：
#       bbox 20 / `unkeyed_match='gt'` / 只统计 box 引脚）：
#         **70.0786（`=1` 旧默认）→ 82.5689（`=0`）＝ +12.49**
#         （comp 0.7057→0.9819 ｜ netH 0.7686→0.8946 ｜ line 0.3492→0.3672
#          ｜ pin 0.7406→0.7252，见 §5.20 的逐通道表）；
#         `--no-vlm` 臂 62.5853 → 75.8737（+13.29，与 oracle 上界逐位一致）；
#         val 19 张 20.2692 → 34.2079（+13.94）。
#       oracle 上界（`work\tools\dev\_pwr_oracle.py --mode s1`，用**我们自己阶段一
#       检出的 bbox**）：68.3369 → 81.6221（+13.29）⇒ 真实现拿到了 oracle 的全部收益。
#     · 判题机复刻尺（`fast.py --engine harness`，= `schematic_parser/official.py`）：
#       67.7370 → 57.26（**−10.48**）—— 这正是 §5.17 当年判负用的那把尺子。
#
#   为什么敢押书面口径：① GT 的 `∅` 号来自源工程的对象计数器（不可恢复）⇒ 判题机
#   **只能**按坐标配这批元件，否则它们对所有队伍恒 fn；② 官方问答是发给参赛队的
#   书面文本，明确写了「坐标匹配」与「非 box 不计分」；③ 两条尺子差 ±10 分级别，
#   书面口径的证据链（题面 + 官方问答 + 业主转述）强于「一台已知写错的判题机」。
#   **风险与回退**：`STAGE2_DROP_UNKEYED=1` 一个环境变量即可退回旧行为（用
#   `fast.py --engine harness` 复核）。
#
# 历史（2026-10-13，判定「丢掉」的那批读数，保留作证据 —— 它们**只对判题机复刻尺**成立）：
#   * 官方 component 在那台判题机上是**纯键匹配**：case 1 里 GT 的 8 个 `∅NNN`
#     （5 gnd + 3 v）全部记 `component.missing`，我们输出的 8 个 `pNNN` 全部记
#     `component.extra`（`component f1 tp 6`、`key_f1 tp 7`、`fn 16 = 22 − 6`）；
#   * `pNNN` 与 `∅NNN` 两臂在那把尺子上**逐位相同**（57.59 / 57.59，基线 66.45）；
#   * 直接抄 GT 的 `∅` 号能到 73.48（+7.03），但号不可恢复；
#   * 把那 10 张的 `pNNN` 元件 + 引脚 + 超图引用整体删掉，用那把尺重打
#     （`work\tools\dev\_offscore.py --engine official`）：**52.3666 → 62.9713（+10.6047）**；
#   * 更早的 220 张实测 `=1` 15.5084 vs `=0` 21.1165（结论相反）当时被当成「旧尺的
#     幻觉」作废 —— 现在看，那条「旧尺」就是**书面口径**，它的方向才是对的。
DROP_UNKEYED = env_on("STAGE2_DROP_UNKEYED", "0")

# ---------------------------------------------------------------------------
# ★★ 兜底键元件（`pNNN`）的**引脚一个都不发**（`STAGE2_PIN_DROP_UNKEYED`，默认 1）。
#
# 与上面 `DROP_UNKEYED` 的分工：`DROP_UNKEYED` 决定**元件框**留不留（留 = 赌书面口径
# 的坐标匹配），本开关只决定**引脚**发不发 —— 因为那批引脚在**任何一把尺**下都是纯 fp：
#   * 判题机复刻尺（`work/_judge.py` 的 judge，按键精确匹配）：我们的 `pNNN` 永远配不上
#     GT 的 `∅NNN` ⇒ 每颗的 `pin_1..pin_k` 全部落进 `pin.extra`。官方 10 张元件级
#     fp/fn 里 **208/226 个事件都是这批无名元**（`work/_compdbg.py`）。
#   * 坐标匹配尺（书面口径 Q5，`unkeyed_match='gt'`）：`rewrite_keys` 把 `pNNN` 改写成
#     `∅NNN` 之后，GT 那些引脚键是**源文档内部号**（实测 `pin_111`，名字 `P_27`），
#     我们的位置号 `pin_1/pin_2` 与名字 `1/2` 依旧一个都配不上 ⇒ 同样纯 fp。
#     实测：把它们的引脚与超图引用剪掉，`judge` +3.75、`judge_gt` +3.75、`protocol2` ±0.00。
#   * 网络侧更早就这么做了（`NET_DROP_UNKEYED=1`，见 :314 的长注释）。
# ⚠️ 反例（因此保留开关）：若把 `NET_DROP_UNKEYED=0` 与本开关一起改，超图里会留下
#    指向「没有引脚表」的元件的引用 —— 两者要么一起留、要么一起剪。
# 离线代理脚本：`work/_unkeyed_proxy.py`（V0 原样 / V1 连元件一起丢 / V2 只丢引脚）。
PIN_DROP_UNKEYED = env_on("STAGE2_PIN_DROP_UNKEYED", "1")

# ★★ 「被洗白的兜底键」也认回无名元件（`STAGE2_PIN_DROP_LAUNDERED`，默认 1）。
#
# 上游（`assigned[cid]["designator"]["canon"]`，实测 `_by = "solve:vlm:enrich"`）会把**已经
# 发过的兜底键 `pNNN`** 当成「读出来的位号」回填，于是 `build_doc` 的 designator 分支拿到一个
# 「有键」的元件、`_unkeyed` 保持 `False` ⇒ 上面那道 `PIN_DROP_UNKEYED` 闸被**绕过**，VLM 臂
# 的交付文档里又多出 `pNNN` 的合成脚号。
# 实证（VLM 臂，官方 10 张）：`pNNN` **元件框 110 个、脚 221 条、引用 0 条**；判题机 `pin.id_f1`
# 层这 221 条全是 fp（同批「脚号读错」只有 3 条）——`work/_pinidset.py`、`work/_undroppins.py`。
# 离线代理（把已有产物里的这批引脚组剪掉，`work/_undroppins.py` + `_offscore.py --engine official`）：
# 官方 10 张 **61.7856 → 65.5257（+3.7401）**，其中 pin 0.596239 → 0.745844（+0.1496），
# comp/netH/netL **逐位不变**；`protocol2` 主尺 **79.30 → 79.30（±0.0000**，它只算 box 引脚）；
# 派生集同向（见 `docs/RESULTS.md` §5.39）。
# `=0` 退回修复前的行为（**只作 A/B 对照**，不要当交付口径）。
PIN_DROP_LAUNDERED = env_on("STAGE2_PIN_DROP_LAUNDERED", "1")

# ---------------------------------------------------------------------------
# ★★ 内部合并类 `tp` 的**文本闸门**（只跟 `STAGE1_MERGE_TP=1` 配套，默认 off）。
#
# 业主原话（m00001）：「testpoint 和跟他搞混的那个则直接合并成一个（YOLO识别阶段），
# 而在**文本绑定时**才决定他到底是哪个」。检测阶段只出一个类 `tp`
# （`stage1/detect.py` 的 `MERGE_TP`），**在这里**落回官方类名。
#
# 为什么选 `assemble.py` 而不是 `assign.py`：`build_doc` 就是「元件键 → 官方交付
# 文档」的那一步，也是**唯一**一处把 `c["type"]` 写进交付产物的地方
# （`:637 components[key] = {"Name":…, "type": ctype, …}`）。放进 `assign.py` 得去
# 改 stage1 的 doc（那层不该知道交付格式），而且阶段三会从 `_assigned/*.json` 重新
# `build_doc` 一遍 —— 只有当决议就在 `build_doc` 里时，两次调用才必然一致。
#
# 三条规则：
#   ① 绑到了 TP 系位号（`assign.py` 的 `TYPE_PREFIX["tp"]`）**且**有 CV 形状证据
#      （子类型落在 `TP_CV_SUBTYPES` 里）→ 官方类名 `testpoint`；
#   ② 绑到了位号但没有形状证据 → `pin`；
#   ③ **没绑到位号 → 直接丢**（不管 `DROP_UNKEYED`）。这就是把原来 stage1 那条
#      「盲目的类型级丢弃」（`detect.py` 顶 `DROP_TYPES`：`pin` 预测 98.9% 是 fp）
#      换成「**有证据的文本闸门**」：位号读出来了才配占一个键，读不出来就别占。
#      「读出来了」的严格定义见 `TP_TRUST_EV`（位号要带 `text`/`conf` 之类真证据，
#      只有 `canon` 的空壳位号不算 —— 融合层会凭空补那种，实测贴错物理器件）。
# 形状证据 = CV 子类型（`stage1/detect.py` 记到元件上的 `subtype`）。
# `cvparts/testpoint.py:10-13` 只有三种：`ring3` 三层细同心圆环（案例 0050，
# `cvparts/__init__.py:12` 记「同心双圆环靶心，43/43 能找到」——43 正好等于 GT
# `testpoint` 总数 43）、`hole` 粗圆环、`dot` 单圆点。
#
# ★★ 2026-10-08 六张派生 case 实测（`work/_fast_out_tpmerge_A6mrg`，stage1 侧统计）：
#   0050 GT testpoint 20 → 27 个 tp，subtype `ring3 22 / 无 5`
#   0011 GT testpoint 12 → 13 个 tp，subtype **`hole 13`**
#   0102 GT testpoint  8 → 17 个 tp，subtype `dot 11 / hole 1 / 无 5`
#   0001 GT pin 8        →  8 个 tp，subtype `无 8`
#   0087 GT pin 68       → 70 个 tp，subtype `无 70`
#   0187 GT pin 23       → 25 个 tp，subtype `无 25`
# ⇒ **判据是「CV 有没有认出 testpoint 那一族形状」，不是「是不是 ring3」**：
#   凡 CV 给了子类型的（`ring3`/`hole`/`dot`）6 张上全部落在 GT `testpoint` 上，
#   一个 CV 子类型都没有的（`0001`/`0087`/`0187`）全部落在 GT `pin` 上（0 例反例）。
#   把 `hole`/`dot` 算成 `pin` 会让 0011 的 12 个 + 0102 的 8 个 testpoint 全变错类。
#   `STAGE2_TP_CV_SUBTYPES` 用来切臂：默认仍是 task 口径的 `ring3`（**不含**
#   `hole`/`dot`）。**放宽到 `ring3,hole,dot` 今天是一个逐位相同的 no-op**：
#   闸门台账（`STAGE2_TP_LEDGER=1`）显示 6 张里真正绑上位号的只有 `无 19 / ring3 12`，
#   `hole`（0011 的 13 个）与 `dot`（0102 的 11+1 个）**一个都没绑上**、早在 `unkeyed`
#   那一步就被丢了。所以按实测口径的正确取值是 `ring3,hole,dot`，但今天不产生分数差异，
#   默认保持 task 原话的 `ring3`；等位号绑定能力上来（例如 CRNN 前缀骨架那类改动）
#   再打开，免得那时 hole/dot 被错判成 `pin`。
TP_MERGED = "tp"
# 同一个环境变量在 `stage1/detect.py` 里也读一次（那里负责合并，这里负责决议）。
MERGE_TP = env_on("STAGE1_MERGE_TP", "0")
TP_CV_SUBTYPES = frozenset(s.strip().lower() for s in
                           env_str("STAGE2_TP_CV_SUBTYPES", "ring3").split(",") if s.strip())
# `STAGE2_TP_TRUST_EVIDENCE`（默认 1）：`tp` 的位号必须**带着读出证据**才算「绑上了」。
# 只有 `canon` 一个字段的空壳位号（融合层凭空补出来的那种）不算，整颗元件丢掉。
# 动机（2026-10-08 实测，`work/_tpmerge_who.py`）：0087 的 stage2 侧只有 8 个 tp
# 拿到 canon，到了 `runs/fusion/_assigned/0087.json` 变成 **38** 个，多出的 30 个形如
# `{"canon": "TP18", "_by": "solve:ocr:designator"}` —— 连 `text`/`conf`/`box` 都没有。
# 官方按 key 匹配元件 ⇒ 键名撞上了 GT（`TP1`/`TP2`/…）却贴在**别的物理器件**上
# （实测 `TP1` GT bbox [1102,631] vs pred [1482.8,799.6]；`TP11` 的 pred 反而落在
# GT `TP1` 的位置），于是 `pairs {"pin->pin": 19}` 里 0 个得分、`fFP` 0→38、
# compF1 0.1923→0.1546、总分 −0.0466。判据 = 除 `canon`/`_by` 外**还有别的字段**：
# `stage2/assign.py:862` 写的位号一定带 `{"text","canon","conf","box","d","fit","box_i"}`，
# 而融合层凭空补的只有 `canon`（+`_by`）。
TP_TRUST_EV = env_on("STAGE2_TP_TRUST_EVIDENCE", "1")
# 决议台账（只用于诊断，不进交付产物）：每条 = 为什么被丢 / 落成了哪个官方类名。
# 默认关（`STAGE2_TP_LEDGER=1` 打开），免得给不关心的人白建 list。
TP_LEDGER_ON = env_on("STAGE2_TP_LEDGER", "0")
TP_LEDGER: List[dict] = []

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
# 新行为（=1，默认）逐引脚采信：某个号是 1..999 的整数、且没跟同一颗 IC 里别的
# 采信号撞，就用它；否则用位置号补，并避开已经用掉的号。**严格弱于全对**（全对时
# 两者输出相同），但把「71 对 1 错」从「全错」救回「71 对」。
BOX_PIN_PARTIAL = env_on("STAGE2_BOX_PIN_PARTIAL", "1")

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
#   * `0`/`off`    = 老行为（位置号补满）；
#   * `1`/`all`    = 一律丢；
#   * `mixed`（**默认**）= **只在「这颗 IC 至少读懂了一个号」时才丢**。一个都没读懂的
#     那些，位置号 `1..N` 本身是个有用的先验（多数 IC 真是 1..N），丢了会伤召回。
#   * `cap`        = **上限版**：发明出来的号只要 **≤ 这颗 IC 真正读到过的最大号** 就留，
#     超过就丢。既砍掉「12 脚封装硬补到 12 个号」这种洪水，又保留「读到 {1,2,3,5,7} 时
#     把 4、6 补出来」这种**填洞**（洞里的号猜中和 GT 撞上的机会不小）。
#
# A/B（`python work\tools\eval\fast.py --no-vlm …`）：
#   10GT(9)：老 44.70(pin 0.1881/netH 0.4746) → `1` **47.67**(pin **0.2720**/netH **0.4971**)
#            → `mixed` **47.67**（与 `1` 逐位相同）
#   19 张  ：老 26.46(pin 0.0935) → `1` **28.31**(pin **0.1320**) → `mixed` **28.31**（同 `1`）
#   220 张 ：老 25.0109(pin 0.0619/netH 0.2078) → `1` 25.0035(pin 0.0657/netH 0.2048)
#            → **`mixed` 25.0117**(pin 0.0649/netH 0.2059) ⇒ 只有 `mixed` 端到端不降。
_DROP_MODE = os.environ.get("STAGE2_BOX_PIN_DROP_UNREAD", "mixed").strip().lower()
BOX_PIN_DROP_UNREAD = _DROP_MODE not in ("", "0", "false", "off", "none")
BOX_PIN_DROP_MIXED_ONLY = _DROP_MODE in ("mixed", "mix", "partial")
BOX_PIN_DROP_CAP = _DROP_MODE in ("cap", "max", "ceiling")

# ★ `STAGE2_BOX_PIN_KEEP_NAMED`（默认关）：`box`(IC) 里**读到了引脚名、但没读出合法号**的桩，
#   在 `BOX_PIN_DROP_UNREAD` 那条闸上**不再丢**，用位置补位号发出去。
#
# 病（§5.56/§5.57 的逐槽取证，官方 10 张 18 根缺 box 脚）：15 根在阶段一就有槽位落在 GT 点
#   5 px 内，其中 **6 根**是「名字读到了、号没读到」：
#   `2 U3.pin_1 GND`（槽 #6 读到名 `GND`）、`2 U4.pin_1 GND`、`6 U2.pin_1 OUT`、`6 U4.pin_1 OUT`
#   —— 这四个名字**完全读对**；`6 U1.pin_1 IN` / `6 U1.pin_2 GND` 读成 `D` / `O`（单字符残读）。
#   它们死在**第一条闸**（`kept=False` ⇒ 直接 `continue`），连「这根桩有真名字」都没看：
#   规则 A（`(型号, 名) → 号`）只在库认识这个名字时才能补号，而 `OUT`/`GND` 不在
#   `EG1192L`/`STC11` 的库名集里 ⇒ 补不出号 ⇒ 桩被丢。
#   发出去的代价：位置补位号**猜中** GT 号 ⇒ `pin.<号>` 名对 = tp；猜不中 ⇒
#   `name_mismatch`（fp+fn，比「不发」的 1 fn 多一个 fp），但网络通道白赚一个成员。
#   ⇒ 必须 A/B 量，不能想当然（同类「一律不丢」的旧档都判负：`DROP_UNREAD` 的老行为、
#   `BOX_PIN_NEED_NAME=0` 的交付官方 10 张 **73.53 vs 73.60**、模式 3 的 **−0.1993**）。
#
# 取值：`0`/`off`（默认）= 现行行为；`1`/`named` = 「名字 ≥2 字符且含字母」就留；
#       `2`/`known` = 再加一道「这是全局引脚名词表认识的名字」（`libpins.is_known_name`
#       ⇒ `GND`/`OUT`/`IN` 过、`D`/`O`/`N`/`H` 这种单字符残读不过）。
#
# ★ 实测判负（§5.57，2026-10-13，两把尺子都掉、默认保持 `0`）：
#   - 判题机复刻尺（官方 10 张）`tok_g6def` 63.3417 → `named` **61.5613（−1.7804）**、
#     `known` **62.0943（−1.2474）**；协议尺 76.39 → **73.67 / 74.40**。
#   - 派生 18 张：`known` 复刻尺 14.5376 → **14.2543（−0.2833）**、协议尺 26.95 → 26.81。
#   - **18 根缺脚一根都没救回来**（`work\_pinname_audit.py` 的「缺脚」计数仍是 18）：
#     位置补位号与 GT 号对不上，多出来的脚全是 `pin.extra`。官方 10 张多出 28/21 根脚，
#     例如 case 7 的 `U1` 多出 `pin_1 = "2R"`（点 `(824, 689.7)`，而 GT `pin_1 AGND` 在
#     `(881, 550)`）—— 这是**垃圾文本槽**，不是我们要救的那 6 根。
#   - 真代价在**网络**：假脚会挂进网络、把连通性搅浑（官方 conn F1 0.7565 →
#     0.7223/0.7336）。⇒ 「让库来补号」这条路的本体不是「名字缺」，而是**号缺**；
#     号没法从库里推出来时，发出去只会多 fp。
#   - 反例教训：`known` 档比 `named` 档少发 7 根脚、却仍掉 1.25 分 ⇒ 问题不在「名字像不像」，
#     而在「号是不是猜的」；未来若要重开，必须先有**读出号**或**号序可信**的新证据。
_KEEP_NAMED_MODE = os.environ.get("STAGE2_BOX_PIN_KEEP_NAMED", "0").strip().lower()
BOX_PIN_KEEP_NAMED = _KEEP_NAMED_MODE not in ("", "0", "false", "off", "none")
BOX_PIN_KEEP_NAMED_KNOWN = _KEEP_NAMED_MODE in ("2", "known", "vocab")


def _named_slot_ok(nm, ctx=None) -> bool:
    """`BOX_PIN_KEEP_NAMED` 的判据：这个「读到的名字」算不算真名字（见上方长注释）。

    注意 `ctx` 必须由调用方传进来：`_LIB_CTX` 是装配函数里的**局部变量**（`assemble.py:1283`），
    模块级函数直接引用它会 `NameError`（2026-10-13 的 `known` 档首跑就踩了这个坑：10 张里
    9 张阶段二失败、总分掉到 8.04）。
    """
    s = str(nm or "").strip()
    if len(s) < 2 or not any(ch.isalpha() for ch in s):
        return False
    if not BOX_PIN_KEEP_NAMED_KNOWN:
        return True
    return bool(ctx is not None and ctx.is_known_name(s))


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
#     （旧的 `BOX_PIN_NEED_NAME = env_on(...)` 已被下面的三态解析取代，见 `_BOX_NEED_NAME_MODE`。）
# ★★★ 「迟到版」`STAGE2_BOX_PIN_NEED_NAME=2`（`late`/`lib`，**现为默认**）：**把闸门挪到规则 B 之后**。
#     动机（2026-10-10，交付账本 `_pinledger.py --run work\_fast_out_fin`）：缺号 37 个里
#     **18 个（49%）**是「号读到了、闸门把桩丢了」。逐槽查 case 2 的 U1/U4 发现形态是
#     **号与名落在相邻的两个候选槽里**（`num='4'` 在 slot 5、`name='SDA'` 在 slot 3）⇒
#     这一支的号是**真的**，只是名字没跟过来；而规则 B（官方 box/IC 库 `(型号, 号) → 名`，
#     只在唯一时才改名、`_libnum.py` 实测 20/20 正确）正好能把这个名字补上，
#     但它在 `:953` 的闸门**之后**执行 ⇒ 现在这些桩根本没机会拿到库名。
#     语义：`0` = 从不丢；`1` = 老行为（没名就丢，库名救不回来）；`2` = 只在
#     「库里也补不出名字」时才丢（仍不发射「拿号当名」的桩 —— 那必然 name_mismatch）。
# ★★★ 交付 A/B（官方 10 张，`work\_pinab12.ps1`，三臂都带 VLM：`补强 10 张、失败 0 张`）：
#     `1` **73.27（与§2.17 的出厂默认逐案逐位相同 ⇒ no-op，A/B 干净）**
#     `2` **73.60**（+3.28，+0.33/案；pin micro 0.6424→**0.6564**；涨 3 平 6 跌 1：
#          case 2 +0.72 / case 4 +2.50 / case 8 +0.58 / case 5 −0.50）
#     `0` 73.53（+2.55；涨 3 平 4 **跌 3**：case 10 −2.17 / case 5 −1.38 / case 2 −0.31
#          ⇒ 「从不丢」会把「拿号当名」的桩发出去，正是 name_mismatch 的 fp+fn）
#     ⇒ **`2` 最优且唯一退步只有 −0.50**，故把默认从 `1` 改成 `2`。
#     天花板（`work\tools\dev\_gateceiling.py`，交付跑 723 个槽）：能补出名的只有 1 条
#     （case 2），能补出号的只有 3 条 ⇒ 这条通道的本质收益就是「把库能认的那几根脚救回来」。
_BOX_NEED_NAME_MODE = os.environ.get("STAGE2_BOX_PIN_NEED_NAME", "2").strip().lower()
BOX_PIN_NEED_NAME = _BOX_NEED_NAME_MODE not in ("", "0", "false", "off", "none")
# ★ 模式 `3` **也必须在「迟到」组里**：`:1335` 的早闸门是「非 LATE 且没名字 ⇒ 丢」，
#   若 3 不置 LATE，桩在早闸门就被丢了、根本走不到下面那条被放宽的迟到闸门
#   （实测：`STAGE2_BOX_PIN_NEED_NAME=3` 若 LATE=False，官方 10 张一根脚都不多）。
BOX_PIN_NEED_NAME_LATE = _BOX_NEED_NAME_MODE in ("2", "late", "lib", "after",
                                                 "3", "kept", "readnum")
# ★★★ 模式 `3`（`kept`/`readnum`）：**只丢「连号都不是读出来的」那些桩**。
#     病：交付档里还留着一批「号读对了、名字没读到、库也补不出名」的桩被迟到闸门丢掉
#     （官方 10 张实测 13 条 `pin.missing`，例如 case 2 `U3` 槽 4 号 '3' 无名、
#     case 6 `U2`/`U4` 槽 1 名 'OUT' 无号、case 7 `U1` 槽 9 名 'N' 无号）。
#     它们**同时**是 netH 的缺成员（`_netloss.py` 的 ②a 12 条）—— 丢掉等于引脚通道记一个
#     `fn`、网络通道再丢一个真成员。发出去：引脚通道变成 `name_mismatch`（fp+fn，
#     与「不发」在 tp+fp+fn 上完全同值，见 `metrics.py:475-476`），网络通道白赚一个成员。
#     位置补出来的号（`kept=False`）仍然丢 —— 那是「假引脚塞进网络」的来源。
# ★ 实测判负（离线 `work\_libslot.py --mode 0` + 官方 10 张，同批对照）：`3` 让引脚
#   597→599、`pin.missing` 13→11、`pin.name_mismatch` 10→**12**、`net.pred_pins` 365→367、
#   `net.soft_tp` 124.59→**124.92**（真成员确实进来了），但**总分 79.1526→78.9533（−0.1993）**
#   —— 引脚通道 `name_mismatch`=fp+fn（与「不发」同值再加一个 fp），且多出 1 张 pred 网
#   本身就是纯 fp。⇒ **默认仍是 `2`，模式 3 只作为已量过的备选留档**（别再默认打开）。
_BOX_NEED_NAME_KEEPREAD = _BOX_NEED_NAME_MODE in ("3", "kept", "readnum")

# ★ 官方元器件库（`cpntLibrary`）引脚先验的**总闸**：`STAGE2_LIB_PINS`，**默认 `1`（开）**。
# ★★ 默认值在 2026-10-09 **从 `0` 翻成 `1`**，依据是 220 张回归（`work\tools\diag\_libpins_cpu.py`
#    + `work\tools\diag\_split220.py`，输入 `work\_full220\runs\stage1` + `work\_f220_A\_assigned`）：
#   官方 10GT 10 张 62.5743→**65.1834（+2.6091，7 涨 0 跌**；pin tp 38→47 / fp 42→38 / fn 98→89）；
#   另 10 张官方（`10_GTcase`）逐位不变；派生 200 张 31.3691→**31.4300（+0.0609**，24 涨 25 跌，
#   最差 0167 −2.924）；官方 20 合计 **+1.3045**；交付链（真 `fusion.run`）官方 9 张 **+1.60**。
#   门槛（官方为正 且 派生不为负）**已过**。
# ★★ 2026-10-14（用户 m42932「以新的这个为参考」）：默认库从 `cpntLibrary.json`（337 键）
#   换成官方 box/IC 库 **`BoxICLibrary_200cases.json`（360 键）** —— `pin_f1` 只考 box 元件，
#   这份库正是那一档。同口径三臂 A/B（只换库文件）：官方 9 张 68.7995 → **69.0883**、
#   派生 220 张 33.2120 → **33.4364**；两库合并（BoxIC 优先 + cpnt 补缺键）无额外收益
#   （官方 9 张逐字节相同、派生 220 张 −0.003）⇒ 不合并，按优先级取第一个存在的文件。
#   库文件路径 `STAGE2_BOXIC_LIB` / `STAGE2_CPNT_LIB`，默认走 `assets.libraries()` 的候选链：
#   **包内 `eda-pcb\data\BoxICLibrary_200cases.json`**（1,826,150 B / 360 键，2026-10-14 起随
#   仓库提交）→ 包内 `eda-pcb\data\cpntLibrary.json`（兜底）→ 本机 `work\data\` → `赛题资料\`。
#   旧的写死默认是本机 `work\data\cpntLibrary.json`，**容器里不存在**（`work\` 不进镜像）
#   ⇒ 默认开也白开。
#   改路径的动机见 `eda-pcb\docs\COMPETITION_RULES.md`（「交付包必须带」那条）。
#   关掉（`STAGE2_LIB_PINS=0`）时本文件不会 import `stage2.libpins`，产物与不挂接这条通道时
#   **逐字节一致**。两条规则与唯一性判据见 `stage2/libpins.py` 的模块 docstring。
#   挂接点在 `build_doc` 的 box 引脚路径：规则 A 在「局部采信 + 位置号补位」之后**就地补号**，
#   规则 B 在出组时**改名**。
LIB_PINS = env_on("STAGE2_LIB_PINS", "1")
#: `=1` 时无论 `STAGE2_DBG_PINS` 都打印本通道补/改了几处（诊断用）。
LIB_PINS_DBG = env_on("STAGE2_LIB_PINS_DBG", "0")

# 兜底键的形状（`p000`…），和 `schematic_parser/metrics.py` 的 `FKEY_RE` 一致。
FKEY_RE = re.compile(r"^p\d{3,4}$")
# **兜底键元件（`pNNN`，GT 里是 `∅NNN`）的引脚要不要写进 net 超图**。
# `net_hypergraph` 的 FP = `total_pred_pins - tp`。
#
# ★★ 默认值在 2026-10 官方最新 QA 之后**从 `1` 翻成 `0`**（见
#    `schematic_parser/metrics.py` 的 `net_pin_collapse_nonbox`）。两个口径下的实测：
#
#    * 旧（字面）口径 —— 非 box 元件的 `组件.引脚号` 按字面当净图节点：
#      GT 电源/地的引脚号是**随机内部号**（`∅200` 的脚是 `pin_198`，官方 Q&A
#      「编号是随机生产的」）⇒ 我们写什么号都配不上，写进去就是纯 FP。
#      220 张实测 `=0` → **21.1165**、`=1` → **22.9068（+1.79）**，收益全在 net 线
#      （`net_hypergraph` 0.1357→0.1862，pred 引脚 23743→9676），component/pin 一动不动。
#      ⇒ 那时的结论是「剪掉」。
#    * 新（折叠）口径 —— 非 box 引脚号折成通配 `*`（最新 QA：「非 box 元件 pinnumber
#      不计分，所以 rcl 等在 hygraph 中相当于不区分引脚；gnd/vcc 同理；**输出规范参考
#      10gt**」）：`(p012,*)` 经坐标匹配被 `rewrite_keys` 改写成 `(∅200,*)` **就能配上**，
#      剪掉反而白丢分。官方 9 张实测（同 stage1、同 VLM 目录、`--workers 1`）：
#        **文档级** 剪 53.4673 → 不剪 **55.8307（+2.363）**
#        **融合层** 剪 62.5201 → 不剪 **64.8641（+2.344）**（netH 0.5726 → 0.6394）
#      上界量法：让 GT 与「删掉全部 gnd/v 成员」的 GT 互评，9 张均值 **−4.4793 分**
#      ⇒ 电源节点在折叠口径下确实值钱。
#    * 代价（诚实记录）：万一官方仍是字面口径，同一改动是 **−4.44**（57.0646 vs 61.5084）。
#      缓解见下面的 `NET_DROP_PWRONLY`：只保留「挂在真实成员上」的电源引用。
# ⚠️ 2026-10-13 到 2026-10-14 之间 `STAGE2_DROP_UNKEYED=1` 曾把这批元件整个丢掉，
#    连 `keymap` 都没有条目 ⇒ `:1104` 的 `keymap.get(cid)` 直接落空，引用本来就不会
#    写进超图，本开关只在 `DROP_UNKEYED=0` 时才有意义。
# ★ 2026-10-14 起 `DROP_UNKEYED` 的默认已是 `0`（按书面口径翻回，见文件顶的长注释与
#    `docs/RESULTS.md` §5.20）⇒ **本开关重新变得关键**：`=1` 会把 `pNNN` 的引脚引用
#    从 hyperGraph 里剪掉。
# ★★ 2026-10-14 晚：官方连通性口径用**真实判分产物**标定成功（`docs/RESULTS.md`
#    §5.21）：官方 `connectivity.f1.soft_tp` = Σ 每对网的 pin-set 软 F1，
#    且**只在「refs 按字面比」下**才能逐位复现（case 1/7/2/3 的 soft_tp 与 f1
#    全中）⇒ 上面那条「折叠口径下电源引用值 +2.34」的老读数**不是官方口径**，
#    它依赖「把非 box 引脚号折成 `*`」这个我们自己加的假设。
#    在标定后的尺子上（官方 10 张、VLM、`bbox20/∅坐标匹配/只算 box 引脚/软 Dice/字面 refs`）：
#      `NET_DROP_UNKEYED=0`（保留引用）**73.4368** → `=1`（剪掉）**76.2000（+2.76）**
#      （netH 0.6338 → 0.7162；同一改动在**旧折叠口径**上仍是 −4.42 ⇒ 符号反转）。
#    机理：`∅NNN` 的脚号是随机内部号，我们坐标匹配后键能对上、**脚号永远对不上**
#    ⇒ 这些引用只增大 |P|、拉低 Dice。**默认改为 `1`**。
#    ⚠ 唯一的反证风险：万一新判题机对「坐标匹配上的无名元件」按位置折算脚号
#    （QA 评测 Q1「建议跟赛题统一」留了口子），同一改动约为 **−4.55**
#    （折叠口径 `=0` 80.8191 / `=1` 76.2669）。要回退只需 `STAGE2_NET_DROP_UNKEYED=0`。
#    ★ 2026-10-17 端到端四臂实测（`docs/RESULTS.md` §5.35 二，同一批真实产物）：
#    本开关 `=0` 且 `STAGE2_PIN_DROP_UNKEYED=0` 时，字面尺 **76.2174（−3.08）**、
#    折叠尺 **83.8524（+4.47）**，对照剪掉的是 79.2965 / 79.3847 ⇒ breakeven 概率 **41%**
#    ⇒ 先拿一次提交当探针读新报告的 connectivity，再决定是否翻（翻只需一个 env）。
NET_DROP_UNKEYED = env_on("STAGE2_NET_DROP_UNKEYED", "1")

# ★ **只由电源/地符号组成的孤立网要不要丢**（`STAGE2_NET_DROP_PWRONLY`，默认 `1`）。
#
# 不剪电源引用之后，那些「整条网只有一个 gnd/v 符号」的碎网会从「被剪空 ⇒ 丢弃」
# 变成「有成员 ⇒ 输出」。GT 里**没有**这种网（每个电源符号都挂在一条有真实元件的
# 网络上）⇒ 它们只可能是 FP。官方 9 张、融合层、在 `NET_DROP_UNKEYED=0` 基础上：
#   全部保留 64.8641（netH 0.6394）→ **丢掉孤网 66.3024（netH 0.6800，+1.4383）**
# 同口径下逐张都变好或持平（1 51.56→53.11、3 78.48→81.44、5 58.40→59.71 …）。
# 旧（字面）口径下也顺带把损失从 −4.44 收窄到 −2.64 ⇒ **两个口径下都更优**。
# `=0` 可退回「连孤网一起输出」的行为做对照（`work\tools\diag\_pwrfilter.py` 的 `keep`）。
NET_DROP_PWRONLY = env_on("STAGE2_NET_DROP_PWRONLY", "1")

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
# ★ 边的端点吸附到本网引脚点的半径（`STAGE2_EDGE_SNAP`，默认 **10.0**）。
# 见 `assemble()` 网络循环里的长注释：GT 边有 20.2% 的端点落在本网引脚 5px 内，
# 我们只有 2.1%；机制是 `extract_wires` 挖空元件框 ⇒ 导线断在框边（差 12~20px）。
# ★ 半径定稿 10（官方 10 张，同一批命令里跑，只改这一个变量）：
#   no-vlm：0 → 57.04（导线 0.1023）｜**10 → 59.01（导线 0.2993）**｜20（旧默认）≈ 58.5（0.25）
#   VLM  ：0 → 63.16（0.1085）｜**10 → 65.21（0.3192）**
#   comp/pin/netH 三通道逐位不变 ⇒ 全部增益都来自 `wire_f1`（它权重 0.10）。
# 0 之所以塌，是因为不吸附时端点留在框边（差 12~20px > 官方 5px 阈值）；20 会吸到隔壁枝。
EDGE_SNAP = float(os.environ.get("STAGE2_EDGE_SNAP", "10.0"))
EDGE_JOIN = float(os.environ.get("STAGE2_EDGE_JOIN", "40.0"))     # 端点相接容差（px）
EDGE_ANG = float(os.environ.get("STAGE2_EDGE_ANG", "6.0"))        # 共线夹角容差（度）

# ★ 跨网焊接（`STAGE2_NET_WELD`，默认 **16 = 开**；`=0` 关）。
#   （2026-10-14 转正：官方 10 张 **+1.49/张**、val 19 张 −0.003/张。）
#
# 为什么：`netH` 的失分分层（`eda-pcb/tools/diag/_netloss.py`）里 ③「键号都在、却没并进
# 正确的网」占 **34/487 = 7.0%**，主体是**我们的网太碎**（pred_count 182 > gt_count 160）。
# 其中一批碎片的**边端点相距只有 6~16px** —— 机制是骨架在「净图擦掉文字框」的位置断开，
# 或者两个分叉点各自比真结点短 8~10px（`snap_edge_ends` 的长注释记过同一现象）。
# ⇒ 若两条网各有边的**端点**落在 6~16px 内、连线又**穿过一个文字框**、且**不穿任何元件框**，
#    它们多半本来就是同一条导线 ⇒ 并成一条（只并网，不动引脚取值）。
#    官方 10 张的净效果全部落在 `net_hypergraph`（0.7162 → **0.7587**），
#    元件/引脚通道逐位不变。
#
# 离线读数（在 `work/_fast_out_nudef` 的最终 pred 上按同一规则改写网络后重打分，
# 尺子 = protocol2 即官方软 Dice 口径；`work/_netpin.py --rule gap`）：
#   官方 10 张：base **76.20** → 规则 **77.69（+1.49/张）**，**6 张变好、0 张变差**
#     （case 10 +3.77 / 3 +3.76 / 7 +2.77 / 6 +2.50 / 4 +1.06 / 2 +1.01；1/5/8/9 不动）
#   派生 val 19 张（留出集）：30.33 → 30.27 / 30.18（**−0.005~−0.008/张**，2 好 2 坏）⇒ 中性。
# 端到端（`work/_weld_ab.ps1` → `work/_weld_ab3.out`；默认口径复验 `work/_weld_def.ps1`
#   → `work/_weld_def.out`）：
#   `weld0`（=0）**76.20** 与 `nudef2` 逐位相同 ✓；`weld1`（=16）**77.69**（与离线预测
#   逐位一致）；`wddef`（默认）**78.07**（= 77.69 + case 7 的 VLM 引脚名方差 +3.85/10）；
#   `wddefno`（`--no-vlm`，默认）**71.00**（对比 `nudefno2` 69.52 ⇒ **+1.48**）。
#   ⚠️ 参考噪声：case 7 的 `/pins/U1/pin_4/pinname` 在 `AGND`/`GND` 之间**跨跑次翻转**
#   （VLM 采样方差），单这一处就值 **±0.38** —— 所有 <0.4 的总分差都不可信。
# ⚠️ 三个条件都**必须**，逐个去掉的读数（同两批数据）：
#   * 去掉「穿文字框」：官方 10 仍 +1.49，val 19 **−0.19/张**
#   * 再去掉「不穿元件框」：val 19 **−0.17/张**（相邻两脚的小元件，如 LED，被当成一条线）
#   * 再去掉 `NET_WELD_COS`（即允许平行同向）：val 19 **−0.38/张** —— 大连接器相邻引脚的
#     平行线间距正好 10px，是这条规则最大的假阳来源（cos ≈ +1.00）
#   * 带宽放到 20px：官方 10 反而 **−2.35/张**（开始把不同网并到一起）
#   * `NET_WELD_MIN` 抬到 10（即只焊 10~16px）：官方 10 +0.86/张、val 19 −0.003/张
#     ⇒ 更保守但少赚一半，维持 6。
# 端点吸附（`EDGE_SNAP`）**必须先做**：离线实验就是在吸附后的最终文档上量的。
#
# ★★ 实装时踩到的两个坑（都已在代码里修掉，端到端只拿到 +0.39 就是它们造成的）：
#   ① `build_doc` 在 `out_nets` 那段已经 `pop("_refs")`（:1275），那里之后 `_merge_nets`
#      只读 `_refs` ⇒ 恒为空 ⇒ 「边并了、成员只剩 root 一份」。修法：`_merge_nets` 改成
#      「`_refs` ∪ hyperGraph 解析出的引用」。
#   ② `_merge_nets` **不重建 `hyperGraph`**（沿用 `base = dict(nets[root])` 的那份），
#      在 `out_nets` 之后调用时评测只读 `hyperGraph` ⇒ 修法：`weld_nets` 收尾时用并集
#      引用重建一次。两个坑合起来让管线只拿到 **+0.39**（76.20 → 76.59，而离线是
#      +1.49/张 = 77.69）—— 诊断见 `work/_weldcmp.py` 与 `work/_weldx2.py`
#      （同一份文档上「探针合并」与「A.weld_nets」逐位对比）。
NET_WELD = float(os.environ.get("STAGE2_NET_WELD", "16"))         # 0 = 关；>0 时当「最大间距」用
NET_WELD_MIN = float(os.environ.get("STAGE2_NET_WELD_MIN", "6.0"))
NET_WELD_COS = float(os.environ.get("STAGE2_NET_WELD_COS", "0.5"))
NET_WELD_TPAD = float(os.environ.get("STAGE2_NET_WELD_TPAD", "3.0"))

# ★ 端点吸附（T 型结解算）—— `net_line` 唯一真正有效的杠杆，2026-10 新增。
# `work/tools/diag/_nledge.py` 把某一对配对 net 的 GT 边与我方边**逐条并排**打印，
# 真相是：**每条边的形状都对，端点却普遍偏 7~13px**，而官方 `metrics._seg_distance`
# 的容差只有 **5px**（`_edge_points` 只取首尾两点，`unordered` 取两个朝向里较小的那个
# 端点距离的 **max**）。偏差的两个来源：① 导线有 ~2px 描边，骨架落在其中一条描边上；
# ② 骨架的分叉点没有解算到真正的交叉处，比真结点短 8~10px —— 例：GT 那根横枝
# `(843,1058)-(944,1058)`，我们给的是 `(853,1057)-(957,1057)`，`d=13.0`。
# 把端点推回「同 net 另一条边的中心线」上就能救回一大批。
# `work/tools/diag/_nldup.py` 官方 9 张离线重打分（基线 50.0598 / 导线 0.0649）：
#   snap12 0.1582 / 50.9922 · **snap20 0.1648 / 51.0589** · snap24 0.1617 / 51.0274 ·
#   snap30 0.1536 / 50.9466 · snap40 0.1475 / 50.8854 ⇒ 取 20（再大会吸到隔壁的枝）。
#   19 张档：off 48.7337 → snap20 49.6574（导线 0.0563 → 0.1487）。
# ★ 2026-10-10 官方尺复测（10 张，同一批命令、只改半径，与上面两段不同源）：0 → 导线
#   0.1023（不吸附时端点留在框边，差 12~20px > 官方 5px 阈值）· 12 → 0.2862 ·
#   **10 → 0.2993（最佳）** · 18 → 0.2684 · 26 → 0.2527 ⇒ 半径默认由 20 改成 **10**。
# **没有一张被它打成 0**（逐张都 ≥ 基线），这是它能过门槛的关键 —— 对比 `par3`
# （把同一条导线的两条描边并成一条中心线）netL 0.0868 却把 case 2 打成 0；
# `par3+snap20` 0.1879 / 51.2898 分数更高但同样爆 0 ⇒ **不合并描边、只做吸附**。
EDGE_SNAP = float(os.environ.get("STAGE2_EDGE_SNAP", "10.0"))         # 吸附半径（px），0=关
EDGE_SNAP_ANG = float(os.environ.get("STAGE2_EDGE_SNAP_ANG", "45.0"))  # 只吸到近似垂直的边


def _ang_deg(d1, d2) -> float:
    """两个方向向量的夹角（度）；任一为零向量时返回 180（不焊）。"""
    n1 = math.hypot(d1[0], d1[1]); n2 = math.hypot(d2[0], d2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = (d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def snap_edge_ends(nets: Dict[str, dict], radius: float = None,
                   max_ang: float = None) -> int:
    """把每条边的端点吸附到**同一个 net 里另一条边**的中心线上，返回改动的端点数。

    只吸到与自身近似垂直的邻居（夹角容差 `max_ang`，默认 45°）：平行的邻居是
    **同一条导线的重复描边**，吸过去只会把枝拉歪（实测 `par` 那一路就是负结果）。
    就地改 `nets[*]["edges"]` 里每条边的首/尾点 —— 官方只看这两点。
    """
    radius = EDGE_SNAP if radius is None else radius
    max_ang = EDGE_SNAP_ANG if max_ang is None else max_ang
    cos_lim = math.cos(math.radians(max_ang))
    moved = 0
    for rec in nets.values():
        edges = rec.get("edges") or {}
        keys = [k for k, v in edges.items() if isinstance(v, (list, tuple)) and len(v) >= 2]
        segs = [(edges[k][0], edges[k][-1]) for k in keys]
        for i, k in enumerate(keys):
            seg = edges[k]
            p0, p1 = seg[0], seg[-1]
            for idx, p in ((0, p0), (-1, p1)):
                o = p1 if idx == 0 else p0
                d0 = (o["x"] - p["x"], o["y"] - p["y"])
                if abs(d0[0]) < 1e-9 and abs(d0[1]) < 1e-9:
                    continue
                n0 = math.hypot(d0[0], d0[1])
                best = None
                for j, (a, b) in enumerate(segs):
                    if j == i:
                        continue
                    vx, vy = b["x"] - a["x"], b["y"] - a["y"]
                    L2 = vx * vx + vy * vy
                    if L2 < 1e-12:
                        qx, qy = a["x"], a["y"]
                    else:
                        t = ((p["x"] - a["x"]) * vx + (p["y"] - a["y"]) * vy) / L2
                        t = max(0.0, min(1.0, t))
                        qx, qy = a["x"] + t * vx, a["y"] + t * vy
                    d = math.hypot(p["x"] - qx, p["y"] - qy)
                    if d > radius or (best is not None and d >= best[0]):
                        continue
                    n1 = math.hypot(vx, vy) or 1e-9
                    if abs((d0[0] * vx + d0[1] * vy) / (n0 * n1)) > cos_lim:
                        continue
                    best = (d, qx, qy)
                if best is not None:
                    seg[idx]["x"] = round(best[1], 2)
                    seg[idx]["y"] = round(best[2], 2)
                    moved += 1
    return moved


# ★★ 边的「结点切分 + 去重」后处理（`STAGE2_EDGE_SPLIT` / `STAGE2_EDGE_DEDUP`）。
#
# 病（官方尺 `net_line_f1` 逐边对读，`work\tools\diag\_linecmp.py --case 1`，
# `pred net_5 <-> gt net_9`：GT 40 边 / 我方 46 边 / tp 12）：
#   ① **跨结点合并**：GT 的边就是**源图的导线段** —— 每条竖抽头处都断开
#      （y=1088 那条横线被切成 x=76/194/312/430/577/725/842/944 共 7 段，
#      x=843 的竖母线被每个分支切成 29~30px 的 13 段）；我方 `merge_collinear`
#      把共线段并成一条 873px / 500.9px / 266.1px 的长边 ⇒ 两端点都对不上任何
#      GT 边（到最近 GT 边 372/236/118）。
#   ② **重复边**：同一条墨迹发两遍（`(949,940.5)-(842,941)` 与 `(949,940.5)-(842,939)`），
#      多出来的那些是纯 fp。
#   ③ 端点系统性过冲 ~5px（我方跟到渲染墨迹末端，GT 是源图连接点）—— 本条**不修**。
#
# 药：**只动 `nets[*]["edges"]`**（不动 `hyperGraph`/引脚 ⇒ 元件/引脚/netH 三个通道
# 逐位不变），先**去重**（共线孪生边不能当切点，否则互相乱切），再**在切点处切开**：
# 切点 = 同一条网里另一条边的端点，落在本线段**严格内部**（`--tol-in` 内，两端各留 2%），
# 且该邻居与本段**不近似共线**（|cos| ≤ `STAGE2_EDGE_SPLIT_COL`）。
#
# 读数（官方 10 张，纯后处理同一份 `_Pred.json` 再打分，`work\tools\diag\_linesplit.py`）：
#   基线 net_line **0.3567**（tp 270 / fp 598 / fn 369）总分 79.0781
#   只去重 0.3638（tp 270 / fp 558）79.1497
#   只切分 0.3781（tp 311 / fp 651 / fn 328）**79.2922**
#   切分+去重 **0.3862**（tp 311 / fp 610 / fn 328）**79.3735**（+0.2954）
#   参数扫描：`tol-in` 2.0 → 0.3843｜6.0 → 0.3854（3.0 最好）·`dedup-tol` 1.0 → 0.3843
#   （2.0 最好）·`col-tol` 0.80 与 0.90 逐位相同（去重后已无共线孪生）。
#   **反例（别重试）**：再加「把 1~3px 缝隙的共线链桥接成长边」→ 79.2014（tp 311→283，
#   GT 自己就带着这些接缝，桥接反而把能配上的碎边弄丢）。
#   **顺带**：GT 边的 fn 几乎全在「已配对的网」里（639 条里只有 4 条来自没配上的网）⇒
#   这一路只该修边几何，别去动网络分组（分组是 netH 的事）。
EDGE_SPLIT = float(os.environ.get("STAGE2_EDGE_SPLIT", "3.0"))    # 切点容差 px，0 = 关
EDGE_SPLIT_COL = float(os.environ.get("STAGE2_EDGE_SPLIT_COL", "0.90"))  # 共线邻居不算切点
EDGE_DEDUP = float(os.environ.get("STAGE2_EDGE_DEDUP", "2.0"))    # 去重容差 px，0 = 关
# ★ 第二遍去重（**切分之后**再跑一次）：切分会把两条只部分重合的长边切成同一批小段，
#   于是同一张网里留下几何一模一样的两条边 —— 第一遍必须在切分前跑（共线孪生不能当
#   切点），所以这批残渣只能在切分后清。0 = 关。官方 10 张：875 条边、net_line
#   0.3862 → 0.3993（tp 311 → 311、fp 610 → 564），总分 79.3735 → 79.5042（+0.1307）。
EDGE_DEDUP2 = float(os.environ.get("STAGE2_EDGE_DEDUP2", "2.0"))  # 第二遍去重容差 px，0 = 关

# ★★ 自由端回缩（`STAGE2_EDGE_TRIM`，默认 3.0 px，0 = 关）。
#
# 病（上面第 ③ 条「端点系统性过冲 ~5px」的量化，`work\_wiretrim.py` / `_linegap.py`）：
#   官方 10 张、交付口径：GT 边 639 / 我方边 874，tp 312 / fp 562 / fn 327（导线 F1 0.4008）；
#   被 5px 门挡掉的近失里 **5~10px 桶有 184 条**，逐例看就是「我方比 GT 多伸出去一点」
#   （GT(843,911)->(944,911) len 101 vs 我方 (949,910.8)->(843,910) len 106 —— 左端到点，
#   右端多 5px，d=5.0 刚好卡在门外；`STAGE1_PIN_OUT_BOX` 恒 12 与 GT 逐案逐侧的外伸不同）。
#
# 药：只把**自由端**（同网内没有别的边贴在这个端点上；结点/共线接缝一律不动）沿边方向
#   朝内回缩 delta px。离线扫描（`python -X utf8 work\_wiretrim.py defconfirm work\_cases10gt.txt 0,2,3,4,5`）：
#     官方 10：d=0 **79.1419**（导线 0.4008，tp/fp/fn 312/562/327）
#              d=2 **79.2969**（0.4163，325/549/314）· d=3 **79.2965**（0.4162，323/551/316）
#              d=4 79.0255 · d=5 78.8356 · d=6 78.7179 ⇒ **3 是拐点，≥4 开始把对的也拉歪**
#     派生 19（臂 cfv，`work/_cases_v4val.txt`）：d=0 30.9589 → d=2 31.0290 → d=3 **31.0784**
#   「所有端」一律回缩在官方 10 上大亏（d=2 79.1148 / d=4 77.9591）⇒ 必须只动自由端。
# 只动 `edges` 首尾两点 ⇒ 元件/引脚/netH 三个通道逐位不变（与上面的切分/去重同性质）。
#
# ★ 端到端 A/B（`work\_trim_ab.ps1`，官方 10 张、VLM 开、`fast.py` 的 protocol2 均值）：
#   `STAGE2_EDGE_TRIM=0` **79.14**（导线 0.4008）＝ 旧基线逐字节相同（11 文件同 10、只差 run.log）
#   `STAGE2_EDGE_TRIM=3` **79.30**（导线 **0.4162**）· 同配置第二次 79.66（那次的引脚 0.7633 是
#   VLM 阶段的运气，第三次复核回落 0.7452）⇒ **可归因的是导线通道 +0.0154 F1 = +0.154 分**，
#   总分的 run-to-run 抖动 ±0.2 来自 VLM 阶段（`trim0` 与 `defconfirm` 逐字节相同 → 关掉时零差异）。
#   派生 19 张（臂 `cfv`/`trimv0`/`trimv`）：导线 0.0727/0.0729 → **0.0847**（+0.12 分，方向同）。
EDGE_TRIM = float(os.environ.get("STAGE2_EDGE_TRIM", "3.0"))  # 自由端回缩 px，0 = 关

# ★★ 端点外推（`STAGE2_EDGE_EXT`，默认 20 px，0 = 关）。
#
# 病（`work\_linelab.py` + `work\_linesample.py`，官方 10 交付口径）：
#   配对网内 GT 边 635 / 配上 321；未配上的 303 条里 **209 条是「一端已 ≤3px、另一端差 9.5px（中位）」
#   （p90 16.0，≤12px 的 147 条）**—— 上面的 `EDGE_TRIM` 只治「过冲」，治不了「不够长」。
#   `work\_lineoracle.py` 量出「端点对端点对齐」的上界：netL 0.4243 → 0.7323（**+3.08 分**）。
# 药：端点沿自身轴向**外推**，逐 px 试探，**只有外推点落到同网另一条边 ≤ `STAGE2_EDGE_EXT_HIT` 内**才生效；
#   只动 `edges` 首尾两点 ⇒ 元件/引脚/netH 三通道逐位不变。
# 读数（官方 10，`work\_edgelab.py` 纯后处理）：`ext:20` 导线 0.4137 → **0.4250**（总分 +0.12）；
#   220 张跨池（`d220known` 臂）：`10GT` 76.21 → **76.30**（导线 0.4077 → 0.4171）、
#   `golden10` 33.72 → 33.72、派生 196 张 27.42 → 27.42、`jlc` 99 张 32.37 → 32.37 ⇒ **跨池零负**。
EDGE_EXT = float(os.environ.get("STAGE2_EDGE_EXT", "20"))          # 端点外推上限 px，0 = 关
EDGE_EXT_HIT = float(os.environ.get("STAGE2_EDGE_EXT_HIT", "3.0"))  # 外推点要落到同网他边这么多 px 内

# ★★ 近共线重叠边合并（`STAGE2_EDGE_OVERLAP`，默认 10 度，0 = 关）。
#
# 病：骨架分叉会把**同一条墨迹**发成两条部分重叠的边（case4 实测
#   `(1689,792)-(1627,792)`、`(1619,792)-(1708,792)`、`(1541,792)-(1612,792)`）；
#   官方口径一对一配对 ⇒ 只能配上一条，其余整条算 fp。`EDGE_DEDUP`(2px) 只去「几何几乎重合」的，
#   部分重叠的漏网（`work\_linelab.py`：配对网里我们 869 条边只有 321 条被用上）。
# 药：同网内两条边夹角 ≤ ang、端点到对方直线 ≤ off、投影重叠 ≥ min_ov ⇒ 合成并集跨度的一条。
# 读数（官方 10，纯后处理）：`overlap:10,4,0.35` 导线 0.4137 → **0.4251**；与 `ext` 叠加 **0.4367**
#   （总分 **80.28**）| 220 张跨池：`10GT` 76.21 → **76.34**、派生 196 张 27.42 → 27.42（导线 0.0463 → 0.0458）、
#   `jlc` 32.37 → 32.36 ⇒ 派生池 −0.01（噪声量级）。
EDGE_OVERLAP = float(os.environ.get("STAGE2_EDGE_OVERLAP", "10"))         # 近共线夹角容差(度)，0 = 关
EDGE_OVERLAP_OFF = float(os.environ.get("STAGE2_EDGE_OVERLAP_OFF", "4.0"))  # 端点到对方直线容差 px
EDGE_OVERLAP_MIN = float(os.environ.get("STAGE2_EDGE_OVERLAP_MIN", "0.35"))  # 投影重叠比例下限

# ★★ 自由端外推（`STAGE2_EDGE_EXT_FREE`，默认 8 px，0 = 关）。
#
# 病（`work\_linesample.py`，官方 10）：303 条未配上的 GT 边里 **209 条是「截断」**，
#   再拆短端邻居：**引脚 95 / 什么都没有 114**。那 114 条的短端 5px 内既无引脚也无同网他边
#   ⇒ `extend_edge_ends` 的「必须落到同网他边 3px 内」闸门恰好把它们全挡在外面，
#   而 GT 的端点就在几像素之外（墨迹还在，只是骨架在那里断了）。
# 药：**只对自由端**（附近 `pin_r` 内无本网引脚、`edge_r` 内无同网他边）沿自身轴直接外推 `radius` px。
#   引脚旁的端点**不动**（那里线本来就该停在引脚上）；只动 `edges` 首尾两点 ⇒ 三通道逐位不变。
# 读数（官方 10，`work\_edgelab.py` 纯后处理，基线 80.04 / 导线 0.4137）：
#   `extfree:8,free` 单独 **80.17**（0.4264）；与 `ext:20;overlap:10,4,0.35` 叠加（**extfree 必须最先**）
#   **80.46（导线 0.4556）**；顺序反过来（`ext;overlap;extfree`）只剩 80.17 —— 外推会把已对齐的端点再推过头。
#   半径扫描（叠加后）：6 → 80.43、**8 → 80.46**、10 → 80.39、15 → 80.19、20 → 79.96、30 → 79.78。
#   无闸门版（`extfree:5,all`，1789 个端点）**77.90**（导线 0.1991）⇒ 自由端这道闸是全部收益的来源。
# ★ **未采纳（默认 0）**：真实管线端到端量下来它是「官方 10 +0.21、val18 −0.2（30.01 → 29.79）、
#   离线 golden10 −0.34（0.1044 → 0.0711）」——盲推会越过真端点在派生池上净亏，
#   而这些收益已被下面的**证据外推**（`STAGE2_EDGE_EXT_ANY`）以更小的跨池代价拿走。
EDGE_EXT_FREE = float(os.environ.get("STAGE2_EDGE_EXT_FREE", "0"))          # 自由端外推 px，0 = 关（未采纳）
EDGE_EXT_FREE_PIN = float(os.environ.get("STAGE2_EDGE_EXT_FREE_PIN", "5.0"))   # 引脚这么近就不算自由端
EDGE_EXT_FREE_EDGE = float(os.environ.get("STAGE2_EDGE_EXT_FREE_EDGE", "3.0"))  # 同网他边这么近就不算自由端


def extend_free_ends(nets: Dict[str, dict], pins: Dict[str, dict], radius: float = None,
                     pin_r: float = None, edge_r: float = None) -> int:
    """自由端沿自身轴向外推 `radius` px（引脚旁 / 贴着同网他边的端点不动）；返回改动端点数。"""
    radius = EDGE_EXT_FREE if radius is None else radius
    pin_r = EDGE_EXT_FREE_PIN if pin_r is None else pin_r
    edge_r = EDGE_EXT_FREE_EDGE if edge_r is None else edge_r
    if radius <= 0:
        return 0
    moved = 0
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = [(k, _seg_ends(v)) for k, v in edges.items()]
        segs = [(k, e) for k, e in segs if e is not None]
        if not segs:
            continue
        pp: List[Tuple[float, float]] = []
        for c, pn in _hg_refs(rec.get("hyperGraph", "")):
            _p = ((pins.get(c) or {}).get("pin_%s" % pn) or {}).get("point")
            if isinstance(_p, dict) and "x" in _p and "y" in _p:
                pp.append((float(_p["x"]), float(_p["y"])))
        for k, ends in segs:
            (ax, ay), (bx, by) = ends
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            seg = edges[k]
            for idx, (px, py), s in ((0, (ax, ay), -1.0), (-1, (bx, by), 1.0)):
                if pp and min(math.hypot(px - qx, py - qy) for qx, qy in pp) <= pin_r:
                    continue
                if _pt_seg_dist(px, py, segs, skip=k) <= edge_r:
                    continue
                seg[idx]["x"] = round(px + s * ux * radius, 2)
                seg[idx]["y"] = round(py + s * uy * radius, 2)
                moved += 1
    return moved


# ★★ 证据外推：跨网 / 引脚（`STAGE2_EDGE_EXT_ANY`，默认 25 px，0 = 关）。
#
# 病：`extend_edge_ends` 的闸门只认**同网**他边 ⇒ 截断的 114 条「短端什么都没有」里，
#   凡是在**别网**的边或引脚附近的那部分（交叉点、汇合点、引脚根部）都被漏掉了。
# 药：逐 px 外推，落点只要贴到**全图任何一条边**（本边除外）或**任何一个引脚点** ≤ `HIT` 就生效。
#   只动 `edges` 首尾两点 ⇒ 元件/引脚/netH 三通道逐位不变。
# 读数（官方 10，`work\_edgelab.py` 纯后处理，基线 80.04 / 导线 0.4137，与 `ext:20;overlap:10,4,0.35` 叠加）：
#   `extany:20,3` → 80.54（0.4632）；半径/容差细扫：`24,2` **80.73（0.4824）**、`25,2` 80.72（0.4808）、
#   `26,2` 80.72、`22,3` 80.70、`25,1.5` 80.67、`30,2` 80.64、`25,2.5` 80.65、`18,2` 80.52；
#   关掉引脚（`25,2,1,0`）80.31、只要边（`25,2,0,1`）80.46 ⇒ **引脚证据值 +0.26、边证据值 +0.18，两者叠加 +0.44**。
# 跨池（纯后处理，220 张 `d220known` 臂；**只有这条仪器是确定性的** —— 端到端派生池有 ±0.3 的 LLM 噪声）：
#   `10GT` 76.21 → **76.90**（导线 0.4077 → 0.4764）｜`golden10` 33.72 → **33.56**（导线 −0.0155）｜
#   派生 196 张 27.42 → 27.40｜`jlc` 99 张 32.37 → **32.38**；拆开看：只要引脚 `10GT` 76.37 / `golden10` 33.63、
#   只要边 `10GT` 76.30 / `golden10` **33.73（+0.01）** ⇒ 代价集中在「引脚那半 × golden10」。
# 端到端（真实管线，`-lean`，`STAGE2_EDGE_EXT_FREE=0` 对照）：官方 10 **80.31 → 80.73**（导线 0.4402 → 0.4821，
#   comp/pin/netH 逐位不变）；golden10 35.98 → 36.42（其中 netH 的 0.3164 → 0.3316 属 LLM 噪声，只有导线是真变化）；
#   val18 29.99 → 29.91（±0.2 噪声内）。
EDGE_EXT_ANY = float(os.environ.get("STAGE2_EDGE_EXT_ANY", "25"))       # 证据外推上限 px，0 = 关
EDGE_EXT_ANY_HIT = float(os.environ.get("STAGE2_EDGE_EXT_ANY_HIT", "2.0"))  # 落点要贴到几何这么多 px 内

# ★★ 自由端吸附到引脚（`STAGE2_PIN_SNAP`，默认 0 = 关）。
#   病（RESULTS §5.74 三）：netH ③ 桶里 6 个「不在任何网」的引脚，其最近边端点在
#   17.7~41.0px 外、夹角 0.4~31.3°；`extend_ends_to_evidence` 的判据是「外推点落在引脚
#   2.0px 内」⇒ 走那么远后横向偏 5~9px，判不中（实测把 EXT_ANY 放大到 45/70，netH 逐位不变）。
#   药：换成「引脚到边所在直线的垂距 ≤ `STAGE2_PIN_SNAP_PERP`」判据，并把端点**直接搬到引脚点**。
#   ★ 读数（`work/_psnap_ab.ps1`，官方 10 张，参考臂 psref = 80.88 / 导线 0.4975 / tp 329）：
#     ps45p8（R=45 PERP=8）**80.53 / 0.4626 / tp 304**、ps45p5 同 80.53 / 0.4626 / 304、
#     ps70p8（R=70）80.54 / 0.4636 / 304 ⇒ **判负（−0.34~−0.35，导线 tp −25）、comp/pin/netH 逐位不变**。
#     机理：GT 的线端落在**元件引脚引线尖**上（case 9 `R1` 的 GT 端 (650,637) 比我们的端点 (649,653)
#     多出 ~16px），把自由端搬到**引脚锚点**反而离开 GT 的真端点，而 `line_tol = 5.0`
#     （`schematic_parser/metrics.py:88`）一超就翻 tp ⇒ 端点「差不多对」时**不动**才是最优。
#     ⇒ 默认 0（关），机制留着；见 RESULTS §5.76。
PIN_SNAP = env_on("STAGE2_PIN_SNAP", "0")
PIN_SNAP_R = float(os.environ.get("STAGE2_PIN_SNAP_R", "45"))         # 向前搜索半径 px
PIN_SNAP_PERP = float(os.environ.get("STAGE2_PIN_SNAP_PERP", "8"))    # 允许的垂距 px


def extend_ends_to_evidence(nets: Dict[str, dict], pins: Dict[str, dict], radius: float = None,
                            hit: float = None) -> int:
    """端点沿自身轴外推，落点贴到**任何边**（本边除外）或**任何引脚点** ≤ `hit` 才生效；返回改动端点数。"""
    radius = EDGE_EXT_ANY if radius is None else radius
    hit = EDGE_EXT_ANY_HIT if hit is None else hit
    if radius <= 0:
        return 0
    allsegs: List[Tuple[str, str, tuple]] = []
    for nk, rec in nets.items():
        for k, v in (rec.get("edges") or {}).items():
            e = _seg_ends(v)
            if e is not None:
                allsegs.append((nk, k, e))
    ppt: List[Tuple[float, float]] = []
    for _c, _pn in (pins or {}).items():
        for _pv in (_pn or {}).values():
            _p = (_pv or {}).get("point")
            if isinstance(_p, dict) and "x" in _p and "y" in _p:
                ppt.append((float(_p["x"]), float(_p["y"])))
    moved = 0
    for nk, rec in nets.items():
        edges = rec.get("edges") or {}
        segs = [(k, _seg_ends(v)) for k, v in edges.items()]
        segs = [(k, e) for k, e in segs if e is not None]
        for k, ends in segs:
            (ax, ay), (bx, by) = ends
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            seg = edges[k]
            for idx, (px, py), s in ((0, (ax, ay), -1.0), (-1, (bx, by), 1.0)):
                for step in range(1, int(radius) + 1):
                    qx, qy = px + s * ux * step, py + s * uy * step
                    ok = False
                    for nk2, k2, e2 in allsegs:
                        if nk2 == nk and k2 == k:
                            continue
                        (cx, cy), (ex_, ey) = e2
                        vx, vy = ex_ - cx, ey - cy
                        L2 = vx * vx + vy * vy
                        if L2 <= 1e-9:
                            d = math.hypot(qx - cx, qy - cy)
                        else:
                            t = max(0.0, min(1.0, ((qx - cx) * vx + (qy - cy) * vy) / L2))
                            d = math.hypot(qx - (cx + t * vx), qy - (cy + t * vy))
                        if d <= hit:
                            ok = True
                            break
                    if not ok:
                        for qx2, qy2 in ppt:
                            if math.hypot(qx - qx2, qy - qy2) <= hit:
                                ok = True
                                break
                    if ok:
                        seg[idx]["x"] = round(qx, 2)
                        seg[idx]["y"] = round(qy, 2)
                        moved += 1
                        break
    return moved


def _pt_seg_dist(px: float, py: float, segs: List[Tuple[str, tuple]], skip: str = "") -> float:
    """点到本网其它边的最短距离（`skip` 那条不算）。"""
    best = float("inf")
    for k2, e2 in segs:
        if k2 == skip:
            continue
        (cx, cy), (ex_, ey) = e2
        vx, vy = ex_ - cx, ey - cy
        L2 = vx * vx + vy * vy
        if L2 <= 1e-9:
            d = math.hypot(px - cx, py - cy)
        else:
            t = max(0.0, min(1.0, ((px - cx) * vx + (py - cy) * vy) / L2))
            d = math.hypot(px - (cx + t * vx), py - (cy + t * vy))
        if d < best:
            best = d
    return best


# ★★ L/T 拐角合拢（`STAGE2_EDGE_CORNER`，默认 12 px，0 = 关）。
# 病：骨架化在**结点**处解不出真交叉位置 —— 一个 T/十字的每一条腿都比真结点**短 8~13 px**
#   （见 `STAGE2_EDGE_SPLIT` 长注释与 `_linelab.py` 的「截断」类：一端已 ≤3px、另一端中位 9.5px）。
# 药：两条同网边各自的**轴延长线**若在 ≤ `STAGE2_EDGE_CORNER` px 内相交，且交点落在对方跨度内
#   （或离对方某个端点 ≤ 同值），就把端点搬到交点 —— 一次把**腿们一起**推到真结点上。
# 位置：**必须跑在 `extend_edge_ends`/`extend_ends_to_evidence` 之前**（离线实测顺序反了收益从 +0.17 掉到 -0.10：
#   先把端点搬到真结点，后面两个外推算子才有正确的起点）。
# 实测（官方 10，纯后处理，叠在 `ext:20;overlap:10,4,0.35;extany:25,2,1,1` 之上）：
#   对照 80.58 / 导线 0.4676 → **先 corner 80.75 / 0.4840**；放最后 80.48 / 0.4570；放中间 80.50 / 0.4593。
# 只动 `edges` 首尾两点 ⇒ 元件/引脚/netH 三通道逐位不变。
EDGE_CORNER = float(os.environ.get("STAGE2_EDGE_CORNER", "12"))          # 拐角合拢半径 px，0 = 关
EDGE_CORNER_INT = float(os.environ.get("STAGE2_EDGE_CORNER_INT", "0.05"))  # 交点落在对方跨度外的容差(比例)
EDGE_CORNER_CROSS = os.environ.get("STAGE2_EDGE_CORNER_CROSS", "1") not in ("0", "", "no", "off")


def snap_ends_to_crossings(nets: Dict[str, dict], radius: float = None,
                           interior: float = None, cross: bool = None) -> int:
    """把端点搬到另一条边的轴延长线交点（治骨架在 T/十字结点处短 8~13px）；返回改动的端点数。

    `cross=True`（默认）时横向边取自**全图所有网** —— 结点另一半被切成别的网时也能合拢；
    `cross=False` 只在**本网**内找横向边（旧行为，离线与跨池读数都略低）。
    只改**被搬端点自己那条边**，横向边与 `hyperGraph` 一律不碰。
    """
    radius = EDGE_CORNER if radius is None else radius
    interior = EDGE_CORNER_INT if interior is None else interior
    cross = EDGE_CORNER_CROSS if cross is None else cross
    if radius <= 0:
        return 0
    allsegs: List[Tuple[str, str, tuple]] = []
    ref: Dict[Tuple[str, str], list] = {}
    for nk, rec in nets.items():
        edges = rec.get("edges") or {}
        for k, v in edges.items():
            e = _seg_ends(v)
            if e is None:
                continue
            allsegs.append((nk, k, e))
            ref[(nk, k)] = v
    if len(allsegs) < 2:
        return 0
    moved = 0
    newpts: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
    for i, (nk, k, ends) in enumerate(allsegs):
        (ax, ay), (bx, by) = ends
        for idx, (px, py), (qx, qy) in ((0, (ax, ay), (bx, by)), (1, (bx, by), (ax, ay))):
            ux, uy = px - qx, py - qy                 # ★ 朝**外**（从边体指向端点外）
            L = math.hypot(ux, uy)
            if L < 1e-6:
                continue
            ux, uy = ux / L, uy / L
            for j, (nk2, k2, ends2) in enumerate(allsegs):
                if j == i:
                    continue
                if not cross and nk2 != nk:
                    continue
                (cx, cy), (ex_, ey_) = ends2
                vx, vy = ex_ - cx, ey_ - cy
                cr = ux * vy - uy * vx
                if abs(cr) < 1e-6:
                    continue
                t = ((cx - px) * vy - (cy - py) * vx) / cr
                if not (0.0 < t <= radius):
                    continue
                X = (px + t * ux, py + t * uy)
                L2 = vx * vx + vy * vy
                tj = ((X[0] - cx) * vx + (X[1] - cy) * vy) / (L2 if L2 > 1e-9 else 1.0)
                ok = -interior <= tj <= 1.0 + interior
                if not ok:
                    for e in ((cx, cy), (ex_, ey_)):
                        dj = math.hypot(e[0] - X[0], e[1] - X[1])
                        if 0.0 < dj <= radius:
                            ok = True
                            break
                if not ok:
                    continue
                newpts[(nk, k, idx)] = X
                moved += 1
                break
    for (nk, k, idx), X in newpts.items():
        seg = ref.get((nk, k))
        if not seg or len(seg) < 2:
            continue
        seg[idx]["x"] = round(X[0], 2)
        seg[idx]["y"] = round(X[1], 2)
    return moved


def extend_edge_ends(nets: Dict[str, dict], radius: float = None,
                     hit: float = None) -> int:
    """端点沿自身轴向外推，**只有外推点落到同网另一条边 `hit` px 内才生效**；返回改动端点数。"""
    radius = EDGE_EXT if radius is None else radius
    hit = EDGE_EXT_HIT if hit is None else hit
    if radius <= 0:
        return 0
    moved = 0
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = [(k, _seg_ends(v)) for k, v in edges.items()]
        segs = [(k, e) for k, e in segs if e is not None]
        for k, ends in segs:
            (ax, ay), (bx, by) = ends
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            seg = edges[k]
            for idx, (px, py), s in ((0, (ax, ay), -1.0), (-1, (bx, by), 1.0)):
                for step in range(1, int(radius) + 1):
                    qx, qy = px + s * ux * step, py + s * uy * step
                    found = False
                    for k2, e2 in segs:
                        if k2 == k:
                            continue
                        (cx, cy), (ex_, ey) = e2
                        vx, vy = ex_ - cx, ey - cy
                        L2 = vx * vx + vy * vy
                        if L2 <= 1e-9:
                            d = math.hypot(qx - cx, qy - cy)
                        else:
                            t = max(0.0, min(1.0, ((qx - cx) * vx + (qy - cy) * vy) / L2))
                            d = math.hypot(qx - (cx + t * vx), qy - (cy + t * vy))
                        if d <= hit:
                            found = True
                            break
                    if found:
                        seg[idx]["x"] = round(qx, 2)
                        seg[idx]["y"] = round(qy, 2)
                        moved += 1
                        break
    return moved


def snap_ends_to_pins(nets: Dict[str, dict], pins: Dict[str, dict],
                      radius: float = None, perp: float = None) -> int:
    """**自由端**沿自身轴向前 `radius` 内有引脚、且该引脚到边所在直线的**垂距** ≤ `perp`
    时，把端点**吸附到引脚点**；返回改动端点数。

    依据（RESULTS §5.74 三 / §5.76）：`extend_ends_to_evidence` 要求**外推的那个点**
    落在引脚点 2.0px 内，而 §5.74 的 5 个「不在任何网」的引脚是 17.7~41.0px、夹角 0.4~31.3°
    ⇒ 走那么远后横向偏 5~9px，**永远判不中**（放大 `EDGE_EXT_ANY` 到 45/70 实测对 netH 零效果）。
    这里换成「引脚到边所在直线的垂距」判据 + 直接把端点搬到引脚点。

    ★★ 别再回头做这个方向（RESULTS §5.125 已给**离线神谕上界**）：在线性判分里
    「GT 端落在元件**引线尖**上、而不是引脚锚点」是系统性的——`_nlcrop.py` 裁图实证
    （case 10 `L1`：GT 端 x=987，我方 x=1003.4，**过伸 16px**；我方 x=1003 已在引线**里面**）。
    `tools/diag/_nlpinoracle.py` 在**已交付产物**上按本函数的语义（含「允许回缩到我方端点
    **身后** backoff px」的加强版）扫了 40 组 (radius 10~45 / perp 3~5 / backoff 0~25)：
    **delta 全为负，−0.0249 ~ −0.0436**（基线 0.5780；`mv` 294~360 个端点被搬动）。
    故本函数**保持默认关**（`STAGE2_PIN_SNAP=0`）。
    """
    radius = PIN_SNAP_R if radius is None else radius
    perp = PIN_SNAP_PERP if perp is None else perp
    if not PIN_SNAP or radius <= 0:
        return 0
    ppt: List[Tuple[float, float]] = []
    for _c, _pn in (pins or {}).items():
        for _pv in (_pn or {}).values():
            _p = (_pv or {}).get("point")
            if isinstance(_p, dict) and "x" in _p and "y" in _p:
                ppt.append((float(_p["x"]), float(_p["y"])))
    if not ppt:
        return 0
    moved = 0
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = [(k, _seg_ends(v)) for k, v in edges.items()]
        segs = [(k, e) for k, e in segs if e is not None]
        if not segs:
            continue
        verts = [p for _k, e in segs for p in e]
        for k, ((ax, ay), (bx, by)) in segs:
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            seg = edges[k]
            if not isinstance(seg, list) or len(seg) < 2:
                continue
            for idx, (px, py) in ((0, (ax, ay)), (1, (bx, by))):
                # 只动自由端：同网别的边的端点贴在这个点上 ⇒ 是结点，不动
                if any(math.hypot(px - qx, py - qy) <= 3.0
                       for (qx, qy) in verts if (qx, qy) != (px, py)):
                    continue
                sx, sy = (ux, uy) if idx == 1 else (-ux, -uy)
                best = None
                for (qx, qy) in ppt:
                    vx, vy = qx - px, qy - py
                    t = vx * sx + vy * sy
                    if t <= 0.0 or t > radius:
                        continue
                    if abs(vx * (-sy) + vy * sx) > perp:
                        continue
                    if best is None or t < best[0]:
                        best = (t, qx, qy)
                if best is None:
                    continue
                seg[idx]["x"] = round(best[1], 2)
                seg[idx]["y"] = round(best[2], 2)
                moved += 1
    return moved


def merge_overlapping_edges(nets: Dict[str, dict], ang: float = None, off: float = None,
                            min_ov: float = None) -> int:
    """同网内近共线且投影重叠的边合成一条（并集跨度）；返回被合并掉的边数。"""
    ang = EDGE_OVERLAP if ang is None else ang
    off = EDGE_OVERLAP_OFF if off is None else off
    min_ov = EDGE_OVERLAP_MIN if min_ov is None else min_ov
    if ang <= 0:
        return 0
    cos_lim = math.cos(math.radians(ang))
    dropped = 0
    for rec in nets.values():
        edges = rec.get("edges") or {}
        live = {}
        for k, v in edges.items():
            e = _seg_ends(v)
            if e is not None:
                live[k] = e
        keys = list(live)
        for ii, ki in enumerate(keys):
            if ki not in live:
                continue
            for kj in keys[ii + 1:]:
                if kj not in live or ki not in live:
                    continue
                (ax, ay), (bx, by) = live[ki]
                (cx, cy), (dx_, dy_) = live[kj]
                ux, uy = bx - ax, by - ay
                Li = math.hypot(ux, uy)
                vx, vy = dx_ - cx, dy_ - cy
                Lj = math.hypot(vx, vy)
                if Li < 1e-6 or Lj < 1e-6:
                    continue
                if abs((ux * vx + uy * vy) / (Li * Lj)) < cos_lim:
                    continue
                ux, uy = ux / Li, uy / Li
                ts, offs = [], []
                for (px, py) in ((cx, cy), (dx_, dy_)):
                    ts.append(((px - ax) * ux + (py - ay) * uy) / Li)
                    offs.append(abs(-(px - ax) * uy + (py - ay) * ux))
                if max(offs) > off:
                    continue
                lo, hi = min(ts), max(ts)
                ov = min(hi, 1.0) - max(lo, 0.0)
                if ov <= 0 or ov / max(1e-9, min(1.0, Lj / Li)) < min_ov:
                    continue
                a0, b0 = min(0.0, lo), max(1.0, hi)
                edges[ki][0]["x"] = round(ax + a0 * Li * ux, 2)
                edges[ki][0]["y"] = round(ay + a0 * Li * uy, 2)
                edges[ki][-1]["x"] = round(ax + b0 * Li * ux, 2)
                edges[ki][-1]["y"] = round(ay + b0 * Li * uy, 2)
                live[ki] = ((edges[ki][0]["x"], edges[ki][0]["y"]),
                            (edges[ki][-1]["x"], edges[ki][-1]["y"]))
                (ax, ay), (bx, by) = live[ki]
                vx2, vy2 = bx - ax, by - ay
                Li = math.hypot(vx2, vy2) or 1e-9
                ux, uy = vx2 / Li, vy2 / Li
                edges.pop(kj, None)
                del live[kj]
                dropped += 1
    return dropped


def _seg_ends(seg):
    """一条边的 (首点, 尾点) 坐标元组；不合法返回 None。"""
    if not isinstance(seg, (list, tuple)) or len(seg) < 2:
        return None
    a, b = seg[0], seg[-1]
    try:
        return ((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])))
    except (TypeError, KeyError, ValueError):
        return None


def _pt_close(p, q, tol: float) -> bool:
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def _cut_point(p, a, b, tol: float):
    """`p` 是否落在 `a-b` 的严格内部（两端各留 2%），且离直线 ≤ tol。"""
    if _pt_close(p, a, tol * 2) or _pt_close(p, b, tol * 2):
        return False
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 <= 1e-9:
        return False
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2
    if not (0.02 < t < 0.98):
        return False
    return abs(a[0] + t * vx - p[0]) <= tol and abs(a[1] + t * vy - p[1]) <= tol


def split_dedup_edges(nets: Dict[str, dict], split_tol: float = None,
                      dedup_tol: float = None,
                      col_tol: float = None,
                      dedup2_tol: float = None) -> Dict[str, int]:
    """就地做「去重 + 结点切分 + 再去重」（见上面长注释），返回 {dup, dup2, split, added}。"""
    split_tol = EDGE_SPLIT if split_tol is None else split_tol
    dedup_tol = EDGE_DEDUP if dedup_tol is None else dedup_tol
    col_tol = EDGE_SPLIT_COL if col_tol is None else col_tol
    dedup2_tol = EDGE_DEDUP2 if dedup2_tol is None else dedup2_tol
    st = {"dup": 0, "dup2": 0, "split": 0, "added": 0}
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = []                      # [(a, b, 原值)]
        for _k, _v in edges.items():
            e = _seg_ends(_v)
            if e is not None:
                segs.append((e[0], e[1], _v))
        if not segs:
            continue
        st0 = (st["dup"], st["split"], st["dup2"])
        # —— ① 去重：两端点（正反皆算）都与已保留的某条重合就丢
        kept: List[tuple] = []
        if dedup_tol > 0:
            for (a, b, _v) in segs:
                dup = False
                for (c, d, _w) in kept:
                    if (_pt_close(a, c, dedup_tol) and _pt_close(b, d, dedup_tol)) or \
                       (_pt_close(a, d, dedup_tol) and _pt_close(b, c, dedup_tol)):
                        dup = True
                        break
                if dup:
                    st["dup"] += 1
                else:
                    kept.append((a, b, _v))
        else:
            kept = list(segs)
        # —— ② 切分：邻居端点落在本段内部且不共线 ⇒ 在此切开
        out: List[tuple] = []          # [(a, b, 原值|None)]
        if split_tol > 0:
            taps = []
            for (a, b, _v) in kept:
                n = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
                u = ((b[0] - a[0]) / n, (b[1] - a[1]) / n)
                taps.append((a, u))
                taps.append((b, u))
            for (a, b, _v) in kept:
                vx, vy = b[0] - a[0], b[1] - a[1]
                n = math.hypot(vx, vy) or 1.0
                ux, uy = vx / n, vy / n
                cuts = []
                for (p, u) in taps:
                    if abs(ux * u[0] + uy * u[1]) > col_tol:
                        continue        # 共线孪生邻居，不是结点
                    if _cut_point(p, a, b, split_tol):
                        cuts.append(p)
                if not cuts:
                    out.append((a, b, _v))
                    continue
                cuts.sort(key=lambda p: (p[0] - a[0]) * vx + (p[1] - a[1]) * vy)
                chain = [a]
                for p in cuts:
                    if math.hypot(p[0] - chain[-1][0], p[1] - chain[-1][1]) > split_tol:
                        chain.append(p)
                chain.append(b)
                if len(chain) < 3:
                    out.append((a, b, _v))
                    continue
                st["split"] += 1
                st["added"] += len(chain) - 2
                for i in range(len(chain) - 1):
                    out.append((chain[i], chain[i + 1], None))
        else:
            out = list(kept)
        # —— ③ 第二遍去重：切分本身会造出新的重合边。
        #    两条只部分重合的长边（第一遍去不掉）在同一批切点上会被切成同一批小段，
        #    于是同一张网里留下几何完全一样的两条边。第一遍去重必须跑在切分**之前**
        #    （共线孪生邻居不能当切点），所以这批残渣只能在这里清。
        #    ★ 安全性：同一张网内的配对是一对一的，两条重合边不可能都配上同一条 GT 边，
        #      丢掉其中一条不可能丢 tp —— 官方 10 张实测 tp 311 → 311，fp 610 → 564
        #      （总分 79.3735 → 79.5042，+0.1307）。
        if dedup2_tol > 0 and len(out) > 1:
            _kept2: List[tuple] = []
            _out2: List[tuple] = []
            for (a, b, _v) in out:
                dup = False
                for (c, d, _w) in _kept2:
                    if (_pt_close(a, c, dedup2_tol) and _pt_close(b, d, dedup2_tol)) or \
                       (_pt_close(a, d, dedup2_tol) and _pt_close(b, c, dedup2_tol)):
                        dup = True
                        break
                if dup:
                    st["dup2"] += 1
                else:
                    _kept2.append((a, b, _v))
                    _out2.append((a, b, _v))
            if st["dup2"]:
                out = _out2
        if (st["dup"] == st0[0] and st["split"] == st0[1]
                and st["dup2"] == st0[2] and len(out) == len(segs)):
            continue                   # 这一条网没动过 ⇒ 原值原样留着
        _new = {}
        for i, (p0, p1, _v) in enumerate(out, 1):
            if _v is not None:
                _new["edge_%d" % i] = _v
            else:
                _new["edge_%d" % i] = [{"x": round(p0[0], 2), "y": round(p0[1], 2)},
                                       {"x": round(p1[0], 2), "y": round(p1[1], 2)}]
        rec["edges"] = _new
    return st


# ★★ 引脚切分（`STAGE2_EDGE_SPLIT_PIN`，默认 0 = 关）。
#
# 病（导线账本 `work/_netled.py`，官方 10 交付口径）：GT 边 639 / 我方边 874，配对网内
#   **「GT 边整条落在我方一条更长边内部」有 183 条**（`inside`，各类近失里最大的一类）。
#   机理：GT 的导线在**引脚处**断开（GT 的 wire 端点 = 引脚点），而我们骨架里那一段是
#   **一条长边**（引脚本体不算 wire ⇒ 没有分支点可切）⇒ GT 两条边、我们只有一条。
#   `split_dedup_edges` 的切点只认**邻居边的端点**，对「引脚落在边的内部」无能为力。
# 药：把每条边的**内部**若有引脚点落在 `tol` 内 ⇒ 在该引脚点处切成两段。放在
#   `split_dedup_edges` 之后、`trim_free_ends` **之前**：两段在新点处互为端点 ⇒
#   回缩那一步认得这是结点、不会把它拉走。只动 `edges` ⇒ 元件/引脚/netH 逐位不变。
#
# 读数（`work/_spin_ab.ps1`，官方 10 张全交付路径 `fast.py --extra "-vlm-pin off"`，参考臂
#   `esref` 80.88 / 导线 0.4975 / tp 329）：
#     `tol=2` **80.80**（导线 0.4894，tp **330**）· `tol=3` **80.81**（0.4903，tp **331**）
#     · `tol=5` 80.81（0.4903，tp 331）
#   ⇒ **tp 确实 +2（329→331），但 fp+fn 涨得更多（≈665→688）⇒ F1 −0.0072、总分 −0.07。**
#   机理：GT 的导线**并不在每个贴上的引脚处断开** —— 引脚搭在一条直导线中间时，GT 的边仍是
#   整条（GT 切的是**导线—导线**的 T 结点）。所以「在引脚点切」多数是假切分（多出一条 fp）。
#   **默认 0（关）**：机制留着，等有「哪些引脚真的在 GT 里切」的判据（见 RESULTS §5.75）。
EDGE_SPLIT_PIN = float(os.environ.get("STAGE2_EDGE_SPLIT_PIN", "0"))  # 引脚切分容差 px，0 = 关


# ★★ 跨网顶点切分（`STAGE2_EDGE_SPLIT_XNET`，默认 0 = 关）。
#   取值：`work/_inside.py` 取证 + `work/_xnet_ab.ps1` / `work/_psnap_ab.ps1` 扫描，见 RESULTS §5.76。
#   ★ 读数（官方 10 张，参考臂 80.88 / 导线 0.4975 / tp 329）：xn3 79.79 / 0.3887 / 305、
#     xn5 79.80 / 0.3894 / 306、xn8 79.92 / 0.4011 / 315；**加了切点去重**后 xnd5
#     79.71 / 0.3803 / 290（仍大负，逐案 2/1/5 涨、3/4/9/10 崩）⇒ **判负（−1.0~−1.2）**。
#   ★ 为什么必负（`work/_xhit.py` 直接量）：xn8 引入 **526 个新顶点，只有 86 个（16.3%）**
#     落在 GT 真切点 3px 内（召回 24.5%）；而度量的代价不对称 —— **假切一次 = tp−1 / fp+2 / fn+1**
#     （一条已配对的边被切两半后两半都过不了「两端点都在 5px 内」），**真切一次只 +1 tp**，
#     ⇒ 精度要 >~67% 才不亏，实测 16% ⇒ 原理上不可能赢。切分类机制整族封口。
EDGE_SPLIT_XNET = float(os.environ.get("STAGE2_EDGE_SPLIT_XNET", "0"))
# 候选点处必须有一条与目标边**非共线**（夹角 ≥ ANG）且长度 ≥ MINLEN 的边，才认它是 T 结点。
EDGE_SPLIT_XNET_ANG = float(os.environ.get("STAGE2_EDGE_SPLIT_XNET_ANG", "30"))
EDGE_SPLIT_XNET_MINLEN = float(os.environ.get("STAGE2_EDGE_SPLIT_XNET_MINLEN", "6"))


def split_edges_at_xnet(nets: Dict[str, dict], tol: float = None,
                        ang: float = None, minlen: float = None) -> Dict[str, int]:
    """把每条边的**内部**穿过「**别的网**的边端点」时，在该点一分为二。

    返回切分计数。端点附近（≤ `tol`）的候选点不切（那已经是结点）。

    依据（`work/_inside.py`，官方 10 交付口径）：netL 账本里 GT 侧 `inside` 那一类
    （GT 有两条边、我们只有一条长边）取到的 **136 个 GT 内部结点里 134 个，
    我方顶点本来就在结点处（d = 1.0~3.2px）**——只是那些顶点属于**另外一条网**
    （另一根导线穿过/搭上这条线），而 `split_dedup_edges` 的切点只在本网内找 ⇒ 漏切。

    ★ 第二轮（`STAGE2_EDGE_SPLIT_XNET_ANG`，默认 30°）：第一版把「所有网的**任何**边端点」
    都当候选，实测大负（`work/_xnet_ab.ps1`：xn3 79.79 / xn8 79.92 vs 参考 80.88）——
    机理是**假切分的代价是 tp−1、fp+2、fn+1（≈ −4 单位），而真切分只赚 +1 tp**，
    所以假切率必须 < 1/3。第一版最大的假切来源：**共线碎片端**（同一条长线被切出的邻段端点
    落在自己身上）与 1~3px 的残段。现在要求候选点处**存在一条与目标边非共线（夹角 ≥ ang）
    且长度 ≥ minlen 的边**（T 形另有一笔横着过来），共线碎片被这一条挡掉。
    """
    tol = EDGE_SPLIT_XNET if tol is None else tol
    ang = EDGE_SPLIT_XNET_ANG if ang is None else ang
    minlen = EDGE_SPLIT_XNET_MINLEN if minlen is None else minlen
    st = {"split": 0, "cand": 0, "rej_col": 0, "rej_net": 0}
    if tol <= 0:
        return st
    cos_min = math.cos(math.radians(max(0.0, min(89.0, ang))))
    # 点桶：把每条边的**两个端点**按 1px 归并，桶里存 (方向, 网名, 边长)
    bkt: Dict[Tuple[int, int], List[tuple]] = {}
    for nm, rec in nets.items():
        for v in (rec.get("edges") or {}).values():
            e = _seg_ends(v)
            if e is None:
                continue
            (p0, p1) = e
            L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if L <= 1e-9:
                continue
            ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
            for a, b in ((p0, p1), (p1, p0)):
                key = (int(round(a[0])), int(round(a[1])))
                bkt.setdefault(key, []).append((ux, uy, nm, L, (a[0], a[1])))
    if not bkt:
        return st
    for nm_a, rec_a in nets.items():
        edges = rec_a.get("edges") or {}
        segs = [(k, _seg_ends(v)) for k, v in edges.items()]
        segs = [(k, e) for k, e in segs if e is not None]
        if not segs:
            continue
        out: List[tuple] = []
        touched = False
        for _k, ((ax, ay), (bx, by)) in segs:
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 <= 1e-9:
                out.append(((ax, ay), (bx, by)))
                continue
            Le = math.sqrt(L2)
            ex, ey = dx / Le, dy / Le
            cuts: List[Tuple[float, float, float]] = []
            for _key, entries in bkt.items():
                qx, qy = entries[0][4]
                if (math.hypot(qx - ax, qy - ay) <= tol
                        or math.hypot(qx - bx, qy - by) <= tol):
                    continue                   # 与端点重合 ⇒ 已经有顶点
                if not any(e_nm != nm_a for _ux, _uy, e_nm, _L, _p in entries):
                    st["rej_net"] += 1
                    continue                   # 候选点必须来自**别的网**
                t = ((qx - ax) * dx + (qy - ay) * dy) / L2
                if t <= 0.0 or t >= 1.0:
                    continue
                fx, fy = ax + t * dx, ay + t * dy
                if math.hypot(qx - fx, qy - fy) > tol:
                    continue
                st["cand"] += 1
                ok = False
                for ux, uy, _e_nm, eL, _p in entries:
                    if eL < minlen:
                        continue
                    if abs(ux * ex + uy * ey) <= cos_min:
                        ok = True
                        break
                if not ok:
                    st["rej_col"] += 1
                    continue
                cuts.append((t, fx, fy))
            if not cuts:
                out.append(((ax, ay), (bx, by)))
                continue
            cuts.sort()
            # 去重：同一处结点常有多个候选点（同一根桩的两个端点等），相距 ≤ tol 只留一个 ——
            # 否则会切出 1~3px 的碎片边（每条碎片 = +1 fp +1 fn）。
            kept: List[Tuple[float, float, float]] = []
            for _t, cx, cy in cuts:
                if kept and math.hypot(cx - kept[-1][1], cy - kept[-1][2]) <= tol:
                    continue
                kept.append((_t, cx, cy))
            if not kept:
                out.append(((ax, ay), (bx, by)))
                continue
            pts = [(ax, ay)] + [(cx, cy) for _t, cx, cy in kept] + [(bx, by)]
            for i in range(len(pts) - 1):
                out.append((pts[i], pts[i + 1]))
            st["split"] += len(kept)
            touched = True
        if touched:
            rec_a["edges"] = {"edge_%d" % (i + 1): [
                {"x": round(p0[0], 2), "y": round(p0[1], 2)},
                {"x": round(p1[0], 2), "y": round(p1[1], 2)}] for i, (p0, p1) in enumerate(out)}
    return st


def split_edges_at_pins(nets: Dict[str, dict], pins: Dict[str, dict],
                        tol: float = None) -> Dict[str, int]:
    """把每条网里**内部穿过引脚点**（≤ `tol`）的边在该引脚处一分为二。

    返回 `{"split": 切点数}`。端点附近（≤ `tol`）的引脚不切（那已经是结点）。
    """
    tol = EDGE_SPLIT_PIN if tol is None else tol
    st = {"split": 0}
    if tol <= 0:
        return st
    ppt: List[Tuple[float, float]] = []
    for _c, _pn in (pins or {}).items():
        for _pv in (_pn or {}).values():
            _p = (_pv or {}).get("point")
            if isinstance(_p, dict) and "x" in _p and "y" in _p:
                ppt.append((float(_p["x"]), float(_p["y"])))
    if not ppt:
        return st
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = [(k, _seg_ends(v)) for k, v in edges.items()]
        segs = [(k, e) for k, e in segs if e is not None]
        if not segs:
            continue
        out: List[tuple] = []
        touched = False
        for _k, ((ax, ay), (bx, by)) in segs:
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 <= 1e-9:
                out.append(((ax, ay), (bx, by)))
                continue
            cuts: List[Tuple[float, float, float]] = []
            for (qx, qy) in ppt:
                t = ((qx - ax) * dx + (qy - ay) * dy) / L2
                if t <= 0.0 or t >= 1.0:
                    continue
                fx, fy = ax + t * dx, ay + t * dy
                if math.hypot(qx - fx, qy - fy) > tol:
                    continue
                if (math.hypot(qx - ax, qy - ay) <= tol
                        or math.hypot(qx - bx, qy - by) <= tol):
                    continue                   # 落在端点附近 ⇒ 已经是结点
                cuts.append((t, fx, fy))
            if not cuts:
                out.append(((ax, ay), (bx, by)))
                continue
            cuts.sort()
            pts = [(ax, ay)] + [(cx, cy) for _t, cx, cy in cuts] + [(bx, by)]
            for i in range(len(pts) - 1):
                out.append((pts[i], pts[i + 1]))
            st["split"] += len(cuts)
            touched = True
        rec["edges"] = {"edge_%d" % (i + 1): [
            {"x": round(p0[0], 2), "y": round(p0[1], 2)},
            {"x": round(p1[0], 2), "y": round(p1[1], 2)}] for i, (p0, p1) in enumerate(out)}
    return st


# ★★ 引脚端推进到元件框边（`STAGE2_END_TO_BOX`，默认 0 = 关）。
#
# 病（`work\_extoracle.py`，官方 10 交付口径）：现在不是 tp 的边里，**只把端点沿自身方向
#   外推 ≤10px 就能多出 69 条 tp**（R=0:329 → R=5:363 → R=10:398 → R=20:403）。
#   逐条看（`work\_extwhy.py`）这些「短掉的 9~10px」几乎全在**端点落在引脚上**的那一端。
# 实证（case 6 `C10`，像素级核对过）：
#   GT 框 `C10.bbox = [1729, 308, 1745, 328]`、我方引脚 `C10.pin_1 = (1736.55, 299.6)`
#   （= 框底边往**外** 8.4px，正是 `STAGE1_PIN_OUT_BOX=12` 那套「框边 ∓ PIN_OUT」的摆放），
#   GT 那条边的端点 = **(1736, 308) = 框边本身**；我方边停在引脚点 ⇒ 系统性短 8~12px。
#   图像核对：x=1737 处墨迹在 image y=790~795 断开（导线描边），y=796 起是**引线**描边，
#   我们的端点 (y_img 810) 落在引线里、GT 的端点 (y_img 802) 也在引线里但更靠里 ⇒ 差 8px。
# 药：**只动「端点落在引脚上（≤ tol）且是同网自由端」的那一端**，沿边方向推进到该引脚的
#   **元件框边**（射线与轴对齐框的第一个交点），推进量 ≤ maxlen。只动 `edges` 首尾两点
#   ⇒ 元件/引脚/netH 三个通道逐位不变。
END_TO_BOX = env_on("STAGE2_END_TO_BOX", "0")
END_TO_BOX_TOL = float(os.environ.get("STAGE2_END_TO_BOX_TOL", "4.0"))
END_TO_BOX_MAX = float(os.environ.get("STAGE2_END_TO_BOX_MAX", "30.0"))
END_TO_BOX_FREE = float(os.environ.get("STAGE2_END_TO_BOX_FREE", "3.0"))
# ★ 只对哪些**元件类型**做（`STAGE2_END_TO_BOX_TYPES`，默认 `c`；`all` = 全做）。
#   依据（`work\_gtendrule.py`，官方 10 的 533 个 GT 悬空端）：GT 的导线端**两种画法都有** ——
#     贴引脚（d_pin≤2）180 个、贴框边（|d_box|≤2）**69 个（全是电容 C\*）**、两者都不是 365 个。
#   按类型：`c` 有 68 贴框边 / 22 贴引脚；`box`（IC）70 贴引脚 / 1 贴框边；`r/gnd/v/led/l`
#   基本两者都不是。⇒ 一刀切（模拟 `work/_btb_sim.py`）**会亏**；只对 `c` 做，按上式
#   netL 0.5157→0.5878（**+0.072 F1 ≈ +0.72 分**）。
END_TO_BOX_TYPES = tuple(
    t.strip().lower() for t in os.environ.get("STAGE2_END_TO_BOX_TYPES", "c").split(",") if t.strip())
END_TO_BOX_ALL = ("all" in END_TO_BOX_TYPES) or (not END_TO_BOX_TYPES)
# ★ 实测（全交付路径 A/B，`work\_btb_ab.ps1`，官方 10 张，2026-10-14）：btref（默认关）
#   80.88 / netL 0.4975；bt1(TOL=4,MAX=30) / bt2(MAX=12) / bt3(TOL=6,MAX=60) **全部 80.88 /
#   netL 0.4967**（comp 0.9819 / pin 0.7731 / netH 0.7749 逐位不变）。逐案摆动极大且正负相抵：
#   case 6 +0.211（0.4912→0.7018）、8 +0.267、4 +0.180、5 +0.086；case 1 −0.218（0.6535→0.4356）、
#   9 −0.203、7 −0.161、3 −0.086、10 −0.083、2 0.000（该案无 c 脚）。
#   逐边账（`work\_btb_diff.py`，注意 `M.net_hypergraph_f1` 返回的是**下标对**）：只做 c 时
#   moved 123 / gain 51（inside→exact）/ loss 55（exact→graze）/ **净 −4**；`TYPES=all` 时
#   moved 306 / gain 52 / loss 128 / **净 −76**。
#   ★ 为何这条线原理上赢不了（`work\_gtgap.py`）：Δ = GT 线端 − 我方线端（沿 引脚→框边 轴）呈**双峰** ——
#   「贴框边画法」组中位 **10.1px**、「贴引脚画法」组中位 **1.0px**，两峰间距 10.1 > 2×line_tol(5.0)
#   ⇒ 任何统一平移都吃不下两种（穷举位移 s：最好只有 s=1.0 的 72/135 = 53%）。三个可得的判别器全否：
#   骨架端 t（box 组中位 6.7 vs pin 组 6.9，无区分度）、引线墨宽（除 case 1 = 3.5 外全 2.0）、
#   stage1 端点贴框边计数（case 1 12/24 vs case 4 0/14，方向相反）⇒ 语料缺陷（同类型两种画法、
#   图上不可分辨，与 §5.72 的框约定同类）。对冲变体（两种位置各发一条边）≈ +0.084 分（123 fp 换 51 tp）
#   —— **故意不取**。默认 0 保持。


def extend_ends_to_box(nets: Dict[str, dict], pins: Dict[str, dict],
                       comps: Dict[str, dict], tol: float = None,
                       maxlen: float = None, types=None) -> Dict[str, int]:
    """把「端点落在引脚上」的自由端沿边方向推进到该引脚的**元件框边**（GT 的电容就是这么画的）。"""
    tol = END_TO_BOX_TOL if tol is None else tol
    maxlen = END_TO_BOX_MAX if maxlen is None else maxlen
    types = END_TO_BOX_TYPES if types is None else types
    only = None if (not types or "all" in tuple(t.lower() for t in types)) \
        else set(t.lower() for t in types)
    st = {"moved": 0, "cand": 0}
    idx: List[Tuple[float, float, List[float]]] = []
    for ck, rec in (pins or {}).items():
        c = (comps or {}).get(ck)
        bb = c.get("bbox") if isinstance(c, dict) else None
        if not (bb and len(bb) == 4):
            continue
        if only is not None and str(c.get("type") or "").lower() not in only:
            continue
        box = [float(v) for v in bb]
        for _pk, pv in (rec or {}).items():
            pt = (pv or {}).get("point") or {}
            x, y = pt.get("x"), pt.get("y")
            if x is None or y is None:
                continue
            idx.append((float(x), float(y), box))
    if not idx:
        return st
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = {k: _seg_ends(v) for k, v in edges.items()}
        segs = {k: e for k, e in segs.items() if e is not None}
        if not segs:
            continue
        ends_all = [p for e in segs.values() for p in e]
        touched = False
        for k, (p0, p1) in segs.items():
            L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if L <= 1e-9:
                continue
            ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
            new = [tuple(p0), tuple(p1)]
            for si, (px, py) in enumerate((p0, p1)):
                # 自由端？（同网别的边端点不在这里）
                busy = any(math.hypot(qx - px, qy - py) <= END_TO_BOX_FREE
                           for j, e2 in segs.items() if j != k for qx, qy in e2)
                if busy:
                    continue
                best = None
                for (cx, cy, box) in idx:
                    d = math.hypot(cx - px, cy - py)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, cx, cy, box)
                if best is None:
                    continue
                st["cand"] += 1
                _d, cx, cy, box = best
                dx, dy = (ux, uy) if si == 1 else (-ux, -uy)   # 朝元件里面走
                x1, y1, x2, y2 = box
                lo, hi = 0.0, 1e9
                ok = True
                for (o, dd, mn, mx) in ((cx, dx, min(x1, x2), max(x1, x2)),
                                        (cy, dy, min(y1, y2), max(y1, y2))):
                    if abs(dd) < 1e-9:
                        if o < mn or o > mx:
                            ok = False
                            break
                        continue
                    t1, t2 = (mn - o) / dd, (mx - o) / dd
                    if t1 > t2:
                        t1, t2 = t2, t1
                    lo, hi = max(lo, t1), min(hi, t2)
                if not ok or hi < lo:
                    continue
                t_enter = lo
                if t_enter <= 0.5 or t_enter > maxlen:
                    continue
                new[si] = (cx + t_enter * dx, cy + t_enter * dy)
                st["moved"] += 1
                touched = True
            if touched and (abs(new[0][0] - p0[0]) > 1e-6 or abs(new[0][1] - p0[1]) > 1e-6
                            or abs(new[1][0] - p1[0]) > 1e-6 or abs(new[1][1] - p1[1]) > 1e-6):
                edges[k] = [{"x": round(new[0][0], 2), "y": round(new[0][1], 2)},
                            {"x": round(new[1][0], 2), "y": round(new[1][1], 2)}]
    return st


def trim_free_ends(nets: Dict[str, dict], delta: float = None,
                   jtol: float = 3.0) -> Dict[str, int]:
    """把导线边的**自由端**沿边方向朝内回缩 `delta` px（见上面长注释）。

    自由端 = 同一条网里没有别的边的端点贴在这个点上、也没有别的边的内部穿过它
    （即不是结点/接缝）。返回 {"trim": 回缩的端点数}。
    """
    delta = EDGE_TRIM if delta is None else delta
    st = {"trim": 0}
    if delta <= 0:
        return st
    for rec in nets.values():
        edges = rec.get("edges") or {}
        segs = []
        for _k, _v in edges.items():
            e = _seg_ends(_v)
            if e is not None:
                segs.append((e[0], e[1]))
        if not segs:
            continue
        out: List[tuple] = []
        for i, (a, b) in enumerate(segs):
            new = []
            for (p, q) in ((a, b), (b, a)):
                junc = False
                for j, (c, d) in enumerate(segs):
                    if j == i:
                        continue
                    if (_pt_close(p, c, jtol) or _pt_close(p, d, jtol)
                            or _cut_point(p, c, d, jtol)):
                        junc = True
                        break
                L = math.hypot(q[0] - p[0], q[1] - p[1])
                if (not junc) and L > delta + 1.0:
                    ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
                    new.append((p[0] + ux * delta, p[1] + uy * delta))
                    st["trim"] += 1
                else:
                    new.append(p)
            out.append((new[0], new[1]))
        _new = {}
        for i, (p0, p1) in enumerate(out, 1):
            _new["edge_%d" % i] = [{"x": round(p0[0], 2), "y": round(p0[1], 2)},
                                   {"x": round(p1[0], 2), "y": round(p1[1], 2)}]
        rec["edges"] = _new
    return st


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
# ★ 竖放 d/led 的 pin_1 在 −SPAN 一侧（默认 0 = 旧行为，见 `stamp_pins` 里的长注释）
PIN_AXIS_DL = env_on("STAGE2_PIN_AXIS_DL", "0")
# ★★ 两脚 d/led 的 pin_1(K) 落在哪一端**从图上读横杠**（默认 0 = 关；判据见
# `stage2/ledpol.py` 的模块 docstring）。`PIN_AXIS_DL` 那种「按轴选符号」的多数票
# 规则在官方 10 张上 **80.88 → 80.81（−0.07）**（`3/D1` 判对、`7/LED1` 与 `10/LED1`
# 判反），因为竖放 LED 可以朝上/朝下两种画法 —— 横杠在哪一端只能看图。
# `ledpol` 的横杠判据：官方 10 张 **判对 7 / 判错 0 / 不判 1**（case 6 LED2）；220 池
# 383 件上（`work/_ledpoolsim.py`，输入就是本函数的两个图章端点、真值取 GT 的 K 点）
# 现行几何默认只对 **211/383 = 55.1%**，判据判出 179 件、换向 80 次（**真该换 63 /
# 误换 17**）⇒ **池上净 +46 件**（每件错号影响 2 个 netH 成员）。端到端 A/B（官方
# 10 张，`work/_led_ab.ps1`~`_led_ab4.ps1`）：臂 `ledref`/`ledoff2`（显式 =0）=
# **80.88**（与交付口径逐位相同）、臂 `led1`/`led1b`（=1）= **80.98**（+0.10；逐案只有
# case 3 变：83.92→84.92，netH 0.6788→0.7074 = +2 成员，其余九案逐位不变）⇒
# **默认 1（上线）**，置 `STAGE2_LED_POLARITY=0` 可回到旧行为（那时连 cv2 都不读图）。
# ⚠ 曾试过「第二信号闸门」（两条横笔里游程更长的那条算横杠，两信号一致就判不出）：
# 换向精度 78.8%→95.8%，但判出件数腰斩 ⇒ 池上净收益 **+46 → +22 件**，已删除该分支
# （那条游程信号的比较式当时还写错了，详见 `docs/RESULTS.md` §5.81 第七节）。
LED_POLARITY = env_on("STAGE2_LED_POLARITY", "1")
# ★ 两脚二极管/LED 的官方脚名**不是号码**：`pin_1 = "K"`（阴极）、`pin_2 = "A"`（阳极），
# 只有 r/c/l 才是 `(1→"1", 2→"2")`。依据：400 张训练 GT 的 d/led 元件全量统计
# （d 192+192、led 177+177，各占该类 100%）与 10GT 全量一致。
# 官方 protocol 的 `pin_match` 明确要求 `pinname exact`（见
# `20261009引入/.../report/submission/*/1.json` 的 protocol 段），贴章路径原来把号码
# 直接当脚名写 ⇒ 每个 d/led 的两脚各吃一次 `pin_name` 失分（10GT 共 8 个）。
PIN_NAME_ALIAS = {("d", "1"): "K", ("d", "2"): "A",
                  ("led", "1"): "K", ("led", "2"): "A",
                  # ★ 2026-10 全池普查（`python -X utf8 work\_nonboxnum.py`，
                  # 域 = `10GT` + `10_GTcase` + `200_train_cases`）：这几个非 box 类型
                  # 的「脚号→脚名」在官方数据里是**确定性的**（不是随机 `P_##`），
                  # 而且是**名≠号**，所以我们按位置发的 1/2/3 全都会吃 `pin_name`：
                  #   crystal_2pin 44/44 = 1→X1, 2→X2
                  #   crystal_3pin  6/6  = 1→INPUT, 2→GND, 3→OUTPUT
                  #   crystal_4pin  8/8  = 1→OSC1, 2→OSC2, 3→GND, 4→GND
                  #   bjt (n=3)    45/45 = 1→B, 2→E, 3→C（另有 5 个 n=2 的随机 `P_##`）
                  #   amp_3pin     10/10 = 1→IN+, 2→IN-, 3→OUT
                  #   amp_5pin     11/11 = 1→IN+, 2→IN-, 3→OUT, 4→VCC, 5→VEE
                  # 而 `gnd`/`v`/`switch`/`other`/`mosfet`/`buzzer`/`battary`/`dc`/
                  # `esd`/`ant`/`net_*`/`m3螺丝` 的号与名都是**随机生成**的
                  # （`pin_119→P_20` 这种）⇒ 那些类型无表可查、只能靠键匹配。
                  # 这张表**只可能把失败变成功**（(类型,号) 对上才会查表；猜错的号本来
                  # 就配不上），所以是零风险的补丁；官方 10 张里没有这些类型，产物逐字节不变。
                  ("crystal_2pin", "1"): "X1", ("crystal_2pin", "2"): "X2",
                  ("crystal_3pin", "1"): "INPUT", ("crystal_3pin", "2"): "GND",
                  ("crystal_3pin", "3"): "OUTPUT",
                  ("crystal_4pin", "1"): "OSC1", ("crystal_4pin", "2"): "OSC2",
                  ("crystal_4pin", "3"): "GND", ("crystal_4pin", "4"): "GND",
                  ("bjt", "1"): "B", ("bjt", "2"): "E", ("bjt", "3"): "C",
                  ("amp_3pin", "1"): "IN+", ("amp_3pin", "2"): "IN-",
                  ("amp_3pin", "3"): "OUT",
                  ("amp_5pin", "1"): "IN+", ("amp_5pin", "2"): "IN-",
                  ("amp_5pin", "3"): "OUT", ("amp_5pin", "4"): "VCC",
                  ("amp_5pin", "5"): "VEE"}
# ★★ 每一侧只保留「沿轴投影绝对值最大」的候选（最贴轴的那个），旁侧幻影脚不发号。
# 2026-10 实测（`work/tools/dev/_stampchk.py`，官方 10 张）：旧逻辑按候选顺序贪心
# 发号，`pts` 的 `side=[F,F,T,F]` 会变成 `[2,1,2,1]` —— **两个真脚都拿到 `2`**、
# 两个垂直旁侧的幻影脚拿到 `1`；`num_of[(cid,0)] == num_of[(cid,2)]` 于是让两条
# stage1 网在 `merge_shared_pins` 里被**合并**（case 1：六颗去耦电容的电源梳
# `net_31` 被并进 U1 的 `net_20` ⇒ 短路），而只含幻影脚的碎片网活下来冒充 `C151.1`。
# 规模：贴章元件 126 ｜ 同号冲突 **28（22.2%）** ｜ 幻影抢号 46。
STAMP_ONE_PER_SIDE = os.environ.get("STAGE2_STAMP_ONE_SIDE", "1").strip().lower() \
    not in ("", "0", "false", "off", "no", "none")


def stamp_pins(c: dict, ctype: str, pts_img: Sequence, H: float,
               mask=None
               ) -> Optional[List[Optional[Tuple[str, Tuple[float, float]]]]]:
    """两脚贴章元件的官方引脚点位。

    返回 `[(引脚号, (x, y)) | None, ...]`，顺序与 `pts_img` 一一对应（图像坐标入参，
    返回的是 **json 坐标**）；`None` = 该候选没拿到号（旁侧幻影脚，调用方必须跳过，
    但**下标不能压缩**）；不适用时返回 None。

    引脚号按官方约定给：−SPAN 一侧是 `1`、+SPAN 一侧是 `2`（`d`/`led` 反过来）。
    我们自己的附着点落在哪一侧，就继承那个号 —— 这一步决定了 net 超图里
    `(元件键, 引脚号)` 的对不对得上。
    """
    if not PIN_STAMP_ON or ctype not in PIN_STAMP_TYPES or not len(pts_img):
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
    axis_y = (not tall) if ctype in PIN_STAMP_PERP else tall
    span = PIN_STAMP_SPAN.get(ctype, PIN_STAMP_DEFAULT_SPAN)
    # pin_1 在轴上的投影符号：+1 表示在 +SPAN 一侧
    s1 = 1.0 if ctype in PIN_STAMP_POS else -1.0
    if PIN_AXIS_DL and ctype in ("d", "led") and axis_y:
        # ★ 竖放的 d/led：pin_1 在 **−SPAN** 一侧（`STAGE2_PIN_AXIS_DL`，默认 0 = 旧行为）。
        #   依据：官方语料统计——d/led **竖放时 pin_1(K) 在 −SPAN 89/99、横放 57/84 在 +SPAN**；
        #   官方 10 张里 `3/D1`、`7/LED1`、`10/LED1` 都是竖放且被统一取 + 的规则判反
        #   （`work/_ledpin.py` 实测 122 同序 / 3 反序；GT 的 `pin_1` 脚名恒为 `K`）。
        #   与 `PIN_NAME_ALIAS` 一致：号翻过来以后，编号 1 的那一端才真是 K 端。
        #   读数（单臂 A/B，官方 10 张、参考臂 80.88）：**80.81（−0.07）**——逐案是
        #   `3/D1` netH 0.6788→**0.7074**（+2 成员 ✓ 规则对）、但 `7/LED1` 0.7819→0.7579、
        #   `10/LED1` 0.7975→0.7712（各 −2 成员 ✗ 规则错）。原因：竖放 LED 可以**朝上/朝下**
        #   两种画法，K 端（横杠）在哪一侧由图决定，与轴无关；语料 89/99 只是多数票。
        #   ⇒ 默认 0（关）。**要拿这 3 个件，必须从图上读出横杠在哪一端**，不是靠轴。
        s1 = -s1
    if axis_y:
        p1, p2 = (cx, cy + s1 * span), (cx, cy - s1 * span)
    else:
        p1, p2 = (cx + s1 * span, cy), (cx - s1 * span, cy)

    # ★ `STAGE2_LED_POLARITY`：d/led 的 pin_1(K) 在哪一端**由图上的横杠决定**。
    #   `mask` 是本案原图的暗红笔画掩膜（`build_doc` 惰性算一次）；判据返回 'A' 表示
    #   横杠在 p2 那端 ⇒ 换向（等价于 s1 取反后重算两个候选点）。判不出来（None）就
    #   保持旧行为。读数见 `stage2/ledpol.py` 与上面 `LED_POLARITY` 的注释。
    if LED_POLARITY and ctype in ("d", "led") and mask is not None:
        try:
            from stage2 import ledpol as _ledpol
            _got = _ledpol.k_end(mask, p1, p2, H)[0]
        except Exception:
            _got = None
        if _got == "A":
            s1 = -s1
            if axis_y:
                p1, p2 = (cx, cy + s1 * span), (cx, cy - s1 * span)
            else:
                p1, p2 = (cx + s1 * span, cy), (cx - s1 * span, cy)

    projs = []
    side = []
    for p in pts_img:
        proj = (float(H) - float(p[1]) - cy) if axis_y else (float(p[0]) - cx)
        projs.append(proj)
        side.append(True if proj * s1 > 0 else False)
    # 这一类元件的 GT 引脚数**恒为 2**（r 3398/2、c 3313/2、l 230/2 …），
    # stage1 给 1 个或 3 个附着点都只是几何噪声 —— 一律按图章输出 pin_1/pin_2。
    #
    # 2026-07 审计提出「`len(side)==1` 时落进 else 只输出 1 个引脚，与本注释矛盾，
    # 改成 `if len(side) != 2 or side[0] == side[1]:`」—— **实测否证、已回退**
    # （10GT 无 VLM：48.35 → 46.88；component 与 pin_f1 **逐位不变**，因为
    # `cfg.pin_box_only=True` 时 r/c/l/d 的引脚根本不进 pin_f1，但 netH 的引用
    # 含全部元件 ⇒ netH tp −14 / fp +14 / fn +14，净 −1.44 分）。
    # 机制：补出来的 `pin_2` 位置是我们**盖章猜**的（`cy ± span`），与 GT 的
    # pin_2 经常对不上，于是既加 fp、又会改变 `net_pin_set` 的交集把原本配对的
    # 网络配对打乱。补桩数是「引脚 f1 的口径」问题，等 `pin_box_only` 之外的
    # 口径确定后再谈，不要在这里凭空补。
    if len(side) == 2 and side[0] == side[1]:
        # 两个附着点落在同一侧（我们自己的点太糙）—— 退回按顺序给号
        side = [True, False]
    if not STAMP_ONE_PER_SIDE:
        seen = [False, False]
        fixed = []
        for want1 in side:
            if seen[0 if want1 else 1]:
                want1 = not want1
            seen[0 if want1 else 1] = True
            fixed.append(want1)
        return [(("1", p1) if want1 else ("2", p2)) for want1 in fixed]
    # 每一侧只留 |proj| 最大的候选（最贴轴的那个）。多余的候选**不发号**：它们的
    # stage1 网因此失去引用，在 assemble 里被当「无引用的网」丢掉，不再冒充
    # `C151.1`；而两个真脚各拿一个号 ⇒ 两条网不再共享引用、不再被 merge。
    best: Dict[bool, Tuple[float, int]] = {}
    for i, want1 in enumerate(side):
        a = abs(projs[i]) if i < len(projs) else 0.0
        cur = best.get(want1)
        if cur is None or a > cur[0]:
            best[want1] = (a, i)
    out: List[Optional[Tuple[str, Tuple[float, float]]]] = [None] * len(pts_img)
    for want1, num, xy in ((True, "1", p1), (False, "2", p2)):
        b = best.get(want1)
        if b is None:
            # 该侧一个候选都没有：把理想图章点补在候选之后（下标 ≥ len(pts_img)
            # ⇒ 没有 stage1 网会引用这个下标），保证引脚文档仍是两脚。
            out.append((num, xy))
        else:
            out[b[1]] = (num, xy)
    return out


# ---------------------------------------------------------------------------
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
            #
            # ★★ 2026-10-14 补：`build_doc` 在 `out_nets` 那段**已经 `pop("_refs")`**
            # （:1255），而 `weld_nets`（跨网焊接）正是在那之后调用 ⇒ 只读 `_refs`
            # 的写法在那里恒为空，合并出来的网**边并了、成员只剩 root 一份**
            # （实测 `work/_weldcmp.py` case 4：15 条边但成员只有 `{C7.1}`，
            # 端到端只拿到 +0.39 而不是离线预测的 +1.49/张）。
            # 因此这里改成「`_refs` ∪ `hyperGraph` 解析出的引用」：正常流程里两者
            # 同源（:1244 同一个 `refs` 写两处）⇒ 近似空操作；只有 `_refs` 被摘掉
            # 或过期时才真正起作用（`merge_shared_pins` 的 `dict(nets[members[0]])`
            # 也会留下过期的 `_refs`，同一族的坑）。
            for r in list(nets[m].get("_refs") or []) + _hg_refs(nets[m].get("hyperGraph", "")):
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
            #
            # ★★ 2026-10-14 补：`build_doc` 在 `out_nets` 那段**已经 `pop("_refs")`**
            # （:1255），而 `weld_nets`（跨网焊接）正是在那之后调用 ⇒ 只读 `_refs`
            # 的写法在那里恒为空，合并出来的网**边并了、成员只剩 root 一份**
            # （实测 `work/_weldcmp.py` case 4：15 条边但成员只有 `{C7.1}`，
            # 端到端只拿到 +0.39 而不是离线预测的 +1.49/张）。
            # 因此这里改成「`_refs` ∪ `hyperGraph` 解析出的引用」：正常流程里两者
            # 同源（:1244 同一个 `refs` 写两处）⇒ 近似空操作；只有 `_refs` 被摘掉
            # 或过期时才真正起作用（`merge_shared_pins` 的 `dict(nets[members[0]])`
            # 也会留下过期的 `_refs`，同一族的坑）。
            for r in list(nets[m].get("_refs") or []) + _hg_refs(nets[m].get("hyperGraph", "")):
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
def build_doc(stage1: dict, assigned: dict, case_id: str = "") -> dict:
    img = stage1.get("image") or {}
    H = float(img.get("h") or 0.0)
    # ★ `STAGE2_LED_POLARITY=1` 时才读原图（算暗红笔画掩膜，供 `stamp_pins` 判 d/led 极性）。
    #   关的时候**连 cv2 都不 import**，产物与不挂这条通道时逐字节一致。
    _led_mask = None
    if LED_POLARITY and img.get("path"):
        try:
            import cv2 as _cv2
            from stage2 import ledpol as _ledpol
            _bgr = _cv2.imread(str(img["path"]))
            if _bgr is not None:
                _led_mask = _ledpol.red_mask(_bgr)
        except Exception:
            _led_mask = None
    comps = stage1.get("components") or []
    a = assigned.get("comps") or {}
    st1_pins = stage1.get("pins") or {}

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
    # ★ `STAGE2_BOX_ALONE_U1`（默认 **0 = 先 A/B**）：整页只有一颗 `box`、它又没读到位号时，
    #   按惯例命名为 `U1`。动机：`10_GTcase`（datasheet 派生 10 张）里 6 张**不印位号**、
    #   只印型号（`0065` 印的是 `LM2578A` 和脚号 1..8，见图），GT 仍按惯例叫 `U1`；
    #   我们发兜底键 `pNNN` ⇒ 该元件配不上 GT，且它的引脚被 `PIN_DROP_UNKEYED`
    #   **整颗丢掉**（实测 `0065`/`0059`/`0042` 的 box 引脚发 0 个），pin F1 直接爆 0
    #   （派生 10 张里 7 张 pin F1 = 0，见 §5.262）。
    BOX_ALONE_U1 = env_on("STAGE2_BOX_ALONE_U1", "1")
    # ★ `STAGE2_BOX_GAP_U1`（默认 **0 = 先 A/B**）：把上面那条从「整页唯一 box」放宽到
    #   「整页 box 数 k、读出 U 号正好 k-1 个、且这些号都落在 1..k 里」——缺的那一个号
    #   就补给没读到号的那颗。`work/_ugap.py`（220 案、GT 当尺子）实测：条件成立 26 次，
    #   **补对 20 / 补错 6**（含 3 次 GT 本身无名）。补对一次能救回该 IC 的全部脚与网
    #   （§5.263 里那几个 +12~+14 的案子就是同类），所以值得单独 A/B。
    BOX_GAP_U1 = env_on("STAGE2_BOX_GAP_U1", "0")
    _n_box = 0
    _used_u: set = set()
    for _c0 in comps:
        _a0 = a.get(_c0["id"]) or {}
        _t0 = str(_a0.get("ctype_fix") or _c0.get("type") or "other").strip().lower()
        if _t0 != "box":
            continue
        _n_box += 1
        _d0 = _a0.get("designator") or {}
        _k0 = str(_d0.get("canon") or _d0.get("text") or "").strip()
        _m0 = re.fullmatch(r"U(\d+)", _k0)
        if _m0:
            _used_u.add(int(_m0.group(1)))
    # `tp` 闸门的台账（只在本函数内累计，退出前打印；`STAGE2_TP_LEDGER=1` 时）。
    _tp_ledger: List[dict] = []

    # ★ `STAGE2_LIB_PINS`：官方元器件库的引脚先验总闸。**默认 `1`（开）** —— 2026-10-09 依据
    # 220 张拆分回归从 0 翻过来（交付口径官方 9 张 +1.60、九张零回归）；2026-10-14 默认库换成
    # 官方 box/IC 库 `BoxICLibrary_200cases.json`（用户 m42932；官方 9 张 68.7995 → 69.0883）。
    # 关掉时 `libpins` 这个模块**根本不会被 import**（上面那个 `LIB_PINS` 常量只是
    # 读一个环境变量，`libpins` 的模块级代码不会去读 1.83 MB 的库文件），于是
    # 产物与不挂接这条通道时逐字节一致。开的时候，整份文档**只加载库一次**
    # （`libpins.LibCtx` 自带惰性 + 幂等 `load()`）。
    # 依据与两条规则见 `stage2/libpins.py` 的模块 docstring。★ 当初默认关的原因是
    # 「收益靠元件型号名被读对，而 stage2 直出的 `Name` 是 OCR 原文，17 个 box 里只有
    # 1 个能命中库」⇒ 收益在**融合层/交付链**（融合会重建文档），不在 stage2 单跑。
    # 库文件路径见 `libpins.lib_path()`（包内 `data/BoxICLibrary_200cases.json` 优先，
    # 缺席退包内 `data/cpntLibrary.json`）。
    _LIB_CTX = None
    if LIB_PINS:
        from stage2 import libpins as _libpins
        _LIB_CTX = _libpins.new_ctx()

    for i, c in enumerate(comps):
        cid = c["id"]
        info = a.get(cid) or {}
        _unkeyed = False
        # `canon` 是前缀矫正后的规范位号（`Rl4` -> `R14`）；没有就用原始读法。
        # 输出键必须用规范位号 —— 官方按 key 匹配元件，一个字符错整条链就丢。
        _d = info.get("designator") or {}
        des = _d.get("canon") or _d.get("text")
        key = des if des else None
        # ★ 内部合并类 `tp` 的**文本闸门**（`STAGE1_MERGE_TP=1` 时才可能出现，
        #   见文件顶 `TP_MERGED` 的长注释）。三条规则里的 ③ 在这里：
        #   **位号没绑上就直接丢**，且绕开 `DROP_UNKEYED` 的兜底键 `pNNN`
        #   —— 合并类的存在意义就是「读到位号才认它」，给它一个假的 `pNNN`
        #   等于把原来那 98.9% 的 fp 又请回来。
        _is_tp = MERGE_TP and str(c.get("type") or "").strip().lower() == TP_MERGED
        # 「绑上了」= 有键 **且** 这个键带着读出证据（见文件顶 `TP_TRUST_EV` 的长注释：
        # 融合层会给 tp 补一个只有 `canon` 的空壳位号，那种键会贴到别的物理器件上）。
        _ev_ok = (not _is_tp) or (not TP_TRUST_EV) or any(
            k not in ("canon", "_by") for k in _d)
        if _is_tp and (not key or not _ev_ok):
            if TP_LEDGER_ON:
                _tp_ledger.append({"cid": cid,
                                   "drop": "unkeyed" if not key else "no-evidence",
                                   "subtype": c.get("subtype"), "des": des,
                                   "by": _d.get("_by")})
            continue
        if not key or key in components:
            # 拿不到位号（或位号撞号）—— 兜底键 `pNNN`（GT 里对应的是无名元 `∅NNN`）。
            # ★ 2026-10-14 起默认 **输出**（`DROP_UNKEYED=0`）：书面口径下官方对 GT 的
            #   `∅NNN` 走**坐标匹配**（QA 评测 Q5），我们的 `pNNN` 靠 bbox 就能被配成
            #   GT 键（`rewrite_keys` 随后把 pins/nets 的引用一起改写），官方 10 张
            #   +12.49。详见文件顶 `DROP_UNKEYED` 的长注释与 `docs/RESULTS.md` §5.20。
            #   `=1` 退回旧行为（判题机复刻尺下 +10.48，那把尺子已被业主认定为写错）。
            if DROP_UNKEYED:
                continue
            _ct_here = str(info.get("ctype_fix") or c.get("type") or "other").strip().lower()
            _gap = None
            if BOX_GAP_U1 and _ct_here == "box" and 0 < len(_used_u) == _n_box - 1 \
                    and _used_u <= set(range(1, _n_box + 1)):
                _miss = sorted(set(range(1, _n_box + 1)) - _used_u)
                if len(_miss) == 1 and ("U%d" % _miss[0]) not in components:
                    _gap = "U%d" % _miss[0]
            if (BOX_ALONE_U1 and _ct_here == "box" and _n_box == 1
                    and "U1" not in components):
                key = "U1"
                _unkeyed = False
                if os.environ.get("STAGE2_DBG_PINS"):
                    print("[boxalone] cid=%s 整页唯一 box 且无位号 → 命名 U1" % cid)
            elif _gap:
                key = _gap
                _unkeyed = False
                if os.environ.get("STAGE2_DBG_PINS"):
                    print("[boxgap] cid=%s 兜底键 → %s（已用 U 号 %s / 共 %d 颗 box）"
                          % (cid, _gap, sorted(_used_u), _n_box))
            else:
                key = "p%03d" % i
                _unkeyed = True
        elif PIN_DROP_LAUNDERED and FKEY_RE.match(str(key)):
            # ★★ 2026-1x 修：兜底键**不总是这里发的** —— 融合/求解层会把已经发过的
            #   `pNNN` 当成「读出来的位号」回填（`assigned[cid]["designator"]["canon"]`，
            #   实测 `_by = "solve:vlm:enrich"`），于是本函数的 `designator` 分支拿到一个
            #   「有键」的元件、`_unkeyed` 保持 `False` ⇒ 下面 :PIN_DROP_UNKEYED 那道闸
            #   被**绕过**，`pNNN` 的合成脚号又回到交付文档里。
            #   实证（`work/_fast_out_trimdef2`，VLM 臂，case 1）：stage2 里 `p012` 的引脚
            #   表是**空的**，融合后变成 `pin_1/pin_2/pin_3`，8 个兜底键元件共 25 个脚；
            #   判题机 `pin.id_f1` 层它们是纯 fp（10 张合计 **221** 个 id-fp 全部来自这里，
            #   而「脚号读错」只有 3 个）——`work/_pinidset.py`。
            #   判据用**键的形状**（与 `schematic_parser/metrics.py:is_unkeyed`、
            #   本文件 `NET_DROP_UNKEYED` 的判据同源）：`pNNN` 不是任何真位号的形状。
            #   ⚠️ 这里**只**认回引脚闸；元件框留不留仍由上面的 `DROP_UNKEYED` 决定
            #   （上游回填的 `pNNN` 在 `DROP_UNKEYED=1` 下本来也不会被丢，属于既有不一致，
            #   不在本次改动范围内 —— 默认 `DROP_UNKEYED=0`，两者行为一致）。
            _unkeyed = True
        keymap[cid] = key
        # ★ `ctype_fix`：位号通道的「内涵互校验」翻了类型时（官方 case 6 的两颗 LED
        #   被 YOLO 判成 `d`，而印字读作 `LED1`/`LED2` —— `LED` 前缀只属于 `led`），
        #   以 `assign` 的记录为准。类型既是输出字段（官方按精确串比），又决定引脚
        #   图章与脚名别名（下面的 `stamp_pins` / `PIN_NAME_ALIAS`）。
        #   没有这个字段时逐位退回原来的 `c.get("type")`（默认关时行为不变）。
        ctype = str(info.get("ctype_fix") or c.get("type") or "other").strip().lower()
        ct_by_cid[cid] = ctype
        if _is_tp:
            # ★ 三条规则里的 ① / ②：有 CV 形状证据（子类型在 `TP_CV_SUBTYPES` 里）
            #   → `testpoint`，否则 → `pin`。证据来自 `stage1/detect.py` 记在元件上的
            #   `subtype`（CV 子类型；YOLO 框被 CV 复用时也补了，见 `detect.py` 里
            #   `best["cv_subtype"]` 那条）。
            #   没有证据**不等于** pin：也可能是 CV 漏了这一颗 ⇒ 落到 ② 的 `pin`，
            #   这是刻意的保守选择 —— `pin` 的 GT 基数（175/32 张）远大于
            #   `testpoint`（43/6 张），押 `pin` 的期望损失最小。
            _sub = str(c.get("subtype") or "").strip().lower()
            ctype = "testpoint" if _sub in TP_CV_SUBTYPES else "pin"
            if TP_LEDGER_ON:
                _tp_ledger.append({"cid": cid, "key": key, "type": ctype,
                                   "subtype": _sub,
                                   "by": "cv:sub" if _sub in TP_CV_SUBTYPES else "default"})
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
        if _unkeyed and PIN_DROP_UNKEYED:
            # 兜底键元件的引脚一个都不发（见文件顶 `PIN_DROP_UNKEYED` 的长注释）。
            continue
        if not pts:
            continue
        # 两脚贴章元件：官方点位是「中心 ±SPAN」的理想图章点，不是走线接点。
        # 直接照图章生成，顺便把 1/2 号按官方约定摆正（见 `stamp_pins` 的长注释）。
        _st = stamp_pins(c, ctype, pts, H, _led_mask)
        if _st is not None:
            _g: Dict[str, dict] = {}
            for k, _item in enumerate(_st):
                if _item is None:
                    continue          # 旁侧幻影候选：不发号、不注册槽位（保持 k 对齐）
                num, xy = _item
                _g["pin_%s" % num] = {
                    "pinname": PIN_NAME_ALIAS.get((ctype, num), num),
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
        for k in range(len(pts)):
            rec = pinfo.get(k) or pinfo.get(str(k)) or {}
            num = ((rec.get("number") or {}).get("text") or "").strip()
            nm = ((rec.get("name") or {}).get("text") or "").strip()
            raw.append({"num": num, "nm": nm})
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
        #   * `box`：OCR 读出的号只要都是 1..999 的**互不相同**整数就采信
        #     （真实引脚号是 1..400 左右的小整数，撞号/越界就说明读废了）；
        #   * 其余：仍要求完整的 1..N 排列 —— 它们的引脚号不进 pinF1，但会进
        #     `net_hyperGraph`，保守的先验在那里仍然更稳。
        nums = [r["num"] for r in raw]
        if ctype == "box":
            ok = bool(nums) and all(n.isdigit() for n in nums)
            if ok:
                try:
                    iv = [int(n) for n in nums]
                except ValueError:
                    ok = False
                else:
                    ok = (len(set(iv)) == len(iv)
                          and all(1 <= v <= 999 for v in iv))
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
                  "drop=%s mixed=%s cap=%s keepnamed=%s nums=%s"
                  % (cid, ctype, len(pts), len(raw), ok, BOX_PIN_PARTIAL,
                     BOX_PIN_DROP_UNREAD, BOX_PIN_DROP_MIXED_ONLY,
                     BOX_PIN_DROP_CAP, _KEEP_NAMED_MODE, [r["num"] for r in raw][:40]))
        _anykept = True
        if not ok:
            if ctype == "box" and BOX_PIN_PARTIAL:
                # 逐引脚：读出来的号只要是 1..999 的整数、没跟别人撞，就采信；
                # 其余用位置号补，并避开已用掉的号（保证键不撞）。
                used, tmp = set(), list(nums)
                kept = [False] * len(tmp)
                for i, n in enumerate(tmp):
                    v = int(n) if (n and n.isdigit()) else 0
                    if 1 <= v <= 999 and v not in used:
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
                    # `fill=True` = 这个号是**位置补位号**（假号），不是 OCR 读出来的。
                    # 规则 A 需要它来区分「号读到了」与「号读废了、拿位置顶的」——
                    # 后者只有名字是真证据（见下面规则 A 的长注释）。
                    raw[k]["fill"] = not kept[k]
                _anykept = any(kept)
                if _LIB_CTX is not None:
                    # ★ **规则 A（补号）**：把「读到了引脚名、但没读出合法号」的桩，
                    # 用官方库的 `(型号, 引脚名) → 号` 换成真号。这里「没读出合法号」
                    # 有**两种**形态，都要抓：
                    #   ① `tmp[i]=""`（一个号都没读出来）；
                    #   ② 读到了号但没通过 1..999/不撞号检查 ⇒ `tmp[i]` 是**位置补位号**
                    #      （`kept=False, fill=True`，比如按顺序填的 `1`,`2`,…）。
                    #   位置号只是占位，真正的证据是那个读到的名字；`_libsim.py` 量到的
                    #   6 个可补引脚，在本管线里正是「有名字 + 位置号」这一种。
                    # 依据：`_libnum.py` 实测「名字在库里唯一映射一个号」的 20 个里
                    # **20/20 都是 GT 号**（精度 100%）；放宽到多号就得猜，猜错 =
                    # fp+fn 双罚，所以本实现只做 uniq（`_libsim.py --mode uniq`）。
                    # `used` = 这颗 IC 里**别处的有效号**（`kept=True` 的那些）：撞号就
                    # 放弃，绝不改别人的号 —— 补号的目的就是把假号换真号，不是把两个
                    # 真脚挤进同一个 `pin_<号>` 键。位置号**不进** `used`（它本来就是假的，
                    # 否则每个目标号都被占着、规则 A 一次都不会触发 —— 实测过）。
                    _lib_used = set(str(q.get("num") or "") for q in raw
                                    if q.get("kept"))
                    _lib_used.discard("")
                    _n = _libpins.complete_pins(name, raw, _LIB_CTX, used=_lib_used)
                    if _n:
                        _anykept = any(q.get("kept") for q in raw)
                        if os.environ.get("STAGE2_DBG_PINS") or LIB_PINS_DBG:
                            print("[libpins] 规则A 元件=%r 补 %d 个引脚号" % (name, _n))
            else:
                for k in range(len(raw)):
                    raw[k]["num"] = str(k + 1)
        # ★ 退化库名守卫（`STAGE2_LIB_REFINE_DEGEN`，档位唯一读取点 `libpins.degen_mode()`）
        #   规则 B 会用库名覆盖我方读数，而**融合层会把型号名吸附成库里同名但不同元件
        #   的键**（§5.48 把阈值放到 4 后，`5 U3` 的 `VOUTAD` → 键 `OUT`），那个键的引脚名
        #   表是纯数字 `1`..`4` ⇒ 读对的 `VIN`/`TAB` 被冲成 `3`/`4`（官方 10 张实测
        #   2 根脚 right→wrong，`work\_pinname_audit.py`）。只挂在 `librefine` 上没用：
        #   它跑到时名字已经是数字了（`ref == cur`），所以守卫必须在规则 B 这里。
        _dmode = _libpins.degen_mode() if _LIB_CTX is not None else "off"
        _shape_block = False
        if _dmode == "shape" and ctype == "box":
            _lkey0 = _LIB_CTX.key_for(name)
            _pairs0 = []
            for _q in raw:
                _n0 = str(_q.get("num") or "")
                if not _n0.isdigit():
                    continue
                _r0 = _LIB_CTX.name_for_num(_lkey0, _n0)
                if _r0:
                    _pairs0.append((str(_q.get("nm") or ""), _r0))
            # 库给的名字全无字母 + 我方读数全都有字母 ⇒ 这个库键可疑，整颗不动。
            if _pairs0 and not any(_libpins.has_letter(_r0) for _c0, _r0 in _pairs0) \
                    and not any(not _libpins.has_letter(_c0) for _c0, _r0 in _pairs0):
                _shape_block = True
        for k, p in enumerate(pts):
            r = raw[k]
            num = r["num"] or str(k + 1)
            # ★ `STAGE2_BOX_PIN_DROP_UNREAD`：把「没读出号、靠位置补出来的」引脚桩
            # **既不写进引脚表、也不进网络**。它只可能变成 `pin.extra`（纯 fp），
            # 而且会把假引脚塞进网络（10GT 实测 netH 0.4746 → 0.4971）。
            # 取值 `1`/`all` = 一律丢；`mixed` = **只在「这颗 IC 至少读懂了一个号」
            # 时才丢**（一个都没读懂的那些，位置号 1..N 本身是个有用的先验）。
            if (not r.get("kept", True)) and BOX_PIN_DROP_UNREAD \
                    and (not BOX_PIN_DROP_MIXED_ONLY or _anykept):
                # ★ `STAGE2_BOX_PIN_KEEP_NAMED`（默认关，见常量区长注释）：这条闸原本
                #   连「这根桩读到了真名字」都不看就丢。开了以后：
                #   `named` 档 = 名字 ≥2 字符且含字母就留 → 用位置补位号发出去；
                #   `known` 档 = 再加一道「全局引脚名词表认识这个名字」的闸（挡 `D`/`O`/`N` 残读）。
                #   猜中号 ⇒ `pin.<号>` 名对 = tp；猜不中 ⇒ `name_mismatch`(fp+fn)，
                #   但网络通道白赚一个成员 ⇒ 收益只能由 A/B 定（同类旧档都判负）。
                if not (BOX_PIN_KEEP_NAMED and _named_slot_ok(r.get("nm"), _LIB_CTX)):
                    continue
            # ★★★ 「迟到闸门」：`BOX_PIN_NEED_NAME_LATE` 时**不在**这里丢，挪到规则 B 之后
            #     （见下方同名判断）—— 让库的 `(型号, 号) → 名` 先有机会把这个桩救活。
            _lib_named = False
            if (ctype == "box" and BOX_PIN_NEED_NAME
                    and not BOX_PIN_NEED_NAME_LATE and not r.get("nm")):
                continue
            nm = r["nm"] or num
            # 非贴章路径上的 d/led 同样必须给官方脚名（K/A），否则 `pin_name` 必失分
            if ctype in ("d", "led"):
                nm = PIN_NAME_ALIAS.get((ctype, str(num)), nm)
            # ★ **规则 B（改名）**：这个桩的**号读到了**（`num` 是 1..999 的整数，或者
            # 规则 A 刚从库里补出来的号），但名字要么空、要么不在库里该型号的名字
            # 集合里 —— 那就用库的 `(型号, 号) → 名`，**只有该号唯一对应一个名字**时
            # 才改写。注意顺序：改名在 `nm = r["nm"] or num` **之后**，所以
            # 「名字空 ⇒ 退化成号」的那种也有机会被换成真名字。
            # 依据：`_libsim.py --fixname` 官方 9 张改名 4 处 **4/4 正确**
            # （pin F1 0.4951→0.5936）；号是多实例聚合过的，一个号对应多个名字时
            # 猜错 = fp+fn 双罚，所以多候选一律不动手。
            if (_LIB_CTX is not None and _libpins.LIB_PINS_NAMEFILL
                    and ctype == "box" and str(num).isdigit()):
                _lkey = _LIB_CTX.key_for(name)
                _lnames = _LIB_CTX.names_of(_lkey)
                if not (_lnames and _libpins.norm(nm) in _lnames):
                    _new = _LIB_CTX.name_for_num(_lkey, num)
                    if _new:
                        # ★ 退化库名守卫（见上面 `_dmode`/`_shape_block` 的注释）。
                        #   `letter`：我方读数有字母、库名纯数字 ⇒ 不改。
                        #   `known` ：同 `letter`，且**我方读数必须是库认识的名字**
                        #             （`VIN`/`TAB`/`SDA` ⇒ 拦；`VWPSTO2`/`IOW2T` 这类库
                        #             不认识的乱码 ⇒ 照旧让库来纠正，派生集需要）。
                        #   `known2`：`known` 加上「我方读数是 ≥2 字母的库认识名字 ⇒
                        #             库名有字母也拦」（官方 2 U3 的 `VCC`←`P67`）。
                        #   判据的唯一实现在 `libpins.degen_block`。
                        _guard = _shape_block or _libpins.degen_block(
                            _LIB_CTX, _dmode, nm, _new)
                        if _guard:
                            if os.environ.get("STAGE2_DBG_PINS") or LIB_PINS_DBG:
                                print("[libpins] 规则B 守卫挡下 元件=%r 号=%s 保留 %r"
                                      "（库名 %r，档 %s）"
                                      % (name, num, r["nm"] or "", _new, _dmode))
                        else:
                            nm = _new
                            _lib_named = True
                            if os.environ.get("STAGE2_DBG_PINS") or LIB_PINS_DBG:
                                print("[libpins] 规则B 元件=%r 号=%s 名改 %r → %r"
                                      % (name, num, r["nm"] or "", _new))
            # ★★★ 「迟到闸门」（下半）：到这里桩还没名字、库也没能补出名字，**才**丢。
            #     语义：`BOX_PIN_NEED_NAME_LATE` 时不再要求「读到的名字」，
            #     而是要求「有名字 —— 读到的或库里补的都算」；仍然**不**发射
            #     「拿号当名」的桩，因为号≠名必然 `name_mismatch` = fp+fn 双罚。
            if (ctype == "box" and BOX_PIN_NEED_NAME and BOX_PIN_NEED_NAME_LATE
                    and not r.get("nm") and not _lib_named
                    and (not _BOX_NEED_NAME_KEEPREAD or not r.get("kept", True))):
                if os.environ.get("STAGE2_DBG_PINS") or LIB_PINS_DBG:
                    print("[libpins] 迟到闸门 元件=%r 号=%s（号读到了、名没有、库也没补出）→ 丢"
                          % (name, num))
                continue
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

    # ★ `tp` 闸门小结（`STAGE2_TP_LEDGER=1` 时打一行；默认不打）。
    # 数字口径：`kept` = 通过文本闸门的 `tp` 元件，`dropped` = 被丢掉的
    # （两种原因：`unkeyed` 没绑到 TP 系位号；`no-evidence` 位号是融合层补的空壳，
    # 见 `TP_TRUST_EV`。**这是 `DROP_TYPES` 那条「盲目类型级丢弃」的替代品**）。
    if MERGE_TP and TP_LEDGER_ON and _tp_ledger:
        _kept = [r for r in _tp_ledger if "type" in r]
        _by: Dict[str, int] = {}
        for r in _kept:
            _by[r["type"]] = _by.get(r["type"], 0) + 1
        _why: Dict[str, int] = {}
        for r in _tp_ledger:
            if "drop" in r:
                _why[r["drop"]] = _why.get(r["drop"], 0) + 1
        print("[assemble] tp 闸门：保留 %d（%s），丢弃 %d（%s）"
              % (len(_kept),
                 " / ".join("%s %d" % (k, v) for k, v in sorted(_by.items())) or "-",
                 len(_tp_ledger) - len(_kept),
                 " / ".join("%s %d" % (k, v) for k, v in sorted(_why.items())) or "-"))
        TP_LEDGER[:] = _tp_ledger

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
                    continue
            if NET_DROP_UNKEYED and FKEY_RE.match(k):
                continue
            t = (k, num_of[(cid, idx)])
            if t not in seen:
                seen.add(t)
                refs.append(t)
        if syms_here:
            syms["net_%d" % j] = syms_here
        # ★ 边端点吸附到**本网自己的引脚点**（`STAGE2_EDGE_SNAP`，默认 0 = 关）。
        # 依据（官方尺，10GT + `work/_fast_out_kA`）：GT 的 639 条边**有 20.2% 的端点在
        # 本网某个引脚点 5px 内**（p25 9.0 / p50 42.0 / p75 74.0），而我们 842 条边只有
        # **2.1%**（p25 19.6 / p50 58.3）。机制：`extract_wires(occlude=[bbox_img])` 把元件框
        # 挖空，导线因此**断在框边**，而官方引脚点在「框边 ∓ 该符号的引脚长度」（12/15/20px），
        # 于是我们的边端点系统性差 12~20px —— 这正是 `just_over(118) + off_8_20(158)` 那两桶。
        # 只吸到**同一张网**的引脚（不是全图最近引脚），避免把端点拽到隔壁网上。
        if EDGE_SNAP > 0 and edges:
            _pp = []
            for _k, _n in refs:
                _p = ((pins.get(_k) or {}).get("pin_%s" % _n) or {}).get("point")
                if _p:
                    _pp.append((float(_p["x"]), float(_p["y"])))
            for _e in edges.values():
                for _i in (0, len(_e) - 1):
                    _q = (_e[_i]["x"], _e[_i]["y"])
                    _best = None
                    for _z in _pp:
                        _dd = math.hypot(_q[0] - _z[0], _q[1] - _z[1])
                        if _dd <= EDGE_SNAP and (_best is None or _dd < _best[0]):
                            _best = (_dd, _z)
                    if _best:
                        _e[_i]["x"] = round(float(_best[1][0]), 2)
                        _e[_i]["y"] = round(float(_best[1][1]), 2)
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
    _trace_dump("00.pre", out_nets, H, case_id)
    out_nets = attach_orphan_rails(out_nets, pins)
    out_nets = merge_shared_pins(out_nets)
    _trace_dump("04.snap3", out_nets, H, case_id)
    if EDGE_SNAP > 0:
        # ★ 端点吸附放在**最后**：这时 net 的成员已经定下来，同 net 里能拿来当
        # 结点的邻居最全（离线实验就是在最终文档上做的，顺序必须一致）。
        snap_edge_ends(out_nets, EDGE_SNAP, EDGE_SNAP_ANG)
    _trace_dump("01.snap3", out_nets, H, case_id)
    if NET_WELD > 0:
        # ★ 跨网焊接（默认关）必须在**吸附之后**：离线实验就是在吸附后的最终文档上量的。
        out_nets = weld_nets(out_nets, components, stage1.get("texts") or (), H, pins)
    _trace_dump("02.weld", out_nets, H, case_id)
    if NET_DROP_PWRONLY:
        # ★ 「整条网只有电源/地符号」的孤立网：GT 里不存在这种网（每个电源符号都挂在
        # 一条有真实元件的网络上），所以它只可能是 FP。**必须放在全部合并之后**——
        # `merge_by_symbol` / `merge_by_label` / `attach_orphan_rails` /
        # `merge_shared_pins` 都可能给一条孤网补上真实成员，早剪会误伤。
        # 实测（官方 9 张、融合层、`NET_DROP_UNKEYED=0`）：全留 64.8641
        # → 剪掉孤网 **66.3024**（netH 0.6394→0.6800，逐张不变差）。见文件顶注释。
        _kept: Dict[str, dict] = {}
        for _v in out_nets.values():
            _keys = [k for k, _p in _hg_refs(_v.get("hyperGraph", ""))]
            _tys = [str((components.get(k) or {}).get("type") or "").strip().lower()
                    for k in _keys]
            if _keys and all(t in POWER_TYPES for t in _tys):
                continue
            _kept["net_%d" % (len(_kept) + 1)] = _v
        out_nets = _kept
    if EDGE_SPLIT > 0 or EDGE_DEDUP > 0 or EDGE_DEDUP2 > 0:
        # ★ 边的「去重 + 结点切分 + 再去重」（`STAGE2_EDGE_SPLIT` / `STAGE2_EDGE_DEDUP` /
        #   `STAGE2_EDGE_DEDUP2`，见长注释）。
        # 必须放在**全部合并/焊接之后**：这时每条网里的边才齐，切点才找得全。
        # 只动 `edges`（首尾两点），`hyperGraph`/引脚一律不碰 ⇒ 元件/引脚/netH 通道逐位不变。
        split_dedup_edges(out_nets)
    if EDGE_SPLIT_PIN > 0:
        # ★ 引脚切分（`STAGE2_EDGE_SPLIT_PIN`，见长注释）：必须在切分/去重**之后**、
        #   自由端回缩**之前** —— 切出来的新端点在回缩那一步才算得上「结点」。
        split_edges_at_pins(out_nets, pins)
    if EDGE_SPLIT_XNET > 0:
        # ★ 跨网顶点切分（`STAGE2_EDGE_SPLIT_XNET`，见长注释）：同一位置 —— 切分/去重之后、
        #   回缩之前。`split_dedup_edges` 只看本网，漏掉「别的网的导线搭在这条线上」的切点。
        split_edges_at_xnet(out_nets)
    _trace_dump("05.split", out_nets, H, case_id)
    if EDGE_TRIM > 0:
        # ★ 自由端回缩（`STAGE2_EDGE_TRIM`，默认 3.0 px，见长注释）：必须在切分/去重
        #   **之后**跑 —— 结点是在切分那一步才定下来的，先回缩会把结点也拉动。
        trim_free_ends(out_nets)
    _trace_dump("06.trim", out_nets, H, case_id)
    if EDGE_CORNER > 0:
        # ★ L/T 拐角合拢（`STAGE2_EDGE_CORNER`，默认 12）：必须跑在 `extend_ends_to_evidence` /
        #   `extend_edge_ends` **之前** —— 先把各条腿搬回真结点，后面两个外推算子才有正确起点
        #   （离线实测：先 corner 80.75，夹在中间 80.50，放最后 80.48）。
        snap_ends_to_crossings(out_nets)
    _trace_dump("07.extfree", out_nets, H, case_id)
    if EDGE_EXT_FREE > 0:
        # ★ 自由端外推（`STAGE2_EDGE_EXT_FREE`）：必须跑在 `extend_edge_ends` **之前** ——
        #   离线实验里顺序反过来（ext;overlap;extfree）会把已对齐的端点再推过头，收益从 +0.42 掉到 +0.13。
        extend_free_ends(out_nets, pins)
    _trace_dump("08.extany", out_nets, H, case_id)
    if EDGE_EXT_ANY > 0:
        # ★ 证据外推（`STAGE2_EDGE_EXT_ANY`）：在「同网他边」外推之前跑，先用全图几何把端点拉到交叉/汇合/引脚根。
        extend_ends_to_evidence(out_nets, pins)
    _trace_dump("09.ext", out_nets, H, case_id)
    if EDGE_EXT > 0:
        # ★ 端点外推（`STAGE2_EDGE_EXT`）：回缩之后跑（先治过冲，再补不足）。
        extend_edge_ends(out_nets)
    _trace_dump("10.pinsnap_box", out_nets, H, case_id)
    if PIN_SNAP:
        # ★ 自由端吸附到引脚（`STAGE2_PIN_SNAP`）：放在所有外推**之后** —— 否则外推会把
        #   刚吸附好的端点又推走。
        snap_ends_to_pins(out_nets, pins)
    if END_TO_BOX:
        # ★ 引脚端推进到元件框边（`STAGE2_END_TO_BOX`，见长注释）：放在**所有回缩/外推之后** ——
        #   回缩（trim）会把自由端往里收 3px，先推进会被它再收回去；GT 的端点就落在框边上。
        extend_ends_to_box(out_nets, pins, components)
    _trace_dump("11.overlap", out_nets, H, case_id)
    if EDGE_OVERLAP > 0:
        # ★ 近共线重叠边合并（`STAGE2_EDGE_OVERLAP`，默认 10 度）：必须最后跑 —— 它会改变边的跨度，
        #   放在切分/回缩之前会把切点与自由端判定带歪。
        merge_overlapping_edges(out_nets)
    _trace_dump("12.final", out_nets, H, case_id)
    if PIN_LEAD_ATTACH or PIN_LEAD_MOVE:
        # ★ 2-pin 符号的「引脚引线」吸附/迁移：必须放在**最末尾** —— 离线取证就是在
        #   出口文档（几何全部定型）上做的，落在这里才能与离线读数逐位一致。
        #   只写 `hyperGraph`，`edges`/引脚一律不碰 ⇒ 元件/引脚/导线三通道逐位不变。
        attach_stamped_leads(out_nets, pins, components)
    _pidx: Dict[str, Dict[str, str]] = {}
    for (_c, _k), _pk in pin_pk.items():
        _pidx.setdefault(_c, {})[str(_k)] = _pk
    return {"case": case_id, "components": components, "pins": pins,
            "nets": out_nets, "__pin_index": _pidx}


def _trace_pts(raw):
    """边在装配层内部的形态是 `[[x, y], ...]`（图像坐标）或 `{'points': [...]}`。"""
    if isinstance(raw, dict):
        raw = raw.get("points")
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for p in raw:
        if isinstance(p, dict):
            if "x" not in p:
                return None
            out.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
        else:
            return None
    return out


_TRACE_DIR = env_str("STAGE2_TRACE_DUMP", "").strip()
_TRACE_N = [0]
_TRACE_CASE = [""]


def _trace_dump(tag, out_nets, h, case_id=""):
    """提取器内部端点归因（§5.126）：把每一步之后的边端点快照写到
    `$STAGE2_TRACE_DUMP/<case>.<NN>.<tag>.json`。

    只在 `STAGE2_TRACE_DUMP` 设了目录时才写，默认不设 ⇒ 零开销、产物逐位不变。
    坐标是**图像坐标**（左上原点），读的时候按 `y_doc = h - y_img` 翻到文档坐标。
    """
    if not _TRACE_DIR:
        return
    try:
        if case_id:
            _TRACE_CASE[0] = case_id
        rows = []
        for nk, v in (out_nets or {}).items():
            e = (v or {}).get("edges") or {}
            vals = e.values() if isinstance(e, dict) else e
            for ei, edge in enumerate(vals):
                p = _trace_pts(edge)
                if p and len(p) >= 2:
                    rows.append([nk, ei, list(p[0]), list(p[-1]), len(p)])
        _TRACE_N[0] += 1
        os.makedirs(_TRACE_DIR, exist_ok=True)
        fp = os.path.join(_TRACE_DIR, "%s.%02d.%s.json"
                          % (_TRACE_CASE[0], _TRACE_N[0], tag))
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump({"tag": tag, "h": h, "edges": rows}, fh)
    except Exception:
        pass


def _trace_step(tag, out_nets, h, case_id=""):
    _trace_dump(tag, out_nets, h, case_id)


RAIL_NAMES = frozenset((
    "gnd", "gnda", "gndd", "agnd", "dgnd", "pgnd", "vss", "vssa",
    "vcc", "vdd", "vddd", "vdda", "vddio", "3v3", "5v", "12v",
    "vbat", "vbus", "vin", "vout"))
RAIL_ATTACH = env_on("STAGE2_RAIL_ATTACH", "1")
# ★ 2-pin 符号的「引脚引线」吸附（`STAGE2_PIN_LEAD_ATTACH`，默认 1）。
#
# 机制：`PIN_STAMP_TYPES` 那几类 2 脚符号的引脚点不是检出来的，而是**按 bbox 轴冲压
# 出来的**（`PIN_STAMP_DEFAULT_SPAN = 20.0`，即框外 20px）。而导线追踪把元件框挖空
# （`extract_wires(occlude=[bbox_img])`），导线往往**停在框边**，于是「框外 20px 的
# 引脚点」与「框内的导线端点」差出 15~20px；装配层要求几何**接触**才把引脚算进网络
# ⇒ 这种脚**一个网都不进**，它对应的 GT 网络白少一个成员（Dice 的分子分母同时受损）。
#
# 依据（`work\_pinattach.py`：在整份出口文档上离线改 `nets[*].hyperGraph`，再用
# `eda-pcb\tools\eval\fast.py --score-only` 秒级重打分；官方 10 张、交付态 83.23）：
#   全图**只有 9 个脚**不属于任何网络（说明成员判定本身已经很好），其中 5 个是 2-pin
#   符号且最近导线 ≤30px：`LED1.1`(case 10，19.1px/29.1°)、`LED1.1`(case 7，
#   17.7px/31.3°)、`LED1.2`×2(32.2/32.4px)、`R2.1`(27.2px)。
#   `R=25 + 夹角≤35°` **只命中那两条 LED1.1** ⇒ 83.23 → **83.33**（netH 0.7806→0.7836，
#   +0.10）；把 R 放到 30 会把 27.2px 的 `R2.1` 也吸进来 ⇒ 只剩 +0.02；
#   **只按距离、不按角度**在 20 张池子上会把一堆「导线从符号**背后**经过」
#   （夹角≈88~180°）的脚也吸上去 ⇒ 10GT 塌到 75.2（−8.0）**⇒ 角度闸门是必须的**。
#   20 张回归（10GT + 池，`work\_fast_out_ec20on`）：角度闸门下池子里的 24 个悬空
#   2-pin 脚**夹角全是 88~180°，一条都不吸** ⇒ 对本规则**零风险**。
# 阈值取自现成常量、不是拟合：`PIN_LEAD_R = PIN_STAMP_DEFAULT_SPAN(20) + cfg.line_tol(5)`；
# `PIN_LEAD_DEG` 只要求「导线大致沿引脚外指方向离开」。
# `=0` 关掉做对照。
#
# ★★ 2026-10-16 扩展（§5.106）：同一套「冲压引线」几何还解释第二种错误 ——
# **纯无线网**（`edges` 为空的网络）里的脚。`LED1.1`/`LED1.2`/`R2.1`/`R1.2`/`R5.2`
# 这类脚被测出来挂在一条**一条边都没有**的网上（上游的导线追踪没给这条网任何几何），
# 而它的冲压引线离**别的**网的导线只有 24~42px ⇒ 成员判定错网。修法有两半：
#   ① `PIN_LEAD_MOVE`：源网络**没有任何边**且脚离别网导线 ≤ `PIN_LEAD_MOVE_R`（夹角同上）
#      ⇒ 把引用搬过去（搬空的无线网随后被删掉）；
#   ② `PIN_LEAD_R` 从 25 提到 40：冲压跨度 20px + 导线缩短 ~20px，三个「无线网的脚」
#      之外还漏着 3 个 27~33px 的悬空脚（`R2.1` case 6、`LED1.2` case 7/10）。
# 离线读数（`tools/diag/_pinattach.py --mode move`，官方 10 张、交付态 83.33）：
#   只搬 6 条（r=25/move_r=50）⇒ **84.00**；再把那 3 个悬空脚按 R=45 吸上 ⇒ **84.18**；
#   `--deg 35` 与 `--deg 999` 逐位相同（6 条搬移的夹角都 ≤35°）；R 从 45 放到 80
#   一条新的都不加（搬移距离上界就是 42.0px）⇒ 阈值不敏感。
# `=0` 各自关掉做对照。
PIN_LEAD_ATTACH = env_on("STAGE2_PIN_LEAD_ATTACH", "1")
PIN_LEAD_MOVE = env_on("STAGE2_PIN_LEAD_MOVE", "1")
PIN_LEAD_R = float(os.environ.get("STAGE2_PIN_LEAD_R", "40.0"))
PIN_LEAD_MOVE_R = float(os.environ.get("STAGE2_PIN_LEAD_MOVE_R", "50.0"))
PIN_LEAD_DEG = float(os.environ.get("STAGE2_PIN_LEAD_DEG", "35.0"))
# ★ 「地分模式」守卫（`STAGE2_RAIL_SPLIT`，默认开）：owner m23945/m02404 的要求是
# **内部按本质模式把 GND 家族拆开跑，输出时再折叠回 `gnd`**（交付 JSON 的 type 一个
# 字节都不改）。这里落地的是拆分后的**唯一一条会改变结构**的判据：跨网焊接
# （`weld_nets`）不得把「本质模式不相交」的两条地网焊成一条 —— 例如 `GND` 与
# `AGND`、`AGND` 与 `DGND`。依据：官方数据集里 `gnd` 只有**一个** type（`10GT/type_list.md`），
# 变体只能从**引脚名/标签**看出来；而 `attach_orphan_rails` 本来就是「同名才挂」
# （精确按 `pinname` 匹配），所以真正会混淆两个变体的只剩几何焊接这一条路。
# 语料事实（2026-10-15，210 个 GT 文件里的网名）：`GND` 52 次、`VSS` 2 次、`PGND` 2 次，
# **`AGND`/`DGND` 0 次** ⇒ 本守卫在我们的语料上是**中性**的（A/B 逐位相同，见 RESULTS §5.27），
# 它的价值只在「测试集真出现变体时不并网」。模式字母：`a` 模拟 / `d` 数字 / `p` 功率 / `e` 大地。
RAIL_SPLIT = env_on("STAGE2_RAIL_SPLIT", "1")
_RAIL_CORES = ("gnd", "vss")
_RAIL_LETTERS = "adpe"
_N_RAIL_BLOCK = [0]


def rail_mode(name) -> tuple:
    """轨名的 `(家族, 本质模式)`；非地家族返回 `None`。

    `rail_mode("GND") == ("gnd", "")`、`rail_mode("AGND") == ("gnd", "a")`、
    `rail_mode("VSSA") == ("gnd", "a")`、`rail_mode("VCC") is None`。
    ★ 这个模式**只进内部判据与账本**，绝不写进交付文档（type 仍折叠为 `gnd`）。
    """
    t = re.sub(r"[^a-z]", "", str(name or "").lower())
    if not t:
        return None
    for core in _RAIL_CORES:
        i = t.find(core)
        if i < 0:
            continue
        pre, suf = t[:i], t[i + len(core):]
        letter = (pre[-1:] if pre else "") or (suf[:1] if suf else "")
        return ("gnd", letter if letter in _RAIL_LETTERS else "")
    return None


def _net_rail_modes(rec: dict, pins: Dict[str, dict]) -> set:
    """一条网的「地模式」集合（从它的引脚名推）。空集 = 这条网没有地家族引脚。"""
    if not pins:
        return set()
    out = set()
    for comp, num in _hg_refs((rec or {}).get("hyperGraph", "")):
        m = rail_mode(_pinname(pins, comp, num))
        if m:
            out.add(m[1])
    return out


def _net_seg_pairs(net: dict) -> List[tuple]:
    """网络里每条边的 (首点, 尾点)（官方 `net_line_f1` 也只取这两点）。"""
    out: List[tuple] = []
    for pts in (net.get("edges") or {}).values():
        if isinstance(pts, list) and len(pts) >= 2:
            try:
                out.append(((float(pts[0]["x"]), float(pts[0]["y"])),
                            (float(pts[-1]["x"]), float(pts[-1]["y"]))))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _bridge_hits(a, b, boxes, pad: float) -> bool:
    """端点 a→b 的直线（取 1/6…5/6 五个采样点）是否落在任一框内。"""
    if not boxes:
        return False
    for t in range(1, 6):
        f = t / 6.0
        x = a[0] + (b[0] - a[0]) * f
        y = a[1] + (b[1] - a[1]) * f
        for (x1, y1, x2, y2) in boxes:
            if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad:
                return True
    return False


def weld_nets(nets: Dict[str, dict], components: Dict[str, dict],
              texts: Sequence = (), H: float = 0.0,
              pins: Dict[str, dict] = None) -> Dict[str, dict]:
    """★ 跨网焊接（`STAGE2_NET_WELD`）：把「端点相距 6~16px、连线穿文字框、不穿元件框、
    且两段不平行同向」的两条网络并成一条。完整依据与逐条件读数见 `NET_WELD` 的长注释。

    只动网络结构（`_refs`/`edges` 取并集），不碰任何引脚取值 ⇒ 引脚/元件通道逐位不变。

    `pins`（可选）只服务 `STAGE2_RAIL_SPLIT` 地模式守卫：给了它才会在焊接前比两条网的
    「地模式集合」，避免把 `GND` 与 `AGND` 这类**本质不同的地**焊成一条（见 `rail_mode`）。
    """
    if NET_WELD <= 0 or len(nets) < 2:
        return nets
    segs = {k: _net_seg_pairs(v) for k, v in nets.items()}
    keys = [k for k in nets if segs[k]]
    if len(keys) < 2:
        return nets
    tboxes: List[tuple] = []
    for t in (texts or ()):
        b = (t or {}).get("bbox_img") if isinstance(t, dict) else None
        if b and len(b) == 4:
            # stage1 的 `bbox_img` 是图像帧（y 向下），文档帧是 y 向上。
            tboxes.append((float(b[0]), H - float(b[3]), float(b[2]), H - float(b[1])))
    cboxes: List[tuple] = []
    for c in (components or {}).values():
        b = (c or {}).get("bbox")
        if b and len(b) == 4:
            cboxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    gap, gmin = NET_WELD, NET_WELD_MIN
    _nblock0 = _N_RAIL_BLOCK[0]
    parent: Dict[str, str] = {k: k for k in nets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_join = 0
    for _i in range(len(keys)):
        for _j in range(_i + 1, len(keys)):
            ka, kb = keys[_i], keys[_j]
            if find(ka) == find(kb):
                continue
            best = None
            bd = gap + 1.0
            for ea in segs[ka]:
                for eb in segs[kb]:
                    for ua in ea:
                        for ub in eb:
                            d = math.hypot(ua[0] - ub[0], ua[1] - ub[1])
                            if d < bd:
                                bd, best = d, (ea, eb, ua, ub)
            if not best or bd > gap or bd < gmin:
                continue
            ea, eb, ua, ub = best
            va = ea[0] if ua is ea[1] else ea[1]
            vb = eb[0] if ub is eb[1] else eb[1]
            da = (ua[0] - va[0], ua[1] - va[1])
            db = (ub[0] - vb[0], ub[1] - vb[1])
            na = math.hypot(da[0], da[1]) or 1.0
            nb = math.hypot(db[0], db[1]) or 1.0
            cos = (da[0] * db[0] + da[1] * db[1]) / (na * nb)
            if cos > NET_WELD_COS:
                continue
            # ★ 地模式守卫（`STAGE2_RAIL_SPLIT`，默认开）：两条网各自的「地模式集合」都非空、
            # 且**不相交** ⇒ 它们是本质不同的地（`{}`↔`{a}` 的 GND vs AGND、`{a}`↔`{d}` 的
            # AGND vs DGND），**拒绝焊接**。集合相交（如 `{a,d}` 与 `{a}`）或有一侧无地家族
            # 引脚（空集）⇒ 不设限，保持原有行为。只做否决、不改任何取值 ⇒ 关掉开关时逐位相同。
            if RAIL_SPLIT and pins:
                _ma = _net_rail_modes(nets[ka], pins)
                _mb = _net_rail_modes(nets[kb], pins)
                if _ma and _mb and not (_ma & _mb):
                    _N_RAIL_BLOCK[0] += 1
                    # 「不删除只削弱」的账本口径（工程哲学 §94：merge/split 必须留事件）：
                    # 这次**没有发生**的合并要记一笔，下游才有机会复核。
                    try:
                        from cvparts.hypo import LEDGER as _LG
                        _LG.record("merge", "weld:%s+%s" % (ka, kb), before=[ka, kb],
                                   reason="地模式不相交 %r vs %r，拒绝焊接"
                                          % (sorted(_ma), sorted(_mb)),
                                   by="stage2.assemble.weld_nets")
                    except Exception:
                        pass
                    continue
            if not _bridge_hits(ua, ub, tboxes, NET_WELD_TPAD):
                continue
            if _bridge_hits(ua, ub, cboxes, 0.0):
                continue
            ra, rb = find(ka), find(kb)
            if ra != rb:
                parent[rb] = ra
                n_join += 1
    if _N_RAIL_BLOCK[0] > _nblock0:
        print("[assemble] 地模式守卫（RAIL_SPLIT）：拒绝焊接 %d 处不同本质模式的地网"
              % (_N_RAIL_BLOCK[0] - _nblock0,))
    if not n_join:
        return nets
    groups: Dict[str, List[str]] = {}
    for k in nets:
        groups.setdefault(find(k), []).append(k)
    out = _merge_nets(nets, groups)
    # ★★ `_merge_nets` 只把 `_refs`/`edges` 并起来，**不重建 `hyperGraph`**
    # （`base = dict(nets[root])` 把 root 的 `hyperGraph` 原样留下）。这条在
    # `build_doc` 的 `out_nets` 阶段之前无所谓 —— 那里反正会在 :1268 用 `_refs`
    # 重建一次；但 `weld_nets` 是**在那之后**调用的，此时 `_refs` 已不是出口、
    # 评测只读 `hyperGraph` ⇒ 并出来的网「边并了、成员只剩 root 一份」。
    # 实测（`work/_weldx2.py` case 10）：探针合并 `+3.77` vs 管线 `+1.68`，
    # 差异全部来自这里；补上重建后两者成员集合逐位一致。
    for _v in out.values():
        _refs = [tuple(_r) for _r in (_v.get("_refs") or [])]
        if not _refs:
            _refs = _hg_refs(_v.get("hyperGraph", ""))
        _v["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in _refs) + ")"
    return out


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


def _nearest_lead_seg(px: float, py: float, segs: Dict[str, list],
                      limit: float):
    """`(px,py)` 到 `segs`（`{网络名: [((x,y),(x,y)), ...]}`）里最近线段的距离与落点。

    返回 `(d, 网络名, 落点x, 落点y)`；`limit` 内一条都没有就返回 `None`。
    与离线探针 `tools/diag/_pinattach.py` 的 `seg_closest` + 取最小同式。
    """
    best = None
    for nm, seg in segs.items():
        for p0, p1 in seg:
            vx, vy = p1[0] - p0[0], p1[1] - p0[1]
            l2 = vx * vx + vy * vy
            if l2 <= 1e-9:
                qx, qy = p0
            else:
                t = ((px - p0[0]) * vx + (py - p0[1]) * vy) / l2
                t = max(0.0, min(1.0, t))
                qx, qy = p0[0] + t * vx, p0[1] + t * vy
            d = math.hypot(px - qx, py - qy)
            if d > limit:
                continue
            if best is None or d < best[0]:
                best = (d, nm, qx, qy)
    return best


def _lead_axis_deg(ox: float, oy: float, on: float,
                   px: float, py: float, qx: float, qy: float) -> float:
    """引脚**外指**方向 `(ox,oy)` 与「引脚 -> 导线落点」方向的夹角（度）。

    退化情形（框心与脚重合、落点与脚重合）返回 `-1.0`：与历史实现一样**放行**。
    """
    if on <= 1e-6:
        return -1.0
    wn = math.hypot(qx - px, qy - py)
    if wn <= 1e-6:
        return -1.0
    cos = (ox * (qx - px) + oy * (qy - py)) / (on * wn)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def attach_stamped_leads(nets: Dict[str, dict], pins: Dict[str, dict],
                         components: Dict[str, dict]) -> int:
    """把**冲压引脚**（2-pin 符号，见 `PIN_STAMP_TYPES`）按「冲压引线」挂/搬到最近的网络。

    两半（长注释见模块常量 `PIN_LEAD_ATTACH` / `PIN_LEAD_MOVE`）：
      ① **迁移**：脚挂在一条**一条边都没有**的网上，而它离别网的导线 ≤ `PIN_LEAD_MOVE_R`
         ⇒ 把引用搬过去；被搬空的无线网随后删掉（它本来就是个空预测）。
      ② **吸附**：脚**不属于任何网络**，最近导线 ≤ `PIN_LEAD_R` ⇒ 挂上去。
    两半都要求导线大致沿引脚**外指**方向离开（夹角 ≤ `PIN_LEAD_DEG`）。
    **只写 `hyperGraph`**：`edges`、引脚坐标一律不碰。

    返回处理过的脚数（迁移 + 吸附）。
    """
    if not nets or not pins or (PIN_LEAD_R <= 0 and PIN_LEAD_MOVE_R <= 0):
        return 0
    member: Dict[tuple, str] = {}
    for nm, rec in nets.items():
        for ref in _hg_refs((rec or {}).get("hyperGraph", "")):
            member[ref] = nm
    segs: Dict[str, list] = {}
    for nm, rec in nets.items():
        seg = []
        for _k, e in ((rec or {}).get("edges") or {}).items():
            if isinstance(e, (list, tuple)) and len(e) == 2:
                try:
                    seg.append(((float(e[0]["x"]), float(e[0]["y"])),
                                (float(e[1]["x"]), float(e[1]["y"]))))
                except (KeyError, TypeError, ValueError, IndexError):
                    pass
        if seg:
            segs[nm] = seg
    if not segs:
        return 0

    def _stamped() -> Iterator[Tuple[str, str, float, float, float, float]]:
        """产出 `(元件, 引脚号, 脚x, 脚y, 框心x, 框心y)`。"""
        for comp, group in (pins or {}).items():
            if not isinstance(group, dict):
                continue
            ctype = str((components.get(comp) or {}).get("type") or "").strip().lower()
            if ctype not in PIN_STAMP_TYPES:
                continue
            box = (components.get(comp) or {}).get("bbox")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            try:
                cx = (float(box[0]) + float(box[2])) / 2.0
                cy = (float(box[1]) + float(box[3])) / 2.0
            except (TypeError, ValueError):
                continue
            for pk, pr in group.items():
                pt = (pr or {}).get("point") or {}
                if pt.get("x") is None or pt.get("y") is None:
                    continue
                try:
                    px, py = float(pt["x"]), float(pt["y"])
                except (TypeError, ValueError):
                    continue
                yield (comp, str(pk).replace("pin_", ""), px, py, cx, cy)

    def _write(nm: str, refs) -> None:
        nets[nm]["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"

    # ① 迁移：源网络没有任何边（纯无线网）⇒ 成员判定没有几何依据
    moved = 0
    dropped = 0
    _mdbg = []
    if PIN_LEAD_MOVE_R > 0:
        emptied = set()
        for comp, num, px, py, cx, cy in _stamped():
            ref = (comp, num)
            cur = member.get(ref)
            if cur is None or cur in segs:
                continue                        # 不属任何网（交给 ②）／本网有导线 ⇒ 不动
            hit = _nearest_lead_seg(px, py, segs, PIN_LEAD_MOVE_R)
            if hit is None:
                continue
            d, tgt, qx, qy = hit
            on = math.hypot(px - cx, py - cy)
            deg = _lead_axis_deg(px - cx, py - cy, on, px, py, qx, qy)
            if deg > PIN_LEAD_DEG:
                continue
            old = _hg_refs(nets[cur].get("hyperGraph", ""))
            if ref in old:
                old.remove(ref)
            _write(cur, old)
            refs = _hg_refs(nets[tgt].get("hyperGraph", ""))
            refs.append(ref)
            _write(tgt, refs)
            member[ref] = tgt
            emptied.add(cur)
            moved += 1
            _mdbg.append((comp, num, cur, tgt, d, deg))
        for nm in list(emptied):
            rec = nets.get(nm) or {}
            if _hg_refs(rec.get("hyperGraph", "")) or (rec.get("edges") or {}):
                continue
            nets.pop(nm, None)
            dropped += 1
        if moved:
            try:
                print("[assemble] 引脚引线迁移：%d 个 2-pin 引脚离开了「没有边的网络」"
                      "（删掉 %d 个被搬空的网）：%s"
                      % (moved, dropped,
                         "  ".join("%s.%s %s->%s d=%.1f deg=%.1f" % t for t in _mdbg[:6])))
            except Exception:
                pass

    # ② 吸附：不属于任何网络的冲压脚
    added = 0
    _dbg = []
    if PIN_LEAD_R > 0:
        for comp, num, px, py, cx, cy in _stamped():
            ref = (comp, num)
            if ref in member:
                continue
            hit = _nearest_lead_seg(px, py, segs, PIN_LEAD_R)
            if hit is None:
                continue
            d, tgt, qx, qy = hit
            on = math.hypot(px - cx, py - cy)
            deg = _lead_axis_deg(px - cx, py - cy, on, px, py, qx, qy)
            if deg > PIN_LEAD_DEG:
                continue
            refs = _hg_refs(nets[tgt].get("hyperGraph", ""))
            refs.append(ref)
            _write(tgt, refs)
            member[ref] = tgt
            _dbg.append((comp, num, d, deg, tgt))
            added += 1
    if added:
        try:
            print("[assemble] 引脚引线吸附：把 %d 个悬空的 2-pin 符号引脚挂到了最近的网络：%s"
                  % (added, "  ".join("%s.%s d=%.1f deg=%.1f -> %s" % t for t in _dbg[:6])))
        except Exception:
            pass
    return moved + added


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
        rec["_refs"] = refs          # ★ 与 hyperGraph 保持一致：`dict(nets[members[0]])`
        rec["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
        out["net_%d" % (len(out) + 1)] = rec
    return out


# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails = []

    def chk(cond, msg):
        chk.n += 1
        if not cond:
            fails.append(msg)

    chk.n = 0

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
    # ★ 夹具这一根是「号没读到、名也没有」的 box 桩。**默认口径会把它丢掉**，两条闸门
    #   各自独立：① `STAGE2_BOX_PIN_DROP_UNREAD=mixed`（这颗 IC 有别的脚读到了号 ——
    #   `case 1 U1` 那批 `pin.missing` 就是这条丢的）；② 迟到闸门
    #   `STAGE2_BOX_PIN_NEED_NAME=2`（没有名字、库也补不出名字）。把两条都关掉，它才按
    #   「位置号当号、号当名」出现 —— 那是 2026-10 前的老口径，旧断言写的就是那个。
    _drop_box = BOX_PIN_DROP_UNREAD or BOX_PIN_NEED_NAME
    chk(("pin_2" in doc["pins"]["U1"]) == (not _drop_box),
        "缺号/缺名 box 桩的闸门口径（DROP_UNREAD=%r MIXED_ONLY=%r NEED_NAME=%r LATE=%r）"
        % (BOX_PIN_DROP_UNREAD, BOX_PIN_DROP_MIXED_ONLY, BOX_PIN_NEED_NAME,
           BOX_PIN_NEED_NAME_LATE))
    if not _drop_box:
        chk(doc["pins"]["U1"]["pin_2"]["pinname"] == "2", "缺名回退引脚号")
    n1 = doc["nets"]["net_1"]
    chk(set(n1["hyperGraph"].strip("()").split(",")) == {"R1.1", "U1.1"},
        "hyperGraph: %r" % (n1["hyperGraph"],))
    chk(len(n1["edges"]) == 2, "net_1 边数: %d" % len(n1["edges"]))
    # ★ 引脚点是**盖章**出来的（`bbox ± 桩长`：R1 左 10-10=0、右 30+10=40 —— 两脚件的
    #   10px 正是源图约定，见 RESULTS §5.18 的 220 case 统计），`STAGE2_EDGE_SNAP`
    #   （默认 10）再把边端点吸到**本网自己的引脚点**上 ⇒ 边首点 = 那个盖章点。
    _e0 = n1["edges"]["edge_1"][0]
    _want = (doc["pins"]["R1"]["pin_1"]["point"] if EDGE_SNAP > 0
             else {"x": 8.0, "y": 25.0})
    chk(_e0 == _want, "边首点: %r（期望 %r，EDGE_SNAP=%r）" % (_e0, _want, EDGE_SNAP))

    # ★ 边的「去重 + 结点切分」（`STAGE2_EDGE_SPLIT` / `STAGE2_EDGE_DEDUP`）：
    #   一条横母线 + 一条垂直抽头（切点）+ 一条反向重复边 ⇒ 去重 1、切分 1、得 3 条边。
    #   `EDGE_SPLIT_COL` 保证共线孪生邻居不当切点（夹具里没有共线邻居）。
    _t = {"net_1": {"edges": {
        "edge_1": [{"x": 0.0, "y": 50.0}, {"x": 100.0, "y": 50.0}],
        "edge_2": [{"x": 40.0, "y": 50.0}, {"x": 40.0, "y": 80.0}],
        "edge_3": [{"x": 100.0, "y": 50.0}, {"x": 0.0, "y": 50.0}],
    }}}
    _st = split_dedup_edges(_t)
    if EDGE_SPLIT > 0 or EDGE_DEDUP > 0:
        chk(_st["dup"] == 1, "边去重计数: %r" % (_st,))
        chk(_st["split"] == 1, "结点切分计数: %r" % (_st,))
        _pts = sorted((e[0]["x"], e[0]["y"], e[1]["x"], e[1]["y"])
                      for e in _t["net_1"]["edges"].values())
        chk(_pts == [(0.0, 50.0, 40.0, 50.0), (40.0, 50.0, 40.0, 80.0),
                     (40.0, 50.0, 100.0, 50.0)], "切分后的边集合: %r" % (_pts,))
        chk(all(len(e) == 2 for e in _t["net_1"]["edges"].values()),
            "切出来的每条边只留两个点")

    # ★ 第二遍去重（`STAGE2_EDGE_DEDUP2`）：两条**部分重合**的长边（第一遍去不掉，
    #   因为端点不同）在同一批切点上会被切成同一批小段 ⇒ 切分后必须再去一次重。
    #   夹具：短母线 (0,50)-(100,50) + 长母线 (0,50)-(140,50) + 抽头 (100,50)-(100,80)
    #   + 反向重复边 (100,50)-(0,50) ⇒ 去重 1（第一遍）、切分 1（长母线在抽头处断开）、
    #   再去重 1（两条短母线小段重合），最后剩 3 条边。
    _t2 = {"net_1": {"edges": {
        "edge_1": [{"x": 0.0, "y": 50.0}, {"x": 100.0, "y": 50.0}],
        "edge_2": [{"x": 0.0, "y": 50.0}, {"x": 140.0, "y": 50.0}],
        "edge_3": [{"x": 100.0, "y": 50.0}, {"x": 100.0, "y": 80.0}],
        "edge_4": [{"x": 100.0, "y": 50.0}, {"x": 0.0, "y": 50.0}],
    }}}
    _st2 = split_dedup_edges(_t2)
    if EDGE_SPLIT > 0 and EDGE_DEDUP > 0 and EDGE_DEDUP2 > 0:
        chk(_st2["dup"] == 1 and _st2["split"] == 1 and _st2["dup2"] == 1,
            "第二遍去重计数: %r" % (_st2,))
        _pts2 = sorted((e[0]["x"], e[0]["y"], e[1]["x"], e[1]["y"])
                       for e in _t2["net_1"]["edges"].values())
        chk(_pts2 == [(0.0, 50.0, 100.0, 50.0), (100.0, 50.0, 100.0, 80.0),
                      (100.0, 50.0, 140.0, 50.0)], "再去重后的边集合: %r" % (_pts2,))

    # ★ 地模式守卫（`STAGE2_RAIL_SPLIT`）：两条**反向共线**、端点相距 10px 的地网，中间跨一条
    #   文字框 —— 这正是 `weld_nets` 会焊的那一形态。两侧引脚名同为 `GND` ⇒ 应并成 1 条；
    #   一边 `GND` 一边 `AGND` ⇒ **本质不同的地**，必须拒焊、仍是 2 条。模式由引脚名推，
    #   输出里不留任何字段（`type` 仍折叠回 `gnd`）。
    def _railfix(na, nb):
        _n = {
            "net_1": {"hyperGraph": "(U1.1)", "_refs": [["U1", "1"]], "pins": [],
                      "edges": {"edge_1": [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}]}},
            "net_2": {"hyperGraph": "(U2.1)", "_refs": [["U2", "1"]], "pins": [],
                      "edges": {"edge_1": [{"x": 60.0, "y": 0.0}, {"x": 110.0, "y": 0.0}]}},
        }
        _p = {"U1": {"pin_1": {"pinname": na}}, "U2": {"pin_1": {"pinname": nb}}}
        _t = [{"bbox_img": [48.0, 95.0, 62.0, 105.0], "text": nb}]
        return weld_nets(_n, {}, _t, 100.0, _p)

    chk(rail_mode("AGND") == ("gnd", "a") and rail_mode("GND") == ("gnd", "")
        and rail_mode("VSSA") == ("gnd", "a") and rail_mode("PGND") == ("gnd", "p")
        and rail_mode("VCC") is None and rail_mode("") is None, "rail_mode 映射")
    if NET_WELD > 0:
        chk(len(_railfix("GND", "GND")) == 1, "同模式地网应焊成 1 条")
        _rg = _railfix("GND", "AGND")
        _ra = _railfix("AGND", "DGND")
        _rv = _railfix("GND", "VCC")
        if RAIL_SPLIT:
            chk(len(_rg) == 2, "GND 与 AGND 不得焊接（RAIL_SPLIT=1）: %r" % (list(_rg),))
            chk(len(_ra) == 2, "AGND 与 DGND 不得焊接（RAIL_SPLIT=1）: %r" % (list(_ra),))
        else:
            chk(len(_rg) == 1, "RAIL_SPLIT=0 时应恢复焊接: %r" % (list(_rg),))
        chk(len(_rv) == 1, "非地家族一侧（VCC）不设限: %r" % (list(_rv),))

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

    print("自检 %d 项：%s" % (chk.n, "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
