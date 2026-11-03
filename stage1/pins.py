"""第一阶段：引脚点提取（纯几何，不训练、不 OCR）。

**为什么必须有这个模块**：官方评分里 pin 0.25 + net_hypergraph 0.35 + net_line 0.10
= **0.70** 全部以「引脚点」为最小单位，而 stage1 之前只输出元件框、一个引脚点都没有
—— 这三块分数在结构上就拿不到。

## 几何依据（220 张 GT 实测，`work\\gt_pin_geom.py`）

* 引脚**从不在框内**（0.0%）；
* 引脚到框的距离 p25 / p50 / p75 = 9 / 10 / 10 px，<=12px 占 **86.1%**
  ⇒ 默认就是「**框外 10px**」；
* `box`（IC）的引脚贴在左右两条**长边**外侧、沿边分布（left 4268 / right 3577）；
* 两脚元件（r/c/l/d/led）的两个引脚在**长轴方向**、离中心 `L/2 + 10`：
  GT 例 R5 bbox [208,340,216,360] → 引脚 y=330/370，中心 y=350，20/2+10 = 20 ✓；
  `∅202` 方框 10×10 → 引脚离中心 15 = 10/2+10 ✓；
* 一脚符号（gnd/v/net_input/testpoint）只有 1 个脚，上下左右不定。

## 主判据：导线触到框边的位置

原理图的基本画法就是「导线画到引脚处就停」，所以**导线与框边的接触点**就是引脚位置。
类型先验只在「一根线都没触到」时兜底（只有一半引脚会接线）。

**接触点必须朝框外推 10px 才是引脚**：接触点在框边上（误差 0px），而 GT 的引脚在框外
10px —— 不推的话每个引脚都差 10px，而 `pin_tol` 默认只有 5px，会全判错。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# GT 实测的默认外推距离；<=12px 覆盖 86.1%
PIN_OUT = 10.0
# ★ 大 IC（`box`）的引脚桩长度。
# 官方 10GT 上同号引脚点误差普查（`work\tools\diag\_pinpos.py`）显示 GT 的 `box`
# 引脚点离框边**左边 14px、右边 20px**，看起来「推到 15px 更对」—— **但那是假的**：
# 逐张量下来只有 case 1 的框检测偏了，其它图的 GT 引脚点就在 ~10px 处。
# 真正的判据是端到端 A/B（`run.py -input work\_gt10_in` 后 `stage2_eval.py`，
# 见 `work\README.md` 的「引脚桩长度」一节）：
#     PIN_OUT_BOX = 10 -> 42.5191（pin F1 0.1394、tp 26、position_mismatch 12）
#     PIN_OUT_BOX = 15 -> 41.4275（pin F1 0.1021、tp 19、position_mismatch 16）
#     PIN_OUT_BOX = 20 -> 40.0155（pin F1 0.0276、tp  5、position_mismatch 31）
# ⇒ 上面那次 A/B 用的是**很旧的代码**。2025-11 在当前代码上重扫了一遍
#   （`work\tools\eval\fast.py --no-vlm`，`work\_fast_base\od*.json`）：
#     10GT 9 张：8 -> pin 0.0998（42.35? 41.35）｜10 -> 0.1492（42.67）
#                ｜**11 -> 0.1586（42.91）**｜**12 -> 0.1602（42.88）**
#                ｜13 -> 0.1413（42.29）｜14 -> 0.1171（41.69）
#     19 张快测：10 -> pin 0.0737（25.99）｜**12 -> pin 0.0789（26.09）**
#   ⇒ **12.0 在两个集合上都赢**（相对 10：10GT pin +7.4%、19 张 pin +7.1%），
#     所以默认改成 12.0。GT 的引脚外推距离实测 p50 = 10 / p75 = 13（`_pincov.py`），
#     12 正好落在两者之间，符合预期。
# 教训：`STAGE1_PIN_OUT_BOX` 曾经被写成默认 15.0，让整条管线白亏 1.09（10GT）。
PIN_OUT_BOX = float(os.environ.get("STAGE1_PIN_OUT_BOX", "12.0"))
# ★ 逐侧腿长（L,R,T,B），2026-10-21 在**提交尺**（`protocol4`，pin_tol=5）下重扫所得。
# 旧尺 `protocol3`（pin_tol=20）下位置几乎不罚分，所以上面那条常数当年只按「整体 F1」调到 12；
# 换成 5px 后位置成为 pin 轴主项，逐侧扫描（`work/_pinleg5.py`，GT10 十张，纯离线改 pins.point
# 后 `--score-only`）：
#   12,12,12,12（旧默认）-> pin 0.7763（总分 83.47）
#   10,10,10,10          -> pin 0.7517（82.86）
#   14,14,14,14          -> pin 0.5738（78.41）
#   14,10,8,10           -> pin 0.7008（81.59）
#   **12,10,8,10**        -> pin **0.8104**（**84.33**）
#   15,12,10,12          -> pin 0.6667（80.74）
# 与独立量的 GT 逐侧外伸（L p50 15 / R 10 / T 8 / B 10）方向一致：左边长、上边最短。
# 设 `STAGE1_PIN_LEG_BOX=""` 即退回单一 `PIN_OUT_BOX`。
PIN_LEG_BOX = tuple(float(v) for v in (os.environ.get("STAGE1_PIN_LEG_BOX", "12,10,8,10") or "").split(",") if v.strip()) or None
# 收尾重摆：stub / merge 补出来的点原本按标量腿长摆，这里统一按「最近边 + 该侧腿长」
# 再摆一次（切向坐标不动）。§5.340.3 里端到端 83.99 与离线 84.33 的差就是这一段。
PIN_LEG_SNAP = int(os.environ.get("STAGE1_PIN_LEG_SNAP", "0"))
PIN_LEG_ALL = int(os.environ.get("STAGE1_PIN_LEG_ALL", "0"))
# 逐**元件**逐侧腿长：从墨迹量出这一颗自己的引线长度，替掉常数 `PIN_LEG_BOX`。
# 动机（§5.357 之后的引脚漏斗）：官方 10 的 21 个位置错里，case1 `U1` 右列 **7 个**
# 全是 `d=(+10.0, ±1)` —— GT 那一侧的腿长是 20px，而我们按常数只推 10px。
# 腿长是逐元件的（同一张图上符号库不同、缩放不同），任何固定 4 元组都必然一半错。
# 量法：`pintrace.find_pins` 已经把「框边外 4px → 该行带内最靠外墨像素」的桩长记在
# `Pin.ln`，于是该侧腿长 = `4 + ln`；按侧取中位数并夹到 `[4, PIN_LEG_INK_MAX]`。
# 用去字墨迹（`text_free_ink`）量，否则紧贴框边的引脚号数字会把腿长虚增到 ~9px。
PIN_LEG_INK = int(os.environ.get("STAGE1_PIN_LEG_INK", "0"))
PIN_LEG_INK_MAX = float(os.environ.get("STAGE1_PIN_LEG_INK_MAX", "26"))
PIN_LEG_INK_MIN = float(os.environ.get("STAGE1_PIN_LEG_INK_MIN", "6"))
LEG_INK_N = {"n": 0}                    # 诊断计数：实际量到墨迹腿长的元件数
# 两脚元件（C/R/L/LED/D）的引脚点直接落到**导线端点**上。
# 动机（§5.359 的 `position_mismatch` 分解）：官方 10 的 38 个位置错里 **28 个是法向**，
# 中位 |d|=9.6px，且集中在两脚件（`1/U1` 右列 7 脚 −10、`3/C1` −9.8、`6/LED1` +10.9、
# `7/C8` −7.0…）——GT 引线长度在 3~22px 之间变化，任何常数腿长都错。
# 两脚件的引线端点就是导线端点（GT 的 `pin.point` 落在导线起端），所以直接在
# 「已检出的线段端点」里找：朝外、`|切向 ≤ PIN_WIRE_TAN|`、距离 ≤ `PIN_WIRE_R`。
# 度 1 端点优先（两条线段的交点不是引线端），找不到就退回原点位。
PIN_WIRE_END = int(os.environ.get("STAGE1_PIN_WIRE_END", "1"))
PIN_WIRE_R = float(os.environ.get("STAGE1_PIN_WIRE_R", "45"))
PIN_WIRE_TAN = float(os.environ.get("STAGE1_PIN_WIRE_TAN", "30"))
# IC（box）也允许「引脚点落到朝外的度 1 导线端点」（`STAGE1_PIN_WIRE_END_IC`）。
# 依据：20261026 新增的 open_GT（13 张干净页）里 191 个 box 脚有 159 个正好落在导线
# 端点上（<=0.5px），而我们的脚停在内侧约 9px ⇒ 那套图的 pin F1 只有 0.017。
PIN_WIRE_END_IC = int(os.environ.get("STAGE1_PIN_WIRE_END_IC", "1"))
# 只吸「真的往外伸出去」的端点：脚点到端点 >= `PIN_WIRE_MIN` 才动。
# 10GT 的脚本来就停在导线端点上（实测 2~3px）⇒ 这条闸让 10GT 逐位不变；
# open_GT 里我们的脚停在内侧 9~20px、而导线已经画到 GT 位置 ⇒ 正好被吸过去。
PIN_WIRE_MIN = float(os.environ.get("STAGE1_PIN_WIRE_MIN", "5"))
# 候选端点必须来自「够长」的线段（真导线）才允许吸；符号自身的小笔画都短于这个值。
PIN_WIRE_SEGLEN = float(os.environ.get("STAGE1_PIN_WIRE_SEGLEN", "25"))
# ★ 本级新加：不许吸「符号自身引线的外端」（`STAGE1_PIN_WIRE_SELF`，默认 0 = 老行为）。
#   判据只看几何：候选端点的**另一端**落在这颗元件的框内（外扩 `_PAD` px）⇒ 这条线段是从
#   本体长出来的引线，它的外端点不是「别处来的导线端点」，吸过去等于把脚推到引线长度上。
#   证据（RESULTS §5.485）：case11 `U1` 右列的 `[[1079,399],[1128,400]]` 就是本件的引线，
#   吸它的外端 1128 使脚比 GT（1095）偏 33px。
PIN_WIRE_SELF = int(os.environ.get("STAGE1_PIN_WIRE_SELF", "0"))
PIN_WIRE_SELF_PAD = float(os.environ.get("STAGE1_PIN_WIRE_SELF_PAD", "4"))
# 只收「度 1」端点（两条线段的交点不是引线端）。new2026 的干净页里 GT 脚点也常常
# 落在「导线与导线相接」的端点上 ⇒ 放宽到全部端点（`STAGE1_PIN_WIRE_FREE=0`）更好。
PIN_WIRE_FREE = int(os.environ.get("STAGE1_PIN_WIRE_FREE", "0"))
# IC 脚的「腿长补齐」（`STAGE1_PIN_LEG_FILL`）：new2026 的 GT（`open_GT`）把脚点画在框边外
# 固定 19~20px 上（p10 19 / p50 20），而没挂到导线端点的脚只有常数腿长（10~12px）⇒ 差 8~10px
# 直接超出 `pin_tol=5`。离线扫描（`work/_legfix.py`，open_GT-1020 13 张）：
#   不补齐 66.26 / pin 0.4474｜target=19..20 + thresh=14..17 补齐 55~56 根 ⇒ **68.28 / 0.5281**
#   thresh=20（补 94 根）⇒ 65.84（把已经对的脚也推走）⇒ 阈值必须卡在 15 附近。
# 注意：这条在 **stage1** 里默认关（`STAGE1_PIN_LEG_FILL=0`）。同一份改动放在 stage1 时
# 会改变 stage2 的引脚候选池、把脚号绑定带歪 ⇒ 实测 66.08（比不开的 66.26 还低）；
# 真正有效的位置是 stage2 装配之后（`STAGE2_PIN_LEG_FILL`，见 `stage2/assemble.py`）。
PIN_LEG_FILL = int(os.environ.get("STAGE1_PIN_LEG_FILL", "0"))
PIN_LEG_FILL_TARGET = float(os.environ.get("STAGE1_PIN_LEG_FILL_TARGET", "19"))
PIN_LEG_FILL_THRESH = float(os.environ.get("STAGE1_PIN_LEG_FILL_THRESH", "15"))
PIN_WIRE_N = {"n": 0, "miss": 0}         # 诊断计数：改到的引脚数 / 没找到的引脚数
# IC 逐元件逐侧腿长，用**符号色**量（`STAGE1_PIN_LEG_COLOR`）。
# 墨迹法失败的原因已定案（§5.358）：引线与导线同为墨、在带内连成一片，`Pin.ln` 量到的是
# 带长。但这些图纸里**引线是符号色（暗红）、导线是绿色**，颜色把它们分开了：
# 从框边向外走，红像素连续段有多长，引线就有多长；红段结束处常常就是导线起点。
# 只在有色页面上生效（黑白页面没有符号色 ⇒ mask 为空 ⇒ 退回常数）。
# 转正（§5.408）：open_GT-1020 离线 68.72 → 71.65、VLM 72.20 → 74.08。上限必须 ≥
# 最长引线（新 GT 的 `TPS61240DRVR` 7 根脚全是 40px），MAX=30 时 case9 反而 −10.7。
# 已知代价：case5 −0.9（框线粗细压短中位数）、case1/13 小幅负；三条加固尝试（段长下限
# MIN / 引线行数下限 NMIN / 导线端点作证）全部实测更差，故保持原始中位数算法。
PIN_LEG_COLOR = int(os.environ.get("STAGE1_PIN_LEG_COLOR", "1"))
PIN_LEG_COLOR_MAX = float(os.environ.get("STAGE1_PIN_LEG_COLOR_MAX", "46"))
PIN_LEG_COLOR_GAP = int(os.environ.get("STAGE1_PIN_LEG_COLOR_GAP", "2"))
# 逐侧腿长的**唯一性**约束（`STAGE1_PIN_LEG_UNI`）：一颗符号的四侧腿长是同一个数（符号
# 库按统一单位画引线），所以「量到饱和（= 探到窗边仍连续墨，说明撞上了别的笔画）或量不到」
# 的那一侧不该用自己那个假数，而应弃权、取本件各侧有效量值的中位数。
PIN_LEG_UNI = int(os.environ.get("STAGE1_PIN_LEG_UNI", "0"))
PIN_LEG_COLOR_N = {"n": 0, "empty": 0}
# 「触到框边」的判定半径
TOUCH_TOL = 6.0
# ★ 本级新加：`box`/`bigic` 单独的触摸半径（`STAGE1_PIN_TOUCH_IC`，默认 0 = 用 `TOUCH_TOL`）。
#   动机（RESULTS §5.485.9）：DC/DC 这类 IC 的脚画在引线外端，引线本身 20~40px，
#   导线是从引线末端接出去的 ⇒ 导线起点离框边 20~40px，`TOUCH_TOL=6` 根本看不见它，
#   于是这个脚只能靠「桩 / 颜色腿长 / 导线端点吸附」猜位置。放宽后它变成一次真触摸。
TOUCH_TOL_IC = float(os.environ.get("STAGE1_PIN_TOUCH_IC", "0") or 0)
# 同一引脚上多个接触点的合并半径（沿框边弧长）
CLUSTER_R = 12.0
# ★ IC（`box`）单独用更小的合并半径。
# 220 张实测 GT 的 `box` 引脚**沿框边的间距 p25=10 p50=10 p75=15 px**
# （`work\tools\diag\_pincov.py`），而 `CLUSTER_R=12` 比这个间距还大 ⇒ 相邻引脚
# 被并成一个簇。实例：`0113` 的 `U4`（MAX7219，24 脚）我们只输出 **7** 个点
# （`work\_submit_out220r21\runs\stage2\0113.json`），而点的位置其实准到 1px
# （我们 x=185.1/305.5 vs GT x=186.0/306.0）——**亏的是覆盖率，不是精度**。
# ⚠️ 历史：这个旋钮**被判负过一次**（10GT 9 张，`--no-vlm`，`work\_fast_base\`）：
#     12.0 42.67 ｜ 6.0 41.68 ｜ 5.0 41.47 ｜ 4.0 41.47
# 19 张快测：12.0 25.99 ｜ 4.0 25.57（pin 0.0737 → 0.0649）。
# 当时的原因：`pin_f1` 的 tp 要「号 + 名 + 点」全对，多出来的引脚绝大多数只是
# `pin.extra`（fp），把 fn 换成 fp+fn 是亏的；拆分只在「名和号本来就读对」的引脚
# 上才有收益，而那部分覆盖率极低。
# ★★ 该结论**在 VLM 引脚通道打开后被推翻**（`-vlm-pin` 默认 on，实测 2026-10-09，
# 交付口径 × 官方 10 张，`work\_pinab*.ps1` / `_cmpab.py`，`work\_fast_base\*.json`）：
#     12.0（旧默认）68.55 ｜ 8.0 69.09(+0.54，**1涨/9平/0跌**) ｜ 4.0/5.0/6.0 69.89(+1.34)
# 其中 4.0 / 5.0 逐位相同（平台），逐案 Δ：2 +3.27 ｜ 3 +2.83 ｜ 5 +0.15 ｜ 6 +6.58
# ｜ 9 +5.44 ｜ **4 −0.44 ｜ 7 −4.46**。机制：IC 密集引脚列（case 6 的 4 颗 SOT-23、
# case 9）在 12px 下被并掉；而 case 7 反过来 —— 半径过小会把同一根真脚的两个触点
# （相距 8px）拆成两根，两根抢同一个文字槽 ⇒ 名字读成垃圾（AGND→'O'/'D'），
# tp 3→2、fn 5→6。VLM 现在能供号供名，拆分才第一次变成净赚。
# ★ 默认值取 **5.0**（平台中点：4.0/5.0 同分，取靠近旧值的一侧；`6.0` 也同分但更接近
# 退化区）。若想零回归，可用 `STAGE1_PIN_CLUSTER_IC=8.0`（只赚 case 9 的 +5.44）。
# ★ 派生集泛化已抽查（12 张，`--no-vlm` + `FUSION_LLM=0`）：`5` vs `12` = **Δ总分 0.00**
# （涨 3 / 平 4 / 跌 5）⇒ 官方集 +1.34、派生集不动。
# ⚠️ 亏的 case 7 是「小半径 ⇒ 簇变多 ⇒ `STUB_SPARSE_RATIO=0.6` 不再触发 ⇒ 丢掉几何桩补出的脚」，
# 试过 `STAGE1_PIN_STUB=union` 把桩全并回来：case 7 救回（69.71）但整体 **−0.90**
# （0 涨 / 7 平 / 3 跌，case 2 76.24→68.52）⇒ 不加桩才是对的。详见 `docs\RESULTS.md` §2.16。
# ★★★★ 2026-10-10 **又翻了回去：默认 5.0 → 12.0**。把桩证据换成「抹掉文字框」的墨迹
#     之后（`STAGE1_PIN_STUB_TEXT=1` + `gapfill`，见下面 ★★★★ 注释），12.0 反超 5.0：
#       交付口径 × 官方 10 张：`free+gapfill+CI=12` **73.27** ｜ `free+gapfill+CI=5` 72.77
#       派生 12 张（`--no-vlm`）：25.82 ｜ 25.80（两个尺子一致指向 12.0）
#     机制：§2.16 那 +1.34 里的一部分本来就是「小半径把**文字簇**拆开 ⇒ 触点变多」的副作用，
#     文字证据被抹掉后这段收益消失，而 gapfill 把欠检的桩脚补回来 ⇒ 半径回到 12.0 不再丢行。
#     ⇒ 旧读数（12.0 68.55 / 4.0=5.0=6.0 69.89）**只在 raw 墨迹下成立**，别再引用。
CLUSTER_IC = float(os.environ.get("STAGE1_PIN_CLUSTER_IC", "12.0"))
# 采样步长（像素）
SAMPLE_STEP = 3.0
# 每个元件最多输出多少个引脚（防毛刺炸开）
MAX_PINS = 64

# ★ 几何引脚桩补检（`STAGE1_PIN_STUB`，默认关）。
# 主判据「导线触到框边」**只能看见接了线的引脚**，而 GT 列出符号的**全部**引脚。
# 220 张实测（`work\tools\diag\_pincov.py`，只算官方 matcher 配上的 358 颗 IC）：
#     GT 引脚 4187 ｜ 我们的引脚点 1901（0.45）
#     `point@5px` 只有 **16.8%**（scorer 的 `cfg.pin_tol` 就是 5.0）
# 而最惨的几张（`0087/U24` 143 脚、`0080/U18` 51 脚）落进了 `pins_from_prior`
# 兜底，只剩 12 个罐头点 —— 因为导线停在引脚桩的**外端**，根本没碰到框。
# `pintrace.find_pins` 找的是「框外朝外的几何小桩」，跟接不接线无关，正好补这一块。
# 代价：桩比真引脚多（40 张抽样 999 桩 vs 750 GT 脚），所以默认关，靠 A/B 决定。
# `union`（把桩并进接触点）实测有**净亏**：19 张 25.99 → 24.68、10GT 42.67 → 39.77
# —— 桩不是真的接线接触点，多出来的点会把网络图搞乱（fp 涨得比 tp 快）。
# ⇒ 只保留 `fill`：**仅在接触点为空**（正在走 `pins_from_prior` 兜底）时才改用桩，
#    这样正好命中 `0087/U24`(143 脚) / `0080/U18`(51 脚) 那类「导线停在桩外端、
#    一个都碰不到框」的符号，不动接触点已经好用的那些 IC。
# `STAGE1_PIN_STUB` = `sparse`（**当前默认**，见下面 ★★）| `0`/`off`（关闭补桩）
#                  | `1`/`fill`（仅兜底时替换）| `union`（全并）。
#
# ★ 2025-10-05 重测（**带 VLM + 当前四旋钮默认**，`fast.py --save …`）：
#     `union` 不再像老读数那样净亏 —— 因为 VLM 的引脚网格通道能从桩上读出真号真名。
#       10GT(9)：默认 58.05 → `union`+`mixed` **58.53**（+0.48，netH 0.4971→0.5051）
#       19 张  ：默认 34.56 → `union`+`mixed` **34.63**（+0.07）
#       隔离臂（10GT）：`DROP_UNREAD=all` 单独 58.05（与 `mixed` 逐位相同）；
#                      `union`+`all` 58.23 < `union`+`mixed` 58.53。
#     `--no-vlm` 下 `union` 仍是亏的（10GT 48.15→47.53、19 张 28.60→28.47）——
#     没有 VLM 时桩只会被 `rebind` 套上错号。**所以 `union` 的价值取决于 VLM 通道。**
#
# ★★ 2025-10-05 「稀疏 IC 才补桩」成为默认（`sparse`，`STAGE1_PIN_STUB_RATIO=0.6`）。
#     触发条件：`len(接触点) < 0.6 * len(几何桩)` —— 只动**明显欠检**的 IC。
#     A/B（全在四旋钮默认之上）：
#       | 集合      | 基线             | `sparse 0.6`      |
#       | 10GT(9) 带VLM | 58.05 (pin .4808) | **59.00 (pin .5040)** |
#       | 19 张 带VLM   | 34.56 (pin .2294) | **35.03 (pin .2357)** |
#       | 10GT(9) 无VLM | 48.15 (pin .2720) | **48.35 (pin .2651)** |
#       | 19 张 无VLM   | 28.60 (pin .1320) | **28.85 (pin .1329)** |
#     比例扫描（10GT 带 VLM）：0.3 → 58.05（等于没触发）｜0.4 → 57.79｜0.5 → 58.23｜
#     **0.6 → 59.00**｜0.7 → 58.18｜0.8 → 57.98｜1.0 → 58.74。0.6 之外都不稳。
#     ⇒ 2026-10-10 交付判决后**默认改成 `gapfill`**（+ 抹字，见 ★★★★）：官方 10 张 69.89→**73.27**。
_STUB_MODE = os.environ.get("STAGE1_PIN_STUB", "gapfill").strip().lower()
STUB_FILL = _STUB_MODE not in ("", "0", "false", "off", "none")
STUB_UNION = _STUB_MODE in ("union", "all", "add")
STUB_SPARSE = _STUB_MODE in ("sparse", "gap", "short")
STUB_SPARSE_RATIO = float(os.environ.get("STAGE1_PIN_STUB_RATIO", "0.6"))

# ★★★ 2026-10-10 把 `CLUSTER_IC` 12→5 试了一轮（后来**又翻回 12**，见上面 CLUSTER_IC
#     的长注释；这条注释保留是因为它暴露的缺口与半径无关）之后暴露出的真缺口：
#     `sparse` 的触发条件比的是**簇计数**（`len(接触点) < 0.6*len(几何桩)`），
#     而簇半径变小 ⇒ 接触点变多 ⇒ 这条判据**反而不再触发**。
#     case 7 的 U1（`MP2236CGJ-Z`，GT 8 脚）就是这样丢掉整行（`_pinfill.py` 复现）：
#       `CI=12`：接触 10 + 补桩 12 = **22** 点（GT 8 脚里对 3）
#       `CI=5` ：接触 15 + 补桩  0 = **15** 点 ⇒ y≈383 那一行（GT `pin_3 SW`/`pin_6 EN`）没了
#     正确判据应是「**有没有没被接触点覆盖的桩**」，而不是比总数 ⇒ 新模式 `gapfill`：
#     只并入「**长桩**（`pintrace.Pin.ln >= STAGE1_PIN_STUB_MINLEN`）**且离任何已有点都
#     超过 `STAGE1_PIN_STUB_GAPTOL` px**」的桩。`ln` = 框边±4px 起点 → 该行带内最外侧
#     墨像素的距离：官方图上真桩 ≥19px，而**引脚号数字只贴框边 ~9px**（case 8 逐像素
#     取证，见 `stage1/pincol.py` 的 `STAGE1_PINC_STUB_REACH` 注释）⇒ 这个阈值剔号字。
#     `STAGE1_PIN_STUB_MINLEN` **默认 8**（只在 `gapfill` 模式下生效）：抹字之后长度
#     阈值变得不敏感（官方 CI=5 的 gap4/gap6/gap8/gap10 全是 127/137/9），8 是
#     「真桩 ≥19px、号字 ~9px」两个分布之间的平台中点。
STUB_GAP = _STUB_MODE in ("gapfill", "gap_only", "covered")
STUB_MINLEN = float(os.environ.get("STAGE1_PIN_STUB_MINLEN", "8"))
STUB_GAPTOL = float(os.environ.get("STAGE1_PIN_STUB_GAPTOL", "4.0"))

# ★★★★ 2026-10-10 桩幻影的真正主因是**文字**，不是长度：`pintrace.find_pins` 只看
#     「框外 `band` 内的墨」，而**引脚号数字 / 型号文字**也画在框边外 ⇒ 整行文字被当成
#     一堆桩。逐桩取证（`work\tools\dev\_stubtext.py`，`pincol.text_free_ink` 抹掉
#     `texts` 里的文字框再喂给 `find_pins`）：
#       case 7 的 U1：raw 22 桩（tp 7 / fp 15，顶边那 11 个是型号文字行）
#                    →**抹字后 8 桩，tp 8 / fp 0 / fn 0**（= GT 的 8 脚，一根不差）
#       case 8 的 U1：raw 31 桩（tp 13 / fp 18）→ 抹字后 8 桩（tp 7 / fp 1 / fn 9）
#     官方 10 张离线（`tools\diag\_stubmode.py --free`，按侧对位 tol 5px）：
#       raw  sparse CI=5  tp 122 fp 178 fn 14  F1 0.5596   ← 当前默认
#       free gapfill CI=5 tp 127 fp 137 fn  9  F1 0.6350
#       free gapfill CI=12 tp 127 fp  77 fn  9  F1 **0.7471**   ← 离线最优
#     ★★★★ 2026-10-10 **交付端到端判决（官方 10 张，`work\_pinab9.out` / `_pinab10.out`）**：
#       base（§2.16 出厂默认 = raw sparse CI=5）                      **69.89**
#       `free + gapfill + MINLEN=8 + CI=5`（`f5`）                    72.77（+2.88，6涨/3平/1跌）
#       `free + sparse  + CI=12`（`f12s`，只抹字 + 大半径）            72.45（+2.56，5涨/3平/2跌）
#       **`free + gapfill + MINLEN=8 + CI=12`（`f12`，采用）        73.27（+3.39，6涨/2平/2跌）**
#       再叠 `STAGE1_PINC_FILL=0`（`f12nf`）                         73.37（只多 case 8 的 +0.98，
#         但派生集是 −0.47 ⇒ 那份 FILL 不动）
#     通道 micro（全 10 案）：pin **0.5250 (57/40/79) → 0.6424 (69/33/67)**、
#     netH 0.7175 (314/74/173) → **0.7289 (322/73/165)**（幻影网络同步减少）；
#     逐案 pin 变好在 2/3/4/6/7/10（case 3 → **1.0000**、case 7 +8.30），只在 case 8 退（−1.67，
#     抹字把「号字正好压在引线墨上」的真桩一起抹掉）；case 1 离线 fp 16→8 但端到端**一动不动**
#     ⇒ 那一案的引脚失分在下游装配/名字绑定，不在候选集。
#     派生 12 张（`--no-vlm`，`work\_findv3.out`）：25.92 → `df12` 25.82 / `df5` 25.80（−0.1/案，
#     基本中性；且两个尺子都判 `CI=12 ≥ CI=5`）⇒ 这就是判决依据。
#     注意：`pincol.enumerate_pins` 那条路也要同样处理（`STAGE1_PINC_*`），但两条路的
#     证据源不同（这里用 `pintrace`，那里用逐列墨计数）。
STUB_FREE = os.environ.get("STAGE1_PIN_STUB_TEXT", "1").strip().lower() not in (
    "", "0", "false", "off", "no", "none")
STUB_FREE_PAD = float(os.environ.get("STAGE1_PIN_STUB_TEXT_PAD", "0.0"))

# ★ 更激进的一档：**整条 `find_pins`**（触点路径 + 桩路径）都用抹掉文字的墨迹。
#   与 `STUB_FREE` 的区别是触点也去字 —— 触点路径同样会被引脚号/型号文字骗到
#   （case 7 的 U1 在 CI=5 下触点 15 个、GT 只有 8 脚）。风险是「引脚号正好压在
#   引线墨上」的真桩会被一起抹掉（case 8 的实测），所以做成独立旋钮、默认关。
#   ★ 2026-10-10 实测（`work\_pinab11.ps1`，交付 × 官方 10 张）：在 `STAGE1_PIN_STUB_TEXT=1`
#     + `gapfill` 已经成立的前提下，`INK_TEXT=1` + CI=12 与 `f12` **逐位相同**（73.27，
#     连 `pin` micro 0.6424 都一样）⇒ 触点路径上**没有**可抹的幻影；CI=5 时反而
#     72.39 < 72.77（`pin` 0.6141 < 0.6295）⇒ 与 `STAGE1_PINC_STUB_REACH` 同向，默认仍 0。
INK_FREE = os.environ.get("STAGE1_PIN_INK_TEXT", "0").strip().lower() not in (
    "", "0", "false", "off", "no", "none")
INK_FREE_PAD = float(os.environ.get("STAGE1_PIN_INK_TEXT_PAD", "0.0"))

# 两脚元件：引脚在长轴两端
TWO_PIN = frozenset({"r", "c", "l", "d", "led", "fuse", "crystal_2pin", "battery",
                     "battary", "net_short", "switch", "buzzer", "amp"})
# 一脚符号
# ★ `"tp"` 是该分支时 `stage1/detect.py` 把 `pin`/`testpoint` 合并出的
#   **内部类**（见 detect.py:100-124）。少了它，开关打开后 `pins_from_prior` 在
#   `:263` 匹配不上、一路掉到 `:274 return []` —— 而它的两个父类都在表里，
#   于是「合并」会**顺手删掉测试点/焊盘的兜底脚**（脚不进 pin F1、但进 net 超图 0.35）。
#   两个父类都按一脚处理，合并类必须同款，否则开关打开前后不是同一套语义。
ONE_PIN = frozenset({"gnd", "v", "vcc", "vbus_in", "net_input", "net_output", "pin",
                     "testpoint", "tp", "net_bidirection", "block", "other"})

# ★★ 「两脚元件的脚必在长轴两端」（`STAGE1_PIN_AXIS`，默认 1 = 开）。
#   动机：`pins_from_touch` 是「哪条导线段贴着框的哪条边，就在那一边生一个脚」——
#   两脚元件周围除了真正的主力引线（长轴两端），还常有**符号自身笔画/极板/短线头**
#   残留在框外（框是检测器给的，常常比符号紧），于是同一个电容会拿到 4 个候选脚
#   （上下真脚 + 左右两个笔画残段）。`build_nets` 会把每个候选脚挂到最近的段上 ⇒
#   **一个两脚元件凭空多出两条单脚网络**，而官方口径 `net_pin_collapse_nonbox=True`
#   把非 IC 的脚号折成 `*`：GT 里一个电容在一条网里只算一个成员，我们却多出两条
#   只含它自己的网 ⇒ `netH` 的 fp 涨、召回不动。
#   证据（官方 10 张，`tools/diag/_pinaxis.py`）：GT 非 IC 元件的脚**100% 沿框主轴**
#   （V/V 100 + H/H 25 = 125/125）；而 stage1 的候选脚里 **232/495（46.9%）在短轴上**。
#   修法：类型不在 `ONE_PIN`、不是 IC、且框主轴比 >= `PIN_AXIS_RATIO` 时，只留主轴
#   两端最近的各一个候选脚，其余进 `pins_latent`（只移出活动集，**不删除**，账本留痕）。
# ★★ 2026-10-10 交付 A/B（`work/_pinab13.ps1`，官方 10 张 + 派生 10 张，`-vlm-pin` 开）：
#   `ax1` = 官方 **8 涨 / 2 平 / 0 跌**，COMMON10 73.27 → **76.17（+2.90/案）**；
#   通道 `net_hypergraph` 0.7289 → **0.7902**、`pin` 0.6424 → 0.6564；派生 10 张 +2.13。
#   机制：幻影单脚网络整批消掉（官方 10 张候选脚 723 → 683；case 1 的最终网 17 → 9、
#   单成员网 12 → 0）。旧读数「`--no-vlm` 下判负」是在「补桩还没抹文字」的口径下量的，
#   已作废。⇒ **默认翻成 1**。
#   ⚠ 2026-10-10 补充（`work/_pinab14.ps1`）：贴章发号修复（`STAGE2_STAMP_ONE_SIDE`，见
#   `stage2/assemble.py`）**修的是同一片幻影网** —— `sfix1`（贴章修复、AXIS 关）与 `ax1`
#   逐案完全相同（都 76.17），且贴章修复已在位时 AXIS 0→1 在 case 1/4/8/9/10 上 VLM-free
#   **逐位相同**；泄漏无关派生 val（19 张 VLM-free）**+0.02/案**（基本中性）。两者都保留
#   （各自单独就到 76.17，合起来 `ax1B` 复现 76.17，不叠加、也不冲突）。
PIN_AXIS = os.environ.get("STAGE1_PIN_AXIS",
                          "1").strip().lower() not in ("", "0", "false", "off", "no", "none")
PIN_AXIS_RATIO = float(os.environ.get("STAGE1_PIN_AXIS_RATIO", "1.2"))

# ★★ 两脚元件的脚是「框边 ∓ PIN_OUT、长轴中线上的一个点」（`STAGE1_PIN_MAJOR`，默认 1）。
#   取证（官方 10GT）：两脚元件 `{r,c,l,d,led,fuse,crystal_2pin}` 125 个、引脚 250 个 ——
#   **250/250 落在 bbox 长轴**，沿轴外伸 p10=p25=p50=p75=p90=max=**10.0**，短轴偏移
#   p50/p90/p99/max=**0.00**（「外伸=10 且短轴偏移=0」238/250=95.2%，其余是四舍五入残留）。
#   所以与其从触点/桩里猜，不如**直接盖章**这两个点。离线按网判（`_leadrule.py --bynet`，
#   官方 10 张）：current 109 OK / 132 落错网 / 32 未挂 → ideal(od=10) 116 / 79 / 55
#   （+7 tp / −53 fp）。
#   交付 A/B（官方 10 张，`maj1` = PIN_AXIS=1 + 贴章修复已在位）：`ax1B` 76.17 → **77.20**
#   （+1.03/案，7 涨 / 2 平 / 1 跌：1 +0.65、4 +0.20、5 +1.56、6 +1.63、7 +2.56、9 +4.26、
#   10 −0.60；3、8 平）；`net_hypergraph` 0.4816 → **0.4961**、20 张合计 47.48 → 48.00；
#   派生 10 张 +0.20。⚠ case 7 那 +2.56 里有 VLM 引脚名抽样的成分（同一臂重跑可达 ±3.85），
#   别单看这一案 —— 机制证据是「250/250 在长轴、外伸恰 10.0」这条几何律。
PIN_MAJOR = os.environ.get("STAGE1_PIN_MAJOR",
                           "1").strip().lower() not in ("", "0", "false", "off", "no", "none")
PIN_MAJOR_TYPES = frozenset({"r", "c", "l", "fuse", "crystal_2pin", "crystal", "d", "led"})

# ★★ 跨过「符号自身引线」去挂网（`STAGE1_PIN_LEAD`，默认 0 = 关；单位 px）。
#
# **问题**：`PIN_MAJOR` 把两脚元件的脚盖章在「框边 ∓ PIN_OUT」= GT 的引脚点（250/250），
# 但**符号画法**会在框边与导线之间留一段 10~20px 的引线笔画（KiCad 的电阻/电容体比
# 引脚点短），导线画到引线**外端**就停 ⇒ 引脚点离最近导线 13~20px，超过
# `nets.PIN_JOIN_DIST=12` ⇒ 这一脚永远挂不上网（`pin_join.orphans`，然后整条网丢失）。
#
# **取证**（官方 case 2，R2 = `c0011`）：`orphans` 里 `{"comp":"c0011","pin":0,"d":18.8}`；
# 逐行判色（图 2063×1116，列 x=573..580）：绿导线止于 y=423、暗红引线 424..443、
# 电阻体 444..480、红引线 481..500、绿导线 501 起 —— 即「导线端 ↔ 引脚点(441.8)」
# 之间 18.8px **全是符号引线**。GT `net_5=(R2.2, U1.3)` 因此整条丢失，case 2 一共
# 8 条 `R.x.2 ↔ U.x` 这样丢掉（conn 权重 0.35，`_conncal`-类读法约 +0.9/案）。
#
# **做法**：把「朝外方向上 `PIN_LEAD` 以内、横向偏移 ≤ `PIN_LEAD_PERP`」的引脚点吸附到
# 最近的导线采样点。只动 **stage1 的附着点** —— 两脚类型在 stage2 走 `stamp_pins`
# （只用 `pts` 判左右侧，沿轴外推不改投影符号），所以**输出引脚逐位不变**、只有网变。
PIN_LEAD = float(os.environ.get("STAGE1_PIN_LEAD", "26"))
PIN_LEAD_PERP = float(os.environ.get("STAGE1_PIN_LEAD_PERP", "3.0"))
# 吸附目标：`end`（默认）= 只用**线段端点**（物理上「导线画到引线外端就停」）；
# `any` = 用整条点云（含段中间），会把「从旁边横穿过去的导线」也当成引线终点。
PIN_LEAD_SRC = os.environ.get("STAGE1_PIN_LEAD_SRC", "end").strip().lower()
# 只吸附到**自由端**（度 1 的导线端点，2px 内没有别的线段端点）。导线在引脚附近
# 与别的段交汇时那个端点不是「引线外端」，把引脚挂上去会并错网（case 3/6 的回归）。
PIN_LEAD_FREE = os.environ.get("STAGE1_PIN_LEAD_FREE", "1").strip().lower() not in (
    "", "0", "false", "off", "no", "none")
# 引线证据：引脚点 → 目标端点 的连线上必须有墨（那段「符号自身引线」的笔画）。
# 没有墨就说明这不是引线，而是别的网的导线端点恰好停在旁边（case 3/6 的回归）。
PIN_LEAD_INK = os.environ.get("STAGE1_PIN_LEAD_INK", "0").strip().lower() not in (
    "", "0", "false", "off", "no", "none")
PIN_LEAD_INK_FRAC = float(os.environ.get("STAGE1_PIN_LEAD_INK_FRAC", "0.6"))
# ★ 判负（2026-10-10，官方 10 张 VLM-free）：本以为是「已挂上网的脚搬走会拆桥 ⇒ 只吸孤儿」，
# 实测 `LEAD_ORPHAN=1` **57.78** vs 关 **58.15**（case 5 +2.05→0、case 10 +1.25→0 全丢，
# 而 case 3/6 的下跌**没有**因此修复）⇒ 真相是：脚常被挂到「路过但不对」的那条线上，
# 吸附正是在把它搬到**引线远侧**那条线上纠正它。默认 **0**（关），只留作实验旋钮。
PIN_LEAD_ORPHAN = os.environ.get("STAGE1_PIN_LEAD_ORPHAN", "0").strip().lower() not in (
    "", "0", "false", "off", "no", "none")
PIN_LEAD_ORPHAN_D = float(os.environ.get("STAGE1_PIN_LEAD_ORPHAN_D", "12.0"))
# 只对 stage2 走 `stamp_pins` 的类型开（`stage2/assemble.py:485 PIN_STAMP_TYPES`）：
# 其余类型的 stage1 引脚点**就是**输出引脚，挪了会直接改 pin_f1。
PIN_LEAD_TYPES = frozenset({"r", "c", "l", "fuse", "crystal_2pin", "d", "led"})


def _seg_ink(ink, a, b, n: int = 8, frac: float = None) -> bool:
    """`a` → `b` 的连线上是否有墨（引线笔画的证据）。`ink` 是图像坐标墨迹掩膜。

    沿线取 `n` 个内点，每点看 3×3 邻域；墨点比例 ≥ `frac` 才算过。
    """
    if ink is None:
        return True
    _f = PIN_LEAD_INK_FRAC if frac is None else float(frac)
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    h, w = ink.shape[:2]
    hit = 0
    tot = 0
    for i in range(1, n + 1):
        t = i / float(n + 1)
        x = int(round(ax + (bx - ax) * t))
        y = int(round(ay + (by - ay) * t))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        tot += 1
        if ink[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
            hit += 1
    if tot == 0:
        return True
    return (hit / float(tot)) >= _f


def _snap_lead(pins, pts, center, lead=None, perp=None, mask=None, ink=None,
               cloud=None):
    """引脚点沿朝外方向吸附到最近的导线采样点（跨过符号自身引线）。

    返回 `(新 pins, 吸附个数)`。`pts` = `sample_segments` 的点云 (M,2)，图像坐标；
    `mask` 非空时只允许吸附到 `mask` 为真的那些点（自由端，见 `PIN_LEAD_FREE`）；
    `ink` 给了就要求「引脚点 → 目标点」的连线上有引线墨迹（见 `PIN_LEAD_INK`）。
    判据：`t = (p-q)·u ∈ (0, lead]` 且横向偏移 `|(p-q) - t·u| ≤ perp`，取 `t` 最小者
    （= 引线外侧最近的导线点）。`t ≤ PIN_JOIN_DIST` 时本来就挂得上，吸附不改所属网。
    `cloud` 给全量点云时只吸**孤儿脚**（到最近导线的距离 > `PIN_LEAD_ORPHAN_D`）：
    已经挂上网的脚再搬走会**拆掉它当时充当的桥**，反而让网数变多（见 `PIN_LEAD_ORPHAN`）。
    """
    _lead = PIN_LEAD if lead is None else float(lead)
    _perp = PIN_LEAD_PERP if perp is None else float(perp)
    out: List[Tuple[float, float]] = []
    n = 0
    cx, cy = float(center[0]), float(center[1])
    for p in pins:
        qx, qy = float(p[0]), float(p[1])
        if PIN_LEAD_ORPHAN and cloud is not None and len(cloud):
            d0 = np.sqrt(((cloud[:, 0] - qx) ** 2 + (cloud[:, 1] - qy) ** 2).min())
            if float(d0) <= PIN_LEAD_ORPHAN_D:
                out.append((qx, qy))
                continue
        ux, uy = qx - cx, qy - cy
        nrm = math.hypot(ux, uy)
        if nrm < 1e-6 or pts is None or len(pts) == 0:
            out.append((qx, qy))
            continue
        ux, uy = ux / nrm, uy / nrm
        dx = pts[:, 0] - qx
        dy = pts[:, 1] - qy
        t = dx * ux + dy * uy
        lat = np.abs(dy * ux - dx * uy)
        ok = (t > 0.0) & (t <= _lead) & (lat <= _perp)
        if mask is not None:
            ok &= mask
        if not bool(ok.any()):
            out.append((qx, qy))
            continue
        idx = int(np.argmin(np.where(ok, t, 1e9)))
        cand = (float(pts[idx, 0]), float(pts[idx, 1]))
        if PIN_LEAD_INK and not _seg_ink(ink, (qx, qy), cand):
            out.append((qx, qy))
            continue
        out.append(cand)
        n += 1
    return out, n


def _axis_split(bbox, pins):
    """按框主轴把候选脚分成「主轴两端」与「短轴两侧」。返回 (keep, drop)。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = abs(x2 - x1), abs(y2 - y1)
    long_axis = "V" if h >= w else "H"
    along, aside = [], []
    for p in pins:
        d_long = abs(p[1] - cy) if long_axis == "V" else abs(p[0] - cx)
        d_short = abs(p[0] - cx) if long_axis == "V" else abs(p[1] - cy)
        (along if d_long >= d_short else aside).append(p)
    if len(along) < 2:
        return list(pins), []
    key = (lambda p: p[1]) if long_axis == "V" else (lambda p: p[0])
    along = sorted(along, key=key)
    keep = [along[0], along[-1]]
    drop = [p for p in pins if p not in keep]
    return keep, drop


# `find_pins` 每次调用都会重置：被 `PIN_AXIS` 移出活动集的候选脚（不删除，只降权）。
LATENT: Dict[str, List[Tuple[float, float]]] = {}


def latent_stats() -> dict:
    """被移出活动集的候选脚小结（给 dbg/账本用）。"""
    return {"n_comps": len(LATENT),
            "n_pins": sum(len(v) for v in LATENT.values()),
            "sample": sorted((k, [list(p) for p in v]) for k, v in LATENT.items())[:6]}


def sample_segments(segments: Sequence[Sequence[Sequence[float]]],
                    step: float = SAMPLE_STEP) -> np.ndarray:
    """把线段按 `step` 像素密集采样成一个点云 (M,2)。

    采样而不是只取端点：导线常常**从元件框中间穿过去**（串联元件），那时接触点
    在段中间，只看端点会漏掉。
    """
    pts: List[Tuple[float, float]] = []
    for s in segments:
        try:
            ax, ay = float(s[0][0]), float(s[0][1])
            bx, by = float(s[-1][0]), float(s[-1][1])
        except (TypeError, IndexError, ValueError):
            continue
        if not (math.isfinite(ax) and math.isfinite(ay)
                and math.isfinite(bx) and math.isfinite(by)):
            continue
        L = math.hypot(bx - ax, by - ay)
        k = max(1, int(math.ceil(L / max(0.5, step))))
        for i in range(k + 1):
            t = i / k
            pts.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    if not pts:
        return np.zeros((0, 2), np.float64)
    return np.asarray(pts, np.float64)


def _edges(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # 顺序固定：0=上(y1) 1=右(x2) 2=下(y2) 3=左(x1)；图像坐标 y 向下
    return [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]


def _normals(bbox):
    """四条边朝**框外**的单位法线（图像坐标，y 向下）。"""
    return [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]


def outline_hits(pts: np.ndarray, bbox, tol: float = TOUCH_TOL):
    """返回 (edge_idx, arc_len, dist, point) —— 落在框边 `tol` 内的采样点。

    用「到**框边**（不是实心矩形）的距离」：只算实心矩形距离的话，框内部远处的
    点距离也是 0，会把整个框内的东西全当成接触点。
    """
    if not len(pts):
        return np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros((0, 2))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # 每条边算一次：参数 t、垂距 d、弧长 s
    best_d = np.full(len(pts), np.inf)
    best_e = np.zeros(len(pts), int)
    best_s = np.zeros(len(pts))
    for e, ((ax, ay), (bx, by)) in enumerate(_edges(bbox)):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            continue
        t = ((pts[:, 0] - ax) * dx + (pts[:, 1] - ay) * dy) / L2
        t = np.clip(t, 0.0, 1.0)
        cx = ax + t * dx
        cy = ay + t * dy
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        s = t * math.sqrt(L2)
        upd = d < best_d
        best_d = np.where(upd, d, best_d)
        best_e = np.where(upd, e, best_e)
        best_s = np.where(upd, s, best_s)
    keep = best_d <= tol
    return best_e[keep], best_s[keep], best_d[keep], pts[keep]


def _cluster(edges: np.ndarray, arc: np.ndarray, gap: float):
    """按 (边号, 沿边弧长) 聚类；同一条边上弧长间距 > gap 就断开。

    返回 [(edge_idx, [(arc, ...)], ...)]，每组内按弧长排序。
    """
    order = np.lexsort((arc, edges))
    out = []
    cur_e = None
    cur_s: List[float] = []
    for k in order:
        e, s = int(edges[k]), float(arc[k])
        if cur_e is None or e != cur_e or (s - cur_s[-1]) > gap:
            if cur_s:
                out.append((cur_e, cur_s))
            cur_e, cur_s = e, [s]
        else:
            cur_s.append(s)
    if cur_s:
        out.append((cur_e, cur_s))
    return out


def _point_on_edge(bbox, e: int, s: float):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if e == 0:
        return (x1 + s, y1)
    if e == 1:
        return (x2, y1 + s)
    if e == 2:
        return (x2 - s, y2)
    return (x1, y2 - s)


def _snap_leg(pins, bbox, legs):
    """把每个点重摆到「最近框边 + 该侧腿长」（切向坐标不动）。"""
    if not legs or len(legs) != 4 or not pins:
        return pins
    x1, y1, x2, y2 = [float(v) for v in bbox]
    lL, lR, lT, lB = [float(v) for v in legs]
    out = []
    for p in pins:
        px, py = float(p[0]), float(p[1])
        dl, dr, dt, db = abs(px - x1), abs(px - x2), abs(py - y1), abs(py - y2)
        m = min(dl, dr, dt, db)
        if m == dl:
            out.append((x1 - lL, py))
        elif m == dr:
            out.append((x2 + lR, py))
        elif m == dt:
            out.append((px, y1 - lT))
        else:
            out.append((px, y2 + lB))
    return out


def _leg_for(nx: float, ny: float, out_dist) -> float:
    """标量直接用；4 元组按法线方向取 (L,R,T,B)。设 `STAGE1_PIN_LEG_BOX=""` 即退回标量。"""
    if isinstance(out_dist, (tuple, list)) and len(out_dist) == 4:
        if nx < -0.5:
            return float(out_dist[0])
        if nx > 0.5:
            return float(out_dist[1])
        if ny < -0.5:
            return float(out_dist[2])
        return float(out_dist[3])
    return float(out_dist)


def pins_from_touch(bbox, pts: np.ndarray, tol: float = TOUCH_TOL,
                    cluster_r: float = CLUSTER_R, out_dist: float = PIN_OUT):
    """导线接触点 -> 引脚（朝框外推 `out_dist`）。"""
    e, s, _d, _p = outline_hits(pts, bbox, tol)
    if not len(e):
        return []
    normals = _normals(bbox)
    pins = []
    for ed, arcs in _cluster(e, s, cluster_r):
        smid = float(np.median(arcs))
        bx, by = _point_on_edge(bbox, ed, smid)
        nx, ny = normals[ed]
        _leg = _leg_for(nx, ny, out_dist)
        pins.append((bx + nx * _leg, by + ny * _leg))
    return pins


def pins_from_prior(bbox, ctype: str):
    """没有任何导线触到时，按类型先验兜底。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = abs(x2 - x1), abs(y2 - y1)
    t = (ctype or "").split(":")[0]
    if t in TWO_PIN:
        # 引脚在长轴两端，离中心 L/2 + 10（GT 实测）
        if h >= w:
            d = h / 2.0 + PIN_OUT
            return [(cx, cy - d), (cx, cy + d)]
        d = w / 2.0 + PIN_OUT
        return [(cx - d, cy), (cx + d, cy)]
    if t in ONE_PIN:
        # 一脚符号：贴较长的那条边中点朝外；gt 里上下不定，取较长的边更常见
        if h >= w:
            return [(cx, y1 - PIN_OUT)]
        return [(cx, y2 + PIN_OUT)]
    # 2026-07 审计修复：这里原来是 `if t in ONE_PIN or True:`，`or True` 让 ONE_PIN 完全失效、
    # 本行 return [] 永不可达，于是**所有**不在 TWO_PIN 的类（实际就是 box 大 IC）在没有导线
    # 触框时被凭空造出 1 个引脚（长边中点外 10px）。那个假脚随后变成一条单脚网络
    # （nets.py 的 MIN_NET_PINS = 1 挡不住），还可能落进 join_dist(12) 被绑到邻近真网上。
    # 自检抓不到：_selftest 用的 "r"/"v" 本来就在 TWO_PIN/ONE_PIN 里。
    # box 的引脚应由 pins_from_touch / _merge_stubs(几何桩) 给出，先验兜底对 box 无意义。
    return []


def _ink_leg_box(bb, ink):
    """从墨迹量这颗元件每一侧的引线长度，返回 `(L, R, T, B)`；量不出返回 None。

    `Pin.ln` 是「框边外 4px 起点 → 该行带内最靠外的墨像素」的长度，所以腿长 = `4 + ln`。
    同一侧可能同时量到「真引线」和「贴着框边的引脚号数字」，真引线更长 ⇒ 取**中位数**
    再夹到 `[PIN_LEG_INK_MIN, PIN_LEG_INK_MAX]`；某侧没量到就留 None 交给常数兜底。
    """
    try:
        from . import pintrace as _pt
    except ImportError:                                   # pragma: no cover
        import pintrace as _pt
    if ink is None:
        return None
    try:
        found = _pt.find_pins(ink, [list(bb)])
    except Exception:                                     # pragma: no cover
        return None
    if not found:
        return None
    sides = {"L": [], "R": [], "T": [], "B": []}
    for p in found:
        if p.dx < -0.5:
            sides["L"].append(4.0 + float(p.ln))
        elif p.dx > 0.5:
            sides["R"].append(4.0 + float(p.ln))
        elif p.dy < -0.5:
            sides["T"].append(4.0 + float(p.ln))
        else:
            sides["B"].append(4.0 + float(p.ln))
    out = []
    for k in ("L", "R", "T", "B"):
        vals = sides[k]
        if not vals:
            out.append(None)
            continue
        v = float(np.median(vals))
        out.append(min(PIN_LEG_INK_MAX, max(PIN_LEG_INK_MIN, v)))
    if all(v is None for v in out):
        return None
    return tuple(out)


def _leg_tuple(base, measured):
    """把测得的逐侧腿长叠加到常数元组上：量不到的那一侧沿用常数。"""
    if measured is None:
        return base
    if not (isinstance(base, (tuple, list)) and len(base) == 4):
        base = (base, base, base, base)
    return tuple(float(base[i]) if measured[i] is None else float(measured[i]) for i in range(4))


def _stub_points(bb, ink, out_dist, min_len=0.0):
    """`pintrace.find_pins` 的几何引脚桩，统一重排到「框外 `out_dist` px」。

    `min_len > 0` 时只保留 `pintrace.Pin.ln >= min_len` 的桩（长桩 = 真引线，
    短桩 = 贴在框边的引脚号数字，见 `STAGE1_PIN_STUB_MINLEN` 注释）。
    """
    try:
        from . import pintrace as _pt
    except ImportError:                                   # pragma: no cover
        import pintrace as _pt
    x0, y0, x1, y1 = [float(v) for v in bb]
    out = []
    for p in _pt.find_pins(ink, [list(bb)]):
        if min_len > 0 and float(getattr(p, "ln", 0.0)) < min_len:
            continue
        if p.dx < 0:
            q = (x0 - out_dist, float(p.y))
        elif p.dx > 0:
            q = (x1 + out_dist, float(p.y))
        elif p.dy < 0:
            q = (float(p.x), y0 - out_dist)
        else:
            q = (float(p.x), y1 + out_dist)
        if not any(math.hypot(q[0] - r[0], q[1] - r[1]) <= 3.0 for r in out):
            out.append(q)
    return out


def _merge_stubs(pins, bb, ink, out_dist):
    """把 `pintrace.find_pins` 找到的几何引脚桩并进 `pins`（3px 内去重）。

    桩的原始点是「框外 4px」，这里一律重排到 GT 的约定位置（框外 `out_dist` px），
    和 `pins_from_touch` 的外推保持一致 —— GT 实测引脚到框距离 p50 = 10px，
    且 100% 在框外。
    """
    out = list(pins)
    for q in _stub_points(bb, ink, out_dist):
        if not any(math.hypot(q[0] - r[0], q[1] - r[1]) <= 3.0 for r in out):
            out.append(q)
    return out


def _color_leg_box(bb, sym):
    """用符号色掩膜量 (L, R, T, B) 四侧引线长度；量不到的侧为 None。

    做法：对每条边、沿边逐行/逐列看框外 `PIN_LEG_COLOR_MAX` 以内的符号色像素，
    取「从框边出发的连续段长」（允许 `PIN_LEG_COLOR_GAP` 个空像素，抗抗锯齿断点），
    同一侧全部段长取中位数。段长 <1 的不计（噪声）。
    """
    if sym is None or not len(sym):
        return None
    H, W = sym.shape[:2]
    x0, y0, x1, y1 = [int(round(float(v))) for v in bb]
    R = int(max(4, PIN_LEG_COLOR_MAX))
    gap = max(0, int(PIN_LEG_COLOR_GAP))
    out = []
    for side in ("L", "R", "T", "B"):
        lens = []
        if side in ("L", "R"):
            ya, yb = max(0, y0), min(H, y1 + 1)
            if side == "L":
                xa, xb, step, base = max(0, x0 - R), max(1, x0), -1, x0
            else:
                xa, xb, step, base = min(W, x1 + 1), min(W, x1 + 1 + R), 1, x1
            if xb - xa < 1 or yb - ya < 2:
                continue
            sub = sym[ya:yb, xa:xb]
            for r in range(sub.shape[0]):
                row = sub[r]
                idx = np.nonzero(row)[0]
                if not idx.size:
                    continue
                # 从框边一侧起算（L 边取最大列号、R 边取最小列号）
                edge_col = row.size - 1 if step < 0 else 0
                if not row[edge_col]:
                    # 允许框边错开 2px：找最靠边的红像素
                    near = idx.max() if step < 0 else idx.min()
                    if abs(int(near) - edge_col) > 2:
                        continue
                    edge_col = int(near)
                c, last, g = edge_col, edge_col, 0
                while 0 <= c + step < row.size:
                    c += step
                    if row[c]:
                        last, g = c, 0
                    else:
                        g += 1
                        if g > gap:
                            break
                L = abs(last - edge_col) + 1
                if L >= 2:
                    lens.append(float(L))
        else:
            xa, xb = max(0, x0), min(W, x1 + 1)
            if side == "T":
                ya, yb, step, base = max(0, y0 - R), max(1, y0), -1, y0
            else:
                ya, yb, step, base = min(H, y1 + 1), min(H, y1 + 1 + R), 1, y1
            if yb - ya < 1 or xb - xa < 2:
                continue
            sub = sym[ya:yb, xa:xb]
            for c in range(sub.shape[1]):
                col = sub[:, c]
                idx = np.nonzero(col)[0]
                if not idx.size:
                    continue
                edge_row = col.size - 1 if step < 0 else 0
                if not col[edge_row]:
                    near = idx.max() if step < 0 else idx.min()
                    if abs(int(near) - edge_row) > 2:
                        continue
                    edge_row = int(near)
                r, last, g = edge_row, edge_row, 0
                while 0 <= r + step < col.size:
                    r += step
                    if col[r]:
                        last, g = r, 0
                    else:
                        g += 1
                        if g > gap:
                            break
                L = abs(last - edge_row) + 1
                if L >= 2:
                    lens.append(float(L))
        out.append(float(np.median(lens)) if len(lens) >= 2 else None)
    if PIN_LEG_UNI:
        sat = R - 0.5
        ok = [v for v in out if v is not None and v < sat]
        if ok and any(v is None or v >= sat for v in out):
            med = float(np.median(ok))
            out = [v if (v is not None and v < sat) else med for v in out]
    if all(v is None for v in out):
        PIN_LEG_COLOR_N["empty"] += 1
        return None
    return tuple(out)


def _leg_fill(pins, bb):
    """把偏短的 IC 脚沿所在边的外法线推到固定腿长（`STAGE1_PIN_LEG_FILL`）。

    new2026 的 GT 把脚点画在框边外 19~20px（`open_GT` 实测 p10 19 / p50 20），挂到导线
    端点的脚已经在正确位置，剩下「到框边距离 < `PIN_LEG_FILL_THRESH`」的脚只有常数腿长
    （10~12px）⇒ 差 8~10px 直接超过 `pin_tol=5`。离线扫描证明阈值必须卡在 15 附近：
    补 55 根 +2.02 分，补 94 根反而 −0.42（把已经对的脚推走）。
    """
    if not pins or not bb or len(bb) < 4:
        return pins, 0
    x0, y0, x1, y1 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    out, moved = [], 0
    for p in pins:
        dl, dr, dt, db_ = abs(p[0] - x0), abs(p[0] - x1), abs(p[1] - y0), abs(p[1] - y1)
        m = min(dl, dr, dt, db_)
        if m >= PIN_LEG_FILL_THRESH:
            out.append(p)
            continue
        if m == dl:
            out.append((x0 - PIN_LEG_FILL_TARGET, p[1]))
        elif m == dr:
            out.append((x1 + PIN_LEG_FILL_TARGET, p[1]))
        elif m == dt:
            out.append((p[0], y0 - PIN_LEG_FILL_TARGET))
        else:
            out.append((p[0], y1 + PIN_LEG_FILL_TARGET))
        moved += 1
    return out, moved


def _snap_to_wire_end(pins, bb, ends, free=None, other=None):
    """把两脚件的引脚点挪到「朝外的、度 1 的导线端点」上（`STAGE1_PIN_WIRE_END`）。

    `pins` 是 `(x, y)` 列表（框边 + 常数腿长）；对每个点，向元件中心的反方向看
    `PIN_WIRE_R` 以内，只在「切向偏移 ≤ `PIN_WIRE_TAN`」的端点里取最近的一个：
    同一颗元件上的两个引脚各自找自己的端点，绝不抢（一对一）。

    `other`（可选，与 `ends` 逐位对齐）= 每个候选端点所在线段的**另一端**；
    `PIN_WIRE_SELF=1` 时，另一端落在这颗元件框内（外扩 `PIN_WIRE_SELF_PAD`）的候选
    直接排除 —— 那是本件自己的引线，它的外端不是导线端点。
    """
    cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
    if free is not None and len(free) and bool(np.any(free)):
        idxs = np.nonzero(free)[0]
    elif free is not None and len(free) and PIN_WIRE_SEGLEN > 0:
        idxs = np.zeros(0, int)               # 没有真导线端点 ⇒ 一个都不吸
    else:
        idxs = np.arange(len(ends), dtype=int)
    if not len(idxs) and PIN_WIRE_SEGLEN <= 0:
        idxs = np.arange(len(ends), dtype=int)
    cand = ends[idxs] if len(idxs) else ends[:0]
    own = None
    if PIN_WIRE_SELF and other is not None and len(idxs) and len(other) == len(ends):
        own = np.asarray(other, np.float64)[idxs]
    used = set()
    out = []
    for p in pins:
        vx, vy = p[0] - cx, p[1] - cy
        L = math.hypot(vx, vy)
        if L < 1e-6:
            out.append(p)
            continue
        ux, uy = vx / L, vy / L
        best, bd = None, PIN_WIRE_R + 1e-6
        for i, e in enumerate(map(tuple, cand)):
            if i in used:
                continue
            if own is not None:
                ox, oy = float(own[i][0]), float(own[i][1])
                if (bb[0] - PIN_WIRE_SELF_PAD <= ox <= bb[2] + PIN_WIRE_SELF_PAD and
                        bb[1] - PIN_WIRE_SELF_PAD <= oy <= bb[3] + PIN_WIRE_SELF_PAD):
                    continue
            dx, dy = e[0] - p[0], e[1] - p[1]
            d = math.hypot(dx, dy)
            if d > bd or d < PIN_WIRE_MIN:
                continue
            along = dx * ux + dy * uy                    # 朝外分量
            across = abs(-dx * uy + dy * ux)             # 切向偏移
            if along < 0.5 * d or across > PIN_WIRE_TAN:
                continue
            best, bd = i, d
        if best is None:
            PIN_WIRE_N["miss"] += 1
            out.append(p)
        else:
            used.add(best)
            PIN_WIRE_N["n"] += 1
            out.append((float(cand[best][0]), float(cand[best][1])))
    return out


def find_pins(components, segments, texts=None, touch_tol: float = TOUCH_TOL,
              cluster_r: float = CLUSTER_R, out_dist: float = PIN_OUT,
              sample_step: float = SAMPLE_STEP, use_prior: bool = True,
              ink=None, sym_ink=None):
    """给每个元件找引脚。返回 {comp_id: [(x, y), ...]}（图像坐标）。

    `texts` 用于「文字不是引脚/桩」的两档处理：`STAGE1_PIN_STUB_TEXT=1` 只让**桩证据**
    去字（默认关，见 `STUB_FREE` 的长注释），`STAGE1_PIN_INK_TEXT=1` 让**整条 find_pins**
    （触点 + 桩）都去字（默认关，见 `INK_FREE`）。

    `ink` 非空且 `STAGE1_PIN_STUB=1` 时，对 `box`/`bigic` 追加几何引脚桩。
    """
    if INK_FREE and ink is not None and texts:
        try:
            from . import pincol as _pc
        except ImportError:                               # pragma: no cover
            import pincol as _pc
        try:
            ink = _pc.text_free_ink(ink, texts, pad=INK_FREE_PAD)
        except Exception:                                 # pragma: no cover
            pass
    pts = sample_segments(segments, sample_step)
    # ★ 两脚元件「引脚点落到导线端点」的候选端点（`STAGE1_PIN_WIRE_END`）。
    _we_pos = _we_free = _we_other = None
    if (PIN_WIRE_END or PIN_WIRE_END_IC) and len(segments):
        _ends = np.asarray(
            [[float(s[0][0]), float(s[0][1])] for s in segments] +
            [[float(s[-1][0]), float(s[-1][1])] for s in segments], np.float64)
        # 与 `_ends` 逐位对齐的「另一端」（前半是起点 ⇒ 另一端是终点，反之亦然）。
        _others = np.asarray(
            [[float(s[-1][0]), float(s[-1][1])] for s in segments] +
            [[float(s[0][0]), float(s[0][1])] for s in segments], np.float64)
        if len(_ends):
            _d2 = ((_ends[:, None, :] - _ends[None, :, :]) ** 2).sum(-1)
            # 只留「真导线」的端点：线段够长（符号里的小笔画都短），
            # 否则 IC 的脚会被吸到符号自身的笔画端点上（实测 10GT 会因此掉到 80）。
            _glen = []
            for s in segments:
                _glen.append(math.hypot(float(s[-1][0]) - float(s[0][0]),
                                        float(s[-1][1]) - float(s[0][1])))
            _glen = np.asarray(_glen, np.float64)
            _ok = _glen >= PIN_WIRE_SEGLEN
            _we_pos = _ends
            _we_other = _others
            _deg1 = (_d2 <= 4.0).sum(1) == 1
            _we_free = (_deg1 if PIN_WIRE_FREE else np.ones(len(_ends), bool)) & \
                np.concatenate([_ok, _ok])
    # ★ 跨引线吸附的目标点集（见 `STAGE1_PIN_LEAD`）：整张图只算一次。
    _lead_tgt = _lead_mask = None
    if PIN_LEAD > 0 and len(segments):
        if PIN_LEAD_SRC == "any":
            _lead_tgt = pts
        else:
            _lead_tgt = np.asarray(
                [[float(s[0][0]), float(s[0][1])] for s in segments] +
                [[float(s[-1][0]), float(s[-1][1])] for s in segments], np.float64)
            if PIN_LEAD_FREE and len(_lead_tgt):
                _d2 = ((_lead_tgt[:, None, :] - _lead_tgt[None, :, :]) ** 2).sum(-1)
                # 度 1 = 2px 内没有别的线段端点（自己算 1 个）
                _lead_mask = (_d2 <= 4.0).sum(1) == 1
    out: Dict[str, List[Tuple[float, float]]] = {}
    LATENT.clear()                     # 每次调用重置（ProcessPool worker 会复用进程）
    for c in components:
        bb = c.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        _ct = str(c.get("type") or "").lower()
        _is_ic = ("box" in _ct or "bigic" in _ct)
        _od = PIN_OUT_BOX if _is_ic else out_dist
        _leg = ((PIN_LEG_BOX or _od) if (_is_ic or PIN_LEG_ALL) else out_dist)
        if PIN_LEG_INK and _is_ic and ink is not None:
            _inkm = ink
            if texts:
                try:
                    from . import pincol as _pc
                except ImportError:                       # pragma: no cover
                    import pincol as _pc
                try:
                    _inkm = _pc.text_free_ink(ink, texts, pad=STUB_FREE_PAD)
                except Exception:                         # pragma: no cover
                    _inkm = ink
            _leg = _leg_tuple(_leg, _ink_leg_box(bb, _inkm))
            LEG_INK_N["n"] += 1
        if PIN_LEG_COLOR and _is_ic and sym_ink is not None:
            _cl = _color_leg_box(bb, sym_ink)
            if _cl is not None:
                _leg = _leg_tuple(_leg, _cl)
                PIN_LEG_COLOR_N["n"] += 1
        # IC 的引脚间距只有 ~10px，不能用 12px 的合并半径（会把邻脚并掉）
        _cr = CLUSTER_IC if _is_ic else cluster_r
        _tt = TOUCH_TOL_IC if (_is_ic and TOUCH_TOL_IC > 0) else touch_tol
        pins = pins_from_touch(bb, pts, _tt, _cr, _leg)
        _ev = {"touch": len(pins), "prior": 0, "stub": 0}
        if not pins and use_prior:
            pins = pins_from_prior(bb, c.get("type") or "")
            _ev["prior"] = len(pins)
        if STUB_FILL and ink is not None and _is_ic:
            # ★ 文字不是桩（见 STUB_FREE 的长注释）：抹掉 `texts` 里的文字框再找桩
            _ink_s = ink
            if STUB_FREE and texts:
                try:
                    from . import pincol as _pc
                except ImportError:                       # pragma: no cover
                    import pincol as _pc
                try:
                    _ink_s = _pc.text_free_ink(ink, texts, pad=STUB_FREE_PAD)
                except Exception:                         # pragma: no cover
                    _ink_s = ink
            _sp = _stub_points(bb, _ink_s, _od,
                               min_len=(STUB_MINLEN if STUB_GAP else 0.0))
            _ev["stub_seen"] = len(_sp)
            if STUB_GAP:
                # gapfill：只补「没被任何已有触点覆盖」的长桩（见上面 ★★★ 注释）
                _add = [q for q in _sp
                        if not any(math.hypot(q[0] - r[0], q[1] - r[1]) <= STUB_GAPTOL
                                   for r in pins)]
                if _add:
                    pins = list(pins) + _add
                    _ev["stub"] = len(_add)
                    _ev["stub_gap"] = len(_add)
            else:
                _few = (STUB_SPARSE and _sp
                        and len(pins) < STUB_SPARSE_RATIO * len(_sp))
                if STUB_UNION or not pins or _few:
                    _b4 = len(pins)
                    pins = _merge_stubs(pins, bb, _ink_s, _od)
                    _ev["stub"] = max(0, len(pins) - _b4)
        # 逐侧腿长收尾（§5.340.3）：stub/merge 补出的点仍按标量腿长摆，这里统一重摆。
        if PIN_LEG_SNAP and _is_ic:
            # `=1` 用常数 4 元组；`=2` 用逐侧**量得**的腿长（`_leg`）。同一侧只可能有一个
            # 腿长（物理事实），所以重摆后再按 3px 去重会把「同一行两个槽」并成一个。
            _lgS = _leg if PIN_LEG_SNAP >= 2 else PIN_LEG_BOX
            if _lgS:
                if not isinstance(_lgS, (tuple, list)) or len(_lgS) != 4:
                    _lgS = (_lgS,) * 4
                pins = _snap_leg(pins, bb, _lgS)
        # 去重（同一位置只留一个），再截断
        dedup: List[Tuple[float, float]] = []
        for p in pins:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= 3.0 for q in dedup):
                dedup.append(p)
        # ★ 两脚元件的脚必在长轴两端（见 `PIN_AXIS` 的长注释）：短轴上的候选脚是把
        #   符号笔画/极板当成了引线，会让一个元件凭空多出单脚网络。**只移出活动集**，
        #   原候选进模块级 `LATENT` 并记 `LEDGER.demote`（工程哲学：不删除、只降权）。
        _latent: List[Tuple[float, float]] = []
        _latent_why = ""
        if PIN_AXIS and not _is_ic:
            _t = str(c.get("type") or "").split(":")[0].lower()
            _b = [float(v) for v in bb]
            _w, _h = abs(_b[2] - _b[0]), abs(_b[3] - _b[1])
            _ratio = (max(_w, _h) / max(1e-6, min(_w, _h)))
            if _t not in ONE_PIN and _ratio >= PIN_AXIS_RATIO and len(dedup) >= 2:
                _keep, _drop = _axis_split(bb, dedup)
                if _drop and len(_keep) >= 2:
                    dedup = _keep
                    _latent = _drop
                    _ev["axis_drop"] = len(_drop)          # 供账本/dbg 计数
                    _latent_why = ("短轴候选脚 %d 个（符号笔画/极板，非长轴引线；"
                                   "GT 非 IC 引脚 100%% 沿主轴）" % len(_drop))
        # ★★ 两脚元件的脚是「框边 ∓ PIN_OUT、长轴中线上的一点」（见 `PIN_MAJOR` 的长注释）：
        #   直接盖章这两个点（每侧恰好一个 ⇒ `stamp_pins` 不再撞号），旧候选整体进 `LATENT`。
        if PIN_MAJOR and not _is_ic:
            _tm = str(c.get("type") or "").split(":")[0].lower()
            if _tm in PIN_MAJOR_TYPES:
                _bm = [float(v) for v in bb]
                _mx, _my = (_bm[0] + _bm[2]) / 2.0, (_bm[1] + _bm[3]) / 2.0
                if (_bm[2] - _bm[0]) >= (_bm[3] - _bm[1]):
                    _ideal = [(_bm[0] - _od, _my), (_bm[2] + _od, _my)]
                else:
                    _ideal = [(_mx, _bm[1] - _od), (_mx, _bm[3] + _od)]
                _gone = [q for q in dedup if q not in _ideal]
                if _gone:
                    _latent = list(_latent) + _gone
                    _latent_why = ("长轴盖章：%d 个旧候选移出活动集（GT 两脚元件引脚恒为"
                                   "「框边 ∓ PIN_OUT、长轴中线」，官方 250/250）" % len(_gone))
                dedup = _ideal
                _ev["major"] = len(_ideal)
        # ★ 跨过符号自身引线去挂网（`STAGE1_PIN_LEAD`，默认关；见该常量的长注释）。
        #   只对 stage2 走 `stamp_pins` 的两脚类型做 ⇒ 输出引脚逐位不变，只有网变。
        if PIN_LEAD > 0 and not _is_ic and dedup and _lead_tgt is not None:
            _tl = str(c.get("type") or "").split(":")[0].lower()
            if _tl in PIN_LEAD_TYPES:
                _cc = ((float(bb[0]) + float(bb[2])) / 2.0,
                       (float(bb[1]) + float(bb[3])) / 2.0)
                _snapped, _nsnap = _snap_lead(dedup, _lead_tgt, _cc,
                                              mask=_lead_mask, ink=ink, cloud=pts)
                if _nsnap:
                    dedup = _snapped
                    _ev["lead"] = _nsnap
        if PIN_WIRE_END and (not _is_ic or PIN_WIRE_END_IC) and _we_pos is not None and dedup:
            dedup = _snap_to_wire_end(dedup, bb, _we_pos, _we_free, _we_other)
            _ev["wire_end"] = len(dedup)
        # ★ IC 脚的「腿长补齐」：没挂到导线端点的脚只有常数腿长（10~12px），而 new2026
        #   的 GT 把脚点画在框边外 19~20px ⇒ 沿所在边外法线推出去（`STAGE1_PIN_LEG_FILL`）。
        if _is_ic and PIN_LEG_FILL and dedup:
            _new, _moved = _leg_fill(dedup, bb)
            if _moved:
                dedup = _new
                _ev["leg_fill"] = _moved
        out[c["id"]] = dedup[:MAX_PINS]
        if _latent:
            LATENT[c["id"]] = list(_latent)
        # ★ 2026-10-10（用户 m42009）：「引脚在等导线交叉验证」必须留下证据链。
        #   每个元件的引脚来源进账本：`touch` = 上游导线真的触到框边（交叉验证通过）；
        #   `prior` = 没有任何导线证据、纯类型先验造出来的（**未确认**，按「只削弱不删除」
        #   记 `score_after=DEMOTE_UNCONFIRMED`，仍留在池里）；`stub` = 几何小桩（半证据）；
        #   三者全 0 = 这个元件在 stage1 里**一个引脚都没有**，下游网络/引脚通道全空 ——
        #   正是「元件与导线互相等」的终点，以前这里什么都不记。
        try:
            from cvparts.hypo import LEDGER, DEMOTE_HARD, DEMOTE_SOFT
            _n = len(out[c["id"]])
            _src = ("touch" if _ev["touch"] else
                    ("prior" if _ev["prior"] else ("stub" if _ev["stub"] else "none")))
            LEDGER.record("evidence", where="pin:%s" % c["id"],
                          before={"type": c.get("type"), "n": _n},
                          after={"src": _src, **{k: _ev[k] for k in ("touch", "prior", "stub")}},
                          reason="引脚来源：%s" % _src,
                          by="stage1/pins.py:find_pins")
            if _src == "none":
                LEDGER.demote(where="pin:%s" % c["id"], value=c.get("id"),
                              score_before=1.0, score_after=DEMOTE_HARD,
                              reason="没有任何导线触点/先验/几何桩 —— 该元件无引脚，下游网络为空",
                              by="stage1/pins.py:find_pins")
            elif _src == "prior":
                LEDGER.demote(where="pin:%s" % c["id"], value=c.get("id"),
                              score_before=1.0, score_after=DEMOTE_SOFT,
                              reason="仅类型先验（无导线交叉验证）",
                              by="stage1/pins.py:find_pins")
            if _latent:
                # 「不删除只削弱」：短轴候选脚移出活动集，但每个都留在账本里，
                # 下游要复核时可以按 where=pins_latent 把它们捞回来。
                LEDGER.demote(where="pins_latent", value=c.get("id"),
                              score_before=1.0, score_after=DEMOTE_SOFT,
                              reason=(_latent_why or
                                      "短轴候选脚 %d 个移出活动集" % len(_latent)),
                              by="stage1/pins.py:find_pins")
                LEDGER.record("evidence", where="pin:%s" % c["id"],
                              before={"n": len(out[c["id"]]) + len(_latent)},
                              after={"n": len(out[c["id"]]), "axis_drop": len(_latent),
                                     "latent": [[round(float(q[0]), 1), round(float(q[1]), 1)]
                                                for q in _latent[:8]]},
                              reason="长轴裁剪：短轴候选脚移出活动集（STAGE1_PIN_AXIS=1）",
                              by="stage1/pins.py:find_pins")
        except Exception:
            pass
    return out



# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    """合成自检。**每次改这个模块都先跑**：本项目已经在「评测函数没自检」上栽过
    三次（网格索引静默漏找、`argmin` 恒为 0、段形状 (M,) vs (M,N)），三次都只有
    真实数据给一个说不清对错的数字。
    """
    fails = []

    def chk(name, got, want, tol=1e-6):
        ok = len(got) == len(want) and all(
            abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
            for a, b in zip(sorted(got), sorted(want)))
        if not ok:
            fails.append(name)
        print("  [%s] %-40s got %s" % ("OK" if ok else "FAIL", name, got))

    box = (100.0, 100.0, 200.0, 300.0)
    print("1) 一根线从左边接到框边中点 -> 引脚应朝框外推 10px")
    chk("左边中点", pins_from_touch(box, sample_segments([[(100, 200), (0, 200)]])),
        [(90.0, 200.0)])

    print("2) 左边两根线，分别在 y=150 / y=250")
    chk("左边两点", pins_from_touch(box, sample_segments([[(100, 150), (0, 150)],
                                                          [(100, 250), (0, 250)]])),
        [(90.0, 150.0), (90.0, 250.0)])

    print("3) 一根线**穿过**框（串联元件）-> 左右各一个引脚")
    chk("穿过", pins_from_touch(box, sample_segments([[(0, 200), (300, 200)]])),
        [(90.0, 200.0), (210.0, 200.0)])

    print("4) 线离框 15px -> 不算触到")
    chk("差 15px", pins_from_touch(box, sample_segments([[(85, 200), (0, 200)]])), [])

    print("5) 线**沿着**左边界画（y 120..200）-> 整段贴边，算 1 个引脚、取中点 y=160")
    # 注意期望值是 160 不是 200：整段都贴边时它本来就是一个「沿边的长接触」，
    # 取这段的中点是合理行为。我第一版把期望写成 200（线的起点），是测试写错了。
    chk("沿边画", pins_from_touch(box, sample_segments([[(100, 200), (100, 120)]])),
        [(90.0, 160.0)])

    print("6) 上边 + 右边各一根 -> 法线方向要对（上边推 -y，右边推 +x）")
    chk("上+右", pins_from_touch(box, sample_segments([[(150, 100), (150, 0)],
                                                       [(200, 200), (300, 200)]])),
        [(150.0, 90.0), (210.0, 200.0)])

    print("7) 类型先验兜底：两脚元件引脚在长轴两端、离中心 L/2+10")
    rbox = (208.0, 340.0, 216.0, 360.0)     # GT 例 R5：引脚 y=330 / 370
    chk("R5 长轴两端", pins_from_prior(rbox, "r"), [(212.0, 330.0), (212.0, 370.0)])

    print("8) 方框电源符号（GT: ∅202 bbox 10x10 -> 引脚离中心 15）")
    chk("方框 1 脚", pins_from_prior((1074.0, 286.0, 1084.0, 296.0), "v"),
        [(1079.0, 276.0)])

    print("9) 两脚元件：有导线触框时也走主轴盖章（`STAGE1_PIN_MAJOR`）")
    comps = [{"id": "c1", "type": "r", "bbox_img": [100, 100, 200, 300]}]
    got = find_pins(comps, [[[100, 200], [0, 200]]], use_prior=True)["c1"]
    if PIN_MAJOR:
        # ★ 行为变更（2026-10-10，默认 1）：两脚元件的脚**不再**取「贴着框的那条导线段」，
        # 而是按 GT 几何律盖章「框边 ∓ PIN_OUT、长轴中线」——官方 10GT 里 250/250 个两脚
        # 引脚都长这样（外伸恰 10.0、短轴偏移 0.00），而贴框的常是文字笔画。见 `PIN_MAJOR`。
        chk("盖章主轴两端", got, [(150.0, 90.0), (150.0, 310.0)])
    else:
        chk("触到就不兜底", got, [(90.0, 200.0)])

    print("10) 一根线都没有时走先验")
    got = find_pins(comps, [], use_prior=True)["c1"]
    chk("没线走先验", got, [(150.0, 90.0), (150.0, 310.0)])

    print("11) 回归：大 IC（box）没有导线触框时**不得**凭空造脚")
    # 这里以前是 `if t in ONE_PIN or True:`，box 会拿到 1 个假脚（长边中点外 10px），
    # 接着变成一条单脚网络、还可能被 join_dist 绑到邻近真网上。见 pins_from_prior 的注释。
    chk("box 不兜底", pins_from_prior((100.0, 100.0, 200.0, 300.0), "box"), [])
    chk("bigic 不兜底", pins_from_prior((100.0, 100.0, 200.0, 300.0), "bigic"), [])
    _b = [{"id": "b1", "type": "box", "bbox_img": [100, 100, 200, 300]}]
    chk("box 无线无墨 -> 0 脚", find_pins(_b, [], use_prior=True)["b1"], [])

    print("12) 长轴裁剪 `_axis_split`：竖长框只留上下两端，左右两个短轴候选被移出")
    keep, drop = _axis_split((68.0, 147.8, 84.0, 167.8),
                             [(76.5, 137.8), (94.0, 158.5), (76.5, 177.8), (58.0, 158.5)])
    chk("竖长框 keep", keep, [(76.5, 137.8), (76.5, 177.8)])
    chk("竖长框 drop", drop, [(94.0, 158.5), (58.0, 158.5)])
    keep, drop = _axis_split((100.0, 100.0, 200.0, 108.0),
                             [(96.0, 104.0), (204.0, 104.0), (150.0, 96.0)])
    chk("横长框 keep", keep, [(96.0, 104.0), (204.0, 104.0)])
    chk("横长框 drop", drop, [(150.0, 96.0)])
    chk("主轴只有一个候选 -> 不裁", _axis_split((100.0, 100.0, 200.0, 300.0),
                                                [(90.0, 200.0), (150.0, 90.0)])[1], [])

    print("")
    print("13) 跨引线吸附 `_snap_lead`：正对外侧、横向 ≤ perp、距离 ≤ lead 才算")
    _ctr = (150.0, 300.0)
    _pts = np.asarray([(150.0, 250.0), (170.0, 260.0), (150.0, 230.0)], dtype=float)
    _got, _n = _snap_lead([(150.0, 270.0)], _pts, _ctr, lead=26.0, perp=3.0)
    chk("正对 20px -> 吸到导线端", _got, [(150.0, 250.0)])
    if _n != 1:
        fails.append("正对 20px 计数")
    chk("lead 太短 -> 不吸", _snap_lead([(150.0, 270.0)], _pts, _ctr,
                                        lead=15.0, perp=3.0)[0], [(150.0, 270.0)])
    chk("旁偏 20px -> 不吸（perp=3）",
        _snap_lead([(150.0, 270.0)], np.asarray([(170.0, 260.0)]),
                   _ctr, lead=26.0, perp=3.0)[0], [(150.0, 270.0)])
    chk("旁偏 20px -> 吸（perp=24）",
        _snap_lead([(150.0, 270.0)], np.asarray([(170.0, 260.0)]),
                   _ctr, lead=26.0, perp=24.0)[0], [(170.0, 260.0)])
    chk("mask 只放行远点 -> 远点超 lead 不吸",
        _snap_lead([(150.0, 270.0)], _pts, _ctr, lead=26.0, perp=3.0,
                   mask=np.asarray([False, False, True]))[0], [(150.0, 270.0)])
    chk("朝内（点已在引脚内侧）-> 不吸",
        _snap_lead([(150.0, 270.0)], np.asarray([(150.0, 285.0)]),
                   _ctr, lead=26.0, perp=3.0)[0], [(150.0, 270.0)])
    # 孤儿判据：已经挂上网的脚（全量点云里有 ≤12px 的点）不许搬走，否则拆桥
    _cloud_near = np.asarray([(150.0, 250.0), (150.0, 265.0)], dtype=float)
    chk("已挂网（5px）-> 不吸",
        _snap_lead([(150.0, 270.0)], _pts, _ctr, lead=26.0, perp=3.0,
                   cloud=_cloud_near)[0], [(150.0, 270.0)])
    chk("孤儿（30px）-> 吸",
        _snap_lead([(150.0, 270.0)], _pts, _ctr, lead=26.0, perp=3.0,
                   cloud=np.asarray([(150.0, 240.0)]))[0], [(150.0, 250.0)])

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
