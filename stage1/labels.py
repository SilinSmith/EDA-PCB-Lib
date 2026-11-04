# -*- coding: utf-8 -*-
"""第一阶段：网络标签（net label）的定位与「逻辑合并」的地基。

**为什么必须有这一层**：拿 GT **自己**的走线去连同一 net 的引脚，也只有
R 0.658 / net 完整率 0.598 —— 剩下 34% 的同 net 引脚纸上**根本没画线**，它们是靠
网络标签（net label）按名字逻辑相连的。任何纯几何提取都拿不到这 34%。

所以导线提完之后还要有一步：**找出画在导线自由端上的那些文字框**，把「文字框 ↔
导线连通分量」绑起来。文字**内容**由第二阶段（`textrec`）填，本模块只负责
**定位与绑定** —— 这样逻辑合并就变成一句 `group by 标签文本`。

语义：一张原理图里，网络标签总是画在一条走线的**自由端**（只接了一个东西的那头），
不会画在走线中段。所以
  1. 把段并成连通分量（`wire_groups`）；
  2. 找出每个分量里**只挂在一条段上**的端点（自由端）；
  3. 每个自由端找 `max_dist` 内最近的一个文字框，即该分量的标签候选。

对外 API：
    wire_groups(segments, tol=GROUP_TOL) -> list[int]        每条段的组号
    free_ends(segments, groups, tol=GROUP_TOL) -> list[(x, y, group)]
    label_candidates(segments, texts, groups=None, max_dist=LABEL_MAX_DIST)
        -> list[dict]  每项 {text_id, bbox_img, center, group, dist, n_free_ends}
    merge_by_label(groups, named_labels) -> dict[int, str]
        把「标签文本相同」的组并成同一个名字（逻辑合并的落点）

自检：`python eda-pcb\stage1\labels.py [--cases N]`
"""
import argparse
try:                                    # 作为包导入（python -m stage1 / submission\run.py）
    from envflag import env_on      # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage1\labels.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.dirname(HERE)
for p in (EDA,):
    if p not in sys.path:
        sys.path.insert(0, p)

GROUP_TOL = float(os.environ.get("STAGE1_GROUP_TOL", "1.5"))   # 段并分量的端点容差（pskel 的 rdp+snap 让共点端点差 1~2px）
LABEL_MAX_DIST = 16.0    # 自由端到文字框的最大距离
ANCHOR_DIST = 6.0

# ------------------------------------------------ 全项目推荐的导线图配置
# 这套数字是 220 张（`work\stage1_all13`）× 多轮扫描定出来的，**别的地方不要再
# 自己写一套图定义**：本项目已经有四份（`stage1_conn_eval.build_groups`、
# `stage1_filter_ablation.conn_stats`、`reconcile.wire_graph`、这里），其中两份
# 各自静默出错、把我误导了半天。现在全部委托到 `wire_groups`。
#
# 实测（`stage1_conn_eval.py --stage1 stage1_all13 --tol 1.5 --ext ... --ext-all`）：
#   ext = 0  R 0.1752 P 0.5317 F1 0.2636  net完整率 0.3425
#   ext = 4  R 0.2508 P 0.3393 F1 0.2884  0.3934
#   ext = 6  R 0.2813 P 0.3161 F1 0.2977  0.4140
#   ext = 8  R 0.3185 P 0.3038 F1 0.3110  0.4333   <- 峰值
#   ext = 10 R 0.3555 P 0.2719 F1 0.3081  0.4556
#   ext = 12 R 0.3881 P 0.2352 F1 0.2929  0.4720
#   ext = 16 R 0.4223 P 0.1681 F1 0.2404  0.4954
#   ext = 32 R 0.5384 P 0.0584 F1 0.1054  0.5745
#   ext = 64 R 0.6655 P 0.0313 F1 0.0597  0.6561
# 对照组：GT 自己的 net_line 当段集合时 R 0.6580 / F1 0.7736 / net完整率 0.5977。
# 注意 ext=64 的 R 已经**超过**这个几何天花板，说明那时是在乱并。
#
# 试过但**没用**的两条（都别重复造）：
#   * 方向性桥接 `weld`（只接端点对端点、方向相对）：ext=0 时 R 只从 0.1752 抬到
#     0.1950 而 P 从 0.5317 掉到 0.4388 —— 实测里主流断法**不是**端点对端点。
#   * 要求被撞的段与射线近似垂直（`ext_cos=0.3`）：ext=8 时 F1 0.3062 < 0.3110。
# ★★ 2026-10-07（下半）：**`10` 是钝补丁，已被规则 (5) 取代，默认改回 `1.5`。**
#   `WIRE_TOL=10` 确实把官方 9 张从 55.07 推到 55.89，但那是拿「过度并网」换来的：
#   `work\tools\diag\_wirenew.py` 把 case 10 从 tol 1.5 抬到 10 时**新并上的 35 对段**
#   打出来，绝大多数长这样 —— `seg216 [70,751]->[71,758]` 与 `seg207 [77,751]->[78,759]`
#   （相隔 6.8px、**平行**）—— 也就是**电容符号的两块极板**。并上它 = 把电容短路，
#   case 10 的 11 脚巨块（横跨 GT 的 `net_2` SW + `net_5` VCC + `net_8`/`net_13` GND）
#   就是这么来的 ⇒ case 10 −0.96 / case 4 −1.10。
#   规则 (5)（`WIRE_OVER`，见下面那段长注释）只吃「端点靠近目标段**且两段不平行**」，
#   把这两件事分开：
#
#     官方 9 张（基线 `wtol_def` = TOL 10 / OVER 0 ⇒ **55.73**，netH 0.5190）
#       OVER=6  55.38 / 8  55.38 / 10 55.99 / **12 56.31** / **14 56.31** / 16 55.96
#       （TOL 回 1.5、`STAGE1_WIRE_OVER_CROSS=0`；TOL 取 1.5/2/3 **逐位相同** ⇒
#        规则 (5) 一开，`WIRE_TOL` 就不再是杠杆了，回到历史值 1.5 最省事）
#       CROSS=1（只吃穿越）时最好只有 55.38、netH 0.5064 ⇒ **必须放开「欠越」**。
#     ⇒ 默认：`WIRE_TOL=1.5` + `WIRE_OVER=12` + `WIRE_OVER_CROSS=0` = **56.31**（netH 0.5339）。
#   历史（保留作对照）：`3` 55.27 / `6` 55.36 / `8` 55.46 / **`10` 55.89** / `12` 55.91 /
#   `14` 55.83 / `20` 51.01 / `30` 47.88（平台 10~12，20 起崩）；逐 case（10 vs 基线）：
#   `1 +3.52`、`7 +3.57`、`3 +1.92`、`5 +0.93`、`9 −0.01`、`2 −0.16`、`8 −0.32`、
#   `10 −0.96`、`4 −1.10`；派生 19 张中性（34.76 vs 34.77）。
WIRE_TOL = float(os.environ.get("STAGE1_WIRE_TOL", "1.5"))
WIRE_EXT = float(os.environ.get("STAGE1_WIRE_EXT", "8.0"))
WIRE_EXT_PERP = float(os.environ.get("STAGE1_WIRE_EXT_PERP", "2.0"))
WIRE_EXT_NEAREST = env_on("STAGE1_WIRE_EXT_NEAREST", "0")
# ---- 定向「抹字断口」桥接（`STAGE1_TEXT_BRIDGE`，默认 0 = 关）----
# 动机（case 3 实证）：`extract_wires(erase=texts, erase_pad=1.0)` 会把**盖在导线上的
# 文本框像素一起抹掉**，留下 9~12px 的断口 —— C1 上脚的竖线 `seg0=[[156.5,508.9],
# [156.5,433.2]]` 顶到 y=433.2，长横线 `seg46=[[812.1,422.5],[152.0,424.0]]` 在 y=424，
# 缺口 9.2px，缺口里正是被抹掉的 `t0018 [154,426,163,430]`；常规规则接不上
# （`WIRE_TOL=1.5` 太短、`WIRE_EXT=8` 又差一点点）。同型碎片在 case 3 有 7 对
# （8.4~11.7px），全部没并上 ⇒ 网络成员少、netH/netL 吃亏。
# **为什么不能靠抬容差**（都已实测）：`WIRE_TOL=10` 是零和交换（官方 9 张 +0.82 /
# 派生 −0.75，且 case 10 把电容两块极板并成 11 脚巨块）；`WIRE_EXT` 20/30 →
# 53.55/50.48；`WIRE_WELD` 24/32 → 54.33/51.86（都更差）。根因是那些断口躺在
# **元件框**里 —— 一抬就把极板并上。所以这里只认**被擦除的文本框**，且要求两个线端
# 各自贴住**同一个**框（见 `_text_bridge_unions`）。**绝不用元件框做桥接。**
TEXT_BRIDGE = float(os.environ.get("STAGE1_TEXT_BRIDGE", "0"))   # 0 = 关；>0 = 允许的最大跨框缺口(px)
TEXT_BRIDGE_PAD = float(os.environ.get("STAGE1_TEXT_BRIDGE_PAD", "3.0"))   # 框外扩，容端点贴着框边
# ★ `weld` / `perp` 的正确配比：**`perp` 必须跟着 `weld` 放大，否则等于没开 weld。**
#
# `_weld_unions` 里 `perp = |out_A × v̂| * dist` 是「位移偏离 A 的外法线延长线多少像素」，
# 而**走线断口几乎都不是端点在一条理想直线上**：骨架在**结点圆点**处会被切断
# （case 1 右侧那根 GND 竖轨在 x=1431/1433 之间左右抖 2px，被切成
# y = 212-224 / 242-254 / 271-283 / 301-312 / 330-372 五段，缺口 17~18px）。
# 缺口 18px + 横向抖 2px ⇒ `perp = (2/18)*18.1 ≈ 2.0`，**压不住老默认的 `perp_tol=1.5`**
# ⇒ 端点明明在 `weld` 半径内也被 `perp` 拒掉。这就是当年「weld 看起来没用」的真因。
#
# ⚠ **但端到端只能在官方 10GT（10 张 KiCad 图）上赚到，在 220 张上是净亏的。**
# 网格（`run.py` 后 `stage2_eval.py`，无 VLM）：
#
#   官方 10GT（`work\_gt10_in`，KiCad 风格）
#     weld=16 perp=1.5（现默认）42.5191  netH tp190 / line tp54
#     weld=24 perp=1.5            42.2083  tp187 / 51      <- perp 不放大，weld 白开
#     weld=24 perp=6              42.7246  tp194 / 53
#     weld=24 perp=10             42.9809  tp196 / 61
#     weld=24 perp=14  ★峰值      43.0507  tp197 / 61
#     weld=24 perp=20             42.7510  tp194 / 60
#     weld=28 perp=17             41.3243  tp179 / 44      <- 再大就乱并
#     weld=32/36/40/48/56 perp=19/22/24/29/34 -> 40.60/40.33/40.25/39.57/38.76
#
#   ★ 全量 220（`work\_submit_in220`，主要是彩色 Altium 图）
#     weld=16 perp=1.5           **24.8651**  netH 0.2081 tp3440 / line 0.0205 tp393
#     weld=24 perp=14              23.9684   netH 0.1807 tp2938 / line 0.0269 tp637
#     （同一对旋钮：10GT 上 +0.53，220 上 **−0.90**；10GT 只占 220 的 4.5%，
#       所以净亏全部来自那 210 张彩色图 —— 它们的走线渲染不吃「结点圆点断口」这套。）
#
# ⇒ **默认改成 12.0 / 1.5（§5.497）。** 16 是历史值：在 13 页主尺上它比 12 差
#   0.14（折叠）/ 0.13（判题机口径，case7 +1.82、case1 −0.15），220 池上差 0.20
#   （0072 +2.07、0001 +0.65、0165 +0.67 都已用单案复核，不是批次抖动）。
#   4/6/8/10/12 **逐位相同** ⇒ 这是一级台阶，不是某个具体值；取 12 离老默认最近。
#   想在 KiCad 风格图纸上更激进，显式设 `STAGE1_WIRE_WELD=24 STAGE1_WIRE_PERP=14`。
WIRE_WELD = float(os.environ.get("STAGE1_WIRE_WELD", "12.0"))
WIRE_PERP = float(os.environ.get("STAGE1_WIRE_PERP", "1.5"))
WIRE_COS = float(os.environ.get("STAGE1_WIRE_COS", "-0.75"))
# 规则 (3) 的**墨迹判据**：两端连线上的采样点里要有多少比例落在墨上才允许焊接。
# `STAGE1_WIRE_INK=0` 关掉（回到纯几何）。`STAGE1_WIRE_INK_FRAC` 调阈值。
WIRE_INK = env_on("STAGE1_WIRE_INK", "1")
WIRE_INK_FRAC = float(os.environ.get("STAGE1_WIRE_INK_FRAC", "0.5"))
# ★★ 规则 (1) 的**十字交叉拆分**（`STAGE1_CROSS_SPLIT`，默认开）。
#
# 规则 (1)「四段端点在 tol 内 ⇒ 并」会把**交叉**当成连接：一条竖直走线穿过
# 一条水平走线时，骨架在交叉处把两条线各切成两段，四个端点相距 1~2px，于是
# 四条段并进同一组 ⇒ **本来独立的两个网络被合成一条**。官方 GT 是分开标的：
# case 2 的 `net_5 = (R2.2, U1.3)`（竖直走线）与 `net_22 = (R1.2, U1.4)`
# （水平走线）在 (576,375) 处十字交叉，图上**没有圆点**，我们却并成了一坨
# `(R1.2, R2.2, U1.3, U1.4)` ⇒ 一个满分配对被稀释、另一个 GT 网直接变 fn。
#
# 判据（两条同时成立才拆）：
#   ① 结点处**有 >= 2 条不同的轴都「穿过去」** —— 同轴在本结点有 >= 2 个端点，
#      或者有线段从结点**中间穿过**（骨架偶尔对同一根导线给出两条重叠副本：一条
#      长段 + 若干被切开的短段，长副本的端点不在结点上，只按端点聚类看不见它 ——
#      case 2 的 `seg 49 [624.3,376]-[444,375]` 正是这样把竖直/水平两半并起来的）；
#   ② 该处**没有实心结点圆点**（`_has_dot`：半径 `STAGE1_CROSS_DOT_R` 的对角方向
#      都有墨 ⇒ 圆点 ⇒ 保持合并）。真结点（T 型、L 角、圆点四叉、单端支）不受影响。
#
# ★ 泛化（`STAGE1_CROSS_TAGS` 默认开 = 根级标签；`STAGE1_CROSS_WIDE` 默认**关**）：
# 判据从「这两段不许并」升级为「**同一个根不许同时持有同一结点的两条不同轴**」
# （`axis_of` 标签 + `_u` 里的根级冲突检查）—— 后者能拦住**经由第三段传递**的
# 跨轴合并（A 轴 ∪ 无关段 ∪ B 轴），那是逐对黑名单拦不住的。多岔（>=3 条轴）本来就是
# 两两判定，自动覆盖。实测（官方 10 张、no-vlm）：只用标签 **case 1 62.01→62.79
# (+0.78)**，其余 9 张逐位不变（case 5 也不动）；把聚簇半径也从 `tol` 放宽到
# `pass_tol` 反而 **case 5 −1.24**（把两个本来无关的近距结点并成一个簇 ⇒ 误判交叉），
# 所以 `WIDE` 默认关。
CROSS_SPLIT = env_on("STAGE1_CROSS_SPLIT", "1")
CROSS_COS = float(os.environ.get("STAGE1_CROSS_COS", "-0.85"))
CROSS_DOT_R = float(os.environ.get("STAGE1_CROSS_DOT_R", "3.0"))
CROSS_DOT_NEED = int(os.environ.get("STAGE1_CROSS_DOT_NEED", "3"))
CROSS_WIDE = env_on("STAGE1_CROSS_WIDE", "0")
# ★ 2026-11-04 曾翻默认 1 → 0（13 页 +0.0439，全在 case 1 的 +0.57）。但池侧单案复现
#   （`fast.py --cases 0058`）：关掉后 `nets` **19 → 11**、最大的网 **118 → 275** 个成员 ——
#   没有交叉标签，两条只是「画着交叉」的线被并成同一个网。池门禁四个开关合跑
#   29.9200 → 29.6782（−0.24），`0058` 一页就占 −4.39。⇒ 已回退到 1。
CROSS_TAGS = env_on("STAGE1_CROSS_TAGS", "1")
CROSS_WIDE_MAX = int(os.environ.get("STAGE1_CROSS_WIDE_MAX", "96"))
# ★ `STAGE1_CROSS_GAP`（默认 0 = 关）：**起簇**半径。骨架在无圆点的十字交叉处给出的
# 端点常常差 **2~4px**（> `WIRE_TOL=1.5`、也 > `CROSS_WIDE` 摊到的 2.25），只按 `ee`
# 端点对聚类就永远看不见这个结点 ⇒ 跨轴合并不被拦。实测 case 4 的 (979,320)：
# 竖直两臂端点差 4.0px、水平长副本从结点中间穿过 ⇒ 规则 (2) 把两轴并成了一个 group 13。
#
# **实测（官方 10 张、no-vlm、`work\_xgap_ab.ps1`）⇒ 净负，默认 0 保持关：**
#
# | 臂 | 总分 | netH | netL | case 1 | case 4 | case 5 | case 6 | case 8 |
# |---|---|---|---|---|---|---|---|---|
# | ref（关） | **80.88** | 0.7749 | 0.4975 | 81.46 | 79.39 | 77.70 | 77.14 | 68.83 |
# | `=3.0` | 80.56 | 0.7663 | 0.4956 | 81.52 | **80.02** | 76.48 | 76.33 | 66.94 |
# | `=4.5` | 80.28 | 0.7574 | 0.4989 | 79.06 | 79.71 | 76.48 | 76.33 | 66.94 |
#
# 唯一赚到的是 case 4（组 66→68：横轴 {seg16, seg70} 与竖轴 {seg52, seg54}+U3.2 引线
# {seg53, seg90} 拆开 ⇒ netL 0.3607→0.4098、netH 0.7412→0.7452 = **+0.63 案内分**）；
# 亏的是四类**误拆**（`work\_xgapbox.py` 逐结点取证 + 拼图
# `work/preview/_cropat/montage_nodes.png`）：
# ① case 5 (1772,499) 图上**有红点**、`_has_dot` 漏判（−1.22）；
# ② case 6 (1736,919) / case 8 (706,449) 是**导线接到引脚引线**的同轴结点，
#    簇内段端到引脚 **0.0px**（各 −0.81 / −1.89）；
# ③ case 1 的十个新结点是 GND / 电容符号笔画（gap 4.5 时 −2.40，gap 3.0 时 +0.06 无害）。
# 加「簇内段端→引脚 ≤2px 就不拆」+「多点圆点判据」两道闸后只剩 case 4 的 +0.63 案内分
# ⇒ 全局 **+0.07 分**（交付分 = 逐案分平均），在 ±0.05 噪声带内，且要给 `wire_groups`
# 新增引脚点管道 ⇒ **不实现**（与 §5.80 一致）。
# §5.338：上面的判负是**旧尺（20px）+ 旧管线**下、半径 3.0/4.5 的读数。提交尺
# `protocol4`（pin/line 5px）重扫：4→82.30 / 6→82.31 / 8→82.85 / **10→83.47**
# / 12→82.80 / 14→82.96 / 20→81.15 / 30→77.57，默认 0 是 81.63（GT10 十张）；
# 池子 50 案 36.41 → 36.61（28 上 22 下，最差单案 −4.9）。逐案 8/10 张 netH 上升，
# 不是单案假象 ⇒ 默认翻成 10。
CROSS_GAP = float(os.environ.get("STAGE1_CROSS_GAP", "10") or 0.0)
# 诊断计数（`wire_detail` 之外用不到；探针读它看「这次拆了几处交叉」）
LAST_X = {"n_cross": 0, "n_blocked": 0}
# ★★ 规则 (5)「端点微越过目标段」的容差（`STAGE1_WIRE_OVER`，0 = 关）。
#
# **为什么需要它、以及它为什么比抬 `WIRE_TOL` 干净。** `WIRE_TOL=10` 在官方 9 张上
# 净 +0.82，但那是拿「过度并网」换来的：`_wirenew.py` 把 case 10 从 tol 1.5 抬到 10
# 时**新并上的 35 对段**打出来，绝大多数的几何长这样 ——
#
#     seg216 [70,751]->[71,758]   seg207 [77,751]->[78,759]   6.8px  平行
#     seg202 [1507,708]->[1508,717]  seg206 [1512,708]->[1512,717]  4.0px  平行
#
# 也就是**电容（和二极管）符号的两块极板**：两条短竖线、相隔 5~7px、互相平行。
# 规则 (2) 只看「端点离线段多近」，于是把它们并起来 —— 等价于**把电容短路**，
# 整个电路的网在那一处被焊成一坨。case 10 的 11 脚巨块（横跨 GT 的
# `net_2` SW 节点 + `net_5` VCC 轨 + `net_8`/`net_13` 两条 GND）就是这么来的。
#
# 而真正想要的并（case 1 的电容引线比 GND 母线低 5px）长这样：
# 引线 `x=194, y 172→238`（竖）、母线 `y=233, x 71→723`（横）—— **引线跨过了母线
# 所在的直线**，只是端点多画了 5px。⇒ 判据是**穿越**，不是**距离**：
#   (a) 段 A 的两个端点在段 B 的支撑直线上**异侧**（A 真的跨过了 B 那条线）；
#   (b) 较近的那个端点到 B 的**线段**距离 <= `WIRE_OVER`；
#   (c) 垂足落在 B 的内部（留 3% 边距）。
# 平行段天然不满足 (a)，所以电容极板永远不会被它并起来。
# `WIRE_TOL=10` 就是这个洞的「钝补丁」：它同时吃掉了 (5) 想要的和不想吃的。
WIRE_OVER = float(os.environ.get("STAGE1_WIRE_OVER", "12"))
# 规则 (5) 是否**要求穿越**。1 = 只吃「端点跨过目标段所在的直线」；
# 0（**现默认**）= 放宽成「不平行即可」（同时吃「欠越」——端点停在目标段前几像素）。
# 两种模式都排除**平行**的段对，所以电容两块极板永远进不来。
# 官方 9 张：CROSS=1 最好 55.38（netH 0.5064）< CROSS=0 的 56.31（netH 0.5339）
# ⇒ 真实图纸上「引线停在母线前几像素」和「越过母线几像素」一样常见，两者都要吃。
WIRE_OVER_CROSS = env_on("STAGE1_WIRE_OVER_CROSS", "0")
WIRE_OVER_COS = float(os.environ.get("STAGE1_WIRE_OVER_COS", "0.9"))


class _UF(object):
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _as_segs(segments):
    """接受两种写法：扁平 `[x1,y1,x2,y2]` 与嵌套 `[[x1,y1],[x2,y2]]`。

    **必须两种都收**：本项目已经在「形状」上栽过两次 —— `stage1_wires_eval.seg_arrays`
    只吃嵌套形，喂扁平形会**静默**返回 `zeros((0,4))`（于是 FN 全是 0）；反方向也是。
    凡是段集合的入口，一律在这里统一，别再让调用方猜。
    """
    arr = np.asarray(segments, np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), np.float64)
    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr.reshape(-1, 4)
    out = []
    for s in segments:
        a, b = s
        out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def _dist_pt_segs(pts, segs):
    """pts (M,2) 到 segs (N,4) 的距离矩阵 (M,N)。"""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    s = np.asarray(segs, np.float64).reshape(-1, 4)
    if not len(p) or not len(s):
        return np.zeros((len(p), len(s)), np.float64)
    x1, y1, x2, y2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L2 = np.where(L2 < 1e-12, 1e-12, L2)
    t = ((p[:, 0][:, None] - x1[None, :]) * dx[None, :]
         + (p[:, 1][:, None] - y1[None, :]) * dy[None, :]) / L2[None, :]
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(p[:, 0][:, None] - (x1[None, :] + t * dx[None, :]),
                    p[:, 1][:, None] - (y1[None, :] + t * dy[None, :]))


def _overshoot_unions(segs, ends, own, over, edge=0.03, cross_only=True, cos_par=0.9):
    """规则 (5)：**端点靠近目标段、且两段不平行**时的并（T 型结点差几像素）。

    `cross_only=True`（默认）时判据见 `WIRE_OVER` 上面那段长注释 —— 核心是**穿越**
    （段 A 的两个端点在段 B 的支撑直线上异侧）。这样电容/二极管符号的两块平行极板
    （相隔 5~7px、平行、不穿越）永远进不来。

    `cross_only=False` 时改为只要求**不平行**（`|cos| < cos_par`）+ 垂足落在 B 内部。
    放宽的理由：真实图纸上引线既可能**越过**母线几像素（穿越），也可能**停在**母线前
    几像素（欠越）—— 后者不满足「穿越」，`cross_only=True` 会漏掉它，这正是
    `ov12t15` 的 netH 0.5064 打不过钝补丁 `WIRE_TOL=10` 的 0.5190 的原因之一。
    平行极板在两种模式下都被排除（`|cos|` 接近 1）。

    返回 `[(i, j), ...]`（段下标对，无序不保证）。
    """
    if over <= 0.0:
        return []
    segs = np.asarray(segs, np.float64).reshape(-1, 4)
    n = len(segs)
    if n < 2:
        return []
    p0 = segs[:, 0:2]
    dv = segs[:, 2:4] - p0
    L = np.hypot(dv[:, 0], dv[:, 1])
    L = np.where(L < 1e-9, 1e-9, L)
    u = dv / L[:, None]
    A0, A1 = segs[:, 0:2], segs[:, 2:4]
    D = _dist_pt_segs(ends, segs)
    out = []
    for i in range(n):
        # side_j(q) = dv_j × (q - p0_j)，即 q 到 j 的支撑直线的有向叉积。
        s0 = dv[:, 0] * (A0[i, 1] - p0[:, 1]) - dv[:, 1] * (A0[i, 0] - p0[:, 0])
        s1 = dv[:, 0] * (A1[i, 1] - p0[:, 1]) - dv[:, 1] * (A1[i, 0] - p0[:, 0])
        if cross_only:
            cand = np.nonzero((s0 * s1) < 0.0)[0]
        else:
            par = np.abs(u @ u[i]) >= cos_par
            cand = np.nonzero(~par)[0]
        if not len(cand):
            continue
        near = np.minimum(D[2 * i, cand], D[2 * i + 1, cand])
        for k in np.nonzero(near <= over)[0]:
            j = int(cand[k])
            ab, L2 = dv[j], max(1e-9, float(dv[j] @ dv[j]))
            for e in (A0[i], A1[i]):
                t = float((e - p0[j]) @ ab / L2)
                if edge <= t <= 1.0 - edge:
                    out.append((i, j))
                    break
    return out


def _ink_along(ends, A, B, ink, n=9, rad=1):
    """沿 A->B 的直线采样 `n` 点，返回每对 `(A,B)` 上「附近有墨」的比例。

    ★ 这是 `_weld_unions` 的**物理判据**，用来替代一味收紧 `perp_tol`：
    结点圆点造成的断口是**画出来的**（圆点本身就是墨），两端之间的直线上必然有墨；
    而「把不相干的两条线凑近」的假配对，连线多半穿过空白。
    坐标是**图像坐标（y 向下）**，所以直接 `ink[y, x]` 索引。
    """
    if ink is None:
        return None
    H, W = ink.shape[:2]
    ts = np.linspace(0.0, 1.0, int(n))
    p0 = np.asarray(ends, np.float64)[A][:, None, :]
    p1 = np.asarray(ends, np.float64)[B][:, None, :]
    pts = p0 + (p1 - p0) * ts[None, :, None]                 # (m,n,2)
    xi = np.round(pts[..., 0]).astype(np.int64)
    yi = np.round(pts[..., 1]).astype(np.int64)
    hit = np.zeros(xi.shape, bool)
    r = int(rad)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xx = np.clip(xi + dx, 0, W - 1)
            yy = np.clip(yi + dy, 0, H - 1)
            hit |= (ink[yy, xx] != 0)
    return hit.mean(axis=1)


def _weld_unions(segs, ends, own, weld, perp_tol, cos_tol, ink=None, ink_frac=0.5):
    """方向性桥接：把「相距 `weld` 以内、且方向对着、且沿着彼此的延长线」的两个
    段端点接起来。

    为什么不能只用距离门槛：`stage1_conn_eval` 实测把连通分量容差从 1px 放到 16px，
    连通 R 从 0.157 涨到 0.534（几何天花板 0.658），**但精度从 0.566 塌到 0.121**
    —— 距离门槛把不相干的网络也并到了一起。真正的走线断口是**沿直线方向**的，
    所以再加两个条件：两端点的外法线要相对（`dot <= cos_tol`），且位移要落在
    这条外法线上（垂直偏移 `perp <= perp_tol`）。

    `ink` 非空时再加一道**墨迹判据**：两端之间的直线上必须有墨（见 `_ink_along`）。
    这道判据是「放宽 `weld`/`perp`」能否只赚不亏的关键。

    `out[m]` = 该端点「离开自己那条段」的单位方向。`own[m]` 是端点 m 所属段的编号，
    m 的奇偶决定它是 p0 端（偶）还是 p1 端（奇）。
    """
    out = np.zeros((len(ends), 2), np.float64)
    i0 = own * 2
    d = segs[:, 2:4] - segs[:, 0:2]
    L = np.hypot(d[:, 0], d[:, 1])
    L = np.where(L < 1e-9, 1e-9, L)
    u = d / L[:, None]
    out[0::2] = -u          # p0 端：离开是 -u
    out[1::2] = u           # p1 端：离开是 +u
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ends)
        pairs = tree.query_pairs(float(weld), output_type='ndarray')
    except Exception:
        pairs = None
    if pairs is None or not len(pairs):
        return []
    A, B = pairs[:, 0], pairs[:, 1]
    ok = own[A] != own[B]
    A, B = A[ok], B[ok]
    if not len(A):
        return []
    v = ends[B] - ends[A]
    dist = np.hypot(v[:, 0], v[:, 1])
    live = dist > 1e-9
    A, B, v, dist = A[live], B[live], v[live], dist[live]
    if not len(A):
        return []
    vh = v / dist[:, None]
    cross = np.abs(out[A, 0] * vh[:, 1] - out[A, 1] * vh[:, 0])
    perp = cross * dist                       # 到外法线延长线的垂直距离
    dot = (out[A] * out[B]).sum(axis=1)
    keep = (perp <= perp_tol) & (dot <= cos_tol)
    if ink is not None:
        frac = _ink_along(ends, A, B, ink)
        if frac is not None:
            keep &= (frac >= float(ink_frac))
    return list(zip(own[A][keep].tolist(), own[B][keep].tolist()))


def _extend_unions(segs, ends, own, out, ext_len, ext_perp, ext_cos=1.0,
                   nearest_only=True, chunk=256):
    """规则 4：端点沿自己的外法线**朝前**延伸 `ext_len`，打到别的段的**身上**
    （垂直偏移 <= `ext_perp`）就接上。

    与规则 (3) 的区别：规则 (3) 只接**端点对端点**；实测里最常见的断法是「竖直的
    引脚走线停在水平总线**上方若干像素**」—— 它对上的是另一条段的**身上**。
    与规则 (2) 的区别：规则 (2) 各向同性，tol 一放大就把不相干的网络也并上
    （实测精度 0.566 -> 0.121）；规则 (4) 只认**正前方**那条窄锥。

    `nearest_only=True` 时（默认）每个端点只接**正前方最近的那一条**段 —— 物理上
    走线延伸出去只会撞上第一样东西；允许一路撞过去会让一条长射线把沿途的东西
    全并起来（实测 ext=64 时连通 R 冲到 0.665、比 GT 自己的 0.658 还高，就是接多了）。
    `ext_cos < 1` 时要求被撞的段与射线**大致垂直**（T 型），用来排除「沿着平行/共线
    段擦过去」的那类。
    """
    N = len(segs)
    if not N or ext_len <= 0:
        return []
    p1, p2 = segs[:, 0:2], segs[:, 2:4]
    d = p2 - p1
    L = np.hypot(d[:, 0], d[:, 1])
    L = np.where(L < 1e-9, 1e-9, L)
    uj = d / L[:, None]                                    # (N,2) 目标段方向
    pairs = []
    for a in range(0, len(ends), chunk):
        b = min(a + chunk, len(ends))
        pe = ends[a:b][:, None, :]
        o = out[a:b][:, None, :]
        v1 = p1[None, :, :] - pe
        v2 = p2[None, :, :] - pe
        t1 = (v1 * o).sum(-1)
        t2 = (v2 * o).sum(-1)
        d1 = o[..., 0] * v1[..., 1] - o[..., 1] * v1[..., 0]
        d2 = o[..., 0] * v2[..., 1] - o[..., 1] * v2[..., 0]
        a1, a2 = np.abs(d1), np.abs(d2)
        den = a1 + a2
        tstar = (t1 * a2 + t2 * a1) / np.where(den < 1e-9, 1.0, den)
        # **`den > eps` 这个条件是必须的**：目标段与本段共线时 d1 = d2 = 0，
        # `d1*d2 <= 0` 成立而 `tstar` 退化成 0，于是「正前方 16px 内的一切共线段」
        # 全被接上（合成自检 C4：三段相隔 200px 也塌成一个分量）。共线情形交给
        # 下面的端点判据处理就够了。
        cross = (d1 * d2 <= 0) & (den > 1e-6)
        tuse = np.where(cross, tstar, np.where(a1 <= a2, t1, t2))
        dmin = np.where(cross, 0.0, np.minimum(a1, a2))
        ok = (tuse >= 0.0) & (tuse <= ext_len) & (dmin <= ext_perp)
        # 交点落在身后但某个端点本身就在正前方窄锥里，也算
        ok |= (t1 >= 0.0) & (t1 <= ext_len) & (a1 <= ext_perp)
        ok |= (t2 >= 0.0) & (t2 <= ext_len) & (a2 <= ext_perp)
        if ext_cos < 1.0:
            dot = (o * uj[None, :, :]).sum(-1)
            ok &= np.abs(dot) <= ext_cos
        if nearest_only:
            # 必须**先**把自己那条段排除再取 argmin：自身端点到自身的距离恒为 0，
            # 否则 argmin 永远选中它、然后被 `own[m] != j` 丢掉，第二名永远没机会
            # （合成自检 C1 就是这么 FAIL 的：竖直走线明明停在总线上方 10px 也不接）。
            cols = np.arange(N)[None, :]
            ok = ok & (cols != own[a:b][:, None])
            tm = np.where(ok, np.maximum(tuse, 0.0), np.inf)
            jb = np.argmin(tm, axis=1)
            bv = tm[np.arange(len(jb)), jb]
            for u in np.nonzero(np.isfinite(bv))[0]:
                pairs.append((int(own[a + int(u)]), int(jb[u])))
        else:
            for (u, v) in np.argwhere(ok):
                m, j = a + int(u), int(v)
                if int(own[m]) != j:
                    pairs.append((int(own[m]), j))
    return pairs


def _has_dot(ink, p, r=None, need=None) -> bool:
    """结点圆点判据：真结点画了实心圆点，**对角**方向（45°/135°/225°/315°）也有墨。

    只看对角是关键 —— 十字交叉的四个**轴向**（与两条线同向）永远有墨，只有对角
    是空白；实心圆点（直径 9~10px）在对角一样是墨。case 2 实测（`STAGE1_CROSS_DOT_R=3`）：
    交叉点 (576,375) 对角 0/4、圆点 (575,355)/(493,375) 对角 4/4、普通走线 0/4。
    `ink` 为 None 时一律当圆点（保守：保持历史行为）。
    """
    if ink is None:
        return True
    r = CROSS_DOT_R if r is None else float(r)
    need = CROSS_DOT_NEED if need is None else int(need)
    h, w = ink.shape[:2]
    hit = 0
    for k in range(4):
        a = np.pi * (0.25 + 0.5 * k)
        x = int(round(float(p[0]) + r * float(np.cos(a))))
        y = int(round(float(p[1]) + r * float(np.sin(a))))
        if 0 <= x < w and 0 <= y < h and ink[y, x]:
            hit += 1
    return hit >= need


def _cross_forbid(segs, ends, own, ee_pairs, ink,
                  cos_par=None, r=None, need=None, pass_tol=None):
    """返回「十字交叉（无圆点）」处**不许并**的段对集合，以及交叉点个数。

    结点处只要有**两条不同的轴都「穿过去」**（同轴端点 >= 2 支，或有线段从结点
    中间穿过），且该处**没有实心结点圆点**（`_has_dot`），就判为交叉：
    只有**同轴**的段能并；两轴**都穿过**时，它们之间的任何合并都被禁止
    （规则 1~5 全适用，由 `wire_groups` 里的 `_u` 拦截）。
    真结点（T 型、L 角、圆点四叉、单端支）一律不动 —— 历史行为。

    ★ 为什么需要「穿过」这一支：骨架对同一根导线偶尔会给出**两条重叠副本**
    （一条长段 + 若干被切开的短段），长副本从结点中间穿过、端点并不落在结点上，
    只按端点聚类看不见它，跨轴合并就经由它发生 —— case 2 的 SDA/SCL 实测：
    `seg 49 [624.3,376]-[444,375]` 从 (576,375) 中间穿过，把竖直的
    `seg 4/seg 10`（R2 顶脚）与水平的 `seg 46/seg 9` 并成了同一个 group 19。
    """
    cos_par = CROSS_COS if cos_par is None else float(cos_par)
    pass_tol = (max(2.0, 1.5 * float(WIRE_TOL)) if pass_tol is None else float(pass_tol))
    axis_cos = abs(cos_par)
    par = {}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry

    for m, j in ee_pairs:
        par.setdefault(m, m)
        par.setdefault(j, j)
        union(m, j)
    # ★ `STAGE1_CROSS_GAP`：把 1.5~GAP px 的端点对也纳入起簇（见同名旋钮注释）。
    # 只影响「是否成簇」；是否拆分仍要求 ≥2 条轴都穿过去 + 无圆点，T 型/L 角不受影响。
    if CROSS_GAP > 0.0 and len(ends) <= 4000:
        D_gap = np.hypot(ends[:, 0][:, None] - ends[:, 0][None, :],
                         ends[:, 1][:, None] - ends[:, 1][None, :])
        for m in range(len(ends)):
            for j in np.nonzero(D_gap[m] <= CROSS_GAP)[0]:
                j = int(j)
                if int(own[m]) != int(own[j]):
                    par.setdefault(int(m), int(m))
                    par.setdefault(j, j)
                    union(int(m), j)
    # ★ 泛化（`STAGE1_CROSS_WIDE`）：聚簇半径从规则 (1) 的 `tol`(1.5) 放宽到 `pass_tol`(2.25)。
    # 骨架在交叉处给出的端点常常差 1.5~2.2px —— 它们**进不了 `ee`**，只按 `ee` 聚类就
    # 永远看不见这个结点（多岔、粗笔画、rdp 抖动都会这样）。放宽只影响「是否成簇」，
    # 是否拆分仍要求**≥2 条轴都穿过去**，所以 T 型（1 条轴穿过）不会被动到。
    # 只对已有簇的邻域做两轮闭包，并把簇规模封顶，避免密集区炸开。
    if CROSS_WIDE and pass_tol > float(WIRE_TOL) and par:
        near = set(int(m) for m in par)
        for _ in range(2):
            add = set()
            for m in sorted(near):
                p = ends[m]
                d = np.hypot(ends[:, 0] - p[0], ends[:, 1] - p[1])
                for k in np.nonzero(d <= pass_tol)[0]:
                    add.add(int(k))
            if add <= near:
                break
            near |= add
        if len(near) <= CROSS_WIDE_MAX:
            near_s = sorted(near)
            for x in range(len(near_s)):
                px = ends[near_s[x]]
                for y in range(x + 1, len(near_s)):
                    py = ends[near_s[y]]
                    if float(np.hypot(px[0] - py[0], px[1] - py[1])) <= pass_tol:
                        par.setdefault(near_s[x], near_s[x])
                        par.setdefault(near_s[y], near_s[y])
                        union(near_s[x], near_s[y])
    cl = {}
    for m in list(par):
        cl.setdefault(find(m), []).append(m)
    forbid, axis_of, n_cross = set(), {}, 0
    n_seg = len(segs)
    for mems in cl.values():
        group = sorted({int(own[m]) for m in mems})
        if len(group) < 2:
            continue
        node = ends[mems].mean(axis=0)
        # 「穿过」结点、端点却不在结点上的段（重叠副本）：点到线段距离 + 投影在内部
        for i in range(n_seg):
            if i in group:
                continue
            a = np.asarray(segs[i, :2], dtype=np.float64)
            b = np.asarray(segs[i, 2:4], dtype=np.float64)
            d = b - a
            L2 = float(d @ d)
            if L2 <= 1e-9:
                continue
            t = float((node - a) @ d) / L2
            if t <= 0.03 or t >= 0.97:
                continue
            if float(np.hypot(*(node - (a + t * d)))) <= pass_tol:
                group.append(int(i))
        group = sorted(set(group))
        if len(group) < 2:
            continue
        # 每支的**轴**（无向）+ 这一支是否「穿过」结点
        axes, is_pass, good = {}, {}, True
        for i in group:
            a = np.asarray(segs[i, :2], dtype=np.float64)
            b = np.asarray(segs[i, 2:4], dtype=np.float64)
            d = b - a
            L2 = float(d @ d)
            if L2 <= 1e-9:
                good = False
                break
            t = float((node - a) @ d) / L2
            if 0.03 < t < 0.97 and float(np.hypot(*(node - (a + t * d)))) <= pass_tol:
                v = d / float(np.sqrt(L2))
                is_pass[i] = True
            else:
                o = b if np.hypot(*(b - node)) >= np.hypot(*(a - node)) else a
                w = o - node
                Lw = float(np.hypot(w[0], w[1]))
                if Lw <= 1e-6:
                    good = False
                    break
                v = w / Lw
                is_pass[i] = False
            if v[0] < 0.0 or (v[0] == 0.0 and v[1] < 0.0):
                v = -v
            axes[i] = v
        if not good:
            continue
        # 按轴聚类（无向：|cos| 够大即同轴）
        sets, used = [], set()
        for i in group:
            if i in used:
                continue
            g = [j for j in group if abs(float(axes[i] @ axes[j])) >= axis_cos]
            used.update(g)
            sets.append(g)
        if len(sets) < 2:
            continue
        # 「穿过去」的轴：有穿过支，或同轴在本结点有 >= 2 个端点
        cont = [bool(any(is_pass[j] for j in g)) or sum(1 for j in g if not is_pass[j]) >= 2
                for g in sets]
        if sum(cont) < 2:
            continue
        if _has_dot(ink, node, r=r, need=need):
            continue
        n_cross += 1
        cid = n_cross
        for x in range(len(sets)):
            if cont[x]:
                for i in sets[x]:
                    axis_of.setdefault(int(i), []).append((cid, x))
        for x in range(len(sets)):
            for y in range(x + 1, len(sets)):
                if not (cont[x] and cont[y]):
                    continue
                for a in sets[x]:
                    for b in sets[y]:
                        forbid.add((a, b) if a < b else (b, a))
    return forbid, n_cross, axis_of


def _closest_on_seg(s, x, y):
    """线段 `s = [x1,y1,x2,y2]` 上离 `(x, y)` 最近的点（投影后夹到段内）。"""
    x1, y1, x2, y2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return x1, y1
    t = ((x - x1) * dx + (y - y1) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return x1 + t * dx, y1 + t * dy


def _text_bridge_unions(segs, ends, own, D, max_d, pad, bridges):
    """端点被「抹字」截断的桥接（规则 6）：两个线端各贴住**同一个**被擦除文本框。

    三重判据，缺一不可：
      (a) 端点-线段距离 > 常规容差（否则规则 2 已经并了）、但在 `max_d` 内；
      (b) **该端点**落在某个被擦除文本框（外扩 `pad`）里、或紧贴它；
      (c) 目标段上离该端点**最近的点**也落在**同一个**框（外扩 `pad`）里。
    (c) 是排他判据：只有「缺口横穿同一个文本框」才成立 —— 电容的两块平行极板、并排
    走线（文本框在旁边而不是在缺口上）都不成立。这正是抬全局容差做不到的区分。
    返回 `[(段下标 i, 段下标 j)]`；调用方一律经 `_u` 合并（仍受 forbid/轴冲突约束）。
    """
    box = np.asarray(bridges, np.float64).reshape(-1, 4)
    if not len(box):
        return []
    x1 = np.minimum(box[:, 0], box[:, 2]) - pad
    y1 = np.minimum(box[:, 1], box[:, 3]) - pad
    x2 = np.maximum(box[:, 0], box[:, 2]) + pad
    y2 = np.maximum(box[:, 1], box[:, 3]) + pad
    out = []
    for m in range(len(ends)):
        i = int(own[m])
        ex, ey = float(ends[m, 0]), float(ends[m, 1])
        hit = np.nonzero((ex >= x1) & (ex <= x2) & (ey >= y1) & (ey <= y2))[0]
        if not len(hit):
            continue
        row = D[m]
        for j in np.nonzero((row > 0.0) & (row <= float(max_d)))[0]:
            k = int(j)
            if k == i:
                continue
            cx, cy = _closest_on_seg(segs[k], ex, ey)
            for h in hit:
                if x1[h] <= cx <= x2[h] and y1[h] <= cy <= y2[h]:
                    out.append((i, k))
                    break
    return out


def wire_groups(segments, tol=GROUP_TOL, weld=0.0, perp_tol=1.5, cos_tol=-0.75,
                ext=0.0, ext_perp=2.0, ext_cos=1.0, ext_nearest=True,
                over=0.0, over_cross=None, over_cos=None, ink=None, ink_frac=0.5,
                bridges=None, bridge=None, bridge_pad=None):
    """把段并成连通分量，返回每条段的组号（组号 = 该组最小的段下标）。

    三种合并：
      (1) 端点-端点 <= tol（pskel 的 rdp/snap 让共点端点差 1~2px）；
      (2) 端点-线段 <= tol（T 型结点）；
      (3) `weld > 0` 时的**方向性桥接**（见 `_weld_unions`，修断口用）；
      (4) `ext > 0` 时的端点外法线延伸；
      (5) `over > 0` 时的**穿越式微越过**（见 `_overshoot_unions` —— 只吃「端点跨过
          目标段所在的直线、且只多画了几像素」，不吃电容极板那种平行近距）；
      (6) `bridge > 0` 时的**抹字断口定向桥接**（见 `_text_bridge_unions` —— 只吃
          「两个线端各贴住同一个被擦除文本框的两侧」，`bridges` 是该框列表）。
    `ink` 非空时给规则 (3) 加一道墨迹判据（见 `_ink_along`）。
    """
    segs = segments if isinstance(segments, np.ndarray) else _as_segs(segments)
    n = len(segs)
    if not n:
        return []
    uf = _UF(n)
    # 索引必须让第 i 段的端点是 ends[2i] / ends[2i+1] —— 用 stack 而不是 concatenate，
    # 否则每一段会和自己的**序号后继**并起来，整张图塌成一个分量（踩过）。
    ends = np.stack([segs[:, :2], segs[:, 2:]], axis=1).reshape(-1, 2)
    own = np.repeat(np.arange(n), 2)
    # (1) 端点-端点（`STAGE1_CROSS_SPLIT` 时把「无圆点的十字交叉」拆成两组）
    ee = []
    if n <= 4000:
        D_ee = np.hypot(ends[:, 0][:, None] - ends[:, 0][None, :],
                        ends[:, 1][:, None] - ends[:, 1][None, :])
        for m in range(len(ends)):
            row = D_ee[m]
            for j in np.nonzero(row <= tol)[0]:
                if own[m] != own[j]:
                    ee.append((int(m), int(j)))
    else:
        try:
            from scipy.spatial import cKDTree
            for a, b in cKDTree(ends).query_pairs(float(tol), output_type='ndarray'):
                if own[a] != own[b]:
                    ee.append((int(a), int(b)))
        except Exception:
            pass
    # ★ 十字交叉（无圆点）的**跨轴禁止**：命中的段对在规则 (1)~(5) 里一律不许并；
    # `axis_of` 给出「段 -> [(结点, 轴号)]」，`_u` 用它做**根级**冲突检查（拦传递合并）。
    forbid, n_cross, axis_of = set(), 0, {}
    if CROSS_SPLIT and ink is not None and (ee or CROSS_GAP > 0.0):
        forbid, n_cross, axis_of = _cross_forbid(segs, ends, own, ee, ink)
    n_blocked = [0]
    LAST_X["n_cross"], LAST_X["n_blocked"] = n_cross, 0
    root_tags = {int(i): dict(t) for i, t in axis_of.items()} if (axis_of and CROSS_TAGS) else {}

    def _u(a, b):
        a, b = int(a), int(b)
        if a == b:
            return
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return
        ta, tb = root_tags.get(ra), root_tags.get(rb)
        if ta and tb:
            for cid, ax in ta.items():
                if cid in tb and tb[cid] != ax:
                    n_blocked[0] += 1
                    return
        if a > b:
            a, b = b, a
        if forbid and (a, b) in forbid:
            n_blocked[0] += 1
            return
        uf.union(ra, rb)
        if ta or tb:
            r = uf.find(ra)
            m = dict(ta or {})
            for cid, ax in (tb or {}).items():
                m.setdefault(cid, ax)
            root_tags[r] = m
            if r != ra:
                root_tags.pop(ra, None)
            if r != rb:
                root_tags.pop(rb, None)

    for m, j in ee:
        _u(own[m], own[j])
    # (2) 端点-线段（T 型结点）
    D = _dist_pt_segs(ends, segs)
    for m in range(len(ends)):
        i = int(own[m])
        row = D[m].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        if float(row[j]) <= tol:
            _u(i, j)
    # (3) 方向性桥接
    if weld > 0.0:
        for a, b in _weld_unions(segs, ends, own, weld, perp_tol, cos_tol,
                                 ink=ink, ink_frac=ink_frac):
            _u(a, b)
    # (4) 端点沿外法线朝前延伸
    if ext > 0.0:
        out = np.zeros((len(ends), 2), np.float64)
        d = segs[:, 2:4] - segs[:, 0:2]
        L = np.hypot(d[:, 0], d[:, 1])
        L = np.where(L < 1e-9, 1e-9, L)
        u = d / L[:, None]
        out[0::2] = -u
        out[1::2] = u
        for a, b in _extend_unions(segs, ends, own, out, ext, ext_perp,
                                   ext_cos=ext_cos, nearest_only=ext_nearest):
            _u(a, b)
    # (5) 穿越式微越过（端点跨过目标段的支撑直线、只多画了几像素）
    if over > 0.0:
        if over_cross is None:
            over_cross = WIRE_OVER_CROSS
        if over_cos is None:
            over_cos = WIRE_OVER_COS
        for a, b in _overshoot_unions(segs, ends, own, over,
                                      cross_only=bool(over_cross), cos_par=over_cos):
            _u(a, b)
    # (6) 抹字断口定向桥接（只认被擦除的文本框，绝不用元件框）
    _br = TEXT_BRIDGE if bridge is None else float(bridge)
    if _br > 0.0 and bridges is not None:
        _bp = TEXT_BRIDGE_PAD if bridge_pad is None else float(bridge_pad)
        for a, b in _text_bridge_unions(segs, ends, own, D, _br, _bp, bridges):
            _u(a, b)
    LAST_X["n_blocked"] = n_blocked[0]
    roots = {}
    out = []
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = r
        out.append(r)
    return out


def free_ends(segments, groups, tol=GROUP_TOL):
    """返回 [(x, y, group)] —— 只挂在一条段上的端点（走线的自由端）。"""
    segs = segments if isinstance(segments, np.ndarray) else _as_segs(segments)
    if not len(segs):
        return []
    ends = np.stack([segs[:, :2], segs[:, 2:]], axis=1).reshape(-1, 2)
    own = np.repeat(np.arange(len(segs)), 2)
    D = _dist_pt_segs(ends, segs)
    out = []
    for m in range(len(ends)):
        i = int(own[m])
        row = D[m].copy()
        row[i] = np.inf
        # 该端点是否落在别的段上（含端点/中段）—— 落在就是结点，不是自由端
        if float(np.min(row)) <= tol:
            continue
        out.append((float(ends[m][0]), float(ends[m][1]), int(groups[i])))
    return out


def _sample_mask(segs, shape, boxes, dist):
    """把 `boxes` 按 `dist` 外扩画成布尔掩膜，并对每条段取 5 个采样点，
    返回「该段是否碰到任一框」的布尔数组。掩膜查询是 O(1)，否则
    「几千条射线 × 几百个框」要几百万次 Python 循环。"""
    H, W = shape
    m = np.zeros((max(1, int(H)), max(1, int(W))), dtype=bool)
    for b in boxes:
        x1 = max(0, int(np.floor(float(b[0]) - dist)))
        y1 = max(0, int(np.floor(float(b[1]) - dist)))
        x2 = min(W, int(np.ceil(float(b[2]) + dist)) + 1)
        y2 = min(H, int(np.ceil(float(b[3]) + dist)) + 1)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    ts = (0.1, 0.3, 0.5, 0.7, 0.9)
    hit = np.zeros(len(segs), dtype=bool)
    for t in ts:
        xs = np.rint(segs[:, 0] + t * (segs[:, 2] - segs[:, 0])).astype(np.int64)
        ys = np.rint(segs[:, 1] + t * (segs[:, 3] - segs[:, 1])).astype(np.int64)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        h = np.zeros(len(segs), dtype=bool)
        h[ok] = m[ys[ok], xs[ok]]
        hit |= h
    return hit


def _shape_of(segs, texts, pad=32):
    x = [float(segs[:, 0].max()), float(segs[:, 2].max())]
    y = [float(segs[:, 1].max()), float(segs[:, 3].max())]
    for b in texts:
        bb = b.get('bbox_img') if isinstance(b, dict) else b
        if bb:
            x.append(float(bb[2]))
            y.append(float(bb[3]))
    return int(max(y) + pad), int(max(x) + pad)


def label_candidates(segments, texts, groups=None, max_dist=LABEL_MAX_DIST,
                     boxes=None, anchor_dist=ANCHOR_DIST, require_anchored=True,
                     inside_pad=1.0, shape=None, min_group_len=0.0, tol=GROUP_TOL):
    """把文字框绑到导线自由端。

    `texts` 是 `stage1.detect` 产出的文字框列表（元素含 `bbox_img`；也接受
    裸的 `[x1,y1,x2,y2]`）。返回按文字框去重后的候选，`group` 取最近的自由端
    所属分量；同一文字框只保留最近的那个自由端。

    `boxes`（元件框，`bbox_img` 或裸框都行）用来做两道门控 —— 没有它们，
    「自由端」这个判据太松：加射线之后每张图有 ~422 个自由端，几乎任何文字框
    都能在 16px 内找到一个，候选数会膨胀到文字框总数的 63%：
      1. **锚定**（`require_anchored`）：一条走线必须一头接在元器件上，另一头才是
         网络标签。整条分量都没碰到任何元件框的，是射线的孤枝，不是走线。
      2. **排除元件内部的文字**（`inside_pad`）：位号 / 参数 / 引脚名都画在元件框
         里或紧贴框边，只有画在框外的文字才可能是网络标签。
    """
    segs = segments if isinstance(segments, np.ndarray) else _as_segs(segments)
    if not len(segs) or not len(texts):
        return []
    if groups is None:
        groups = wire_groups(segs, tol)
    if not groups:
        return []

    comp_boxes = []
    if boxes:
        for b in boxes:
            bb = b.get('bbox_img') if isinstance(b, dict) else b
            if bb:
                comp_boxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])

    anchored = None
    if require_anchored and comp_boxes:
        shp = shape or _shape_of(segs, texts)
        hit = _sample_mask(segs, shp, comp_boxes, anchor_dist)
        anchored = set(int(g) for g in np.asarray(groups)[hit])

    fe = free_ends(segs, groups, tol)
    if min_group_len > 0.0:
        # 分量总长太短的当作碎片（1~3px 的孤枝），它的端点不是真正的走线端
        glen = {}
        slen = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        for i, g in enumerate(groups):
            glen[int(g)] = glen.get(int(g), 0.0) + float(slen[i])
        fe = [t for t in fe if glen.get(t[2], 0.0) >= min_group_len]
    if anchored is not None:
        fe = [t for t in fe if t[2] in anchored]
    if not fe:
        return []

    boxes, ids, centers = [], [], []
    for t in texts:
        if isinstance(t, dict):
            b = t.get('bbox_img')
            tid = t.get('id')
        else:
            b, tid = t, None
        if not b:
            continue
        x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        if comp_boxes:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            inside = False
            for cb in comp_boxes:
                if (cb[0] - inside_pad <= cx <= cb[2] + inside_pad
                        and cb[1] - inside_pad <= cy <= cb[3] + inside_pad):
                    inside = True
                    break
            if inside:
                continue
        boxes.append((x1, y1, x2, y2))
        ids.append(tid)
        centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    if not centers:
        return []

    fe_xy = np.asarray([(a, b) for a, b, _ in fe], np.float64)
    fe_grp = [g for _, _, g in fe]
    C = np.asarray(centers, np.float64)
    grid = np.asarray(boxes, np.float64)
    # ★ 2026-07 审计修复：这里原来先算了一个 `(n_text, n_text)` 的点到框距离矩阵
    #   （`box_dist = np.hypot(dx, dy)`），紧接着就 `del box_dist` —— 纯粹白算。
    #   大图上有上千个文字框，这一步是 O(n_text²) 的内存与时间，且结果从没被用过
    #   （下面用的是「自由端 -> 文字框」的 `d_fb`）。整段删掉。

    # 自由端 -> 文字框 的距离
    d_fb = np.hypot(
        np.maximum(np.maximum(grid[None, :, 0] - fe_xy[:, 0][:, None], 0.0),
                   np.maximum(fe_xy[:, 0][:, None] - grid[None, :, 2], 0.0)),
        np.maximum(np.maximum(grid[None, :, 1] - fe_xy[:, 1][:, None], 0.0),
                   np.maximum(fe_xy[:, 1][:, None] - grid[None, :, 3], 0.0)))   # (n_free, n_text)

    best = {}
    for fi in range(len(fe)):
        for ti in np.nonzero(d_fb[fi] <= max_dist)[0]:
            ti = int(ti)
            d = float(d_fb[fi][ti])
            cur = best.get(ti)
            if cur is None or d < cur['dist']:
                best[ti] = {'text_id': ids[ti], 'bbox_img': list(boxes[ti]),
                            'center': list(centers[ti]), 'group': int(fe_grp[fi]),
                            'free_end': [float(fe_xy[fi][0]), float(fe_xy[fi][1])],
                            'dist': d, 'n_free_ends': 0}
    for ti, rec in best.items():
        rec['n_free_ends'] = int(np.sum(d_fb[:, ti] <= max_dist))
    out = sorted(best.values(), key=lambda r: r['dist'])
    return out


def merge_by_label(groups, named_labels):
    """逻辑合并的落点：把「标签文本相同」的导线分量合成同一个网络名。

    `groups` 是 `wire_groups` 的输出（每条段的组号）；
    `named_labels` 是 {组号: 标签文本}（文本由第二阶段识别后填进来）。
    返回 {组号: 网络名}，没有标签的组保留 `None`。
    """
    name_of = {}
    for g, lab in named_labels.items():
        if lab:
            name_of[int(g)] = lab
    return {int(g): name_of.get(int(g)) for g in set(groups)}


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=r'Y:\workspace\EDA\work\stage1_all13')
    ap.add_argument('--cases', type=int, default=20)
    a = ap.parse_args()
    import json
    files = sorted(f for f in os.listdir(a.stage1) if f.endswith('.stage1.json'))
    files = files[::max(1, len(files) // a.cases)][:a.cases]
    t0 = time.time()
    n_fe = n_raw = n_lab = n_txt = n_seg = 0
    per = []
    for f in files:
        d = json.load(open(os.path.join(a.stage1, f), encoding='utf-8'))
        segs = _as_segs(d['wire_detail']['segments'])
        texts = d.get('texts') or []
        comps = d.get('components') or []
        if not len(segs):
            continue
        g = wire_groups(segs)
        fe = free_ends(segs, g)
        raw = label_candidates(segs, texts, g, require_anchored=False, boxes=None)
        labs = label_candidates(segs, texts, g, boxes=comps)
        n_seg += len(segs)
        n_fe += len(fe)
        n_txt += len(texts)
        n_raw += len(raw)
        n_lab += len(labs)
        per.append((f.replace('.stage1.json', ''), len(segs), len(texts), len(fe),
                    len(raw), len(labs)))
    dt = time.time() - t0
    print(f'cases {len(files)}  段 {n_seg}  文字框 {n_txt}  自由端 {n_fe}')
    print(f'每张：段 {n_seg / len(files):.0f}  文字框 {n_txt / len(files):.0f}  '
          f'自由端 {n_fe / len(files):.0f}')
    print(f'候选（无门控）{n_raw} = 每张 {n_raw / len(files):.1f}（文字框的 {n_raw / n_txt * 100:.1f}%）')
    print(f'候选（锚定+排除元件内文字）{n_lab} = 每张 {n_lab / len(files):.1f}'
          f'（文字框的 {n_lab / n_txt * 100:.1f}%）')
    print(f'耗时 {dt:.1f}s = {dt / len(files) * 1000:.0f} ms/张')
    print('\n逐张：case  段  文字框  自由端  无门控  门控后')
    for r in per:
        print('  {:<28} {:>5} {:>6} {:>6} {:>7} {:>6}'.format(*r))


if __name__ == '__main__':
    _main()
