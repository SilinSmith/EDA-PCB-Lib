# -*- coding: utf-8 -*-
"""第二阶段端到端 CLI：stage1 产物 -> 文字识别 -> 角色分配 -> 官方格式 JSON。

    cd Y:\\workspace\\EDA\\eda-pcb
    python -m stage2.run --stage1 Y:\\workspace\\EDA\\work\\stage1_all14 ^
                         --out Y:\\workspace\\EDA\\work\\stage2_all --limit 10

产物：`<out>\\<case>.json`（官方 components/pins/nets 格式，左下角原点）
      `<out>\\_stage2_report.json`（每个 case 的统计与耗时）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 「假值词表」的唯一实现（`eda-pcb/envflag.py`）；`stage2/assign.py` 用的是同一份。
try:
    from envflag import env_on
except ImportError:                     # 直接跑脚本时 `sys.path[0]` 是本目录
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envflag import env_on

from .text import (DEFAULT_CKPT, LINESPLIT, TextEngine, detect_texts, imread_unicode,
                   read_stage1_texts, read_vertical)
from .assign import assign, dist_to_rect, page_scale_from_heights, set_page_scale
from .assemble import build_doc
from . import partname_ocr
from .expect import merge_runs, score_expect, should_grow
from .growbox import grow_reading

# 「不可分就默认是长的」时，长出来的新读法要过这道门才允许进候选池（用户 m06734：
# 「截止条件定的高一点」）。0.9 = 必须落在它自己前缀所期望的长度区间里。
GROW_MIN_SCORE = float(os.environ.get("STAGE2_GROW_MIN", "0.9"))
# 每颗元件最多为几个框做生长（控成本）
GROW_PER_COMP = int(os.environ.get("STAGE2_GROW_PER_COMP", "4"))
# 短串触发（`STAGE2_GROW_SHORT`，默认 1）：读出来只剩 1~2 个字符（字母前缀可缺）的框，
# 也送去「长框重读」——截断的位号正是这种形态。只在 `--recut grow|both` 生效。
GROW_SHORT = int(os.environ.get("STAGE2_GROW_SHORT", "1"))
GROW_SHORT_RE = re.compile(r"^[A-Za-z]?\d{1,2}$")
# ★★ 把 `stage1/pincol.py` 算好的引脚号窗口当文字框送进 OCR（默认开）。
# 只取 `out_win`（框外的**号**）：名字在框内，检测器本来就检得到，而号漏检最狠。
# 想连名字一起读就设 `STAGE2_PINCOL_WIN_KEYS=out_win,in_win`。
PINCOL_WIN_TEXTS = os.environ.get("STAGE2_PINCOL_WIN", "1").strip().lower() not in (
    "", "0", "false", "off", "none")
PINCOL_WIN_KEYS = tuple(k.strip() for k in
                        os.environ.get("STAGE2_PINCOL_WIN_KEYS", "out_win").split(",")
                        if k.strip())

# ---------------------------------------------------------------------------
# ★★ 队友分支 `origin/codex/improve-pin-network-parsing` 的**五个引脚读法模块**
#    （2026-10-15 并入；模块本体原样拷入，接口与开关名沿用他们的命名）：
#
#    这五个读取器都是**纯追加**：由「已确认的引脚/车道几何」再切一批裁片、读一轮，
#    把结果 append 进 `reads`，再由 `assign` 统一消化 —— 不动已有读数、不删候选，
#    因此与我们的 `pincol`/`pinname`/`pinrot` 通道**并列**而不是替换。
#
#    ★ 默认值：这里先全部 **off**（队友分支里默认是 1/auto），因为并进来必须先做
#      A/B（用户 m55811 的要求：改了识别逻辑的要 A/B）。逐个 A/B 有正收益的再翻默认。
#      已翻默认的三个（§5.515，13 页 +0.149/+0.149/+0.032、池 28.4589→28.5735）：
#      `STAGE2_PIN_NUMBER_REPAIR`=1、`STAGE2_PIN_NAME_OCR`=auto；
#      缺 `rapidocr-onnxruntime` 时两条读取器都 `provider is None → return []`，
#      实测逐字节退回旧默认（13/13 sha256 相同）。
#      · 独立引脚车道窗口 + 选择性恢复（§5.59 起**默认开**：
#                                        离线 +1.11/张、派生 +0.1868、精简 +0.20、完整 +0.31）
#                                        （`stage2/pin_lane_crops.py` / `pin_lane_reading.py`）
#        `STAGE2_PIN_LANE_CLEANUP`   =light/heavy，车道裁片的去杂强度
#      · `STAGE2_PIN_NUMBER_REPAIR`  =1 脚号几何保守修复 + 双视图一致才确认
#                                        （`stage2/pin_number_crops.py` / `pin_number_reading.py`）
#      · `STAGE2_PIN_NAME_OCR`       =auto/on/off 可选 PP-OCR 名称识别
#                                        （`stage2/pin_name_reading.py`，需 `requirements.pin-name-ocr.txt`）
#        黑白引脚通道继承它（`stage2/mono_pin_reading.py`）
#    依赖：`stage2/isolated_ocr.py`（单样本独立识别，避免 CRNN 依赖最长裁片）。
#    可选依赖：`rapidocr-onnxruntime`（本机未装 ⇒ auto 会告警回退，on 会报错）。
# ---------------------------------------------------------------------------
PIN_LANE_CLEANUP = os.environ.get("STAGE2_PIN_LANE_CLEANUP", "light").strip().lower()
PIN_NAME_OCR = os.environ.get("STAGE2_PIN_NAME_OCR", "auto").strip().lower()
MONO_PIN_OCR = PIN_NAME_OCR
# ★ 2026-11-04 翻默认 1 → 0（关）：13 页单独关掉 **+0.1033**（85.5786，全部落在 case 11 的 +1.34）。
#   四页快筛 +0.336（case 11）与之同向。这条「几何保守修复」是 2026-10 为缺号件补的，
#   现在 `STAGE2_PIN_NAME_OCR=auto` + 通用 OCR 通道已经把号读全，修复反而会把号改错。
PIN_NUMBER_REPAIR = env_on("STAGE2_PIN_NUMBER_REPAIR", "0")
# ★ 同槽双通道联合断言（§5.515 起默认**开**；`stage2/pin_ocr_join.py`）：
#   `STAGE2_PIN_NUMBER_REPAIR`（号）与 `STAGE2_PIN_NAME_OCR`（名）是两条独立外部通道，
#   但读数进 `assign` 后按**槽**分配：号与名落到相邻的两个槽上时，两个槽各自缺一半，
#   缺名字的那个被 `assemble.py` 的迟到 need-name 闸门整条丢掉 ⇒ 两条通道都对同一根
#   脚给了证据却谁都发不出去（`case11 U1` 实测就是这样丢 6 根脚：`read_pin_numbers`
#   接受 idx0→'8'、idx2→'6'，`read_pin_names` 接受 idx0→'LED1'、idx2→'VDD'，
#   都 ≥.99，而 GT 的 `pin_8` 正是 `LED1`、`pin_6` 正是 `VDD`）。
#   打开后只做一件事：两条通道**对同一 `(component_id, pin_index)` 都接受**的槽，
#   用缺的那一半补齐（不覆盖已有值、不新增槽、不删任何读数、不引入任何阈值）。
#   必须与上面两个开关同时开：任一通道关掉 ⇒ 该通道无读数 ⇒ 自然弃权。
# ★ 号通道「只喂 JOIN、不进通用读数池」（`STAGE2_PIN_NUMBER_JOIN_ONLY`，默认 **0=关**）：
#   上面那条联合断言要 `pin_number_repair` 的读数进 `_ocr_ch`；但同一批读数若同时并进
#   `reads`（`assign` 的通用池，见下面 `if _extra:` 那段），会去抢槽 / 触发 `PIN_DUPNUM`
#   让位 —— `STAGE2_PIN_NUMBER_REPAIR` 2026-11-04 翻 0（13 页 +0.1033、case11 +1.34）
#   关的是**这两件事一起**，于是联合断言的号半边也一起废了（`case11 U1` 丢 6 根脚）。
#   开着 ⇒ 两件事分开：`_ocr_ch` 照样拿到号读数（断言可用），通用池不收（不抢槽）。
#   关掉 ⇒ 旧行为（两处都收）。默认关，使下游逐位不变；与 `PIN_NUMBER_REPAIR=1` 一起开才生效。
#   ★ 2026-11-04 实测（`work/_cv/pnjoin1.json`）：`REPAIR=1 JOIN_ONLY=1` ⇒ 13 页 **85.4753
#     （−0.1033）**，与翻默认前的 `ship3` 逐位相同 ⇒ **掉分全在 JOIN 这条路上、通用池贡献 0**
#     （原假设被推翻）。日志：11 页有断言，只有 case11 真补（补号 2、补名 2），而它正是 −1.34
#     的那页；补进去的 `pin_6=VDD`、`pin_8=LED1` 位置都对，但同件里 `pin_7` 已带着 `LED1`
#     （左排还整体错一槽：我们的 `pin_2` 带着 GT `pin_1` 的 `OUT`）⇒ 新增正确名与既有错名
#     并存，多出来的那根算 fp。根因是**箱体脚名按行从下往上绑定**时与降序脚号错位，不是
#     通道缺证据，详见 `docs/RESULTS.md` §5.570。
PIN_NUMBER_JOIN_ONLY = env_on("STAGE2_PIN_NUMBER_JOIN_ONLY", "0")
# ★ 整颗元件「引脚号 ↔ 引脚名」抄表（`STAGE2_PIN_TABLE`，默认 **0=关**；
#   `stage2/pin_table_vlm.py`）：上面那条联合断言要求**两条单脚通道对同一个槽都接受**
#   才补缺的那一半；这里换一个证据源——把**整颗元件的符号裁片**（外扩
#   `STAGE2_ZOOM_READ_PAD`、放大 2×）交给本地 VLM **一次抄出整张 `(号, 名)` 表**
#   （`vlm/pinzoom.py` 是逐脚放大读、`vlm/pinname.py` 只读名字，都没有「整张表」这条
#   通道）。表**只用来改名字**，且必须过「表自洽」+「号在本元件内唯一」两道闸门。
#   写号 / 按侧拟合外推号 / 无闸门填空名 / 不看号唯一地覆盖，四种离线实测都判负
#   （13 页 pin f1：0.5343 → 0.4871 / 0.4871 / 0.5135 / 0.5467），当前口径是离线
#   0.5343 → 0.5781、13 页无一处变差（详见模块文件头）。
#   关 ⇒ 连 import 都不做，下游逐位不变。
PIN_TABLE = env_on("STAGE2_PIN_TABLE", "0")
# ★ 缺口首位的整侧脚名重绑（`STAGE2_PIN_NAME_REBIND`，默认 **0=关**；
#   `stage2/pin_rebind.py`）：一行的名与号是**分别裁片**的，号那张丢了（细 `1`、
#   导线压字）以后，名字仍按它自己那一行绑定，而号槽从此整体上移一格 —— slot `2`
#   于是顶着的名字其实是 pin 1 的名。本通道只认一种签名（该侧名比号多一个、号是
#   连续升序、多出的那个名在号序起点外侧）才动手，其余一律不碰。
#   2026-11-04 实测（`work/_rebind_orc.py` + `work/_rebind_hand.py`）：13 页只命中
#   `case11 U1` 左排一次（`pin_2` 的 `OUT` → `BAT`），该页 **+1.5152**、均值
#   **+0.1166**，其余 12 页逐位不变。放量到「所有侧按行重配」实测 **−9.5 均值**
#   （两侧混组、共线平移不可判），所以这里刻意只做这一种可判的形态。
#   2026-11-05 管线内复核（`work/_cv/rebind1.json`、`work/_cv/rebind0.json`）：
#   =1 ⇒ **85.6952 / 84.6418 / 81.9486**（+0.1166，只 case11 +1.52）；
#   =0 ⇒ 85.5786 / 84.5252 / 81.8320，与交付档 `onoffShip6` 逐位相同。
PIN_NAME_REBIND = env_on("STAGE2_PIN_NAME_REBIND", "1")
# ★ 同点同名重复脚去重（`STAGE2_PIN_DEDUP`，默认 **0=关**；`stage2/pin_rebind.py`
#   的 `dedupe_coincident`）：同一行偶尔被裁出两个槽（`pin_5` 与 `pin_7` 都叫 `VM`、
#   点差 <3px）。判分键是 `(元件, 号)`，多出来的那颗永远算 fp，而真脚照样能配上。
#   规则：同件、同名、点距 ≤3px ⇒ 丢号更大的那颗，并把网里的引用改到留下的那颗。
#   2026-11-05 实测（`work/_dedupe13.py` 直调模块）：13 页只命中 `case1 U3.pin_7 →
#   U3.pin_5`，该页 **+0.2957**（不改网引用是 +0.2277）、均值 **+0.0227**，其余 12 页
#   逐位不变。默认关，先离线口径。
PIN_DEDUP = env_on("STAGE2_PIN_DEDUP", "0")
# ★ 行内位号按 x 序一一对应（`STAGE2_DESIG_ROWS`，默认 **0=关**；`stage2/desig_rows.py`）：
#   同 type、y 差 ≤15px、x 邻距 <120px 的一排元件里，位号印在各自右侧时「逐颗捡最近
#   文字框」会整体右偏一颗（左邻框系统性近 ~3px），且 `assign` 先到先得。三条同时成立
#   （簇内元件数 == 可用位号数 / 位号没被簇外占用 / 现认框的 x 单调）才按 x 序重配，
#   并把该元件的 `desig_slots` 收成一条（否则 fusion 的全局解按旧候选表再解一遍，
#   改 `designator` 等于没改）。离线 13 页只命中 5 个簇，case1 的 6 元簇改对 4 颗、
#   改错 0，其余 4 个簇现状已对（不动）。
#   2026-11 取证：最终键由 `stage2/rowbind.py`（经 `fusion/run.py:619` 常开调用）
#   用 stage2 的错配框再改一次，配合本开关后该排 `labels == bodies`，rowbind 自己跳过。
#   2026-11-05 管线 A/B（`work/_onoff13.py`）：13 页 85.6952 → 85.7406（+0.0454），
#   official 84.6418 → 84.6880、judge 81.9486 → 81.9888；逐案只有 case1 **+0.885**
#   （1897 那颗回来，最终 pred 与 GT 该排完全一致），其余 12 页 0.00。
#   ⚠ 2026-11-05 池门禁判否：`work/_poolgate.py final1 --pool-ref ship3` ⇒ **29.5455
#   （−0.3027）**；离线归因（`work/_poolab.py`）把 compbox 关掉重算仍是 29.9200（=ship3，
#   零影响），所以 -0.37 全来自本开关，最大单案 `0042` **−6.06**、`0003` −1.66、
#   `0005` −1.56、`0091` −1.37。池页的「一排同 type 元件」常是**同值不同位号**的
#   电阻阵列，按 x 序重配会把本来对的位号打乱 ⇒ **默认回 0**，代码保留给 13 页档用。
DESIG_ROWS = env_on("STAGE2_DESIG_ROWS", "0")
# ★ 文字锚定的补件（`STAGE2_TEXT_ANCHOR`，默认 **1=开**；`stage2/text_anchor.py`）：
#   位号文字读到了、附近（默认 30px 内）却没有任何元件时，按**红色墨迹连通域**就地补一颗。
#   动机（`work/_textanchor.py`，13 页）：缺的 16 个 GT 件里有 3 个是「文字读得很准、
#   元件根本没检出」——case1 `L2`（位号 0.99 / 距最近元件 35.8px）、case11 `SW1`
#   （1.00 / 45.8px）、case12 `VOUT`（`VOUT` 0.75 / 96px）：四段半圆的电感、双触点
#   轻触开关、五边形输出端口这三族字形检测器系统性失明（`work/_hunt2.py` 在 ±130px
#   裁片、imgsz 640/1280、conf 0.01 下也一个都不吐）。
#   离线上界（`work/_rescueorc.py`）：补 GT 框 +1.97 / 补墨迹框 +1.61（三页合计，
#   均值 +0.15 / +0.12）；端口因墨迹连着导线（框宽 121 vs GT 82）暂不启用。
#   2026-11-05 管线 A/B（`work/_onoff13.py`）：13 页 86.6369 → **86.7607 / 85.2248 / 82.9913
#   （+0.1238）**，逐案只有 case1 **+0.33**（`L2`）、case11 **+1.29**（`SW1`），其余 11 页逐位不变。
#   池门禁：`work/_poolgate.py anchor5 --pool-ref mergeline STAGE2_TEXT_ANCHOR=1` ⇒ **29.6345**
#   （mergeline 29.6381，差 −0.0036 在噪声 0.0072 内）；全类型版 `anchor` 是 29.5997
#   （0001 五颗 `TP*` −0.55、0011 三颗 `D*` −0.44）⇒ 交付层只放 `l/switch`。
TEXT_ANCHOR = env_on("STAGE2_TEXT_ANCHOR", "1")
#   默认 1：池门禁 `wf4pool` = 29.6510，与 `dfill` 29.6510 **30 案逐位相同**（可证中性）。
#   未加「只认竖排」闸门时是 29.5502（−0.1008）：池页 0018 补了 8 颗水平读出的 `VCC/GND`
#   （那页 GT 只有 10 颗 gnd、3 颗 v，候选全是元件体内的引脚名）⇒ 每颗都是 extra。
TEXT_WORDS = env_on("STAGE2_TEXT_WORDS", "1")
#   实测（13 页，STAGE2_VTEXT=1 STAGE2_TEXT_WORDS=1）：local **87.3329** / official
#   **85.8063** / judge **83.5829**（基线 87.2413 / 85.7086 / 83.4913）；逐案只 case11
#   **+1.19**（竖排 `NDGND` → 暗墨迹细杆定位，框中心离 GT ∅695 7.5px ⇒ tp），其余 12 页逐位不变。
#   三条判据都是实测买来的：① 只补「GT 那边也是无名件」的类型（`gnd` 83 颗、`v` 51 颗全是
#   `∅NNN`）——`net_input`/`net_output` 的 GT 全是具名（`VIN`/`VOUT`/`BAT`），无名合成件在
#   `unkeyed_match='gt'` 下配不上、只会 +1 fp（case12 即使轮廓框已压在 GT `VIN_4` 上也 −0.51）；
#   ② 闭合轮廓优先（case12 的箭头是 25x75，先验 54x20 只会框到下面那根横条，中心差 37px）；
#   ③ 与既有件重叠的轮廓丢掉（否则 case11 会选中 LED2 的外框 109x65）。把词写进 `value` 后
#   case11 反而 −1.28（文本比对把件配到别的 GT 上）⇒ 不带 value。
# ★ 网标词锚定（`STAGE2_TEXT_WORDS`，默认 **0=关**；`stage2/text_anchor.py: rescue_words`）：
#   「文字 → 类型 → 导线末端/暗墨迹 → 按类型先验补件」，覆盖**没有位号**的符号（GND/VCC/
#   VIN/VOUT/3V3）。动机：case11 的两颗竖排 GND（`('GNDGND', 0.994)`、离最近元件 96px）
#   检测器只给 `c 0.015`；case12 有 6 条竖排网标（VIN×2、GND×4，conf 0.99~1.00）。
#   锚点不能只用导线末端（`work/_wireorc.py`：最近末端 21.5px 外，case11 −0.95），
#   要「末端 → 附近暗墨迹块 → 与类型先验取并集」；墨迹掩膜也从纯红改成「红 ∪ 暗」。
# ★ 位号回填（`STAGE2_DESIG_FILL`，默认 **0=关**；`stage2/text_anchor.fill_designators`）：
#   容灾——「件在、位号没读到」时不要整颗丢键，把 12px 内、类型前缀一致的**孤立位号文字**
#   回填成 `designator`。只改 `a["comps"][cid]["designator"]`，不动几何、不增删件。
#   实测：13 页 87.2015 → **87.2413（+0.0398）**，只改 1 件 case1 `p029 led → LED2`
#   （文字距离 0.0px、conf 0.947）⇒ 该案 +0.52；池门禁 `dfill` = **29.6510**，与 `shape1`
#   逐位相同（池上没有可回填的候选）⇒ 默认 1。
DESIG_FILL = env_on("STAGE2_DESIG_FILL", "1")
# ★ 竖排文字列补读（`STAGE2_VTEXT`，默认 **0=关**；`stage2/text.py: read_vertical`）：
#   识别端与检测端都只处理正立文字，竖排串会被切成单字（case11 那串 `GND GND` 在 reads
#   里是 x≈1219 的一列 `C/2/Q/D`）。把竖长框聚成一列、整体转 90° 再读：`('GND GND', 0.93)`；
#   反向对照横排 `GND` 正读 0.9996、转 90° 读空 ⇒ 只在竖长框上补试旋转是安全的。
#   case12 顺带量到 6 条竖排网标全丢（`VIN`×2、`GND`×4，conf 0.99~1.00，见 `work/_vcol_audit.py`）。
#   默认 1（与 `TEXT_WORDS` 配套）：竖排读法**只**作为网标词锚定的输入，不并入通用
#   `reads`（并入会改写 `desig_candidates`：池 0181 −0.22、0165 +0.02）。隔离后池上与不挂
#   这条臂逐位相同（`wf4pool` 29.6510 = `dfill` 29.6510，30 案逐位同）。
VTEXT = env_on("STAGE2_VTEXT", "1")
# ★ box 件「型号名文字框」重读（`STAGE2_PARTNAME_OCR`，§本次，默认 **0=关**；
#   `stage2/partname_ocr.py`）：把 `partname.box` 那颗文字框单独重裁一次送 PP-OCR，
#   在 `assemble.build_doc` 的 `Name` 最终成型处按档位覆盖：
#     `0` = 关（产物与不挂这条通道时逐字节相同）
#     `1` = 保守：只有 `libpins.norm(PP 文本)` 精确命中「形态良好」的库键才采用 PP 原串
#     `2` = 进取：`1` 之外再加「与现名相同」或「conf ≥ 0.90 且纯可打印 ASCII 且 ≥3 字符」
PARTNAME_OCR = partname_ocr.level()


# ★ 小字「放大重读」（`STAGE2_ZOOM_READ`，**默认 4 = 开**；0 表示关）。
#   依据（`work/_tinydesig.py` + 直接取证，页 `0001` 的 `C6`）：位号字高只有 **8px**，
#   整页检测器在 1× 上根本不吐这个框（整页 909 个读数里没有 `C6`）；把元件周围
#   ±40px 的窗口放大 **4×** 后再跑 `detect_texts(min_h=2)` + 同一 CRNN，读出
#   `C6:0.98`。2×/3× 不够（历史上「上采样无效」的结论是 1×/2×/3× 窗口尺度下做的）。
#   逐元件裁剪 ⇒ 每页只多 ~100 次小图前向，代价可控；命中的框坐标除回原原点并并入
#   `reads`，交给 `assign` 当普通候选参与打分（不直接改键）。
#   转正读数（§5.397.10，`MODE=refine`、`ROLES=designator,value`）：官方 10 张
#   90.71→**90.80**（只有 case8 变：72.86→73.72，pin 0.3226→0.3871；其余 9 张逐位不变，
#   10 张流水线总耗时 106s vs 108s 无差别），池 50 案 34.285→**34.33**。
#   ★ §5.397.13：`ROLES` 解码头从 `designator,value` 扩到**再加 `pin_name,pin_number`**
#   （官方 10 张 90.80→**91.04**、池 34.33→**34.38**）。角色表是 CRNN 的**解码提示**，
#   同一张低置信度小图在不同角色表下会吐出不同字符：case1 那个 `C154` 在只有
#   desig+value 时被读成 `C14`（少一位），加上引脚角色后读回 `C154`
#   （comp 0.8636→0.9091、netH 0.9685→0.9875）。
ZOOM_READ = float(os.environ.get("STAGE2_ZOOM_READ", "4") or 0)
ZOOM_READ_PAD = float(os.environ.get("STAGE2_ZOOM_READ_PAD", "40") or 40)
ZOOM_READ_MINH = int(os.environ.get("STAGE2_ZOOM_READ_MINH", "2") or 2)
ZOOM_READ_ROLES = tuple(x.strip() for x in os.environ.get(
    "STAGE2_ZOOM_READ_ROLES", "designator,value,pin_name,pin_number").split(",")
    if x.strip())
# 只把**形状像位号**的放大读数并进候选池（`desig`=位号形状 / `all`=不筛，用于对照）。
# 教训：不筛就并（`all`）时官方 case1 85.54 → **75.63**（引脚 0.7308→0.5385）——
# 放大窗口把**引脚名**也读成 designator/value，而 `assign` 的槽位是一对一竞争，
# 这些盒子会抢走引脚名槽位。筛到位号形状后污染面才等于目标面。
ZOOM_READ_KEEP = os.environ.get("STAGE2_ZOOM_READ_KEEP", "desig").strip().lower()
ZOOM_READ_CONF = float(os.environ.get("STAGE2_ZOOM_READ_CONF", "0.6") or 0.6)
# `STAGE2_ZOOM_READ_MODE`：`add`=把放大读数当**新候选**并入（历史实现，实测会把
#   槽位竞争搅乱：官方 case1 引脚 0.7308→0.5385）；`refine`=只**重读已存在的框**
#   （低置信的），不变动候选数量 ⇒ 不会引入竞争者。默认 `refine`。
ZOOM_READ_MODE = os.environ.get("STAGE2_ZOOM_READ_MODE", "refine").strip().lower()
ZOOM_REFINE_BELOW = float(os.environ.get("STAGE2_ZOOM_REFINE_BELOW", "0.7") or 0.7)
ZOOM_REFINE_GAIN = float(os.environ.get("STAGE2_ZOOM_REFINE_GAIN", "0.05") or 0.05)
ZOOM_REFINE_MAX = int(os.environ.get("STAGE2_ZOOM_REFINE_MAX", "0") or 0)
_ZOOM_DESIG_RE = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$")
_ZOOM_PINNUM_RE = re.compile(r"^(\d{1,3}[A-Z]?|[A-Z]\d{1,2})$")
_ZOOM_PINNAME_RE = re.compile(r"^[A-Z][A-Z0-9_/+.\-]{1,15}$")
# `desig` 只要位号形状；`pin` 只要脚号/脚名形状；`both` 两者都收；`all` 不筛（对照用）。
_ZOOM_KEEP_RE = {
    "desig": {"designator": _ZOOM_DESIG_RE},
    "pin": {"pin_number": _ZOOM_PINNUM_RE, "pin_name": _ZOOM_PINNAME_RE},
    "both": {"designator": _ZOOM_DESIG_RE, "pin_number": _ZOOM_PINNUM_RE,
             "pin_name": _ZOOM_PINNAME_RE},
}.get(ZOOM_READ_KEEP, {})

# ★ 多尺度文字检测（`STAGE2_MS_TEXT`，默认关`""`）。
#   动机：池页（KiCad/Altium 导出件）的位号是 6~9px 的细字，`cvparts.textseg` 在 1× 上
#   对它的召回不足——`work/_multiscale.py` 实测页 0001 的 GT 键串整页只读出 36/47，
#   把整页 CUBIC 放大 2×/3× 再检测、框除回 1× 后并集到 **41/47**（0013 是 2/18 → 7/18）。
#   与 `STAGE2_ZOOM_READ`（逐元件窗口）不同，这里**整页**跑，交给 `assign` 正常竞争。
#   值 = 逗号分隔的放大倍数，如 `2` / `2,3`；只收 1× 高度 ≤ `STAGE2_MS_TEXT_MAXH` 的框
#   （大标签 1× 已经检得出，重复加只会搅乱槽位），与已有框 IoU ≥ `STAGE2_MS_TEXT_IOU`
#   的丢弃（同一串不重复入池）。
MS_TEXT = tuple(float(x) for x in (os.environ.get("STAGE2_MS_TEXT", "2,3").split(",")) if x.strip())
MS_TEXT_MAXH = float(os.environ.get("STAGE2_MS_TEXT_MAXH", "20") or 20)
MS_TEXT_IOU = float(os.environ.get("STAGE2_MS_TEXT_IOU", "0.4") or 0.4)
MS_TEXT_ROLES = tuple(x.strip() for x in os.environ.get(
    "STAGE2_MS_TEXT_ROLES", "designator,value").split(",") if x.strip())
MS_TEXT_MINH = int(os.environ.get("STAGE2_MS_TEXT_MINH", "4") or 4)
# 三道护栏（默认全开）：多尺度只用来「补新位号」，不然会把 1× 已经读到的串再入池一遍、
# 或者让落在符号本体里的脚名去抢 designator/value 槽位——官方 case1 实测：不加护栏
# `STAGE2_MS_TEXT=2` 85.54→**81.72**（netH 0.9633→0.8757，pin 0.7308→0.7037）。
MS_TEXT_CONF = float(os.environ.get("STAGE2_MS_TEXT_CONF", "0.75") or 0.75)
MS_TEXT_OUTSIDE = env_on("STAGE2_MS_TEXT_OUTSIDE", "1")   # 只收落在元件框外的框
MS_TEXT_KEEP_RE = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$") if env_on(
    "STAGE2_MS_TEXT_KEYS", "1") else None

# ★ 页面尺度归一（`STAGE2_PAGE_SCALE_AUTO`，**默认关**——§5.387 判否）
#   `assign` 的归属/抢读半径是 1× 标定的绝对像素量，换 DPI 就整体失效；这里按
#   整页文字高度中位数估尺度注入。§5.387 实测：把尺度**强制**成真值（0.7×/1.4×）
#   后 GT10 降级集分数**一分不动**（69.54 / 59.30），而估准了反而会过冲到 2.3×
#   把 pin 打崩（1.4× 集 59.53 → 33.43）⇒ 这族半径不是瓶颈，开关默认关。
PAGE_SCALE_AUTO = env_on("STAGE2_PAGE_SCALE_AUTO", "0")
PAGE_SCALE_DBG = env_on("STAGE2_PAGE_SCALE_DBG", "0")
# ★ 队友 13 个模块里的「读数器」注册表（`origin/codex/improve-pin-network-parsing`）。
#   契约见各模块 docstring 顶部的【接口】块，也见 `process()` 里的用法注释。
#   每项 = (名字, 开关, 模块, 函数, 需要的角色, 额外 kwargs)；
#   kwargs 的值可以是 `roles -> value` 的函数（按角色决定要不要读名字/号）。
PIN_READERS = [
    ("pin_lane", True, "pin_lane_reading", "read_pin_lanes",
     ("pin_name", "pin_number"),
     {"include_names": lambda roles: roles is None or "pin_name" in roles,
      "include_numbers": lambda roles: roles is None or "pin_number" in roles,
     "cleanup": PIN_LANE_CLEANUP, "grey_evidence": True}),
    ("pin_number_repair", PIN_NUMBER_REPAIR, "pin_number_reading", "read_pin_numbers",
     ("pin_number",), {"grey_evidence": True}),
    ("pin_name_ocr", PIN_NAME_OCR != "off", "pin_name_reading", "read_pin_names",
     ("pin_name",), {"mode": PIN_NAME_OCR, "grey_evidence": True}),
    ("mono_pin_ocr", MONO_PIN_OCR != "off", "mono_pin_reading", "read_mono_pin_texts",
     ("pin_name", "pin_number"),
     {"mode": MONO_PIN_OCR,
      "include_names": lambda roles: roles is None or "pin_name" in roles,
      "include_numbers": lambda roles: roles is None or "pin_number" in roles}),
]


def _load_reader(mod: str):
    """按模块名加载队友读取器（先包相对导入，再退化绝对导入）。"""
    import importlib
    try:
        return importlib.import_module("." + mod, __package__ or "stage2")
    except Exception:
        return importlib.import_module("stage2." + mod)




# 这些是「参数」类元件，它们旁边没有长器件名，不参与生长
_VALUE_TYPES = {"r", "c", "l", "fuse", "crystal", "crystal_2pin",
                "crystal_3pin", "crystal_4pin"}


def _longest_read(reads) -> str:
    best = ""
    for _role, (t, _c) in (reads or {}).items():
        if t and len(t) > len(best):
            best = t
    return best


def recut_candidates(doc, reads, engine, bgr, max_new: int = 400):
    """**期望驱动的重切**（用户 m06470）：把同一行里相邻的**碎片框**拼成更长的框再读一遍。

    背景：`work\\reports\\错例集_导线与元件.md` 里 7645 条 `text_mismatch` 全是
    「长器件名被切碎」（GT `RP2350B` → 我们 `GP/2`）。`expect.merge_runs` 用
    「读出来很短 / 没小写 / 带分隔符」判碎片、用「拼起来更像个器件名」判该不该拼，
    这里把它接到管线上：拼出来的框补一次 OCR，再整体重新分配。
    """
    comps = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
    if not comps:
        return []
    groups = {}
    for r in reads:
        b = r["bbox"]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        j = min(range(len(comps)),
                key=lambda i: dist_to_rect(cx, cy, comps[i]["bbox_img"]))
        groups.setdefault(j, []).append(r)
    seen = {tuple(round(float(v), 3) for v in r["bbox"]) for r in reads}
    boxes_new = []
    for j, rs in groups.items():
        if len(rs) < 2:
            continue
        boxes = [list(r["bbox"]) for r in rs]
        texts = [_longest_read(r.get("reads")) for r in rs]
        hs = sorted(max(1.0, b[3] - b[1]) for b in boxes)
        for cand in merge_runs(boxes, texts, char_h=hs[len(hs) // 2]):
            key = tuple(round(float(v), 3) for v in cand["bbox"])
            if key in seen:
                continue
            seen.add(key)
            boxes_new.append(cand["bbox"])
            if len(boxes_new) >= max_new:
                break
    if not boxes_new:
        return []
    return engine.read(bgr, boxes_new)


def grow_candidates(doc, reads, engine, bgr, max_new: int = 600):
    """**沿阅读方向再框几个字**（用户 m06734）。

    与 `recut_candidates` 的区别很重要：那个是「拼**相邻的检测框**」，而真实情况是
    旁边**根本没有第二个框** —— 切分器只给出盖住名字中段的一个框（GT `RP2350B`
    读成 `GP/2`）。所以这里直接看像素：`growbox.grow_reading` 沿同一行往左右扩，
    停在「贯穿式竖线」（导线/符号边框）或空列预算用完的地方。

    两个「高」门槛：①只有 `expect.should_grow` 认为「这不像一个干净位号」的框才动
    ②长出来的新读法必须 `score_expect >= GROW_MIN_SCORE`（默认 0.9）才进候选池。
    """
    comps = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
    if not comps:
        return []
    groups: dict = {}
    for r in reads:
        b = r["bbox"]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        j = min(range(len(comps)),
                key=lambda i: dist_to_rect(cx, cy, comps[i]["bbox_img"]))
        groups.setdefault(j, []).append((dist_to_rect(cx, cy, comps[j]["bbox_img"]), r))
    seen = {tuple(round(float(v), 3) for v in r["bbox"]) for r in reads}
    boxes_new = []
    for j, rs in groups.items():
        if (comps[j].get("type") or "") in _VALUE_TYPES:
            continue                      # 参数类元件旁边没有长器件名
        rs.sort(key=lambda x: x[0])
        n = 0
        for _d, r in rs:
            if n >= GROW_PER_COMP:
                break
            _rd = _longest_read(r.get("reads"))
            # ★ 截断的位号会读成「看起来干净」的短串（`open_GT` case1 的印刷 `U1` 只剩 `1`、
            #   池子 `R59` 只剩 `R9`），`should_grow` 对它一律 False ⇒ 这里追加一条短串触发。
            if not (should_grow(_rd) or (GROW_SHORT and GROW_SHORT_RE.match(str(_rd).strip()))):
                continue
            g = grow_reading(bgr, r["bbox"])
            if not g:
                continue
            key = tuple(round(float(v), 3) for v in g)
            if key in seen:
                continue
            seen.add(key)
            boxes_new.append(g)
            n += 1
            if len(boxes_new) >= max_new:
                break
    if not boxes_new:
        return []
    out = []
    for r in engine.read(bgr, boxes_new):
        best = max((score_expect(t) for t, _c in (r.get("reads") or {}).values()),
                   default=0.0)
        if best >= GROW_MIN_SCORE:
            out.append(r)
    return out


def process(case: str, stage1_dir: Path, out_dir: Path, engine: TextEngine,
            roles=None, do_ocr: bool = True, min_conf: float = 0.0,
            texts: str = "both", recut: str = "off") -> dict:
    t0 = time.time()
    doc = json.loads((stage1_dir / ("%s.stage1.json" % case)).read_text(encoding="utf-8"))
    png = (doc.get("image") or {}).get("path")
    bgr = imread_unicode(png)

    # 文字框的来源：`stage1` = 复用 stage1 产物里的 `texts`（`cvparts.textbox` 出的，
    # 召回高、框偏小）；`detect` = 用 `cvparts.textseg` 当场重切（框紧，但少切 ~8%
    # 的框）；`union` = 以 stage1 框保底、能在 textseg 框里找到对应就把框收紧；
    # `both` = `union` 的结果再补回被换掉的那些 stage1 框（两个切法都留着）；
    # `auto` = 有 stage1 产物就复用，否则当场重切。
    #
    # **默认值是 `both`（2025-10-04 改）**。这个决定量过三次：
    #   旧管线（总分 ~6.7、`PIN_STAMP` 之前）：union 6.6592 < stage1 6.7307。
    #   中管线（总分 ~15.95）220 张：union **15.9525** | stage1 15.5567 |
    #     auto 同 stage1 | detect 15.4180。
    #   官方口径管线（2025-10-04）220 张四臂：
    #     both   **23.3815**（compF1 0.5144 tp 5158 fp 4470 fn 4451 netH 0.1940）
    #     union   22.9068（0.5077 / 5077 / 4551 / 4532 / 0.1862）
    #     stage1  22.7717（0.5005 / 5054 / 4574 / 4555 / 0.1889）
    #     auto    22.7717（同 stage1）
    #     detect  22.4571（0.4930 / 4923 / 4705 / 4686 / 0.1876）
    #   ⇒ union 一直是「只留 textseg 那一版」里最好的；`both`（两版都留）再 +0.475。
    #   别再来回翻。代价：0.74 s/张 → 0.87 s/张。
    #
    # `both` 的动机：`union` 是**替换**，同一个位置只留 textseg 那一版，可 textseg
    # 的切法在这一版 CRNN 上是**逐框随机**地更好或更差 —— `0001 c0052`（`STAGE2_DBG_CID`
    # 实测）位置上有 `[603,856,620,864]`（stage1 框，`pin_name` 读得 `TP1` 正确）
    # 和 `[602,855,619,863]`（textseg 框，只差 1px，五个角色全读成 `11`/`O`/`C1`）。
    # 一刀切就把好的那版丢了；两个都交给 `assign` 打分去选反而涨分。
    boxes, src = [], texts
    if texts in ("stage1", "auto"):
        boxes = read_stage1_texts(doc)
        src = "stage1"
    elif texts in ("union", "both", "multi"):
        from .text import merge_boxes, merge_boxes_both, merge_boxes_multi
        base = read_stage1_texts(doc)
        det = detect_texts(bgr)
        if texts == "both":
            boxes = merge_boxes_both(base, det)
        elif texts == "multi":
            boxes = merge_boxes_multi(base, det)
        else:
            boxes = merge_boxes(base, det)
        src = texts
    if not boxes:
        boxes = detect_texts(bgr)
        src = "detect"

    # ★ 叠行注释块按行切开（`stage2/text.split_stacked`，`TEXT_LINESPLIT` 默认开）。
    #   两行注释（位号 + 型号/值）被检测器当成一个框时，纵向压扁后位号必然读错。
    if LINESPLIT:
        from .text import split_stacked as _split
        _n0 = len(boxes)
        boxes = _split(bgr, boxes)
        if len(boxes) != _n0:
            src += "+line%d" % (len(boxes) - _n0)

    # ★★ **pincol 的引脚号窗口直接切、直接读**（`stage1/pincol.py`，默认开）。
    #
    # 引脚号是 6x8 px 的数字，`cvparts.textbox` 对它的召回很低（0143 的 U1：GT 64 个
    # 号只检出约 33 个），检不到的号就没有候选、`assign` 也就无从归位。而 pincol
    # 已经把每个号的窗口算好了 ⇒ 不必等检测器，按窗口切开收紧（定长 +3px 纵向外扩，
    # 曲线见 `work\reports\尺度归一_诊断.md`）再交给同一个 CRNN 读。
    if PINCOL_WIN_TEXTS:
        try:
            from stage1 import pincol as _pc
            _wb = _pc.win_boxes(bgr, doc.get("pincol") or {}, keys=PINCOL_WIN_KEYS)
        except Exception:
            _wb = []
        if _wb:
            boxes = list(boxes) + [b for b, _c, _k in _wb]
            src += "+pincolwin"

    t1 = time.time()
    if do_ocr:
        reads = engine.read(bgr, boxes) if roles is None else engine.read(bgr, boxes, roles)
    else:
        reads = [{"bbox": b, "reads": {}} for b in boxes]

    # ★ 竖排文字列补读（`STAGE2_VTEXT`，默认关）：见 `stage2/text.py: read_vertical`。
    _vreads: List[Dict] = []
    if do_ocr and VTEXT:
        try:
            _vreads = read_vertical(bgr, engine, boxes)
        except Exception:
            _vreads = []
        # 只交给网标词锚定用（见下方 `rescue_words`）：并入通用 `reads` 会改写
        # 位号回填候选（池 0181 −0.22、0165 +0.02），使这条臂在池上不可证中性。

    # ★ 小字放大重读（`STAGE2_ZOOM_READ`，默认关）：见文件头该开关的注释。
    #   只对**已检出的元件**做局部放大，把新增读数并入候选池；坐标一律除回原尺度，
    #   与整页读数同处一个坐标系，`assign` 无需知道它从哪来。
    # ★ 多尺度整页文字检测（`STAGE2_MS_TEXT`，默认关）：见开关块的注释。
    #   把整页放大 → 检测 → 框除回 1× → 去重 → 只读 `MS_TEXT_ROLES` → 并入候选池。
    if do_ocr and MS_TEXT:
        try:
            import cv2 as _cv2                      # noqa: PLC0415
            _n_new = 0
            _have = list(boxes)

            def _iou(a, b):
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                if ix <= 0 or iy <= 0:
                    return 0.0
                inter = ix * iy
                ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                return inter / ua if ua > 0 else 0.0

            for _s in MS_TEXT:
                _up = _cv2.resize(bgr, None, fx=_s, fy=_s, interpolation=_cv2.INTER_CUBIC)
                _bs = detect_texts(_up, min_h=int(MS_TEXT_MINH * _s),
                                   max_h=int(34 * _s))
                _add = []
                for _b in _bs:
                    _b1 = (_b[0] / _s, _b[1] / _s, _b[2] / _s, _b[3] / _s)
                    if _b1[3] - _b1[1] > MS_TEXT_MAXH:
                        continue
                    if MS_TEXT_OUTSIDE:
                        _mx = 0.5 * (_b1[0] + _b1[2])
                        _my = 0.5 * (_b1[1] + _b1[3])
                        if any(c["bbox_img"][0] - 2 <= _mx <= c["bbox_img"][2] + 2 and
                               c["bbox_img"][1] - 2 <= _my <= c["bbox_img"][3] + 2
                               for c in (doc.get("components") or []) if c.get("bbox_img")):
                            continue
                    if any(_iou(_b1, _h) >= MS_TEXT_IOU for _h in _have):
                        continue
                    _have.append(_b1)
                    _add.append((_b1, (_b[0], _b[1], _b[2], _b[3])))
                if not _add:
                    continue
                _got = engine.read(_up, [a[1] for a in _add], MS_TEXT_ROLES)
                _seen = set()
                for _r0 in reads:
                    _v0 = (_r0.get("reads") or {}).get("designator")
                    if _v0 and _v0[0]:
                        _seen.add(_v0[0].strip())
                for (_b1, _b2), _r in zip(_add, _got):
                    _rd = _r.get("reads") or {}
                    if MS_TEXT_KEEP_RE:
                        _dv = _rd.get("designator")
                        if not _dv or not _dv[0] or _dv[1] < MS_TEXT_CONF \
                                or not MS_TEXT_KEEP_RE.match(_dv[0].strip()):
                            continue
                        if _dv[0].strip() in _seen:
                            continue
                        _seen.add(_dv[0].strip())
                        _rd = {"designator": _dv}
                    boxes.append(_b1)
                    reads.append({"bbox": _b1, "reads": _rd, "ms": int(_s)})
                    _n_new += 1
            if _n_new:
                src += "+ms%s(%d)" % ("/".join("%g" % s for s in MS_TEXT), _n_new)
        except Exception as _e:                      # noqa: BLE001
            src += "+msfail(%s)" % type(_e).__name__

    # ★ 小字放大重读（`STAGE2_ZOOM_READ`，默认关）：见文件头该开关的注释。
    #   只对**已检出的元件**做局部放大，把新增读数并入候选池；坐标一律除回原尺度，
    #   与整页读数同处一个坐标系，`assign` 无需知道它从哪来。
    if do_ocr and ZOOM_READ > 1.0 and ZOOM_READ_MODE == "refine":
        # 只重读**已经存在**的框：把元件窗口放大后，用同一 CRNN 再读一遍这些框，
        # 只在「新读法置信度明显更高」时替换。候选数量不变 ⇒ 不动槽位竞争。
        try:
            import cv2 as _cv2                      # noqa: PLC0415
            _zc = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
            _plan = {}
            for _i, _r0 in enumerate(reads):
                _b0 = _r0["bbox"]
                _mx = max([v[1] for v in (_r0.get("reads") or {}).values() if v] or [0.0])
                if _mx >= ZOOM_REFINE_BELOW:
                    continue
                _j = min(range(len(_zc)),
                         key=lambda k: dist_to_rect(0.5 * (_b0[0] + _b0[2]),
                                                    0.5 * (_b0[1] + _b0[3]),
                                                    _zc[k]["bbox_img"]))
                _plan.setdefault(_j, []).append(_i)
            _n_ref = 0
            for _j, _idxs in _plan.items():
                if ZOOM_REFINE_MAX and _n_ref >= ZOOM_REFINE_MAX:
                    break
                _bb = _zc[_j]["bbox_img"]
                _x0 = max(0, int(_bb[0] - ZOOM_READ_PAD))
                _y0 = max(0, int(_bb[1] - ZOOM_READ_PAD))
                _x1 = min(bgr.shape[1], int(_bb[2] + ZOOM_READ_PAD))
                _y1 = min(bgr.shape[0], int(_bb[3] + ZOOM_READ_PAD))
                if _x1 - _x0 < 8 or _y1 - _y0 < 8:
                    continue
                _sub = _cv2.resize(bgr[_y0:_y1, _x0:_x1], None,
                                   fx=ZOOM_READ, fy=ZOOM_READ,
                                   interpolation=_cv2.INTER_CUBIC)
                _mb = [[(reads[_i]["bbox"][0] - _x0) * ZOOM_READ,
                        (reads[_i]["bbox"][1] - _y0) * ZOOM_READ,
                        (reads[_i]["bbox"][2] - _x0) * ZOOM_READ,
                        (reads[_i]["bbox"][3] - _y0) * ZOOM_READ] for _i in _idxs]
                _mb = [[max(0.0, b[0]), max(0.0, b[1]),
                        min(float(_sub.shape[1]), b[2]),
                        min(float(_sub.shape[0]), b[3])] for b in _mb]
                if any(b[2] - b[0] < 2 or b[3] - b[1] < 2 for b in _mb):
                    _idxs = [i for i, b in zip(_idxs, _mb) if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
                    _mb = [b for b in _mb if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
                    if not _idxs:
                        continue
                for _i, _nr in zip(_idxs, engine.read(
                        _sub, _mb, roles if roles is not None else ZOOM_READ_ROLES)):
                    _old = reads[_i].get("reads") or {}
                    _new = _nr.get("reads") or {}
                    _om = max([v[1] for v in _old.values() if v] or [0.0])
                    _nm = max([v[1] for v in _new.values() if v] or [0.0])
                    if _nm >= _om + ZOOM_REFINE_GAIN:
                        reads[_i]["reads"] = _new
                        reads[_i]["zoom"] = 1
                        _n_ref += 1
            if _n_ref:
                src += "+zrefine%g(%d)" % (ZOOM_READ, _n_ref)
            if os.environ.get("STAGE2_ZOOM_READ_DBG"):
                sys.stderr.write("[zrefine] lowconf=%d refined=%d\n"
                                 % (sum(len(v) for v in _plan.values()), _n_ref))
        except Exception as _e:
            src += "+zrefinefail(%s)" % type(_e).__name__
    elif do_ocr and ZOOM_READ > 1.0:
        try:
            import cv2 as _cv2                      # noqa: PLC0415
            _zc = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
            _seen = {tuple(round(float(v), 2) for v in b) for b in boxes}
            _added = 0
            for _c in _zc:
                _bb = _c["bbox_img"]
                _x0 = max(0, int(_bb[0] - ZOOM_READ_PAD))
                _y0 = max(0, int(_bb[1] - ZOOM_READ_PAD))
                _x1 = min(bgr.shape[1], int(_bb[2] + ZOOM_READ_PAD))
                _y1 = min(bgr.shape[0], int(_bb[3] + ZOOM_READ_PAD))
                if _x1 - _x0 < 8 or _y1 - _y0 < 8:
                    continue
                _sub = _cv2.resize(bgr[_y0:_y1, _x0:_x1], None,
                                   fx=ZOOM_READ, fy=ZOOM_READ,
                                   interpolation=_cv2.INTER_CUBIC)
                _zb = detect_texts(_sub, min_h=ZOOM_READ_MINH)
                if not _zb:
                    continue
                for _r in engine.read(_sub, _zb, ZOOM_READ_ROLES):
                    _rd = _r.get("reads") or {}
                    if ZOOM_READ_KEEP != "all":
                        _hit = []
                        for _role, v in _rd.items():
                            if not (v and v[0] and v[1] >= ZOOM_READ_CONF):
                                continue
                            _want = _ZOOM_KEEP_RE.get(_role)
                            if _want and _want.match(str(v[0]).strip().upper()):
                                _hit.append((_role, v))
                        if not _hit:
                            continue
                        _role, _v = max(_hit, key=lambda t: t[1][1])
                        _rd = {_role: (str(_v[0]).strip().upper(), float(_v[1]))}
                    _b = [_r["bbox"][0] / ZOOM_READ + _x0, _r["bbox"][1] / ZOOM_READ + _y0,
                          _r["bbox"][2] / ZOOM_READ + _x0, _r["bbox"][3] / ZOOM_READ + _y0]
                    _k = tuple(round(float(v), 2) for v in _b)
                    if _k in _seen:
                        continue
                    _mid = (0.5 * (_b[0] + _b[2]), 0.5 * (_b[1] + _b[3]))
                    if any(
                            (0.5 * (b[0] + b[2]) - _mid[0]) ** 2
                            + (0.5 * (b[1] + b[3]) - _mid[1]) ** 2 <= 36.0 for b in boxes):
                        continue
                    _seen.add(_k)
                    boxes.append(_b)
                    reads.append({"bbox": _b, "reads": _rd})
                    _added += 1
            if _added:
                src += "+zoom%g(%d)" % (ZOOM_READ, _added)
            if os.environ.get("STAGE2_ZOOM_READ_DBG"):
                sys.stderr.write("[zoom] comps=%d added=%d\n" % (len(_zc), _added))
        except Exception as _e:                     # 失败就退回整页读数，不拖垮管线
            src += "+zoomfail(%s)" % type(_e).__name__
    # ★ 队友分支的追加式读取器 —— **统一注册表**（开关与清单见文件头的 PIN_READERS）。
    #
    # 为什么是注册表：这些读取器/裁片器是「可重组的部件」，只有接口一致才能被 workflow
    # 自由连线（用户 m40034 第 2 项）。契约（13 个模块 docstring 里也都写着）：
    #     读数器   read_xxx(doc, reads, engine, bgr, *, min_conf=0.0, **opts) -> List[dict]
    #     裁片器   xxx_boxes(doc, image, reads=None, **opts)                  -> List[dict]
    #     返回     纯追加：只返回**新增**候选（{"bbox":…, "reads":{role:(text,conf)}}），
    #              绝不改入参、绝不删既有读数 ⇒ 关掉时下游逐位不变
    #     开关     只在**这里**读 env（模块自身不读 env）
    # 关掉或角色不匹配都整项跳过；单项异常只打印，不拖垮其它读取器。
    _ocr_ch = {}          # JOIN 用：两条 PP-OCR 通道各自的原始读数（JOIN 关 ⇒ 恒空）
    if do_ocr:
        for _name, _on, _mod, _fn, _need, _kw in PIN_READERS:
            if not _on:
                continue
            if roles is not None and not any(_r in roles for _r in _need):
                continue
            try:
                _m = _load_reader(_mod)
                _args = {k: (v(roles) if callable(v) else v) for k, v in _kw.items()}
                _extra = list(getattr(_m, _fn)(doc, reads, engine, bgr,
                                               min_conf=min_conf, **_args))
            except Exception as e:                     # 单个读取器失败不影响整张图
                print("[stage2] 读取器 %s 失败：%s" % (_name, e))
                continue
            if roles is not None:                      # 只留被请求的角色
                _extra = [dict(r, reads={k: v for k, v in r.get("reads", {}).items()
                                         if k in roles})
                          for r in _extra if any(k in r.get("reads", {}) for k in roles)]
            if _name in ("pin_number_repair", "pin_name_ocr"):
                _ocr_ch[_name] = list(_extra)
            if _extra and not (_name == "pin_number_repair" and PIN_NUMBER_JOIN_ONLY):
                reads = list(reads) + _extra

    t2 = time.time()

    for r in reads:
        for k, (txt, conf) in list(r["reads"].items()):
            if conf < min_conf:
                r["reads"][k] = ("", conf)

    # ★ 页面尺度归一（`STAGE2_PAGE_SCALE_AUTO`，默认开）：把 `assign` 里那串
    #   绝对像素半径（`NEAR`/`FAR`/`STEAL_R`/`VALUE_STEAL_R`/`PART_STEAL`）按整页
    #   文字高度中位数缩放到 1× 标定尺度。1× 页面中位高度 = `STAGE2_TEXT_MED`(10)
    #   ⇒ 尺度 1.0、产物逐位不变；1.4× 官方 case1 由 21.21 起、0.7× 由 69.54 起
    #   再量（见 docs/RESULTS.md §5.387）。
    if PAGE_SCALE_AUTO:
        # ★ 尺度估计用**检测器自己**的整页框高（`min_h=3/max_h=60`），不用 `reads`
        #   的高度：`reads` 走的是 stage1 文字框，框的分布随页面画法剧烈变化
        #   （1× 官方 case1 中位 5px、case2/8 中位 10px），拿它当尺子会把同一批
        #   1× 页面推出 0.5~1.0 的假尺度。标定值 `STAGE2_TEXT_MED`=10.8 是 1×
        #   十页同一统计量的均值（`work/_scaleprobe.py`）。
        if reads:
            _hs = [b[3] - b[1] for b in
                   detect_texts(bgr, min_h=3, max_h=60)]
            _ps = page_scale_from_heights(_hs)
            if PAGE_SCALE_DBG:
                print("[pagescale] %.3f median_h=%.1f n=%d"
                      % (_ps, sorted(_hs)[len(_hs) // 2] if _hs else 0.0, len(_hs)))
        else:
            set_page_scale(1.0)      # 没有任何文字框 ⇒ 回到标定尺度，避免上一案残留

    a = assign(doc.get("components") or [], doc.get("pins") or {}, reads,
               pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    if recut in ("expect", "grow", "both") and do_ocr:
        extra = []
        if recut in ("expect", "both"):
            extra += recut_candidates(doc, reads, engine, bgr)
        if recut in ("grow", "both"):
            extra += grow_candidates(doc, reads, engine, bgr)
        if extra:
            for r in extra:
                for k, (txt, conf) in list(r["reads"].items()):
                    if conf < min_conf:
                        r["reads"][k] = ("", conf)
            reads = list(reads) + list(extra)
            a = assign(doc.get("components") or [], doc.get("pins") or {}, reads,
                       pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    # ★ 引脚名救援（`stage2/pinname.py`，默认**开**，2026-10-16）：
    #   「号读对了、名字没读到」的 box 脚以前会被 `assemble.py` 的 NEED_NAME 闸门
    #   整条丢掉（官方 9 张：缺号 51 里有 23 根 = 45%）。这里从脚点往体内切带、
    #   剔掉体描边、按行分组只取离脚最近的一行，读一次；过闸门才写回名字。
    #   必须在 `build_doc` **之前**（名字要进交付文档），且只动 `a` 不删任何东西。
    #   默认值依据：官方 10 张 --no-vlm 60.5854 → 61.0000（+0.4146/张，3 涨 0 跌）、
    #   派生 val 19 张 17.0047 → 17.0322（+0.0274，4 涨 0 跌）；置 0 时逐位不变。
    try:
        from . import pinname as _pinname
        _pn = _pinname.rescue(bgr, doc, a, engine)
    except Exception as _e:                      # noqa: BLE001
        _pn = {"on": True, "error": "%s: %s" % (type(_e).__name__, _e)}
    # ★ 竖直脚旋转重读（`stage2/pinrot.py`，`STAGE2_PINROT` 默认关）：上下边的脚
    #   文字是旋转 90° 印的（官方 10GT：case 6 有 6 根、case 10 有 3 根、case 7 有 2 根），
    #   轴对齐裁片只能读成 'L'/'33' 这类碎片 —— case 6 引脚通道整张 0 分。这里把整图
    #   转 90°（顺/逆各一轮）、在旋转帧里取体内/体外条带重读号与名，过闸门才覆盖。
    try:
        from . import pinrot as _pinrot
        _pr = _pinrot.fix(bgr, doc, a, engine)
    except Exception as _e:                      # noqa: BLE001
        _pr = {"on": True, "error": "%s: %s" % (type(_e).__name__, _e)}
    # ★ 脚名「事后」重读（`stage2/pinnamereread.py`，`STAGE2_PINNAME_REREAD` 默认关；§5.467）：
    #   §5.466 把补读塞在 `TextEngine.read()` 里，读数一变 `assign` 的角色就重排
    #   （case8 `slot1` 的号 11→15），名字更对了反而掉 2.46。这里等 `assign` 把角色与号
    #   都定死之后再重读，只覆盖 `name.text`，**不碰 number、不增删槽**。
    try:
        from . import pinnamereread as _pnnr
        _nrr = _pnnr.fix(bgr, doc, a, engine)
    except Exception as _e:                      # noqa: BLE001
        _nrr = {"on": True, "error": "%s: %s" % (type(_e).__name__, _e)}
    # ★ 竖排网名补读（`stage2/pwlabel.py`，默认**开**）：
    #   电源符号的网名（`+5V`/`VCC`/`3.3V`）在这一版图里是**旋转 90° 竖排**印的，
    #   横排 CRNN 只读得出 `'R'` 这类碎片（open_GT case2 实测 0.47 → 转帧 0.95）。
    #   官方 GT 的电源符号都带 `value`，我们缺 35 个 `v` ⇒ 官方口径最大的一处缺口。
    #   纯追加：只给「本来没有网名」的符号写 `partname`，写不进就什么都不做。
    try:
        from . import pwlabel as _pwlabel
        _pl = _pwlabel.fill(bgr, doc, a, engine, reads=reads)
    except Exception as _e:                      # noqa: BLE001
        _pl = {"on": True, "error": "%s: %s" % (type(_e).__name__, _e)}
    # ★ 行内位号/值重绑（`stage2/rowbind.py`）：**不在这里调**。最终位号由
    #   `fusion.claims` 的 `solve` 全局重裁，stage2 里改什么都不留痕（`claims.py:713`
    #   把 `assigned` 全部降级成候选）——落在 `fusion/run.py` 的 `_assemble` 之后。
    # ★ 同槽双通道联合断言（见文件头常量处注释）：
    #   放在 `build_doc` 之前、且在上面所有「单通道」修法（pinname/pinrot/
    #   pinnamereread）之后 —— 只补它们各自补不出来的那一半，已有的值一律不覆盖。
    if _ocr_ch:
        try:
            from . import pin_ocr_join as _poj
            _ty = {str(c.get("id")): c.get("type") for c in doc.get("components") or []}
            _jn = _poj.join(a, _ocr_ch, _ty, min_conf=min_conf)
            if _jn.get("assert"):
                print("[stage2] 同槽双通道断言：%d 个槽（补号 %d、补名 %d；"
                      "已有名字跳过 %d、同号占位跳过 %d、非箱体 %d）"
                      % (_jn["assert"], _jn["filled_number"], _jn["filled_name"],
                         _jn["skip_named"], _jn["skip_claimed"], _jn["skipped_type"]))
        except Exception as _e:                      # noqa: BLE001
            print("[stage2] 同槽双通道断言失败：%s" % (_e,))
    # ★ 整颗元件「引脚号 ↔ 引脚名」抄表（`STAGE2_PIN_TABLE`，默认 0；见文件头常量处注释）：
    #   同样放在 `build_doc` 之前、所有单通道修法之后。只改已有槽的**名字**，
    #   不动坐标、不动号、不增删槽；端点不可用 / 表不可信 ⇒ 整条通道 no-op。
    if PIN_TABLE:
        try:
            from . import pin_table_vlm as _ptv
            _pt = _ptv.apply(bgr, doc, a)
            _ab = ",".join("%s=%d" % (k, v)
                           for k, v in sorted((_pt.get("abstain") or {}).items()))
            print("[stage2] 整颗元件抄表：箱体 %d 颗、问 %d 次 %.1fs、改名 %d 只；弃权 %s%s"
                  % (_pt.get("comps", 0), _pt.get("asked", 0), _pt.get("secs", 0.0),
                     _pt.get("renamed", 0), _ab or "-",
                     ("；错误 %s" % _pt["error"]) if _pt.get("error") else ""))
        except Exception as _e:                      # noqa: BLE001
            print("[stage2] 整颗元件抄表失败：%s" % (_e,))
    # ★ 缺口首位的整侧脚名重绑（见文件头常量处注释）：只改已有槽的**名字**，
    #   不动坐标、不动号、不增删槽；签名不成立 ⇒ no-op。
    if PIN_NAME_REBIND:
        try:
            from . import pin_rebind as _prb
            _rb = _prb.rebind(a)
            if _rb["changed"]:
                print("[stage2] 缺口首位重绑：%d 处（%s）"
                      % (_rb["changed"], "；".join("%s.pin_%s %s->%s"
                                                   % (d["component"], d["number"],
                                                      d["old"], d["new"])
                                                   for d in _rb["details"])))
        except Exception as _e:                      # noqa: BLE001
            print("[stage2] 缺口首位重绑失败：%s" % (_e,))
    # ★ 行内位号按 x 序一一对应（见常量处注释；默认关）。
    if DESIG_ROWS:
        try:
            from . import desig_rows as _dr
            _rs = _dr.fix(doc, a)
            if _rs["changed"]:
                print("[stage2] 行内位号重配：%d 簇 / %d 颗（%s）"
                      % (len(_rs["details"]), _rs["changed"],
                         "；".join("%s x%.0f %s->%s" % (d["cid"], d["x"], d["old"], d["new"])
                                   for g in _rs["details"] for d in g["fix"])))
        except Exception as _e:                       # noqa: BLE001
            print("[stage2] 行内位号重配失败：%s" % (_e,))
    # ★ 文字锚定的补件（见文件头 `TEXT_ANCHOR` 注释）：只追加合成元件，不改既有件。
    if TEXT_ANCHOR:
        try:
            from . import text_anchor as _ta
            _tres = _ta.rescue(doc, a, reads, bgr)
            if TEXT_WORDS:
                _wres = _ta.rescue_words(doc, a, list(reads) + list(_vreads), bgr)
                print("[stage2] 网标锚定：补 %d 颗 / 近距 %d、低置信 %d、无末端 %d、尺寸 %d、重叠 %d、非网标 %d"
                      % (len(_wres["added"]), _wres["near"], _wres["low_conf"],
                         _wres["no_wire"], _wres["bad_size"], _wres["overlap"], _wres["no_type"]))
                _tres["added"].extend(_wres["added"])
            print("[stage2] 文字锚定：补 %d 颗 / 近距跳过 %d、低置信 %d、无墨迹 %d、尺寸 %d、重键 %d、非位号 %d"
                  % (len(_tres["added"]), _tres["near"], _tres["low_conf"], _tres["no_ink"],
                     _tres["bad_size"], _tres["dup"], _tres["no_type"]))
            # 旁路文件交给交付层补（融合会把合成件按同框消重吃掉，见 `text_anchor` 头注释）。
            if _tres["added"]:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / ("%s.anchor.json" % case)).write_text(
                    json.dumps({"added": _tres["added"]}, ensure_ascii=False), encoding="utf-8")
            if _tres["added"]:
                print("[stage2] 文字锚定补件：%d 颗（%s）"
                      % (len(_tres["added"]), "；".join(
                          "%s/%s d=%.0f" % (d["key"], d["type"], d["d"])
                          for d in _tres["added"])))
        except Exception as _e:                      # noqa: BLE001
            print("[stage2] 文字锚定补件失败：%s" % (_e,))
    # ★ 位号回填（见文件头 `DESIG_FILL` 注释）：候选文字写侧车，交付层再改名。
    if DESIG_FILL:
        try:
            from . import text_anchor as _ta2
            _dc = _ta2.desig_candidates(reads)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / ("%s.desig.json" % case)).write_text(
                json.dumps({"reads": _dc}, ensure_ascii=False), encoding="utf-8")
            print("[stage2] 位号回填候选：%d 条" % len(_dc))
        except Exception as _e:                      # noqa: BLE001
            print("[stage2] 位号回填候选失败：%s" % (_e,))
    pred = build_doc(doc, a, case)
    pin_index = pred.pop("__pin_index", None)
    # ★ 同点同名重复脚去重（见文件头常量处注释）：只丢号更大的那颗，网引用同步改写。
    if PIN_DEDUP:
        try:
            from . import pin_rebind as _prb2
            _dd = _prb2.dedupe_coincident(pred)
            if _dd["dropped"]:
                print("[stage2] 同点同名脚去重：%d 个（%s）"
                      % (_dd["dropped"], "；".join("%s -> %s" % kv
                                                   for kv in sorted(_dd["alias"].items()))))
        except Exception as _e:                      # noqa: BLE001
            print("[stage2] 同点同名脚去重失败：%s" % (_e,))
    t3 = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("%s.json" % case)).write_text(
        json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    # ★ 旁路文件：引脚索引 → 交付文档里的 `pin_<N>` 键。融合层要改引脚号/名就
    # 必须知道这个映射，而它是 `build_doc` 内部「结构期望重排」的产物，外部
    # 反推不出来（实测三种反推法覆盖率 0.560 / 0.462 / 0.318）。
    # ⚠ 必须 `pop` 后再写 `<case>.json`：交付文档的字节不能有任何变化。
    if pin_index is not None:
        (out_dir / ("%s.pinindex.json" % case)).write_text(
            json.dumps(pin_index, ensure_ascii=False), encoding="utf-8")
    # 原始分配结果单独存一份：阶段三要拿它里的 `desig_slots` 做 LLM 仲裁，
    # 再把选定值回填后重新 `build_doc`（改名会牵动 `pins` 的键和 net 的 hyperGraph，
    # 所以只能重装配，不能事后改字符串）。
    (out_dir / "_assigned").mkdir(exist_ok=True)
    # ★ 「不删除，只削弱」账本（m40034 第四项）：随 `_assigned` 旁路落盘，**不进交付物**
    #   （`pred` 已在上面 `build_doc` 时定稿，这里只动旁路 dict `a`）。
    #   内容 = `cvparts.hypo.LEDGER`（改写/截断/丢弃事件 + 降权分）+ `structmatch`
    #   的降权候选与保留池（`HIST_RESERVE` = 重扫队列的输入）。
    try:
        from cvparts.hypo import LEDGER as _LG
        from . import structmatch as _sm
        # ⚠ 不能写 `from . import assign as _as`：`stage2/__init__` 里有
        #   `from .assign import assign`，包属性 `stage2.assign` 已被**函数**遮蔽，
        #   那样拿到的是函数（实测 `AttributeError: 'function' object has no
        #   attribute 'LAST_CONFORM'`）。直接按名字取模块级列表对象即可。
        from .assign import LAST_CONFORM as _lc
        from .assign import LAST_CONFORM_EXTRA as _le
        _nd = _LG.dump()
        _nd["demotions_pin"] = len(_sm.HIST_DEMOTED)
        _nd["reserve_pin"] = len(_sm.HIST_RESERVE)
        _nd["demoted"] = _sm.HIST_DEMOTED[-60:]
        _nd["reserve"] = _sm.HIST_RESERVE[-60:]
        _nd["residual"] = list(_sm.LAST_RESIDUAL)[-40:]
        # ★ 内涵「遗漏检测」（m48732）：本来按内涵
        #   规则（位号前缀 ⇄ 元件类型互证、削开头胶连）能拿到键、却因为判型/取证
        #   不对而最终没挂上键的元件。官方 10 张上它恰好点出 4 颗被丢掉的 LED
        #   （case 6 ×2、case 7、case 10）。**只写旁路报告**，不进交付物。
        _nd["conform_audit"] = list(_lc)
        # ★ 内涵「多余检测」（同一开关）：我方发出去的、内涵上必然判错的元件
        #   （`d`/`led` 的 Name 槽在官方 10GT 里是型号，我方 `name = key`）。
        _nd["conform_extra"] = list(_le)
        # ★ 大一统交叉校验层（用户 m50175，`CV_UNIFIED` 默认 0=关）：把已有的各种
        #   候选池/降权痕迹**只读**投影成统一 `Registry`（5 种并行候选表示 → `Claim`），
        #   跑内置校验（几何判据走 `GridIndex`/`AlongIndex`），产出 `Verdict` 台账。
        #   **只往旁路 dict `a` 里加 `unified` 键**：`pred` 早在 `:263 build_doc`
        #   定稿，交付 `<case>.json` 的字节不受影响；自带 try，坏了不牵连 `no_delete`。
        try:
            if os.environ.get("CV_UNIFIED", "0").strip().lower() not in (
                    "", "0", "false", "off", "none"):
                from cvparts import unified as _uf
                _reg = _uf.Registry("case%s" % case)
                _reg.add_many(_uf.project_stage1(doc))
                _reg.add_many(_uf.project_assigned(a))
                _reg.register(*_uf.default_checks())
                _reg.run()
                _nd["unified"] = _reg.to_json()
        except Exception as _ue:                  # noqa: BLE001
            _nd["unified"] = {"error": "%s: %s" % (type(_ue).__name__, _ue)}
        a["no_delete"] = _nd
    except Exception as _e:
        a["no_delete"] = {"error": "%s: %s" % (type(_e).__name__, _e)}
    (out_dir / "_assigned" / ("%s.json" % case)).write_text(
        json.dumps(a, ensure_ascii=False), encoding="utf-8")

    n_des = sum(1 for v in a["comps"].values() if v.get("designator"))
    n_val = sum(1 for v in a["comps"].values()
                if v.get("value") or v.get("partname"))
    n_pn = sum(1 for v in a["comps"].values()
               for p in (v.get("pins") or {}).values() if p.get("name"))
    n_pnum = sum(1 for v in a["comps"].values()
                 for p in (v.get("pins") or {}).values() if p.get("number"))
    n_pins = sum(len(v.get("pins") or {}) for v in a["comps"].values())
    return {
        "case": case, "n_components": len(doc.get("components") or []),
        "n_texts": len(boxes), "texts_src": src,
        "n_designator": n_des, "n_value_name": n_val,
        "n_pin_name": n_pn, "n_pin_number": n_pnum, "n_pins": n_pins,
        "n_net_labels": len(a["net_labels"]), "n_nets": len(pred["nets"]),
        "labels_merged": sum(1 for v in pred["nets"].values() if v.get("label")),
        "pinname_rescue": _pn,
        "pinrot": _pr,
        "t_read": round(t2 - t1, 2), "t_assign": round(t3 - t2, 2),
        "t_total": round(t3 - t0, 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第二阶段：文字识别 + 组装")
    ap.add_argument("--stage1", required=True, help="stage1 产物目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="CRNN 权重（默认取 stage2.text.DEFAULT_CKPT）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 case")
    ap.add_argument("--cases", default="", help="逗号分隔的 case id")
    ap.add_argument("--no-ocr", action="store_true", help="只做定位不做识别（对照臂）")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--texts", default="both", choices=("stage1", "detect", "union", "both", "multi", "auto"),
                    help="文字框来源（默认 both）：stage1=复用 stage1 产物（召回高、框偏小）；"
                         "detect=用 cvparts.textseg 当场重切（框紧）；union=以 stage1 框保底、"
                         "能用 textseg 框收紧就收紧；both=union 再补回被换掉的 stage1 框；"
                         "multi=both 再补上每个检出框的「外扩版」（**实测是负的**：23.0447 vs both 23.5009，留着只作对照）"
                         "（实测最好，+0.475）；auto=有 stage1 产物就复用")
    ap.add_argument("--recut", default="off", choices=("off", "expect", "grow", "both"),
                    help="expect=拼相邻碎片框再读一遍；grow=沿阅读方向把邻近笔画框进来再读；"
                         "both=两者都做（默认 off）")
    args = ap.parse_args(argv)

    s1 = Path(args.stage1)
    out = Path(args.out)
    files = sorted(s1.glob("*.stage1.json"))
    cases = [f.name[: -len(".stage1.json")] for f in files]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c in want]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有可跑的 case（--stage1 指向的目录里没有 *.stage1.json）")
        return 1

    engine = TextEngine(ckpt=args.ckpt, device=args.device)
    print("[stage2] %d 个 case  识别器: %s" % (len(cases), engine.describe()[11:]))
    if PARTNAME_OCR:
        print("[stage2] 型号名重读（STAGE2_PARTNAME_OCR=%d）：box 件 partname 文字框重送 PP-OCR"
              % PARTNAME_OCR)

    rows, failed, t0 = [], [], time.time()
    for i, c in enumerate(cases, 1):
        try:
            r = process(c, s1, out, engine, do_ocr=not args.no_ocr,
                        min_conf=args.min_conf, texts=args.texts, recut=args.recut)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print("  [%d/%d] %s 失败: %r" % (i, len(cases), c, exc))
            failed.append({"case": str(c), "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        rows.append(r)
        print("  [%d/%d] %s  元件 %d  文字 %d  位号 %d  值/名 %d  引脚名 %d  "
              "引脚号 %d  标签 %d  网络 %d  %.1fs"
              % (i, len(cases), c, r["n_components"], r["n_texts"], r["n_designator"],
                 r["n_value_name"], r["n_pin_name"], r["n_pin_number"],
                 r["n_net_labels"], r["n_nets"], r["t_total"]))

    wall = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / "_stage2_report.json").write_text(
        json.dumps({"cases": rows, "wall_s": round(wall, 1),
                    "n_cases": len(rows), "failed": failed},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    if rows:
        n = len(rows)
        # 2026-07 审计修复：这两个「赋值率」原来的分母是 `n_pin_name + n_pin_number`
        # —— 那是**成功数**之和，不是引脚总数 ⇒ 名字和号都读全时两个率都显示 0.5，
        # 任何调参都看不出真实的 0.9x 或 0.1x。改用 process() 新返回的 `n_pins`。
        _npins = max(1, sum(r.get("n_pins", 0) for r in rows))
        print("\n合计 %d 张，墙钟 %.1fs（%.2fs/张）；位号赋值率 %.3f，"
              "引脚名赋值率 %.3f，引脚号赋值率 %.3f"
              % (n, wall, wall / n,
                 sum(r["n_designator"] for r in rows) / max(1, sum(r["n_components"] for r in rows)),
                 sum(r["n_pin_name"] for r in rows) / _npins,
                 sum(r["n_pin_number"] for r in rows) / _npins))
    if failed:
        print("\n⚠️ 有 %d/%d 张 case 抛异常、**没有产物**（列出前 20）：%s"
              % (len(failed), len(cases), failed[:20]))
    print("产物 -> %s" % out)
    # 2026-07 审计修复：原来无论多少 case 失败都返回 0，「跑完了」≠「220 张都在」，
    # 缺文件在提交里只能表现为掉分。现在有失败就以非 0 退出，调用方（run.py -workers）
    # 才能把这一层报上去。
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
