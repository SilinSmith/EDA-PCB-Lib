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
import re
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
# ★★ 引脚号的**旋转补读**（`STAGE2_PIN_NUM_ROT`，**默认 0 = 关**）。
#
# 病（`work\_pnum_REPORT.md` §4，池案 0003 / U2 顶边那 13 个号）：竖排（整行旋转 90°）的
# 引脚号被检测器按**横排**文字框交出来，CRNN 直立读出来是「两个字并成一列」的乱码 ⇒ 过不了
# `assign.PINNUM_RE`，在号轮 `if not got: continue` 被静默丢掉（池 30 案共 B4+B5 = 192 槽，
# 占 stage1 已检槽点的 20.5%）。框**检出了、位置对、尺寸对**，缺的只是一次旋转。
#
# 触发与接受**都不新增阈值**：直立读 `fit_pinnum()<=0` 才补读（既有判据）；补出来的串仍要过
# 既有 `PINNUM_RE`，之后照旧进 `assign._best_read` / `_score` 与全部几何门。旋转只试 90°/270°
# 两个方向（同一张裁片做两次高宽互换的遍历），没有「角度步长」这类参数。
# 尺寸守门用 `stage1.pincol.NUM_MAX_W`（既有：引脚号文字框的最大边长，42px）。**不看**
# `NUM_MAX_H`：竖排裁片本身就是宽高互换的，高 26 那条是按直立框量的上界，单体框实测
# 34×26 / 34×27（`_pnum_REPORT.md` §4.2）的短边正好越界，照搬会把要救的框全挡掉。
PIN_NUM_ROT = float(os.environ.get("STAGE2_PIN_NUM_ROT", "0")) > 0
# ★★ 文字框的「**有色墨迹**」统计（由 `STAGE2_PINNAME_INK` 打开）。
#
# 病（10GT case 5 的 `U3`，见 `docs/RESULTS.md` §5.91）：pin1 名字框那一行有两个候选
# —— 真名 `ADJ(GND)`（67x13）与**红点引脚标记**（8x8，CRNN 读 `Q`）。名轮把标记绑给
# pin1、把真名挤到 pin2 ⇒ 两条名字双双错位。
# **几何分不开**（合法名 `L` 是 7x10、红点是 8x8），但**墨迹颜色分得开**：实测同一张图
#   * 真名框（`FB`/`P`/`EN`/`L`，10 个）本项 = **0.000**；
#   * 红点（`U3` 两处 + `p022`）= **0.641 / 0.484 / 0.500**。
# 判据与 `tight_ink_box` 同源（`max-min >= SAT` 且灰度 < DARK），只是这里算**占比**。
# 结果作为 `cfrac` 挂在 `read()` 返回的每条记录上（`assign.py` 的名轮据此踢掉标记框）。
TEXT_INK_SAT = float(os.environ.get("STAGE2_TEXT_INK_SAT", "60"))
TEXT_INK_DARK = float(os.environ.get("STAGE2_TEXT_INK_DARK", "160"))
# ★★ 同一份统计的**只供名轮**入口（`STAGE2_PINNAME_INK`，默认 **1 = 开**，§5.550：13 页 82.97→83.72 的主修手 +0.43）。
#   这条明确写着「只为名轮的 `cfrac` 闸门开」，便于单独 A/B；实现上等价（`cfrac` 只被
#   `assign.PINNAME_CFRAC`(:2813 / :3023) 消费），与整条颜色通道共用同一份统计。
#   动机（新识别器回归）：`open_GT case6` 的 `U4`/`U1` 各有一个**红圈极性标记**被读成
#   脚名（`o1` 0.44 / `o` 0.80）并抢到脚名槽 —— 实测红点框 `cfrac` 0.250/0.512、
#   真名框全 0.000，正是 `PINNAME_CFRAC`(0.20) 能分开的那一类。
PINNAME_INK = os.environ.get("STAGE2_PINNAME_INK", "1").strip().lower() in ("1", "true", "yes", "on")
TEXT_INK = PINNAME_INK

ENS_MARGIN = float(os.environ.get("STAGE2_ENS_MARGIN", "0.05") or 0.0)
ENS_ROLES = tuple(r.strip() for r in
                  (os.environ.get("STAGE2_ENS_ROLES") or "designator,value,pin_number,pin_name").split(",")
                  if r.strip())
ENS_DBG = os.environ.get("STAGE2_ENS_DBG", "0") not in ("0", "", "off")
_ENS: List[Tuple[str, str, float, float]] = []
# ★★ 第三识别通道：通用 OCR（PP-OCR 识别头，`stage2/genocr.py`；`STAGE2_GEN_OCR`，默认
#   **"2"**，§5.567 起翻默认；"0" = 完全不加载、不调用）。两套模型的置信度**不同尺度，
#   禁止直接比数字**，判据全部落在字符串形状上：
#     `=1` 保守：我方读法**不合该角色语法**而通用读法合 ⇒ 换；
#     `=2` 再加「双方都合语法、通用读法更长更完整」（`04`→`104`），增量必须 <= 2 字（防粘连邻框）。
#   角色只到 `designator/value/partname/pin_name`：`pin_number` 那一路有自己的切描边与
#   旋转补读（`TIGHT_ROLES` / `PIN_NUM_ROT`），换读数会打乱那套量过的几何判据。
#   13 页 84.5427 → **84.9945（+0.4518）**（case1 +0.54 / case5 +3.03 / case11 +2.31，
#   其余 10 页逐字节不变，FIX 5 / BREAK 0）；池 29.1209 → **29.8722（+0.7513）**。
GEN_OCR_LEVEL = 0 if (os.environ.get("STAGE2_GEN_OCR", "2").strip()
                      in ("", "0", "off", "False", "false")) else (
    2 if os.environ.get("STAGE2_GEN_OCR", "2").strip().startswith("2") else 1)
GEN_OCR_ROLES = ("designator", "value", "partname", "pin_name")
GEN_OCR_DBG = os.environ.get("STAGE2_DBG_GENOCR", "0") not in ("0", "", "off")
#: `STAGE2_DBG_GENOCR=1` 时把替换记录另存一份 JSONL（`_cv.py` 会吞掉子进程 stdout，
#: 出「逐条修复/弄坏清单」要靠这个文件；不在 DBG 档时一个字节都不写）。
GEN_OCR_LOG = (os.environ.get("STAGE2_GENOCR_LOG") or "").strip()
GLYPH_VOTE_MIN = int(os.environ.get("STAGE2_GLYPH_VOTE_MIN", "3") or 3)
GLYPH_VOTE_TAU = float(os.environ.get("STAGE2_GLYPH_VOTE_TAU", "3") or 3)
GLYPH_VOTE_PURITY = float(os.environ.get("STAGE2_GLYPH_VOTE_PURITY", "0.6") or 0.6)
GLYPH_VOTE_CONF = float(os.environ.get("STAGE2_GLYPH_VOTE_CONF", "0.35") or 0.35)
GLYPH_VOTE_OVERRIDE = float(os.environ.get("STAGE2_GLYPH_VOTE_OVERRIDE", "0.6") or 0.6)
GLYPH_VOTE_BEAM = int(os.environ.get("STAGE2_GLYPH_VOTE_BEAM", "8") or 8)
GLYPH_VOTE_MAX = int(os.environ.get("STAGE2_GLYPH_VOTE_MAX", "4000") or 4000)
# ★★ 「图上有几个字形」约束解码（`STAGE2_READ_LENFIX`，默认 **1**，§5.427）。
#   贪心 CTC 会整字吞掉窄字/夹在中间的字（`TIM→TM`、`TP4333→TP433`），七个权重一致、
#   补白也救不回；但字形段数（列投影连通段）与模型无关，是可靠的独立信号。
READ_LENFIX = os.environ.get("STAGE2_READ_LENFIX", "1") not in ("0", "", "off", "False", "false")
LENFIX_ROLES = tuple(r.strip() for r in
                     (os.environ.get("STAGE2_READ_LENFIX_ROLES")
                      or "pin_name,designator,value,partname,net_label,pin_number").split(",")
                     if r.strip())
LENFIX_INK = float(os.environ.get("STAGE2_READ_LENFIX_INK", "0.8") or 0.8)
LENFIX_MINW = int(os.environ.get("STAGE2_READ_LENFIX_MINW", "2") or 2)
LENFIX_NARROW = int(os.environ.get("STAGE2_READ_LENFIX_NARROW", "4") or 4)
LENFIX_MAXN = int(os.environ.get("STAGE2_READ_LENFIX_MAXN", "14") or 14)
LENFIX_MAXDIFF = int(os.environ.get("STAGE2_READ_LENFIX_MAXDIFF", "3") or 3)
LENFIX_K = int(os.environ.get("STAGE2_READ_LENFIX_K", "10") or 10)
LENFIX_BEAM = int(os.environ.get("STAGE2_READ_LENFIX_BEAM", "24") or 24)
LENFIX_DBG = os.environ.get("STAGE2_READ_LENFIX_DBG", "0") not in ("0", "", "off")
LENFIX_MARGIN = float(os.environ.get("STAGE2_READ_LENFIX_MARGIN", "3.0") or 0.0)


# ---------------------------------------------------------------------------
def _role_fit(role: str, text: str) -> float:
    """该角色下的**语法**适配度（只用既有判据，不看置信度 —— 两套模型的分数不可比）。"""
    if not text:
        return 0.0
    # ⚠ 不能写 `from . import assign`：`stage2/__init__` 的 `from .assign import assign`
    #   把包属性 `stage2.assign` 遮蔽成**函数**（`run.py:776` 记过同一个坑）。
    if role == "value":
        from .assign import fit_value
        return fit_value(text)
    if role == "designator":
        # 读文字时还不知道器件类型：空 type 走 `canon_designator` 的通用形态分支。
        from .assign import fit_designator
        return fit_designator(text, "")
    if role == "partname":
        return 1.0 if (len(text) >= 4 and re.search(r"[A-Za-z]", text)) else 0.0
    if role == "pin_name":
        from .assign import fit_pinname
        return fit_pinname(text)
    return 0.0


def _beam_rank(tc: Tuple[str, float]) -> float:
    """beam 候选的**总发射对数概率**：`_beam` 给的 conf 是 `exp(ssum/n)`，还原成 `ssum`。"""
    t, conf = tc
    if not t:
        return -1e9
    return len(t) * float(np.log(max(1e-6, min(1.0, float(conf)))))


def _ctc_logprob(lp: np.ndarray, label: str, itos: Dict[int, str]) -> float:
    """标准 CTC 前向：给定 `log_softmax` 矩阵与候选串，返回 `log p(label)`。

    `_beam` 里的 conf 是**发射概率的几何平均**（与路径概率不成比例），不能用来比较
    「多一个字」和「少一个字」的候选 —— 这里按定义算，够快（T≤40、L≤14）。
    """
    if not label:
        return -1e9
    stoi = {ch: i for i, ch in itos.items()}
    ids = [stoi.get(ch, -1) for ch in label]
    if any(i < 0 for i in ids):
        return -1e9
    L = len(ids)
    neg = -1e18
    # 扩展标签：blk, l1, blk, l2, ..., lL, blk
    ext = [0]
    for i in ids:
        ext.extend([i, 0])
    S = len(ext)
    prev = np.full(S, neg)
    prev[0] = lp[0, 0]
    if S > 1:
        prev[1] = lp[0, ext[1]]
    for t in range(1, lp.shape[0]):
        cur = np.full(S, neg)
        for s in range(S):
            a = prev[s]
            if s >= 1:
                a = np.logaddexp(a, prev[s - 1])
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]:
                a = np.logaddexp(a, prev[s - 2])
            cur[s] = a + lp[t, ext[s]]
        prev = cur
    out = prev[S - 1]
    if S > 1:
        out = np.logaddexp(out, prev[S - 2])
    return float(out)


# ★★ 单位字母的**墨迹判别**（`STAGE2_VALUE_MU`，默认 1 = 开）。
#
# 病：GT 里微符号恒为 **U+03BC**（池子 200 页 110 个、GT10 1 个），我们只输出 ASCII `u`；
# 反过来 `10nF` 被读成 `10uF`（GT10 case1 `C158`，§5.370）。CRNN 在 8-10px 上把
# `μ`/`u`/`n` 当同一个形状，是**字符级**混淆——整串 top-k、同页投票都吃过（都判否）。
# 但这两个字的形态在像素上是**可分**的：
#   * `μ` 有降部：`tail`（字形底边低于基线的像素）中位 3px，`u`/`n` **恒 0**
#     ⇒ 池子 60 页 153 个样本上 `tail>=2` 命中 72% 的 μ、**误报 0**。
#   * `n` 拱在顶上、`u` 碗在底下：`mb`（上 1/3 墨量 − 下 1/3 墨量）n 中位 **+0.03**、
#     u 中位 **−0.11**，`mb<=-0.02` 判 n/u **94.8%**。
# 只在字符串形状是「数字 + 单位 + 尾字母」（`12uF`/`0.1uF`/`4.7<n>H`）时才动。
VALUE_MU = os.environ.get("STAGE2_VALUE_MU", "1").strip().lower() in ("1", "true", "yes", "on")
VALUE_MU_TAIL = float(os.environ.get("STAGE2_VALUE_MU_TAIL", "2.0") or 2.0)
VALUE_UNIT_ROLES = ("value",)
_UNIT_STATS: Dict[str, int] = {}
# ★★ 「值尾部那个读不出来的字母」补全（`STAGE2_VALUE_TAIL`，默认 **1 = 开**；§5.491）。
#
# 病：`open_GT` case13 的 C2/C6 值字是 `22uf`，但 `f` 是一根细竖笔，CRNN 在 32px 下**读不出**
# 它 —— 贪心串是 `22u`，`_length_fix` 明知「图上有 4 个字形」也只能在 beam 里挑，而 beam 给的
# 4 字候选是 `220u`/`22um`/`22u1`（全错），其中 `22um` 还会被 `canon_value` 削回 `22u`。
# ⇒ 这条路的瓶颈不是「分辨 f 还是 F」，而是**根本没有字符能命名那个字形**。
#
# 做法：只在「读数 = 数字 + 单位字母（p/n/u/µ）」且「字形段数 = 读数长度 + 1」时才动手 ——
# 即图上确实多一个没读出来的字形、位置在末尾；然后用**字宽**决定大小写：
# 该字形列宽 < `VALUE_TAIL_W` × 同框数字列宽中位数 ⇒ 小写 `f`，否则大写 `F`
# （实测 case13：`22uf` 尾字宽 7px、数字 9px ⇒ 0.78；`100nF` 尾字宽 10–11px ⇒ ≥1.1）。
# ⚠ 必须同时开 `STAGE2_VALUE_TAILRE=1`（`assign.VALUE_RE` 才认小写尾字母），否则 `canon_value`
# 会把刚补上的 `f` 再削掉 —— 两个旋钮是一对，单开任何一个都逐位惰性（§5.491.5 两条对照档）。
#
# 2026-11 落默认：主尺 80.36 → **80.48**（只有 case13 +1.54，全尺只改 2 个值且都改对）；
# 池 30 案 Δ −0.02 落在批次噪声内，且逐案复核证明**本档在池上一处都没触发**（0003/0004 的
# 预测与无 env 档逐字节相同）⇒ 没有可复现的池伤害，翻默认。
VALUE_TAIL = int(os.environ.get("STAGE2_VALUE_TAIL", "1") or 0)
VALUE_TAIL_W = float(os.environ.get("STAGE2_VALUE_TAIL_W", "0.85") or 0.85)
VALUE_DOT = int(os.environ.get("STAGE2_VALUE_DOT", "1") or 0)
VALUE_DOT_W = float(os.environ.get("STAGE2_VALUE_DOT_W", "0.45") or 0.45)
VALUE_DOT_BOT = float(os.environ.get("STAGE2_VALUE_DOT_BOT", "0.45") or 0.45)
_DOTSRC_RE = re.compile(r"^(?P<a>\d)(?P<b>\d+)(?P<u>[pnu\u00b5\u03bc]?)$")
_TAILSRC_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<u>[pnu\u00b5\u03bc])$")
_TAIL_DONE_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)\s*[pnu\u00b5\u03bcmkKMGR]?[FHV\u03a9AWRJfhvr]?$")
GLYPH_VOTE_ROLES = tuple(r.strip() for r in os.environ.get(
    "STAGE2_GLYPH_VOTE_ROLES", "designator,pin_number,pin_name,value").split(",") if r.strip())


_STATS: Dict[str, int] = {}
_EXAMPLES: List = []


def glyph_sig(gray: np.ndarray, box: Box, wh: Tuple[int, int] = (16, 32)):
    """框内墨迹的二值签名（按框均值二值化后缩放），用于同形聚类。"""
    h, w = gray.shape[:2]
    x0 = max(0, int(np.floor(box[0])))
    y0 = max(0, int(np.floor(box[1])))
    x1 = min(w, int(np.ceil(box[2])))
    y1 = min(h, int(np.ceil(box[3])))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    c = gray[y0:y1, x0:x1]
    c = cv2.resize(c, wh, interpolation=cv2.INTER_AREA)
    return (c < float(c.mean())).astype(np.uint8).ravel()


def glyph_vote(reads: List[Dict], cands: Dict, sigs: Dict) -> int:
    """同形簇内投票改写读数。返回被改写的框-角色数（诊断用）。"""
    order = list(sigs)
    clusters: List[Dict] = []
    for key in order:
        s = sigs[key]
        hit = None
        for cl in clusters:
            if int(np.count_nonzero(cl["rep"] ^ s)) <= GLYPH_VOTE_TAU:
                hit = cl
                break
        if hit is None:
            clusters.append({"cnt": s.astype(np.int16), "n": 1, "rep": s.copy(),
                             "members": [key]})
            continue
        hit["cnt"] += s
        hit["n"] += 1
        hit["rep"] = (2 * hit["cnt"] >= hit["n"]).astype(np.uint8)
        hit["members"].append(key)

    n_fix = 0
    why = {"clusters": len(clusters), "big": 0, "twovote": 0, "pure": 0}
    for cl in clusters:
        if cl["n"] < GLYPH_VOTE_MIN:
            continue
        why["big"] += 1
        votes: Dict[str, List[float]] = {}
        for key in cl["members"]:
            top = cands.get(key) or []
            if not top:
                continue
            t, sc = top[0]
            if not t or float(sc) < GLYPH_VOTE_CONF:
                continue
            v = votes.setdefault(t, [0.0, 0.0])
            v[0] += 1.0
            v[1] += float(sc)
        if len(votes) < 2:
            continue
        why["twovote"] += 1
        total = sum(v[0] for v in votes.values())
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
        best, (cnt, _s) = ranked[0]
        if total < GLYPH_VOTE_MIN or cnt / total < GLYPH_VOTE_PURITY:
            continue
        why["pure"] += 1
        for i, r in cl["members"]:
            cur = reads[i].get(r)
            if not cur or cur[0] == best:
                continue
            if float(cur[1]) > GLYPH_VOTE_OVERRIDE:
                continue                      # 自己读得很像就信自己
            if not any(t == best for t, _s in (cands.get((i, r)) or [])):
                continue                      # 该框自己没考虑过这个串 ⇒ 不改
            if os.environ.get("STAGE2_GLYPH_VOTE_DBG", "0") not in ("0", "", "off"):
                _STATS[r] = _STATS.get(r, 0) + 1
                if len(_EXAMPLES) < 24:
                    _EXAMPLES.append((r, cur[0], best, round(float(cur[1]), 2), cl["n"]))
            reads[i][r] = (best, float(cur[1]))
            n_fix += 1
    if os.environ.get("STAGE2_GLYPH_VOTE_DBG", "0") not in ("0", "", "off"):
        print("[glyph] 漏斗 %s" % why)
    return n_fix

# ★ 「叠行注释块按行切开」（见 `split_stacked`）：高 ≥ LINESPLIT_H × 页面中位字高的框才切。
LINESPLIT = os.environ.get("TEXT_LINESPLIT", "1").strip().lower() in ("1", "true", "yes", "on")
LINESPLIT_H = float(os.environ.get("TEXT_LINESPLIT_H", "1.7"))
LAST_SPLIT = [0, 0]        # [切开的框数, 切出的行数]


def colored_frac(bgr: np.ndarray, box: Box, sat: float = None, dark: float = None) -> float:
    """框内**有色墨迹**占比（0~1）：`(max-min) >= sat` 且灰度均值 `< dark` 的像素比例。

    灰度图、空框、越界框一律 0.0。**不改框本身**，只作为候选的附加属性供 `assign` 用。
    """
    if bgr is None or getattr(bgr, "ndim", 2) != 3:
        return 0.0
    sat = TEXT_INK_SAT if sat is None else sat
    dark = TEXT_INK_DARK if dark is None else dark
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(int(bgr.shape[1]), x2); y2 = min(int(bgr.shape[0]), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    m = ((sub.max(2) - sub.min(2)) >= sat) & (sub.mean(2) < dark)
    return float(m.mean())


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


def _glyph_runs(dark: np.ndarray) -> List[Tuple[int, int]]:
    """列投影的连通段（= 逐个字形的左右边界）。"""
    cols = dark.any(axis=0)
    runs, s = [], None
    for i, v in enumerate(cols):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(cols)))
    return runs


def _unitglyph_features(bgr: np.ndarray, box: Box) -> Optional[Dict[str, float]]:
    """量框内**倒数第二个字形**（`12uF` 里的 `u`）的形态，供 `μ`/`u` 判别。

    - `tail`：该字形底边低于框内基线的像素数。`μ` 有降部（p50 3px），`u`/`n` 没有（恒 0）。
    - `mb`：该字形**上三分之一与下三分之一的墨量之差**。`n` 的拱在顶上（+0.03），
      `u` 的碗在底下（−0.11）——池子 153 个样本上 `mb<=-0.02` 判 n/u 准确 94.8%。
    - `cy`：墨迹重心（0=字形顶，1=基线），同一现象的另一视角。

    基线取该框所有列底边的 75 分位（下划线、`%` 之类的低悬笔不拉低基线）。
    """
    if bgr is None or getattr(bgr, "ndim", 2) != 3:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
    x2, y2 = min(int(bgr.shape[1]), x2 + 1), min(int(bgr.shape[0]), y2 + 1)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    dark = sub.min(2) < 170      # 只按「暗」判墨（值的蓝字是**有色**的，加饱和闸会把它全切掉）
    runs = _glyph_runs(dark)
    if len(runs) < 3:
        return None
    a, b = runs[-2]
    seg = dark[:, a:b]
    rows = np.nonzero(seg.any(1))[0]
    if len(rows) == 0:
        return None
    bottoms = [int(np.nonzero(dark[:, c])[0][-1]) for c in range(dark.shape[1]) if dark[:, c].any()]
    if not bottoms:
        return None
    base = float(np.percentile(bottoms, 75))
    ys = np.nonzero(seg)[0]
    top, bot = float(rows[0]), float(rows[-1])
    span = max(1.0, bot - top)
    third = max(1.0, span / 3.0)
    return {"tail": float(rows[-1]) - base,
            "mb": float((ys - top < third).mean()) - float((ys - top > span - third).mean()),
            "cy": float((ys - top).mean() / span),
            "w": float(b - a)}


_MU_RE = re.compile(r"^(?P<a>\d+(?:\.\d+)?)\s*(?P<u>[uUnN\u03bc\u00b5])\s*(?P<t>[FfHhMmRr])$")
UNIT_MU = "\u03bc"


def fix_unit_text(bgr: np.ndarray, txt: str, box: Box,
                  tail_min: float = None) -> str:
    """用字形墨迹改掉 `12uF` 里的单位字母：降部 → `μ`；`n`/`u` 按上下墨量判。

    只在字符串形状是「数字 + 单位 + 尾字母」时才动，其余原样返回。
    """
    if not txt:
        return txt
    tail_min = VALUE_MU_TAIL if tail_min is None else tail_min
    m = _MU_RE.match(txt.strip())
    if not m:
        return txt
    u = m.group("u")
    f = _unitglyph_features(bgr, box)
    if f is None:
        return txt
    new = u
    if u in "uU" and f["tail"] >= tail_min:
        new = UNIT_MU
    if new == u:
        return txt
    return txt[:m.start("u")] + new + txt[m.end("u"):]


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


def split_stacked(bgr: np.ndarray, boxes: List[Box],
                  min_frac: Optional[float] = None) -> List[Box]:
    """把「两行以上叠在一起」的文字框按**墨迹行缝**切开，一行一框。

    依据（`work/_lineblock.py`，50 评测案）：未键对的 536 颗具名元件里 **160 颗（29.9%）**
    近旁就有这种「高 ≥ 2.2× 页面中位字高」的框，而框上的位号读法全是 `C`/`U`/`1`/`UF3`
    这类残串 —— 两行注释（位号 + 型号/值）被当成**一行**去认，纵向被压扁后谁都读不对。
    这是检测层的合并错误，不是识别参数问题：正确做法是按行切开，一行一个候选。
    """
    if not boxes:
        return boxes
    hs = sorted(float(b[3]) - float(b[1]) for b in boxes)
    med = hs[len(hs) // 2] if hs else 8.0
    lim = (min_frac if min_frac is not None else LINESPLIT_H) * med
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    out: List[Box] = []
    for b in boxes:
        h = float(b[3]) - float(b[1])
        if h < lim or h < 6.0:
            out.append(b)
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in b]
        H, W = gray.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 3 or y1 - y0 < 4:
            out.append(b)
            continue
        sub = gray[y0:y1, x0:x1]
        ink = (sub < TEXT_INK_DARK).sum(axis=1)
        thr = max(1, int(0.02 * (x1 - x0)))
        bands: List[Tuple[int, int]] = []
        cur = None
        for i, v in enumerate(ink):
            if v > thr:
                cur = [i, i] if cur is None else [cur[0], i]
            elif cur is not None:
                bands.append((cur[0], cur[1]))
                cur = None
        if cur is not None:
            bands.append((cur[0], cur[1]))
        bands = [bb for bb in bands if bb[1] - bb[0] >= 3]
        if len(bands) < 2:
            out.append(b)
            continue
        if any((bb[1] - bb[0]) > 1.6 * med + 2 for bb in bands):
            out.append(b)      # 切出来一段就比一行还高 ⇒ 不是「叠行」，别动
            continue
        for i0, i1 in bands:
            out.append((float(b[0]), float(y0 + i0) - 1.0,
                        float(b[2]), float(y0 + i1) + 1.0))
        LAST_SPLIT[0] += 1
        LAST_SPLIT[1] += len(bands)
    return out


def read_stage1_texts(stage1_doc: dict) -> List[Box]:
    out: List[Box] = []
    for t in stage1_doc.get("texts") or []:
        b = t.get("bbox_img") if isinstance(t, dict) else t
        if b and len(b) == 4:
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    # §5.208.4(a)：`texts_wide`（`EDA_TEXTBOX_INK_WIDE`）是**只给读数用**的额外文字框，
    # 原样追加、不做几何去重（§5.207.2：框级去重会把冗余投票扔掉）。
    #
    # §5.208.5(4)：`EDA_TEXTBOX_INK_WIDE_MODE=replace` 改成**替换式** —— 把与 wide 框
    # IoU ≥ `STAGE2_WIDE_TOL`（默认 0.5）的旧框丢掉再放 wide 框，**框总数基本不变**，
    # 只让内容更准（本线在检测侧最后一个「不加框」的动作）。
    wide: List[Box] = []
    for t in stage1_doc.get("texts_wide") or []:
        b = t.get("bbox_img") if isinstance(t, dict) else t
        if b and len(b) == 4:
            wide.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    if wide:
        mode = os.environ.get("EDA_TEXTBOX_INK_WIDE_MODE", "append").strip().lower()
        if mode == "replace":
            tol = float(os.environ.get("STAGE2_WIDE_TOL", "0.5"))
            out = [b for b in out if not any(_iou(b, w) >= tol for w in wide)]
        out.extend(wide)
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
        self._rec2 = None
        self._tried2 = False
        self._rec3 = None
        self._tried3 = False
        self._recr = None
        self._triedr = False
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

    def recognizer2(self):
        """位号专用识别器（`STAGE2_CKPT_DESIG`）。用途：只换数字/名字的权重时把位号
        读数钉在旧权重上，观察「识别增益能否变现而不洗牌键」。未设置时返回 None。"""
        if self._tried2:
            return self._rec2
        self._tried2 = True
        p = (os.environ.get("STAGE2_CKPT_ROLES_CKPT") or os.environ.get("STAGE2_CKPT_DESIG") or "").strip()
        if not p or not Path(p).is_file():
            return None
        try:
            from textrec.infer import Recognizer
            self._rec2 = Recognizer.load(Path(p), device=self.device)
        except Exception:                                        # noqa: BLE001
            self._rec2 = None
        return self._rec2

    def recognizer_ens(self):
        """`STAGE2_CKPT_ENS` 的第二权重（装同一份代码、不同 epoch/语料），未设置返回 None。"""
        if self._tried3:
            return self._rec3
        self._tried3 = True
        p = (os.environ.get("STAGE2_CKPT_ENS") or "").strip()
        if not p or not Path(p).is_file():
            return None
        try:
            from textrec.infer import Recognizer
            self._rec3 = Recognizer.load(Path(p), device=self.device)
        except Exception:                                        # noqa: BLE001
            self._rec3 = None
        return self._rec3

    # ------------------------------------------------------------------
    def _pin_num_rot(self, rec, gray: np.ndarray, boxes: Sequence[Box],
                     crops: Dict[int, Box],
                     reads: List[Dict[str, Tuple[str, float]]]
                     ) -> Dict[int, Tuple[str, float]]:
        """`STAGE2_PIN_NUM_ROT`：直立读不出合法脚号的框，补一次 90°/270° 旋转读。

        只做**补读**：不改任何既有读数（结果单独返回、由 `read` 挂在框的 `pin_number_rot`
        键上），不改框、不动几何。触发（`fit_pinnum<=0`）与接受（`PINNUM_RE`）都是既有判据，
        这里只多花两次 CRNN 前向，判断权全部留给 `assign`。
        """
        from .assign import fit_pinnum                     # 既有判据，lazy 避免循环 import
        from stage1 import pincol as _pincol               # 既有尺寸窗（见 `PIN_NUM_ROT`）
        out: Dict[int, Tuple[str, float]] = {}
        hi, wi = gray.shape[:2]
        for i, cb in crops.items():
            b = boxes[i]
            if max(b[2] - b[0], b[3] - b[1]) > _pincol.NUM_MAX_W:
                continue                                  # 一排字并成的大框：旋转也读不成一个号
            cur = reads[i].get("pin_number")
            if cur is not None and fit_pinnum(cur[0]) > 0.0:
                continue                                  # 直立已经读出合法号，不碰
            x1 = max(0, int(round(cb[0]))); y1 = max(0, int(round(cb[1])))
            x2 = min(wi, int(round(cb[2]))); y2 = min(hi, int(round(cb[3])))
            if not (x2 > x1 and y2 > y1):
                continue
            sub = gray[y1:y2, x1:x2]
            best = None
            for k in (1, 3):                              # 90° / 270°：高宽互换的两个方向
                r = np.rot90(sub, k)
                got = rec.read_many(
                    r, [((0.0, 0.0, float(r.shape[1]), float(r.shape[0])), "pin_number")])
                txt, conf = got[0]
                t = (txt or "").strip()
                if not t or fit_pinnum(t) <= 0.0:
                    continue
                if best is None or float(conf) > best[1]:
                    best = (t, float(conf))
            if best is not None:
                out[i] = best
        return out

    # ------------------------------------------------------------------
    def _gen_ocr(self, gray: np.ndarray, boxes: Sequence[Box],
                 keys: Sequence[Tuple[int, str]], reads: List[Dict]) -> None:
        """`STAGE2_GEN_OCR`：通用 OCR 第三通道，逐 `(框, 角色)` 决定是否替换 CRNN 读法。

        同一张裁片只送一次（PP-OCR 识别与角色无关），归一化与语法判据按角色各做一遍。
        只改 `reads`，不碰框与几何；`STAGE2_DBG_GENOCR=1` 时打印并落 JSONL。
        """
        try:
            from . import genocr
        except Exception:                                      # noqa: BLE001
            return
        want = [k for k in keys if k[1] in GEN_OCR_ROLES]
        if not want or not genocr.available():
            return
        idx = sorted({i for i, _r in want})
        got = genocr.read_many(gray, [(boxes[i], "gen") for i in idx])
        by = dict(zip(idx, got))
        for i, r in want:
            cur = reads[i].get(r)
            old = (cur[0] if cur else "") or ""
            txt, conf = by.get(i, ("", 0.0))
            new = genocr.fix_confusions(txt, r)
            if not new or new == old:
                continue
            fit_old = _role_fit(r, old)
            if fit_old <= 0.0:
                if _role_fit(r, new) <= 0.0:
                    continue
            elif GEN_OCR_LEVEL < 2 or len(new) <= len(old) or len(new) > len(old) + 2:
                continue
            reads[i][r] = (new, float(conf))
            if GEN_OCR_DBG:
                print("[genocr] %-10s %r -> %r (conf %.3f)" % (r, old, new, float(conf)))
                if GEN_OCR_LOG:
                    try:
                        import json
                        with open(GEN_OCR_LOG, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(
                                {"role": r, "old": old, "new": new,
                                 "conf": round(float(conf), 4),
                                 "box": [float(v) for v in boxes[i]]},
                                ensure_ascii=False) + "\n")
                    except Exception:                      # noqa: BLE001
                        pass

    # ------------------------------------------------------------------
    def read(self, bgr: np.ndarray, boxes: Sequence[Box],
             roles: Sequence[str] = ROLES) -> List[Dict]:
        """-> [{"bbox": Box, "reads": {role: (text, conf)}}, ...]"""
        boxes = [tuple(float(v) for v in b) for b in boxes]
        rec = self.recognizer()
        if rec is None or not boxes:
            out0 = [{"bbox": b, "reads": {}} for b in boxes]
            if TEXT_INK:
                for i, r in enumerate(out0):
                    r["cfrac"] = colored_frac(bgr, boxes[i])
            return out0

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        reads: List[Dict[str, Tuple[str, float]]] = [dict() for _ in boxes]
        rot_crops: Dict[int, Box] = {}
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
            if PIN_NUM_ROT and "pin_number" in roles:
                # 旋转补读的裁片必须与号轮**完全一致**（切描边 + `PINNUM_PAD_FRAC_Y`），
                # 否则量到的增益里会混进「换了裁片」这一项。
                rot_crops[i] = (cb_ty if (cb_t is not None and "pin_number" in TIGHT_ROLES)
                                else cb)
            for r in roles:
                if cb_t is not None and r in TIGHT_ROLES:
                    items.append((cb_ty if r == "pin_number" else cb_t, r))
                else:
                    items.append((cb, r))
                keys.append((i, r))
        rec2 = self.recognizer2()
        frozen = set(t.strip() for t in (os.environ.get("STAGE2_CKPT_ROLES") or "designator").split(",") if t.strip())
        if rec2 is None or not any(r in frozen for _i, r in keys):
            passes = [(rec, items, keys)]
        else:
            sel = [k for k in range(len(items)) if keys[k][1] in frozen]
            oth = [k for k in range(len(items)) if keys[k][1] not in frozen]
            passes = [(rec2, [items[k] for k in sel], [keys[k] for k in sel]),
                      (rec, [items[k] for k in oth], [keys[k] for k in oth])]
        for rr, its, kys in passes:
            for s in range(0, len(its), self.chunk):
                part = its[s:s + self.chunk]
                got = rr.read_many(gray, part)
                for (i, r), (txt, conf) in zip(kys[s:s + self.chunk], got):
                    reads[i][r] = (txt, float(conf))
        rec3 = self.recognizer_ens()
        if rec3 is not None:
            # `STAGE2_CKPT_ENS`：第二个权重独立重读同一批裁片，逐 (框,角色) 按
            # 「置信度领先 ENS_MARGIN 才换」合并。两个模型误差不相关，换的只是
            # 一方明显更确定的那些格子。
            for s in range(0, len(items), self.chunk):
                got = rec3.read_many(gray, items[s:s + self.chunk])
                for (i, r), (txt, conf) in zip(keys[s:s + self.chunk], got):
                    cur = reads[i].get(r)
                    if cur is None or r not in ENS_ROLES:
                        continue
                    if float(conf) > cur[1] + ENS_MARGIN:
                        if ENS_DBG:
                            _ENS.append((cur[0], txt, cur[1], float(conf)))
                        reads[i][r] = (txt, float(conf))
        if GEN_OCR_LEVEL > 0:
            self._gen_ocr(gray, boxes, keys, reads)
        if VALUE_DOT:
            self._fix_dot(gray, boxes, roles, reads)
        if VALUE_TAIL:
            # ⚠ 必须排在 `_length_fix` **之前**：长度修补会把 `22u` 改成 `22um`/`220u`（错字），
            # 之后 `_TAILSRC_RE` 就不再匹配 `数字+单位` 形状，`canon_value` 再把 `m` 削掉 ⇒
            # 白忙一场（§5.491 第一版就是 case13 C6 没补上）。
            self._fix_tail(gray, boxes, roles, reads)
        if READ_LENFIX:
            self._length_fix(gray, boxes, roles, rec, reads)
        if VALUE_MU:
            self._fix_units(bgr, boxes, roles, reads)
        # 放最后：触发要看**最终**的直立引脚号读数（`_length_fix` 等已经改完）。
        rot = self._pin_num_rot(rec, gray, boxes, rot_crops, reads) if PIN_NUM_ROT else {}
        return [{"bbox": boxes[i], "reads": reads[i],
                 **({"pin_number_rot": rot[i]} if i in rot else {}),
                 **({"cfrac": colored_frac(bgr, boxes[i])} if TEXT_INK else {})}
                for i in range(len(boxes))]

    # ------------------------------------------------------------------
    def _length_fix(self, gray: np.ndarray, boxes: Sequence[Box], roles: Sequence[str],
                    rec, reads: List[Dict]) -> None:
        """`STAGE2_READ_LENFIX`（默认 1）：**用「图上有几个字形」约束解码**，救回被 CTC 吞掉的字。

        病（`open_GT`，2026-10 实测）：贪心 CTC 解码会**整字吞掉**窄字或夹在两个字中间的字 ——
        `TIM` 读成 `TM`（case12 `U1` 的 `pin_1`）、`TP4333` 读成 `TP433`（case11 `U2` 的型号）。
        换权重无用（7 个 ckpt 一致读错），补白无用（`_padscan` 只会引入 `T→1` 混淆）。
        但**图里有几个字是能数的**：列投影的连通段数就是字数，与模型无关。

        做法：贪心串的字数 ≠ 字形段数时，取同一裁片的 `log_softmax` 做 n-best beam 解码，
        在「字数正好等于段数」的候选里挑分最高的；再乘两道图像侧的罚：
          * 窄段（宽 ≤ `LENFIX_NARROW`）配宽字母 ⇒ 罚（`W`/`M` 不可能只有 3px 宽）；
          * 相邻重复字母 ⇒ 罚（这套图的文字里没有 `TT`/`MM` 这类叠字，beam 里却大量产生）。
        只有候选串和原读法不同才替换；仍保留原置信度（置信度是发射概率的几何平均，
        换了字符串以后它的含义不变，只是不再对应那条路径）。
        """
        if rec is None:
            return
        want = [r for r in roles if r in LENFIX_ROLES]
        if not want:
            return
        H, W = gray.shape[:2]
        for i, b in enumerate(boxes):
            for r in want:
                cur = reads[i].get(r)
                if cur is None or not cur[0]:
                    continue
                txt, conf = cur[0], float(cur[1])
                cb = pad_box(b, self.pad_min, self.pad_frac,
                             pad_min_y=self.pad_min_y, pad_frac_y=self.pad_frac_y)
                x0, y0, x1, y1 = (int(round(v)) for v in cb)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W, x1), min(H, y1)
                g = gray[y0:y1, x0:x1]
                if g.size == 0 or g.shape[1] < 4:
                    continue
                dark = g < max(1.0, float(g.mean()) * LENFIX_INK)
                runs = [t for t in _glyph_runs(dark) if (t[1] - t[0]) >= LENFIX_MINW]
                n = len(runs)
                if n < 2 or n > LENFIX_MAXN or n == len(txt):
                    continue
                if abs(n - len(txt)) > LENFIX_MAXDIFF:
                    continue
                try:
                    lp = rec.logits_many(gray, [(tuple(cb), r)])[0]
                except Exception:                                    # noqa: BLE001
                    continue
                lp = np.asarray(lp)
                itos = getattr(getattr(rec, "charset", None), "itos", None) or {}
                base_lp = _ctc_logprob(lp, txt, itos)
                hyps = rec._beam(lp, k=LENFIX_K, beam=LENFIX_BEAM)
                # 只接受「在贪心串里插字」得到的候选（不允许改字/删字）：
                # 吞字是**插入**型错误，改字是另一类问题（`_BEAM` 里它常常更差）。
                cands = []
                for t, _c in hyps:
                    if len(t) != n or t == txt:
                        continue
                    _i = 0
                    for _ch in t:
                        if _i < len(txt) and _ch == txt[_i]:
                            _i += 1
                    if _i == len(txt):
                        cands.append(t)
                if not cands:
                    continue
                _best, _best_lp = None, -1e18
                for t in cands:
                    clp = _ctc_logprob(lp, t, itos)
                    if clp < base_lp - LENFIX_MARGIN:
                        continue
                    if clp > _best_lp:
                        _best, _best_lp = t, clp
                if _best is None:
                    continue
                if LENFIX_DBG:
                    print("[lenfix] %-10s %r(%d) -> %r(%d) n=%d d=%.2f"
                          % (r, txt, len(txt), _best, len(_best), n, _best_lp - base_lp))
                reads[i][r] = (_best, conf)

    # ------------------------------------------------------------------
    def _fix_units(self, bgr: np.ndarray, boxes: Sequence[Box], roles: Sequence[str],
                   reads: List[Dict]) -> None:
        """`STAGE2_VALUE_MU`：用字形墨迹改单位字母（详见上面的长注释）。"""
        want = [r for r in roles if r in VALUE_UNIT_ROLES]
        if not want:
            return
        for i, b in enumerate(boxes):
            for r in want:
                cur = reads[i].get(r)
                if cur is None or not cur[0]:
                    continue
                new = fix_unit_text(bgr, cur[0], b)
                if new != cur[0]:
                    _UNIT_STATS["changed"] = _UNIT_STATS.get("changed", 0) + 1
                    reads[i][r] = (new, cur[1])

    # ------------------------------------------------------------------
    def _fix_tail(self, gray: np.ndarray, boxes: Sequence[Box], roles: Sequence[str],
                  reads: List[Dict]) -> None:
        """`STAGE2_VALUE_TAIL`：给「图上有、读法里没有」的值尾字母补上 `f`/`F`（详见上面的长注释）。"""
        want = [r for r in roles if r in VALUE_UNIT_ROLES]
        if not want:
            return
        H, W = gray.shape[:2]
        for i, b in enumerate(boxes):
            for r in want:
                cur = reads[i].get(r)
                if cur is None or not cur[0]:
                    continue
                txt, conf = cur[0].strip(), float(cur[1])
                if not _TAILSRC_RE.match(txt):
                    continue
                cb = pad_box(b, self.pad_min, self.pad_frac,
                             pad_min_y=self.pad_min_y, pad_frac_y=self.pad_frac_y)
                x0, y0, x1, y1 = (int(round(v)) for v in cb)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W, x1), min(H, y1)
                g = gray[y0:y1, x0:x1]
                if g.size == 0 or g.shape[1] < 4:
                    continue
                dark = g < max(1.0, float(g.mean()) * LENFIX_INK)
                runs = [t for t in _glyph_runs(dark) if (t[1] - t[0]) >= LENFIX_MINW]
                if len(runs) != len(txt) + 1:
                    continue
                a, bb2 = runs[-1]
                seg = dark[:, a:bb2]
                rows = np.nonzero(seg.any(1))[0]
                if len(rows) < 4 or (rows[-1] - rows[0] + 1) < 0.6 * dark.shape[0]:
                    continue
                wid = [t[1] - t[0] for t in runs]
                wd = float(np.median(wid[:-1])) if len(wid) > 1 else 0.0
                if wd <= 0.0:
                    continue
                letter = "f" if float(wid[-1]) < VALUE_TAIL_W * wd else "F"
                new = txt + letter
                if not _TAIL_DONE_RE.match(new):
                    continue
                _UNIT_STATS["tail"] = _UNIT_STATS.get("tail", 0) + 1
                reads[i][r] = (new, conf)

    # ------------------------------------------------------------------
    def _fix_dot(self, gray: np.ndarray, boxes: Sequence[Box], roles: Sequence[str],
                 reads: List[Dict]) -> None:
        """`STAGE2_VALUE_DOT`：读法里丢了小数点时按字形段补回（§5.500）。

        病：`case13 C8` 的 GT 是 `2.7nf`，我们读成 `27n`。裁剪剖面的字形段是
        `w=10 / w=2(只占底部两行) / w=10 / w=9 / w=7` —— 那条又窄又贴底的段就是小数点。
        判据：段数 = 读数长度 + 2、其中**恰好一条**宽度 ≤ `VALUE_DOT_W` × 段宽中位数、
        且该段只在裁剪高度底部 `VALUE_DOT_BOT` 以内有墨。补完仍要过 `_TAIL_DONE_RE`。
        """
        want = [r for r in roles if r in VALUE_UNIT_ROLES]
        if not want:
            return
        H, W = gray.shape[:2]
        for i, b in enumerate(boxes):
            for r in want:
                cur = reads[i].get(r)
                if cur is None or not cur[0]:
                    continue
                txt, conf = cur[0].strip(), float(cur[1])
                m = _DOTSRC_RE.match(txt)
                if not m:
                    continue
                cb = pad_box(b, self.pad_min, self.pad_frac,
                             pad_min_y=self.pad_min_y, pad_frac_y=self.pad_frac_y)
                x0, y0, x1, y1 = (int(round(v)) for v in cb)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W, x1), min(H, y1)
                g = gray[y0:y1, x0:x1]
                if g.size == 0 or g.shape[1] < 6:
                    continue
                dark = g < max(1.0, float(g.mean()) * LENFIX_INK)
                runs = [t for t in _glyph_runs(dark) if (t[1] - t[0]) >= 1]
                if len(runs) != len(txt) + 2:
                    continue
                wid = [t[1] - t[0] for t in runs]
                wd = float(np.median(wid))
                if wd <= 2.0:
                    continue
                cand = [k for k, w in enumerate(wid) if w <= VALUE_DOT_W * wd]
                if len(cand) != 1:
                    continue
                k = cand[0]
                seg = dark[:, runs[k][0]:runs[k][1]]
                rows = np.nonzero(seg.any(1))[0]
                if len(rows) == 0 or float(rows[0]) < VALUE_DOT_BOT * dark.shape[0]:
                    continue
                new = m.group("a") + "." + m.group("b") + m.group("u")
                if not _TAIL_DONE_RE.match(new):
                    continue
                _UNIT_STATS["dot"] = _UNIT_STATS.get("dot", 0) + 1
                reads[i][r] = (new, conf)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        self.recognizer()
        return "TextEngine(ckpt=%s, %s)" % (self.ckpt, self.note or "ok")
