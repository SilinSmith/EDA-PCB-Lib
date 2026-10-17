# -*- coding: utf-8 -*-
"""用 VLM 给 stage2 的预测补上两件从没读对过的东西：**IC 位号** 与 **IC 引脚号/名**。

为什么必须一起做（实测）：

* `pin_f1`（权重 0.25）的 `flat()` 里有 `if comp not in box_comps: continue`，
  而 `box_comps` 取自 **GT** 的 box 元件键。我们 box 键命中率只有 **0.309**
  ⇒ 引脚读得再准，键不对就一分不得。第一次只补引脚：`tp` 69 → **54**，总分倒退。
* VLM 读位号（裁片 = 预测 bbox 外扩 60，坐标翻正后）**0.65 与 GT 键一致**，
  读出非空 0.75，0.16 s/个。
* VLM 逐引脚读（每个已检出引脚点裁一个不对称窗口、打红圈、8 个拼网格）
  **编号 0.9697 / 编号+名称 0.8636**，0.81 s/元件。

坐标坑（踩过一次）：预测的 `bbox` / `point` 是**左下角 y 向上**，与 GT 一致；
拿去索引图像（y 向下）前必须 `y_img = H - y`，否则裁片上下镜像、读出来全是垃圾。

220 张官方公开集实测（`work\\stage2_v17` 为输入）：

    基线        23.9559   component F1 0.5225  pin F1 0.0474 tp  69
    VLM 补强后  24.5211   component F1 0.5227  pin F1 0.0692 tp 166   (+0.565)
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from . import client

# 与 `stage2/assemble.py:89` 同款判据：值必须含字母 / Ω / µ（GT 里 r/c 的值一个纯数字都没有）
VALUE_LIKE = re.compile(r"[A-Za-z\u03A9\u2126\u00B5\u03BC]")

IC = ("box", "ic", "boxic")
VAL_TYPES = ("r", "c", "l", "fuse", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin")
COLORS = {"red": (0, 0, 235), "magenta": (235, 0, 235), "cyan": (235, 235, 0),
          "blue": (235, 0, 0), "green": (0, 180, 0)}

DESIG_PROMPT = (
    "这是电路原理图里的**一个元器件**（IC / 模块）。\n"
    "在该元器件外框**之外**（通常紧贴右上角或左上角）印着它的**位号**，"
    "形如「字母前缀 + 数字」，例如 U1、U5、IC2、J3、SW1。\n"
    "只回那个位号本身，不要解释、不要换行、不要任何其它字符。\n"
    "如果图里印的是 `U?`、`J?` 这种带问号的未标注形式，就原样回 `U?`。\n"
    "完全找不到就回 NONE。")

# 分立元件的「值」。四条硬约束都是踩出来的（见 README）：
#   ① 裁片必须够大（`value_margin` 默认 90）—— 45px 时会把 `100nF` 的尾巴 `F` 切掉，
#      模型于是把 `100n` 规范化成 `10nF`，逐字命中率从 0.94 掉到 0.79；
#   ② 必须明说「原样照抄、不要规范化」（`100nF` 不许写成 `0.1uF`）；
#   ③ 必须明说「不要读容差/耐压行」（否则常把 `10%`、`16V` 当成值）；
#   ④ 必须明说「只报画面中心那个元件的」，否则会读到旁边那颗的位号/值。
VALUE_PROMPT = (
    "这是电路原理图里**一个分立元件**（电阻 R / 电容 C / 电感 L / 晶振 Y / 保险丝 F）的局部放大图。\n"
    "图里会有好几个类似的元件挨在一起，每个旁边印两行字：\n"
    "  第 1 行 = **位号**（短的「字母+数字」，如 C5、R12、L1）；\n"
    "  第 2 行 = **值**（元件参数，如 10K、4.7uF、0.1uF、15pF、100nF）。\n"
    "先找到**画面中心**那个元件，只报**它自己**的位号与值，不要报旁边元件的。\n"
    "值的字符**原样照抄，一个字符都不要改、不要规范化**：\n"
    "图上写 `100nF` 就写 `100nF`（不要写成 `0.1uF`，也不要写成 `10nF`）；\n"
    "图上写 `0.1uF` 就写 `0.1uF`。\n"
    "**绝对不要**把第三行（容差/耐压，如 `10%`、`16V`、`50V`）当成值。\n"
    "只回一行 `位号=值`。找不到值就回 `位号=NONE`；什么都没有就回 NONE。\n"
    "不要解释、不要换行。")

# IC 的器件型号（写进 `Name`）。官方 PDF 第 8 页：「'ic'：比较 'Name'」，
# 而 `box` 现在只有 2/996 个 tp —— 我们的 `Name` 全是引脚名/器件值（`HS96L03W2C03 -> VCC1`）。
# `_vlmname.py` 实测：margin 8/scale 4 → 0.2333；margin 40/scale 2 → 0.4333；
# margin 80/scale 2 → **0.4667（任一候选）**；margin 140 与 80 持平。
NAME_PROMPT = (
    "这是电路原理图里**一个芯片/模块/连接器**的局部放大图。\n"
    "它的外框旁边印着两样东西：**位号**（如 U1、J3、CN2）与**器件型号 / part number**"
    "（如 STM32F103C8T6、AMS1117-3.3、HC-PZ254-11.5L-1x3P、TYPE-C 16PIN）。\n"
    "请只回**器件型号本身**，不要回位号，不要回引脚名（VDD/VCC/GND），不要回参数值。\n"
    "型号里的字符**原样照抄**，不要补全、不要改写、不要加空格。\n"
    "如果型号分好几行印（如 `MAX14571` / `MAX14572`），每行回一个。\n"
    "只回型号文本，一行一个，不要解释、不要任何其它字符。\n"
    "完全找不到型号就回 NONE。")

PINS_PROMPT = (
    "这是一张由多个小格拼成的图，每一格是电路原理图里**一个引脚**附近的放大图。\n"
    "每一格里有一个**红色圆圈**，圈住的就是这一格要读的那个引脚。\n"
    "请对每一格读出两件事：\n"
    "1) 编号：紧贴该引脚、印在元器件外框**之外**的那串数字（引脚编号）\n"
    "2) 名称：元器件外框**内部**、与该引脚同一行的信号名文字（如 VDD、GND、SPI_MOSI）\n"
    "只回**一行**，格式 `格号:编号:名称`，多格之间用英文逗号分隔。例如：\n"
    "0:12:VDD,1:13:GND,2:44:SPI_CLK\n"
    "不要 JSON、不要换行、不要解释、不要任何其它字符。")

DESIG_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")
# `desig_fallback` 补读的那些元件里绝大多数是分立元件，而 `DESIG_PROMPT` 开头就写着
# 「**一个元器件**（IC / 模块）」，模型看到一颗电容就直接回 NONE —— 实测 `0001` 的
# `C6`/`C16`/`C8`/`R1`/`R2` 全回 NONE（放大 2x 也一样），而图里 `C6 4.7uF` 印得清清楚楚。
# 换成按分立元件写的提示词（沿用 `VALUE_PROMPT` 里验过的四条约束）。
PASSIVE_TYPES = frozenset((
    "r", "c", "l", "fuse", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin",
    "d", "led", "testpoint", "tp", "switch", "bjt", "mosfet", "pin", "amp", "amp_3pin",
    "amp_5pin", "amp_7pin", "buzzer", "esd", "battary", "jumper", "motor", "dc",
    "transfomer", "block", "opt", "seg", "ant", "circle_header", "other",
    "mosfet_npn", "mosfet_pnp", "bjt_npn", "bjt_pnp",
))
DESIG_PROMPT_PASSIVE = (
    "这是电路原理图里**一个分立元件**（电阻 / 电容 / 电感 / 二极管 / 晶振 / 保险丝 / "
    "测试点 / 三极管 / 开关）的局部放大图。\n"
    "图里往往有好几个长得很像的元件挨在一起，每个旁边印着它的**位号**"
    "（短的「字母前缀 + 数字」，如 C5、R12、L1、D3、Y1、TP1、Q2）。\n"
    "位号旁边通常还有一两行**参数/容差**（如 `4.7uF`、`10K`、`0603`、`16V`、`10%`、`50V`），"
    "**不要把参数当成位号**。\n"
    "请只找**画面正中**那个元件自己的位号，绝对不要报旁边元件的。\n"
    "只回那个位号本身（例如 `C5`），不要解释、不要换行、不要任何其它字符。\n"
    "完全找不到就回 NONE。")
# 位号没读出来时 `assemble.py` 会兜底成 `pNNN`。这种键**永远不可能**和 GT 对上
# （`unkeyed_match='gt'` 只放行 GT 自己的 `∅NNN`），所以拿任何别的信息去换都是有赚无赔。
FKEY_RE = re.compile(r"^p\d{3,4}$")


def norm(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s or "")).upper()


def crop_dir(bgr, pt, center, outer, inner, cross, cross_perp=None, along_pad=None,
             dir_uv=None):
    """不对称窗口：沿「远离器件中心」方向外扩 `outer`（拿引脚编号，印在外框外），
    朝中心内伸 `inner`（拿引脚名称，印在外框内），**垂直于该方向**两边各留 `cross_perp`。

    ★ 为什么把 `cross` 拆成「垂直半宽」和「沿线余量」——
    旧实现用同一个 `cross`（默认 58→按器件缩放后 24）**同时**加在 x 和 y 上，于是窗口
    在**垂直方向总高 = 2×cross ≈ 48px**。而 QFP 的引脚间距常常只有 ~20px，
    48px 的窗口**必然把上下相邻的引脚行一起框进来** ⇒ 模型看到 2~3 行引脚编号/名称，
    报回来的名字是**别的行**的（10GT 引脚错的 55 处里 41 处正是这种 name_mismatch：
    位置对、名字错，而 `pin` 权重占 0.25）。几何实测（`work\\tools\\diag\\_pinwin.py`，
    10GT 全部带引脚的元件、124 个引脚，用 GT 引脚点当「行」的代理）：
    窗口里混进**别的引脚**的平均个数 **旧 2.09 → 新 1.31**，有混进的引脚
    **86.3% → 72.6%**；渲染对照见 `work\\preview\\_pincrop_new.png`
    （旧格 2~5 排编号，新格 1~2 排）。

    **退化保证**：`cross_perp=None` 时**逐位等于旧行为**（x、y 两边各留 `cross`），
    这是 A/B 里 `pin_pitch_frac=0` 那一臂依赖的性质。
    """
    h, w = bgr.shape[:2]
    px, py = float(pt[0]), float(pt[1])
    if dir_uv is not None:
        # ★★ 真正的病根：`u` 原来取「引脚 → **包围盒中心**」的方向。对一颗
        # 177×317 的窄高 IC，靠下边的引脚给出的 `u` 是 **斜的**（实测 (-101,-118)），
        # 于是「沿引脚」与「垂直引脚」都变成斜向，窗口的轴对齐包围盒在**两个轴上都胖**，
        # 垂直方向收窄几乎不起作用（实测 old=(84,78) → new=(83,75)，白改）。
        # 正解是让 `u` = **引脚所在那条边的内法向**（轴对齐）：直角矩形本体上，
        # 离引脚最近的那条边就是他所在的边。偏离量由 `_pinwin.py` 量化。
        ux, uy = float(dir_uv[0]), float(dir_uv[1])
        n = (ux * ux + uy * uy) ** 0.5 or 1.0
        ux, uy = ux / n, uy / n
    else:
        dx, dy = float(center[0]) - px, float(center[1]) - py
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-3:
            dx, dy, n = 1.0, 0.0, 1.0
        ux, uy = dx / n, dy / n
    if cross_perp is None:                       # —— 旧行为，逐位不变 ——
        ax0, ay0 = px - ux * outer, py - uy * outer
        ax1, ay1 = px + ux * inner, py + uy * inner
        x0 = max(0, int(min(ax0, ax1) - cross)); x1 = min(w, int(max(ax0, ax1) + cross))
        y0 = max(0, int(min(ay0, ay1) - cross)); y1 = min(h, int(max(ay0, ay1) + cross))
        if x1 <= x0 or y1 <= y0:
            return None
        return bgr[y0:y1, x0:x1].copy(), (px - x0, py - y0)
    cp = float(cross_perp)
    # 沿线方向的余量**保持旧值 `cross`**（不缩），于是这次改动只作用于「垂直于引脚的
    # 那个方向」—— 它只会丢掉相邻引脚行，**绝不会**丢掉本引脚沿线上的任何东西
    # （编号在外、名称在内，都在 `outer`/`inner` 覆盖范围内，且余量与旧版逐位相同）。
    ap = cross if along_pad is None else max(0.0, float(along_pad))
    ax0, ay0 = px - ux * (outer + ap), py - uy * (outer + ap)
    ax1, ay1 = px + ux * (inner + ap), py + uy * (inner + ap)
    pxx, pyy = -uy * cp, ux * cp
    xs = [ax0 + pxx, ax0 - pxx, ax1 + pxx, ax1 - pxx]
    ys = [ay0 + pyy, ay0 - pyy, ay1 + pyy, ay1 - pyy]
    x0 = max(0, int(min(xs))); x1 = min(w, int(max(xs)))
    y0 = max(0, int(min(ys))); y1 = min(h, int(max(ys)))
    # ★ 单调性保证：新框永远取「新 ∩ 旧」。
    # 斜向引脚（`u` 不平行于坐标轴）时，旋转矩形的**轴对齐包围盒**可能在某个轴上反而
    # 比旧框大（旧框在该轴上只有 `2×cross`，新框有 `2×|u⊥分量|×cp` 加上沿线投影）。
    # 实测 297 个引脚里有 6 个会这样。取交集把这 6 个钉回旧尺寸，
    # 于是「新窗口在两个轴上都不大于旧窗口」成为恒等式，收紧是**可证**的。
    ox0 = max(0, int(min(px - ux * outer, px + ux * inner) - cross))
    ox1 = min(w, int(max(px - ux * outer, px + ux * inner) + cross))
    oy0 = max(0, int(min(py - uy * outer, py + uy * inner) - cross))
    oy1 = min(h, int(max(py - uy * outer, py + uy * inner) + cross))
    x0, x1 = max(x0, ox0), min(x1, ox1)
    y0, y1 = max(y0, oy0), min(y1, oy1)
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1].copy(), (px - x0, py - y0)


def adaptive_window(bbox, inner=78.0, cross=58.0, outer=34.0, ref=762.0):
    """小 IC（4 脚那种 ~84x84）在固定窗口下会把整颗连同走线框进一格、标记淹没在
    连线里（实测 `2/U1` 只有 0.25、`10/U1` 0.44）。按器件尺寸缩放窗口。"""
    w = abs(float(bbox[2]) - float(bbox[0])); h = abs(float(bbox[3]) - float(bbox[1]))
    k = max(0.35, min(1.0, max(w, h) / ref))
    return max(12.0, outer * k), max(30.0, inner * k), max(20.0, cross * k)


def mark(im, pt, r=4, t=2, color=(0, 0, 235)):
    """标记圈越小越好：`mr=4..5` 时 编号 0.9615 / 编号+名称 0.8846，`mr=9` 掉到
    0.846 / 0.615，`mr>=14` 直接盖住引脚编号。"""
    cv2.circle(im, (int(round(pt[0])), int(round(pt[1]))), int(r), color, int(t),
               cv2.LINE_AA)
    return im


def tile(crops, cols=4, side=150, pad=8, label_h=26):
    import numpy as np
    if not crops:
        return None
    rows = (len(crops) + cols - 1) // cols
    cw = side + pad * 2
    ch = side + pad * 2 + label_h
    out = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for i, c in enumerate(crops):
        r, q = divmod(i, cols)
        if c is None:
            continue
        hh, ww = c.shape[:2]
        s = min(side / max(1, ww), side / max(1, hh))
        im = cv2.resize(c, (max(1, int(ww * s)), max(1, int(hh * s))),
                        interpolation=cv2.INTER_CUBIC)
        ih, iw = im.shape[:2]
        y = r * ch + label_h + pad + max(0, (side - ih) // 2)
        x = q * cw + pad + max(0, (side - iw) // 2)
        out[y:y + ih, x:x + iw] = im
        cv2.putText(out, "#%d" % i, (q * cw + pad, r * ch + label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.rectangle(out, (q * cw, r * ch), (q * cw + cw - 1, r * ch + ch - 1),
                      (200, 200, 200), 1)
    return out


def parse_grid(txt):
    """`0:12:VDD,1:13:GND` -> {0: ('12','VDD'), ...}；也容忍 JSON 数组。"""
    out = {}
    t = client.strip_fence((txt or "").strip())
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            for it in json.loads(m.group(0)):
                if isinstance(it, dict):
                    i = it.get("id", it.get("index"))
                    out[int(i)] = (str(it.get("number") or ""), str(it.get("name") or ""))
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    for seg in re.split(r"[,\n;]+", t):
        mm = re.match(r"\s*[\[\(]?\s*#?(\d{1,3})\s*[:：]\s*([^:：]*?)\s*[:：]\s*(.*?)\s*[\]\)]?\s*$",
                      seg)
        if mm:
            out[int(mm.group(1))] = (mm.group(2), mm.group(3))
        else:
            mm = re.match(r"\s*#?(\d{1,3})\s*[:：]\s*(.*?)\s*$", seg)
            if mm:
                out[int(mm.group(1))] = ("", mm.group(2))
    return out


def read_designator(bgr, ib, opts, stat, ctype=""):
    """读位号。先用 `desig_margin`，读不到（含 `NONE`/`U?`/乱码）就把外扩翻倍再试一次 ——
    `-Datasheet` 拆页那种整页图里元件很小，60px 常常框不到位号。

    `ctype` 用来换提示词：`DESIG_PROMPT` 是为 IC 写的，套到电容/电阻上模型会回 NONE。"""
    h, w = bgr.shape[:2]
    prompt = DESIG_PROMPT_PASSIVE if ctype in PASSIVE_TYPES else DESIG_PROMPT
    txt = ""
    for mult in (1.0, 2.2):
        m = opts["desig_margin"] * mult
        x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
        x1 = min(w, int(ib[2] + m)); y1 = min(h, int(ib[3] + m))
        if x1 <= x0 or y1 <= y0:
            continue
        txt, _dt = client.ask(bgr[y0:y1, x0:x1], prompt, model=opts["model"])
        stat["desig"] += 1
        lines = (txt or "").strip().splitlines()
        t = re.sub(r"[^0-9A-Za-z?]", "", lines[0]) if lines else ""
        if t and DESIG_RE.match(t):
            return t
    return None


def read_value(bgr, ib, opts, stat):
    """读分立元件的「值」。返回 (位号, 值) 或 (None, None)。

    裁片外扩默认 90px —— 这个数不是拍的：45 / 55 / 65 / 80 / 100 / 140 / 200 扫下来
    逐字命中率是 0.836 / 0.875 / 0.923 / 0.942 / 0.933 / 0.894 / 0.827（120 样本），
    而 25px（旧值）只有 0.212 —— 因为 25px 会把 `100nF` 尾巴的 `F` 切掉。"""
    h, w = bgr.shape[:2]
    m = opts["value_margin"]
    x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
    x1 = min(w, int(ib[2] + m)); y1 = min(h, int(ib[3] + m))
    if x1 <= x0 or y1 <= y0:
        return None, None
    im = client.upscale(bgr[y0:y1, x0:x1], opts["value_scale"])
    txt, _dt = client.ask(im, VALUE_PROMPT, model=opts["model"])
    stat["value"] += 1
    lines = (txt or "").strip().splitlines()
    t = lines[0].strip() if lines else ""
    if not t or "=" not in t:
        return None, None
    lhs, rhs = t.split("=", 1)
    key = re.sub(r"[^0-9A-Za-z]", "", lhs)
    val = rhs.strip()
    val = re.sub(r"[\r\n\t]", "", val)
    if not val or val.upper() == "NONE":
        return (key or None), None
    return (key or None), val


# ---- 「请求合并」第一个落地通道：值通道拼图 ------------------------------
# 现状：220 张里值通道要发 3492 次请求（一格一次），是全管线第二大开销
# （位号 4094 / 值 3492 / 引脚网格 1051 / 型号 1021 ≈ 9658 次）。
# 引脚通道本来就是拼图的（4025 个引脚 → 1051 次），值通道跟着做。
# ⚠️ 拼图会让**每格分辨率下降**，而值通道对裁片尺寸极其敏感
# （margin 25 → 0.212 vs margin 80 → 0.9423），所以默认 `value_batch=1`（关），
# 必须先用 `fast.py` / `_vlmab2.py` 做 A/B 证明每格精度不掉再打开。
VALUE_PROMPT_GRID = (
    "这是一张由若干**电路原理图局部放大图**拼成的网格，每格左上角有编号 `#0` `#1` …。\n"
    "每一格里是**一个分立元件**（电阻/电容/电感/二极管/晶振/保险丝/测试点）和它旁边的标注。\n"
    "请**逐格**报告该格的 `位号=值`，一行一格，格式固定为：\n"
    "  #0: C12=100nF\n"
    "  #1: R5=10K\n"
    "规则（与只看一格时完全相同）：\n"
    "1. 值**原样照抄**，绝不换算、绝不规范化 —— 图上写 `100nF` 就写 `100nF`，"
    "不要写成 `0.1uF`，更不要写成 `10nF`。\n"
    "2. 绝对不要把容差/耐压/封装行（`10%` `16V` `50V` `0603` `1%`）当成值。\n"
    "3. 只报**该格画面正中**那个元件自己的位号与值，不要报邻格的。\n"
    "4. 该格看不清、或没有位号/值，就写 `#3: NONE`。\n"
    "只输出这些行，不要任何解释。"
)


def parse_valgrid(txt):
    """`#0: C12=100nF` -> {0: ('C12', '100nF')}；`NONE` -> `(None, None)`。"""
    out = {}
    t = client.strip_fence((txt or "").strip())
    for seg in re.split(r"[\n;]+", t):
        mm = re.match(r"\s*#?\s*(\d{1,3})\s*[:：]\s*(.*?)\s*$", seg)
        if not mm:
            continue
        i, body = int(mm.group(1)), mm.group(2).strip()
        if not body or body.upper() == "NONE":
            out[i] = (None, None)
            continue
        if "=" in body:
            lhs, rhs = body.split("=", 1)
            k = re.sub(r"[^0-9A-Za-z]", "", lhs)
            v = re.sub(r"[\r\n\t]", "", rhs.strip())
        else:
            k, v = "", body
        out[i] = (k or None, None if (not v or v.upper() == "NONE") else v)
    return out


def read_values(bgr, items, opts, stat):
    """`items = [(键, ib), …]` -> 等长的 `[(模型自报位号, 值), …]`。

    `value_batch <= 1` 时逐格调用（= 旧行为）；否则把 `value_batch` 格拼成一张图问一次。
    """
    n = int(opts.get("value_batch") or 1)
    if n <= 1 or len(items) <= 1:
        return [read_value(bgr, ib, opts, stat) for _k, ib in items]
    h, w = bgr.shape[:2]
    m = opts["value_margin"]
    cols = max(1, int(opts.get("value_cols") or 2))
    side = max(64, int(opts.get("value_side") or 520))
    out = [(None, None)] * len(items)
    for s in range(0, len(items), n):
        chunk = items[s:s + n]
        crops = []
        for _k, ib in chunk:
            x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
            x1 = min(w, int(ib[2] + m)); y1 = min(h, int(ib[3] + m))
            if x1 <= x0 or y1 <= y0:
                crops.append(None)
                continue
            crops.append(client.upscale(bgr[y0:y1, x0:x1], opts["value_scale"]))
        g = tile(crops, cols=cols, side=side)
        if g is None:
            continue
        txt, _dt = client.ask(g, VALUE_PROMPT_GRID, model=opts["model"])
        stat["value"] += 1
        got = parse_valgrid(txt)
        for i in range(len(chunk)):
            if i in got:
                out[s + i] = got[i]
    # **补漏**：拼图里模型没答出来的格子，退回逐格再问一次。
    # `_valbatch.py` 实测：batch=6 时「写回数量」会从 63/84 掉到 41/84（精度不掉，
    # 0.984 vs 0.976）—— 掉的是**覆盖率**。补漏把覆盖率补回来，
    # 只对没答出来的格子多花调用，总调用数仍然远低于「一格一次」。
    if opts.get("value_refill", True) and n > 1:
        for i, (key, ib) in enumerate(items):
            if out[i][1] is not None:      # 已经拿到值了（哪怕后面会被门控掉）
                continue
            out[i] = read_value(bgr, ib, opts, stat)
    return out


def read_name(bgr, ib, opts, stat):
    """读 IC 的**器件型号**（写进 `Name`）。返回型号字符串或 None。

    依据官方 PDF 第 8 页：`ic` 类比 `Name`，而 `box` 现在 tp 只有 2/996。
    `_vlmname.py` 实测（60 个 GT box 元件）：margin 80 / scale 2 首行 0.4333、
    任一候选 0.4667；margin 40 起就平了。返回**第一行**（调用方自行决定是否试其它候选）。
    """
    h, w = bgr.shape[:2]
    m = opts["name_margin"]
    x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
    x1 = min(w, int(ib[2] + m)); y1 = min(h, int(ib[3] + m))
    if x1 <= x0 or y1 <= y0:
        return None
    im = client.upscale(bgr[y0:y1, x0:x1], opts["name_scale"])
    txt, _dt = client.ask(im, NAME_PROMPT, model=opts["model"])
    stat["name"] += 1
    # ★ 2026-07 审计修复（与 `cv_` 那次同族的「静默计数器」）：`client.ask` 把**所有**
    # HTTP/超时/JSON 失败压成一个字符串返回、**永不抛**（`vlm\client.py` 的兜底
    # `except Exception as e: return "ERR:%s" % repr(e)[:80], ...`）。本函数是四个 VLM
    # 回读通道里**唯一**没挡这个前缀的 —— 于是 `ERR:<HTTPError 400 ...>` 会被原样
    # 写进 `cv_["Name"]`，`name_set` 还照样 +1。实测（纯逻辑复刻 client 的格式化）：
    # `"ERR:<HTTPError 401: 'Unauthorized'>"` 命中 `VALUE_LIKE`（:35 只要有一个字母就
    # 命中）、`t in IC` 为假 ⇒ 两个过滤器全放行。生产路径上 `-vlm-name` 默认就是 `on`
    # （`submission\run.py` 的 `default="on"` → `"name": True`），220 张 × 上千次调用，
    # 一次中途失败（某张裁片 400／OOM／单次超时）就把那颗元件的 Name 永久写坏。
    # 更糟的是 `name_set/name_calls` 会在**彻底失败时趋近 1.0** —— 全管线唯一用来
    # 证明型号通道有效的那个数字，恰好在坏掉时最好看。
    if str(txt or "").startswith("ERR:"):
        stat["name_err"] = stat.get("name_err", 0) + 1
        return None
    for line in (txt or "").strip().splitlines():
        t = line.strip().strip("`").strip()
        if not t or t.upper() == "NONE":
            continue
        if not VALUE_LIKE.search(t):        # 纯数字/符号，不可能是型号
            continue
        if t.lower() in IC:                 # 模型把类型词当型号回了（:424/:553 都先 .lower()，此处漏了）
            continue
        return t
    return None


def _pin_pitch(points, default=0.0):
    """这颗粒子**自己**的引脚间距：每个点到最近邻距离的中位数。

    QFP/SOP 的引脚排成列，最近邻必然是同一边的相邻脚 ⇒ 中位数就是引脚间距。
    只有 1 个点时无法估计，返回 `default`（调用方据此退回旧行为）。"""
    n = len(points)
    if n < 2:
        return float(default)
    P = [(float(a[0]), float(a[1])) for a in points]
    best = []
    for i in range(n):
        xi, yi = P[i]
        best.append(min(((xi - P[j][0]) ** 2 + (yi - P[j][1]) ** 2) ** 0.5
                        for j in range(n) if j != i))
    best.sort()
    return best[len(best) // 2]


def _edge_dir(bbox, pt):
    """引脚所在边的**内法向**（轴对齐单位向量）：取离引脚最近的那条边，方向指向体内。

    直角矩形本体（这里所有 IC 都是）上这是精确的；角上的引脚会并列最近，
    按 左/右/上/下 的顺序取先命中的那条，两个候选都是合理的近似。
    `bbox` 必须是**图像空间** `(x0, y0, x1, y1)`。"""
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    x, y = float(pt[0]), float(pt[1])
    dl, dr = x - x0, x1 - x
    dt, db = y - y0, y1 - y
    m = min(dl, dr, dt, db)
    if m == dl:
        return (1.0, 0.0)          # 贴左边 ⇒ 体内在右
    if m == dr:
        return (-1.0, 0.0)
    if m == dt:
        return (0.0, 1.0)
    return (0.0, -1.0)


def read_pins(bgr, points, ctr, win, opts, stat, bbox=None):
    """逐引脚取景 → 拼图 → 让 VLM 读「编号:名称」。

    ★ 垂直半宽按**这颗粒子自己的引脚间距**给（`0.45×pitch`），而不是沿用 `win[2]`：
    旧实现的窗口垂直总高 `2×cross`（缩放后常 40~48px）会把相邻引脚行一起框进来，
    模型于是报回**别的行**的名字 —— 10GT 上引脚错的 55 处里 41 处是 name_mismatch。
    这里用 `min(win[2], …)` 保证**只会更窄**，任何器件都不会因为这一改动而看到更多行。"""
    pitch = _pin_pitch(points)
    cross = float(win[2])
    frac = float(opts.get("pin_pitch_frac", 0.45))
    cmin = float(opts.get("pin_cross_min", 12.0))
    # `frac<=0` 或估不出间距 ⇒ `cross_perp=None`，**逐位回到旧行为**（A/B 的对照臂）
    cp = None if (pitch <= 0 or frac <= 0) else min(cross, max(cmin, frac * pitch))
    stat.setdefault("pitch", []).append(round(pitch, 1))
    stat.setdefault("cross_perp", []).append(None if cp is None else round(cp, 1))
    use_edge = bool(opts.get("pin_edge_dir", True)) and bbox is not None
    crops, idx = [], []
    for i, q in enumerate(points):
        uv = _edge_dir(bbox, q) if use_edge else None
        c = crop_dir(bgr, q, ctr, win[0], win[1], cross, cross_perp=cp, dir_uv=uv)
        if c is None:
            continue
        im, rel = c
        mark(im, rel, opts["mr"], opts["mt"], COLORS[opts["color"]])
        crops.append(im); idx.append(i)
    ans = {}
    per, cols = opts["per"], opts["cols"]
    for s in range(0, len(crops), per):
        sub = crops[s:s + per]
        subidx = idx[s:s + per]
        g = tile(sub, cols=cols)
        txt, _dt = client.ask(g, PINS_PROMPT, model=opts["model"])
        stat["grid"] += 1
        for i, (num, name) in parse_grid(txt).items():
            if 0 <= i < len(sub):
                ans[subidx[i]] = (num, name)
    return ans


def enrich_doc(pred, png_path, opts, stat, log=None):
    """就地改写一份预测文档：先按 VLM 读到的位号改名（同步 `pins` 与 `nets.hyperGraph`），
    再逐引脚补号/名。返回 (n_rename, n_pin)。"""
    from stage2.text import imread_unicode

    bgr = imread_unicode(png_path)
    H = float(bgr.shape[0])
    comps = pred.get("components") or {}
    pins = pred.get("pins") or {}
    n_ren = n_pin = 0

    renames = {}
    if opts["designator"]:
        # 默认只给 IC 读位号；`desig_all` 打开后所有**带位号**的元件都读
        # （r/c/l/d/led/switch/… 的 GT 键是 R1/C5 这种，键对不上就是 missing+extra）。
        # 电源类（gnd / v / net_*）GT 用的是 ∅NNN 无名键，靠坐标匹配，不读。
        no_desig = frozenset({"gnd", "v", "net_input", "net_output",
                              "net_short", "net_bidirection", "net_passive"})
        reads = {}
        for key, cv_ in comps.items():
            ct = str(cv_.get("type") or "").lower()
            if ct in no_desig:
                continue
            if not opts.get("desig_all"):
                # `desig_fallback`：现在的键是兜底键 `pNNN` 的元件一律补读位号。
                # 这类元件 220 张里有 2601 个，其中 92% 在 10px 内就有文字框、
                # 但 `assign.py` 因为「同一个框只能给一个人」的全局去重没给它 ——
                # 而 `pNNN` 键**永远不可能**和 GT 对上，换掉它只有赚没有赔。
                want = ct in IC or (opts.get("desig_fallback")
                                    and FKEY_RE.match(str(key)) is not None)
                if not want:
                    continue
            bb = cv_.get("bbox")
            if not bb:
                continue
            ib = (bb[0], H - bb[3], bb[2], H - bb[1])       # y 向上 -> 图像坐标
            got = read_designator(bgr, ib, opts, stat, ct)
            if got:
                reads[key] = got
        # 冲突消解：先只认「无人争抢」的读数。多颗 IC 读出同一个位号，说明至少
        # 有一颗读到了邻居（`0001` 的 U3 就常被读成邻居的 `U1`）—— 这种整组放弃，
        # 而不是随机挑一个去抢占。`U?`/`J?` 这种未标注形式在 `read_designator`
        # 里已经被 `DESIG_RE` 挡掉。
        claims = {}
        for k, g in reads.items():
            claims.setdefault(g, []).append(k)
        solo = {g: ks[0] for g, ks in claims.items() if len(ks) == 1 and ks[0] != g}
        renames = {}
        # 第一轮：目标键本来就空着的，直接改。
        for g, old in sorted(solo.items()):
            if g not in comps:
                renames[old] = g
        # 第二轮：目标键被别的元件占着。只有当那个占用者**自己也被改名搬走**
        # （它的真名读出来了）才让位 —— 硬抢占会在 `comps` 字典里撞键、丢掉一个元件。
        for g, old in sorted(solo.items()):
            if g in renames.values() or g not in comps:
                continue
            holder = comps.get(g)
            if reads.get(g) == g:          # 占用者自我确认，它是真的
                continue
            if g in renames:               # 占用者自己也要搬走 -> 位子腾出来了
                renames[old] = g
        if renames:
            comps = {renames.get(k, k): v for k, v in comps.items()}
            pins = {renames.get(k, k): v for k, v in pins.items()}
            for net in (pred.get("nets") or {}).values():
                hg = net.get("hyperGraph")
                if isinstance(hg, str):
                    for old, new in renames.items():
                        hg = re.sub(r"(?<![0-9A-Za-z_])%s\." % re.escape(old),
                                    "%s." % new, hg)
                    net["hyperGraph"] = hg
            n_ren = len(renames)

    # ---- 分立元件的「值」 -------------------------------------------------
    # 220 张 A/B：现行管线在 GT 值非空的元件上写对 0.577，VLM 是 0.942；
    # 而且**严格碾压** —— 120 样本里「只 VLM 对 38 / 只管线对 0」，所以直接覆盖，
    # 不做取舍。代价是 GT 值本来就是空的那 16 个里多写 8 个值（那些 GT 本身没标）。
    n_val = 0
    late = {}
    if opts["value"]:
        vitems = []
        for key, cv_ in comps.items():
            if str(cv_.get("type") or "").lower() not in VAL_TYPES:
                continue
            bb = cv_.get("bbox")
            if not bb:
                continue
            vitems.append((key, (bb[0], H - bb[3], bb[2], H - bb[1])))
        for (key, _ib), (kk, val) in zip(vitems, read_values(bgr, vitems, opts, stat)):
            # ⚠️⚠️ 2025-10-05 修（**这是 `_submit_out220z` 之后端到端回归 −0.88 的真凶**）：
            # 上面那个循环的循环变量是 `for key, cv_ in comps.items()`，`vitems` 里只存了
            # `(key, inner_box)`，**没存 `cv_`**；下面这个循环又用了 `cv_` —— 于是它绑定的
            # 永远是上面那个循环**最后一次迭代**留下的那颗元件。实测（case 0002）：
            #   `[val] key=p001 cv_is_comps_key=False -> '100n'` … 六次全是 False。
            # ⇒ 值确实被读对了，却全写在错误的元件上（`value_set` 计数照样 +1，把 bug 藏住了）。
            cv_ = comps.get(key)
            if cv_ is None:
                continue
            # **顺手捡位号**（零额外成本）：值通道让模型回的是 `位号=值` 一整行。
            # 如果这颗元件的键还是 `pNNN` 兜底（说明专门的位号读取失败了），
            # 而模型自报的位号形状合法、又没人占，就把它当成一次补读。
            # 兜底键本来就一分不得，换掉只有可能变好。
            if (opts.get("value_use_key") and kk and DESIG_RE.match(kk)
                    and FKEY_RE.match(str(key)) and kk not in comps):
                late.setdefault(key, kk)
            # **位号一致门控**（这是值通道从负转正的关键）：
            # 模型回的是 `位号=值`，如果它报的位号不是这颗元件自己的，说明它读到了邻居。
            # `_vlmval3.py` 的 400 样本实测：全采纳 0.6400 / 0.7050（scale 1 / 4），
            # 只采纳「位号对上」的 0.7125 / **0.7950**（管线 0.5025）。
            # ⚠️ 2025-10-05 修：`late` 的改名原先在**值通道之后**（:512）才应用，于是键还是
            # `pNNN` 的元件永远过不了这个门控 —— 220 张带 VLM 实测 `value_set=0`，
            # 值通道整体哑火（`_submit_out220z` 是更早版本的 enrich 跑的，所以复现不出）。
            # 现在把「这颗元件的键是兜底键、且模型自报的位号正是它的待改名」也判为过门。
            _own = bool(kk) and (kk.upper() == str(key).upper() or late.get(key) == kk)
            if opts["value_key_gate"] and kk and not _own:
                continue
            if val and VALUE_LIKE.search(val):
                cv_["value"] = val
                n_val += 1
        stat["value_set"] = n_val

    # 应用上面捡到的位号（冲突规则与专门读取那轮一致：目标键必须真的空着）
    if late:
        taken = set(comps) - set(late)
        keep = {}
        for old, new in late.items():
            if new in taken or new in keep.values():
                continue
            keep[old] = new
        if keep:
            comps = {keep.get(k, k): v for k, v in comps.items()}
            pins = {keep.get(k, k): v for k, v in pins.items()}
            for net in (pred.get("nets") or {}).values():
                hg = net.get("hyperGraph")
                if isinstance(hg, str) and hg:
                    for old, new in keep.items():
                        hg = re.sub(r"(?<![0-9A-Za-z_])%s\." % re.escape(old),
                                    "%s." % new, hg)
                    net["hyperGraph"] = hg
            renames.update(keep)
            n_ren = len(renames)

    # ---- IC 的「器件型号」（写 `Name`） -----------------------------------
    # 官方 PDF 第 8 页：`ic` 比 `Name`。`box` 现在 tp 只有 2/996（我们的 Name 是引脚名/器件值），
    # 而 `_vlmname.py` 实测读数 0.4333 ⇒ 估 +0.37 分。风险极低：几乎没有可改坏的。
    n_nam = 0
    if opts["name"]:
        for key, cv_ in comps.items():
            if str(cv_.get("type") or "").lower() not in IC:
                continue
            bb = cv_.get("bbox")
            if not bb:
                continue
            ib = (bb[0], H - bb[3], bb[2], H - bb[1])
            val = read_name(bgr, ib, opts, stat)
            if val:
                cv_["Name"] = val
                n_nam += 1
        stat["name_set"] = n_nam

    # ★ 引脚通道（见 `PIN_CHANNEL` 的定义处）：默认**关**。关掉时下面的循环体一次都不走，
    # `pins` 原样返回 —— 位号/值/型号三条通道在此之前已经改完，不受影响。
    _pin_ren = {}            # {元件键: {旧号: 新号}}，循环结束后重写 nets[*].hyperGraph
    if opts.get("pin", PIN_CHANNEL):
        # ★ 给融合层留一个**出处标记**：`fusion/claims.py:_add_vlm` 的 (d) 段只对带
        #   `_pin_channel` 的文档把 `pins` 里的号/名当候选收进池子。没有这个标记，
        #   基线臂（引脚通道关着）的文档也会被收一遍 CRNN 读数，A/B 就不干净了。
        pred["_pin_channel"] = True
    for key, cv_ in (list(comps.items()) if opts.get("pin", PIN_CHANNEL) else []):
        if str(cv_.get("type") or "").lower() not in IC:
            continue
        cur = pins.get(key) or {}
        if len(cur) < opts["min_pins"]:
            continue
        bb = cv_.get("bbox")
        if not bb:
            continue
        points, keep = [], []
        for pk, info in cur.items():
            pt = (info or {}).get("point") or {}
            try:
                points.append((float(pt["x"]), H - float(pt["y"])))
                keep.append((pk, info))
            except Exception:  # noqa: BLE001
                pass
        if not points:
            continue
        win = adaptive_window(bb, opts["inner"], opts["cross"], opts["outer"], opts["ref"])
        ctr = ((bb[0] + bb[2]) / 2.0, H - (bb[1] + bb[3]) / 2.0)
        # `bb` 是 **JSON 空间**，`points` 已经翻到图像空间 ⇒ 交给 `read_pins` 的框也要翻
        bb_img = (bb[0], H - bb[3], bb[2], H - bb[1])
        ans = read_pins(bgr, points, ctr, win, opts, stat, bbox=bb_img)
        # VLM 读到的号优先；读不到就**原样保留**（丢一个引脚就多一个 `pin.missing`，
        # 实测会让 `pin.missing` 302→308、总分倒退）。
        newpins = {}
        used = set()
        _pin_ren_before = dict(_pin_ren.get(key) or {})
        _merge = bool(opts.get("pin_merge"))
        for i, (pk, info) in enumerate(keep):
            num = name = ""
            if i in ans:
                num, name = (ans[i][0] or "").strip(), (ans[i][1] or "").strip()
            old_name = str((info or {}).get("pinname") or "")
            if _merge:
                # ★ 合并：**号用我们自己的**（几何/CRNN），只把模型读到的名并进来。
                newpins[pk] = {"pinname": name or old_name, "point": info["point"]}
                continue
            if num and num in used:
                num = ""
            if num:
                used.add(num)
                # ★ 2026-10-13（§5.228）：**改名撞键不许吃脚**。旧写法直接
                #   `newpins["pin_%s" % num] = ...`，而 `else` 分支又会把「号没读出来/
                #   号被去重」的引脚按**原键**塞回同一个 dict —— 只要某个原键恰好等于
                #   别人刚读出的新键，后者就把前者**覆盖掉**，这颗元件的引脚数静默减少。
                #   实测（220 张批 vs 20 张串行，同参数同案子）`10GT/1` 的 U1 少了 7 个脚
                #   （26 → 19）、该案 −19.75 分；批里「改写引脚」350 次 vs 串行 156 次，
                #   撞键机会正比于改写量。这里改成撞了就让位（退回原键，仍撞就加后缀）。
                _tgt = "pin_%s" % num
                if _tgt in newpins:
                    _tgt = pk
                    if _tgt in newpins:
                        _j = 2
                        while "%s_%d" % (pk, _j) in newpins:
                            _j += 1
                        _tgt = "%s_%d" % (pk, _j)
                if _tgt != pk:
                    _pin_ren.setdefault(key, {})[
                        pk[4:] if pk.startswith("pin_") else pk] = num
                newpins[_tgt] = {"pinname": name or old_name, "point": info["point"]}
                n_pin += 1
            else:
                newpins[pk] = info
        if newpins:
            # ★ 2026-10-13（§5.228）：最后一道保险 —— 改写后引脚数**变少**就整颗放弃改写。
            #   一颗粒 IC 少一个脚 = 多一条 `pin.missing`，而 `pin.f1` 是 0.25 权重；
            #   实测 `10GT/1` 在负载饱和的批里就丢过 7 个脚（−19.75 分）。
            #   宁可保留「号没读对」的原样，也不接受「少一个脚」。
            if len(newpins) >= len(keep):
                pins[key] = newpins
            else:
                # 改写被否 ⇒ 这颗元件的引脚改名记录也必须回滚，否则
                # `nets[*].hyperGraph` 会被改成**不存在的**新脚号（比不改更糟）。
                if _pin_ren_before:
                    _pin_ren[key] = _pin_ren_before
                else:
                    _pin_ren.pop(key, None)

    # ★ 引脚键改名后必须同步 `nets[*].hyperGraph`（`VLM_PIN_REBUILD`，默认开）。
    # 不改的话，被 VLM 读对号的那些引脚**进不了任何网络**：实测 `net.pred_pins`
    # 替换前后都是 220、net_hypergraph 三臂逐位相同（0.4987），引脚通道的收益
    # 一滴都流不到 0.35 权重的网络项上。位号改名那条路在 `:591-597` 早就这么做了。
    if _pin_ren and (opts.get("pin_rebuild", PIN_REBUILD)):
        for net in (pred.get("nets") or {}).values():
            hg = net.get("hyperGraph")
            if not isinstance(hg, str):
                continue
            for ck, mp in _pin_ren.items():
                for old, new in mp.items():
                    hg = re.sub(
                        r"(?<![0-9A-Za-z_])%s\.%s(?![0-9A-Za-z_])"
                        % (re.escape(ck), re.escape(old)),
                        "%s.%s" % (ck, new), hg)
            net["hyperGraph"] = hg

    pred["components"] = comps
    pred["pins"] = pins
    if log is not None:
        log.append((pred.get("case"), n_ren, stat["desig"], stat["grid"], n_pin, renames,
                    stat["value"], stat["value_set"], stat["name"], stat["name_set"],
                    stat.get("name_err", 0)))
    return n_ren, n_pin


# ★ `VLM_PIN_MERGE`：引脚通道是「**替换**」还是「**合并**」。
#
# 老行为（=0，替换）：`newpins["pin_%s" % num] = {...}` —— 模型给的号**整体覆盖**我们
#   几何/CRNN 的号。220 张实测：VLM 让「名」变好（`_pinname.py`：differ 780→631、
#   equal 232→280），却让「号」变差（`number_absent` 75.8%→80.4%），因为模型给号
#   一旦读歪，我们本来对的号也被丢掉。
# 新行为（=1，合并）：**保留我们自己的键 `pk`**，只把模型读到的 `name` 并进来
#   （`name or old_name`）。号仍由几何/CRNN 决定，名由 VLM 补。
#
# 依据：10GT 上「点」已经很好（`_pincov.py`：`point@5` 69.9%、`point@15` 81.6%、`num_ok`
# 87.5%、`both_ok` 63.2%），瓶颈在**名**（`_namefix.py`：逐字相等只有 48.4%）。
_PIN_MERGE_ENV = os.environ.get("VLM_PIN_MERGE", "0").strip().lower()
PIN_MERGE = _PIN_MERGE_ENV not in ("", "0", "false", "off", "none")

# ★ `VLM_PIN_REBUILD`（默认开）：replacement 模式会把引脚键 `pin_13` 改成 `pin_113`，
# 但 `nets[*].hyperGraph` 是**字符串** `"(U1.13,...)"`。不改它，读对的引脚就
# **进不了任何网络** —— 实测 `net.pred_pins` 在替换前后都是 220、`net_hypergraph`
# 一动没动（2026-10 官方 9 张：none 0.4987 / pinmerge 0.4987 / pinrepl 0.4987），
# 即引脚通道的分全丢在网络门口。位号改名那条路早就在 `:591-597` 做了同样的重写。
_PIN_REBUILD_ENV = os.environ.get("VLM_PIN_REBUILD", "1").strip().lower()
PIN_REBUILD = _PIN_REBUILD_ENV not in ("", "0", "false", "off", "none")

# ★★ 引脚通道总开关（2025-1x）：**默认关**。
#
# 为什么关：pincol 打开后 stage1 已经能给大 IC 交出**几何正确**的引脚列
# （0143 的 U1：GT 点 5px 命中 3/64 → 63/64，五号窗/名字窗都按等距格点算好）。
# 而 VLM 的引脚通道是「一次性看到 N 格、回一份 `格号:编号:名称`」——
# 在大 IC 上它**只回得出一小部分**，并且（默认 replace 模式）会把我们自己
# 本来对的钥匙整批替换掉。0143 实测（pincol + `STAGE2_BOX_PIN_NEED_NAME` 全开）：
#   不开 VLM                                        → 总分 **39.06**，pin F1 0.2857
#   `VLM_PIN_PER=8  VLM_PIN_COLS=4`（老默认）        → **35.47**，U1 只剩 13 个脚、pin F1 0.0000
#   `VLM_PIN_PER=4  VLM_PIN_COLS=2`                 → **36.41**，pin F1 0.0377
#   `VLM_PIN_PER=2  VLM_PIN_COLS=1`                 → **35.47**，U1 13 个脚、唯一名 11
# 三种网格都一样差 ⇒ **不是「一次问几格」的问题，是这条路本身在有大 IC 的图上净负。**
# 而同一个 VLM 的**位号通道**在 0143 上是 **component F1 0.5833 → 0.8750（值 +8.75 分）**，
# 跟引脚通道无关 —— 所以这里只摘掉引脚那一段，位号 / 值 / 型号三条通道照留。
# 代价：`pinname` 少了 VLM 这条来源，只能靠 CRNN 读框内的名字。
#
# ★ 2026-10 补测（**端到端交付口径**，且融合层已修掉「文档的 `pins` 没人读」那个 bug）：
#   官方 10GT 9 张 **58.6304 → 61.5112（+2.88）**，pin F1 0.3920→0.4858（tp 37→45）；
#   派生 220 张 **29.4819 → 29.4099（−0.07）**。两半拆开（`FUSION_VLM_PIN_NUM/NAME`）：
#   官方档**只补号 +1.951 且九张没有一张变差**、只补名 +1.316、全开 +2.881、空跑 0.000。
#   ⇒ 上面「在有大 IC 的图上净负」这个判断在**派生档上依然成立**（那正是 220 的构成），
#   本行默认值因此不改；**官方档该开**，并把两半开关交给 `fusion/claims.py`。
#   见 `docs/RESULTS.md` §2.5、`docs/DIAGNOSTICS.md`「写进了产物 ≠ 有人读」。
_PIN_CHANNEL_ENV = os.environ.get("VLM_PIN_CHANNEL", "0").strip().lower()
PIN_CHANNEL = _PIN_CHANNEL_ENV not in ("", "0", "false", "off", "none")

DEFAULTS = dict(designator=True, desig_all=False, desig_margin=60.0,
                # ★ `per`/`cols` 是**引脚网格**的「一次问几格 / 几列」。`per=8,cols=4` 是
                # 老默认（4 列 × 2 行）。大 IC（0143 的 U1 有 66 个脚）上模型一次看 8 格
                # 会串行（实测回 19 个答案、其中 `PC0` 重复 3 次）。
                # 改成环境变量是为了能不改 CLI 就 A/B：`VLM_PIN_PER` / `VLM_PIN_COLS`。
                per=int(os.environ.get("VLM_PIN_PER", "8")),
                cols=int(os.environ.get("VLM_PIN_COLS", "4")),
                min_pins=2, pin_merge=PIN_MERGE, pin_rebuild=PIN_REBUILD,
                pin=PIN_CHANNEL,
                # **引脚名窗口：`inner` 是最大的一根旋钮**（实测，`work\tools\diag\_pinsweep.py`，
                # 在 **GT 的引脚点上**读，所以只反映窗口几何、不含点检测误差）：
                #   10GT 的 4 张有 IC 键的图（8/4/3/10，46 个引脚）按「引脚名完全一致」算
                #     inner= 78（旧默认） 0.522 ｜ 120 → 0.761 ｜ **160 → 0.891** ｜ 220 → 0.848
                #     ｜ 300 → 0.870 ｜ 420 → 0.717（太大反而把邻排卷进来）
                #   号与名都对的「都」：旧 0.522 → **160 时 0.826**
                #   `cross`（± 垂直范围）几乎不改名，58 与 6 只在 inner 大时差 0.07。
                # 根因：这些原理图的引脚名是 `/VOUT`/`/NRESET`/`/VDDREG` 这种带前缀的长串，
                # inner=78 会把右边**截断**（`/VOUT`→`/VOU`、`/MISO`→`/MIS`、`/NRESET`→`/NRE`）。
                outer=34.0, inner=160.0, cross=58.0, ref=762.0, mr=4, mt=2,
                # 引脚窗口的**垂直半宽**：不再沿用 `cross`，而取「这颗粒子自己的引脚间距
                # × `pin_pitch_frac`」（`read_pins` 里 `min(cross, …)`，只会更窄）。
                # 旧实现垂直总高 `2×cross`（缩放后常 40~116px）而 QFP 间距常 19~30px
                # ⇒ 几何上确实会框进相邻引脚行（`_pinwin_u.py` 量的「混进别的引脚点」
                # 从 2.09 降到 0.00），**但换成真模型就翻车了**。
                # ★★ 4 组件 × 4 臂真跑 VLM（`work\tools\diag\_pinab.py 8:U1 4:U2 10:U1 1:U1`，
                #   36 个引脚；`名对/号对/都对`）：
                #     OLD  frac=0    edge=0 → 30/25/23
                #     EDGE frac=0    edge=1 → 30/25/24
                #     HALF frac=0.45 edge=0 → 30/24/**22**
                #     NEW  frac=0.45 edge=1 → 30/13/**11**   ← case 1 的号对 12/15 掉到 1/15
                # ⇒ ①**名对四臂全是 30/36，一点没动** —— 「框进邻行导致名字错」这个
                #      假设被证伪，几何不是 name_mismatch 的杠杆；
                #    ②号对才是被这组改动影响的量，而 `frac=0.45`+`edge=1` 同时开是**灾难**
                #      （大 IC 上窗口收成 27px×… 的窄条后，模型把编号读成隔几行的别的脚）；
                #    ③样本只有 36 个脚，EDGE 的 +1 是噪声，不能当收益。
                # 所以两个开关**默认都关**（真·旧行为）：机制与 A/B 臂留着，等有更大的
                # 夹具再定。置 0/False 关闭。
                pin_pitch_frac=0.0,
                # 垂直半宽的**下限**：引脚**编号**常常印在引脚线上方（垂直偏移约 6~12px），
                # 收得比它更紧就会连自己的编号一起裁掉（`_pincropab.py` 渲染可见：
                # 下限 6 时若干格只剩红圈与一截线，编号没了）。12 仍远小于节距的一半
                # （最小节距 19 ⇒ 相邻引脚线在 ±19），所以不会把邻行拉回来。
                pin_cross_min=12.0,
                # 引脚窗口的**朝向**：`u`（沿引脚、朝体内）取「引脚所在边的内法向」而不是
                # 「引脚 → 包围盒中心」。后者对窄高 IC 是**斜的**，会让窗口在两个轴上都胖。
                # 置 False 关闭（退回旧行为）。见上面那张四臂表：单独开只是噪声级。
                pin_edge_dir=False,
                color="red", model=client.MODEL,
                # 值：**默认开**。关键是「位号一致」门控 —— 模型回的是 `位号=值`，
                # 只采纳它报的位号就是这颗元件自己的那些，其余（读到邻居）丢掉。
                # 220 张端到端 A/B（`work\stage2_v17` 为输入，只开值通道）：
                #   不开 23.9559（text_mismatch 961）→ 开门控 **24.9051**（tp 5527，text_mismatch 666）
                # 对照：不带门控的旧版是 **负的**（23.6404，text_mismatch 1066）。
                value=True, value_margin=80.0, value_scale=4, value_key_gate=True,
                # **请求合并**：把 `value_batch` 格值裁片拼成一张图一次问完。
                # 默认 1 = 关（旧行为）。打开前必须先 A/B 证明每格精度不掉 ——
                # 值通道对裁片尺寸极其敏感，而拼图会让每格变小。
                value_batch=1, value_cols=2, value_side=520, value_refill=True,
                # 值通道顺手捡位号：只对键还是 `pNNN` 兜底的元件生效（零额外调用）。
                value_use_key=True,
                # IC 器件型号（写 Name）：实测读数 0.4333，box 现在 tp 只有 2/996。
                name=False, name_margin=80.0, name_scale=2,
                # 兜底键补读位号：只对**现在的键是 `pNNN`** 的元件读（默认按 IC 类型读）。
                # 220 张里这类元件 2601 个，92% 在 10px 内就有文字框却拿不到位号。
                desig_fallback=False,
                # 开跑前的一次**响亮**自检（`client.preflight`）：端点 401 / 不吃图 / 只产
                # reasoning 三种情况下，`enrich_dir` 的统计全都是「三个 set 归零」，与
                # 「图里真没东西」长得一模一样。默认开；只在离线复现时关。
                preflight=True)


def enrich_dir(indir, outdir, png_of, opts=None, workers=6, limit=0, cases=None,
               verbose=True):
    """对目录里的每份 `<case>.json` 做 VLM 补强，写到 `outdir`。返回统计 dict。"""
    o = dict(DEFAULTS)
    o.update(opts or {})
    indir, outdir = Path(indir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # 阶段三要读 `<s2>/_assigned/<case>.json`，所以补强目录必须把它一并镜像过来
    # （内容不变；补强只改 `<case>.json` 的 components/pins/nets）。
    src_asg = indir / "_assigned"
    if src_asg.is_dir():
        dst_asg = outdir / "_assigned"
        dst_asg.mkdir(parents=True, exist_ok=True)
        for f in src_asg.glob("*.json"):
            (dst_asg / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    wanted = set(cases or [])
    lookup = png_of if callable(png_of) else (lambda k: (png_of or {}).get(k))
    jobs = []
    for f in sorted(indir.glob("*.json")):
        if wanted and f.stem not in wanted:
            continue
        jobs.append((f, lookup(f.stem)))
        if limit and len(jobs) >= limit:
            break
    log = []
    # 起跑前先问一次端点。失败就抛 —— 不能让 400 张图「1 秒跑完、全部分项 0」地静默劣化。
    if o.get("preflight", True) and jobs:
        client.preflight()

    def work(job):
        f, png = job
        try:
            pred = json.loads(f.read_text(encoding="utf-8"))
            if not png:
                # 找不到对应 PNG：只能原样拷贝。**这是一次「没补强」**，必须计数，
                # 否则导出阶段看到的是一个和输入逐位相同的文件、发现不了。
                n_failed.append((f.stem, "no-png"))
                (outdir / f.name).write_text(json.dumps(pred, ensure_ascii=False),
                                             encoding="utf-8")
                return
            stat = dict(desig=0, grid=0, value=0, value_set=0, name=0, name_set=0)
            enrich_doc(pred, png, o, stat, log)
            (outdir / f.name).write_text(json.dumps(pred, ensure_ascii=False),
                                         encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print("  !! %s: %r" % (f.stem, e))
            n_failed.append((f.stem, "%s: %s" % (type(e).__name__, e)))
            try:
                (outdir / f.name).write_text(f.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass

    n_failed = []          # ★ 2026-07 审计修复：两条失败路径以前都是**完全静默**的
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    stats = dict(n=len(log), n_failed=len(n_failed), failed=n_failed[:20],
                 renamed=sum(r[1] for r in log),
                 desig_calls=sum(r[2] for r in log),
                 grid_calls=sum(r[3] for r in log),
                 pins=sum(r[4] for r in log),
                 value_calls=sum(r[6] for r in log),
                 value_set=sum(r[7] for r in log),
                 name_calls=sum(r[8] for r in log),
                 name_set=sum(r[9] for r in log),
                 name_err=sum(r[10] for r in log))
    return stats
