"""整颗元件「引脚号 ↔ 引脚名」抄表 + 互证改名（`STAGE2_PIN_TABLE`，**默认关**）。

问题（open_GT 13 页实测）：箱体件的号与名是**两条独立的 OCR 通道**，各自按槽落位，
于是同一根脚常常「号落在 A 槽、名落在 B 槽」，而 `assemble.py` 的迟到 need-name
闸门只放「有名字」的槽 ⇒ 号读对的槽没名字被丢、名字读对的槽号错了被发成别的键。

本模块换一个证据源：把**整颗元件**（外扩边距、放大）交给本地 VLM **一次抄出整张
`(号, 名)` 表**（`vlm/pinzoom.py` 是逐脚放大读、`vlm/pinname.py` 只读名字，都没有
「整张表」这条通道）。表**只用来改名字**，且必须过下面两道闸门：

1. **表自洽闸门**（整个元件）：把「我方已读到的号」拿去查表，记 `agree`（查到的名与
   我方读到的名一致）、`dis`（不一致）；`agree + dis < MIN_VOTES` 或 `dis > agree`
   ⇒ 整颗弃权。判据来自实测：`case2` 的三颗 74HC 符号让 VLM 多吐了一行（读数 `1 1`
   顶在最前，整张表**下移一位**），此时四个号**全部对不上**我方读数（agree 0 /
   dis 4）；而正确的表（`case5 U2` 7/7、`case12 U1` 9/9）只在**个别脚**上不一致。
2. **号唯一**（逐槽）：该槽的号在**本元件内**只能出现在这一个槽上。`assemble.py`
   的 partial 修复按 `(not nm, -len(nm), i)` 决定谁留号，换名会改变这个竞争——
   `case12 U1` 实测：给一个「号 8 与邻槽重复」的槽改个更长的名，它就在排序里抢走了
   邻槽的 `pin_8`，几何从对的槽换到错的槽（0.800 → 0.600）。号唯一 ⇒ 这一项不可能
   翻转，唯一的改动是「本来就发得出去的那一槽，名字变对」。

★ 五条「不做 / 试过更差」的负结果（离线复刻 `work/_pt_try.py`，逐元件 tp/fp/fn 与
13 页 pin f1；基线 = 现状 113/101/96 = 0.5343）——写在这里免得后人再试：

* **不写号**：按侧等差拟合把表里的号外推到没读到号的槽上，0.5343 → **0.4871**
  （`case5 U1/U2` 的 `5/L`↔`6/VIN` 被拟合反了方向，`case2` 整表下移时全错）。
* **无闸门地填空名**（只填空名、不看表自洽）：**0.5135** —— 填名会让槽从「被丢弃」
  变成「被发出」，位置不对就是 fn→fp+fn。
* **不看号唯一地覆盖+填空**：0.5467，`case12 U1` 0.800 → **0.522** 崩。
* **要名字互证**（表名必须在同元件另一槽被读到过）：0.5485，`case12 U1` → 0.600 崩
  （互证挡不住重复号那一条）；再加号唯一 = 0.5579。
* 互证去掉、只留「自洽闸门 + 号唯一 + 允许给空名槽补名」（**本模块当前口径**）：
  **0.5781**，13 页无一处变差（`case1` .342→.387、`case4` .667→.800、
  `case5` .486→.579、`case9` .667→.833、`case10` .640→.741、`case11` .222→.278）。

改动只有 `assigned[cid]["pins"][idx]["name"]`（名字），**不动坐标、不动号、不增删槽**；
开关不设时 `run.py` 连 import 都不做，下游逐位不变。
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

# 只处理「算分的」箱体件（与 `schematic_parser/metrics.py` 的 `IC_TYPES` 同集合）。
TYPES = ("box", "ic", "boxic")

# 裁片外扩与放大：外扩沿用 `run.py:142` 的 `STAGE2_ZOOM_READ_PAD`（同名同默认 40，
# 就是逐脚放大读用的那一个边距）；放大 2× 是探针实测（`work/_vlmtable13.py`，
# 13 页 open_GT 上 34 颗箱体件）——1× 时小字的脚名糊成一片。
PAD = float(os.environ.get("STAGE2_ZOOM_READ_PAD", "40") or 40)
SCALE = float(os.environ.get("STAGE2_PIN_TABLE_SCALE", "2") or 2)

# 互证闸门要「至少两只脚」的表自洽证据：单只脚的比较是抛硬币，不够格改名字。
MIN_VOTES = 2

PROMPT = ("这张图是一个电路元件符号。请把**每一条引脚**的编号和名称都抄下来。"
          "引脚画在符号四周：编号是阿拉伯数字，名称是旁边的英文/数字短词。"
          "输出格式：每行一条，`编号 名称`，用单个空格分隔；只输出这些行，"
          "不要列表符号、不要解释。名称看不清写 `?`。没有引脚就输出 NONE。")

# VLM 输出的一行：号（≤2 字母前缀 + 1~3 位数字 + 可选 1 字母）+ 名（非空无空格）。
# 与 `work/_vlmtable13.py` 的解析完全一致（同一份实测有效的口径）。
_ROW_RE = re.compile(r"^([A-Za-z]{0,2}\d{1,3}[A-Za-z]?)\s+(\S+)$")

_PROBED: Optional[bool] = None


# --------------------------------------------------------------------------- 解析


def parse(text: str) -> List[Tuple[str, str]]:
    """VLM 正文 → `[(号, 名)]`；名是 `?` 的行丢弃（那只是「看不清」的占位）。"""
    out: List[Tuple[str, str]] = []
    for ln in str(text or "").splitlines():
        ln = ln.strip().strip("`-* ")
        if not ln or ln.upper() == "NONE":
            continue
        m = _ROW_RE.match(ln)
        if not m:
            continue
        num, name = m.group(1), m.group(2)
        if name == "?":
            continue
        out.append((num, name))
    return out


def table(rows: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """`[(号, 名)]` → 号唯一的表（先到先得，保序）。

    **号可以是字母数字**（`case11 USB1` 的 `A12`/`B5`/`B12`：连接器脚号本来就不是
    纯数字），照 `assemble` 的 `pin_<号>` 口径原样保留，不做任何改写。
    名是**纯数字**的行一律丢：`work/_pt_gather.jsonl` 里这样的行共 5 条、全是
    `1 1`（VLM 把脚号抄成了脚名），其中池案 `0006` 的 `U9` 因此把名 `3` 写成 `1`
    并掉了 0.42 分（net 侧 `pred_count` 34→31）。
    """
    seen, out = set(), []
    for num, name in rows:
        num, name = str(num or "").strip(), str(name or "").strip()
        if not num or not name or num in seen:
            continue
        if re.fullmatch(r"[0-9]+", name):
            continue
        seen.add(num)
        out.append((num, name))
    return out


def _norm(s) -> str:
    """名字归一：只留字母数字并小写（`D+` → `d`、`V_out` → `vout`）。"""
    return re.sub(r"[^0-9a-z]", "", str(s or "").lower())


def _same(a, b) -> bool:
    """表自洽闸门用的宽松相等：归一后相同，或一方是另一方的前缀
    （`open_GT` 的脚名有 `D-`/`EH`/`EP` 这类短词，VLM 常抄成 `D`/`EPAD`）。"""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb or na.startswith(nb) or nb.startswith(na)


# --------------------------------------------------------------------------- 几何


def _bbox(c: dict, image_h: Optional[float]) -> List[float]:
    """元件框，**y 向下**（stage1 的 `bbox_img` 就是这个坐标；`bbox_json` 要翻）。"""
    bb = c.get("bbox_img") or []
    if len(bb) >= 4:
        return [float(v) for v in bb[:4]]
    bb = c.get("bbox_json") or []
    if len(bb) >= 4 and image_h:
        x0, y0, x1, y1 = [float(v) for v in bb[:4]]
        return [x0, float(image_h) - y1, x1, float(image_h) - y0]
    return []


def _crop(bgr, bbox: Sequence[float]):
    """整颗元件裁片（外扩 `PAD`、放大 `SCALE`）；太小/太大都放弃。"""
    H, W = bgr.shape[:2]
    x0 = max(0, int(round(float(bbox[0]) - PAD)))
    y0 = max(0, int(round(float(bbox[1]) - PAD)))
    x1 = min(W, int(round(float(bbox[2]) + PAD)))
    y1 = min(H, int(round(float(bbox[3]) + PAD)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub = bgr[y0:y1, x0:x1]
    if SCALE > 1:
        sub = cv2.resize(sub, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    return sub


# --------------------------------------------------------------------------- 判定


def repair(tab: Sequence[Tuple[str, str]], slots: Dict[int, Tuple[str, str]]
           ) -> Tuple[Dict[int, str], Dict[str, int]]:
    """表 + 我方逐槽读数 → `{槽下标: 新名字}`（第二个返回值是弃权原因计数）。

    `slots` = `{槽下标: (号, 名)}`（我们自己的读数，可能就是空的）。
    纯函数，`work/_pt_try.py` 用它做离线复刻。
    """
    why: Dict[str, int] = {}
    lut = {num: name for num, name in tab}
    if not lut:
        return {}, {"table_empty": 1}

    agree = dis = 0
    for num, nm in slots.values():
        num, nm = str(num or "").strip(), str(nm or "").strip()
        if num in lut and nm:
            if _same(nm, lut[num]):
                agree += 1
            else:
                dis += 1
    if agree + dis < MIN_VOTES:
        return {}, {"too_few_votes": 1}
    if dis > agree:
        return {}, {"table_shuffled": 1}

    # 号在本元件内出现的次数：只允许改「号唯一」的槽（见文件头第 2 条闸门）。
    cnt: Dict[str, int] = {}
    for num, _nm in slots.values():
        num = str(num or "").strip()
        if num:
            cnt[num] = cnt.get(num, 0) + 1

    out: Dict[int, str] = {}
    for i, (num, nm) in sorted(slots.items()):
        num, nm = str(num or "").strip(), str(nm or "").strip()
        if not num or num not in lut:
            continue
        if cnt.get(num) != 1:                 # 号在本元件内不唯一 ⇒ 不动（见文件头）
            why["num_shared"] = why.get("num_shared", 0) + 1
            continue
        new = lut[num]
        if _norm(new) == _norm(nm):            # 已经一致（含两侧都空）⇒ 不动
            continue
        if not nm:
            why["filled"] = why.get("filled", 0) + 1
        out[int(i)] = new
    if out:
        why["renamed"] = len(out)
    return out, why


def apply(bgr, doc: dict, assigned: dict, client=None, opts: Optional[dict] = None) -> dict:
    """每个已分配的箱体件抄一次表、过闸门改名字；返回统计 dict（`run.py` 打印）。

    `assigned`（`assign()` 的产物）**原地修改**：只改 `comps[cid]["pins"][idx]["name"]`
    的 `text`，**不动坐标、不动号、不增删槽**。端点不可用 / 任何异常 ⇒ 整体 no-op。
    """
    global _PROBED
    stats = {"on": True, "comps": 0, "asked": 0, "renamed": 0,
             "abstain": {}, "error": "", "secs": 0.0}
    if bgr is None or not isinstance(assigned, dict):
        stats["error"] = "no-image"
        return stats
    if client is None:
        try:
            from vlm import client as client                    # noqa: PLC0415
        except Exception as e:                                  # noqa: BLE001
            stats["error"] = "client: %s" % (e,)
            return stats
    if _PROBED is None:
        try:
            _PROBED = bool(client.probe())
        except Exception:                                       # noqa: BLE001
            _PROBED = False
    if not _PROBED:
        stats["error"] = "probe"
        return stats

    timeout = float((opts or {}).get("timeout") or 0) or None
    image_h = ((doc.get("image") or {}).get("h") or None)
    types = {str(c.get("id")): (c.get("type") or "").strip().lower()
             for c in doc.get("components") or []}
    bboxes = {str(c.get("id")): _bbox(c, image_h) for c in doc.get("components") or []}
    a_comps = assigned.get("comps") or {}
    for cid, comp in a_comps.items():
        cid = str(cid)
        if types.get(cid, "") not in TYPES:
            continue
        pins = (comp or {}).get("pins") or {}
        if not pins:
            continue
        bbox = bboxes.get(cid) or []
        if len(bbox) < 4:
            continue
        stats["comps"] += 1
        sub = _crop(bgr, bbox)
        if sub is None:
            stats["abstain"]["crop"] = stats["abstain"].get("crop", 0) + 1
            continue
        try:
            txt, secs = client.ask(sub, PROMPT, timeout=timeout)
        except Exception as e:                                  # noqa: BLE001
            stats["abstain"]["ask_error"] = stats["abstain"].get("ask_error", 0) + 1
            stats["error"] = "%s: %s" % (type(e).__name__, e)
            continue
        stats["asked"] += 1
        stats["secs"] += float(secs or 0.0)
        rows = parse(txt)
        if not rows:
            stats["abstain"]["rows_0"] = stats["abstain"].get("rows_0", 0) + 1
            continue
        slots = {}
        for i, rec in pins.items():
            rec = rec or {}
            slots[int(i)] = (((rec.get("number") or {}).get("text") or "").strip(),
                             ((rec.get("name") or {}).get("text") or "").strip())
        new, why = repair(rows, slots)
        for k, v in why.items():
            if k != "renamed":
                stats["abstain"][k] = stats["abstain"].get(k, 0) + int(v)
        for i, name in sorted(new.items()):
            rec = pins.get(i)
            if rec is None:
                rec = pins.get(str(i))
            if rec is None:                       # 槽不存在 ⇒ 绝不新增
                continue
            old = rec.get("name") or {}
            conf = float(old.get("conf") or 1.0) if isinstance(old, dict) else 1.0
            rec["name"] = {"text": name, "conf": conf,
                           "box": list(old.get("box") or []) if isinstance(old, dict) else [],
                           "d": 0.0, "pin_table": True}
            rec["pin_table"] = True
            stats["renamed"] += 1
    return stats
