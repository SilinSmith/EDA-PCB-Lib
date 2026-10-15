# -*- coding: utf-8 -*-
"""元件「值」文本的类型内规范化（`FUSION_VALUE_FIX`，默认 **1** = 开；`0` 关，A/B 用）。

## 病：离线档的值里有一类是**物理上不可能**的读数

`-offline` 档的值是 CRNN 原文（`stage2/assemble.py:1332-1343` 只在 `ctype in VALUE_TYPES`
且 `VALUE_LIKE.search(vv)` 时取 `info["value"]["text"]`）。官方 10 张离线档 11 条值错里
**6 条**不是「数错」而是「单位读错」，而且错得**物理上不可能**（`work\\_offtext.py`）：

| 案例 | GT | 离线档 | 类型 | 为什么不可能 |
| --- | --- | --- | --- | --- |
| 5 L1 / 5 L2 / 7 L1 / 9 L1 / 10 L1 | `10uH` | `10uF` | `l` 电感 | 电感的单位只能是亨，`F` 是电容的单位 |
| 3 L1 | `3.3uH` | `3.30uH` | `l` 电感 | 尾零是**写法**差，不是读数差 |

剩下 5 条（`10nF←10uF` ×3、`0.1μF←0uF`、`47K←4.7K`）是**真读错**（数字/小数点/前缀），
本条**不碰** —— 猜数字就是在赌博，而单位家族是**由类型决定的约束**，不是猜测。

## 药：两条纯机械规则（只用到类型 + 单位家族）

1. **单位家族必须与类型相容**：`l` / `crystal_*` 里的电感记号只能是亨族 → 末尾单位字母
   `F` 改 `H`；`c` 只能是法族 → 末尾 `H` 改 `F`。前缀（`p n u μ m`）不动。
2. **数字部分的尾零规范化**：`3.30` → `3.3`、`0.10` → `0.1`（GT 的写法是去尾零的）。

两条都只在**严格单值**形态上做（`^[0-9]+(\\.[0-9]+)?[前缀]?[单位]$`，整串必须吃干净）；
带 `/`、空格、多段（`100nF/50V`）、或任何其它形态**一律不碰**。

## 守卫

- 只动 `ctype in VALUE_TYPES`（`r/c/l/fuse/crystal_*`；`gnd` 等**永不写值**，见
  `stage2/assemble.py:29-30` 与 `:878-883`）；
- 改完**必须变短或等长**（只可能删尾零 / 换一个字母），不做任何插入；
- 旧值留 `value["_old_text"]`，并向 `cvparts.hypo.LEDGER` 记一条 `value_fix`（留痕）。

## 实测（A/B，2026-10-20 转正默认 1；`docs/RESULTS.md` §5.44）

| 臂 | 官方 10 张（`official` 尺） | protocol2 | 派生 18 张 | 触发 |
| --- | --- | --- | --- | --- |
| `-offline`（含 ⓪.93 型号名吸附） | 60.2329 | 72.00 | 13.9056 | — |
| `-offline` + 值规范 | **60.9822（+0.7493）** | 72.00 | 13.9056（逐位相同） | 官方 10 张 6 条 / 派生 10 条 |
| 默认档（VLM 开） | 65.5141 | 79.66 | — | — |
| 默认档 + 值规范 | 65.5257 | 79.30 | — | 导出 `value` **0 条不同** |

- 逐案（离线）：case3（`3.30uH→3.3uH`）+1.1539、case5（两条 `10uF→10uH`）+2.3077、
  case7 +1.2500、case9 +1.7648、case10 +1.0169，**无一处回退**（`comp` 通道 0.428942 →
  0.453920，pin/conn/wire 三项逐位不变 ⇒ 改动确实只落在「值」上）。
- **VLM 档零影响**：`tok_full` 与 `tok_fullvf` 导出的 `value` 字段**0 条不同** ⇒ 与 ⓪.93
  同理，VLM 的 `value` claim 一旦存在就压过本步，只在「没人读过这个值」时兜底。
- 派生 18 张里 10 条触发但分数逐位不变 —— 因为派生集 ~92% 的元件键本身就没对上，
  值不参与比较 ⇒ **不存在回退风险**。
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

try:                                            # 包内正常路径
    from envflag import env_on
except ImportError:                             # 直接按文件路径加载本模块时
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from envflag import env_on                  # type: ignore

ON = env_on("FUSION_VALUE_FIX", "1")

# 单位家族（末尾字母）→ 哪个类型家族可以用它
FARAD = ("F",)                                  # 法拉：c / crystal_*（去耦、定时电容）
HENRY = ("H",)                                  # 亨：l
LETTERS = "pnu\u03bcm"                          # 前缀：皮/纳/微/毫
# 单位字母（含欧姆族 `R/K/M/k` 与 Ω/Ω）。**只用来判「这个尾巴是不是一个已知单位」**：
# 欧姆族对 r/fuse 合法，但电容器件读到 `K` 的话是「不属于任何物理家族」⇒ 不猜、不动。
UNITS = "FHRKkM\u03a9\u2126"

# 严格单值：数字 + 前缀? + 单位（整串吃干净）
_SINGLE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([" + LETTERS + r"]?)([" + UNITS + r"]?)$")

# 类型 → 允许的单位家族（**只有 `c` 与 `l` 有物理上唯一的家族**；晶振的值是频率，
# 尾部字母不可推，故不给家族规则、只做尾零规范化）
def _family(ctype: str) -> Tuple[str, ...]:
    t = (ctype or "").strip().lower()
    if t == "l":
        return HENRY
    if t == "c":
        return FARAD
    return ()                                   # r / fuse / crystal_*：不碰单位


def _trim_num(num: str) -> str:
    """去尾零：`3.30`→`3.3`、`0.10`→`0.1`、`1.00`→`1`（只可能变短）。"""
    if "." not in num:
        return num
    out = num.rstrip("0").rstrip(".")
    return out or num


def fix_one(text: str, ctype: str) -> Tuple[Optional[str], str]:
    """`(新值 | None, 理由)`。`None` = 不动。"""
    s = str(text or "").strip()
    if not s:
        return None, "空"
    m = _SINGLE.match(s)
    if not m:
        return None, "非严格单值形态（带 / 空格或多段）⇒ 不碰"
    num, pre, unit = m.group(1), m.group(2), m.group(3)
    fam = _family(ctype)
    new_num, new_unit, why = _trim_num(num), unit, []
    if new_num != num:
        why.append("去数字尾零")
    if fam and unit and unit not in fam:
        # 单位家族不相容 ⇒ 换成该类型唯一的合法单位字母（前缀不动）
        if unit in FARAD or unit in HENRY:
            new_unit = fam[0]
            why.append("单位家族与类型不符（%s → %s）" % (unit, fam[0]))
        else:
            return None, "单位 %r 不属于任何物理家族 ⇒ 不猜" % unit
    if not why:
        return None, "本就规范"
    if not new_unit and fam and not unit:
        return None, "类型有单位家族但读数没有单位 ⇒ 不猜"
    out = new_num + pre + new_unit
    if len(out) > len(s):
        return None, "规范化后变长 ⇒ 放弃"
    return out, "；".join(why)


def fix_assigned(assigned: dict, s1_doc: Optional[dict] = None, log=None) -> dict:
    """就地把 `assigned["comps"][cid]["value"]["text"]` 规范化。"""
    rep: dict = {"on": bool(ON), "n_seen": 0, "n_fix": 0, "n_skip": 0,
                 "n_other": 0, "rows": []}
    if not ON:
        return rep
    types: Dict[str, str] = {}
    if isinstance(s1_doc, dict):
        comps = s1_doc.get("components")
        if isinstance(comps, dict):
            comps = list(comps.values())
        for c in (comps or []):
            if isinstance(c, dict) and c.get("id"):
                types[str(c["id"])] = str(c.get("type") or "").strip().lower()
    try:
        from cvparts.hypo import LEDGER
    except Exception:                            # noqa: BLE001
        LEDGER = None
    for cid, comp in (assigned.get("comps") or {}).items():
        if not isinstance(comp, dict):
            continue
        val = comp.get("value")
        if not isinstance(val, dict):
            continue
        txt = str(val.get("text") or "").strip()
        rep["n_seen"] += 1
        ctype = types.get(str(cid), "")
        if ctype and ctype not in ("r", "c", "l", "fuse", "crystal_2pin",
                                   "crystal_3pin", "crystal_4pin", "crystal"):
            rep["n_other"] += 1
            continue
        if not txt:
            rep["n_skip"] += 1
            continue
        new, why = fix_one(txt, ctype)
        if not new:
            rep["n_skip"] += 1
            if len(rep["rows"]) < 6 and "不碰" not in why and "不猜" not in why:
                rep["rows"].append({"cid": cid, "type": ctype, "old": txt,
                                    "new": None, "why": why})
            continue
        val["_old_text"] = txt
        val["text"] = new
        rep["n_fix"] += 1
        if len(rep["rows"]) < 6:
            rep["rows"].append({"cid": cid, "type": ctype, "old": txt,
                                "new": new, "why": why})
        if log:
            log("  [value_fix] %s(%s): %r → %r（%s）" % (cid, ctype, txt, new, why))
        if LEDGER is not None:
            try:
                LEDGER.record("value_fix", "comp.%s.value" % cid, before=[txt],
                              after=[new], reason=why, by="fusion.value_fix")
            except Exception:                    # noqa: BLE001
                pass
    return rep
