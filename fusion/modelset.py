# -*- coding: utf-8 -*-
"""用**读到的引脚名集合**去官方符号库反推 IC 型号名（`FUSION_MODEL_SET`，默认 **1** = 开；`0` 关，A/B 用）。

## 病：名字读歪 ⇒ 库整条路线作废，而且救不回来（不只是「近失串」）

`fusion/namefix.py`（§5.43）只能救**编辑距离 ≤2 的近失串**。离线档还有一批名字是
**另一种错误**：CRNN 把框内的**引脚功能名**或别的文字当成了型号名 —— 官方 10 张实测
`1 U1 = 'VREFA+'`（GT `MSP432E401YTPDT`）、`8 U1 = 'NRESET'`（GT `PAW3395DM-T6QU`）、
`5 U2 = 'TPSW1240DR'`（GT `TPS61240DRVR`）、`4 U2 = '15ND'`（GT `AMS1117-3.3`）…
这类串到最近库键的距离 ≥3，Levenshtein 一条都吸不回来。

代价最大的恰恰是它们：`1 U1` 一颗就吃掉 **11 根 GT 引脚**（`work\\_pinlib2.py` 归因：
离线 10 张缺的 38 根 box 引脚里 **18 根**是「只差型号名」—— 库里明明有那个号，
只是我们查库的键是错的；另有 12 根是图上名和号**都没读出来**、6 根 GT 型号根本不在库）。

## 药：引脚名集合本身就是指纹

库（`data/BoxICLibrary_200cases.json`，360 键）每个键带整张符号的引脚表。元件框里读到的
**名**（`GND`/`VDDC`/`VBAT`…）与**号**，就是这颗 IC 的指纹：拿它们去库里找命中最多、
且**唯一最优**的键。读数（`work\\_modelset.py`，离线臂 `tok_offvf`，官方 10 张）：

| 元件 | 我们的名 | GT 名 | 读到/丢掉的非数字名 | 最优（名命中/号命中） | 次优 | 判词 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 U1 | `VREFA+` | `MSP432E401YTPDT` | GND×7/GNDA/VBAT/VDDC×2 | **MSP432E401YTPDT(5/17)** | STM32F103C8T6(3/8) | 唯一 ⇒ 动 |
| 5 U2 | `TPSW1240DR` | `TPS61240DRVR` | VOUT/FB/P/GND/EN/VIN/L | **TPS61240DRVR(6/6)** | TPS61089RNRR(5/6) | 唯一 ⇒ 动 |
| 8 U1 | `NRESET` | `PAW3395DM-T6QU` | MOS/MOTION/LED/P/MSO/NESE/NRESE/SCLK/NC×2 | **PAW3395DM-T6QU(9/11)** | 2.4GHz(4/14) | 唯一 ⇒ 动 |
| 5 U3 | `VOUTAD` | `AMS1117-3.3` | Q/TAB/VIN/ADIGND/Q | AMS1117-3.3(2/4) | AMS1117-5.0(2/4) | 打平 ⇒ 不动 |
| 2 U4 | `HSWLO3W2C` | `HS96L03W2C03` | SDA/VCC/GND/SCL | BH1750(4/3) | HS96L03W2C03(4/3) | 打平 ⇒ 不动 |
| 4 U1/U2/U3、10 U2 | … | … | … | 最优都**不是** GT | … | 不动 |

⇒ 只动「名字命中数**至少 `MIN` 个**、且**严格**优于次优 ≥`MARGIN`」的那些（默认 2 / 1）。
`HSWLO3W2C`→`HS96L03W2C03` 这类打平的不动 —— 宁可留着错，也不赌。

## 守卫（与 `namefix` 同形）

1. 只动 `box/boxic/ic`；
2. 读出来的串**已经是库键**（精确或归一化命中）⇒ 一个字符都不动（读对的不可能被改歪）；
3. 名命中数 `sn ≥ FUSION_MODEL_SET_MIN`（默认 2），且 `sn_best - sn_second ≥ FUSION_MODEL_SET_MARGIN`
   （默认 **2**）—— **要求名严格占优**，号只当并列时的旁证（不作为单独动手理由）；
4. 纯数字的「名」（`1`/`2`/`3`）不进指纹 —— 库键里遍地都是，没有信息量；
5. 只改 `partname.text`，旧值留 `partname["_old_text"]`，`cvparts.hypo.LEDGER` 记
   `model_set`（含旧名/新名/命中数与次优键，便于复核）。

## 实测（A/B，2026-10-21 转正默认 1 + MARGIN 2；`docs/RESULTS.md` §5.45）

| 臂 | 官方 10 张（`official` 尺） | 派生 18 张 | VLM 档命中 |
| --- | --- | --- | --- |
| 关（基线） | 60.9822 | 13.9056 | — |
| MARGIN=1 | 62.0627（+1.0805） | — | `tok_full`/`tok_lean` 各 **3** 颗 |
| **MARGIN=2（转正）** | **62.0224（+1.0402）** | **14.6526（+0.7470）** | `tok_full`/`tok_lean` **0** 颗 |

- 生效的只有「名命中数严格占优 ≥2」的元件：离线官方 10 张里是 `1 U1 VREFA+ → MSP432E401YTPDT`
  （+6.5166）与 `8 U1 NRESET → PAW3395DM-T6QU`（+3.8855），两把尺子都为正。
- MARGIN=1 多出来的 case 5/6/7/10 四颗是「3 正确 3 错误」的赌博，净贡献只有 +0.04 分；
  而 MARGIN=2 让 VLM 档**结构性**不动手（那两档的型号名本来就被 VLM 读对、已是库键）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:                                            # 包内正常路径
    from envflag import env_on, env_str
except ImportError:                             # 直接按文件路径加载本模块时
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from envflag import env_on, env_str          # type: ignore

try:
    from stage2 import libpins as LB
except ImportError:                              # pragma: no cover
    import sys as _sys2

    _sys2.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stage2 import libpins as LB             # type: ignore

# `os.environ.get` 一律直写：`tools/dev/gen_knobs.py` 的正则只认它 / `envflag.*`。
# ★ 2026-10-21 转正（默认 0 → 1）读数：官方 10 张 60.9822 → **62.0224（+1.0402）**、
#   派生 18 张 13.9056 → **14.6526（+0.7470）**（两把尺子都为正）。
# ★ `MARGIN` 默认同时 1 → **2**：MARGIN=1 在官方 10 张上多 +0.04，但 VLM 档（`tok_full`/
#   `tok_lean`）会动 3 颗「把错名换成另一个错名」的元件；MARGIN=2 时那两档命中 **0**（实测），
#   只留 case 1 / case 8 两颗唯一且正确的大件 ⇒ 用 0.04 分换「VLM 各档结构性零影响」。
ON = env_on("FUSION_MODEL_SET", "1")
MIN = int(os.environ.get("FUSION_MODEL_SET_MIN", "2") or "2")
MARGIN = int(os.environ.get("FUSION_MODEL_SET_MARGIN", "2") or "2")
# `FUSION_MODEL_SET_TIE`（默认 **0**，A/B 用；§5.465）：名命中**打平**时允许出手，但只允许
# 在「打平的那两个库键，就我们读到的那些**号**给出的**名**完全一致」时。病（`open_GT` case5
# `U3`）：`AMS1117-3.3` 与 `AMS1117-5.0` 各命中 2 个名 ⇒ 按 MARGIN 的守卫不出手，于是
# `Name` 留着错的 `VOUTAD`、脚名留着 `ADIGN`/`VIN`（GT `ADJ(GND)`/`VOUT(TAB)`）—— 这两个
# 变体的**引脚表在图上完全一样**，取哪个键对「名/号」没有任何区别，唯一不确定的只是 `Name`
# 那串后缀，所以可以按名排序取第一个（确定性），而不是一律不动。
TIE = env_on("FUSION_MODEL_SET_TIE", "0")
BOX = {"box", "boxic", "ic"}

_IDX: Optional[Dict[str, dict]] = None
_NORMK: Optional[Dict[str, str]] = None
_LIB_INFO = {"path": "", "n": 0, "err": ""}


def index() -> Dict[str, dict]:
    """`库键 → {'names': set(norm(name)), 'nums': set(str(number))}`（惰性，读不动就空）。"""
    global _IDX, _NORMK
    if _IDX is not None:
        return _IDX
    import json

    try:
        p = LB.lib_path()
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        _LIB_INFO.update({"path": str(p), "n": len(raw), "err": ""})
        idx: Dict[str, dict] = {}
        nk: Dict[str, str] = {}
        for k, v in (raw or {}).items():
            names: Set[str] = set()
            nums: Set[str] = set()
            bynum: Dict[str, str] = {}
            for pin in ((v or {}).get("pins") or []):
                nm = LB.norm((pin or {}).get("name"))
                nu = str((pin or {}).get("number") or "").strip()
                if nm:
                    names.add(nm)
                if nu:
                    nums.add(nu)
                    if nm:
                        bynum.setdefault(nu, nm)
            idx[str(k)] = {"names": names, "nums": nums, "bynum": bynum}
            nk.setdefault(LB.norm(k), str(k))
        _IDX = idx
        _NORMK = nk
    except Exception as exc:                     # noqa: BLE001 —— 库缺席不该炸管线
        _LIB_INFO.update({"err": "%s: %s" % (type(exc).__name__, exc)})
        _IDX = {}
        _NORMK = {}
    return _IDX


def read_names(comp: dict) -> Tuple[Set[str], Set[str]]:
    """从 `assigned` 的元件记录里取「读到的名 / 读到的号」（都归一化；纯数字名丢掉）。"""
    names: Set[str] = set()
    nums: Set[str] = set()
    for st in ((comp or {}).get("pins") or {}).values():
        if not isinstance(st, dict):
            continue
        nm = st.get("name")
        t = str(nm.get("text") or "").strip() if isinstance(nm, dict) else ""
        if t and not t.isdigit():
            names.add(LB.norm(t))
        nu = st.get("number")
        u = str(nu.get("text") or "").strip() if isinstance(nu, dict) else ""
        if u:
            nums.add(u)
    return names, nums


def infer_one(comp: dict, idx: Optional[Dict[str, dict]] = None
              ) -> Tuple[Optional[str], int, int, Optional[str], int]:
    """`(库键 | None, 名命中, 号命中, 次优键, 次优名命中)`。

    只在「名命中 ≥ MIN 且严格优于次优 ≥ MARGIN」时返回库键。
    """
    idx = index() if idx is None else idx
    if not idx:
        return None, 0, 0, None, 0
    rn, ru = read_names(comp)
    if not rn:
        return None, 0, 0, None, 0
    cands: List[Tuple[int, int, str]] = []
    for k, e in idx.items():
        sn = len(rn & e["names"])
        if not sn:
            continue
        cands.append((sn, len(ru & e["nums"]), k))
    if not cands:
        return None, 0, 0, None, 0
    cands.sort(key=lambda t: (-t[0], -t[1], t[2]))
    sn, su, key = cands[0]
    sn2 = cands[1][0] if len(cands) > 1 else 0
    key2 = cands[1][2] if len(cands) > 1 else None
    if sn >= max(MIN, 1) and (sn - sn2) >= max(MARGIN, 1):
        return key, sn, su, key2, sn2
    # 打平档（`FUSION_MODEL_SET_TIE`，默认 0）：只在「两个候选键就我们读到的号给出的名
    # 完全一致」时出手（见常量区的注释）。
    if TIE and key2 and sn >= max(MIN, 1) and sn == sn2:
        e1, e2 = idx.get(key) or {}, idx.get(key2) or {}
        b1, b2 = e1.get("bynum") or {}, e2.get("bynum") or {}
        seen = [u for u in ru if u in b1 and u in b2]
        if seen and all(b1[u] == b2[u] for u in seen):
            return key, sn, su, key2, sn2
    return None, sn, su, key2, sn2


def snap_assigned(assigned: dict, s1_doc: Optional[dict] = None,
                  log=None) -> dict:
    """就地把 `assigned["comps"][cid]["partname"]["text"]` 换成推断出的库键。"""
    rep: dict = {"on": bool(ON), "min": MIN, "margin": MARGIN,
                 "n_seen": 0, "n_have": 0, "n_names": 0, "n_hit": 0,
                 "n_skip": 0, "n_other": 0, "rows": [], "lib": dict(_LIB_INFO)}
    if not ON:
        return rep
    idx = index()
    rep["lib"] = dict(_LIB_INFO)
    if not idx:
        rep["err"] = "库不可用"
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
        pn = comp.get("partname")
        rep["n_seen"] += 1
        ctype = types.get(str(cid), "")
        if ctype and ctype not in BOX:
            rep["n_other"] += 1
            continue
        txt = str((pn or {}).get("text") or "").strip() if isinstance(pn, dict) else ""
        if not isinstance(pn, dict):
            rep["n_skip"] += 1
            continue
        if txt and LB.norm(txt) in (_NORMK or {}):
            rep["n_have"] += 1                       # 已经是库键 ⇒ 一个字都不动
            continue
        hit, sn, su, key2, sn2 = infer_one(comp, idx)
        if not hit:
            rep["n_names"] += 1 if sn else 0
            rep["n_skip"] += 1
            if sn and len(rep["rows"]) < 8:
                rep["rows"].append({"cid": cid, "old": txt, "new": None,
                                    "sn": sn, "su": su, "second": key2,
                                    "sn2": sn2, "why": "命中不足/不唯一"})
            continue
        pn["_old_text"] = txt
        pn["text"] = hit
        pn["_by"] = "fusion:model_set"
        rep["n_hit"] += 1
        if len(rep["rows"]) < 8:
            rep["rows"].append({"cid": cid, "old": txt, "new": hit, "sn": sn,
                                "su": su, "second": key2, "sn2": sn2})
        if log:
            log("  [model_set] %s: %r → %s（名命中 %d vs 次优 %s %d，号命中 %d）"
                % (cid, txt, hit, sn, key2 or "-", sn2, su))
        if LEDGER is not None:
            try:
                LEDGER.record("model_set", "comp.%s.partname" % cid, before=[txt],
                              after=[hit],
                              reason=("引脚名集合对齐库：命中 %d 名 / %d 号，次优 %s(%d)"
                                      % (sn, su, key2 or "-", sn2)),
                              by="fusion.model_set")
            except Exception:                    # noqa: BLE001
                pass
    return rep
