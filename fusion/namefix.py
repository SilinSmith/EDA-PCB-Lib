# -*- coding: utf-8 -*-
"""IC 型号名「吸附」到官方符号库闭词表（`FUSION_NAME_SNAP`，默认 **1** = 开；`0` 关，A/B 用）。

## 病：离线档的名字是 CRNN 原文，库路线整条死掉

`--no-vlm` / `-offline` 档里 box 的 `Name` 直接来自 `assigned[...]["partname"]["text"]`
（`stage2/assemble.py:1334`），也就是**没经过 VLM 校准的 CRNN 原文**。官方 10 张实测
（`work\\_offtext.py`）：21 个 box 元件里只有 **1** 个名字对，另 20 个是近失串 ——
`HSWL03W2C03` ← `HS96L03W2C03`、`SY1813BADC` ← `SY8113BADC`、`XC8205P32MR` ← `XC6206P332MR`、
`VREFA+`（把框内的引脚功能名当成了型号）…

后果**不止名字那一项分**：`stage2/pinlib.repair_slots`（库给号、图给序，官方 10 张
同名组 93.6% 可复现）与 `fusion/librefine`（`(型号, 号) → 库名`）都靠
`libpins.key_for(型号)` 拿库键 —— 型号读歪 = **整条库路线死掉**。证据：

| 档 | `pin_libseq.n_key`（10 张） | `librefine.n_key` |
| --- | --- | --- |
| 默认档 `tok_full` | 1,4,1,3,3,2,1,1,1,2（合计 19） | 26,15,6,10,18,4,0,17,6,11 |
| 离线档 `tok_off` | **0 ×8**、2、1 | **0 ×8**、4、6 |

## 药：官方符号库是闭词表（360 键），把近失串吸附回来

`work\\_namesnap.py`（离线重放 + `official` 尺）：21 个 box 里已对 1 / **修好 6** /
未动 12（其中 2 个吸到了别的库键 —— 但**基线本来也是错的**，不算回退）⇒
官方 10 张 **59.1761 → 59.8993（+0.7232）**。

本步放在 `fusion/run.py` 里 `PB.repair_slots` **之前**，所以名字一修好，库给号 / 库改名
两条通道会在同一张图上跟着活过来（那部分收益是额外算的）。

## 守卫（宁可不吸，也不要把读对的串改歪）

1. 只动 `box/boxic/ic`（型号名只对这类有意义；阻容的 `Name` = 位号，绝不能碰）；
2. 读出来的串**已经是库键**（精确或归一化命中）⇒ 一个字符都不动；
3. 归一化 Levenshtein 距离 `d ≤ FUSION_NAME_SNAP_TOL`（默认 2）**且**最优严格优于次优
   （`d2 - d ≥ 1`）—— 两个库键一样近时宁可不动（`79WM12` 对 `78M12`/`79M12` 就是这种）；
4. 读数归一化长度 ≥ `FUSION_NAME_SNAP_MINLEN`（默认 4，挡住 `SC1L` 这种碎片）；
5. 只改 `partname.text`，旧文本留在 `partname["_old_text"]` 可查，并往
   `cvparts.hypo.LEDGER` 记一条 `name_snap`（`docs/reports/工程哲学.md` §94：
   改写要留痕、不静默丢）。

## 实测（A/B，2026-10-20 转正默认 1；`docs/RESULTS.md` §5.43）

| 臂 | 官方 10 张（`official` 尺） | protocol2 | 派生 18 张 |
| --- | --- | --- | --- |
| `-offline` | 59.1761 | 70.04 | 13.7885 |
| `-offline` + 吸附 | **60.2329（+1.0568）** | **72.00（+1.96）** | **13.9056（+0.1171）** |
| 默认档（VLM 开） | 65.5141 | 79.66 | — |
| 默认档 + 吸附 | 65.5257 | 79.30 | — |

- **落位决定成败**：第一版把本步放在 `rep["iterate"]` 之后、`PB.repair_slots` 之前，结果
  `pin_libseq`/`librefine` 确实活了，但**导出的 `Name` 一个字没变** ⇒ 只 +0.3336。因为
  `CL.from_stage2`（`fusion/run.py:300`）在吸附之前就把 `comp.<cid>.partname` 的候选值从
  `assigned` 抄进假设池，而 `fusion/doc.py:195-222` 的 claim 分支又把旧值写回 `build_doc`
  的 `Name`。⇒ 必须放在 `run_case` 里 `assigned = _load(...)` 之后、`assigned0` deepcopy
  **之前**，让池子/底稿/claim 三处看到同一份吸附后的文本。
- **可加性**：纯名字重放（`work\\_namesnap.py`）= +0.7232，库通道复活 = +0.3336，两效应相加
  = +1.0568 ✓（独立可加）。
- **VLM 档零影响（不是"影响很小"，是逐位相同）**：`-vlm-name off` 与 `-lean` 两臂开关前后
  四个通道**逐位相等**（61.0941 / 64.4768 与各自基线完全一致），`tok_full` 与 `tok_fullsnap`
  导出的 21 个 box `Name` 逐条相同且全对 ⇒ VLM 的 `partname` claim 一旦存在就压过本步，
  **本步只在"没人读过这个名字"时兜底**，不可能把读对的名字改歪。
- `box` 名字对错（`work\\_offtext.py`，离线 21 个）：错 20 → **14**、对 1 → **7**；
  `pin_libseq.n_key` 合计 4 → 9、`librefine.n_key` 合计 14 → 40。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# `os.environ.get` 一律直写：`tools/dev/gen_knobs.py` 的正则只认它 / `envflag.*`，
# 包一层就会让开关从 `docs/KNOBS.md` 里整条消失（同 `stage2/pinlib.py:93` 的记）。
ON = env_on("FUSION_NAME_SNAP", "1")
TOL = int(os.environ.get("FUSION_NAME_SNAP_TOL", "2") or "2")
MINLEN = int(os.environ.get("FUSION_NAME_SNAP_MINLEN", "4") or "4")
BOX = {"box", "boxic", "ic"}

_KEYS: Optional[List[str]] = None
_NORM: Optional[Dict[str, str]] = None
_KEYS_NORM: Optional[List[str]] = None
_LIB_INFO = {"path": "", "n": 0, "err": ""}


def lib_keys() -> Tuple[List[str], Dict[str, str], List[str]]:
    """库键 / `norm(键) → 键` / 归一化键表（惰性，读不动就返回空）。"""
    global _KEYS, _NORM, _KEYS_NORM
    if _KEYS is not None:
        return _KEYS, _NORM or {}, _KEYS_NORM or []
    import json

    try:
        p = LB.lib_path()
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        _LIB_INFO.update({"path": str(p), "n": len(raw), "err": ""})
        _KEYS = [str(k) for k in raw]
    except Exception as exc:                     # noqa: BLE001 —— 库缺席不该炸管线
        _LIB_INFO.update({"err": "%s: %s" % (type(exc).__name__, exc)})
        _KEYS = []
    _NORM = {}
    for k in _KEYS:
        _NORM.setdefault(LB.norm(k), k)
    _KEYS_NORM = list(_NORM)
    return _KEYS, _NORM, _KEYS_NORM


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_key(text: str, tol: int = None) -> Tuple[Optional[str], int, int]:
    """`(吸附到的库键 | None, 最优距离, 次优距离)`（全部按 `libpins.norm` 归一化后比）。"""
    keys, _nm, kn = lib_keys()
    if not keys:
        return None, 999, 999
    t = LB.norm(text)
    if not t:
        return None, 999, 999
    if t in (_NORM or {}):                       # 已经是库键 ⇒ 不动
        return None, 0, 0
    tol = TOL if tol is None else tol
    scored = sorted((_lev(t, k), k) for k in kn)
    d, best = scored[0]
    d2 = scored[1][0] if len(scored) > 1 else 999
    if d <= tol and (d2 - d) >= 1 and len(t) >= MINLEN:
        return (_NORM or {})[best], d, d2
    return None, d, d2


def snap_assigned(assigned: dict, s1_doc: Optional[dict] = None,
                  log=None) -> dict:
    """就地把 `assigned["comps"][cid]["partname"]["text"]` 吸附到库键上。

    `s1_doc` 用来判元件类型（stage1 的 `components` 是**列表**，`id ↔ type`）；
    没有它就只处理「`partname.text` 命中库键归一化前缀」的元件（保守退化）。
    """
    rep: dict = {"on": bool(ON), "tol": TOL, "minlen": MINLEN,
                 "n_seen": 0, "n_have": 0, "n_lib": 0, "n_snap": 0,
                 "n_skip": 0, "n_other": 0, "rows": [], "lib": dict(_LIB_INFO)}
    if not ON:
        return rep
    keys, _nm, _kn = lib_keys()
    rep["lib"] = dict(_LIB_INFO)
    if not keys:
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
        if not isinstance(pn, dict):
            continue
        txt = str(pn.get("text") or "").strip()
        rep["n_seen"] += 1
        ctype = types.get(str(cid), "")
        if ctype and ctype not in BOX:
            rep["n_other"] += 1
            continue
        if not txt:
            rep["n_skip"] += 1
            continue
        if LB.norm(txt) in (_NORM or {}):
            rep["n_have"] += 1
            continue
        hit, d, d2 = best_key(txt)
        if not hit:
            rep["n_skip"] += 1
            if len(rep["rows"]) < 6:
                rep["rows"].append({"cid": cid, "old": txt, "d": d, "d2": d2,
                                    "new": None, "why": "距离/裕度不足"})
            continue
        pn["_old_text"] = txt
        pn["text"] = hit
        rep["n_snap"] += 1
        if len(rep["rows"]) < 6:
            rep["rows"].append({"cid": cid, "old": txt, "new": hit, "d": d,
                                "d2": d2, "old_text_kept": True})
        if log:
            log("  [name_snap] %s: %r → %s (d=%d, d2=%d)" % (cid, txt, hit, d, d2))
        if LEDGER is not None:
            try:
                LEDGER.record("name_snap", "comp.%s.partname" % cid, before=[txt],
                              after=[hit],
                              reason="OCR 近失串吸附到官方符号库闭词表（d=%d）" % d,
                              by="fusion.name_snap")
            except Exception:                    # noqa: BLE001
                pass
    return rep
