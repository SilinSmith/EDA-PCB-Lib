# -*- coding: utf-8 -*-
"""引脚号「库给多值、图给顺序」回填（`FUSION_PIN_LIBSEQ`）——**在融合层做**。

## 病（官方 10 张、protocol2 尺、`wddef` 基线 78.07；逐脚取证 `work\_pincmp.py --run work\_fast_out_wddef --case 1 --comp U1`）

★ 先说清楚**病的主要形态不是「读错」而是「根本没 emit」**：交付档（VLM 开）case 1 的
`U1`（MSP432E401YTPDT，GT 26 个脚）我方只有 **19** 个脚，缺口是

  * **7 根完全没进文档**（`114 GND` / `90 VDD` / `113 VDD` / `8 VDDA` / `87 VDDC` /
    `115 VDDC` / `9 VREFA+`）：CRNN 少读（前导 1 被吃掉：号框实测宽 34px = 3 个字、
    识别出的串只有 2 个字；或干脆没读数）⇒ `stage2/assemble.py` 的
    `BOX_PIN_DROP_UNREAD`（`:1141-1143`）把「没读到合法号」的桩**整条丢掉**；
  * **1 根号错**：`46`（GT `48 GND`）—— 而库规则 B（`:1162-1172`）还会把它的名字
    一起带偏成 `PF4`（库 `name_for_num(46)`）；
  * **1 根号成了非数字**：`S0`（GT `90 VDD`）。

误读/丢桩一根脚同时吃 **fp + fn** 两次罚分（真值号变 `missing`、错号变 `extra`），
所以这是引脚通道里最贵的一批。另：`--no-vlm` 档上误读更多（`114→14`、`113→13`、
`115→15`、`87→67`、`68→66`），但那不是交付档的形态。

## 机制：库给多值、图给顺序

库里**同名引脚的号集合**是已知的（`MSP432E401YTPDT` 的 `GND = {17,48,55,58,80,114}`、
`VDD = {7,16,26,…,122}`、`VDDC = {87,115}`），而图上这些同名脚**自上而下恰好是库里
号的升序**。GT 实测（`work\\_liborder.py`，官方 10 张）：box 元件的 **94 个同名组里
88 个（93.6%）** 能逐脚复现 GT。

## 为什么在融合层而不是阶段二

阶段二 `assign()` 跑在 **VLM 之前**（`submission/run.py` 顺序：阶段一/二 → 阶段三 →
VLM 补强 → 融合层），那时 `partname` 还没被 VLM 修好 —— `--no-vlm` 下 case 1 的 `U1`
型号名读成 `VREFA+`（错），`key_for` 拿不到库键，整条路是死的（实测：接在 `assign()`
里时官方 10 张**逐位不变**）。融合层这一步（`fusion/run.py` 的 ⑩ 库名回填）拿到的
`doc["components"][cid]["Name"]` 才是 VLM 校准过的（IC 型号名 21/21 全对），
所以库键在这里命中。

## 保守性（三道守卫 + 两条硬规矩）

只处理「该同名组的槽位数与库里同名脚个数**相等**」的组，且**只动读数不在库集合里的
槽**（空号 / 非数字 / 误读），可读且在库集合里的号**一个都不动**。在此之上三道守卫：

  1. **乱序拒绝**：组内「可读且在库集合里」的号，按槽位 y 升序必须**严格递增**。
     反例 `EG1192L` 的 `VIN`（库 `[1,9]`、图上 `9,1`，符号倒着画）会在这里被挡下
     ⇒ 判成「我读的顺序与库的顺序不一致」，**整组不动**（`MODE=1` 下也是自洽即不动）。
  2. **组外占用拒绝**：候选号若已被**同元件别的槽**读走，整组放弃。实战踩点：
     `case 10 U1 EG1192L` 的 `VB` 组库里是 `6`，而图上 `VR` 那根已经读到 `6` ⇒
     老写法把 `VB` 也改成 `6`，两根脚撞号 ⇒ `build_doc` 少一根脚（引脚数 9 → 8）。
  3. **撞号终检**：写入后整颗元件的号集合若仍有重复，**整颗放弃**（`blocked`）。

分配规则：待填槽位（读数不在集合里）按 y 升序，依次领走剩下的候选号（候选升序）。
此外 **`MODE=1` 时若组内没有可疑读数就不动手**。写入时**就地改文本、保留原有
`box`/`conf`/`d`**（下游 `priors`/`solve`/`claims` 拿这些框做几何先验，抹成 `None`
等于顺手毁掉别的证据），旧读数留在 `_old_text` 里可查。

## 规矩（沿用工程哲学）

只改键与名，绝不增删引脚、绝不碰 `point`/`edges`/`components`；每次改写都往
`cvparts.hypo.LEDGER` 记一条 `kind="libseq"` 的事件（before/after/why），
所以「改写」在旁路账本里可查（见 `docs/reports/工程哲学.md` §94）。
"""
from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:                                        # 包内正常路径
    from . import libpins as _libpins
except ImportError:                         # 直接跑脚本 / 单文件导入
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from stage2 import libpins as _libpins
    except ImportError:
        import libpins as _libpins                      # type: ignore


def _env(name: str, dflt: str) -> str:
    """读字符串开关。

    ★ 一律走 `os.environ.get` 而不是另起一层名字：`tools/dev/gen_knobs.py` 的正则只认
    `os.environ.get(...)` / `envflag.*`，包一层就会让开关从 `docs/KNOBS.md` 里**整条
    消失**（`stage2/libpins.py:76-79` 记过同一个坑）。
    """
    return (os.environ.get(name, dflt) or dflt).strip()


# `FUSION_PIN_LIBSEQ`：0 = 关（历史行为）、1 = 只在组内读数可疑时补/纠（**默认档**，
# 2026-10-15 转正）、2 = 只要脚数相等就整组按库顺序重写（更激进，未定案）。
# 转正读数（`work\_libslot_ab.ps1` / `work\_libslot_val.ps1`，protocol2 尺）：
# 官方 10 张 `wddef2` 78.07 → **78.64（+0.565/张，case 1 一张就 +5.65、其余九张逐位不变）**；
# 派生 19 张 30.32 → **30.54（+0.22，落在 ±0.4 噪声带内但不为负）**。
MODE = int(os.environ.get("FUSION_PIN_LIBSEQ", "1") or "1")
# 同名组名与库名做模糊匹配的阈值（`VDA` ↔ `VDDA` = 0.857）。只在库里查不到该归一化
# 名时才用，且要求「最优明显优于次优」（`VDD` vs `VDDC` 这种近名不能互相抢）。
FUZZY = float(os.environ.get("FUSION_PIN_LIBSEQ_FUZZY", "0.85") or 0.85)
FUZZY_MARGIN = float(os.environ.get("FUSION_PIN_LIBSEQ_MARGIN", "0.06") or 0.06)

_MAPS: Optional[dict] = None


def _sort_nums(nums: Sequence[str]) -> List[str]:
    """按「先短后长、再字典序」排 = 正整数的升序（1,2,…,16,…,101,113,122）。"""
    return sorted(set(str(n) for n in nums), key=lambda s: (len(s), s))


def _build_maps(raw: dict) -> dict:
    out: Dict[str, dict] = {}
    for key, rec in (raw or {}).items():
        pins = (rec or {}).get("pins") if isinstance(rec, dict) else None
        if not pins:
            continue
        byname: Dict[str, List[str]] = {}
        canon: Dict[str, str] = {}
        bynum: Dict[str, str] = {}
        for p in pins:
            if not isinstance(p, dict):
                continue
            nm = str(p.get("name") or "").strip()
            num = str(p.get("number") or "").strip()
            if not nm or not num:
                continue
            n = _libpins.norm(nm)
            byname.setdefault(n, []).append(num)
            canon.setdefault(n, nm)
            bynum.setdefault(num, nm)
        if byname:
            out[str(key)] = {"byname": {k: _sort_nums(v) for k, v in byname.items()},
                             "canon": canon, "bynum": bynum}
    return out


def maps() -> dict:
    """惰性加载 `{库键: {byname, canon, bynum}}`（库不可用返回 `{}`）。"""
    global _MAPS
    if _MAPS is None:
        _MAPS = {}
        try:
            import json
            raw = json.loads(Path(_libpins.lib_path()).read_text(encoding="utf-8"))
            _MAPS = _build_maps(raw)
        except Exception as e:                              # noqa: BLE001
            print("[pinlib] 库不可用（%s）—— 本通道跳过" % e)
    return _MAPS


def _ctx():
    try:
        return _libpins.new_ctx()
    except Exception:                                       # noqa: BLE001
        return None


def _resolve_name(n: str, byname: dict):
    """归一化脚名 → 库里的归一化脚名（精确优先，其次唯一模糊）。"""
    if not n:
        return None
    if n in byname:
        return n
    best_k, best_r, second_r = None, 0.0, 0.0
    for k in byname:
        r = difflib.SequenceMatcher(None, n, k).ratio()
        if r > best_r:
            best_k, second_r, best_r = k, best_r, r
        elif r > second_r:
            second_r = r
    if best_k and best_r >= FUZZY and (best_r - second_r) >= FUZZY_MARGIN:
        return best_k
    return None


def _num_of(pin_key: str) -> str:
    s = str(pin_key or "").strip()
    if s.startswith("pin_"):
        s = s[4:]
    return s


def _y_of(rec: Any) -> float:
    try:
        return float(((rec or {}).get("point") or {}).get("y") or 0.0)
    except Exception:                                       # noqa: BLE001
        return 0.0


def _rewrite_refs(hg: str, cid: str, k2n: Dict[str, str]) -> str:
    """把 `(U1.14,…)` 里的 `U1.14` 换成 `U1.114`（只动本元件、只动整段数字）。"""
    if not hg or not k2n:
        return hg
    pat = re.compile(r"(?<![\w.])%s\.(\d+)" % re.escape(str(cid)))

    def _sub(m):
        old = m.group(1)
        return "%s.%s" % (cid, k2n.get(old, old))

    return pat.sub(_sub, hg)


def repair_doc(doc: dict, ctx=None, mode: Optional[int] = None,
               log=None) -> Dict[str, Any]:
    """就地按「库集合 + 图上顺序」重排 `doc` 里 box 元件的**引脚号键**。

    * 数据来源：`doc["components"][cid]["Name"]`（VLM 校准过的型号名）；
    * 顺序依据：`doc["pins"][cid]["pin_<号>"]["point"]["y"]` —— 文档坐标是
      **y-up**（与 GT 同一帧），所以**号小的在上面 = y 大**，按 y 降序配对库升序；
    * 改键之后同步改写 `doc["nets"][*]["hyperGraph"]` 里的 `(cid.旧号 → cid.新号)`。

    返回统计 dict（含 `rows` = 逐处改动，便于探针/文档引用）。
    """
    m = MODE if mode is None else int(mode)
    st: Dict[str, Any] = {"on": m > 0, "mode": m, "n_comp": 0, "n_key": 0,
                          "n_grp": 0, "n_chg": 0, "rows": [], "blocked": []}
    if m <= 0 or not isinstance(doc, dict):
        return st
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}
    if not comps or not pins:
        return st
    ctx = ctx or _ctx()
    if ctx is None:
        st["err"] = "libpins 不可用"
        return st
    allm = maps()
    nets = doc.get("nets") or {}
    for cid in list(pins.keys()):
        st["n_comp"] += 1
        cname = str(((comps.get(cid) or {}).get("Name")) or "").strip()
        if not cname:
            continue
        try:
            key = ctx.key_for(cname)
        except Exception:                                   # noqa: BLE001
            key = None
        if not key:
            continue
        mp = allm.get(str(key))
        if not mp:
            continue
        st["n_key"] += 1
        byname, canon, bynum = mp["byname"], mp["canon"], mp["bynum"]
        rec = pins.get(cid) or {}
        # 按脚名归一 → 库名 分组
        groups: Dict[str, List[str]] = {}
        cur: Dict[str, str] = {}
        for pk, v in rec.items():
            num = _num_of(pk)
            if not num:
                continue
            cur[num] = pk
            nm = _libpins.norm(str((v or {}).get("pinname") or ""))
            if not nm:
                continue
            rn = _resolve_name(nm, byname)
            if rn:
                groups.setdefault(rn, []).append(pk)
        plan: List[tuple] = []                  # (pin_key, old_num, new_num, libname)
        used_new: Dict[str, str] = {}
        for lname, pkeys in groups.items():
            cand = byname.get(lname) or []
            if not cand or len(cand) != len(pkeys):
                continue
            st["n_grp"] += 1
            olds = [_num_of(pk) for pk in pkeys]
            if olds == list(cand):
                continue
            cset = set(cand)
            suspicious = any((not o) or (o not in cset) for o in olds)
            if m == 1 and not suspicious:
                continue
            order = sorted(pkeys, key=lambda pk: -_y_of(rec.get(pk)))
            pair = list(zip(order, cand))
            clash = [n for _, n in pair if n in used_new]
            if clash:                           # 跨组抢同一个号 ⇒ 整组放弃
                st["blocked"].append({"cid": cid, "grp": lname, "dup": clash})
                continue
            for pk, n in pair:
                used_new[n] = pk
            for pk, n in pair:
                plan.append((pk, _num_of(pk), n, bynum.get(n) or canon.get(lname)))

        # 冲突体检：最终键集合里不许出现重复数字
        final: Dict[str, str] = {}
        for pk, old, new, _ln in plan:
            final[new] = pk
        if len(final) != len(plan):
            st["blocked"].append({"cid": cid, "grp": "*", "dup": "final"})
            continue

        k2n: Dict[str, str] = {}
        for pk, old, new, ln in plan:
            if old == new:
                continue
            k2n[old] = new
        if not k2n:
            continue
        newrec: Dict[str, Any] = {}
        for pk, v in rec.items():
            num = _num_of(pk)
            nn = k2n.get(num, num)
            nv = dict(v or {})
            for _pk, _old, _new, _ln in plan:
                if _new == nn and _ln and _libpins.norm(str(nv.get("pinname") or "")) \
                        != _libpins.norm(_ln):
                    nv["pinname"] = _ln          # 号改了，名跟着库走（VDA → VDDA）
                    break
            newrec["pin_%s" % nn] = nv
        pins[cid] = newrec
        for nk, nv in nets.items():
            if isinstance(nv, dict) and nv.get("hyperGraph"):
                nv["hyperGraph"] = _rewrite_refs(str(nv["hyperGraph"]), cid, k2n)
        for pk, old, new, ln in plan:
            if old == new:
                continue
            st["n_chg"] += 1
            row = {"cid": cid, "lib": key, "pin": "pin_%s" % new, "old": old,
                   "new": new, "name": ln}
            st["rows"].append(row)
            try:
                from cvparts.hypo import LEDGER
                LEDGER.record("libseq", "pin:%s.%s" % (cid, new), before=old,
                              after=new, why="库同名组按图上 y 降序回填库升序号"
                              "（库键 %s）" % key, by="stage2.pinlib", reason="libseq")
            except Exception:                               # noqa: BLE001
                pass
    if log is not None:
        log.append(st)
    return st


# ---------------------------------------------------------------------------
# 槽位版（融合层，`FUSION_PIN_LIBSEQ` 的正路）
# ---------------------------------------------------------------------------
# 为什么必须有槽位版：**文档版救不回被丢掉的脚**。
# `stage2/assemble.py:1133-1143` 对 `ctype == "box"` 的每根桩有一条闸
#   `if (not r.get("kept", True)) and BOX_PIN_DROP_UNREAD and (...): continue`
# 而 `kept` 只在「号是 1..999 的整数且不撞号」时为真（:1079-1084）。于是
# 「名字读对了、号没读出来」的桩（`VDDC`×2、`VDD`×1、`GND`×1、`VDA`）在
# **规则 A 救不动**（库的 `(型号,名)→号` 只做唯一映射，`VDD` 有 14 个号 ⇒ 放弃）
# 时就被丢在装配里 —— 交付文档里**连键都不存在**，文档版无从下手。
# 官方 10 张 case 1 的 `U1` 实测：26 根 GT 脚里 7 根没 emit
# （8 VDDA / 9 VREFA+ / 87,115 VDDC / 90,113 VDD / 114 GND；见 `_pincmp.py`）。
# 在**槽位**上把号补齐，`kept` 自然为真 ⇒ 那几根脚重新进文档。


def _slot_field(rec: Any, fld: str) -> dict:
    v = (rec or {}).get(fld)
    return v if isinstance(v, dict) else {}


def _slot_text(rec: Any, fld: str) -> str:
    """槽位的 `number`/`name` 文本（`assigned` 里是 `{"text":…,"box":…}`）。"""
    v = (rec or {}).get(fld)
    if isinstance(v, dict):
        return str(v.get("text") or "").strip()
    return str(v or "").strip()                     # 兼容裸字符串


def _slot_y(rec: Any) -> Optional[float]:
    """槽位在**图像帧**里的 y（取 `number`/`name` 的框；都没有返回 `None`）。

    ⚠️ 这里的帧与 `assigned` 的框一致 —— 阶段二的槽位框与阶段一的
    `texts[].bbox_img` **同帧**（实测：按原值比对命中、按 `H-y` 翻帧 0 命中），
    所以**不要**翻帧。图像帧里 y 越小越靠上 ⇒ 号小的在上面 = **y 升序**。
    """
    for fld in ("number", "name"):
        box = _slot_field(rec, fld).get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 2:
            try:
                return float(box[1])
            except Exception:                           # noqa: BLE001
                pass
    return None


def repair_slots(assigned: dict, ctx=None, mode: Optional[int] = None,
                 log=None) -> Dict[str, Any]:
    """在**槽位**上按「库同名脚集合 + 图上 y 顺序」补齐/纠正引脚号，并在必要时纠名。

    数据来源：`assigned["comps"][cid]["partname"]["text"]`（融合层已由 VLM 校准，
    IC 型号名 21/21 全对）；顺序依据：槽位 `number`/`name` 框的 y（**图像帧**，
    号小的在上面 = y 升序）。返回统计 dict（含 `rows`）。

    保守性（与 `repair_doc` 一致）：只在「同名组脚数 == 库里同名脚个数」时动手；
    `MODE=1` 时还要求组内**存在可疑读数**（空号、非数字号、或号不在库集合里）。
    """
    m = MODE if mode is None else int(mode)
    st: Dict[str, Any] = {"on": m > 0, "mode": m, "n_comp": 0, "n_key": 0,
                          "n_grp": 0, "n_num": 0, "n_name": 0,
                          "rows": [], "blocked": [], "skip": []}
    if m <= 0 or not isinstance(assigned, dict):
        return st
    comps = assigned.get("comps") or {}
    if not comps:
        return st
    ctx = ctx or _ctx()
    if ctx is None:
        st["err"] = "libpins 不可用"
        return st
    allm = maps()
    for cid, c in comps.items():
        st["n_comp"] += 1
        pins = (c or {}).get("pins")
        if not isinstance(pins, dict) or len(pins) < 2:
            continue
        pname = _slot_text(c, "partname")
        if not pname:
            continue
        try:
            key = ctx.key_for(pname)
        except Exception:                               # noqa: BLE001
            key = None
        if not key:
            continue
        mp = allm.get(str(key))
        if not mp:
            continue
        st["n_key"] += 1
        byname, canon = mp["byname"], mp["canon"]
        idxs = sorted(pins.keys(), key=lambda x: (len(str(x)), str(x)))
        # ---- ① 按脚名归一 → 库名 分组（只有**读到名字**的槽位进组）------------
        groups: Dict[str, List[str]] = {}
        for i in idxs:
            nm = _slot_text(pins[i], "name")
            if not nm:
                continue
            rn = _resolve_name(_libpins.norm(nm), byname)
            if rn:
                groups.setdefault(rn, []).append(i)
        plan: List[Any] = []                            # (idx, old, new, libname)
        used_new: Dict[str, str] = {}
        # 组**外**已占用的有效号：任何计划都不许撞上它们（撞号 = `build_doc`
        # 里 `pk in group` 会给键加后缀、或那个桩被判 `kept=False` 直接丢 ⇒ 少一根脚）。
        # 实战踩点：`case 10 U1 EG1192L` 的 `VB` 组库里是 `6`，但图上 `VR` 那根已经
        # 读到 `6` ⇒ 老的写法把 `VB` 也改成 `6`，两根脚撞号 ⇒ 引脚数 9 → 8。
        foreign: Dict[str, str] = {}
        for i in idxs:
            n = _slot_text(pins[i], "number")
            if n.isdigit():
                foreign.setdefault(n, i)
        for lname, members in groups.items():
            cand = list(byname.get(lname) or [])
            if not cand or len(cand) != len(members):
                if cand:
                    st["skip"].append({"cid": cid, "grp": lname,
                                       "why": "n=%d lib=%d" % (len(members), len(cand))})
                continue
            st["n_grp"] += 1
            ys = [_slot_y(pins[i]) for i in members]
            if all(y is not None for y in ys):
                order = [i for _, i in sorted(zip(ys, members))]
            else:                                       # 框缺失 ⇒ 退回槽位序号
                order = list(members)
            cur = [_slot_text(pins[i], "number") for i in order]
            cset = set(cand)
            if cur == list(cand):
                continue
            # ① 组内**可读且在库集合里**的号必须按 y 序**严格递增**（乱序说明这张
            #    符号是倒着画的、或库条目与图上的符号不是同一版 ⇒ 整组放弃）。
            cidx = {n: j for j, n in enumerate(cand)}
            known = [cidx[n] for n in cur if n in cset]
            if known != sorted(set(known)):
                st["skip"].append({"cid": cid, "grp": lname, "why": "乱序 %s" % known})
                continue
            # ② 组外占用：候选号被别的槽位读走了 ⇒ 放弃（避免撞号）
            bad = [n for n in cand if n in foreign and foreign[n] not in members]
            if bad:
                st["blocked"].append({"cid": cid, "grp": lname, "dup": bad})
                continue
            # ③ 待填槽位 = 读数**不在库集合里**的那些（空号 / 非数字 / 误读），
            #    按 y 序；它们依次领走剩下的候选号（候选本身是升序）。
            need = [i for i, n in zip(order, cur) if n not in cset]
            if not need:
                continue                                # 全部可读 ⇒ 自洽，不动
            left = [n for n in cand if n not in set(cur)]
            if len(left) != len(need):
                st["skip"].append({"cid": cid, "grp": lname,
                                   "why": "need=%d left=%d" % (len(need), len(left))})
                continue
            if m == 1 and not any((not o) or (o not in cset) for o in cur):
                continue
            pair = list(zip(need, left))
            clash = [n for _, n in pair if n in used_new]
            if clash:
                st["blocked"].append({"cid": cid, "grp": lname, "dup": clash})
                continue
            for i, n in pair:
                used_new[n] = i
                plan.append((i, _slot_text(pins[i], "number"), n,
                             canon.get(lname) or lname))
        # ---- ② 有号无名：库能唯一给出名字就补（为了过 `BOX_PIN_NEED_NAME`）----
        named: List[Any] = []
        for i in idxs:
            if _slot_text(pins[i], "name") or not _slot_text(pins[i], "number"):
                continue
            num = _slot_text(pins[i], "number")
            hits = [nm for nm, nums in byname.items() if num in nums]
            if len(hits) == 1:
                named.append((i, num, canon.get(hits[0]) or hits[0]))
        if not plan and not named:
            continue
        # ---- ③ 撞号终检：写完之后这颗元件的号集合里不许有重复（有就整颗放弃）----
        _new = dict((i, _slot_text(pins[i], "number")) for i in idxs)
        for i, _old, n, _ln in plan:
            _new[i] = n
        _vals = [v for v in _new.values() if v]
        if len(set(_vals)) != len(_vals):
            dup = sorted(v for v in set(_vals) if _vals.count(v) > 1)
            st["blocked"].append({"cid": cid, "grp": "*", "dup": dup})
            continue
        # ---- ③ 写入（只写 `number`/`name` 两个字段，绝不增删槽位）-------------
        # ★ 就地改文本、**保留原有的 `box`/`conf`/`d`**：下游（`priors` / `solve` /
        #   `claims`）会拿这些框做几何先验，把框抹成 `None` 等于顺手毁掉别的证据。
        rows: List[dict] = []
        for i, old, new, ln in plan:
            rec = dict(pins[i] or {})
            _n = rec.get("number")
            nb = dict(_n) if isinstance(_n, dict) else {}
            nb.update({"text": new, "_by": "libseq", "_libseq": True})
            if old:
                nb["_old_text"] = old
            rec["number"] = nb
            old_nm = _slot_text(rec, "name")
            if ln and _libpins.norm(old_nm) != _libpins.norm(ln):
                _m = rec.get("name")
                mb = dict(_m) if isinstance(_m, dict) else {}
                mb.update({"text": ln, "_by": "libseq", "_libseq": True})
                if old_nm:
                    mb["_old_text"] = old_nm
                rec["name"] = mb
                st["n_name"] += 1
            pins[i] = rec
            if old != new:
                st["n_num"] += 1
            rows.append({"cid": cid, "lib": key, "slot": i, "old": old,
                         "new": new, "name": ln})
        for i, num, nm in named:
            rec = dict(pins[i] or {})
            _m = rec.get("name")
            mb = dict(_m) if isinstance(_m, dict) else {}
            mb.update({"text": nm, "_by": "libseq", "_libseq": True})
            rec["name"] = mb
            pins[i] = rec
            st["n_name"] += 1
            rows.append({"cid": cid, "lib": key, "slot": i, "old": num,
                         "new": num, "name": nm, "why": "补名"})
        st["rows"].extend(rows)
        for r in rows:
            try:
                from cvparts.hypo import LEDGER
                LEDGER.record("libseq", "slot:%s.%s" % (cid, r["slot"]),
                              before=r["old"], after=r["new"],
                              why="库同名组按图上 y 升序回填库升序号（库键 %s）"
                              % key, by="stage2.pinlib", reason="libseq")
            except Exception:                           # noqa: BLE001
                pass
    if log is not None:
        log.append(st)
    return st


# ---------------------------------------------------------------------------
def _selftest() -> int:
    """合成库自检：模糊名 + 整组回填 + 自洽不动 + 脚数不等不动 + 关闸不动。"""
    global _MAPS, _ctx
    saved_maps, saved_ctx = _MAPS, _ctx
    _MAPS = {
        "FAKE1": {
            "byname": {"gnd": ["17", "48", "55"], "vdd": ["7", "16"],
                       "vdda": ["8"]},
            "canon": {"gnd": "GND", "vdd": "VDD", "vdda": "VDDA"},
            "bynum": {"17": "GND", "48": "GND", "55": "GND", "7": "VDD",
                      "16": "VDD", "8": "VDDA"},
        }
    }

    class _Ctx:
        def key_for(self, s):
            return "FAKE1" if str(s).strip().upper().startswith("FAKE") else None

    _ctx = lambda: _Ctx()                                   # type: ignore
    bad: List[str] = []
    n = [0]

    def chk(cond, msg):
        n[0] += 1
        if not cond:
            bad.append(msg)

    def _doc(pins, nets=None, name="FAKE1"):
        return {"components": {"U1": {"Name": name, "type": "box"}},
                "pins": {"U1": pins},
                "nets": nets or {}}

    # ① 前导 1 被吃掉（14 实为 114 的同名组 + 一个 46 实为 48）⇒ 整组按 y 降序回填
    #    y-up：号小的在上面 = y 大。
    d = _doc({
        "pin_17": {"pinname": "GND", "point": {"x": 0, "y": 90}},
        "pin_46": {"pinname": "GND", "point": {"x": 0, "y": 60}},
        "pin_55": {"pinname": "GND", "point": {"x": 0, "y": 30}},
    }, nets={"net_1": {"hyperGraph": "(U1.17,C1.1)"},
             "net_2": {"hyperGraph": "(U1.46,U1.55)"}})
    s = repair_doc(d, mode=1)
    chk(sorted(d["pins"]["U1"].keys()) == ["pin_17", "pin_48", "pin_55"],
        "组内可疑未回填：%s" % sorted(d["pins"]["U1"].keys()))
    chk(d["nets"]["net_2"]["hyperGraph"] == "(U1.48,U1.55)",
        "net 引用未改写：%s" % d["nets"]["net_2"]["hyperGraph"])
    chk(d["pins"]["U1"]["pin_48"]["point"] == {"x": 0, "y": 60}, "不许碰 point")
    chk(s["n_chg"] == 1 and len(s["rows"]) == 1, "改动计数不对 %s" % s["n_chg"])
    # ② 组内读数都在库集合里 ⇒ 不动（EG1192L 的 VIN 倒画反例）
    d2 = _doc({"pin_48": {"pinname": "GND", "point": {"x": 0, "y": 90}},
               "pin_17": {"pinname": "GND", "point": {"x": 0, "y": 60}},
               "pin_55": {"pinname": "GND", "point": {"x": 0, "y": 30}}})
    s2 = repair_doc(d2, mode=1)
    chk(sorted(d2["pins"]["U1"].keys()) == ["pin_17", "pin_48", "pin_55"]
        and s2["n_chg"] == 0, "自洽组被误改 %s" % sorted(d2["pins"]["U1"].keys()))
    # ③ 模糊名 VDA → 库 VDDA，且脚数相等 ⇒ 补号
    d3 = _doc({"pin_8": {"pinname": "VDDA", "point": {"x": 0, "y": 10}}})
    chk(_resolve_name("vda", _MAPS["FAKE1"]["byname"]) == "vdda", "模糊名未解析")
    repair_doc(d3, mode=2)
    chk("pin_8" in d3["pins"]["U1"], "VDDA 组被误动")
    # ④ 脚数与库不等 ⇒ 不动
    d4 = _doc({"pin_14": {"pinname": "GND", "point": {"x": 0, "y": 10}}})
    repair_doc(d4, mode=1)
    chk("pin_14" in d4["pins"]["U1"], "脚数不等时不该动手")
    # ⑤ 关闸不动
    d5 = _doc({"pin_14": {"pinname": "GND", "point": {"x": 0, "y": 10}},
               "pin_17": {"pinname": "GND", "point": {"x": 0, "y": 5}},
               "pin_55": {"pinname": "GND", "point": {"x": 0, "y": 1}}})
    repair_doc(d5, mode=0)
    chk("pin_14" in d5["pins"]["U1"], "关闸时不该动手")
    # ⑥ 型号名命中不了库键 ⇒ 不动
    d6 = _doc({"pin_14": {"pinname": "GND", "point": {"x": 0, "y": 10}},
               "pin_17": {"pinname": "GND", "point": {"x": 0, "y": 5}},
               "pin_55": {"pinname": "GND", "point": {"x": 0, "y": 1}}},
              name="UNKNOWN")
    repair_doc(d6, mode=1)
    chk("pin_14" in d6["pins"]["U1"], "库键未知时不该动手")

    # ---- 槽位版 -----------------------------------------------------------
    def _sl(num=None, nm=None, y=0.0):
        def _f(t):
            return {"text": t, "conf": 0.0, "box": [0.0, float(y), 9.0, float(y) + 8]}
        return {"number": _f(num) if num is not None else None,
                "name": _f(nm) if nm is not None else None}

    def _asg(slots, name="FAKE1"):
        return {"comps": {"b0": {"designator": {"text": "U1"},
                                 "partname": {"text": name},
                                 "pins": {str(i): v for i, v in enumerate(slots)}}}}

    # ⑦ 组内可疑（空号 + 不在库集合里的 46）⇒ 只动「读数不在库里」的两槽：
    #    按 y 序（槽 1 在上、槽 0 在下）依次领走剩下的候选 17/48，可读的 55 不动
    a = _asg([_sl("46", "GND", 60.0), _sl(None, "GND", 30.0), _sl("55", "GND", 10.0)])
    s7 = repair_slots(a, mode=1)
    got = [_slot_text(a["comps"]["b0"]["pins"][str(i)], "number") for i in range(3)]
    chk(got == ["48", "17", "55"], "槽位组内回填不对：%s" % got)
    chk(s7["n_num"] == 2 and s7["n_grp"] == 1, "槽位改动计数不对 %s" % s7["n_num"])
    # ⑧ 自洽（都在库集合里，只是顺序不同）⇒ mode 1 不动
    a2 = _asg([_sl("48", "GND", 60.0), _sl("17", "GND", 30.0), _sl("55", "GND", 10.0)])
    s8 = repair_slots(a2, mode=1)
    got2 = [_slot_text(a2["comps"]["b0"]["pins"][str(i)], "number") for i in range(3)]
    chk(got2 == ["48", "17", "55"] and s8["n_num"] == 0, "自洽槽位组被误改 %s" % got2)
    # ⑨ 关闸不动
    a3 = _asg([_sl("46", "GND", 60.0), _sl(None, "GND", 30.0), _sl("55", "GND", 10.0)])
    repair_slots(a3, mode=0)
    chk(_slot_text(a3["comps"]["b0"]["pins"]["0"], "number") == "46", "关闸时不该动手")
    # ⑩ 有号无名 ⇒ 库唯一给名（槽位数 ≥2 才进通道）
    a4 = _asg([_sl("17", None, 10.0), _sl("48", "GND", 30.0)])
    repair_slots(a4, mode=1)
    chk(_slot_text(a4["comps"]["b0"]["pins"]["0"], "name") == "GND", "有号无名未补名")
    # ⑪ 模糊名 VDA → VDDA：补号 8 并把名纠成库的规范名
    a5 = _asg([_sl(None, "VDA", 10.0), _sl(None, "GND", 30.0)])
    repair_slots(a5, mode=1)
    chk(_slot_text(a5["comps"]["b0"]["pins"]["0"], "number") == "8"
        and _slot_text(a5["comps"]["b0"]["pins"]["0"], "name") == "VDDA",
        "模糊名组未纠名补号：%s" % a5["comps"]["b0"]["pins"]["0"])
    _MAPS, _ctx = saved_maps, saved_ctx
    print("自检 %d 项：%s" % (n[0], "全部通过" if not bad else "失败"))
    for b in bad:
        print("  FAIL", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
