# -*- coding: utf-8 -*-
"""**号/名联修的库权威校正**（`FUSION_PIN_LIBFIX`，默认 **0=关**，A/B 见 `docs/RESULTS.md` §5.49）。

## 病（`work\\_namebad2.py` 在官方 10 张上逐脚量出来的）
`stage2/librefine.py` 已经做「(型号, 号) → 库名」，但它有两条前提挡住了本类失分：

1. **库里那个号必须有唯一名**（`name_for_num`）。官方库里一号多名极常见：
   `AMS1117-3.3` 的 1 号有 `ADJ / ADJ(GND) / ADJ/GND / GND / GND/ADJ / VSS/ADJ`
   六种写法 ⇒ `name_for_num` 返回 `None`；而 CRNN 恰好把 GT 的 `ADJ(GND)` 读成截断的
   `ADJ(GN`（VLM 档 case 5 `U3.pin_1`，位置只差 2.6 px）。
2. **脚的号本身读错**时整条规则都不适用（号不在库号集合里 ⇒ 也是 `None`）：
   离线档 `6 U1`（型号 `79M12`）我们的脚键是 `pin_33` 名 `L`，GT 是 `pin_3` 名 `OUT`；
   `10 U1`（`EG1192L`）我们 `pin_24` 名 `VA`，GT 是 `pin_4` 名 `VIA`（位置只差 2.5 px）。

离线档 8 根「号对·位对·名错」的脚里，库键命中 2 根、而库按号给的名**还是同一份错读**
（`2 U3` 的型号被读成 `STC11` —— 一个**假库键**，库于是把 `P67/4` 又灌了一遍）⇒
单靠 `librefine` 已经榨干，剩下只能靠下面两条新规则。

## 药（两条，都只在**型号名精确命中库键**时动）
* **A 一号多名 ⇒ 模糊挑一个**：库对该号给出多个名字时取相似度最高者，要求
  `≥ FUSION_PIN_LIBFIX_RATIO`（默认 0.75）且**严格优于次优** `≥ MARGIN`（默认 0.05）；
  「只差大小写/标点」的一律保留我方读法（判题逐字符比较，与 `librefine` 同规矩）。
* **B 号读错 ⇒ 先修号**：若我们的号不在该型号的库号集合里，而**去掉一位多余数字**
  （重复数字 `33`，或前导多读的 `24` / `115`：`num.endswith(L)` 且长度只差 1）能得到库号
  `L`，且 `pin_L` 未被本元件占用、库里 `L` 号有**唯一**名、当前名是「库里没有的碎片」
  或与库名模糊相近 —— 则把脚键改成 `pin_L`、名写成库名，并同步改写网络里的
  `hyperGraph` 引用（`U1.24 → U1.4`）。

## 规矩
只改 `pinname` 与**脚键的号**；绝不增删脚、绝不碰 `point` / `bbox` / `value` / `Name`；
每次改动记 `cvparts.hypo.LEDGER.record("pin_libfix", …)`（可查、可回滚）。
**库名必须含字母**（`_named`）—— 官方库里有 `STC11` 的 4 号名 `'4'`、`SA812` 的 0 号名 `'0'`
这类垃圾条目，采信它只会把对的读法改错（派生 18 张 `0117 U1 pin_10 'PA1' → pin_0 '0'`）。
自检：`python -X utf8 -m fusion.pinfix`（在 `eda-pcb` 下）。

## 实测（2026-10-21 转正默认 **1**；`docs/RESULTS.md` §5.49）
四把尺子全为正、无一处回归，且**不花任何 LLM/VLM 调用**（纯查表 + 字符串）：

| 尺子 | 臂 | 前 → 后 | Δ |
| --- | --- | --- | --- |
| 判题机复刻（官方 10 张） | 离线 | 62.0224 → 62.1625 | **+0.1401** |
| 判题机复刻（官方 10 张） | 默认 VLM | 65.5141 → 65.6629 | **+0.1488** |
| protocol2（官方 10 张） | 离线 | 73.37 → 73.83 | **+0.46** |
| protocol2（官方 10 张） | 默认 VLM | 79.66 → 80.02 | **+0.36** |
| 判题机复刻（派生 18 张） | 离线 | 13.9056 → 13.9056 | **±0.0000** |

命中（离线 4 处 / VLM 2 处 / 派生 0 处）：`10 U1 pin_24 'VA' → pin_4 'VIA'`（`EG1192L`）、
`6 U1 pin_33 'L' → pin_3 'OUT'`（`79M12` 离线 / `78M12` VLM）、
`5 U3 pin_1 'ADJ(GN' → 'ADJ(GND)'`（`AMS1117-3.3`，规则 A）、
case 8 的 `pin_22/pin_26 → pin_2/pin_6`（`NC`，位置仍超差故不涨分）。
"""
from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from stage2 import libpins as LP
except ImportError:                                          # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from stage2 import libpins as LP                     # type: ignore
    except ImportError:
        LP = None                                            # type: ignore


def _on() -> bool:
    # 默认 **1 = 开**（2026-10-21 转正；四把尺子全正见模块 docstring 的「实测」表）
    return (os.environ.get("FUSION_PIN_LIBFIX", "1") or "").strip().lower() \
        not in ("", "0", "false", "off", "no", "none")


def _num_of(pin_key: str) -> str:
    s = str(pin_key or "").strip()
    if s.startswith("pin_"):
        s = s[4:]
    return s if s else ""


_TABLE: Optional[Dict[str, Any]] = None


def _lib_file() -> Path:
    if LP is not None:
        try:
            return Path(LP.lib_path())
        except Exception:                                    # noqa: BLE001
            pass
    return Path(__file__).resolve().parents[1] / "data" / "BoxICLibrary_200cases.json"


def _table(path=None) -> Dict[str, Any]:
    """`norm(型号) → {"key": 原写法, "nums": {号: [名字…]}}`（一号多名全部保留）。"""
    global _TABLE
    if isinstance(path, dict):                               # 允许直接传预建表（自检/复用）
        return path
    if path is None and _TABLE is not None:
        return _TABLE
    f = Path(path) if path else _lib_file()
    out: Dict[str, Any] = {}
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return out
    for k, v in (raw or {}).items():
        nums: Dict[str, List[str]] = {}
        for pin in ((v or {}).get("pins") or []):
            num = str((pin or {}).get("number") or "").strip()
            nm = str((pin or {}).get("name") or "").strip()
            if not num or not nm:
                continue
            have = {LP.norm(x) for x in nums.get(num, [])} if LP else set(nums.get(num, []))
            if (LP.norm(nm) if LP else nm) not in have:
                nums.setdefault(num, []).append(nm)
        if nums:
            out[LP.norm(k) if LP else k] = {"key": k, "nums": nums}
    if path is None:
        _TABLE = out
    return out


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").upper(), (b or "").upper()).ratio()


def _rank(cur: str, cands: List[str]) -> List[Tuple[float, str]]:
    out = [(_ratio(cur, c), c) for c in cands]
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def _named(s: str) -> bool:
    """库名得像个人写的：必须含字母。

    官方库里有一批**垃圾条目**（`STC11` 的 4 号名是 `'4'`、`SA812` 的 0 号名是 `'0'`、
    7 号名是 `'7'`），照它改只会把对的读法改成错的 —— 派生 18 张的
    `0117 U1 pin_10 'PA1' → pin_0 '0'` 就是这么掉下去的（−0.0021）。
    """
    return any(ch.isalpha() for ch in (s or ""))


def _renumber(doc: dict, cid: str, old: str, new: str) -> int:
    """把 `cid.old` 的网络引用改成 `cid.new`（`hyperGraph` 文本 + `pins` 列表）。"""
    n = 0
    old_tok, new_tok = "%s.%s" % (cid, old), "%s.%s" % (cid, new)
    for nv in (doc.get("nets") or {}).values():
        if not isinstance(nv, dict):
            continue
        hg = nv.get("hyperGraph")
        if isinstance(hg, str) and hg.strip():
            toks = [t.strip() for t in hg.strip().strip("()").split(",") if t.strip()]
            if old_tok in toks:
                toks = [new_tok if t == old_tok else t for t in toks]
                nv["hyperGraph"] = "(" + ",".join(toks) + ")"
                n += 1
        pl = nv.get("pins")
        if isinstance(pl, list):
            for i, t in enumerate(pl):
                if str(t).strip() == old_tok:
                    pl[i] = new_tok
                    n += 1
    pi = doc.get("__pin_index")
    if isinstance(pi, dict):                                 # 维护内部索引（判题机不看）
        for slot in pi.values():
            if isinstance(slot, dict):
                for k2, v2 in list(slot.items()):
                    if v2 == ("pin_%s" % old):
                        slot[k2] = "pin_%s" % new
    return n


def fix(doc: dict, log=None, table=None) -> Dict[str, Any]:
    """就地校正 `doc` 里「型号名精确命中库键」的元件的引脚名/号，返回统计。"""
    # ★ 两个阈值**必须**用 `os.environ.get` 直写在调用点上：`tools/dev/gen_knobs.py`
    #   的正则只认 `os.environ.get(...)` / `envflag.*`，包一层函数（同 `namefix.py` 的记）
    #   会让这两个开关从 `docs/KNOBS.md` 里整条消失。
    st: Dict[str, Any] = {"on": _on(), "ratio": 0.75, "margin": 0.05,
                          "n_seen": 0, "n_key": 0, "n_name": 0, "n_rekey": 0,
                          "n_refnum": 0, "rows": []}
    try:
        st["ratio"] = float(os.environ.get("FUSION_PIN_LIBFIX_RATIO", "0.75") or "0.75")
    except Exception:                                        # noqa: BLE001
        pass
    try:
        st["margin"] = float(os.environ.get("FUSION_PIN_LIBFIX_MARGIN", "0.05") or "0.05")
    except Exception:                                        # noqa: BLE001
        pass
    if not st["on"] or not isinstance(doc, dict) or LP is None:
        return st
    tab = _table(table)
    if not tab:
        st["err"] = "库加载失败"
        return st
    pins = doc.get("pins") or {}
    for cid, comp in (doc.get("components") or {}).items():
        if not isinstance(comp, dict):
            continue
        rows = pins.get(cid) or {}
        if not isinstance(rows, dict) or not rows:
            continue
        st["n_seen"] += len(rows)
        ent = tab.get(LP.norm(comp.get("Name") or ""))
        if not ent:                                          # 型号名必须**精确**命中库键
            continue
        st["n_key"] += len(rows)
        nums: Dict[str, List[str]] = ent["nums"]
        allnames = {LP.norm(x) for names in nums.values() for x in names}
        used = set(rows)
        for pk in list(rows):
            rec = rows.get(pk)
            if not isinstance(rec, dict):
                continue
            num = _num_of(pk)
            if not num:
                continue
            cur = str(rec.get("pinname") or "")
            newnum = newname = None
            why = ""
            if num in nums:                                  # A：一号多名 ⇒ 模糊挑一个
                cands = nums[num]
                if LP.norm(cur) not in {LP.norm(x) for x in cands}:
                    rk = _rank(cur, cands)
                    best = rk[0][0]
                    second = rk[1][0] if len(rk) > 1 else 0.0
                    if best >= st["ratio"] and (best - second) >= st["margin"] \
                            and _named(rk[0][1]):
                        newname, why = rk[0][1], "一号多名模糊选择 %.3f/%.3f" % (best, second)
            else:                                            # B：号读错 ⇒ 先修号
                cands_b: List[Tuple[str, str]] = []
                for L in nums:
                    if not L.isdigit() or not num.isdigit():
                        continue
                    if len(num) - len(L) != 1 or not num.endswith(L):
                        continue
                    if ("pin_%s" % L) in used:
                        continue
                    names = nums[L]
                    if len({LP.norm(x) for x in names}) != 1:
                        continue
                    nm = names[0]
                    if not _named(nm):                       # 垃圾库名（`'0'`/`'4'`/`'7'`）不采信
                        continue
                    junk = LP.norm(cur) not in allnames
                    near = bool(cur) and _ratio(cur, nm) >= st["ratio"]
                    if junk or near:
                        cands_b.append((L, nm))
                if len(cands_b) == 1:
                    newnum, newname = cands_b[0]
                    why = "号读错 %s→%s（库名唯一）" % (num, newnum)
            if not newname:
                continue
            if newnum and newnum != num:
                newkey = "pin_%s" % newnum
                rows[newkey] = rec
                del rows[pk]
                used.discard(pk)
                used.add(newkey)
                st["n_refnum"] += _renumber(doc, cid, num, newnum)
                st["n_rekey"] += 1
            rec["pinname"] = newname
            st["n_name"] += 1
            st["rows"].append({"cid": cid, "was_key": pk,
                               "now_key": "pin_%s" % newnum if newnum else pk,
                               "was": cur, "now": newname, "key": ent["key"], "why": why})
    if st["n_name"]:
        try:
            from cvparts.hypo import LEDGER
            LEDGER.record("pin_libfix", "pins", before=st["rows"][:40],
                          after={"n_name": st["n_name"], "n_rekey": st["n_rekey"]},
                          reason="(型号,号) 库权威：一号多名模糊选择 + 号读错改键",
                          by="fusion:pinfix")
        except Exception:                                    # noqa: BLE001
            pass
    if log and st["n_name"]:
        log("  [pinfix] 库权威校正 %d 处（改名 %d / 改号 %d，键命中 %d/%d 个脚）"
            % (st["n_name"], st["n_name"] - st["n_rekey"], st["n_rekey"],
               st["n_key"], st["n_seen"]))
    return st


# ── 自检 ──────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import tempfile
    bad: List[str] = []
    fake = {"FIX1": {"pins": [{"name": "ADJ", "number": "1"},
                              {"name": "ADJ(GND)", "number": "1"},
                              {"name": "VOUT", "number": "2"},
                              {"name": "OUT", "number": "3"}],
                     "same_hash_components": []},
            "FIX2": {"pins": [{"name": "A", "number": "1"},
                              {"name": "B", "number": "1"}],
                     "same_hash_components": []}}
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "lib.json"
        f.write_text(json.dumps(fake), encoding="utf-8")
        tab = _table(f)
        doc = {"components": {"U1": {"Name": "FIX1", "type": "box"},
                              "U2": {"Name": "没这个型号", "type": "box"}},
               "pins": {"U1": {"pin_1": {"pinname": "ADJ(GN", "point": {"x": 1, "y": 1}},
                                       "pin_2": {"pinname": "VOUT", "point": {"x": 2, "y": 2}},
                                       "pin_33": {"pinname": "L", "point": {"x": 3, "y": 3}}},
                        "U2": {"pin_1": {"pinname": "NOPE"}}},
               "nets": {"net_1": {"hyperGraph": "(U1.33,C1.1)"}},
               "__pin_index": {"c0000": {"0": "pin_33"}}}
        os.environ["FUSION_PIN_LIBFIX"] = "1"
        s = fix(doc, table=tab)
        p1 = doc["pins"]["U1"]
        if p1["pin_1"]["pinname"] != "ADJ(GND)":
            bad.append("一号多名该模糊挑出 ADJ(GND)：%s" % p1["pin_1"]["pinname"])
        if "pin_3" not in p1 or "pin_33" in p1:
            bad.append("号读错该改键 33→3：%s" % sorted(p1))
        if p1.get("pin_3", {}).get("pinname") != "OUT":
            bad.append("改号后名该写库名 OUT：%s" % (p1.get("pin_3") or {}).get("pinname"))
        if p1["pin_2"]["pinname"] != "VOUT":
            bad.append("已是库名不该动")
        if doc["nets"]["net_1"]["hyperGraph"] != "(U1.3,C1.1)":
            bad.append("网络引用该跟着改：%s" % doc["nets"]["net_1"]["hyperGraph"])
        if doc["__pin_index"]["c0000"]["0"] != "pin_3":
            bad.append("__pin_index 该跟着改")
        if p1["pin_1"]["point"] != {"x": 1, "y": 1} or p1["pin_3"]["point"] != {"x": 3, "y": 3}:
            bad.append("不许碰 point")
        if doc["pins"]["U2"]["pin_1"]["pinname"] != "NOPE":
            bad.append("型号落不到库键时不该动")
        if (s["n_name"], s["n_rekey"]) != (2, 1):
            bad.append("计数不对 %s" % s)
        # 关：一个字都不许动
        doc2 = {"components": {"U1": {"Name": "FIX1"}},
                "pins": {"U1": {"pin_33": {"pinname": "L"}}}}
        os.environ["FUSION_PIN_LIBFIX"] = "0"
        s0 = fix(doc2, table=tab)
        if s0["n_name"] or "pin_33" not in doc2["pins"]["U1"]:
            bad.append("总闸关时不该动手")
        # FIX2：1 号两名、都不可能模糊中（A/B）⇒ 不动，且 pin_1 已存在 ⇒ 也不许改号
        doc3 = {"components": {"U1": {"Name": "FIX2"}}, "pins": {"U1": {"pin_1": {"pinname": "??"}}}}
        os.environ["FUSION_PIN_LIBFIX"] = "1"
        s3 = fix(doc3, table=tab)
        if s3["n_name"] or doc3["pins"]["U1"]["pin_1"]["pinname"] != "??":
            bad.append("两名都没把握时不该动 %s" % s3)
        os.environ.pop("FUSION_PIN_LIBFIX", None)
    print("pinfix 自检：%s" % ("全部通过" if not bad else "失败 " + " / ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
