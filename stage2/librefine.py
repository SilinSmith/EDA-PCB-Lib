# -*- coding: utf-8 -*-
"""**装配之后的库名回填**：用官方库把已经 emit 的引脚名按 `(型号, 号) → 库名` 校准一次。

## 为什么需要它（实测，不是猜的）

* `stage2/libpins.py` 的规则 B（`names_for`）只在 `assemble.build_doc` 里跑；而
  **VLM 补强发生在融合层之前**（`submission/run.py` 的顺序：阶段一/二 → 阶段三 →
  VLM 补强 → 融合层 → 导出），VLM 会在库里已经改写过的引脚上**再写一遍**自己的读数
  （`run.log`：「改写引脚 93 个」）⇒ 库里已经校准好的名字被 VLM 的读法覆盖。
* no-vlm 配置下 `key_for(我方型号名)` **全部返回 None**（型号名自己就读错了），库先验
  整条路是死的；但 **VLM 配置下 IC `Name` 21/21 全对**
  （`work\\tools\\diag\\_libopp.py --pred work\\_vlmname`：元件名 `ok 21 100%`），
  `key_for` 一定命中真键 ⇒ 库先验在**交付口径**下是活的，只是没人用它去复核 VLM 的读数。
* 官方 10 张、VLM 臂的**离线后处理**实测（`work\\tools\\dev\\_libpost.py`，直接改同一份
  `_Pred.json` 再打分；基线 68.3369）：

  | 档 | 改写处数 | 库答案==GT | 总分 | pin F1 / tp / fp / fn |
  |---|---|---|---|---|
  | 关 | 0 | — | 68.3369 | 0.6724 / 78 / 40 / 58 |
  | `missing`（当前名不在库里才改） | 3 | 3 | 68.7156 | 0.6876 / 80 / 38 / 56 |
  | `consist`（+ 名指向的号与本脚号矛盾） | 6 | 6 | 69.0554 | 0.7012 / 82 / 36 / 54 |
  | **`any`（只挡「仅大小写/标点不同」）** | **9** | **8**（另 1 处是我方多出的脚，GT 无此脚） | **69.4341** | **0.7163 / 84 / 34 / 52** |

  为什么 `any` 敢改「当前名其实是这个型号的合法名」的脚（如 c8 pin7 `NC → VDDIO`、
  c10 pin2 `VIN → GND`）：**号是本脚的硬证据**，库说这个号叫 `VDDIO`，而 `NC` 是同一
  颗芯片在**别的号**上的名字 ⇒ 名/号不自洽。唯一必须挡住的是「只差大小写」那一类
  （`VIN → Vin`、`VOUT → Vout`）：官方判据 `metrics.py:466` 是**逐字**比较，库里的写法
  不一定等于 GT 的写法。

## 规矩（沿用工程哲学）

只加不改：本模块**只改 `pinname` 字段**，绝不删除/新增引脚、绝不碰 `point`、`net`、
`components`；每一次改写都往 `cvparts.hypo.LEDGER` 记一条 `record(kind="librefine")`，
所以「削弱/改写」在旁路账本里可查。总闸默认 **1（开）**（2026-10 转正，读数见 `_on()`）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from . import libpins as LP
except Exception:                                            # pragma: no cover
    LP = None                                                # type: ignore


# ★ 这两个开关**必须**在调用点直接写字面量环境名：`tools/dev/gen_knobs.py` 的 `PAT` 只认
#   `os.environ.get("X", …)` / `env_on("X", …)` 这类**字面量第一参数**；套一层自建包装
#   （例如 `_env("X", …)`）会让 `docs/KNOBS.md` 整条漏掉本开关（2026-10 踩过：
#   `STAGE2_LIB_REFINE` 当时没进表）。自检直接改 `os.environ` 来模拟两档。
def _on() -> bool:
    # 2026-10 转正：官方 10 张（VLM 交付档）66.4498 → 66.7807（+0.3310/张，4 涨 6 平 0 跌；
    # `pin` F1 0.7272 → 0.7404、tp 310 → 316）；派生 val 19 张 19.2974 → 19.3987
    # （+0.1013/张，5 涨 13 平 1 跌，唯一那 1 跌是 VLM 写值的方差：两臂 `写值 117/210` vs
    # `115/210`）；`--no-vlm` 退路上逐位相同（60.9970 → 60.9970，型号名读错 ⇒ 全部 miss）。
    return (os.environ.get("STAGE2_LIB_REFINE", "1") or "").strip().lower() \
        not in ("", "0", "false", "off", "no", "none")


def _when() -> str:
    """档位：`any`（默认，只挡仅大小写不同）/ `consist` / `missing`。"""
    w = (os.environ.get("STAGE2_LIB_REFINE_WHEN", "any") or "").strip().lower()
    return w if w in ("any", "consist", "missing") else "any"


def _num_of(pin_key: str) -> str:
    """`pin_12` → `12`；裸 `12` → `12`；其它 → `""`。"""
    s = str(pin_key or "").strip()
    if s.startswith("pin_"):
        s = s[4:]
    return s if s else ""


def refine(doc: dict, ctx=None, log=None) -> Dict[str, Any]:
    """就地校准 `doc["pins"][cid][pin_<号>]["pinname"]`，返回统计。

    只在下面四条**同时**成立时改写（缺一不动）：
      1. 总闸 `STAGE2_LIB_REFINE=1`；
      2. 该元件型号名能命中库键（精确/归一化/模糊，`LibCtx.key_for`）；
      3. 库对该 **号** 有唯一名（`name_for_num`，一号多名 ⇒ 库里就没这个映射）；
      4. 档位判据（见 `_when`），且库名与当前名**不是只差大小写**。
    """
    st: Dict[str, Any] = {"on": _on(), "when": _when(), "n_seen": 0, "n_key": 0,
                          "n_chg": 0, "blocked": 0, "case_only": 0, "rows": []}
    if not st["on"] or not isinstance(doc, dict):
        return st
    if LP is None:                                           # pragma: no cover
        st["err"] = "libpins 不可用"
        return st
    ctx = ctx or LP.new_ctx()
    if not ctx.load():
        st["err"] = "库加载失败"
        return st
    when = st["when"]
    pins = doc.get("pins") or {}
    for cid, comp in (doc.get("components") or {}).items():
        if not isinstance(comp, dict):
            continue
        rows = pins.get(cid) or {}
        if not rows:
            continue
        st["n_seen"] += len(rows)
        key = ctx.key_for(comp.get("Name") or "")
        if not key:
            continue
        st["n_key"] += len(rows)
        nm_set = {LP.norm(x) for x in ctx.names_of(key)}
        for pk, rec in rows.items():
            if not isinstance(rec, dict):
                continue
            num = _num_of(pk)
            if not num:
                continue
            cur = str(rec.get("pinname") or "")
            ref = ctx.name_for_num(key, num)
            if not ref or ref == cur:
                continue
            if cur and LP.norm(cur) == LP.norm(ref):
                st["case_only"] += 1                          # 只差大小写/标点 ⇒ 保留我方读法
                continue
            if when == "missing":
                if cur and LP.norm(cur) in nm_set:
                    st["blocked"] += 1
                    continue
            elif when == "consist":
                if cur and LP.norm(cur) in nm_set:
                    other = ctx.num_for_name(key, cur)
                    if other in (None, "", num):
                        st["blocked"] += 1
                        continue
            rec["pinname"] = ref
            st["n_chg"] += 1
            st["rows"].append({"cid": cid, "pin": pk, "num": num,
                               "was": cur, "now": ref, "key": key})
    if st["n_chg"]:
        try:
            from cvparts.hypo import LEDGER
            LEDGER.record("librefine", "pins", before=st["rows"][:40],
                          after={"n_chg": st["n_chg"], "when": when},
                          reason="(型号,号)→库名 校准 VLM/OCR 写下的引脚名",
                          by="stage2:librefine")
        except Exception:                                    # noqa: BLE001
            pass
    if log and st["n_chg"]:
        log("  [librefine] 校准引脚名 %d 处（档 %s，键命中 %d/%d 个脚）"
            % (st["n_chg"], when, st["n_key"], st["n_seen"]))
    return st


# ── 自检 ──────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import json
    import tempfile  # noqa: F401
    bad = []
    fake = {
        "AA1": {"pins": [{"name": "VDD", "number": "1"}, {"name": "PA00", "number": "2"},
                         {"name": "X", "number": "3"}, {"name": "Y", "number": "3"},
                         {"name": "VIN", "number": "4"}]},
    }
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        f = Path(td) / "lib.json"
        f.write_text(json.dumps(fake), encoding="utf-8")
        if LP is None:
            print("librefine 自检：libpins 不可用")
            return 1
        LP.LIB_PINS = True
        ctx = LP.new_ctx(f)
        doc = {"components": {"U1": {"Name": "AA1"}, "U2": {"Name": "不存在的型号"}},
               "pins": {"U1": {"pin_1": {"pinname": "vdd", "point": {"x": 0, "y": 0}},
                               "pin_2": {"pinname": "NOPE", "point": {"x": 1, "y": 1}},
                               "pin_3": {"pinname": "X", "point": {"x": 2, "y": 2}},
                               "pin_4": {"pinname": "VIN", "point": {"x": 3, "y": 3}}},
                        "U2": {"pin_1": {"pinname": "NOPE"}}}}
        # 关：一个字都不许动
        os.environ["STAGE2_LIB_REFINE"] = "0"
        s0 = refine(doc, ctx)
        if s0["n_chg"] != 0 or doc["pins"]["U1"]["pin_2"]["pinname"] != "NOPE":
            bad.append("总闸关时不该动手")
        os.environ["STAGE2_LIB_REFINE"] = "1"
        os.environ["STAGE2_LIB_REFINE_WHEN"] = "any"
        s = refine(doc, ctx)
        # pin_1 vdd→VDD 是「只差大小写」⇒ 保留；pin_2 NOPE→PA00 改；pin_3 号3多名⇒库里无映射⇒不动；pin_4 VIN 已是库名⇒不动
        if doc["pins"]["U1"]["pin_1"]["pinname"] != "vdd":
            bad.append("只差大小写不该改")
        if doc["pins"]["U1"]["pin_2"]["pinname"] != "PA00":
            bad.append("应改成库名 PA00")
        if doc["pins"]["U1"]["pin_3"]["pinname"] != "X":
            bad.append("一号多名时库里无唯一映射，不该动")
        if doc["pins"]["U2"]["pin_1"]["pinname"] != "NOPE":
            bad.append("型号落不到库键时不该动")
        if s["n_chg"] != 1 or s["case_only"] != 1:
            bad.append("计数不对 %s" % s)
        good = doc["pins"]["U1"]["pin_2"]["point"]
        if good != {"x": 1, "y": 1}:
            bad.append("不许碰 point")
        # missing 档：pin_4 的名字 VIN 是库名 ⇒ 不改；把 pin_2 的当前名改成库名后再跑应为 0 改动
        os.environ["STAGE2_LIB_REFINE"] = "1"
        os.environ["STAGE2_LIB_REFINE_WHEN"] = "missing"
        doc2 = {"components": {"U1": {"Name": "AA1"}},
                "pins": {"U1": {"pin_2": {"pinname": "VDD"}}}}
        s2 = refine(doc2, ctx)
        if s2["n_chg"] != 0 or s2["blocked"] != 1:
            bad.append("missing 档该挡住库内合法名 %s" % s2)
        os.environ.pop("STAGE2_LIB_REFINE", None)
        os.environ.pop("STAGE2_LIB_REFINE_WHEN", None)
    print("librefine 自检：%s" % ("全部通过" if not bad else "失败 " + " / ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
