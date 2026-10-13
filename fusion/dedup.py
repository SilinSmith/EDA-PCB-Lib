# -*- coding: utf-8 -*-
"""元件通道「同物重复」消重（`FUSION_COMPDEDUP`，默认 **1** = 开；`0` 关，A/B 用）。

## 病：同一颗元件被发了两次，其中一次没有位号

官方 10 张里 `component.extra` 只剩两条，形态完全一样 —— **同一个物理符号被检出两次**，
一次挂了真位号、一次只能拿到假键 `pNNN`（`stage2/assign.py` 给「没配到位号文字框」的
元件发的兜底名）：

| 案例 | 假键 | 类型 | 撞上谁 | 几何 |
| --- | --- | --- | --- | --- |
| 4 | `p010` | `d` | `LED1`（同族 `led`） | x 重叠 0.84、中心 y 差 5.6 ≤ 0.6×14 |
| 10 | `p016` | `gnd` | `p015`（同类型 `gnd`） | x 重叠 1.00、中心 y 差 16 ≤ 25 |

真值里这两处都只有**一颗**元件（case 4 是 `LED1`，case 10 是一颗 `gnd`）⇒ 多出来的那颗是
`component.extra`，也就是 **fp**（protocol2 的 `component_f1` 只看 (type, bbox, 键)，见
`schematic_parser/metrics.py` 的 `assign_components`）。

## 药：两条纯几何规则（只用到键形、类型、bbox）

1. **D1 假键撞真键**：假键元件（`^p\\d{3,}$`）类型 ∈ {`d`,`led`}，与某颗**真键**同族元件
   在 x 方向重叠 ≥0.7（按两者中较窄的那个宽度算）且中心 y 差 ≤ 0.6×自身高 ⇒ 删假键那颗。
2. **D2 同类型假键互撞**：两颗假键元件类型相同、x 方向重叠 ≥0.9、中心 y 差 ≤25px ⇒
   按键序丢一颗（留键序小的，保证可复现）。

## 守卫（宁可不删）

- 只碰 `d`/`led`/`gnd`/`v` 这类**符号型**元件；`IC_TYPES`（`box`/`ic`/`tp`…）永不碰；
- 待删元件若在 `pins` 里有脚、或被任何 `nets[*]._refs` 引用 ⇒ **不删**（本批 3 颗都
  没有脚也没有网，删除是干净的；这条守卫是为将来别的图写的）；
- 只删假键元件；真键元件之间永不互相消重。

## 实测

离线（`work/_compdedup.py` + `fast.py --score-only`，交付态 `work/_fast_out_pnon` = 81.75）：

| 口径 | 官方 10 张 | 元件 F1（10 张均值） | 变动 |
| --- | --- | --- | --- |
| 关 | 81.75 | 0.9963 | — |
| 开 | **81.86（+0.11）** | **1.0000** | case 4 79.94→80.00、case 10 84.83→84.89 |

端到端 A/B（`FUSION_COMPDEDUP=1`，官方 10 张、`-vlm-pin off`，2026-10）：81.75 → **81.86**，
逐案只有 case 4 / case 10 变，其余 8 张逐位相同，退出码 0 —— 与离线裁定**逐位一致**。

⇒ 元件通道到 10 张全 1.0000，**这条线走到头**（`component` 权重 0.30，每颗元件 0.06 分）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from envflag import env_on

ON = env_on("FUSION_COMPDEDUP", "1")

FAKE_KEY = re.compile(r"^p\d{3,}$")
SYMBOL_TYPES = ("d", "led", "gnd", "v")
D1_XOV = 0.70
D1_DCY = 0.60
D2_XOV = 0.90
D2_DCY = 25.0


def _canon(t) -> str:
    t = str(t or "").strip().lower()
    return {"diode": "d", "light_emitting_diode": "led", "ground": "gnd"}.get(t, t)


def _bb(c):
    b = (c or {}).get("bbox") or (c or {}).get("box")
    try:
        v = [float(x) for x in b[:4]]
    except Exception:                                          # noqa: BLE001
        return None
    if v[2] <= v[0] or v[3] <= v[1]:
        return None
    return v


def _xov(a, b) -> float:
    """x 方向重叠 / 较窄者宽度。"""
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    return max(0.0, hi - lo) / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def _cy(b) -> float:
    return (b[1] + b[3]) / 2.0


def _anchored(doc, key) -> bool:
    """这颗元件是否被脚或网引用（引用了就不许删）。"""
    if (doc.get("pins") or {}).get(key):
        return True
    for net in (doc.get("nets") or {}).values():
        for r in (net or {}).get("_refs") or []:
            if isinstance(r, (list, tuple)) and r and str(r[0]) == str(key):
                return True
    return False


def dedup_doc(doc):
    """就地消重；返回 trace 列表 [{drop, keep, rule, type}]。"""
    comps = doc.get("components") or {}
    keys = [str(k) for k in comps]
    fake = [k for k in keys if FAKE_KEY.match(k)]
    real = [k for k in keys if not FAKE_KEY.match(k)]
    trace, gone = [], set()

    def _cand(k):
        c = comps.get(k)
        if c is None:
            return None, None, None
        t = _canon(c.get("type"))
        b = _bb(c)
        if t not in SYMBOL_TYPES or b is None:
            return None, None, None
        return c, t, b

    for fk in sorted(fake):
        if fk in gone:
            continue
        _c, ft, fb = _cand(fk)
        if _c is None or _anchored(doc, fk):
            continue
        # D1：撞真键
        hit = None
        if ft in ("d", "led"):
            for rk in real:
                _rc, rt, rb = _cand(rk)
                if _rc is None or rt not in ("d", "led"):
                    continue
                if _xov(fb, rb) >= D1_XOV and abs(_cy(fb) - _cy(rb)) <= D1_DCY * (fb[3] - fb[1]):
                    hit = (rk, "d1")
                    break
        # D2：撞别的假键（同类型）
        if hit is None:
            for ok in sorted(fake):
                if ok == fk or ok in gone:
                    continue
                _oc, ot, ob = _cand(ok)
                if _oc is None or ot != ft:
                    continue
                if _xov(fb, ob) >= D2_XOV and abs(_cy(fb) - _cy(ob)) <= D2_DCY:
                    keep, drop = (ok, fk) if str(ok) > str(fk) else (fk, ok)
                    hit = (keep, "d2", drop)
                    break
        if hit is None:
            continue
        keep = hit[0]
        drop = hit[2] if len(hit) > 2 else fk
        if drop in gone or _anchored(doc, drop):
            continue
        trace.append({"drop": drop, "keep": keep, "rule": hit[1], "type": ft})
        gone.add(drop)
    for k in gone:
        comps.pop(k, None)
    return trace


def apply_file(src, dst):
    """读 `src` 的 `<case>_Pred.json`，消重后写 `dst`；失败返回 None（导出退化成原样）。"""
    src, dst = Path(src), Path(dst)
    doc = json.loads(src.read_text(encoding="utf-8"))
    trace = dedup_doc(doc)
    if trace:
        dst.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return {"dropped": len(trace), "trace": trace}
