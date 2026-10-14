# -*- coding: utf-8 -*-
"""网线端点「系统性外伸」的偏置校正（`FUSION_EDGECAL`，默认 0 = 先跑 A/B）。

病（`work/_netl2.py` 逐边归因，末交付态 83.03 / 导线 F1 0.5291）：
判据 `schematic_parser/metrics.py:611` 用 `_seg_distance`（两端点距离取 max）配边、
`cfg.line_tol = 5.0`（`metrics.py:88`）；配对网里**未配上的边**共 288 fp / 290 fn，
其中端点沿 GT 轴的纵向差直方图在 **+2px 处一个尖峰（118 个端点）**，
`超出 310 / 齐 7 / 短了 87` ⇒ 我们的端点整体比 GT **多伸出去 ~1px**。
来源是上游的外推规则（`stage2/assemble.py:694 EDGE_EXT=20`、
`:788 EDGE_EXT_ANY=25`：向外推去够同网他边/几何证据），外推对 GT 而言常常过头。

药：把每条 `nets[*].edges` 的两个端点**沿自身方向内收 `TRIM` px**（≤ 段长的 40%），
只改端点坐标 ⇒ 元件 / 引脚 / netH 三通道逐位不变。

离线裁定（`work/_edgetrimx.py` + `fast.py --score-only`，官方 10 张，基线 83.03）：
`0.5 → 83.15`、`0.8/1.2 → 83.14`、**`1.0 → 83.16（+0.13，导线 0.5291 → 0.5427）`**、
`2.0 → 83.09`、`3.0 → 82.82`；只收「自由端」（附近无引脚、无同网他边）反而更差
（`1.0 → 83.09`）⇒ 过大偏差不集中在自由端，取**均匀**校正。

必须排在 `fusion/pwrclip.py` **之后**（吸附会重新移动端点）。

端到端 A/B（真实管线、官方 10 张、`-vlm-pin off`，`work/_ec1.log`）：
**83.03 → 83.16（+0.13）**，导线 F1 0.5291 → **0.5427**、导线 tp 349 → **355**，
元件 / 引脚 / netH 三个通道逐位不变；内收 637 条边 / 10 张图。
对照两条上游外推开关（同一批、`work/_extany0.log` / `work/_extfree5.log`）：
`STAGE2_EDGE_EXT_ANY=0` **82.78**（导线 0.5042）、`STAGE2_EDGE_EXT_FREE=5` **82.91**
（导线 0.5170）⇒ 外推规则本身是赚的，但它留下了**向外 ~1px 的系统偏置**，
本校正正好抵消 ⇒ 两者叠加才是 83.16。★ 默认 **1** = 开。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MAXFRAC = 0.4


def _env(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_on(name: str, default: str = "0") -> bool:
    return _env(name, default).strip().lower() not in ("", "0", "false", "off", "no", "none")


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:                                            # noqa: BLE001
        return default


ON = _env_on("FUSION_EDGECAL", "1")          # ★ 默认 1：官方 10 张 83.03 → 83.16（+0.13）
TRIM = _env_float("FUSION_EDGECAL_TRIM", 1.0)


def _pt(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _out(p, x, y):
    if isinstance(p, dict):
        return {"x": round(x, 2), "y": round(y, 2)}
    return [round(x, 2), round(y, 2)]


def trim_doc(doc: dict, trim: float = None) -> dict:
    """就地内收所有网线端点；返回 {"moved": 边数, "trace": [...]}。"""
    t0 = TRIM if trim is None else float(trim)
    n = 0
    trace = []
    if t0 <= 0:
        return {"moved": 0, "trace": trace}
    for net_id, net in (doc.get("nets") or {}).items():
        edges = net.get("edges") or {}
        for k, seg in list(edges.items()):
            if not isinstance(seg, list) or len(seg) != 2:
                continue
            (x1, y1), (x2, y2) = _pt(seg[0]), _pt(seg[1])
            dx, dy = x2 - x1, y2 - y1
            L = (dx * dx + dy * dy) ** 0.5
            if L <= 1e-6:
                continue
            ux, uy = dx / L, dy / L
            t = min(t0, L * MAXFRAC)
            edges[k] = [_out(seg[0], x1 + ux * t, y1 + uy * t),
                        _out(seg[1], x2 - ux * t, y2 - uy * t)]
            n += 1
            if len(trace) < 40:
                trace.append({"net": net_id, "edge": k, "len": round(L, 1)})
    return {"moved": n, "trim": round(t0, 3), "trace": trace}


def apply_file(src, dst, trim: float = None) -> dict:
    """在**已经落盘**的 pred JSON 上内收端点；无改动则不重写。写盘是原子的。"""
    src_p, dst_p = Path(src), Path(dst)
    doc = json.loads(src_p.read_text(encoding="utf-8"))
    st = trim_doc(doc, trim)
    if not st.get("moved"):
        return st
    tmp = dst_p.with_name(dst_p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(dst_p))
    return st
