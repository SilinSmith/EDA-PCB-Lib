# -*- coding: utf-8 -*-
"""网线端点「系统性外伸」的偏置校正（`FUSION_EDGECAL`，默认 1 = 开）。

病（`work/_netl2.py` 逐边归因，基线 83.03 / 导线 F1 0.5291）：
判据 `schematic_parser/metrics.py:611` 用 `_seg_distance`（两端点距离取 max）配边、
`cfg.line_tol = 5.0`（`metrics.py:88`）；配对网里**未配上的边**共 288 fp / 290 fn，
端点沿 GT 轴的纵向差直方图在 **+2px 处一个尖峰（118 个端点）**，
`超出 310 / 齐 7 / 短了 87` ⇒ 我们的端点整体比 GT **多伸出去 ~1px**。
来源是上游的外推规则（`stage2/assemble.py:694 EDGE_EXT=20`（推到同网他边 3px 内）、
`:788 EDGE_EXT_ANY=25`，另有 `:681 EDGE_TRIM=3`（自由端回缩）、`:711 EDGE_EXT_FREE=8`）。

药：把每条 `nets[*].edges` 的**每个端点**沿自身方向内收 ——
端点在**同网另一条边的 `NEAR2` px 内**（即上游外推正好停在这里，GT 则是零间隙接上）
⇒ 收 `TRIM_NEAR` px；其余端点收 `TRIM` px。只改端点坐标 ⇒ 元件 / 引脚 / netH 三通道不变。

离线裁定（`work/_edgetrimx.py` + `fast.py --score-only`，官方 10 张，基线 83.03）：
| 规则 | 总分 | 导线 F1 |
|---|---|---|
| 均匀内收 1.0px | 83.16 | 0.5427 |
| 均匀内收 2.0 / 3.0px | 83.09 / 82.82 | — |
| 只收「自由端」1.0px | 83.09 | 0.5354 |
| 吸到同网他边（snap 6px） | 83.00 | 0.5267 |
| snap + 均匀 1px（mix） | 83.11 | 0.5371 |
| 「两端都贴边」才收 3px（cond） | 83.22 | 0.5480 |
| **逐端：贴边收 4.0px、其余收 1.0px（perend，本实现）** | **83.23** | **0.5490** |
| 同上 x2=4.5 / 5.0 | 83.23 / 83.12 | 0.5498 / 0.5382 |
| 同上 x=0.7 / x2=2.0 | 83.20 / 83.16 | 0.5463 / 0.5422 |

必须排在 `fusion/pwrclip.py` **之后**（吸附会重新移动端点）。

端到端 A/B（真实管线、官方 10 张、`-vlm-pin off`，`work/_ec1.log`）：
均匀 1px 版 **83.03 → 83.16（+0.13）**（导线 0.5291 → 0.5427、tp 349 → 355，内收 637 条边）；
对照两条上游外推开关（同一批、`work/_extany0.log` / `work/_extfree5.log`）：
`STAGE2_EDGE_EXT_ANY=0` **82.78**、`STAGE2_EDGE_EXT_FREE=5` **82.91**
⇒ 外推规则本身是赚的，但它留下了**向外 ~1px 的系统偏置**，本校正正好抵消。
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


ON = _env_on("FUSION_EDGECAL", "1")          # ★ 默认 1：官方 10 张 83.03 → 83.23（+0.20）
TRIM = _env_float("FUSION_EDGECAL_TRIM", 1.0)          # 普通端点内收 px
TRIM_NEAR = _env_float("FUSION_EDGECAL_TRIM_NEAR", 4.0)  # 贴着同网他边的端点内收 px
NEAR2 = _env_float("FUSION_EDGECAL_NEAR", 3.5)          # 「贴着」判定半径 px


def _pt(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _out(p, x, y):
    if isinstance(p, dict):
        return {"x": round(x, 2), "y": round(y, 2)}
    return [round(x, 2), round(y, 2)]


def _dist_to_other(pt, k, segs):
    """pt 到**同网他边**的最近距离（无他边时返回 1e9）。"""
    best = 1e9
    px, py = pt
    for k2, (q0, q1) in segs.items():
        if k2 == k:
            continue
        ax, ay = q0
        bx, by = q1
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 <= 1e-9:
            t = 0.0
        else:
            t = ((px - ax) * dx + (py - ay) * dy) / L2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        qx, qy = ax + dx * t, ay + dy * t
        d = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
        if d < best:
            best = d
    return best


def trim_doc(doc: dict, trim: float = None, trim_near: float = None,
             near: float = None) -> dict:
    """就地内收网线端点（贴边端收得多）；返回 {"moved": 边数, "trim": ...}。"""
    t0 = TRIM if trim is None else float(trim)
    t1 = TRIM_NEAR if trim_near is None else float(trim_near)
    nr = NEAR2 if near is None else float(near)
    n = 0
    trace = []
    if t0 <= 0 and t1 <= 0:
        return {"moved": 0, "trace": trace}
    for net_id, net in (doc.get("nets") or {}).items():
        edges = net.get("edges") or {}
        segs = {}
        for k, seg in edges.items():
            if isinstance(seg, list) and len(seg) == 2:
                segs[k] = (_pt(seg[0]), _pt(seg[1]))
        for k, seg in list(edges.items()):
            if k not in segs:
                continue
            k0, k1 = segs[k]
            dx, dy = k1[0] - k0[0], k1[1] - k0[1]
            L = (dx * dx + dy * dy) ** 0.5
            if L <= 1e-6:
                continue
            ux, uy = dx / L, dy / L
            new = []
            for pt, sgn in ((k0, 1.0), (k1, -1.0)):
                amt = t1 if _dist_to_other(pt, k, segs) <= nr else t0
                t = min(amt, L * MAXFRAC) if amt > 0 else 0.0
                new.append((pt[0] + sgn * ux * t, pt[1] + sgn * uy * t))
            if (abs(new[0][0] - k0[0]) < 1e-9 and abs(new[1][0] - k1[0]) < 1e-9):
                continue
            edges[k] = [_out(seg[0], new[0][0], new[0][1]),
                        _out(seg[1], new[1][0], new[1][1])]
            n += 1
            if len(trace) < 40:
                trace.append({"net": net_id, "edge": k, "len": round(L, 1)})
    return {"moved": n, "trim": round(t0, 3), "trim_near": round(t1, 3), "trace": trace}


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
