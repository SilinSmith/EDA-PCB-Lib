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

## §T 型接点伸长（`FUSION_EDGECAL_JUNC`，默认 **12 px**；设 0 关闭）

病（`work/_line_diag13.py` + `work/_line_anchor.py`，13 页 `open_GT-1106`，基线导线 F1
0.6258 = tp 492 / fp 316 / fn 252）：252 条未命中 GT 边里 **75 条只差 5~12px**、
51 条差 12~28px，且这些边的**法向残差 p50 只有 0.3px、95.2% ≤ 2px** —— 线是对的，
错的只有**沿线方向的弧长**；其中 **61.9% 是我们停短了**。形态是「我们的一条边停在
同网另一条边**近旁几像素处**，而 GT 那条边一直画到交点（T 型接点）为止」。

药：把端点沿**自身线方向**伸长到它与**本网另一条边**的最近交点，条件是交点落在
他边**线段内部**（`s∈[0,L]`）、距离 `≤ JUNC` px、两线夹角 `|sin| ≥ JUNC_PERP`。
只伸长、不缩短（oracle 拆分：伸长 +0.0782 / 缩短 +0.0120）。只动端点坐标 ⇒
元件/引脚/netH 三轴不变（`net_pin_set` 只读 `hyperGraph`）。

离线裁定（`work/_line_proto.py`，13 页、官方 `evaluate_case` + `CFG`，基线 0.6258）：
| 规则 | mean line | Δ | tp/fp/fn |
|---|---|---|---|
| 同网交点、只伸长，r=8/10/12 | 0.6302 | **+0.00438** | 496/312/248 |
| 同上 r=15 / r=18 | 0.6313 | **+0.00552** | 497/311/247 |
| 同上 r=20 | 0.6303 | +0.00452 | 496/312/248 |
| 同上 r=25（悬崖，勿用） | 0.5958 | −0.02996 | — |
| 放宽到「交点可落在线段外 4px」 | — | +0.00035 | ⇒ `s∈[0,L]` 是承重约束 |
| 要求对方线**自身端点**在交点 ≤2px（真 T） | 0.6301 | +0.00431 | 略低于上者 |
| 允许跨网交点 | — | 与同网**逐键相同** ⇒ 有用的交点全在同网内 |
| 单用 `dedup`(tol=5) / `minlen`(L=12) | — | +0.0011 / +0.0014 | 另一机制，未启用 |

判死（同一批实测，勿重跑）：引脚吸附 −0.0060~−0.0196、沿自身线到引脚 ≈0、
沿自身线到元件框边 −0.0368、同网共线填缝 −0.0009、删被包含子段 −0.0023。

门闸实测（13 页 `work/_cv`，同一棵树，`linej12off5.json` → `linej12t2.json`）：
`JUNC=12`（`JUNC_STUB` 取默认 2）= line 0.6258→**0.6297(+0.00394)**、mean 88.047→88.086、
official 86.322→86.361、judge 84.304→84.344；comp/pin/netH **逐位不变**；逐页只有
1/6/11 上升（+0.0061 / +0.0118 / +0.0333），**无任何页面下降**、无任何页增加/删除边。
另一棵上游树（`linej12off2` → `linej12t`）复现同一 +0.00394。

`JUNC_STUB=0`（不要求真 T）实测 line **+0.00534**，但 case 5 −0.0052：伸长把一对原本重合
的边扰开后，上游去重逻辑多留了一条假边（`net_2`，66→67 条）；case 1 的 +0.0296 里也含
同样机制的「复活边」。⇒ 想要更大收益可设 0，但要接受这种偶发假边；保守用默认 2。

**默认仍是 0 = 关**（本篇只改端点、不改开关默认），推荐值 **12**（离线 +0.00438、
门闸 +0.00394，距 r=25 悬崖 13px）；追离线 +0.0055 可试 15（门闸未验）。
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


# ★ 2026-11-04 曾翻默认 1 → 0：13 页合跑 **+0.0828**（2 +0.71 / 9 +0.63 / 12 +0.24 / 13 +0.19），
#   但同一次里 **case 5 −0.68**、case 4 −0.17、case 10 −0.08 —— 单页跌幅是增益页的 10 倍，
#   属「按页换分」而非机制改进。按「无单页 −>0.05」的纪律**回退到 1**；本轮的正式交付只留
#   `STAGE2_PIN_NUMBER_REPAIR=0`（只有 case 11 +1.34，其余 12 页逐位不动）。
ON = _env_on("FUSION_EDGECAL", "1")
TRIM = _env_float("FUSION_EDGECAL_TRIM", 1.0)          # 普通端点内收 px
TRIM_NEAR = _env_float("FUSION_EDGECAL_TRIM_NEAR", 4.0)  # 贴着同网他边的端点内收 px
NEAR2 = _env_float("FUSION_EDGECAL_NEAR", 3.5)          # 「贴着」判定半径 px
# ★ 2026-11-07：`FUSION_EDGECAL_JUNC`（端点沿自身线伸长到**同网 T 型交点**的半径 px，
#   **默认 12 px**；设 0 关闭）。依据见文件顶注「§T 型接点伸长」。为 0 时 `extend_doc` 直接早退，
#   本文件输出与改动前逐键相同。
JUNC = _env_float("FUSION_EDGECAL_JUNC", 12.0)
JUNC_PERP = _env_float("FUSION_EDGECAL_JUNC_PERP", 0.05)  # 交点处 |sin(夹角)| 下限
# 只认**真 T 型接点**：交点必须落在他边**端点** `JUNC_STUB` px 内（`0` = 不设此约束）。
# 原理图里「一条线从另一条线中腰穿过」不成接点（无 junction 点），GT 也不会在那里收笔。
JUNC_STUB = _env_float("FUSION_EDGECAL_JUNC_STUB", 2.0)


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


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L <= 1e-6:
        return None
    return dx / L, dy / L


def _cross_t(d, e, a, b, perp):
    """直线 `e + t*d` 与线段 `a..b` 的交点参数 t（交点须落在段内）；不交返回 None。

    同交给交点到他边**近端**的距离 `min(s, L-s)`（判「真 T 型」用）。
    """
    u = _unit(a, b)
    if u is None:
        return None
    den = d[0] * u[1] - d[1] * u[0]
    if abs(den) < perp:                                   # 近平行：不是交点
        return None
    wx, wy = a[0] - e[0], a[1] - e[1]
    t = (wx * u[1] - wy * u[0]) / den
    s = (wx * d[1] - wy * d[0]) / den
    L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    if s < 0.0 or s > L:
        return None
    return t, min(s, L - s)


def extend_doc(doc: dict, r: float = None, perp: float = None,
               stub: float = None) -> dict:
    """把端点沿**自身线方向**伸长到与本网其它边的最近交点（T 型接点）。

    只伸长不缩短；交点必须落在**他边线段内部**（`s∈[0,L]`）、距他边近端 `≤ stub` px
    （`stub=0` 关掉该限制）、半径 `r` px 内取最近。
    返回 {"extended": 边数, ...}；只改端点坐标 ⇒ 元件/引脚/netH 三通道不变。
    """
    rr = JUNC if r is None else float(r)
    pp = JUNC_PERP if perp is None else float(perp)
    sb = JUNC_STUB if stub is None else float(stub)
    out = {"extended": 0, "junc": round(rr, 3), "stub": round(sb, 3), "trace": []}
    if rr <= 0:
        return out
    for net_id, net in (doc.get("nets") or {}).items():
        edges = net.get("edges") or {}
        segs = {}
        for k, seg in edges.items():
            if isinstance(seg, list) and len(seg) == 2:
                segs[k] = (_pt(seg[0]), _pt(seg[1]))
        if len(segs) < 2:                                 # 单边网没有「他边」
            continue
        for k, seg in list(edges.items()):
            if k not in segs:
                continue
            p0, p1 = segs[k]
            cur = [p0, p1]
            for end in (0, 1):
                e, o = cur[end], cur[1 - end]
                d = _unit(o, e)
                if d is None:
                    continue
                best = None
                for k2, (a, b) in segs.items():
                    if k2 == k:
                        continue
                    hit = _cross_t(d, e, a, b, pp)
                    if hit is None:
                        continue
                    t, near = hit
                    if t <= 0.0 or t > rr:
                        continue
                    if sb > 0 and near > sb:              # 不是真 T 型接点（他边只是穿过）
                        continue
                    if best is None or t < best:
                        best = t
                if best is not None:
                    cur[end] = (e[0] + best * d[0], e[1] + best * d[1])
            if cur[0] != p0 or cur[1] != p1:
                edges[k] = [_out(seg[0], cur[0][0], cur[0][1]),
                            _out(seg[1], cur[1][0], cur[1][1])]
                out["extended"] += 1
                if len(out["trace"]) < 40:
                    out["trace"].append({"net": net_id, "edge": k})
    return out


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


def apply_file(src, dst, trim: float = None, junc: float = None) -> dict:
    """在**已经落盘**的 pred JSON 上内收端点（+ 可选 T 型接点伸长）；无改动则不重写。"""
    src_p, dst_p = Path(src), Path(dst)
    doc = json.loads(src_p.read_text(encoding="utf-8"))
    st = trim_doc(doc, trim)
    ex = extend_doc(doc, junc)                            # 必须在内收**之后**：伸长位置以收完的线为准
    if ex.get("extended"):
        st["extended"] = ex["extended"]
        st["junc"] = ex["junc"]
    if not st.get("moved") and not ex.get("extended"):
        return st
    tmp = dst_p.with_name(dst_p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(dst_p))
    return st
