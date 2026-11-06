# -*- coding: utf-8 -*-
"""库表补脚：对库能查到的 IC/盒体件，把**库里有、我们没检出的脚**按格点补上。

动机（§5.591）：13 页 31 个 IC/box 件里 21 个能 `LibCtx.key_for(Name)` 命中，
且**命中的库脚数与 GT 脚数完全相等**；其中 9 根脚我们漏检（都是无引线的 NC 行或
屏蔽脚）。补脚的上限实测：位置用 GT、名字用库表 ⇒ **judge 尺 +0.6377**，而库表给出的
脚名与 GT **9/9 完全一致**（`work/_pinadd_liborc.py`）。

规则的保守性（三条全过才补，宁缺勿错）：
1. 同一行/列（坐标差 ≤3px）里至少 2 根带整数号的脚，用中位间距拟合格点，且
   `号 = off + step*槽号`（step=±1）对该行**每个**脚都严格成立；
2. 缺号的槽必须在 [最小槽-1, 最大槽+1] 内且当前为空（6px 内没有别的脚）；
3. 库表能给出该号的脚名 —— 无名补脚在判分器里是 fp+fn 双扣（§5.588）。

坐标系：`doc` 里的 `pins[*].point` 是 **y-up** JSON 帧（与预测文件一致），本模块
不读图，所以不需要翻转。
"""
from __future__ import annotations

import statistics

try:  # 交付包里 `envflag` 是顶层模块；作为包导入时走相对路径
    from envflag import env_on
except ImportError:  # pragma: no cover
    from ..envflag import env_on

from . import libpins as L

ON = env_on("STAGE2_PINFILL", "1")
BOX_TYPES = ("box", "ic", "boxic", "box_ic")
COORD_TOL = 3.0
MIN_PITCH = 3.0
FREE_TOL = 6.0


def _pin_num(key):
    return key[4:] if key.startswith("pin_") else key


def _pins(doc, comp):
    v = (doc.get("pins") or {}).get(comp) or {}
    return v if isinstance(v, dict) else {}


def _recs(pins):
    out = []
    for k, rec in pins.items():
        p = (rec or {}).get("point") or {}
        if p.get("x") is None or p.get("y") is None:
            continue
        out.append((_pin_num(k), float(p["x"]), float(p["y"])))
    return out


def _lines(recs):
    """按「同一行 y / 同一列 x」聚组；不构成行或列的组合丢弃。"""
    left = list(recs)
    out = []
    while left:
        head = left.pop(0)
        grp = [head]
        for r in list(left):
            if abs(r[2] - head[2]) <= COORD_TOL or abs(r[1] - head[1]) <= COORD_TOL:
                grp.append(r)
                left.remove(r)
        ys = [g[2] for g in grp]
        xs = [g[1] for g in grp]
        if max(ys) - min(ys) <= COORD_TOL:
            out.append(("H", statistics.fmean(ys), grp))
        elif max(xs) - min(xs) <= COORD_TOL:
            out.append(("V", statistics.fmean(xs), grp))
    return out


def _one_line(axis, coord, grp, missing, ctx, key, recs):
    if len(grp) < 2:
        return []
    ints = []
    for n, x, y in grp:
        if not str(n).isdigit():
            return []
        ints.append((int(n), x, y))
    ints.sort(key=lambda t: t[1] if axis == "H" else t[2])
    pos = [t[1] if axis == "H" else t[2] for t in ints]
    gaps = [b - a for a, b in zip(pos, pos[1:]) if b - a > 1.0]
    if not gaps:
        return []
    pitch = statistics.median(gaps)
    if pitch < MIN_PITCH:
        return []
    p0 = pos[0]
    slots = [int(round((p - p0) / pitch)) for p in pos]
    if len(set(slots)) != len(slots):
        return []
    off = ints[0][0] - slots[0]
    step = None
    for (n, _x, _y), s in zip(ints, slots):
        if s == slots[0]:
            continue
        cand = (n - off) / (s - slots[0])
        if cand not in (1, -1):
            return []
        if step is None:
            step = cand
        elif step != cand:
            return []
    if step is None:
        return []
    smin, smax = min(slots), max(slots)
    adds = []
    for want in missing:
        if not str(want).isdigit():
            continue
        n_w = int(want)
        k = (n_w - off) / step
        if k != int(k):
            continue
        k = int(k)
        if k < smin - 1 or k > smax + 1 or k in slots:
            continue
        p = p0 + k * pitch
        px = p if axis == "H" else coord
        py = coord if axis == "H" else p
        if any(abs(px - r[1]) <= FREE_TOL and abs(py - r[2]) <= FREE_TOL for r in recs):
            continue
        nm = ctx.name_for_num(key, want)
        if not nm:
            continue
        adds.append((want, px, py, nm))
    return adds


def fill(doc, ctx=None):
    stats = {"on": ON, "comps": 0, "added": 0, "keys": 0, "skipped": 0}
    if not ON:
        return stats
    pins_all = doc.get("pins") or {}
    comps = doc.get("components") or {}
    own = ctx is None
    if own:
        ctx = L.LibCtx()
        try:
            ctx.load()
        except Exception as e:  # noqa: BLE001
            stats["error"] = "%s: %s" % (type(e).__name__, e)
            return stats
    for comp in list(pins_all.keys()):
        c = comps.get(comp) or {}
        if str(c.get("type") or "") not in BOX_TYPES:
            continue
        name = str(c.get("Name") or "")
        if not name:
            continue
        key = ctx.key_for(name)
        if not key:
            continue
        stats["keys"] += 1
        pins = _pins(doc, comp)
        recs = _recs(pins)
        if not recs:
            continue
        have = {r[0] for r in recs}
        missing = [n for n in ctx.nums_all(key) if n not in have]
        if not missing:
            continue
        adds = []
        for axis, coord, grp in _lines(recs):
            adds.extend(_one_line(axis, coord, grp, missing, ctx, key, recs))
        if not adds:
            stats["skipped"] += 1
            continue
        stats["comps"] += 1
        for num, x, y, nm in adds:
            pins["pin_%s" % num] = {"pinname": nm, "point": {"x": x, "y": y},
                                    "src": "libfill"}
            stats["added"] += 1
    return stats
