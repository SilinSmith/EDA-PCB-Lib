# -*- coding: utf-8 -*-
"""引脚第④关（位置 ≤ `cfg.pin_tol` = 5.0 px）到底能不能靠**纯几何**救回来。

背景（`_pinfunnel.py` 已量）：GT 引脚 9356 条里，能走到第④关的只有 353 条
（②号配上 1083 → ③名相等 353），而这 353 条里 **175 条距离 > 5px** 被判
`pin.position_mismatch` —— 也就是说「元件键对、引脚号对、引脚名也对」的三关全过的
引脚，**有一半死在最后一个像素级判据上**，而 tp 只有 178。

引脚点是我方在 `eda-pcb\\stage2\\assemble.py:588-668` 里算出来的
（`point: {"x": px, "y": round(H - py, 2)}`），**不是识别出来的** —— 所以如果
「真值点总是落在我方元件框的边上」这件事成立，那么把点**投影到框边**就是一次
零成本（不必重跑任何模型）的 tp 增益。

本脚本对同一批「三关全过」的引脚，逐个候选变换量 tp：

  T0  原样（基线）
  T1  把我方点**吸附到我方元件框的最近边**
  T2  把我方点吸附到**真值元件框**的最近边（这是「框是准的」的上界，用来判断
      到底是「点算错了」还是「框本身就偏了」）
  T3  把我方点吸附到**我方元件框外扩 pad 像素**后的最近边（pad = 1..6）
  T4  把我方点沿着「从元件中心指向该点」的射线**推到框边**（等价于按方向归一）

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_pinsnap.py
    python work\\tools\\diag\\_pinsnap.py --pred work\\_fusion_out --top 12
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES,             # noqa: E402
                                      assign_components, canon_type,
                                      pin_key_to_number, _point)
import stage2_eval as SE                                                # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
TOL = 5.0


def _box(b):
    if not b or len(b) < 4:
        return None
    v = [float(x) for x in b[:4]]
    if v[2] < v[0]:
        v[0], v[2] = v[2], v[0]
    if v[3] < v[1]:
        v[1], v[3] = v[3], v[1]
    return v


def _nearest_on_rect(pt, box):
    """点到矩形**边界**的最近点；点在框内时吸到最近的那条边。"""
    if box is None:
        return pt
    x, y = pt
    x0, y0, x1, y1 = box
    cx = min(max(x, x0), x1)
    cy = min(max(y, y0), y1)
    if x0 <= x <= x1 and y0 <= y <= y1:            # 框内 → 吸边
        cands = [(x, y0), (x, y1), (x0, y), (x1, y)]
        return min(cands, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    # 框外 → 四个角/边上的夹取点里取最近的（此处 (cx,cy) 已在边界上或角上）
    cands = [(cx, y0), (cx, y1), (x0, cy), (x1, cy)]
    return min(cands, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)


def _radial(pt, box, ctr):
    """沿着 中心→点 的方向把点推到框边上（方向不变，距离归一）。"""
    if box is None:
        return pt
    x, y = pt
    dx, dy = x - ctr[0], y - ctr[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return pt
    x0, y0, x1, y1 = box
    ts = []
    if abs(dx) > 1e-6:
        for bx in (x0, x1):
            t = (bx - ctr[0]) / dx
            if t > 0:
                yy = ctr[1] + t * dy
                if y0 - 1e-6 <= yy <= y1 + 1e-6:
                    ts.append(t)
    if abs(dy) > 1e-6:
        for by in (y0, y1):
            t = (by - ctr[1]) / dy
            if t > 0:
                xx = ctr[0] + t * dx
                if x0 - 1e-6 <= xx <= x1 + 1e-6:
                    ts.append(t)
    if not ts:
        return pt
    t = min(ts)
    return (ctr[0] + t * dx, ctr[1] + t * dy)


def _expand(box, pad):
    return [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad] if box else None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    rows = []                 # (case, comp, num, gtpt, ourpt, ourbox, gtbox, ctr)
    st = collections.Counter()

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None or (a.split and case.split != a.split):
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            continue
        box_comps = {k for k, v in (gt.get("components") or {}).items()
                     if canon_type((v or {}).get("type")) in IC_TYPES}
        pk_pins = {k2g.get(pk, pk): grp for pk, grp in (pr.get("pins") or {}).items()}
        for gk in box_comps:
            ggrp = (gt.get("pins") or {}).get(gk) or {}
            pgrp = pk_pins.get(gk) or {}
            if not isinstance(ggrp, dict) or not isinstance(pgrp, dict):
                continue
            gnum = {pin_key_to_number(k): v for k, v in ggrp.items()}
            pnum = {pin_key_to_number(k): v for k, v in pgrp.items()}
            gbox = _box((((gt.get("components") or {}).get(gk)) or {}).get("bbox"))
            # 我方框：先按键取，取不到就退回「同键」的预测元件框
            pbox = _box((((pr.get("components") or {}).get(gk)) or {}).get("bbox"))
            for num in set(gnum) & set(pnum):
                g, p = gnum[num], pnum[num]
                if not isinstance(g, dict) or not isinstance(p, dict):
                    continue
                if g.get("pinname") != p.get("pinname"):
                    continue                      # 只看三关全过、只剩④的
                ga, pa = _point(g.get("point")), _point(p.get("point"))
                if ga is None or pa is None:
                    continue
                ctr = None
                if pbox:
                    ctr = ((pbox[0] + pbox[2]) / 2.0, (pbox[1] + pbox[3]) / 2.0)
                rows.append((case.case_id, gk, num, ga, pa, pbox, gbox, ctr))

    def d2(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def score(fn, label):
        tp = 0
        for r in rows:
            q = fn(r)
            if q is not None and d2(q, r[3]) <= TOL:
                tp += 1
        st[label] = tp
        return tp

    n = len(rows)
    print("== 三关全过、只剩④的引脚 n = %d ==" % n)
    print("  T0 原样                 tp %4d  %6.1f%%" % (score(lambda r: r[4], "T0"), 100.0 * st["T0"] / max(1, n)))
    print("  T1 吸附我方框边         tp %4d  %6.1f%%" % (score(lambda r: _nearest_on_rect(r[4], r[5]), "T1"), 100.0 * st["T1"] / max(1, n)))
    print("  T2 吸附真值框边(上界)   tp %4d  %6.1f%%" % (score(lambda r: _nearest_on_rect(r[4], r[6]), "T2"), 100.0 * st["T2"] / max(1, n)))
    for pad in (1, 2, 3, 4, 6, 8):
        k = "T3_pad%d" % pad
        print("  T3 吸附我方框边+外扩%-2d  tp %4d  %6.1f%%"
              % (pad, score(lambda r, p=pad: _nearest_on_rect(r[4], _expand(r[5], p)), k),
                 100.0 * st[k] / max(1, n)))
    print("  T4 沿中心→点 推到框边   tp %4d  %6.1f%%"
          % (score(lambda r: _radial(r[4], r[5], r[7]) if r[7] else None, "T4"),
             100.0 * st["T4"] / max(1, n)))

    # T5 全局平移扫描：如果「我方点相对真值点有一个系统的常数偏移」，那么一次
    # 常数平移就是零成本增益。注意这**不是**径向/切向按框变化的偏差（T1/T2 已经
    # 否掉了那条路），而是纯粹的整体错位。
    print("\n== T5 全局平移扫描（tp / %d） ==" % n)
    best = (st["T0"], 0, 0)
    print("   dx:", " ".join("%5d" % dx for dx in range(-4, 5)))
    for dy in range(-4, 5):
        row = []
        for dx in range(-4, 5):
            tp = 0
            for r in rows:
                if d2((r[4][0] + dx, r[4][1] + dy), r[3]) <= TOL:
                    tp += 1
            row.append(tp)
            if tp > best[0]:
                best = (tp, dx, dy)
        print(" dy%-3d" % dy, " ".join("%5d" % v for v in row))
    print("  → 最优平移 dx=%d dy=%d ⇒ tp %d（原样 %d）"
          % (best[1], best[2], best[0], st["T0"]))
    near = [r for r in rows if TOL < d2(r[4], r[3]) <= 12]
    far = [r for r in rows if d2(r[4], r[3]) > 12]
    print("  ④关超差 %d 条 = 近失(5<d≤12) %d + 远离(>12) %d"
          % (len(rows) - st["T0"], len(near), len(far)))

    # 误差方向分解：以「我方框中心 → 点」为径向，拆成 径向 / 切向
    rad = collections.Counter()
    for r in rows:
        ga, pa, box, ctr = r[3], r[4], r[5], r[7]
        if not box or not ctr:
            rad["无框"] += 1
            continue
        vx, vy = pa[0] - ctr[0], pa[1] - ctr[1]
        nv = math.hypot(vx, vy) or 1.0
        ux, uy = vx / nv, vy / nv
        ex, ey = ga[0] - pa[0], ga[1] - pa[1]
        rl = ex * ux + ey * uy                 # 正 = 真值点比我方更靠外
        tl = -ex * uy + ey * ux                # 切向
        rad["径向更外" if rl > 1.5 else ("径向更内" if rl < -1.5 else "径向基本齐")] += 1
        rad["切向>2" if abs(tl) > 2 else "切向<=2"] += 1
        if abs(rl) > 1.5:
            bin_ = "径向%" + ("%.0f" % (round(abs(rl) / 5.0) * 5))
            rad[bin_] += 1
    print("\n== 误差方向分解（真值点 − 我方点，投到我方框的径向/切向） ==")
    for k in ("径向更外", "径向更内", "径向基本齐", "切向>2", "切向<=2", "无框"):
        if rad[k]:
            print("  %-10s %5d" % (k, rad[k]))
    inner = sorted([k for k in rad if k.startswith("径向") and k[-1].isdigit()])
    if inner:
        print("  径向幅度分桶：", [(k, rad[k]) for k in inner])

    # 逐条样例
    if a.top:
        bad = [r for r in rows if d2(r[4], r[3]) > TOL]
        print("\n== ④关超差样例（case, 元件, 号, 我方点, 真值点, 距离, 我方框） ==")
        for r in bad[:a.top]:
            print("   %-8s %-6s %-4s (%.1f,%.1f) -> (%.1f,%.1f)  d=%.1f  box=%s"
                  % (r[0], r[1], r[2], r[4][0], r[4][1], r[3][0], r[3][1],
                     d2(r[4], r[3]),
                     [round(v) for v in r[5]] if r[5] else None))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
