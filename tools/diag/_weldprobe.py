# -*- coding: utf-8 -*-
"""量「方向性桥接（规则 3 `weld`）」被哪一道闸门卡住。

动机：规则 (5) 落地后 case 10 的拓扑失分（③）仍有 18 个成员，`_segap.py` 量到这些
碎片到最近段的缺口是 **12.9 ~ 61px**。但 `_weld_unions` 有三道闸门 ——
  (a) 两端点距离 <= `WIRE_WELD`（16）
  (b) `perp = dist * |sin(位移与外法线的夹角)| <= WIRE_PERP`（1.5，极紧）
  (c) 两端点的外法线要相对（`dot <= WIRE_COS` = -0.75）
  (d) 沿两端连线的墨迹比例 >= `WIRE_INK_FRAC`（0.5）
本工具对每个「碎片网」的端点，找**别的组**的最近端点，把 (a)(b)(c) 三项原始值
都打出来，并标出是哪一项先把它挡住的。

⚠ 墨迹判据 (d) **不在这里复算**（需要原图；`wire_detail` 里没有 ink），
所以「四道闸门全过」的行也可能被 (d) 挡掉。要看 (d) 得走端到端 A/B。

用法：
    python work\\tools\\diag\\_weldprobe.py --dir work\\_fast_out_off9_ov --case 10 --maxdist 70
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "eda-pcb", "stage1"))

import labels as L  # noqa: E402


def as_segs(raw):
    out = []
    for s in raw:
        arr = np.asarray(s, np.float64).reshape(-1, 2)
        if len(arr) >= 2:
            out.append(np.concatenate([arr[0], arr[-1]]))
    return np.array(out, np.float64) if out else np.zeros((0, 4), np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "_fast_out_off9_ov"))
    ap.add_argument("--case", default="10")
    ap.add_argument("--maxdist", type=float, default=70.0)
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--only", default="", help="只列「挡住原因」里含这个子串的行，如 weld")
    a = ap.parse_args()

    p = os.path.join(a.dir, "runs", "stage1", a.case + ".stage1.json")
    s1 = json.load(open(p, encoding="utf-8"))
    segs = as_segs((s1.get("wire_detail") or {}).get("segments") or [])
    n = len(segs)
    if not n:
        print("没有段", p)
        return 1

    g = L.wire_groups(segs, tol=L.WIRE_TOL, weld=L.WIRE_WELD, perp_tol=L.WIRE_PERP,
                      cos_tol=L.WIRE_COS, ext=L.WIRE_EXT, ext_perp=L.WIRE_EXT_PERP,
                      ext_cos=1.0, ext_nearest=bool(L.WIRE_EXT_NEAREST),
                      over=L.WIRE_OVER, over_cross=L.WIRE_OVER_CROSS,
                      over_cos=L.WIRE_OVER_COS)
    g = np.asarray(g)

    ends = np.stack([segs[:, :2], segs[:, 2:]], axis=1).reshape(-1, 2)
    own = np.repeat(np.arange(n), 2)
    d = segs[:, 2:4] - segs[:, 0:2]
    Ln = np.hypot(d[:, 0], d[:, 1])
    Ln = np.where(Ln < 1e-9, 1e-9, Ln)
    u = d / Ln[:, None]
    out = np.zeros_like(ends)
    out[0::2] = -u
    out[1::2] = u

    from scipy.spatial import cKDTree
    tree = cKDTree(ends)
    pairs = tree.query_pairs(float(a.maxdist), output_type='array' if False else 'ndarray')

    # ★ 决定性问题：中间那道缺口是不是**被一颗检测到的元件占着**？
    # 一个「共线、相对、相隔 28px」的缺口有两种来源，而它们必须区别对待：
    #   (1) **结点圆点**把导线切了（KiCad 画法）⇒ 应该桥接；
    #   (2) **串联元件的本体**（电阻/电容横跨在导线上）⇒ **绝不能桥接**，
    #       否则左脚的网和右脚的网被短路。
    # 纯几何分不开这两者 —— 但「缺口里有没有一颗我们的元件框」能分开。
    comps = []
    for c in (s1.get("components") or []):
        bb = c.get("bbox_img") or c.get("bbox")
        if bb and len(bb) >= 4:
            comps.append((c.get("id"), c.get("type"), np.asarray(bb[:4], np.float64)))
    comps = np.array(comps, dtype=object) if comps else np.zeros((0, 3), object)

    def comp_between(p, q, n=9):
        ts = np.linspace(0.0, 1.0, int(n))[1:-1]
        for t in ts:
            pt = p + t * (q - p)
            for cid, ctype, bb in comps:
                if bb[0] <= pt[0] <= bb[2] and bb[1] <= pt[1] <= bb[3]:
                    return "%s(%s)" % (cid, ctype)
        return ""

    rows = []
    for m, k in pairs:
        if g[own[m]] == g[own[k]]:
            continue
        v = ends[k] - ends[m]
        dist = float(np.hypot(v[0], v[1]))
        if dist <= 1e-9:
            continue
        vh = v / dist
        perp = float(abs(out[m, 0] * vh[1] - out[m, 1] * vh[0]) * dist)
        dot = float(out[m] @ out[k])
        fails = []
        if dist > L.WIRE_WELD:
            fails.append("weld")
        if perp > L.WIRE_PERP:
            fails.append("perp")
        if dot > L.WIRE_COS:
            fails.append("dot")
        rows.append((dist, perp, dot, m, k, fails, comp_between(ends[m], ends[k])))
    # 每个端点只留最近的一条
    best = {}
    for r in rows:
        for e in (r[3], r[4]):
            if e not in best or r[0] < best[e][0]:
                best[e] = r
    rows = sorted(best.values())
    if a.only:
        rows = [r for r in rows if a.only == ("+".join(r[5]) or "四闸全过(待墨迹)")]
    print("case %s：段 %d · 端点 %d · 跨组配对(<=%.0fpx) %d"
          % (a.case, n, len(ends), a.maxdist, len(rows)))
    print("现默认：WELD=%.1f PERP=%.1f COS=%.2f EXT=%.1f EXT_PERP=%.1f  TOL=%.1f OVER=%.1f"
          % (L.WIRE_WELD, L.WIRE_PERP, L.WIRE_COS, L.WIRE_EXT, L.WIRE_EXT_PERP,
             L.WIRE_TOL, L.WIRE_OVER))
    cnt = {}
    for r in rows:
        key = "+".join(r[5]) or "四闸全过(待墨迹)"
        cnt[key] = cnt.get(key, 0) + 1
    print("挡住原因分布：")
    for k in sorted(cnt, key=lambda x: -cnt[x]):
        print("   %-32s %d" % (k, cnt[k]))
    print("\n最近 %d 条（dist / perp / dot / 端点段号 / 挡住 / 缺口里的元件）:" % a.show)
    for dist, perp, dot, m, k, fails, hit in rows[:a.show]:
        print("  %6.1fpx  perp=%6.2f  dot=%6.2f  seg%-4d<->seg%-4d  %-16s %s"
              % (dist, perp, dot, own[m], own[k],
                 "+".join(fails) or "四闸全过(待墨迹)", hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
