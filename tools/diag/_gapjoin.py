# -*- coding: utf-8 -*-
"""离线验证「同网相邻共线边的缺口处理」，三种做法分别打分：

  snap   把两侧相向端点都挪到会合点（中点）—— 让它们首尾相接，不动边的条数；
  fill   在缺口处**新增**一条边，正好覆盖缺口（最贴近 GT「在交叉处切开、首尾相接」的结构）；
  merge  把两条边合成一条（跨越缺口到两端）—— §5.90 `_edgejoin` 的大范围合并是负的，
         这里只测**极紧**的闸门（缺口 ≤4px、横向 ≤1.5px）时是否变正。

背景：case 1 x≈843 的 GT 竖线是 13 条 29~30px、首尾相接的边；我方只 9 条，缺的是
[115,144]/[173,203]/[380,410]/[438,468] 四段（缺口 27~30px），根因是骨架在交叉处被
`thin_stroke_mask` 打洞（交叉处墨迹粗 ⇒ dt 大 ⇒ 被当粗线剔除）后断成 5~10px 的碎片、
再被 MIN_EDGE_LEN/extent 过滤丢掉。

用法:
  python -X utf8 work\\_gapjoin.py --mode fill  --g 35 --lat 2 --ang 12 --out gf35
  python -X utf8 work\\_gapjoin.py --mode snap  --g 6  --lat 1.5 --ang 12 --out gs6
  python -X utf8 work\\_gapjoin.py --mode merge --g 4  --lat 1.5 --ang 8  --out gm4
  python -X utf8 eda-pcb\\tools\\eval\\fast.py --score-only --cases "1,2,3,4,5,6,7,8,9,10" --out gf35
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def edges_of(net):
    """net["edges"] -> [(key, (x0,y0), (x1,y1), raw0, raw1)]，取首末两点。"""
    e = net.get("edges")
    out = []
    if isinstance(e, dict):
        items = list(e.items())
    elif isinstance(e, list):
        items = list(enumerate(e))
    else:
        return out
    for k, v in items:
        if not v or len(v) < 2:
            continue
        p0 = (float(v[0]["x"]), float(v[0]["y"]))
        p1 = (float(v[-1]["x"]), float(v[-1]["y"]))
        out.append((str(k), p0, p1, v[0], v[-1]))
    return out


def _unit(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    if L <= 1e-9:
        return None, 0.0
    return (dx / L, dy / L), L


def pair_gap(a, b, ang_max):
    """两条边的沿轴缺口信息；不可比时 None。返回 (gap, perp, iA, iB, mid)。"""
    ua, La = _unit(a[0], a[1])
    ub, Lb = _unit(b[0], b[1])
    if ua is None or ub is None:
        return None
    if abs(ua[0] * ub[1] - ua[1] * ub[0]) > math.sin(math.radians(ang_max)):
        return None
    pa = [a[0][0] * ua[0] + a[0][1] * ua[1], a[1][0] * ua[0] + a[1][1] * ua[1]]
    pb = [b[0][0] * ua[0] + b[0][1] * ua[1], b[1][0] * ua[0] + b[1][1] * ua[1]]
    a_lo, a_hi = min(pa), max(pa)
    b_lo, b_hi = min(pb), max(pb)
    if a_hi <= b_lo:
        iA = 0 if pa[0] >= pa[1] else 1
        iB = 0 if pb[0] <= pb[1] else 1
        gap = b_lo - a_hi
    elif b_hi <= a_lo:
        iA = 0 if pa[0] <= pa[1] else 1
        iB = 0 if pb[0] >= pb[1] else 1
        gap = a_lo - b_hi
    else:
        return None
    if gap < 0.5:
        return None
    qa, qb = a[iA], b[iB]
    v = (qb[0] - qa[0], qb[1] - qa[1])
    perp = abs(v[0] * (-ua[1]) + v[1] * ua[0])
    mid = ((qa[0] + qb[0]) / 2.0, (qa[1] + qb[1]) / 2.0)
    return (gap, perp, iA, iB, mid)


def _dot(p, q):
    return [p[0] * q[0] + p[1] * q[1]]


def _keynum(k):
    try:
        return int(str(k).rsplit("_", 1)[-1])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out_leadmove"))
    ap.add_argument("--out", default="gap")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--mode", default="fill", choices=["fill", "snap", "merge", "report"])
    ap.add_argument("--g", type=float, default=35.0)
    ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=2.0)
    ap.add_argument("--ang", type=float, default=12.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    src = Path(a.pred)
    dst = ROOT / "work" / ("_fast_out_" + a.out)
    if a.mode != "report":
        (dst / "final_results").mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)

    want = set(x for x in a.cases.split(",") if x)
    gh = Counter()
    nh = Counter()
    tot = Counter()
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or "")
        if cid and want and cid not in want:
            continue
        pp = src / "final_results" / (cid + "_Pred.json")
        if not pp.exists():
            continue
        doc = json.loads(pp.read_text(encoding="utf-8"))
        nets = doc.get("nets") or {}
        for nm in sorted(nets):
            net = nets[nm]
            es = edges_of(net)
            cand = []
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    r = pair_gap((es[i][1], es[i][2]), (es[j][1], es[j][2]), a.ang)
                    if r is None:
                        continue
                    gap, perp, iA, iB, mid = r
                    tot["cand"] += 1
                    if a.mode == "report":
                        gh[min(int(round(gap)), 60)] += 1
                        nh[min(int(round(perp)), 12)] += 1
                        continue
                    if perp > a.lat or gap > a.g or gap < a.gmin:
                        continue
                    cand.append((gap, es[i][0], iA, es[j][0], iB, mid))
            if a.mode == "report":
                continue
            cand.sort(key=lambda t: t[0])
            used = set()

            def _raw(key):
                v = net["edges"].get(key)
                return v if (isinstance(v, list) and len(v) >= 2) else None

            for gap, ka, iA, kb, iB, mid in cand:
                va, vb = _raw(ka), _raw(kb)
                if va is None or vb is None:
                    continue
                if a.mode in ("snap", "merge") and ((ka, iA) in used or (kb, iB) in used):
                    continue
                if a.mode == "snap":
                    va[iA]["x"], va[iA]["y"] = round(mid[0], 2), round(mid[1], 2)
                    vb[iB]["x"], vb[iB]["y"] = round(mid[0], 2), round(mid[1], 2)
                    used.add((ka, iA)); used.add((kb, iB))
                elif a.mode == "fill":
                    qa, qb = va[iA], vb[iB]
                    mx = max([_keynum(k) for k in net["edges"]] + [0])
                    net["edges"]["edge_%d" % (mx + 1)] = [
                        {"x": round(float(qa["x"]), 2), "y": round(float(qa["y"]), 2)},
                        {"x": round(float(qb["x"]), 2), "y": round(float(qb["y"]), 2)}]
                elif a.mode == "merge":
                    fa, fb = va[1 - iA], vb[1 - iB]
                    net["edges"][ka] = [
                        {"x": round(float(fa["x"]), 2), "y": round(float(fa["y"]), 2)},
                        {"x": round(float(fb["x"]), 2), "y": round(float(fb["y"]), 2)}]
                    net["edges"].pop(kb, None)
                    used.add((ka, iA)); used.add((kb, iB))
                tot[a.mode] += 1
                if not a.quiet and tot[a.mode] <= 10:
                    print("   case %-3s %s %s+%s gap=%5.1f mid=(%.1f,%.1f)"
                          % (cid, nm, ka, kb, gap, mid[0], mid[1]))
        if a.mode != "report":
            (dst / "final_results" / pp.name).write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    if a.mode == "report":
        print("[gapjoin] 报告 pred=%s ang<=%.0f" % (src, a.ang))
        print("  沿轴缺口直方图: %s"
              % ", ".join("%d:%d" % (k, gh[k]) for k in sorted(gh)))
        print("  横向偏移直方图: %s" % ", ".join("%d:%d" % (k, nh[k]) for k in sorted(nh)))
        print("  候选 %d" % tot["cand"])
        return 0
    print("[gapjoin] mode=%s g=[%.1f,%.1f] lat=%.1f ang=%.0f -> %s"
          % (a.mode, a.gmin, a.g, a.lat, a.ang, dst))
    print("[gapjoin] 处理 %d 处（候选 %d）" % (tot[a.mode], tot["cand"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
