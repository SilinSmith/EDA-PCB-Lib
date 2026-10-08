# -*- coding: utf-8 -*-
"""_wiresnapaudit.py —— 追 EDGE_SNAP 为什么没吃掉那 5~10px。

三个问题：
  Q1 `snap_edge_ends` 到底改了几个端点？（用它自己的签名判据，但只统计不改）
  Q2 它把端点吸到了哪里？（吸到的邻居边与自己什么关系、位移多少、吸完离 GT 端点还有多远）
  Q3 我们输出的端点分布 vs GT 端点：<=1/3/5/10px 各占多少？（决定候选 (a) 的天花板）
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import _edge_points  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)

W = ROOT / "work"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
RADIUS, MAXANG = 20.0, 45.0


def ang(d1, d2):
    n1 = math.hypot(*d1)
    n2 = math.hypot(*d2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = (d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pt_seg(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1]), (a[0], a[1]), 0.0
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2
    tc = max(0.0, min(1.0, t))
    foot = (a[0] + tc * vx, a[1] + tc * vy)
    return math.hypot(p[0] - foot[0], p[1] - foot[1]), foot, t


def edges_pts(doc):
    out = []
    for k, n in (doc.get("nets") or {}).items():
        for ek, e in (n.get("edges") or {}).items():
            p = _edge_points(e)
            if p is not None and len(p) >= 2:
                out.append((k, ek, (p[0][0], p[0][1]), (p[-1][0], p[-1][1])))
    return out


tot = {"ends": 0, "snappable": 0, "moved_in": 0, "toward_gt": 0, "away_gt": 0}
print("%-5s %6s %9s %9s %9s %9s" % ("case", "端点数", "可吸", "吸后更近GT", "吸后更远", "位移中位"))
per_case = []
for cid in GT9:
    gt = load_json(exact[cid].json_path)
    pr = json.loads((W / "_f220_A" / ("%s.json" % cid)).read_text(encoding="utf-8"))
    H = float((pr.get("image") or {}).get("h") or 0)
    if not H:
        H = float((json.loads((W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % cid))
                              .read_text(encoding="utf-8")).get("image") or {}).get("h") or 0)
    ge = [((p[0], H - p[1]), (q[0], H - q[1]))
          for _k, _e, p, q in edges_pts(gt)]
    gends = [pt for e in ge for pt in e]
    pe = edges_pts(pr)
    ents = [e for e in pe]
    n_snap, n_in, n_tow, n_away, moves = 0, 0, 0, 0, []
    for idx, (k, ek, a, b) in enumerate(ents):
        for which, pt in (("a", a), ("b", b)):
            other = b if which == "a" else a
            d_self = (pt[0] - other[0], pt[1] - other[1])
            best = None
            for j, (k2, ek2, a2, b2) in enumerate(ents):
                if j == idx:
                    continue
                if k2 != k:
                    continue                       # 只在同一个 net 内找邻居
                d2 = (b2[0] - a2[0], b2[1] - a2[1])
                if ang(d_self, d2) > MAXANG:
                    continue
                d, foot, t = pt_seg(pt, a2, b2)
                if d <= RADIUS and (best is None or d < best[0]):
                    best = (d, foot, t)
            if best is None:
                continue
            n_snap += 1
            d, foot, t = best
            if d <= 0.5:
                continue
            n_in += 1
            moves.append(d)
            d0 = min(math.hypot(pt[0] - ge0[0], pt[1] - ge0[1]) for ge0 in gends)
            d1 = min(math.hypot(foot[0] - ge0[0], foot[1] - ge0[1]) for ge0 in gends)
            if d1 < d0 - 0.3:
                n_tow += 1
            elif d1 > d0 + 0.3:
                n_away += 1
    moves.sort()
    med = moves[len(moves) // 2] if moves else 0.0
    print("%-5s %6d %9d %9d %9d %9.2f" % (cid, len(ents) * 2, n_snap, n_tow, n_away, med))
    tot["ends"] += len(ents) * 2
    tot["snappable"] += n_snap
    tot["moved_in"] += n_in
    tot["toward_gt"] += n_tow
    tot["away_gt"] += n_away
    per_case.append({"case": cid, "ends": len(ents) * 2, "snappable": n_snap,
                     "moved": n_in, "toward_gt": n_tow, "away_gt": n_away, "med": med})
print("合计: 端点 %d ｜ 满足吸附判据 %d（%.0f%%）｜ 真会动 %d ｜ 其中吸完更靠 GT 端点 %d / 更远 %d"
      % (tot["ends"], tot["snappable"], 100.0 * tot["snappable"] / tot["ends"],
         tot["moved_in"], tot["toward_gt"], tot["away_gt"]))

print()
print("Q3 输出端点 vs GT 端点最近距离分布（同 net 内邻居吸附的天花板）")
for cid in GT9:
    gt = load_json(exact[cid].json_path)
    pr = json.loads((W / "_f220_A" / ("%s.json" % cid)).read_text(encoding="utf-8"))
    H = float((json.loads((W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % cid))
                          .read_text(encoding="utf-8")).get("image") or {}).get("h") or 0)
    gends = [(p[0], H - p[1]) for _k, _e, p, q in edges_pts(gt)] + \
            [(q[0], H - q[1]) for _k, _e, p, q in edges_pts(gt)]
    pends = [e[2] for e in edges_pts(pr)] + [e[3] for e in edges_pts(pr)]
    b = {"1": 0, "3": 0, "5": 0, "10": 0}
    for p in pends:
        d = min(math.hypot(p[0] - g[0], p[1] - g[1]) for g in gends)
        for t in b:
            if d <= float(t):
                b[t] += 1
    n = len(pends)
    print("  %-5s 端点 %4d ｜ <=1px %3.0f%% ｜ <=3px %3.0f%% ｜ <=5px %3.0f%% ｜ <=10px %3.0f%%"
          % (cid, n, 100.0 * b["1"] / n, 100.0 * b["3"] / n, 100.0 * b["5"] / n, 100.0 * b["10"] / n))
(W / "_wiresnapaudit.json").write_text(json.dumps(per_case, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
