# -*- coding: utf-8 -*-
"""_wiremeet.py —— 候选 (a)/(b) 端点吸附到**支撑直线交点**：穷举参数看有没有正收益。

对每个端点，在本 net 内找最近的非平行邻居（两支撑直线夹角在 [minang, maxang]），
把端点搬到两支撑直线的**交点**（T 型结点的真解），位移 <= maxmove。
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)
W = ROOT / "work"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
HS = {c: float((json.loads((W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % c))
                           .read_text(encoding="utf-8")).get("image") or {}).get("h") or 0)
      for c in GT9}
CFG5 = EvalConfig(line_tol=5.0)


def norm(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n > 1e-12 else (0.0, 0.0)


def pt_seg_d(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


def ang(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))))


def isect(p, d, a, b):
    e = (b[0] - a[0], b[1] - a[1])
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / den
    return (p[0] + t * d[0], p[1] + t * d[1])


def run(radius, maxmove, minang, maxang):
    tp = fp = fn = moved = 0
    for cid in GT9:
        H = HS[cid]
        doc = json.loads((W / "_f220_A" / ("%s.json" % cid)).read_text(encoding="utf-8"))
        for rec in (doc.get("nets") or {}).values():
            arr = []
            for ek, v in (rec.get("edges") or {}).items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    arr.append([ek, (float(v[0]["x"]), H - float(v[0]["y"])),
                                (float(v[-1]["x"]), H - float(v[-1]["y"]))])
            for i, (ek, a, b) in enumerate(arr):
                d0 = norm((b[0] - a[0], b[1] - a[1]))
                if d0 == (0.0, 0.0):
                    continue
                for w in (1, 2):
                    p = arr[i][w]
                    best = None
                    for j, (ek2, a2, b2) in enumerate(arr):
                        if j == i:
                            continue
                        v2 = norm((b2[0] - a2[0], b2[1] - a2[1]))
                        if v2 == (0.0, 0.0):
                            continue
                        th = ang(d0, v2)
                        if th < minang or th > maxang:
                            continue
                        if pt_seg_d(p, a2, b2) > radius:
                            continue
                        q = isect(p, d0, a2, b2)
                        if q is None:
                            continue
                        dd = math.hypot(p[0] - q[0], p[1] - q[1])
                        if dd > maxmove or (best is not None and dd >= best[0]):
                            continue
                        best = (dd, q)
                    if best is not None:
                        arr[i][w] = best[1]
                        moved += 1
            for ek, a, b in arr:
                rec["edges"][ek] = [{"x": round(a[0], 2), "y": round(H - a[1], 2)},
                                    {"x": round(b[0], 2), "y": round(H - b[1], 2)}]
        s = evaluate_case(load_json(exact[cid].json_path), doc, cid, CFG5)
        d = s.net_line.as_dict()
        tp += d["tp"]; fp += d["fp"]; fn += d["fn"]
    return moved, tp, fp, fn


print("基线（不动产物）：tp 148 fp 857 fn 405 F1 0.1900")
print("%-7s %-8s %-7s %-7s | %6s %6s %6s %6s %8s"
      % ("radius", "maxmove", "minang", "maxang", "移动", "tp", "fp", "fn", "F1"))
for radius, maxmove, minang, maxang in [
        (20, 8, 30, 90), (20, 12, 30, 90), (20, 20, 30, 90), (20, 40, 30, 90),
        (40, 12, 40, 90), (40, 20, 40, 90), (40, 40, 40, 90),
        (60, 20, 60, 90), (60, 40, 60, 90), (80, 30, 30, 90), (20, 30, 70, 89)]:
    mv, tp, fp, fn = run(radius, maxmove, minang, maxang)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    print("%-7d %-8d %-7d %-7d | %6d %6d %6d %6d %8.4f"
          % (radius, maxmove, minang, maxang, mv, tp, fp, fn, f1))
