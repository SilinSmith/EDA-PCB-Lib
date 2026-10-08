# -*- coding: utf-8 -*-
"""_wirecand.py —— 端点精度的三个最小候选，官方 9 张离线重算（不动管线/不改产物）。

候选定义（全部只改 `nets[*]["edges"]` 的首尾两点，官方只看这两点）：
  base     现产物原样
  snapc    复刻 `stage2/assemble.py:259-307 snap_edge_ends`（把端点投到**同 net 邻居的
           有限线段**上、只在 45° 内吸、半径 20）—— 用来验证「现管线真的在做这件事」
  snapX    同半径，但投影到邻居的**支撑直线**（不 clip）⇒ 取到的才是真交点/垂足
  meetL    把「本边的支撑直线」与「最近的非平行邻居的支撑直线」求交（T 型结点的正解），
           位移上限 MAXMOVE 像素
  clus     同 net 内所有端点按 CLUSTER 半径聚类，簇内取质心（候选 c）
每种候选都跑 line_tol = 5 / 10 / 20 的 tp/fp/fn，并报 mean score。

输出：`work\\_wirecand_out.txt`、`work\\_wirecand.json`
"""
import copy
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
HS = {}
for cid in GT9:
    d = json.loads((W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % cid))
                   .read_text(encoding="utf-8"))
    HS[cid] = float((d.get("image") or {}).get("h") or 0)


def pred_raw(cid):
    return json.loads((W / "_f220_A" / ("%s.json" % cid)).read_text(encoding="utf-8"))


def norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-12 else (0.0, 0.0)


def line_isect(p, d, a, b):
    """直线 (p,d) 与直线 (a,b) 的交点；平行返回 None。"""
    e = (b[0] - a[0], b[1] - a[1])
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / den
    return (p[0] + t * d[0], p[1] + t * d[1])


def pt_seg_d(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


def angle(u, v):
    c = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.degrees(math.acos(c))


def to_img(doc, cid):
    """把预测文档的边点从 JSON 坐标（y 向上）转成图像坐标（y 向下）就地处理。
    返回 {净名: [(ei, p0, p1), ...]} 与写回函数。"""
    H = HS[cid]
    nets = doc.get("nets") or {}
    store = {}
    for nk, rec in nets.items():
        arr = []
        for ek, v in (rec.get("edges") or {}).items():
            if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                continue
            a = (float(v[0]["x"]), H - float(v[0]["y"]))
            b = (float(v[-1]["x"]), H - float(v[-1]["y"]))
            arr.append([ek, a, b])

        def write(arr=arr, rec=rec, H=H):
            for ek, a, b in arr:
                rec["edges"][ek] = [{"x": round(a[0], 2), "y": round(H - a[1], 2)},
                                    {"x": round(b[0], 2), "y": round(H - b[1], 2)}]
        store[nk] = (arr, write)
    return store


def cand_snapc(cid, doc, radius=20.0, maxang=45.0):
    """复刻现管线：投到邻居**有限线段**上，夹角 < maxang，半径 radius。"""
    m = 0
    for nk, (arr, _w) in to_img(doc, cid).items():
        for i, (ek, a, b) in enumerate(arr):
            for which in (1, 2):
                p = arr[i][which]
                o = arr[i][2] if which == 1 else arr[i][1]
                d0 = norm((o[0] - p[0], o[1] - p[1]))
                if d0 == (0.0, 0.0):
                    continue
                best = None
                for j, (ek2, a2, b2) in enumerate(arr):
                    if j == i:
                        continue
                    v = (b2[0] - a2[0], b2[1] - a2[1])
                    if math.hypot(*v) < 1e-9:
                        q = a2
                    else:
                        t = max(0.0, min(1.0, ((p[0] - a2[0]) * v[0] + (p[1] - a2[1]) * v[1])
                                              / (v[0] ** 2 + v[1] ** 2)))
                        q = (a2[0] + t * v[0], a2[1] + t * v[1])
                    d = math.hypot(p[0] - q[0], p[1] - q[1])
                    if d > radius or (best is not None and d >= best[0]):
                        continue
                    if angle(d0, norm(v)) > maxang:
                        continue
                    best = (d, q)
                if best is not None:
                    arr[i][which] = best[1]
                    m += 1
    _w()
    return m


def cand_snapx(cid, doc, radius=20.0, maxang=45.0):
    """投影到邻居的**支撑直线**（不 clip）。"""
    m = 0
    for nk, (arr, _w) in to_img(doc, cid).items():
        for i, (ek, a, b) in enumerate(arr):
            for which in (1, 2):
                p = arr[i][which]
                o = arr[i][2] if which == 1 else arr[i][1]
                d0 = norm((o[0] - p[0], o[1] - p[1]))
                if d0 == (0.0, 0.0):
                    continue
                best = None
                for j, (ek2, a2, b2) in enumerate(arr):
                    if j == i:
                        continue
                    v = norm((b2[0] - a2[0], b2[1] - a2[1]))
                    if v == (0.0, 0.0) or angle(d0, v) > maxang:
                        continue
                    # 点到支撑直线的垂足
                    t = (p[0] - a2[0]) * v[0] + (p[1] - a2[1]) * v[1]
                    q = (a2[0] + t * v[0], a2[1] + t * v[1])
                    d = math.hypot(p[0] - q[0], p[1] - q[1])
                    if d > radius or (best is not None and d >= best[0]):
                        continue
                    best = (d, q)
                if best is not None:
                    arr[i][which] = best[1]
                    m += 1
    _w()
    return m


def cand_meetl(cid, doc, radius=40.0, maxang=60.0, maxmove=15.0):
    """本边支撑直线 × 最近非平行邻居支撑直线 = T 型真交点。"""
    m = 0
    for nk, (arr, _w) in to_img(doc, cid).items():
        for i, (ek, a, b) in enumerate(arr):
            d0 = norm((b[0] - a[0], b[1] - a[1]))
            if d0 == (0.0, 0.0):
                continue
            for which in (1, 2):
                p = arr[i][which]
                best = None
                for j, (ek2, a2, b2) in enumerate(arr):
                    if j == i:
                        continue
                    if angle(d0, norm((b2[0] - a2[0], b2[1] - a2[1]))) < maxang:
                        continue                      # 共线/近平行不参与求交
                    if pt_seg_d(p, a2, b2) > radius:
                        continue
                    q = line_isect(p, d0, a2, b2)
                    if q is None:
                        continue
                    d = math.hypot(p[0] - q[0], p[1] - q[1])
                    if d > maxmove or (best is not None and d >= best[0]):
                        continue
                    best = (d, q)
                if best is not None:
                    arr[i][which] = best[1]
                    m += 1
    _w()
    return m


def cand_clus(cid, doc, radius=6.0):
    """同 net 内端点聚类取质心（只动 >=2 个点的簇）。"""
    m = 0
    for nk, (arr, _w) in to_img(doc, cid).items():
        pts = []
        for i, (ek, a, b) in enumerate(arr):
            pts.append((i, 1, a))
            pts.append((i, 2, b))
        used = [False] * len(pts)
        for x in range(len(pts)):
            if used[x]:
                continue
            lab = [x]
            used[x] = True
            for y in range(x + 1, len(pts)):
                if used[y]:
                    continue
                if math.hypot(pts[x][2][0] - pts[y][2][0], pts[x][2][1] - pts[y][2][1]) <= radius:
                    used[y] = True
                    lab.append(y)
            if len(lab) < 2:
                continue
            cx = sum(pts[t][2][0] for t in lab) / len(lab)
            cy = sum(pts[t][2][1] for t in lab) / len(lab)
            for t in lab:
                arr[pts[t][0]][pts[t][1]] = (cx, cy)
                m += 1
    _w()
    return m


CANDS = [("base", None), ("snapc", cand_snapc), ("snapx", cand_snapx),
         ("meetL", cand_meetl), ("clus", cand_clus)]
TOLS = [5.0, 10.0, 20.0]

lines = []
print("%-6s %-22s %s" % ("候选", "每张改动端点数", "line_tol=5 / 10 / 20 的 tp/fp/fn 与 mean"))
for name, fn in CANDS:
    row = {"cand": name, "per": {}, "by_tol": {}}
    docs = {}
    for cid in GT9:
        d = pred_raw(cid)
        if fn is not None:
            row["per"][cid] = fn(cid, d)
        docs[cid] = d
    out = []
    for tol in TOLS:
        cfg = EvalConfig(line_tol=tol)
        tp = fp = fn_ = 0
        sc = 0.0
        for cid in GT9:
            s = evaluate_case(load_json(exact[cid].json_path), docs[cid], cid, cfg)
            d = s.net_line.as_dict()
            tp += d["tp"]; fp += d["fp"]; fn_ += d["fn"]; sc += s.score
        f1 = 2 * tp / max(1, 2 * tp + fp + fn_)
        row["by_tol"][str(tol)] = {"tp": tp, "fp": fp, "fn": fn_, "f1": round(f1, 4),
                                   "score": round(sc / len(GT9), 2)}
        out.append("tol%-3.0f tp%-4d fp%-4d fn%-4d F1 %.4f (score %.2f)" % (tol, tp, fp, fn_, f1, sc / 9))
    lines.append("%-6s %-22s\n     %s" % (name, sum(row["per"].values()) if row["per"] else "-",
                                          "\n     ".join(out)))
    print(lines[-1])
(W / "_wirecand_out.txt").write_text("\n".join(lines), encoding="utf-8")
print()
print("写盘：work\\_wirecand_out.txt")
