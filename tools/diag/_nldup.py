"""_nldup.py —— `net_line` 的真凶：我们把**笔画的两条边**都当成导线输出。

`work\\tools\\diag\\_nldiag.py` 量出来的事实：官方 9 张 GT 共 553 条边、我们 996 条
（**多 80%**），而两边点距离 ≤5px 的只有 62 条。看两条相邻的预测边就明白了：

    edge_1 [(1696,386),(1633,386)]     ← 同一条横线的**下边**
    edge_2 [(1634,384),(1691,384)]     ← 同一条横线的**上边**（差 2px）

骨架化 / 描边取边的时候把导线画出来的**两条描边**各当了一条线段，而 GT 的 `edges`
是**一条中心线**（`edge_1 [(1960,520),(2051,520)]`）。`metrics.net_line_f1` 是
**一对一**指派 ⇒ 一条 GT 边最多命中一条预测边，另一条必然变 `fp`。

本脚本做**离线重打分**（不改管线）：把预测文档里每个 net 的 `edges` 按不同策略
重写一遍，再调 `evaluate_case` 看 `net_line` 怎么动。

  python work\\tools\\diag\\_nldup.py --out work\\_fast_out_novlm9
  python work\\tools\\diag\\_nldup.py --arm off,par2,par3,par5,par3col
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json           # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case           # noqa: E402
from stage2_eval import build_index, resolve                             # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

OFFICIAL = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


# ---------------------------------------------------------------- 边的基本量
def _pq(e):
    """一条边 -> (p, q)，p 在前（按 x 再按 y 排序），点数不足返回 None。

    两种写法都要吃：`[{"x","y"},{"x","y"}]`（文档里的原样）与 `((x,y),(x,y))`
    （内部已归一化的元组，`rewrite` 会传这种）。
    """
    if not isinstance(e, (list, tuple)) or len(e) < 2:
        return None
    a, b = e[0], e[1]
    if isinstance(a, (list, tuple)):
        pa = (float(a[0]), float(a[1]))
        pb = (float(b[0]), float(b[1]))
    else:
        pa = (float(a["x"]), float(a["y"]))
        pb = (float(b["x"]), float(b["y"]))
    return (pa, pb) if pa <= pb else (pb, pa)


def _kind(p, q):
    return "h" if abs(q[0] - p[0]) >= abs(q[1] - p[1]) else "v"


def _para(p, q):
    """平行轴上的位置（h 取 y，v 取 x）与该边的跨度区间。"""
    if _kind(p, q) == "h":
        return 0.5 * (p[1] + q[1]), (min(p[0], q[0]), max(p[0], q[0]))
    return 0.5 * (p[0] + q[0]), (min(p[1], q[1]), max(p[1], q[1]))


def merge_parallel(edges, off_tol=3.0, min_ov=0.35):
    """把「方向相同、平行轴相差 ≤off_tol、投影区间重叠 ≥min_ov」的边并成一条中心线。

    并出来的新边 = 跨度的并集 + 平行轴的**按长度加权平均**（两条描边的中线）。
    """
    items = []
    for e in edges:
        pq = _pq(e)
        if pq is None:
            continue
        p, q = pq
        off, span = _para(p, q)
        items.append(dict(p=p, q=q, k=_kind(p, q), off=off, a=span[0], b=span[1]))

    kept = [False] * len(items)
    groups = []
    order = sorted(range(len(items)), key=lambda i: (items[i]["k"], items[i]["off"]))
    for i in order:
        if kept[i]:
            continue
        it = items[i]
        grp = [i]
        kept[i] = True
        changed = True
        while changed:                      # 单链即可，但允许传递闭包（多描边）
            changed = False
            for j in range(len(items)):
                if kept[j] or items[j]["k"] != it["k"]:
                    continue
                ref = [items[g] for g in grp]
                if all(abs(items[j]["off"] - r["off"]) > off_tol for r in ref):
                    continue
                lo = max(min(r["a"] for r in ref), items[j]["a"])
                hi = min(max(r["b"] for r in ref), items[j]["b"])
                inter = max(0.0, hi - lo)
                base = min(min(r["b"] - r["a"] for r in ref),
                           items[j]["b"] - items[j]["a"])
                if base > 0 and inter / base < min_ov:
                    continue
                grp.append(j)
                kept[j] = True
                changed = True
        groups.append(grp)

    out = []
    for grp in groups:
        if len(grp) == 1:
            it = items[grp[0]]
            out.append([it["p"], it["q"]])
            continue
        ws = [max(1e-6, items[g]["b"] - items[g]["a"]) for g in grp]
        off = sum(items[g]["off"] * w for g, w in zip(grp, ws)) / sum(ws)
        a = min(items[g]["a"] for g in grp)
        b = max(items[g]["b"] for g in grp)
        k = items[grp[0]]["k"]
        if k == "h":
            out.append([(a, off), (b, off)])
        else:
            out.append([(off, a), (off, b)])
    return out


def weld(runs, join_tol=40.0, ang_tol=6.0):
    """共线碎边焊接（和 `stage2/assemble.py:207 merge_collinear` 同一套判据）。

    `runs` 每项 `((x,y),(x,y))`。返回同样形态的列表：端点相接（≤join_tol）
    且方向共线（夹角 ≤ang_tol）的两条并成一条，反复迭代到不动。
    """
    def ang(d1, d2):
        n1 = (d1[0] ** 2 + d1[1] ** 2) ** 0.5
        n2 = (d2[0] ** 2 + d2[1] ** 2) ** 0.5
        if n1 < 1e-9 or n2 < 1e-9:
            return 180.0
        c = (d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    out = [[r[0], r[1]] for r in runs if len(r) >= 2]
    changed = True
    while changed:
        changed = False
        n = len(out)
        for i in range(n):
            a, b = out[i]
            for k in range(i + 1, n):
                c, d = out[k]
                for (p, o, q, r2) in ((a, b, c, d), (a, b, d, c),
                                      (b, a, c, d), (b, a, d, c)):
                    if math.hypot(p[0] - q[0], p[1] - q[1]) > join_tol:
                        continue
                    if ang((p[0] - o[0], p[1] - o[1]),
                           (r2[0] - q[0], r2[1] - q[1])) > ang_tol:
                        continue
                    out[i] = [o, r2]
                    del out[k]
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return out


def _proj(p, a, b):
    """点 `p` 到线段 `ab` 的最近点（夹在线段内）与距离。"""
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return a, math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2
    t = max(0.0, min(1.0, t))
    q = (a[0] + t * vx, a[1] + t * vy)
    return q, math.hypot(p[0] - q[0], p[1] - q[1])


def snap_ends(runs, radius=12.0, max_ang=45.0):
    """把每条边的端点**吸附到同 net 里另一条边的中心线上**（T 型结的补法）。

    `_nledge.py` 的实测：我们每条边的形状都对，但端点普遍偏 **7~13px**
    （描边一侧 → 2px；骨架的分叉点没有解算到真正的交叉处 → 8~10px），
    而 `metrics._seg_distance` 的容差是 **5px** 且只比**两个端点**。所以
    只要把端点推回交叉处，一大批边就能从「差一点」变成 TP。

    `max_ang`：只吸附到**近似垂直**的那条边上（平行的邻居是重复描边，不是结点）。
    """
    out = [[r[0], r[1]] for r in runs if len(r) >= 2]
    for i, seg in enumerate(out):
        for k in (0, 1):
            p, o = seg[k], seg[1 - k]
            d0 = (o[0] - p[0], o[1] - p[1])
            n0 = math.hypot(*d0) or 1e-9
            best = None
            for j, other in enumerate(out):
                if j == i:
                    continue
                q, d = _proj(p, other[0], other[1])
                if d > radius or (best is not None and d >= best[0]):
                    continue
                d1 = (other[1][0] - other[0][0], other[1][1] - other[0][1])
                n1 = math.hypot(*d1) or 1e-9
                cos = abs((d0[0] * d1[0] + d0[1] * d1[1]) / (n0 * n1))
                if cos > math.cos(math.radians(max_ang)):
                    continue
                best = (d, q)
            if best is not None:
                seg[k] = (round(best[1][0], 2), round(best[1][1], 2))
    return out


def rewrite(doc, arm):
    """按臂名重写 `nets[*].edges`，返回 (新文档, 统计)。

    臂名语法：`off` | 任意顺序的 `par<tol>` / `col<join>` / `snap<r>` 用 `+` 连接，
    例如 `par3`、`par3+col40`、`col40`、`par3+snap12`、`par3+snap12+col16`。
    """
    if arm == "off":
        return doc, {}
    st = {"before": 0, "after": 0}
    steps = []
    for tok in arm.split("+"):
        tok = tok.strip()
        if not tok:
            continue
        num = ""
        for ch in tok:
            if ch.isdigit() or ch == ".":
                num += ch
        if tok.startswith("par"):
            steps.append(("par", float(num or 3.0)))
        elif tok.startswith("col"):
            steps.append(("col", float(num or 40.0)))
        elif tok.startswith("snap"):
            steps.append(("snap", float(num or 12.0)))

    doc = copy.deepcopy(doc)
    for net in (doc.get("nets") or {}).values():
        ed = net.get("edges") or {}
        pts = [_pq(v) for v in ed.values()]
        pts = [p for p in pts if p is not None]
        st["before"] += len(pts)
        merged = pts
        for kind, v in steps:
            if not merged:
                break
            if kind == "par":
                merged = merge_parallel([[p, q] for p, q in merged], off_tol=v)
            elif kind == "col":
                merged = weld(merged, join_tol=v)
            elif kind == "snap":
                merged = snap_ends(merged, radius=v)
        st["after"] += len(merged)
        net["edges"] = {"edge_%d" % (i + 1): [{"x": round(p[0], 2), "y": round(p[1], 2)},
                                              {"x": round(q[0], 2), "y": round(q[1], 2)}]
                        for i, (p, q) in enumerate(merged)}
    return doc, st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work\\_fast_out_novlm9")
    ap.add_argument("--cases", default="")
    ap.add_argument("--arm", default="off,par2,par3,par5")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    cases = [c for c in a.cases.split(",") if c] or OFFICIAL
    arms = [x for x in a.arm.split(",") if x]
    cfg = EvalConfig()
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    loaded = []
    for cid in cases:
        case, _how = resolve(cid, exact, digits)
        if case is None:
            case, _how = resolve(cid.lstrip("0") or "0", exact, digits)
        if case is None:
            continue
        pp = outdir / "final_results" / (case.case_id + "_Pred.json")
        if not pp.exists():
            print("! 没有预测 %s" % pp)
            continue
        loaded.append((case, load_json(Path(case.json_path)), load_json(pp)))

    print("%-10s %8s | %s" % ("arm", "均分", " ".join("%-14s" % c.case_id for c, _, _ in loaded)))
    print("-" * 110)
    base = {}
    for arm in arms:
        row, nl, tot = [], [], []
        for case, gt, pr in loaded:
            p2, _st = rewrite(pr, arm)
            sc = evaluate_case(gt, p2, case_id=case.case_id, cfg=cfg)
            nl.append(sc.net_line.f1)
            tot.append(sc.score)
            row.append(sc.net_line.f1)
        if arm == arms[0]:
            for c, v in zip(loaded, nl):
                base[c[0].case_id] = v
        print("%-10s %8.4f | %s" % (arm, statistics.mean(nl),
                                    " ".join("%-14.4f" % v for v in row)))
        print("%-10s %8.4f | 导线 F1（总分均值 %.4f）"
              % ("", sum(tot) / len(tot), sum(tot) / len(tot)))
    print("-" * 110)
    for arm in arms:
        if arm == arms[0]:
            continue
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
