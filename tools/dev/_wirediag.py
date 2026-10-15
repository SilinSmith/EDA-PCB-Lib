"""官方 wire 通道（net_line）失分归因：GT 边为什么配不上。

在「官方尺」下，wire 只在**connectivity 配上的网对**里做方向无关的一对一边匹配，
代价 = 两端点最大距离，阈值 5px。这里把每个 GT 边的最优代价拆成桶：

    tp            <=5px（就是官方记的 tp）
    just_over     5~8px   ← 差一点点
    off_8_20      8~20px  ← 几何偏差
    off_20         >20px  ← 完全不是同一条
    net_unmatched 它所在的网没被配上（再多 tp 也拿不到）
    no_pred_edge  配上的网里 pred 一条边都没有

用法（在 Y:\\workspace\\EDA 下）：

    python -X utf8 eda-pcb\\tools\\dev\\_wirediag.py --run work\\_fast_out_dropunk
    python -X utf8 eda-pcb\\tools\\dev\\_wirediag.py --run work\\_fast_out_dropunk --show 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import official as OFF  # noqa: E402

GT_ROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
THRESHOLD = 5.0
BUCKETS = ("tp", "just_over", "off_8_20", "off_20", "net_unmatched", "no_pred_edge")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def bucket(c: float) -> str:
    if c <= THRESHOLD:
        return "tp"
    if c <= 8.0:
        return "just_over"
    if c <= 20.0:
        return "off_8_20"
    return "off_20"


def end_pattern(g, p):
    """把 GT 边 g 与 pred 边 p 的两端距离（方向无关）排序后归类。

    one_end_ok  = 一端 <=2px、另一端 >5px  ⇒ 线**短了一截**（端点被削掉）
    both_mid    = 两端都在 2~5px         ⇒ 整条线**平移**了几像素
    one_end_far = 一端 <=2px、另一端 >12px ⇒ 一条线只覆盖了一半，另半截是别的边
    both_far    = 两端都 >5px
    """
    def sw(x, y):
        return max(math.hypot(x[0][0] - y[0][0], x[0][1] - y[0][1]),
                   math.hypot(x[1][0] - y[1][0], x[1][1] - y[1][1]))
    a = sw(g, p)
    b = sw(g, (p[1], p[0]))
    if a <= b:
        d1 = math.hypot(g[0][0] - p[0][0], g[0][1] - p[0][1])
        d2 = math.hypot(g[1][0] - p[1][0], g[1][1] - p[1][1])
    else:
        d1 = math.hypot(g[0][0] - p[1][0], g[0][1] - p[1][1])
        d2 = math.hypot(g[1][0] - p[0][0], g[1][1] - p[0][1])
    lo, hi = min(d1, d2), max(d1, d2)
    if hi <= 5.0:
        return "tp"
    if lo <= 2.0 and hi <= 12.0:
        return "one_end_ok"
    if lo <= 2.0:
        return "one_end_far"
    if lo <= 5.0:
        return "both_mid"
    return "both_far"


def _pt_seg(d, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return math.hypot(d[0] - ax, d[1] - ay)
    t = ((d[0] - ax) * dx + (d[1] - ay) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(d[0] - (ax + t * dx), d[1] - (ay + t * dy))


def _cover(edge, segs, tol=3.0, step=2.0):
    """GT 边（或 pred 边）被 segs 覆盖的比例（沿线采样）。"""
    (x0, y0), (x1, y1) = edge
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(L / step) + 1)
    hit = 0
    for i in range(n + 1):
        t = i / float(n)
        d = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        if any(_pt_seg(d, a, b) <= tol for a, b in segs):
            hit += 1
    return hit / float(n + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work\\_fast_out_dropunk")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--show", type=int, default=0, help="每案打印 N 条差一点点的例子")
    ap.add_argument("--ends", action="store_true", help="按两端距离模式归类（诊断端点削短）")
    ap.add_argument("--cover", action="store_true",
                    help="GT 边被 pred 边覆盖的比例（判断「几何对不对 / 只是切分不同」）")
    ap.add_argument("--ctol", type=float, default=3.0, help="覆盖判定的点到线段容差")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    total: Counter = Counter()
    ep: Counter = Counter()
    cover_stat: list = []
    n_gt_net = n_pred_net = 0
    for cid in cases:
        gt_p = GT_ROOT / cid / ("%s.json" % cid)
        pred_p = ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)
        if not (gt_p.exists() and pred_p.exists()):
            print("skip %s" % cid)
            continue
        gt, pred = load(gt_p), load(pred_p)
        _conn, pairs = OFF.connectivity_f1(gt, pred, collapse_nonbox=False)
        ge, pe = OFF._edges_of(gt), OFF._edges_of(pred)
        n_gt_net += len(ge)
        n_pred_net += len(pe)
        pmap = dict(pairs)
        local: Counter = Counter()
        examples = []
        for gid, gsegs in ge.items():
            pid = pmap.get(gid)
            if pid is None:
                local["net_unmatched"] += len(gsegs)
                total["net_unmatched"] += len(gsegs)
                continue
            psegs = pe.get(pid) or []
            if not psegs:
                local["no_pred_edge"] += len(gsegs)
                total["no_pred_edge"] += len(gsegs)
                continue
            for g in gsegs:
                costs = [OFF._edge_cost(g, p) for p in psegs]
                j = min(range(len(costs)), key=lambda k: costs[k])
                b = bucket(costs[j])
                local[b] += 1
                total[b] += 1
                ep[end_pattern(g, psegs[j])] += 1
                if b != "tp" and args.show and len(examples) < args.show:
                    p = psegs[j]
                    examples.append((round(costs[j], 1), g, p))
        n_gt = sum(len(v) for v in ge.values())
        n_pred = sum(len(v) for v in pe.values())
        sc = OFF.evaluate_case(gt, pred, cid, collapse_nonbox=False)
        if args.cover:
            gset = [s for v in ge.values() for s in v]
            pset = [s for v in pe.values() for s in v]
            gc = [_cover(e, pset, tol=args.ctol) for e in gset]
            pc = [_cover(e, gset, tol=args.ctol) for e in pset]
            full = sum(1 for c in gc if c >= 0.95)
            dead = sum(1 for c in gc if c < 0.5)
            pjunk = sum(1 for c in pc if c < 0.5)
            print("case %-3s GT|coverage %.3f 全覆(>=.95) %d/%d 半覆以下 %d | "
                  "pred 覆盖(反向) %.3f 无对应(<.5) %d/%d"
                  % (cid, sum(gc) / max(1, len(gc)), full, len(gc), dead,
                     sum(pc) / max(1, len(pc)), pjunk, len(pc)))
            cover_stat.append((sum(gc) / max(1, len(gc)), full, len(gc), dead,
                               sum(pc) / max(1, len(pc)), pjunk, len(pc)))
        print("case %-3s gt_edge %3d pred_edge %3d | tp %3d just5-8 %3d off8-20 %3d off>20 %3d "
              "net_unmatched %3d | wireF1 %.4f (tp %d fp %d fn %d)"
              % (cid, n_gt, n_pred, local["tp"], local["just_over"], local["off_8_20"],
                 local["off_20"], local["net_unmatched"], sc.wire.f1,
                 sc.wire.tp, sc.wire.fp, sc.wire.fn))
        for c, g, p in examples:
            print("      %.1fpx  GT (%.0f,%.0f)-(%.0f,%.0f)  →  P (%.0f,%.0f)-(%.0f,%.0f)"
                  % (c, g[0][0], g[0][1], g[1][0], g[1][1],
                     p[0][0], p[0][1], p[1][0], p[1][1]))
    print()
    print("== 汇总（%d 案，GT 网 %d / pred 网 %d）==" % (len(cases), n_gt_net, n_pred_net))
    tot = sum(total.values())
    for b in BUCKETS:
        v = total[b]
        print("  %-14s %5d  %5.1f%%" % (b, v, 100.0 * v / max(1, tot)))
    print("  GT 边合计 %d" % tot)
    if args.ends:
        print()
        print("== 端点模式（GT 边 → 该网对里最优 pred 边的两端距离）==")
        for k in ("tp", "one_end_ok", "both_mid", "one_end_far", "both_far"):
            v = ep[k]
            print("  %-12s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, sum(ep.values()))))
    if cover_stat:
        n = len(cover_stat)
        mg = sum(r[0] for r in cover_stat) / n
        mp = sum(r[4] for r in cover_stat) / n
        fg = sum(r[1] for r in cover_stat)
        ng = sum(r[2] for r in cover_stat)
        dg = sum(r[3] for r in cover_stat)
        jp = sum(r[5] for r in cover_stat)
        npd = sum(r[6] for r in cover_stat)
        print()
        print("== 覆盖（容差 %.1fpx）==" % args.ctol)
        print("  GT 边平均被覆盖率 %.3f ｜ 全覆(>=95%%) %d/%d ｜ 半覆以下 %d"
              % (mg, fg, ng, dg))
        print("  pred 边平均覆盖 GT %.3f ｜ 无对应(<50%%) %d/%d"
              % (mp, jp, npd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
