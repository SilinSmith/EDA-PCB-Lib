# -*- coding: utf-8 -*-
"""_wireskel.py —— 「骨架到底有没有画出这根线」的逐 case 上限测量。

对每张 case：
  * GT net_line 实例数
  * 用 GT 边坐标（y 翻转到图像坐标）去比对 stage1 的 `wire_detail.segments`
    （图像坐标），口径 = 评测器同款「两端点 max 距离 <= tol」
    统计 tol=5 / 10 / 20 / 40 下的命中率
  * 再统计「GT 边被拆成了几段预测段」（一条 GT 边上压着几个预测段的中点）
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


def gt_edges_img(doc, H):
    """GT 边坐标是 y 向上的（stage2 写预测时做了 y=H-y）；这里转成图像坐标。"""
    out = []
    for k, n in (doc.get("nets") or {}).items():
        for ek, e in (n.get("edges") or {}).items():
            p = _edge_points(e)
            if p is None:
                continue
            out.append((k, ek, (p[0][0], H - p[0][1]), (p[-1][0], H - p[-1][1])))
    return out


def s1_segs(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = (d.get("wire_detail") or {}).get("segments") or []
    out = []
    for s in segs:
        a, b = s[0], s[1]
        out.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return d, out


def segd(p, q):
    """评测器同款：两端点 max 距离（无序）。"""
    f = max(math.hypot(p[0][0] - q[0][0], p[0][1] - q[0][1]),
            math.hypot(p[1][0] - q[1][0], p[1][1] - q[1][1]))
    r = max(math.hypot(p[0][0] - q[1][0], p[0][1] - q[1][1]),
            math.hypot(p[1][0] - q[0][0], p[1][1] - q[0][1]))
    return min(f, r)


def pt_seg_d(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


jobs = []
for c in ["1", "2", "3", "4", "5", "7", "8", "9", "10"]:
    jobs.append((c, "GT9", W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % c)))
for c in ["0092", "0048", "0019", "0024", "0086", "0042", "0075", "0083", "0090", "0057",
          "0062", "0032", "0028", "0175", "0198", "0105", "0106", "0036", "0120"]:
    jobs.append((c, "D19", W / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % c)))

print("%-6s %-4s | %4s | %s | %s" % ("case", "grp", "gtE", "命中率 tol=5/10/20/40", "GT 边被几段预测段覆盖(中位/最大)"))
rows = []
for cid, grp, sp in jobs:
    if cid not in exact or not sp.exists():
        print("skip", cid, sp.exists())
        continue
    gt = load_json(exact[cid].json_path)
    s1doc, segs = s1_segs(sp)
    H = float((s1doc.get("image") or {}).get("h") or 0)
    ge = gt_edges_img(gt, H)
    if not ge:
        continue
    hits = {"5": 0, "10": 0, "20": 0, "40": 0}
    cover = []
    for _k, _e, a, b in ge:
        best = 1e18
        for s in segs:
            d = segd((a, b), s)
            if d < best:
                best = d
            if best <= 5:
                break
        for tol in hits:
            if best <= float(tol):
                hits[tol] += 1
        # 压在这条 GT 边上的预测段数（段中点到 GT 边距离 <= 4px）
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        n = sum(1 for s in segs
                if pt_seg_d(((s[0][0] + s[1][0]) / 2.0, (s[0][1] + s[1][1]) / 2.0), a, b) <= 4.0)
        cover.append(n)
    n = len(ge)
    cover.sort()
    line = ("%-6s %-4s | %4d | %4.0f%% %4.0f%% %4.0f%% %4.0f%% | %4.1f / %d   (段数 %d, H=%.0f)"
            % (cid, grp, n, 100.0 * hits["5"] / n, 100.0 * hits["10"] / n,
               100.0 * hits["20"] / n, 100.0 * hits["40"] / n,
               cover[n // 2], cover[-1], len(segs), H))
    print(line)
    rows.append({"case": cid, "grp": grp, "gtE": n, "hit": hits, "cover_med": cover[n // 2],
                 "cover_max": cover[-1], "n_seg": len(segs), "H": H})

(W / "_wireskel_out.txt").write_text(
    "\n".join("%s" % r for r in rows), encoding="utf-8")
(W / "_wireskel.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

for grp in ("GT9", "D19"):
    sub = [r for r in rows if r["grp"] == grp]
    if sub:
        n = sum(r["gtE"] for r in sub)
        print("[%s] GT 边 %d ｜ tol5 %.1f%% / tol10 %.1f%% / tol20 %.1f%% / tol40 %.1f%%"
              % (grp, n, 100.0 * sum(r["hit"]["5"] for r in sub) / n,
                 100.0 * sum(r["hit"]["10"] for r in sub) / n,
                 100.0 * sum(r["hit"]["20"] for r in sub) / n,
                 100.0 * sum(r["hit"]["40"] for r in sub) / n))
