# -*- coding: utf-8 -*-
"""_wiregeo.py —— 几何失配到底长什么样（抽样打点 + 长度分布）。

对几张代表 case，打印：
  * GT 边的「长度」分布（短桩 / 长轨）
  * 每条 GT 边到最近预测边的最短严格距离（未设阈值），按长度分桶
  * 前 8 条距离 > 5px 的 GT 边，连同它最近的预测边原样打出来
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import _edge_points, _seg_distance  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)

CASES = [("0092", ROOT / "work/_f220_A/0092.json"),
         ("0024", ROOT / "work/_f220_A/0024.json"),
         ("0032", ROOT / "work/_f220_A/0032.json"),
         ("5", ROOT / "work/_fast_out_reg9/final_results/5_Pred.json")]


def edges(d):
    out = []
    for k, n in (d.get("nets") or {}).items():
        for ek, e in (n.get("edges") or {}).items():
            p = _edge_points(e)
            if p is not None and len(p) >= 2:
                out.append((k, ek, p[0], p[-1]))
    return out


def L(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


for cid, pp in CASES:
    gt = load_json(exact[cid].json_path)
    pr = json.loads(Path(pp).read_text(encoding="utf-8"))
    ge, pe = edges(gt), edges(pr)
    if not pe:
        print("=== %s 预测 0 条边（gt %d 条）" % (cid, len(ge)))
        print("    GT 前 5 条: %s" % [ (g[2], g[3]) for g in ge[:5] ])
        continue
    print("=== %s  GT 边 %d  预测边 %d" % (cid, len(ge), len(pe)))
    # 长度分布
    gl = [L(g[2], g[3]) for g in ge]
    pl = [L(p[2], p[3]) for p in pe]
    print("   GT 边长 中位 %.1f  短(<10px) %d/%d  长(>50px) %d/%d"
          % (sorted(gl)[len(gl) // 2], sum(1 for v in gl if v < 10), len(gl),
             sum(1 for v in gl if v > 50), len(gl)))
    print("   预测边长 中位 %.1f  短(<10px) %d/%d  长(>50px) %d/%d"
          % (sorted(pl)[len(pl) // 2], sum(1 for v in pl if v < 10), len(pl),
             sum(1 for v in pl if v > 50), len(pl)))
    far = []
    for g in ge:
        best, bj = 1e18, None
        for j, p in enumerate(pe):
            d = _seg_distance([g[2], g[3]], [p[2], p[3]], True)
            if d < best:
                best, bj = d, j
        far.append((best, g, bj))
    far.sort(key=lambda t: -t[0])
    n5 = sum(1 for f in far if f[0] <= 5)
    print("   GT 边到最近预测边 <=5px 的: %d/%d (%.0f%%)" % (n5, len(ge), 100.0 * n5 / len(ge)))
    print("   最远的 6 条（GT 边 → 最近预测边 距离）:")
    for d, g, j in far[:6]:
        p = pe[j] if j is not None else None
        print("     d=%7.1f  GT %s/%s %s→%s(长%.1f)   |  pred %s/%s %s→%s(长%.1f)"
              % (d, g[0], g[1], tuple(round(v, 1) for v in g[2]),
                 tuple(round(v, 1) for v in g[3]), L(g[2], g[3]),
                 p[0] if p else "-", p[1] if p else "-",
                 tuple(round(v, 1) for v in p[2]) if p else None,
                 tuple(round(v, 1) for v in p[3]) if p else None,
                 L(p[2], p[3]) if p else 0))
