# -*- coding: utf-8 -*-
"""_wirecover.py —— 我们的一条 GT 边被几条预测边覆盖？预测边比 GT 边短多少？

假设：官方只看**每条边的两个端点**，tolerance 5px。我们把一条 GT 导线切成了
2~4 条更短的边，切点（= 我们解出的结点）与 GT 的结点位置差 5~10px
⇒ 两侧端同时判负。那么「修端点」救不回来，**修边数（把碎边并回一条）才是正解**。
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, _edge_points, assign_components, rewrite_keys, \
    net_hypergraph_f1  # noqa: E402

exact = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    if c.json_path:
        exact.setdefault(c.case_id, c)
W = ROOT / "work"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
D19 = ["0092", "0048", "0019", "0024", "0086", "0042", "0075", "0083", "0090",
       "0057", "0062", "0032", "0028", "0175", "0198", "0105", "0106", "0036", "0120"]
cfg = EvalConfig()


def ep(doc):
    out = {}
    for nk, n in (doc.get("nets") or {}).items():
        arr = [e for e in (_edge_points(v) for v in (n.get("edges") or {}).values())
               if e is not None]
        out[nk] = arr
    return out


def L(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def near(p, seg, tol=5.0):
    a, b = seg
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return L(p, a) <= tol
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    q = (a[0] + t * vx, a[1] + t * vy)
    return L(p, q) <= tol


print("== GT 边被几条预测边覆盖（预测端点落在 GT 边上 tol=5px 内算覆盖）==")
print("%-5s %6s %8s %8s %10s %10s" % ("case", "GT边", "覆盖>=2", "覆盖=1", "GT边长中位", "预测边长中位"))
tot = [0, 0, 0, 0]
for cid in GT9 + D19:
    gt = load_json(exact[cid].json_path)
    pp = (W / "_fast_out_reg9" / "final_results" / ("%s_Pred.json" % cid)) if cid in GT9 \
        else (W / "_f220_A" / ("%s.json" % cid))
    if not pp.exists():
        continue
    pr = json.loads(pp.read_text(encoding="utf-8"))
    gt2, pred = gt, pr
    comp, km = assign_components(gt2, pred, cfg, {})
    pred2 = rewrite_keys(pred, km)
    _n, pairs = net_hypergraph_f1(gt2, pred2, cfg, {})
    ge, pe = ep(gt2), ep(pred2)
    gnames = list(ge)
    pnames = list(pe)
    n2 = n1 = 0
    gl, pl = [], []
    for i, j in pairs:
        for a, b in ge[gnames[j]]:
            gl.append(L(a, b))
            k = sum(1 for seg in pe[pnames[i]]
                    if near(a, seg) or near(b, seg))
            if k >= 2:
                n2 += 1
            else:
                n1 += 1
        for a, b in pe[pnames[i]]:
            pl.append(L(a, b))
    m = lambda v: sorted(v)[len(v) // 2] if v else 0
    print("%-5s %6d %8d %8d %10.1f %10.1f" % (cid, n2 + n1, n2, n1, m(gl), m(pl)))
    tot[0] += n2 + n1; tot[1] += n2; tot[2] += n1
    tot[3] += len(pe[pnames[i]]) if pairs else 0
print("合计: GT 边 %d ｜ 被 >=2 条预测边覆盖 %d (%.0f%%) ｜ 恰好 1 条 %d"
      % (tot[0], tot[1], 100.0 * tot[1] / max(1, tot[0]), tot[2]))
