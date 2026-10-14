# -*- coding: utf-8 -*-
"""§5.112 取证：把「落在符号 bbox 里」的导线端点**拉回 bbox 边框**（离线神谕/规则测）。

§5.111 证据：160 条「只差一端」的 GT 边里，GT 那头有 48 处**正好落在某个组件 bbox 边框上**
（24 处 0px、21 处 1px）；而我方那头 149/160 离任何 bbox 边框 ≥9px（多数落在我方引脚点上，
即「框外 ~12px」）。`_linetrim.py` 把**所有**这种端点裁到所属组件边框 ⇒ −0.1240（因为 IC(box)
的 GT 锚点在框**外** 5~20px，裁进去就错）。这里改成**只裁「端点已经在某个符号 bbox 内部」**的
情况（IC 的端点不会在 IC 体内），并可限制符号尺寸 `--maxspan`。

用法（cwd = Y:\\workspace\\EDA\\eda-pcb）：
    python -X utf8 ..\\work\\_bbtrim.py --run work/_fast_out_gsd9 --maxspan 60 --insets 0,2,4
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--maxspans", default="60,0")
ap.add_argument("--insets", default="0,2,4")
ap.add_argument("--near", type=float, default=2.0, help="端点距 bbox 边框 ≤ 这个值也算「在里面」")
ap.add_argument("--min_move", type=float, default=2.0)
ap.add_argument("--min_len", type=float, default=5.0)
args = ap.parse_args()

MAXSPANS = [float(s) for s in args.maxspans.split(",") if s.strip()]
INSETS = [float(s) for s in args.insets.split(",") if s.strip()]

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()


def comps(doc):
    out = []
    for cid, comp in (doc.get('components') or {}).items():
        if not isinstance(comp, dict):
            continue
        for key in ('bbox', 'bbox_img', 'box', 'rect'):
            v = comp.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                try:
                    x0, y0, x1, y1 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
                except Exception:
                    continue
                bb = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                out.append((str(cid), bb, (bb[2] - bb[0]) * (bb[3] - bb[1]),
                            str(comp.get('type') or '').lower()))
                break
    out.sort(key=lambda r: r[2])
    return out


def inside(p, bb, near):
    return (bb[0] - near <= p[0] <= bb[2] + near) and (bb[1] - near <= p[1] <= bb[3] + near)


def entry_point(a, b, bb, steps=200):
    for i in range(steps + 1):
        t = i / float(steps)
        p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if inside(p, bb, 0.5):
            return t, p
    return None, None


def mutate(pred, maxspan, inset):
    mut = copy.deepcopy(pred)
    cs = comps(pred)
    n_moved = 0
    for _, net in (mut.get('nets') or {}).items():
        if not isinstance(net, dict):
            continue
        for _, ev in (net.get('edges') or {}).items():
            if not isinstance(ev, list) or len(ev) < 2:
                continue
            a, b = M._point(ev[0]), M._point(ev[1])
            if not a or not b:
                continue
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if L < args.min_len:
                continue
            out = []
            for end, far in ((a, b), (b, a)):
                target = None
                for cid, bb, area, ctype in cs:            # 面积从小到大
                    if maxspan > 0 and max(bb[2] - bb[0], bb[3] - bb[1]) > maxspan:
                        continue
                    if not inside(end, bb, args.near):
                        continue
                    tt, pt = entry_point(far, end, bb)
                    if pt is None:
                        continue
                    if inset > 0:
                        tt2 = min(1.0, tt + inset / max(L, 1e-6))
                        pt = (a[0] + tt2 * (b[0] - a[0]), a[1] + tt2 * (b[1] - a[1]))
                    if math.hypot(pt[0] - far[0], pt[1] - far[1]) < args.min_len:
                        continue
                    if math.hypot(pt[0] - end[0], pt[1] - end[1]) < args.min_move:
                        continue
                    target = pt
                    break
                out.append(target if target is not None else end)
            if out[0] != a or out[1] != b:
                n_moved += 1
            ev[0] = {'x': round(out[0][0], 2), 'y': round(out[0][1], 2)}
            ev[1] = {'x': round(out[1][0], 2), 'y': round(out[1][1], 2)}
    return mut, n_moved


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
sel = [c for c in sel if c in cases]
acc = {(ms, ins): [0.0, 0.0, 0] for ms in MAXSPANS for ins in INSETS}
base_line = base_total = 0.0
for cid in sel:
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    bs = M.evaluate_case(gt, pred, cid, cfg)
    base_line += bs.net_line.f1
    base_total += bs.score
    parts = []
    for ms in MAXSPANS:
        for ins in INSETS:
            mut, nm = mutate(pred, ms, ins)
            sc = M.evaluate_case(gt, mut, cid, cfg)
            acc[(ms, ins)][0] += sc.net_line.f1
            acc[(ms, ins)][1] += sc.score
            acc[(ms, ins)][2] += nm
            parts.append("ms%-4g ins%-3g line %.4f(%+.4f) tot %+.2f mv%d"
                         % (ms, ins, sc.net_line.f1, sc.net_line.f1 - bs.net_line.f1,
                            sc.score - bs.score, nm))
    print("case %-3s base line %.4f tot %.2f" % (cid, bs.net_line.f1, bs.score))
    for p in parts:
        print("    " + p)
n = max(1, len(sel))
print("\n== bbox trim (n=%d) base line %.4f tot %.4f" % (len(sel), base_line / n, base_total / n))
for ms in MAXSPANS:
    for ins in INSETS:
        sl, st, mv = acc[(ms, ins)]
        print("   maxspan %-4g inset %-3g  line %.4f (%+.4f)  total %+.4f pts  moved %d"
              % (ms, ins, sl / n, (sl - base_line) / n, (st - base_total) / n, mv))
