# -*- coding: utf-8 -*-
"""§5.112 取证：**电源符号的 bbox 边框**是不是 GT 的导线端点锚点？

背景（§5.111）：160 条「只差一端」的 GT 边里，GT 那头落在**某个组件 bbox 边框上**的 48 处
（0px 24 处 / 1px 21 处），落在 GT 引脚点上的 22 处；而我方那头 98/160 落在**我方引脚点**上、
149/160 离任何 bbox 边框 ≥9px。`_linetrim.py` 把它一律裁到「所属组件的 bbox 边框」⇒ 净亏
−0.1240（因为 IC(box) 的 GT 锚点在框**外** 5~20px）。本探针只对**电源类/小符号**做这件事：

  若某个导线端点落在我方引脚点 2.5px 内，且该脚的元件 type ∈ --types（或 bbox 跨度 ≤ --maxspan），
  就把端点挪到「线段进入该 bbox 的那个交点」再沿方向进 --inset px 处。

用法（cwd = Y:\\workspace\\EDA\\eda-pcb）：
    python -X utf8 ..\\work\\_pwrborder.py --run work/_fast_out_gsd9 --insets 0,2,4,6,8
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
ap.add_argument("--types", default="gnd,v,vcc,power,earth,ground,vbus_in,net_input,net_output")
ap.add_argument("--maxspan", type=float, default=0.0,
                help=">0 时也接受 bbox 跨度（max(w,h)）不超过它的元件")
ap.add_argument("--pin_tol", type=float, default=2.5)
ap.add_argument("--insets", default="0,2,4,6,8")
ap.add_argument("--min_len", type=float, default=5.0)
args = ap.parse_args()

TYPES = {s.strip().lower() for s in args.types.split(",") if s.strip()}
INSETS = [float(s) for s in args.insets.split(",") if s.strip()]

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cfg = M.EvalConfig()


def comps(doc):
    """cid -> (bbox, type)"""
    out = {}
    for cid, comp in (doc.get('components') or {}).items():
        if not isinstance(comp, dict):
            continue
        bb = None
        for key in ('bbox', 'bbox_img', 'box', 'rect'):
            v = comp.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                try:
                    x0, y0, x1, y1 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
                except Exception:
                    continue
                bb = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                break
        if bb is None:
            continue
        t = str(comp.get('type') or '').lower()
        out[str(cid)] = (bb, t)
    return out


def pins_of(doc):
    out = []
    for comp, group in (doc.get('pins') or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, v in group.items():
            p = M._point((v or {}).get('point'))
            if p:
                out.append((str(comp), p))
    return out


def inside(p, bb):
    return bb[0] - 0.5 <= p[0] <= bb[2] + 0.5 and bb[1] - 0.5 <= p[1] <= bb[3] + 0.5


def entry_point(a, b, bb, steps=200):
    """线段 a->b 上第一个落在 bb 内的点（a 在框外时）。"""
    for i in range(steps + 1):
        t = i / float(steps)
        p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if inside(p, bb):
            return t, p
    return None, None


def mutate(pred, inset):
    """返回 (mut, n_moved, n_seen)"""
    mut = copy.deepcopy(pred)
    cmap = comps(pred)
    pins = pins_of(pred)
    n_moved = 0
    n_seen = 0
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
                for comp, p in pins:
                    if math.hypot(p[0] - end[0], p[1] - end[1]) > args.pin_tol:
                        continue
                    rec = cmap.get(comp)
                    if rec is None:
                        continue
                    bb, t = rec
                    ok = t in TYPES or (args.maxspan > 0 and
                                        max(bb[2] - bb[0], bb[3] - bb[1]) <= args.maxspan)
                    if not ok:
                        continue
                    n_seen += 1
                    tt, pt = entry_point(far, end, bb)
                    if pt is None:
                        continue
                    if inset > 0:
                        tt2 = min(1.0, tt + inset / max(L, 1e-6))
                        pt = (a[0] + tt2 * (b[0] - a[0]), a[1] + tt2 * (b[1] - a[1]))
                    # 别把边裁得太短
                    if math.hypot(pt[0] - far[0], pt[1] - far[1]) < args.min_len:
                        continue
                    target = pt
                    break
                out.append(target if target is not None else end)
            if out[0] != a or out[1] != b:
                n_moved += 1
            ev[0] = {'x': round(out[0][0], 2), 'y': round(out[0][1], 2)}
            ev[1] = {'x': round(out[1][0], 2), 'y': round(out[1][1], 2)}
    return mut, n_moved, n_seen


sel = [s.strip() for s in args.cases.split(",") if s.strip()]
sel = [c for c in sel if c in cases]
per_inset = {inset: [0.0, 0.0, 0] for inset in INSETS}   # sum line, sum total, moved
base_line = base_total = 0.0
for cid in sel:
    gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
    pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)),
                          encoding="utf-8"))
    bs = M.evaluate_case(gt, pred, cid, cfg)
    base_line += bs.net_line.f1
    base_total += bs.score
    row = []
    for inset in INSETS:
        mut, nm, ns = mutate(pred, inset)
        sc = M.evaluate_case(gt, mut, cid, cfg)
        per_inset[inset][0] += sc.net_line.f1
        per_inset[inset][1] += sc.score
        per_inset[inset][2] += nm
        row.append("inset %-4g line %.4f (%+.4f) tot %.2f (%+.2f) moved %d"
                   % (inset, sc.net_line.f1, sc.net_line.f1 - bs.net_line.f1,
                      sc.score, sc.score - bs.score, nm))
    print("case %-3s base line %.4f tot %.2f | %s" % (cid, bs.net_line.f1, bs.score,
                                                      " ; ".join(row)))
n = max(1, len(sel))
print("\n== power/small-symbol border trim (n=%d) : base line %.4f tot %.4f"
      % (len(sel), base_line / n, base_total / n))
for inset in INSETS:
    sl, st, mv = per_inset[inset]
    print("   inset %-4g  line %.4f (%+.4f)   total %.4f (%+.4f pts)   edges moved %d"
          % (inset, sl / n, (sl - base_line) / n, st / n, (st - base_total) / n, mv))
