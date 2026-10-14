# -*- coding: utf-8 -*-
"""§5.112 取证：某个坐标附近到底有什么？—— GT/我方元件、引脚、导线端点 + 裁图。

用法（cwd = Y:\\workspace\\EDA\\eda-pcb）：
    python -X utf8 ..\\work\\_whatis.py --case 2 --x 576 --y 690 --pad 45
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="2")
ap.add_argument("--x", type=float, default=576.0)
ap.add_argument("--y", type=float, default=690.0)
ap.add_argument("--pad", type=float, default=45.0)
ap.add_argument("--run", default="work/_fast_out_gsd9")
ap.add_argument("--scale", type=int, default=7)
ap.add_argument("--out", default="")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
c = cases[args.case]
gt = normalise_doc(json.load(open(str(c.json_path), encoding="utf-8")))
pred = json.load(open(str(ROOT / args.run / "final_results" / ("%s_Pred.json" % args.case)),
                      encoding="utf-8"))
cfg = M.EvalConfig()
_, km = M.assign_components(gt, pred, cfg, {})
pred = M.rewrite_keys(pred, km)


def bbox_of(comp):
    for key in ('bbox', 'bbox_img', 'box', 'rect'):
        v = comp.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            b = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
    return None


def dist_to_bb(p, bb):
    dx = max(bb[0] - p[0], 0.0, p[0] - bb[2])
    dy = max(bb[1] - p[1], 0.0, p[1] - bb[3])
    return math.hypot(dx, dy)


q = (args.x, args.y)
for tag, doc in (("GT  ", gt), ("PRED", pred)):
    print("== %s components within %.0fpx of (%.0f,%.0f)" % (tag, args.pad + 25, q[0], q[1]))
    rows = []
    for cid, comp in (doc.get('components') or {}).items():
        bb = bbox_of(comp)
        if bb is None:
            continue
        d = dist_to_bb(q, bb)
        if d <= args.pad + 25:
            key = str(comp.get('Name') or comp.get('value') or '')
            rows.append((d, str(cid), str(comp.get('type') or ''), bb, key))
    for d, cid, t, bb, key in sorted(rows)[:8]:
        print("   d=%6.1f  %-10s type=%-6s bbox=(%.0f,%.0f,%.0f,%.0f) %s"
              % (d, cid, t, bb[0], bb[1], bb[2], bb[3], key))
    print("   pins within %.0fpx:" % (args.pad,))
    for comp, group in (doc.get('pins') or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, v in group.items():
            p = M._point((v or {}).get('point'))
            if p and math.hypot(p[0] - q[0], p[1] - q[1]) <= args.pad:
                print("      %-8s %-10s (%.1f,%.1f) d=%.1f"
                      % (comp, pk, p[0], p[1], math.hypot(p[0] - q[0], p[1] - q[1])))
    print("   edge ends within %.0fpx:" % (args.pad,))
    for nn, net in (doc.get('nets') or {}).items():
        if not isinstance(net, dict):
            continue
        for ek, ev in (net.get('edges') or {}).items():
            if not isinstance(ev, list) or len(ev) < 2:
                continue
            a, b = M._point(ev[0]), M._point(ev[1])
            if not a or not b:
                continue
            for p in (a, b):
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d <= args.pad:
                    print("      %-10s %-8s (%.1f,%.1f) d=%.1f  seg (%.0f,%.0f)->(%.0f,%.0f)"
                          % (nn, ek, p[0], p[1], d, a[0], a[1], b[0], b[1]))

im = Image.open(str(c.png_path if hasattr(c, 'png_path') else c.json_path).replace('.json', '.png')) \
    if False else None
# 直接按目录找图
img_path = Path(str(c.json_path)).parent / ("%s.png" % args.case)
im = Image.open(str(img_path))
W, H = im.size
x0 = max(0, int(args.x - args.pad)); x1 = min(W, int(args.x + args.pad))
r0 = max(0, int(H - (args.y + args.pad))); r1 = min(H, int(H - (args.y - args.pad)))
crop = im.crop((x0, r0, x1, r1)).resize(((x1 - x0) * args.scale, (r1 - r0) * args.scale),
                                        Image.NEAREST)
out = args.out or str(ROOT / "work" / "_crops" / ("whatis_%s_%d_%d.png"
                                                  % (args.case, int(args.x), int(args.y))))
Path(out).parent.mkdir(parents=True, exist_ok=True)
crop.save(out)
print("\nimage %s  size=%dx%d  crop rows %d..%d cols %d..%d  -> %s"
      % (img_path.name, W, H, r0, r1, x0, x1, out))
