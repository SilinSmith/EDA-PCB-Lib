# -*- coding: utf-8 -*-
"""§5.111 取证（三）：把某个 case / 元件的 GT 引脚点与我们的点画在同一张裁图上。

绿圈 = GT 的引脚点（`<case>.json`），红圈 = 我方 `*_Pred.json` 的点，
蓝框 = GT 框，橙框 = 我方框。用于判断「偏移是沿轴的外推距离差」还是「认错了桩」。

用法：python -X utf8 work\\_pinview.py --case 8 --comp U1 [--run gsd9] [--pad 40] [--scale 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, normalise_doc   # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, rewrite_keys  # noqa: E402


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get('x'), raw.get('y'))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return float(raw[0]), float(raw[1])


ap = argparse.ArgumentParser()
ap.add_argument("--case", default="8")
ap.add_argument("--comp", default="U1")
ap.add_argument("--run", default="gsd9")
ap.add_argument("--pad", type=float, default=40.0)
ap.add_argument("--scale", type=float, default=4.0)
ap.add_argument("--out", default="")
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
cas = cases[args.case]
gt_doc = json.load(open(str(cas.json_path), encoding="utf-8"))
pred_doc = json.load(open(str(ROOT / "work" / ("_fast_out_%s" % args.run)
                               / "final_results" / ("%s_Pred.json" % args.case)),
                          encoding="utf-8"))
gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
cfg = EvalConfig()
comp, keymap = assign_components(gt, pred, cfg, {})
pred2 = rewrite_keys(pred, keymap)
gb = (gt['components'].get(args.comp) or {}).get('bbox')
pb = (pred2['components'].get(args.comp) or {}).get('bbox')
if not gb and not pb:
    sys.exit("没有这个元件的框：%s" % args.comp)

img = cv2.imread(str(cas.png_path), cv2.IMREAD_COLOR)
H = img.shape[0]
boxes = [b for b in (gb, pb) if b]
x0 = min(b[0] for b in boxes) - args.pad
y0 = min(b[1] for b in boxes) - args.pad
x1 = max(b[2] for b in boxes) + args.pad
y1 = max(b[3] for b in boxes) + args.pad
# GT 的 y 是文档坐标（自下而上），图像坐标要翻转
print("GT box(img-y) %s   pred box %s" % (gb, pb))
gpins = gt['pins'].get(args.comp) or {}
ppins = pred2['pins'].get(args.comp) or {}
for pk, g in sorted(gpins.items()):
    a = _pt((g or {}).get('point'))
    p = ppins.get(pk)
    b = _pt((p or {}).get('point')) if isinstance(p, dict) else None
    print("  %-8s gt=%-16s pred=%-16s name=%s/%s"
          % (pk, a, b, (g or {}).get('pinname'),
             (p or {}).get('pinname') if isinstance(p, dict) else None))

x0i, y0i = int(max(0, x0)), int(max(0, y0))
x1i, y1i = int(min(img.shape[1], x1)), int(min(H, y1))
crop = img[y0i:y1i, x0i:x1i].copy()


def draw(bbox, colour, tag, dy=0.0, label=""):
    if not bbox:
        return
    bx0, by0, bx1, by1 = [float(v) for v in bbox]
    # 文档 y -> 图像 y
    p0 = (int(bx0 - x0i), int((H - by1) - y0i + dy))
    p1 = (int(bx1 - x0i), int((H - by0) - y0i + dy))
    cv2.rectangle(crop, p0, p1, colour, 1)
    if label:
        cv2.putText(crop, label, (p0[0], max(10, p0[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


draw(gb, (255, 128, 0), "gt", label="GT")
draw(pb, (0, 165, 255), "pr", label="PRED")
for pk, g in gpins.items():
    a = _pt((g or {}).get('point'))
    if a:
        cv2.circle(crop, (int(a[0] - x0i), int((H - a[1]) - y0i)), 3, (0, 220, 0), 1)
        cv2.putText(crop, str(pk).replace("pin_", ""),
                    (int(a[0] - x0i) + 3, int((H - a[1]) - y0i) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)
for pk, p in ppins.items():
    b = _pt((p or {}).get('point')) if isinstance(p, dict) else None
    if b:
        cv2.circle(crop, (int(b[0] - x0i), int((H - b[1]) - y0i)), 3, (0, 0, 255), 1)
        cv2.putText(crop, str(pk).replace("pin_", "") + "*",
                    (int(b[0] - x0i) + 3, int((H - b[1]) - y0i) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

big = cv2.resize(crop, None, fx=args.scale, fy=args.scale,
                 interpolation=cv2.INTER_NEAREST)
out = Path(args.out) if args.out else (ROOT / "work" / "_crops" / ("pinview_%s_%s.png" % (args.case, args.comp)))
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), big)
print("写出 %s  (%d x %d, scale %.1f)" % (out, big.shape[1], big.shape[0], args.scale))
