# -*- coding: utf-8 -*-
"""0095：把 GT 框和新 textseg 框并排打出来。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _textseg_eval import ROLE_OF, cases  # noqa: E402

png = js = None
for p, j in cases(0):
    if p.parent.name == "0095":
        png, js = p, j
bgr = imread_unicode(png)
d = json.loads(Path(js).read_text(encoding="utf-8"))
gts = []
for t in (d.get("texts") or {}).values():
    loc = t.get("location") or {}
    c = (t.get("content") or "").strip()
    role = ROLE_OF.get(str(t.get("details") or ""))
    if not role or not c:
        continue
    gts.append((tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2")), c))
ob = [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
nb = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr)]
print(f"img {bgr.shape}  GT {len(gts)}  旧 {len(ob)}  新 {len(nb)}")
print("\n-- GT 前 12 --")
for b, c in gts[:12]:
    print("   ", [round(v, 1) for v in b], c)
print("\n-- 旧 前 12 --")
for b in ob[:12]:
    print("   ", [round(v, 1) for v in b])
print("\n-- 新 前 12 --")
for b in nb[:12]:
    print("   ", [round(v, 1) for v in b])
print("\n-- 新按 x 排序前 12 --")
for b in sorted(nb, key=lambda t: (t[1], t[0]))[:12]:
    print("   ", [round(v, 1) for v in b])
xs = [b[0] for b in nb]
print(f"\n新框 x1 范围 {min(xs):.0f}..{max(xs):.0f}")
gxs = [b[0][0] for b in gts]
print(f"GT 框 x1 范围 {min(gxs):.0f}..{max(gxs):.0f}")
