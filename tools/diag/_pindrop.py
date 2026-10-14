# -*- coding: utf-8 -*-
"""单个 box 元件的「引脚数丢失」取证：stage1 检测到的点 → stage2 编号 → 交付。

用于回答：GT 有 3 个脚（1/2/3），我们只交 2 个（2/3）——那个脚是**没检测到**、
还是**检测到但没有号**、还是**有号但被丢**。

用法：
    python -X utf8 work\\_pindrop.py --run gsd9 --case 6 --comp U1
    python -X utf8 work\\_pindrop.py --run gsd9 --case 8 --comp U1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import EvalConfig, canon_type, rewrite_keys, assign_components  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gsd9")
ap.add_argument("--case", default="6")
ap.add_argument("--comp", default="U1")
a = ap.parse_args()

gt = None
for c in discover(DEFAULT_DATA_ROOT):
    if str(c.case_id) == str(a.case) and c.split == "10GT":
        gt = normalise_doc(json.load(open(str(c.json_path), encoding="utf-8")))
if gt is None:
    raise SystemExit("no GT")

base = ROOT / "work" / ("_fast_out_%s" % a.run)
pred = normalise_doc(json.load(open(str(base / "final_results" / ("%s_Pred.json" % a.case)),
                                     encoding="utf-8")))
cfg = EvalConfig()
_, keymap = assign_components(gt, pred, cfg, {})
print("keymap(gt->pred 之外)：", {k: v for k, v in list(keymap.items())[:8]})
pred2 = rewrite_keys(pred, keymap)

cid = a.comp
gcomp = (gt.get("components") or {}).get(cid) or {}
gbb = gcomp.get("bbox")
print("\nGT %s type=%r bbox=%s" % (cid, gcomp.get("type"), gbb))
for pk, info in sorted((gt.get("pins") or {}).get(cid, {}).items()):
    print("   GT %-6s %s" % (pk, json.dumps(info, ensure_ascii=False)))
print("\n交付 pred %s bbox=%s" % (
    cid, ((pred2.get("components") or {}).get(cid) or {}).get("bbox")))
for pk, info in sorted((pred2.get("pins") or {}).get(cid, {}).items()):
    print("   PR %-6s %s" % (pk, json.dumps(info, ensure_ascii=False)))

# stage1 侧：按 bbox 中心最近找同一个元件
s1p = base / "runs" / "stage1" / ("%s.stage1.json" % a.case)
if not s1p.exists():
    raise SystemExit("no stage1 doc")
s1 = json.load(open(str(s1p), encoding="utf-8"))
print("\nstage1 顶层键：", sorted(s1))
comps = s1.get("components") or []
print("stage1 元件数 =", len(comps))


def center(c):
    bb = c.get("bbox_img") or c.get("bbox")
    if isinstance(bb, dict):
        return ((bb["x0"] + bb["x1"]) / 2.0, (bb["y0"] + bb["y1"]) / 2.0)
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
    return (-1e9, -1e9)


def gcenter(bb):
    if isinstance(bb, dict):
        return ((bb["x0"] + bb["x1"]) / 2.0, (bb["y0"] + bb["y1"]) / 2.0)
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


gc = gcenter(gbb)
best, bd = None, 1e18
for c in comps:
    x, y = center(c)
    d = ((x - gc[0]) ** 2 + (y - gc[1]) ** 2) ** 0.5
    if d < bd:
        best, bd = c, d
print("最近的 stage1 元件：id=%s type=%r dist=%.1f bbox_img=%s" % (
    best.get("id"), best.get("type"), bd, best.get("bbox_img")))
scid = best.get("id")

for name in ("pins", "pin_evidence", "pin_color", "pin_mono", "pincol"):
    sec = s1.get(name) or {}
    hit = sec.get(scid) if isinstance(sec, dict) else None
    if hit is None:
        continue
    print("\n--- stage1[%s][%s]（%d 项）" % (name, scid, len(hit)))
    if isinstance(hit, dict):
        for k, v in sorted(hit.items()):
            print("    %-8s %s" % (k, json.dumps(v, ensure_ascii=False)[:300]))
    else:
        for i, v in enumerate(hit):
            print("    [%d] %s" % (i, json.dumps(v, ensure_ascii=False)[:300]))
