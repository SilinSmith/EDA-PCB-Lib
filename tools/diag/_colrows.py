# -*- coding: utf-8 -*-
"""§5.115 取证：三方「引脚行（切向坐标）」对齐率 —— GT vs 我方交付 vs 队友彩色实测。

对每个 case 的每个 GT box 元件、每条框边：
  gt_rows   = GT 在该边上的引脚数（按 GT bbox 归边）
  ours_hit  = 我方交付预测点里，切向与某 GT 行相差 ≤5px 的 GT 行数
  meas_hit  = 队友 `stage1/pincolor.extract_color_pins` 的彩色实测点里，同样口径的行数
  ours_n / meas_n = 法向（框外为正）配对后的中位绝对差（越小越接近 GT 约定）
用来回答：他们的**实测行**是否比我方预测更贴 GT（行），而他们输在哪（法向）。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_colrows.py [--run sjndef] [--cases 1,2,...,10] [--detail]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, IC_TYPES, discover, normalise_doc  # noqa: E402
from schematic_parser.metrics import canon_type                                      # noqa: E402
from stage1 import pincolor                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="sjndef")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--detail", action="store_true")
ap.add_argument("--tol", type=float, default=5.0)
a = ap.parse_args()

SIDES = ("top", "bottom", "left", "right")


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get("x"), raw.get("y"))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _bbox(comp):
    for key in ("bbox_img", "bbox", "box"):
        b = (comp or {}).get(key)
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            try:
                return [float(v) for v in b[:4]]
            except (TypeError, ValueError):
                pass
    return None


def side_of(p, bb):
    x0, y0, x1, y1 = bb
    x, y = p
    cand = {
        "top": abs(y - y0), "bottom": abs(y - y1),
        "left": abs(x - x0), "right": abs(x - x1),
    }
    return min(cand, key=lambda s: cand[s])


def tangential(side, p):
    return p[0] if side in ("top", "bottom") else p[1]


def normal(side, p, bb):
    x0, y0, x1, y1 = bb
    x, y = p
    return {"top": y0 - y, "bottom": y - y1, "left": x0 - x, "right": x - x1}[side]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c

tot = {k: 0 for k in ("gt", "ours", "meas", "ourspair", "measpair")}
norm_ours, norm_meas = [], []
print("%-4s %-6s | %-18s | %-18s | %s" % ("case", "comp", "GT行/我方命中", "GT行/实测命中", "法向中位差 我方/实测"))
print("-" * 96)
for cid in a.cases.split(","):
    if cid not in cases:
        continue
    cas = cases[cid]
    gt = normalise_doc(json.load(open(str(cas.json_path), encoding="utf-8")))
    img = cv2.imread(str(cas.png_path), cv2.IMREAD_COLOR)
    H = img.shape[0]
    # GT 文档是「y 向上」，stage1 产物/彩色实测是「y 向下（图像坐标）」。
    # 比较实测列时必须把 GT 翻到图像坐标：y_img = H - y_doc，bbox 同步翻转。
    gt_img = {"components": {}, "pins": {}}
    for k, v in (gt.get("components") or {}).items():
        bb = _bbox(v)
        c2 = dict(v)
        if bb is not None:
            c2["bbox_img"] = [bb[0], H - bb[3], bb[2], H - bb[1]]
        gt_img["components"][k] = c2
    for k, pins in (gt.get("pins") or {}).items():
        gt_img["pins"][k] = {}
        for pk, info in (pins or {}).items():
            p = _pt((info or {}).get("point"))
            gt_img["pins"][k][pk] = {"point": [p[0], H - p[1]] if p else None}
    sp = ROOT / "work" / ("_fast_out_%s" % a.run) / "runs" / "stage1" / ("%s.stage1.json" % cid)
    if not sp.exists():
        print("%-4s  (no stage1 %s)" % (cid, sp.name))
        continue
    st = json.loads(sp.read_text("utf-8"))
    comps = list(st.get("components") or [])
    measured = pincolor.extract_color_pins(img, comps)
    pp = ROOT / "work" / ("_fast_out_%s" % a.run) / "final_results" / ("%s_Pred.json" % cid)
    pred = normalise_doc(json.loads(pp.read_text("utf-8"))) if pp.exists() else {"pins": {}}

    for key, gcomp in (gt.get("components") or {}).items():
        if canon_type((gcomp or {}).get("type")) not in IC_TYPES:
            continue
        gbb = _bbox(gcomp)
        if gbb is None:
            continue
        # 匹配必须用图像坐标（stage1 产物是图像坐标），否则只有纵向居中的元件能配上
        gbb_match = _bbox(gt_img["components"][key]) or gbb

        best, bi = None, None
        for c in comps:
            bb = _bbox(c)
            if bb is None:
                continue
            v = iou(gbb_match, bb)
            if best is None or v > best:
                best, bi = v, c
        if bi is None or best < 0.3:
            print("%-4s %-6s  (no stage1 match)" % (cid, key))
            continue
        ours = [p for p in (_pt((v or {}).get("point")) for v in
                            ((pred.get("pins") or {}).get(key) or {}).values()) if p]
        meas_recs = measured.get(str(bi.get("id"))) or []
        meas = [_pt(r.get("point_img")) for r in meas_recs]
        meas = [p for p in meas if p]
        tips = [p for p in (_pt(r.get("red_tip_img")) for r in meas_recs) if p]

        # 统一到图像坐标比：GT 翻转、我方交付翻转、彩色实测本来就是图像坐标。
        gbb_i = _bbox(gt_img["components"][key]) or gbb

        def _flip(pins_map):
            out = []
            for v in (pins_map or {}).values():
                p = _pt((v or {}).get("point"))
                if p:
                    out.append((p[0], H - p[1]))
            return out

        gt_pts = _flip((gt.get("pins") or {}).get(key))
        ours_pts = _flip((pred.get("pins") or {}).get(key))
        rows = {}
        for side in SIDES:
            grows = [p for p in gt_pts if side_of(p, gbb_i) == side]
            orows = [p for p in ours_pts if side_of(p, gbb_i) == side]
            mrows = [p for p in meas if side_of(p, gbb_i) == side]
            rows[side] = (sorted(tangential(side, p) for p in grows),
                          sorted(tangential(side, p) for p in orows),
                          sorted(tangential(side, p) for p in mrows))
            for gt_t in rows[side][0]:
                tot["gt"] += 1
                if any(abs(gt_t - t) <= a.tol for t in rows[side][1]):
                    tot["ours"] += 1
                if any(abs(gt_t - t) <= a.tol for t in rows[side][2]):
                    tot["meas"] += 1
            # 法向配对（切向 <=tol 配上的对，取切向最近的那个）
            for gt_p in grows:
                gtt, gtn = tangential(side, gt_p), normal(side, gt_p, gbb_i)
                for cand, acc in ((orows, norm_ours), (meas, norm_meas)):
                    hit = [p for p in cand if side_of(p, gbb_i) == side
                           and abs(tangential(side, p) - gtt) <= a.tol]
                    if hit:
                        best2 = min(hit, key=lambda p: abs(tangential(side, p) - gtt))
                        acc.append(abs(normal(side, best2, gbb_i) - gtn))
                        tot["ourspair" if cand is orows else "measpair"] += 1
            if a.detail and (rows[side][0] or rows[side][1] or rows[side][2]):
                print("      %-5s GT %s" % (side, [round(v) for v in rows[side][0]]))
                print("            ours %s" % [round(v) for v in rows[side][1]])
                print("            meas %s" % [round(v) for v in rows[side][2]])
        oh = sum(1 for side in SIDES for t in rows[side][0]
                 if any(abs(t - u) <= a.tol for u in rows[side][1]))
        mh = sum(1 for side in SIDES for t in rows[side][0]
                 if any(abs(t - u) <= a.tol for u in rows[side][2]))
        ng = sum(len(rows[s][0]) for s in SIDES)
        print("%-4s %-6s | %3d / %3d          | %3d / %3d          | 点数 GT %d 我方 %d 实测 %d" % (
            cid, key, ng, oh, ng, mh, len(gt_pts), len(ours_pts), len(meas)))


def _fmt(d):
    return "%.2f (n=%d)" % (statistics.median(d), len(d)) if d else "-"


print("-" * 96)
print("汇总：GT 行 %d｜我方命中 %d（%.1f%%）｜彩色实测命中 %d（%.1f%%）"
      % (tot["gt"], tot["ours"], 100.0 * tot["ours"] / max(1, tot["gt"]),
         tot["meas"], 100.0 * tot["meas"] / max(1, tot["gt"])))
print("法向中位绝对差：我方 %s｜彩色实测 %s（配对样本 %d / %d）"
      % (_fmt(norm_ours), _fmt(norm_meas), tot["ourspair"], tot["measpair"]))
