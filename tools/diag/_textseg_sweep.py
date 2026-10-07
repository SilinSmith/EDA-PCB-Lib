# -*- coding: utf-8 -*-
"""扫描 `cvparts.textseg` 的关键开关：墨迹阈值模式 / 字形短边下限 / 最小面积。

对每个 config 报：检出框数、真值框命中率（best IoU >= 0.05）、IoU 中位、
以及「外扩后 OCR exact」和「外扩前 OCR exact」。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_textseg_sweep.py [--limit 10] [--only A,B]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import TextEngine, imread_unicode, pad_box  # noqa: E402
from _textseg_eval import ROLE_OF, cases, iou_matrix  # noqa: E402

CONFIGS = {
    "old": None,
    "A_fix170_d3": dict(ink_mode="fixed", min_dim=3, min_area=6),
    "E_max170_d2_ln15": dict(ink_mode="max", min_dim=2, min_area=4, line_len=15),
    "F_resid6": dict(ink_mode="max", min_dim=2, min_area=4, line_len=15,
                     resid_min_h=6, resid_min_dim=1),
    "G_resid5": dict(ink_mode="max", min_dim=2, min_area=4, line_len=15,
                     resid_min_h=5, resid_min_dim=1),
    "H_resid4_a3": dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
                        resid_min_h=4, resid_min_dim=1),
}


def gt_of(js):
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    out = []
    for t in (d.get("texts") or {}).values():
        loc = t.get("location") or {}
        c = (t.get("content") or "").strip()
        role = ROLE_OF.get(str(t.get("details") or ""))
        if not role or not c or len(c) > 24:
            continue
        try:
            b = tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2"))
        except (TypeError, KeyError):
            continue
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            continue
        out.append((b, c, role))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    names = [n for n in (args.only.split(",") if args.only else CONFIGS) if n]

    eng = TextEngine()
    rr = eng.recognizer()

    ndet = {n: 0 for n in names}
    gtb = {n: [] for n in names}
    hits = {(n, s): 0 for n in names for s in ("", "_pad")}
    cnts = {(n, s): 0 for n in names for s in ("", "_pad")}
    hits[("GT_pad", "")] = 0
    cnts[("GT_pad", "")] = 0
    per_case = {n: {} for n in names}
    n_gt = 0

    for png, js in cases(args.limit):
        gts = gt_of(js)
        if not gts:
            continue
        n_gt += len(gts)
        bgr = imread_unicode(png)

        # 天花板臂：真值紧框 + 外扩，直接读
        by = {}
        for i, (_b, _c, role) in enumerate(gts):
            by.setdefault(role, []).append(i)
        for role, idx in by.items():
            boxes = [pad_box(gts[i][0]) for i in idx]
            outs = rr.read_many(bgr, [(b, role) for b in boxes])
            for j, i in enumerate(idx):
                cnts[("GT_pad", "")] += 1
                if (outs[j][0] or "").strip() == gts[i][1]:
                    hits[("GT_pad", "")] += 1

        for n in names:
            if CONFIGS[n] is None:
                det = [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
            else:
                det = [tuple(float(v) for v in b)
                       for b in textseg.detect_boxes(bgr, **CONFIGS[n])]
            ndet[n] += len(det)
            M = iou_matrix([g[0] for g in gts], det) if det else np.zeros((len(gts), 0))
            best = {}
            hit = 0
            for i, (b, c, role) in enumerate(gts):
                row = M[i] if M.size else np.zeros(0)
                j = int(np.argmax(row)) if row.size else -1
                bi = float(row[j]) if j >= 0 else 0.0
                gtb[n].append(bi)
                if bi >= 0.05:
                    hit += 1
                    best[i] = (det[j], role, c)
            per_case[n][png.parent.name] = (len(gts), hit, len(det))

            by = {}
            for i, (bx, role, _c) in best.items():
                by.setdefault(role, []).append(i)
            for s in ("", "_pad"):
                for role, idx in by.items():
                    boxes = [best[i][0] for i in idx]
                    if s == "_pad":
                        boxes = [pad_box(b) for b in boxes]
                    outs = rr.read_many(bgr, [(b, role) for b in boxes])
                    for j, i in enumerate(idx):
                        cnts[(n, s)] += 1
                        if (outs[j][0] or "").strip() == best[i][2]:
                            hits[(n, s)] += 1

    print("\n=== 检测 ===")
    print(f"  真值框 {n_gt}")
    for n in names:
        g = np.asarray(gtb[n])
        print(f"  {n:18s} 检出 {ndet[n]:5d}  命中 {int((g >= 0.05).sum()):5d}/"
              f"{g.size} = {float((g >= 0.05).mean()):.3f}  IoU p50 {np.median(g):.3f}")

    print("\n=== OCR exact ===")
    print(f"  {'GT_pad (天花板)':18s} —          "
          f"{hits[('GT_pad','')]/max(1,cnts[('GT_pad','')]):.4f} "
          f"({hits[('GT_pad','')]}/{cnts[('GT_pad','')]})  "
          f"端到端 {hits[('GT_pad','')]/max(1,n_gt):.4f}")
    for n in names:
        a = hits[(n, "")] / max(1, cnts[(n, "")])
        b = hits[(n, "_pad")] / max(1, cnts[(n, "_pad")])
        e = hits[(n, "_pad")] / max(1, n_gt)
        print(f"  {n:18s} 原样 {a:.4f} ({hits[(n,'')]}/{cnts[(n,'')]})   "
              f"外扩 {b:.4f} ({hits[(n,'_pad')]}/{cnts[(n,'_pad')]})   "
              f"端到端 {e:.4f}")

    print("\n=== 逐 case 命中/真值 ===")
    keys = sorted({k for n in names for k in per_case[n]})
    print(f"  {'case':8s} " + "  ".join(f"{n[:16]:>16s}" for n in names))
    for k in keys:
        row = "  ".join(f"{per_case[n].get(k, (0, 0, 0))[1]:6d}/"
                        f"{per_case[n].get(k, (0, 0, 0))[0]:<4d}" for n in names)
        print(f"  {k:8s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
