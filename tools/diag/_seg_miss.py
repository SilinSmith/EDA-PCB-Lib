# -*- coding: utf-8 -*-
"""新 textseg 漏掉了哪些真值文字框？（best IoU < 阈值 的真值框）

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_seg_miss.py [--limit 10] [--tol 0.05] [--show 40]
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
from stage2.text import imread_unicode  # noqa: E402
from _textseg_eval import D300, ROLE_OF, cases, iou_matrix  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--case", default="")
    ap.add_argument("--cfg", default="")
    args = ap.parse_args(argv)

    from cvparts import textseg
    cfg = {}
    if args.cfg:
        for kv in args.cfg.split(","):
            k, v = kv.split("=")
            if "." in v:
                cfg[k] = float(v)
            elif v.lstrip("-").isdigit():
                cfg[k] = int(v)
            elif v in ("True", "False"):
                cfg[k] = v == "True"
            else:
                cfg[k] = v

    by_role = {}
    n_gt = 0
    shown = 0
    for png, js in cases(args.limit):
        if args.case and png.parent.name != args.case:
            continue
        d = json.loads(Path(js).read_text(encoding="utf-8"))
        gts = []
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
            gts.append((b, c, role))
        if not gts:
            continue
        bgr = imread_unicode(png)
        old = [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
        new = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **cfg)]
        Mo = iou_matrix([g[0] for g in gts], old)
        Mn = iou_matrix([g[0] for g in gts], new)
        for i, (b, c, role) in enumerate(gts):
            n_gt += 1
            bi = float(Mn[i].max()) if Mn.size else 0.0
            if bi >= args.tol:
                continue
            oi = float(Mo[i].max()) if Mo.size else 0.0
            by_role[role] = by_role.get(role, 0) + 1
            if shown < args.show:
                w = b[2] - b[0]
                h = b[3] - b[1]
                print(f"  {png.parent.name:22s} {role:12s} {c!r:22s} "
                      f"wh={w:5.1f}x{h:5.1f} 新{bi:.3f} 旧{oi:.3f} "
                      f"box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")
                shown += 1
    miss = sum(by_role.values())
    print(f"\n真值框 {n_gt}  漏掉（新 best IoU < {args.tol}）{miss} = {miss/max(1,n_gt):.3f}")
    for k, v in sorted(by_role.items(), key=lambda kv: -kv[1]):
        print(f"    {k:12s} {v:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
