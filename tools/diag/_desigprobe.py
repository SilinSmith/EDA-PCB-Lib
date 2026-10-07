# -*- coding: utf-8 -*-
"""对单张图里**每一颗 IC/box 元件**直接问一次位号，和管线当时拿到的读数并排看。

用来回答：「这颗 IC 明明印着 U1，为什么我们输出 `pNNN` / `UB` / `U0`？」

    . work\\tools\\dev_env.ps1
    python work\\tools\\diag\\_desigprobe.py --case 0113 --pred work\\_submit_out220z\\runs\\stage2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import discover                      # noqa: E402
from vlm import client, enrich as E                             # noqa: E402
from stage2.text import imread_unicode                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pred", required=True, help="stage2 预测目录（每张 <case>.json）")
    ap.add_argument("--margin", type=float, default=0.0, help="0 = 用 enrich 的默认值")
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover()}
    c = cases.get(a.case)
    if c is None or not c.png_path:
        print("找不到 case %s 的 PNG" % a.case)
        return 1
    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    pr = json.loads((pred_dir / ("%s.json" % a.case)).read_text(encoding="utf-8"))

    bgr = imread_unicode(c.png_path)
    H = float(bgr.shape[0])
    print("PNG %s  %s" % (c.png_path, bgr.shape))
    opts = dict(E.DEFAULTS) if hasattr(E, "DEFAULTS") else {}
    if a.margin:
        opts["desig_margin"] = a.margin
    print("desig_margin = %s  model = %s" % (opts.get("desig_margin"), opts.get("model")))
    print()
    stat = {"desig": 0}

    comps = pr.get("components") or {}
    rows = []
    for key, cv in comps.items():
        ct = str(cv.get("type") or "").lower()
        if ct != "box":
            continue
        bb = cv.get("bbox")
        if not bb:
            continue
        ib = (bb[0], H - bb[3], bb[2], H - bb[1])
        got = E.read_designator(bgr, ib, opts, stat, ct)
        rows.append((key, cv, got))
    print("%-10s %-26s %-10s %s" % ("我们键", "bbox", "VLM 读", "Name"))
    for key, cv, got in sorted(rows, key=lambda r: -r[1]["bbox"][1] if r[1].get("bbox") else 0):
        print("%-10s %-26s %-10s %r"
              % (key, str(cv.get("bbox")), got or "(读不出)", cv.get("Name")))
    print()
    print("desig 请求数 = %d" % stat["desig"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
