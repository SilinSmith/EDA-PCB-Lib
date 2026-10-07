# -*- coding: utf-8 -*-
"""量「墨迹高度 g / 标注框高度 H」的稳定性。

`crop_gray` 的裁剪高度 = 1.5 × 传给它的框高（`PAD_Y_FRAC=0.25` 两边各一份）。
训练侧传的是标注框 H，线上侧传的是 stage1 紧框 + `pad_box` 外扩。
要做**尺度归一**，就得把线上那张裁剪变回训练那张 ⇒ 需要知道 H 与墨迹高 g 的关系。

如果 g/H 集中在一个常数附近，就能用「先量 g，再按 H ≈ g/r̂」反推标注框，
从而在不知道标注框的情况下复刻训练裁剪 —— 而且**不用重训**。

    python work\\tools\\diag\\_inkband.py
    python work\\tools\\diag\\_inkband.py --show 12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from textrec import config as C                                   # noqa: E402
from textrec.dataset import _ink_band, crop_gray                  # noqa: E402


def pct(v, qs=(5, 25, 50, 75, 95)):
    if not v:
        return "-"
    a = np.asarray(v, dtype="float64")
    return " ".join("%.2f" % np.percentile(a, q) for q in qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "work" / "ocr_real" / "manifest.jsonl"))
    ap.add_argument("--split", default="")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.split:
        recs = [r for r in recs if r.get("split") == a.split]
    print("条目 %d" % len(recs))

    per = defaultdict(lambda: {"r": [], "H": [], "g": [], "none": 0})
    shown = 0
    for r in recs:
        b = r["bbox"]
        H = max(1.0, float(b[3]) - float(b[1]))
        im = Image.open(r["image"]).convert("L")
        W, Hi = im.size
        x1 = int(max(0, min(W - 1, round(b[0]))))
        y1 = int(max(0, min(Hi - 1, round(b[1]))))
        x2 = int(max(x1 + 1, min(W, round(b[2]))))
        y2 = int(max(y1 + 1, min(Hi, round(b[3]))))
        got = _ink_band(im, y1, y2, x1, x2, W, Hi)
        role = r.get("role")
        if got is None:
            per[role]["none"] += 1
            continue
        _ny1, _ny2, g = got
        per[role]["r"].append(g / H)
        per[role]["H"].append(H)
        per[role]["g"].append(g)
        if a.show and shown < a.show:
            shown += 1
            print("   %-6s %-10s H=%5.1f g=%5.1f  g/H=%.2f  %r"
                  % (r.get("case"), role, H, g, g / H, r.get("text")))

    print("\n%-12s %5s %5s   %s" % ("role", "n", "none", "g/H  p5 p25 p50 p75 p95"))
    allr = []
    for k in sorted(per):
        d = per[k]
        print("%-12s %5d %5d   %s" % (k, len(d["r"]), d["none"], pct(d["r"])))
        allr += d["r"]
    print("%-12s %5d %5s   %s" % ("ALL", len(allr), "", pct(allr)))
    print("\ng p5 p25 p50 p75 p95:", pct([g for k in per for g in per[k]["g"]]))
    print("H p5 p25 p50 p75 p95:", pct([h for k in per for h in per[k]["H"]]))


if __name__ == "__main__":
    main()
