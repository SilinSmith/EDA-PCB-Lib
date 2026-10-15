"""Are the k-means 'gnd modes' really modes -- or degrees of GT box misalignment?

For every GT `gnd` box in the full dataset we measure
  ink      : fraction of dark pixels INSIDE the box
  ink25    : fraction of dark pixels inside the box grown by 25 px (leak check)
  off      : distance (px) from box centre to the ink centroid inside the grown box
  side     : box w/h in px
and cross-tabulate against the schematic source in the file name.

Read-only.  Writes nothing.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
GROW = 25
import os as _os

# ⚠ 教训：这份数据集里符号线画的是**浅灰**（GND 符号的笔画是 151/152），
# 早先用 128 当"墨迹"阈值会把 14.5% 的 gnd 框误判成空框（肉眼一看框里明明有符号）。
# 背景是纯白 255，所以取 200 才安全。
INK_THR = int(_os.environ.get("GNDBOX_INK", "200"))


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def box_px(box, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    iy1, iy2 = sorted((h - y2, h - y1))
    x1, x2 = sorted((x1, x2))
    return x1, iy1, x2, iy2


def main() -> None:
    cases = S.discover()
    print(f"cases: {len(cases)}")
    recs = []
    for c in cases:
        try:
            gray = np.asarray(Image.open(c.png_path).convert("L"))
        except Exception as e:  # noqa: BLE001
            print("skip", c.case_id, e)
            continue
        if not c.json_path:
            continue
        doc = S.load_json(c.json_path)
        comps = (doc or {}).get("components") or {}
        for key, v in comps.items():
            if str(v.get("type") or "").lower() != "gnd":
                continue
            bb = v.get("bbox")
            if not bb or len(bb) != 4:
                continue
            h, w = gray.shape
            x1, y1, x2, y2 = box_px(bb, h)
            a, b = max(0, int(y1)), min(h, int(round(y2)))
            cc, dd = max(0, int(x1)), min(w, int(round(x2)))
            sub = gray[a:b, cc:dd]
            if sub.size == 0:
                continue
            ink = float((sub < INK_THR).mean())
            ga, gb = max(0, int(y1) - GROW), min(h, int(round(y2)) + GROW)
            gc, gd = max(0, int(x1) - GROW), min(w, int(round(x2)) + GROW)
            gsub = gray[ga:gb, gc:gd]
            mask = gsub < INK_THR
            ink25 = float(mask.mean())
            if mask.any():
                ys, xs = np.nonzero(mask)
                cy = float(ys.mean()) + ga
                cx = float(xs.mean()) + gc
                off = float(np.hypot(cx - (x1 + x2) / 2.0, cy - (y1 + y2) / 2.0))
            else:
                off = float("nan")
            recs.append(dict(case=c.key, key=key, ink=ink, ink25=ink25, off=off,
                             w=float(x2 - x1), h=float(y2 - y1), src=source_of(c.png_path)))
    n = len(recs)
    print(f"gnd GT boxes measured: {n}")
    ink = np.array([r["ink"] for r in recs])
    ink25 = np.array([r["ink25"] for r in recs])
    off = np.array([r["off"] for r in recs])
    print("\n-- ink fraction inside the GT box --")
    for lo, hi in [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.10),
                   (0.10, 0.20), (0.20, 0.35), (0.35, 1.01)]:
        m = (ink >= lo) & (ink < hi)
        print(f"  [{lo:.3f},{hi:.3f})  n={m.sum():5d}  {100*m.sum()/n:5.1f}%")
    print(f"  ink<=0.005 (essentially empty): {(ink <= 0.005).sum()} ({100*(ink<=0.005).sum()/n:.1f}%)")
    print(f"  ink<=0.02 : {(ink<=0.02).sum()} ({100*(ink<=0.02).sum()/n:.1f}%)")
    print(f"  median ink {np.median(ink):.4f}   mean ink {ink.mean():.4f}")
    print("\n-- after growing the box by 25 px --")
    print(f"  empty still: {(ink25 <= 0.005).sum()}   median {np.median(ink25):.4f}")
    print("\n-- distance from box centre to ink centroid (grown box, px) --")
    for t in (3, 5, 8, 12, 20):
        print(f"  off > {t:2d}px : {(off > t).sum():5d}  {100*(off>t).sum()/n:5.1f}%")
    print(f"  median off {np.nanmedian(off):.2f}px  mean {np.nanmean(off):.2f}px")
    print("\n-- cross tab: ink bucket x source --")
    buckets = [("empty<=1%", lambda x: x <= 0.01),
               ("1-5%", lambda x: 0.01 < x <= 0.05),
               ("5-20%", lambda x: 0.05 < x <= 0.20),
               ("rich>20%", lambda x: x > 0.20)]
    srcs = Counter(r["src"] for r in recs)
    hdr = "  %-10s" % "source" + "".join("%12s" % b[0] for b in buckets) + "%8s" % "total"
    print(hdr)
    for s in sorted(srcs, key=lambda k: -srcs[k]):
        row = [r for r in recs if r["src"] == s]
        cells = []
        for _, f in buckets:
            cells.append("%12d" % sum(1 for r in row if f(r["ink"])))
        print("  %-10s" % s + "".join(cells) + "%8d" % len(row))
    print("\n-- box side lengths --")
    ws = Counter(round(r["w"]) for r in recs)
    hs = Counter(round(r["h"]) for r in recs)
    print("  top w:", ws.most_common(8))
    print("  top h:", hs.most_common(8))
    print("\n-- worst offenders (ink==0, off large) --")
    bad = sorted([r for r in recs if r["ink"] <= 0.005], key=lambda r: -(r["off"] or 0))
    for r in bad[:12]:
        print(f"  {r['case']:>28} {r['key']:<8} w={r['w']:.0f} h={r['h']:.0f} "
              f"off={r['off']:.1f} ink25={r['ink25']:.4f} src={r['src']}")


if __name__ == "__main__":
    main()
