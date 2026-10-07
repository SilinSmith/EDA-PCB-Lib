"""按 case 解剖 testpoint 的三种画法：每个 GT 附近都有哪些圆、半径多少、有没有孔。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
import schematic_parser as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
INK = 170


def contours_of(gray):
    inv = cv2.threshold(gray, INK, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, hier = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out, None
    hier = hier[0]
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p <= 0:
            continue
        circ = 4 * math.pi * a / (p * p) if p > 0 else 0
        (cx, cy), r = cv2.minEnclosingCircle(c)
        out.append({
            "i": i, "cx": float(cx), "cy": float(cy), "r": float(r),
            "area": float(a), "circ": float(circ),
            "child": int(hier[i][2]),
            "parent": int(hier[i][3]),
        })
    return out, hier


def main():
    agg = {}
    for cid in ("0050", "0011", "0102", "0012", "0056", "0131"):
        c = S.find_case(cid, S.DEFAULT_DATA_ROOT)
        if not c:
            continue
        doc = S.load_json(c.json_path)
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cs, hier = contours_of(gray)
        tps = [(k, v) for k, v in doc["components"].items()
               if str(v.get("type")) == "testpoint"]
        print(f"\n=== {cid}  {W}x{H}  testpoint {len(tps)} 个  圆轮廓 {len(cs)} 个 ===")
        rows = []
        for k, v in tps[:6]:
            x1, y1, x2, y2 = [float(t) for t in v["bbox"]]
            gcx, gcy = (x1 + x2) / 2, H - (y1 + y2) / 2
            near = [q for q in cs if math.hypot(q["cx"] - gcx, q["cy"] - gcy) <= 60
                    and q["circ"] >= 0.5 and q["area"] >= 12]
            near.sort(key=lambda q: math.hypot(q["cx"] - gcx, q["cy"] - gcy))
            desc = []
            for q in near[:4]:
                d = math.hypot(q["cx"] - gcx, q["cy"] - gcy)
                has_hole = q["child"] != -1
                hole_r = None
                if has_hole:
                    hc = cs[q["child"]]
                    hole_r = hc["r"]
                desc.append(f"d{d:.0f} r{q['r']:.0f} c{q['circ']:.2f} "
                            f"{'HOLE' if has_hole else '    '}"
                            f"{'' if hole_r is None else f'(hr{hole_r:.0f})'}")
            rows.append((k, gcx, gcy, desc))
            agg.setdefault(cid, []).append(near[:4])
        for k, gcx, gcy, desc in rows:
            print(f"  {k:<6} ({gcx:6.0f},{gcy:6.0f})  " + " | ".join(desc))

    print("\n=== 每 case 汇总（所有 testpoint 的最近 3 层圆半径）===")
    for cid, lst in agg.items():
        rs = [[round(q["r"]) for q in near[:3]] for near in lst]
        print(f"  {cid}: {rs}")


if __name__ == "__main__":
    main()
