# -*- coding: utf-8 -*-
"""局部对照：某条竖线（x≈X）上 GT 的边切点 vs 我方边切点 vs stage1 段跨度。

用法: python -X utf8 work\_cutsx.py --case 1 --x 843 --half 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def cases_by_id():
    out = {}
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
        out[cid] = c
    return out


def _vspan(p, q, X, half):
    """竖直/近竖直段在 x≈X 处的 y 跨度；水平段返回 None。"""
    (x0, y0), (x1, y1) = p, q
    if abs(x0 - x1) > abs(y0 - y1):      # 水平为主
        if abs((y0 + y1) / 2.0 - 0) < 0 and False:
            return None
        return None
    if min(x0, x1) - half <= X <= max(x0, x1) + half:
        return (min(y0, y1), max(y0, y1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--x", type=float, default=843.0)
    ap.add_argument("--half", type=float, default=3.0)
    ap.add_argument("--run", default="work/_fast_out_leadmove")
    a = ap.parse_args()
    c = cases_by_id()[a.case]
    H = None
    import cv2
    H = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR).shape[0]

    def rows(doc, flip):
        out = []
        for nm, net in (doc.get("nets") or {}).items():
            for ek, e in (net.get("edges") or {}).items():
                pts = [(float(t["x"]), float(t["y"])) for t in e]
                p0, p1 = pts[0], pts[-1]
                if flip:
                    p0 = (p0[0], H - p0[1]); p1 = (p1[0], H - p1[1])
                sp = _vspan(p0, p1, a.x, a.half)
                if sp:
                    out.append((sp[0], sp[1], nm, ek, round(p0[0], 1), round(p1[1], 1)))
        return sorted(out)

    gt = json.loads(Path(str(c.json_path)).read_text(encoding="utf-8"))
    pr = json.loads((ROOT / a.run / "final_results" / ("%s_Pred.json" % a.case)).read_text(encoding="utf-8"))
    st = json.loads((ROOT / a.run / "runs" / "stage1" / ("%s.stage1.json" % a.case)).read_text(encoding="utf-8"))
    segs = []
    for s in ((st.get("wire_detail") or {}).get("segments") or []):
        p0 = (float(s[0][0]), float(s[0][1])); p1 = (float(s[-1][0]), float(s[-1][1]))
        sp = _vspan(p0, p1, a.x, a.half)
        if sp:
            segs.append((sp[0], sp[1], round(p0[0], 1), round(p1[0], 1)))
    gtr = rows(gt, True)
    prr = rows(pr, True)
    print("== GT 竖边（x≈%.0f）%d 条 ==" % (a.x, len(gtr)))
    for y0, y1, nm, ek, x0, x1 in gtr:
        print("   y[%7.1f,%7.1f] 长 %6.1f  %s.%s" % (y0, y1, y1 - y0, nm, ek))
    print("== 我方竖边（x≈%.0f）%d 条 ==" % (a.x, len(prr)))
    for y0, y1, nm, ek, x0, _ in prr:
        print("   y[%7.1f,%7.1f] 长 %6.1f  %s.%s" % (y0, y1, y1 - y0, nm, ek))
    print("== stage1 竖直段（x≈%.0f）%d 条 ==" % (a.x, len(segs)))
    for y0, y1, xa, xb in sorted(segs):
        print("   y[%7.1f,%7.1f] 长 %6.1f  x %.1f->%.1f" % (y0, y1, y1 - y0, xa, xb))


if __name__ == "__main__":
    main()
