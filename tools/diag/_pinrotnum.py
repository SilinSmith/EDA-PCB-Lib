# -*- coding: utf-8 -*-
"""case 6/7/10 竖直脚的「号」与「名」在 4 朝向下的读数，并对上 GT 参考。
用法：python -X utf8 eda-pcb\tools\diag\_c6num.py --case 6 [--sat 70]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from stage2.text import TextEngine                      # noqa: E402
import stage2.pinname as PN                             # noqa: E402


def read_img(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def side_of(pt, bb):
    x0, y0, x1, y1 = bb
    dl, dr = abs(pt[0] - x0), abs(pt[0] - x1)
    dt, db = abs(pt[1] - y0), abs(pt[1] - y1)
    m = min(dl, dr, dt, db)
    return "L" if dl == m else "R" if dr == m else "T" if dt == m else "B"


def out_box(pt, bb, half, depth=40.0, inset=2.0):
    """朝**体外**的条带（号印在体外）。"""
    x0, y0, x1, y1 = bb
    s = side_of(pt, bb)
    if s == "B":
        return [pt[0] - half, y1 + inset, pt[0] + half, y1 + inset + depth]
    if s == "T":
        return [pt[0] - half, y0 - inset - depth, pt[0] + half, y0 - inset]
    if s == "L":
        return [x0 - inset - depth, pt[1] - half, x0 - inset, pt[1] + half]
    return [x1 + inset, pt[1] - half, x1 + inset + depth, pt[1] + half]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="6")
    ap.add_argument("--run", default=r"work\_fast_out_c6")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sat", type=float, default=70.0)
    ap.add_argument("--gt", default="")
    a = ap.parse_args()
    run = ROOT / a.run
    s1 = json.loads((run / "runs" / "stage1" / ("%s.stage1.json" % a.case)).read_text(encoding="utf-8"))
    asg = json.loads((run / "runs" / "stage2" / "_assigned" / ("%s.json" % a.case)).read_text(encoding="utf-8"))
    bgr = read_img(s1["image"]["path"])
    H = bgr.shape[0]
    eng = TextEngine(device=a.device)
    rec = eng.recognizer()
    gt = {}
    if a.gt:
        gj = json.loads(Path(a.gt).read_text(encoding="utf-8"))
        for ck, cv_ in (gj.get("components") or {}).items():
            for pk, pv in ((gj.get("pins") or {}).get(ck) or {}).items():
                p = (pv or {}).get("point") or {}
                gt[(float(p.get("x", 0)), float(p.get("y", 0)))] = (
                    str(pk), str((pv or {}).get("pinname") or ""))
    pins = s1.get("pins") or {}
    comps = {c.get("id"): c for c in (s1.get("components") or []) if isinstance(c, dict)}
    for cid, cc in (asg.get("comps") or {}).items():
        c = comps.get(cid) or {}
        if (c.get("type") or "") != "box":
            continue
        bb = c.get("bbox_img")
        if not bb:
            continue
        bb = [float(v) for v in bb]
        pts = pins.get(cid) or []
        key = ((cc.get("designator") or {}).get("text") or "").strip()
        print("== case %s %s bb=%s" % (a.case, key, [round(v) for v in bb]))
        for i, p in enumerate(pts):
            pt = (float(p[0]), float(p[1]))
            s = side_of(pt, bb)
            gtxt = ""
            for gp, (gnum, gname) in gt.items():
                if math.hypot(gp[0] - pt[0], (H - gp[1]) - pt[1]) <= 12:
                    gtxt = "GT %s/%s" % (gnum, gname)
            rec0 = ((cc.get("pins") or {}).get(str(i)) or {})
            cur = "num=%s name=%s" % (
                (((rec0.get("number") or {}) or {}).get("text") or "-"),
                (((rec0.get("name") or {}) or {}).get("text") or "-"))
            outs = []
            for tag, box in (("内", PN._strip_box(pt, bb, 14.0, depth=46.0)),
                             ("外", out_box(pt, bb, 12.0))):
                for nm, rot in (("0", None), ("R", cv2.ROTATE_90_CLOCKWISE),
                                ("L", cv2.ROTATE_90_COUNTERCLOCKWISE)):
                    x0, y0, x1, y1 = [int(round(v)) for v in box]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(bgr.shape[1], x1), min(H, y1)
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        outs.append("%s%s:空" % (tag, nm))
                        continue
                    sub = bgr[y0:y1, x0:x1]
                    img = sub if rot is None else cv2.rotate(sub, rot)
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    h, w = g.shape[:2]
                    tb = PN._tight_ink_grp(img, [0.0, 0.0, float(w), float(h)],
                                           near_y=None, pad=1.0) if a.sat >= 70 else None
                    if tb is None:
                        # 松阈值（红色号码：sat 高）
                        m = ((np.max(img, axis=2).astype(np.int16)
                              - np.min(img, axis=2).astype(np.int16)) < 160) & (g < 200)
                        m = m.astype(np.uint8) * 255
                        if int((m > 0).sum()) < 6:
                            outs.append("%s%s:无墨" % (tag, nm))
                            continue
                        tb = PN._tight_ink_grp(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR),
                                               [0.0, 0.0, float(w), float(h)], near_y=None)
                        if tb is None:
                            outs.append("%s%s:无墨" % (tag, nm))
                            continue
                    try:
                        t, cf = rec.read_many(g, [(tb, "pin_name")])[0]
                    except Exception as e:                              # noqa: BLE001
                        outs.append("%s%s:err" % (tag, nm))
                        continue
                    outs.append("%s%s:'%s'(%.2f)" % (tag, nm, (t or "").strip(), float(cf)))
            print("   pin%-3d %-2s %-16s %-22s %s"
                  % (i, s, cur, gtxt, " ".join(outs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
