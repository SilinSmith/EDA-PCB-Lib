# -*- coding: utf-8 -*-
"""_pinocr.py -- how good is our OCR for *box* (IC) pin numbers, and how often is
the part name (`Name` of a box component) mis-picked?

Reads a saved run directory (``work/_fast_out_<arm>``) plus the 10 official GT files.

  python -X utf8 eda-pcb/tools/dev/_pinocr.py --run work/_fast_out_w10_base

Reports (per case, and in total):
  * OCR pin-number tokens on box components: digits / letter-mixed / other
  * of the digit tokens, how many hit a GT pin number on the same row (|dy|<=8)
  * of the letter-mixed tokens, what the GT number on that row actually is
  * `Name` (part name) of each box component vs GT `Name`, and whether the text
    box sits inside the component bbox (a pin name picked up as the part name)
"""

import argparse
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
IMG_DIR = ROOT / "work" / "_fast_in"
NUM = re.compile(r"^\d+$")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def img_h(case):
    for p in (IMG_DIR / ("%s.png" % case),
              GT_DIR / str(case) / ("%s.png" % case)):
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                return im.shape[0]
    return None


def gt_index(case):
    """designator -> {"type","name","bbox","pins":{num:(x,y)} } in the output frame."""
    d = load(GT_DIR / str(case) / ("%s.json" % case))
    out = {}
    for key, c in (d.get("components") or {}).items():
        out[key] = {"type": (c.get("type") or "").lower(),
                    "name": c.get("Name"), "bbox": c.get("bbox"), "pins": {}}
    for key, ps in (d.get("pins") or {}).items():
        rec = out.setdefault(key, {"type": "", "name": None, "bbox": None, "pins": {}})
        for pk, v in (ps or {}).items():
            num = pk[4:] if str(pk).startswith("pin_") else str(pk)
            pt = v.get("point") or {}
            rec["pins"][num] = (pt.get("x"), pt.get("y"))
    return out


def gt_num_at_row(g, x, y, dy=8.0, dx=30.0):
    best, bd = None, 1e9
    for num, (px, py) in g["pins"].items():
        if px is None or py is None:
            continue
        if abs(py - y) <= dy and abs(px - x) <= dx:
            dd = abs(py - y) + 0.05 * abs(px - x)
            if dd < bd:
                best, bd = num, dd
    return best


def in_box(tb, bb):
    if not tb or not bb:
        return None
    cx = (tb[0] + tb[2]) / 2.0
    cy = (tb[1] + tb[3]) / 2.0
    return (bb[0] <= cx <= bb[2]) and (bb[1] <= cy <= bb[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = ROOT / run
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]

    T = {"digits": 0, "hit": 0, "miss": 0, "nomatch": 0,
         "mixed": 0, "mixed_fixable": 0, "other": 0,
         "box": 0, "name_ok": 0, "name_bad": 0, "name_null": 0,
         "name_inside": 0, "empty": 0}
    print("%-5s %-34s %-16s %s" % ("case", "GT part name", "our part name", "note"))
    for case in cases:
        H = img_h(case)
        s1 = load(run / "runs" / "stage1" / ("%s.stage1.json" % case))
        asg = load(run / "runs" / "stage2" / "_assigned" / ("%s.json" % case))
        gt = gt_index(case)
        st1_pins = s1.get("pins") or {}
        st1_comps = s1.get("components") or []
        if isinstance(st1_comps, dict):
            st1_comps = list(st1_comps.values())
        # cid -> bbox of stage1 component (output frame)
        cbb = {}
        for c in st1_comps:
            cbb[c.get("id")] = c.get("bbox_json") or c.get("bbox_img")
        for cid, info in (asg.get("comps") or {}).items():
            des = (info.get("designator") or {})
            key = des.get("canon") or des.get("text")
            if not key or key not in gt:
                continue
            g = gt[key]
            if g["type"] != "box":
                continue
            T["box"] += 1
            pn = (info.get("partname") or {}).get("text")
            note = []
            if not pn:
                T["name_null"] += 1
                note.append("no-partname")
            elif pn == g["name"]:
                T["name_ok"] += 1
            else:
                T["name_bad"] += 1
                note.append("NAME")
                if in_box((info.get("partname") or {}).get("box"), g["bbox"]):
                    T["name_inside"] += 1
                    note.append("inside-box")
            if a.show or note:
                print("%-5s %-34s %-16s %s" % (case, str(g["name"])[:34], str(pn)[:16],
                                               " ".join(note)))
            pts = st1_pins.get(cid) or []
            pinfo = info.get("pins") or {}
            if not pts:
                continue
            for k in range(len(pts)):
                rec = pinfo.get(k) or pinfo.get(str(k)) or {}
                tok = ((rec.get("number") or {}).get("text") or "").strip()
                px, py_img = float(pts[k][0]), float(pts[k][1])
                py = H - py_img
                if not tok:
                    T["empty"] += 1
                    continue
                if NUM.match(tok):
                    T["digits"] += 1
                    gn = gt_num_at_row(g, px, py)
                    if gn is None:
                        T["nomatch"] += 1
                    elif gn == tok:
                        T["hit"] += 1
                    else:
                        T["miss"] += 1
                        if a.show:
                            print("      k=%-3d read=%-6s GT=%-6s (x=%.0f y=%.1f)"
                                  % (k, tok, gn, px, py))
                else:
                    has_digit = any(ch.isdigit() for ch in tok)
                    T["mixed" if has_digit else "other"] += 1
                    gn = gt_num_at_row(g, px, py)
                    if has_digit:
                        T["mixed_fixable"] += 1
                    print("      %s k=%-3d read=%-6s GT=%-6s (x=%.0f y=%.1f)"
                          % ("MIX" if has_digit else "OTH", k, tok, gn, px, py))
    print()
    print("box comps          %d" % T["box"])
    print("  Name ok/bad/null %d / %d / %d   (of bad: inside-box %d)"
          % (T["name_ok"], T["name_bad"], T["name_null"], T["name_inside"]))
    print("pin-number tokens  digits %d | letter-mixed %d | other %d | empty %d"
          % (T["digits"], T["mixed"], T["other"], T["empty"]))
    print("  digit tokens     hit %d / miss %d / no-GT-row %d"
          % (T["hit"], T["miss"], T["nomatch"]))
    print("  letter-mixed     %d  (all printable above)" % T["mixed_fixable"])


if __name__ == "__main__":
    main()
