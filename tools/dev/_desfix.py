# -*- coding: utf-8 -*-
"""_desfix.py -- how many missing GT component keys are *one confusion away* from a
text we actually read?

Motivation: case 7's LED1 is detected (`led` 0.771, 5px off GT) but stage2 gives it
`designator: null` and `partname.text = "7ED1"` -- the designator text was OCR'd with
`L -> 7`, so it fails the designator gate and the component is dropped (DROP_UNKEYED).

  python -X utf8 work/_desfix.py --run leadV --cases 1,2,3,4,5,6,7,8,9,10
"""
import argparse
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
DIG2LET = {"0": "OQD", "1": "IL", "2": "Z", "3": "E", "4": "A", "5": "SB",
           "6": "G", "7": "LTF", "8": "B", "9": "Gg"}
LET2DIG = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "E": "3",
           "A": "4", "S": "5", "B": "8", "G": "6", "T": "7", "F": "7", "g": "9"}
DES = re.compile(r"^[A-Za-z]{1,4}\d{1,4}$")


def repairs(t):
    """yield candidate designators one confusion away from text t."""
    t = (t or "").strip().replace(" ", "")
    if not t or len(t) > 7:
        return
    out = []
    for i, ch in enumerate(t):
        if ch.isdigit():
            for let in DIG2LET.get(ch, ""):
                out.append(t[:i] + let + t[i + 1:])
        elif ch in LET2DIG:
            out.append(t[:i] + LET2DIG[ch] + t[i + 1:])
    for c in out:
        if DES.match(c):
            yield c


def texts_of(comp):
    out = []
    for fld in ("designator", "value", "partname"):
        v = comp.get(fld)
        if isinstance(v, dict) and v.get("text"):
            out.append(str(v["text"]))
    for v in (comp.get("pins") or {}).values():
        for f2 in ("number", "name"):
            q = (v or {}).get(f2)
            if isinstance(q, dict) and q.get("text"):
                out.append(str(q["text"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="leadV")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()
    run = ROOT / "work" / ("_fast_out_%s" % a.run)
    n_miss = n_fix = 0
    rows = []
    for case in a.cases.split(","):
        gt = json.load(open(GT_DIR / case / ("%s.json" % case), encoding="utf-8"))
        pr = json.load(open(run / "final_results" / ("%s_Pred.json" % case),
                           encoding="utf-8"))
        gc = gt.get("components") or {}
        pc = pr.get("components") or {}
        miss = [k for k in gc if k not in pc and not k.startswith("\u2205")]
        asg = json.load(open(run / "runs" / "stage2" / "_assigned" / ("%s.json" % case),
                            encoding="utf-8"))
        pool = {}
        for cid, comp in (asg.get("comps") or {}).items():
            for t in texts_of(comp):
                pool.setdefault(t, cid)
        for k in miss:
            n_miss += 1
            hit = None
            for t, cid in pool.items():
                if k in set(repairs(t)):
                    hit = (t, cid)
                    break
            if hit:
                n_fix += 1
            rows.append((case, k, str(gc[k].get("type")), hit))
    print("missing GT keys (non-\u2205): %d ; one-confusion-away from a read text: %d"
          % (n_miss, n_fix))
    for r in rows[:a.show]:
        print("   %-4s %-8s %-10s %s" % (r[0], r[1], r[2], r[3] if r[3] else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
