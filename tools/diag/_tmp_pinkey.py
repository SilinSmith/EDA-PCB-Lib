import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from schematic_parser.data import find_case, load_json   # noqa: E402

PRED = r"work\_submit_out220r21\runs\stage2"

for cid in ("0113", "0004"):
    c = find_case(cid)
    gt = load_json(c.json_path)
    pr = load_json(os.path.join(PRED, cid + ".json"))
    cs = gt["components"]
    gp = gt["pins"]
    for gk, gv in cs.items():
        if "box" not in str(gv.get("type", "")).lower():
            continue
        gpin = gp.get(gk) or {}
        if len(gpin) < 10:
            continue
        gb = [float(x) for x in gv["bbox"]]
        best = None
        for k, v in pr["components"].items():
            b = [float(x) for x in v["bbox"]]
            d = sum(abs(b[i] - gb[i]) for i in range(4))
            if best is None or d < best[0]:
                best = (d, k)
        pk = best[1]
        o = pr["pins"].get(pk) or {}
        print("== %s GT %s (Name=%s) npins=%d ; ours %s npins=%d"
              % (cid, gk, gv.get("Name"), len(gpin), pk, len(o)))
        gpts = [(float(v["point"]["y"]), k, v.get("pinname")) for k, v in gpin.items()]
        opts = [(float(v["point"]["y"]), k, v.get("pinname")) for k, v in o.items()]
        for gy, gk2, gn in sorted(gpts):
            b2 = None
            for oy, ok, on in opts:
                d = abs(oy - gy)
                if b2 is None or d < b2[0]:
                    b2 = (d, ok, on, oy)
            mark = "OK " if (b2[0] <= 5 and b2[1] == gk2) else "   "
            print("   %s GT y=%6.1f %-9s %-12s || ours y=%6.1f %-9s %-12s d=%.1f"
                  % (mark, gy, gk2, gn, b2[3], b2[1], b2[2], b2[0]))
        print()
