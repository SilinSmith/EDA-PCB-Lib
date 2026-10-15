# -*- coding: utf-8 -*-
"""_symbolpin.py -- can a symbol's pin length be learned from the provided 200
cases and replayed on a new case?

GT pin points are *not* ink tips: for a `box` (IC) symbol they sit on
``bbox edge -/+ the symbol's pin length``.  That length is a property of the
symbol drawing, so it can be fitted from the 200 provided cases (which come with
GT) and looked up by part name at run time.

Scale: the eval images are rendered at different zooms, so the length is stored
relative to the pin pitch (100 mil grid) of the same symbol:

    k = offset_px / pitch_px

Modes
  --learn   scan the 200 provided GT files, write `data/pin_geom.json`
  --eval    on the 10 official GT cases, compare actual offsets with the
            prediction k * pitch (and with the flat PIN_OUT_BOX=12 baseline)
"""

import argparse
import collections
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "200_train_cases"
GT10 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
OUT = ROOT / "eda-pcb" / "data" / "pin_geom.json"


def sides(bb, pt):
    """outward distances of `pt` from each bbox edge (output frame, y up)."""
    x0, y0, x1, y1 = bb
    x, y = pt
    return {"L": x0 - x, "R": x - x1, "B": y0 - y, "T": y - y1}


def side_of(bb, pt, max_off=45.0):
    d = sides(bb, pt)
    s, v = max(d.items(), key=lambda kv: kv[1])
    return (s, v) if 0.0 <= v <= max_off else (None, v)


def pitch_of(points, side, bb, tol=6.0):
    """median gap between consecutive pins of one side (px per pin row)."""
    x0, y0, x1, y1 = bb
    if side in ("L", "R"):
        vals = sorted(p[1] for p in points)
    else:
        vals = sorted(p[0] for p in points)
    gaps = [b - a for a, b in zip(vals, vals[1:]) if 0.5 < (b - a) < 80.0]
    if not gaps:
        return None
    gaps.sort()
    return gaps[len(gaps) // 2]


def comp_pins(doc):
    """key -> (rec, {num: (x, y)})"""
    out = {}
    for k, c in (doc.get("components") or {}).items():
        out[k] = (c, {})
    for k, ps in (doc.get("pins") or {}).items():
        rec = out.setdefault(k, ({}, {}))
        for pk, v in (ps or {}).items():
            num = pk[4:] if str(pk).startswith("pin_") else str(pk)
            pt = (v or {}).get("point") or {}
            rec[1][num] = (pt.get("x"), pt.get("y"))
    return out


def learn():
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    files = glob.glob(str(TRAIN / "*" / "*_target_named.json"))
    n_comp = n_pin = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for key, (rec, pins) in comp_pins(d).items():
            if str(rec.get("type") or "").lower() != "box":
                continue
            name = str(rec.get("Name") or "")
            bb = rec.get("bbox")
            if not name or not bb or len(bb) != 4 or not pins:
                continue
            n_comp += 1
            by_side = collections.defaultdict(list)
            for num, pt in pins.items():
                if pt[0] is None:
                    continue
                s, v = side_of(bb, pt)
                if s:
                    by_side[s].append((pt, v))
            for s, lst in by_side.items():
                pts = [p for p, _ in lst]
                offs = [v for _, v in lst]
                pitch = pitch_of([p for p in pts], s, bb)
                offs.sort()
                med = offs[len(offs) // 2]
                n_pin += len(offs)
                e = {"n": len(offs), "off": round(med, 3), "pitch": pitch}
                if pitch:
                    e["k"] = round(med / pitch, 4)
                table[name][s].append(e)
    out = {}
    for name, sides_ in table.items():
        rec = {}
        for s, lst in sides_.items():
            tot = sum(e["n"] for e in lst)
            offs = sorted(e["off"] for e in lst)
            ks = sorted(e["k"] for e in lst if "k" in e)
            rec[s] = {"n": tot, "off": round(offs[len(offs) // 2], 3),
                      "k": round(ks[len(ks) // 2], 4) if ks else None,
                      "k_n": len(ks)}
        out[name] = rec
    os.makedirs(OUT.parent, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print("learned %d symbols from %d box comps / %d pins -> %s"
          % (len(out), n_comp, n_pin, OUT))
    # coverage of the 14 symbols the 10 eval cases use
    want = set()
    for c in [str(i) for i in range(1, 11)]:
        d = json.load(open(GT10 / c / (c + ".json"), encoding="utf-8"))
        for rec in (d.get("components") or {}).values():
            if str(rec.get("type") or "").lower() == "box":
                want.add(str(rec.get("Name")))
    print("eval box symbols: %d, covered by table: %d"
          % (len(want), len(want & set(out))))
    for w in sorted(want):
        print("   %-24s %s" % (w, json.dumps(out.get(w), ensure_ascii=False) if w in out else "-"))


def evaluate(table):
    tot = {"pins": 0, "flat12": 0, "table": 0, "kmiss": 0, "nok": 0}
    rows = []
    for c in [str(i) for i in range(1, 11)]:
        d = json.load(open(GT10 / c / (c + ".json"), encoding="utf-8"))
        for key, (rec, pins) in comp_pins(d).items():
            if str(rec.get("type") or "").lower() != "box":
                continue
            name = str(rec.get("Name") or "")
            bb = rec.get("bbox")
            if not bb or not pins:
                continue
            by_side = collections.defaultdict(list)
            for num, pt in pins.items():
                if pt[0] is None:
                    continue
                s, v = side_of(bb, pt)
                if s:
                    by_side[s].append((pt, v))
            for s, lst in by_side.items():
                pts = [p for p, _ in lst]
                pitch = pitch_of(pts, s, bb)
                ent = (table.get(name) or {}).get(s) or {}
                pred = None
                if ent.get("k") is not None and pitch:
                    pred = ent["k"] * pitch
                for (pt, actual) in lst:
                    tot["pins"] += 1
                    tot["flat12"] += (abs(actual - 12.0) <= 5.0)
                    if pred is None:
                        tot["nok"] += 1
                        continue
                    ok = abs(actual - pred) <= 5.0
                    tot["table"] += ok
                    if not ok:
                        tot["kmiss"] += 1
                        rows.append((c, key, name, s, round(actual, 1),
                                     round(pred, 1), round(pitch, 1) if pitch else None,
                                     ent.get("k")))
    print("box GT pins %d | flat12 within 5px: %d (%.1f%%) | table k*pitch: %d (%.1f%%)"
          % (tot["pins"], tot["flat12"], 100.0 * tot["flat12"] / max(1, tot["pins"]),
             tot["table"], 100.0 * tot["table"] / max(1, tot["pins"])))
    print("  no table entry/pitch: %d   |  table misses: %d" % (tot["nok"], tot["kmiss"]))
    print("  case  cid    symbol                   side  actual  pred  pitch  k")
    for r in sorted(rows)[:40]:
        print("  %-5s %-6s %-24s %-4s %6.1f %6.1f %6s %s"
              % (r[0], r[1], r[2][:24], r[3], r[4], r[5], r[6], r[7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.learn:
        learn()
    if a.eval:
        with open(OUT, encoding="utf-8") as fh:
            evaluate(json.load(fh))


if __name__ == "__main__":
    main()
