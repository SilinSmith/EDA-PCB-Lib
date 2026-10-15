# -*- coding: utf-8 -*-
"""_symgeom.py -- learn a per-symbol pin geometry signature from the 200 training GT
and measure its coverage / exactness on the 10 official GT cases.

signature of a component = frozenset of (pin_number, pinname, side, offset_abs)
  side        : which bbox edge the pin hangs off ('L','R','T','B')
  offset_abs  : outward distance from that edge, in pixels
  along_norm  : (position along the edge - lo) / (hi - lo), 0..1

  python -X utf8 work/_symgeom.py --mode build
  python -X utf8 work/_symgeom.py --mode cover
"""
import argparse
import collections
import json
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "200_train_cases"
GT10 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
CACHE = ROOT / "work" / "_symgeom_table.json"


def target(case_dir):
    f = list(case_dir.glob("*_target*.json"))
    return f[0] if f else None


def sig(bbox, pins):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    out = []
    for pid, p in (pins or {}).items():
        num = pid.split("_", 1)[1]
        pt = p.get("point") or {}
        x, y = float(pt.get("x", 0.0)), float(pt.get("y", 0.0))
        d = {"L": x0 - x, "R": x - x1, "T": y - y1, "B": y0 - y}
        side = min(d, key=d.get)
        off = round(d[side], 1)
        if side in ("L", "R"):
            lo, hi, pos = y0, y1, y
        else:
            lo, hi, pos = x0, x1, x
        an = (pos - lo) / (hi - lo) if hi > lo else 0.0
        out.append((num, str(p.get("pinname")), side, off, round(an, 4)))
    return out


def collect(dirs):
    per = collections.defaultdict(collections.Counter)
    for cd in dirs:
        f = target(cd)
        if not f:
            continue
        j = json.load(open(f, encoding="utf-8"))
        comps = j.get("components") or {}
        pins = j.get("pins") or {}
        for k, c in comps.items():
            ps = pins.get(k) or {}
            if not ps:
                continue
            key = (str(c.get("Name")), str(c.get("type")))
            s = tuple(sorted(sig(c.get("bbox"), ps)))
            per[key][s] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cover")
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "build":
        per = collect([d for d in sorted(TR.iterdir()) if d.is_dir()])
        out = {}
        for (nm, ty), cnt in per.items():
            variants = [{"n": n, "sig": [list(x) for x in s]}
                        for s, n in cnt.most_common(4)]
            out["%s|%s" % (nm, ty)] = {"n_variants": len(cnt),
                                      "total": sum(cnt.values()),
                                      "variants": variants}
        CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print("symbols with pins: %d ; written %s" % (len(out), CACHE))
        return 0
    tab = json.loads(CACHE.read_text(encoding="utf-8"))
    n_comp = n_cov = n_ok = 0
    n_pin = n_pin_cov = n_pin_ok = 0
    miss_names = collections.Counter()
    for case in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        j = json.load(open(GT10 / case / ("%s.json" % case), encoding="utf-8"))
        comps = j.get("components") or {}
        pins = j.get("pins") or {}
        rows = []
        for k, c in comps.items():
            ps = pins.get(k) or {}
            if not ps:
                continue
            n_comp += 1
            n_pin += len(ps)
            nm, ty = str(c.get("Name")), str(c.get("type"))
            e = tab.get("%s|%s" % (nm, ty))
            if not e:
                miss_names[nm] += 1
                continue
            n_cov += 1
            n_pin_cov += len(ps)
            want = set((str(x[0]), str(x[1])) for x in sig(c.get("bbox"), ps))
            got = None
            for v in e["variants"]:
                s = set((str(x[0]), str(x[1])) for x in v["sig"])
                if s == want:
                    got = v
                    break
            off_ok = None
            if got:
                n_ok += 1
                n_pin_ok += len(ps)
            rows.append((k, nm, ty, len(ps), bool(got)))
        bad = [r for r in rows if not r[4]]
        print("case %-2s comps(with pins) %-3d covered %-3d exact(number+name set) %-3d | miss %s"
              % (case, len(rows), sum(1 for r in rows if tab.get("%s|%s" % (r[1], r[2]))),
                 sum(1 for r in rows if r[4]), [r[1] for r in bad][:6]))
    print()
    print("components: %d ; name+type in table: %d ; same number+name set: %d"
          % (n_comp, n_cov, n_ok))
    print("pins      : %d ; in covered comps : %d ; in exact comps: %d"
          % (n_pin, n_pin_cov, n_pin_ok))
    print("frequent missing names:", miss_names.most_common(a.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
