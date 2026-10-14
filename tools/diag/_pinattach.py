# -*- coding: utf-8 -*-
"""Pin -> net attachment rules, scored offline (no VLM, seconds).

Why (see RESULTS 5.105): a 2-pin symbol gets its pin points *stamped* at
bbox +/- 20px (STAGE2_PIN_STAMP_SPAN), and the drawn wire can stop ~15..20px
short of that point -- the schematic still means "connected".  Our pipeline
joins a pin to a net only when the geometry touches, so such pins end up in
**no net at all** and their GT net loses one member.

This probe attaches those pins offline (a pure `hyperGraph` edit -- component /
pin / net_line channels must stay bit-identical) and re-scores with
`tools/eval/fast.py --score-only`.

Modes
  touch   rebuild every hyperGraph from M.parse_hypergraph (round-trip self test)
  orphan  append refs for pins that no net references
  move    also relocate a pin whose nearest net is not the one it sits in

Gates
  --r         nearest wire-segment distance (px)
  --deg       max angle between the pin's outward axis and the wire direction
              (999 = off)
  --only-stamp  only 2-pin stamped symbols (r/c/l/fuse/crystal_2pin/d/led)

Usage
  python -X utf8 work/_pinattach.py --mode orphan --r 0 --report      # census
  python -X utf8 work/_pinattach.py --mode orphan --r 25 --only-stamp --out pa25
  python -X utf8 tools/eval/fast.py --score-only --cases 1,2,3,4,5,6,7,8,9,10 --out pa25
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

# archived under eda-pcb/tools/diag/ => repo root is 3 levels up
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402

STAMPED = frozenset(("r", "c", "l", "fuse", "crystal_2pin", "d", "led"))
POWER_PREFIX = "\u2205"


def edges_of(net):
    out = []
    e = net.get("edges")
    if isinstance(e, dict):
        for k, seg in e.items():
            if isinstance(seg, list) and len(seg) == 2:
                out.append((k, (float(seg[0]["x"]), float(seg[0]["y"])),
                            (float(seg[1]["x"]), float(seg[1]["y"]))))
    return out


def seg_closest(px, py, p0, p1):
    """-> (distance, closest point)"""
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    l2 = vx * vx + vy * vy
    if l2 <= 1e-9:
        cx, cy = p0
    else:
        t = max(0.0, min(1.0, ((px - p0[0]) * vx + (py - p0[1]) * vy) / l2))
        cx, cy = p0[0] + t * vx, p0[1] + t * vy
    return math.hypot(px - cx, py - cy), (cx, cy)


def canon_ref(comp, pin):
    """`pins[*]` keys are `pin_1`; hyperGraph refs carry the bare number."""
    return comp, M.pin_key_to_number(pin)


def dump(refs):
    return "(" + ",".join("%s.%s" % (c, p) for c, p in refs) + ")"


def append_ref(hg, ref):
    hg = (hg or "").strip()
    if hg.endswith(")") and len(hg) > 2:
        return hg[:-1] + "," + "%s.%s" % ref + ")"
    return "(" + "%s.%s" % ref + ")"


def centre(comps, comp):
    b = (comps.get(comp) or {}).get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            x0, y0, x1, y1 = [float(v) for v in b]
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        except (TypeError, ValueError):
            return None
    return None


def axis_angle(px, py, cxy, wx, wy):
    """Angle (deg) between the pin's outward axis and the wire direction."""
    if cxy is None:
        return 0.0
    ox, oy = px - cxy[0], py - cxy[1]
    on = math.hypot(ox, oy)
    wx_, wy_ = wx - px, wy - py
    wn = math.hypot(wx_, wy_)
    if on < 1e-6 or wn < 1e-6:
        return 0.0
    cos = (ox * wx_ + oy * wy_) / (on * wn)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out_finalrd8"))
    ap.add_argument("--out", default="pa")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--mode", default="orphan",
                    choices=("touch", "orphan", "move"))
    ap.add_argument("--r", type=float, default=20.0)
    ap.add_argument("--deg", type=float, default=999.0)
    ap.add_argument("--min-gain", type=float, default=8.0)
    ap.add_argument("--only-stamp", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    src = Path(a.pred)
    dst = ROOT / "work" / ("_fast_out_" + a.out)
    (dst / "final_results").mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)

    tot = {"orphan": 0, "move": 0, "stamp": 0, "nets": 0}
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or "")
        if cid not in set(a.cases.split(",")):
            continue
        pp = src / "final_results" / (cid + "_Pred.json")
        if not pp.exists():
            continue
        doc = json.loads(pp.read_text(encoding="utf-8"))
        comps = doc.get("components") or {}
        pins = doc.get("pins") or {}
        nets = doc.get("nets") or {}
        segs = {nm: edges_of(net) for nm, net in nets.items()}
        member = {}
        for nm, net in nets.items():
            for ref in M.parse_hypergraph(net.get("hyperGraph", "")):
                member[ref] = nm

        n_o = n_m = n_s = 0
        for comp in sorted(pins):
            if comp.startswith(POWER_PREFIX) or comp not in comps:
                continue
            ctype = str((comps[comp] or {}).get("type") or "").lower()
            if a.only_stamp and ctype not in STAMPED:
                continue
            cxy = centre(comps, comp)
            for pk in sorted(pins[comp] or {}):
                pt = (pins[comp][pk] or {}).get("point") or {}
                if pt.get("x") is None:
                    continue
                px, py = float(pt["x"]), float(pt["y"])
                ranked = []
                for nm, segl in segs.items():
                    hits = [seg_closest(px, py, p0, p1) for _ek, p0, p1 in segl]
                    if hits:
                        d, cp = min(hits, key=lambda h: h[0])
                        ranked.append((d, cp, nm))
                if not ranked:
                    continue
                ranked.sort(key=lambda t: t[0])
                dist, cp, best = ranked[0]
                ref = canon_ref(comp, pk)
                cur = member.get(ref)
                if a.report and cur is None:
                    ang = axis_angle(px, py, cxy, cp[0], cp[1])
                    if dist <= 60.0:
                        print("   case %-3s %-8s %-4s %-6s d=%5.1f deg=%5.1f  "
                              "(2nd %s d=%.1f)"
                              % (cid, ref[0], ref[1], ctype, dist, ang,
                                 ranked[1][2] if len(ranked) > 1 else "-",
                                 ranked[1][0] if len(ranked) > 1 else -1))
                if cur is None and a.mode != "touch":
                    ang = axis_angle(px, py, cxy, cp[0], cp[1])
                    if dist <= a.r and ang <= a.deg:
                        nets[best]["hyperGraph"] = append_ref(
                            nets[best].get("hyperGraph"), ref)
                        member[ref] = best
                        n_o += 1
                        n_s += 1 if ctype in STAMPED else 0
                        if not a.quiet and n_o <= 8:
                            print("   case %-3s +%s.%-4s -> %s (d=%.1f deg=%.1f %s)"
                                  % (cid, ref[0], ref[1], best, dist, ang, ctype))
                elif a.mode == "move" and cur is not None and cur != best:
                    dcur = min([seg_closest(px, py, p0, p1)[0]
                                for _ek, p0, p1 in segs.get(cur, [])] or [1e9])
                    if dist <= a.r and dcur - dist >= a.min_gain:
                        old = M.parse_hypergraph(nets[cur].get("hyperGraph", ""))
                        if ref in old:
                            old.remove(ref)
                        nets[cur]["hyperGraph"] = dump(old)
                        nets[best]["hyperGraph"] = append_ref(
                            nets[best].get("hyperGraph"), ref)
                        member[ref] = best
                        n_m += 1
                        if not a.quiet and n_m <= 8:
                            print("   case %-3s ~%s.%-4s %s -> %s (%.1f -> %.1f)"
                                  % (cid, ref[0], ref[1], cur, best, dcur, dist))

        if a.mode == "touch":
            for nm, net in nets.items():
                net["hyperGraph"] = dump(M.parse_hypergraph(net.get("hyperGraph", "")))

        if not (a.report and a.r <= 0):
            (dst / "final_results" / pp.name).write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        tot["orphan"] += n_o
        tot["move"] += n_m
        tot["stamp"] += n_s
        tot["nets"] += len(nets)

    print("[pinattach] mode=%s r=%.1f deg=%.0f min_gain=%.1f only_stamp=%s -> %s"
          % (a.mode, a.r, a.deg, a.min_gain, a.only_stamp, dst))
    print("[pinattach] 新增 %d 条（其中 2-pin 符号 %d 条）、迁移 %d 条，网数 %d"
          % (tot["orphan"], tot["stamp"], tot["move"], tot["nets"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
