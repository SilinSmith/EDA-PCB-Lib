"""Can a ground-named PIN anchor the ground SYMBOL on the same net?

Architecture being tested: "confirm one GND symbol per sheet, then traverse the
sheet for look-alikes".  The seed is supposed to be a pin whose *name* is GND /
VSS / AGND ... -- a name is far more vendor-stable than a glyph.

For every GT pin whose pinname looks like a ground:
  follow its net -> all edge points -> the far end of the net
  is that far end within N px of a GT gnd component's bbox?
And the reverse, which is what actually sizes the architecture:
  for every GT gnd component, does its net carry a ground-named pin at all?

Also runs the identical test for power (VCC/VDD) -> `v`, as a control: if the
method works for power but not ground, the problem is the data, not the method.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
GROUND = re.compile(r"GND|VSS|V_?SS|EARTH|AGND|DGND|PGND", re.I)
POWER = re.compile(r"VCC|VDD|VBAT|VUSB|VIN|3V3|5V|1V8|12V|VREF|VBUS|AVDD|DVDD", re.I)
NEAR = 14.0


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def bbox_dist(px, py, box):
    """box is [x1,y1,x2,y2] in JSON (bottom-left) coords; distance in px."""
    x1, y1, x2, y2 = [float(v) for v in box]
    lx, hx = min(x1, x2), max(x1, x2)
    ly, hy = min(y1, y2), max(y1, y2)
    dx = max(lx - px, 0.0, px - hx)
    dy = max(ly - py, 0.0, py - hy)
    return float(np.hypot(dx, dy))


def pin_point(pin):
    p = pin.get("point") or {}
    return float(p.get("x", 0.0)), float(p.get("y", 0.0))


def main() -> None:
    stat = defaultdict(lambda: defaultdict(int))
    dists: dict[str, list[float]] = defaultdict(list)
    gnd_kind = Counter()
    per_src = defaultdict(lambda: defaultdict(int))

    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        comps = doc.get("components") or {}
        pins = doc.get("pins") or {}
        nets = doc.get("nets") or {}
        if not comps:
            continue
        src = source_of(c.json_path)

        # net index: "<comp>.<suffix>" -> net key ; flat list of pins
        token_net: dict[str, str] = {}
        net_tokens: dict[str, list[tuple[str, str]]] = {}
        for nk, net in nets.items():
            toks = S.parse_hypergraph(net.get("hyperGraph", ""))
            net_tokens[nk] = toks
            for t in toks:
                token_net[f"{t[0]}.{t[1]}"] = nk

        # far point of a net = the edge endpoint farthest from a given point
        def far_point(nk, px, py):
            best, bd = (px, py), -1.0
            if nk is None:
                return best
            for pts in ((nets.get(nk) or {}).get("edges") or {}).values():
                if not isinstance(pts, list):
                    continue
                for p in pts:
                    if not isinstance(p, dict):
                        continue
                    x, y = float(p.get("x", 0.0)), float(p.get("y", 0.0))
                    d = float(np.hypot(x - px, y - py))
                    if d > bd:
                        bd, best = d, (x, y)
            return best

        gnd_boxes = [(k, comp) for k, comp in comps.items()
                     if str(comp.get("type", "")).strip() == "gnd"]
        pow_boxes = [(k, comp) for k, comp in comps.items()
                     if str(comp.get("type", "")).strip() in ("v", "vbus_in", "vcc")]

        def cls_of(name: str) -> str | None:
            if GROUND.search(name):
                return "gnd"
            if POWER.search(name):
                return "pow"
            return None

        # ---- forward: ground-named pin -> does its net reach a gnd symbol? --
        for ck, plist in pins.items():
            if not isinstance(plist, dict):
                continue
            for pk, pin in plist.items():
                if not isinstance(pin, dict):
                    continue
                nm = str(pin.get("pinname", ""))
                kind = cls_of(nm)
                if kind is None:
                    continue
                suffix = pk[4:] if pk.startswith("pin_") else pk
                nk = token_net.get(f"{ck}.{suffix}")
                px, py = pin_point(pin)
                fx, fy = far_point(nk, px, py)
                target = gnd_boxes if kind == "gnd" else pow_boxes
                best = min((bbox_dist(fx, fy, comp["bbox"]) for _, comp in target), default=1e9)
                key = f"{kind}_near"
                stat[key]["pins"] += 1
                dists[kind].append(best)
                if best <= NEAR:
                    stat[key]["hit"] += 1
                    per_src[src][f"{kind}_hit"] += 1
                per_src[src][f"{kind}_pin"] += 1

        # ---- reverse: every gnd/v component, does its net carry such a pin? -
        for lab, boxes in (("gnd", gnd_boxes), ("pow", pow_boxes)):
            for ck, comp in boxes:
                gnd_kind[f"{lab}:{'other' if ck in pins else 'no_pins_entry'}"] += 1
                plist = pins.get(ck) or {}
                toks: list[tuple[str, str]] = []
                for pk in (plist if isinstance(plist, dict) else {}):
                    suffix = pk[4:] if pk.startswith("pin_") else pk
                    nk = token_net.get(f"{ck}.{suffix}")
                    if nk:
                        toks.extend(net_tokens.get(nk, []))
                other_comp = {t[0] for t in toks if t[0] != ck}
                if not other_comp:
                    stat[f"{lab}_rev"]["isolated"] += 1
                    stat[f"{lab}_rev"]["total"] += 1
                    per_src[src][f"{lab}_rev_total"] += 1
                    continue
                has = False
                for oc in other_comp:
                    for pk, pin in (pins.get(oc) or {}).items():
                        if not isinstance(pin, dict):
                            continue
                        if GROUND.search(str(pin.get("pinname", ""))):
                            has = True
                            break
                    if has:
                        break
                stat[f"{lab}_rev"]["total"] += 1
                per_src[src][f"{lab}_rev_total"] += 1
                if has:
                    stat[f"{lab}_rev"]["anchored"] += 1
                    per_src[src][f"{lab}_rev_hit"] += 1

    print("=== forward: a ground/power-named pin, far end of its net, distance to a gnd/v symbol ===")
    for kind in ("gnd", "pow"):
        n = stat[f"{kind}_near"]["pins"]
        h = stat[f"{kind}_near"]["hit"]
        d = np.asarray(dists.get(kind, []), dtype=float)
        if n:
            print(f"  {kind}-named pins: {n:6d}  far-end within {NEAR:.0f}px: {h:6d}  "
                  f"= {h/n*100:5.1f}%")
            for q in (10, 25, 50, 75, 90):
                print(f"      dist p{q:<3d} {np.percentile(d, q):7.1f} px")
    print("\n=== reverse: can each gnd / v symbol be anchored at all? ===")
    for lab, human in (("gnd", "gnd symbols"), ("pow", "v/power symbols")):
        t = stat[f"{lab}_rev"]["total"]
        a = stat[f"{lab}_rev"]["anchored"]
        iso = stat[f"{lab}_rev"]["isolated"]
        if t:
            print(f"  {human:20s} total {t:6d}   net carries a gnd-named pin: {a:6d} "
                  f"= {a/t*100:5.1f}%   (net has no other component: {iso} = {iso/t*100:.1f}%)")

    print("\n=== by source (anchor coverage of gnd symbols) ===")
    print(f"{'source':18s} {'gnd total':>10s} {'anchored':>9s} {'rate':>7s} "
          f"{'gndpins':>8s} {'far<14px':>9s}")
    for s in SRC_ORDER + ["unknown"]:
        d = per_src.get(s)
        if not d or not d.get("gnd_rev_total"):
            continue
        t = d["gnd_rev_total"]; a = d["gnd_rev_hit"]
        pn = d.get("gnd_pin", 0); ph = d.get("gnd_hit", 0)
        print(f"{s:18s} {t:10d} {a:9d} {a/max(t,1)*100:6.1f}% {pn:8d} {ph:9d}")

    print("\n=== sanity: how many gnd/v components have a pins entry at all ===")
    for k, v in gnd_kind.most_common():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
