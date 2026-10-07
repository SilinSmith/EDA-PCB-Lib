"""Measure the statistics a synthetic sheet must reproduce, from the 220 real cases.

Answers three questions the generator currently guesses at:
  1. how densely is a real sheet populated (components per megapixel, bbox coverage)
  2. how much wire does a real net actually have (edges per net, wire length, and whether
     power/ground nets are wired at all -- they usually are not, they are implied by the
     symbol name)
  3. how long is a typical wire segment
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("10GT", "10_GTcase", "200_train_cases")
POWERISH = {"v", "gnd"}


def find_json(d: Path):
    cands = sorted(d.glob("*_target*.json")) or sorted(d.glob("*.json"))
    return cands[0] if cands else None


def main() -> int:
    from PIL import Image

    n_comp_mp, cov, nets_per_mp = [], [], []
    edges_by_pins = defaultdict(list)
    zero_edge_by_type = Counter()
    seg_lengths = []
    pins_per_net = Counter()
    edges_per_net = Counter()
    gnd_edges, v_edges, sig_edges = [], [], []
    comp_area_frac = []

    cases = 0
    for split in SPLITS:
        for d in sorted((ROOT / split).iterdir()):
            if not d.is_dir():
                continue
            jp = find_json(d)
            if not jp:
                continue
            doc = json.loads(jp.read_text(encoding="utf-8"))
            pngs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            if not pngs:
                continue
            with Image.open(pngs[0]) as im:
                W, H = im.size
            mp = W * H / 1e6
            comps = doc.get("components", {})
            nets = doc.get("nets", {})
            cases += 1
            n_comp_mp.append(len(comps) / mp)
            nets_per_mp.append(len(nets) / mp)

            sheet = W * H
            covered = 0.0
            for c in comps.values():
                b = c.get("bbox") or [0, 0, 0, 0]
                covered += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            comp_area_frac.append(covered / sheet)

            types = {k: (c.get("type") or "") for k, c in comps.items()}
            for net in nets.values():
                edges = net.get("edges") or {}
                npins = len([t for t in (net.get("hyperGraph") or "").strip("()").split(",") if t.strip()])
                pins_per_net[min(npins, 30)] += 1
                edges_per_net[min(len(edges), 30)] += 1
                edges_by_pins[min(npins, 30)].append(len(edges))
                if not edges:
                    zero_edge_by_type["net"] += 1
                # classify by the types of the components this net touches
                touched = {types.get(t.rsplit(".", 1)[0], "?")
                           for t in (net.get("hyperGraph") or "").strip("()").split(",") if t.strip()}
                if touched and touched <= POWERISH:
                    gnd_edges.append(len(edges))
                elif "v" in touched:
                    v_edges.append(len(edges))
                else:
                    sig_edges.append(len(edges))
                for e in edges.values():
                    pts = [(float(p["x"]), float(p["y"])) for p in e]
                    for a, b in zip(pts, pts[1:]):
                        seg_lengths.append(math.dist(a, b))

    def q(vals, name, unit=""):
        if not vals:
            return
        vals = sorted(vals)
        n = len(vals)
        p = lambda f: vals[min(n - 1, int(f * n))]
        print(f"{name:26} n={n:6}  mean={st.mean(vals):9.2f}  p10={p(.1):8.2f}  "
              f"p50={p(.5):8.2f}  p90={p(.9):9.2f}  max={vals[-1]:10.2f}{unit}")

    print(f"cases analysed: {cases}")
    q(n_comp_mp, "components / Mpx")
    q(nets_per_mp, "nets / Mpx")
    q([f * 100 for f in comp_area_frac], "component area / sheet", "%")
    q(seg_lengths, "wire segment length", " px")

    print("\n--- nets with zero edges (implied by symbol name, not wired) ---")
    tot = sum(edges_per_net.values())
    print(f"  {zero_edge_by_type['net']} / {tot} nets = {100*zero_edge_by_type['net']/tot:.1f}%")
    for name, vals in (("gnd-only nets", gnd_edges), ("v-touching nets", v_edges), ("signal nets", sig_edges)):
        z = sum(1 for x in vals if x == 0)
        print(f"  {name:16} n={len(vals):6}  zero-edge={z:5} ({100*z/max(1,len(vals)):5.1f}%)  mean edges={st.mean(vals) if vals else 0:.2f}")

    print("\n--- edges per net, by pin count ---")
    for k in sorted(edges_by_pins):
        v = edges_by_pins[k]
        z = sum(1 for x in v if x == 0)
        print(f"  pins={k:3}  nets={len(v):5}  mean_edges={st.mean(v):6.2f}  zero={100*z/len(v):5.1f}%")

    print("\n--- distribution of net sizes (pins) ---")
    tt = sum(pins_per_net.values())
    for k in sorted(pins_per_net)[:16]:
        print(f"  {k:3} pins: {pins_per_net[k]:6}  ({100*pins_per_net[k]/tt:5.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
