"""Second probe: the structural details the generator must match exactly.

  * is an edge always a straight two-point segment?
  * how big is a component bbox, per type (this calibrates symbol scale in pixels)
  * how many pins does a component have, per type
  * what fraction of pins ever appear in a net
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("10GT", "10_GTcase", "200_train_cases")


def main() -> int:
    edge_len = Counter()
    bbox_by_type = defaultdict(list)
    pins_by_type = defaultdict(list)
    pin_total = 0
    pin_in_net = 0
    hyper_tokens_bad = Counter()
    nets_per_case = []
    comps_per_case = []

    for split in SPLITS:
        for d in sorted((ROOT / split).iterdir()):
            if not d.is_dir():
                continue
            cands = sorted(d.glob("*_target*.json")) or sorted(d.glob("*.json"))
            if not cands:
                continue
            doc = json.loads(cands[0].read_text(encoding="utf-8"))
            comps = doc.get("components", {})
            pins = doc.get("pins", {})
            nets = doc.get("nets", {})
            nets_per_case.append(len(nets))
            comps_per_case.append(len(comps))

            for net in nets.values():
                for e in (net.get("edges") or {}).values():
                    edge_len[len(e)] += 1

            for key, c in comps.items():
                b = c.get("bbox") or [0, 0, 0, 0]
                bbox_by_type[c.get("type") or "?"].append((b[2] - b[0], b[3] - b[1]))
                pg = pins.get(key) or {}
                pins_by_type[c.get("type") or "?"].append(len(pg))
                pin_total += len(pg)

            in_net = set()
            for net in nets.values():
                hg = (net.get("hyperGraph") or "").strip()
                for tok in hg.strip("()").split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    if "." not in tok:
                        hyper_tokens_bad[tok[:24]] += 1
                        continue
                    comp, num = tok.rsplit(".", 1)
                    in_net.add((comp, num))
            pin_in_net += len(in_net)

    print("=== edge point counts (is an edge one straight segment?) ===")
    tot = sum(edge_len.values())
    for k in sorted(edge_len)[:8]:
        print(f"  {k} points: {edge_len[k]:7}  ({100*edge_len[k]/tot:6.2f}%)")
    print(f"  total edges: {tot}")

    print("\n=== component bbox size, per type (px) ===")
    print(f"{'type':14} {'n':>6} {'w_p50':>7} {'h_p50':>7} {'w_p90':>7} {'h_p90':>7} {'pins_p50':>9} {'pins_max':>9}")
    for t, v in sorted(bbox_by_type.items(), key=lambda kv: -len(kv[1]))[:18]:
        ws = sorted(x[0] for x in v)
        hs = sorted(x[1] for x in v)
        pv = sorted(pins_by_type[t])
        g = lambda a, f: a[min(len(a) - 1, int(f * len(a)))]
        print(f"{t:14} {len(v):6} {g(ws,.5):7.0f} {g(hs,.5):7.0f} {g(ws,.9):7.0f} {g(hs,.9):7.0f} "
              f"{g(pv,.5):9.0f} {pv[-1]:9}")

    print(f"\ncomponents/sheet : mean={st.mean(comps_per_case):.1f}  p50={st.median(comps_per_case):.0f}")
    print(f"nets/sheet       : mean={st.mean(nets_per_case):.1f}  p50={st.median(nets_per_case):.0f}")
    print(f"pins total       : {pin_total}")
    print(f"pins named in nets: {pin_in_net}  ({100*pin_in_net/max(1,pin_total):.1f}% of all pins)")
    if hyper_tokens_bad:
        print("malformed hyperGraph tokens (top):", hyper_tokens_bad.most_common(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
