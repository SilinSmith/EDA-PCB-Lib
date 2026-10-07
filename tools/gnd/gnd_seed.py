"""Two follow-ups on the "seed one GND, then traverse the sheet" plan.

1. Per-SHEET seeding: an anchor is only useful if a sheet has at least one.
   (A per-symbol rate of 27% on a sheet with 7 symbols is a per-sheet rate of
   ~89% -- the right denominator is the sheet, not the symbol.)

2. Within-sheet self-similarity, which is the whole premise of "traverse the
   sheet for look-alikes": take one gnd crop, NCC it against the OTHER gnd crops
   of the SAME sheet, versus gnd crops of OTHER sheets.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

GROUND = re.compile(r"GND|VSS|EARTH", re.I)
SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
TILE = 32
MIN_SIDE, MAX_SIDE = 9.0, 46.0


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def ncc(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


def crop(g, box):
    x1, y1, x2, y2 = [float(v) for v in box]
    h = g.shape[0]
    iy1, iy2 = h - y2, h - y1
    x1, x2 = sorted((x1, x2)); iy1, iy2 = sorted((iy1, iy2))
    if not (MIN_SIDE <= max(x2 - x1, iy2 - iy1) <= MAX_SIDE):
        return None
    sub = g[max(0, int(iy1)):int(iy2), max(0, int(x1)):int(x2)]
    if sub.size == 0 or min(sub.shape) < 4:
        return None
    return np.asarray(Image.fromarray(sub).resize((TILE, TILE), Image.BILINEAR), dtype=np.float32)


def main() -> None:
    sheets = []           # (src, anchored_any, n_gnd, crops[])
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
        token_net = {}
        net_tokens = {}
        for nk, net in nets.items():
            toks = S.parse_hypergraph(net.get("hyperGraph", ""))
            net_tokens[nk] = toks
            for t in toks:
                token_net[f"{t[0]}.{t[1]}"] = nk

        gnds = {k: v for k, v in comps.items() if str(v.get("type", "")).strip() == "gnd"}
        if len(gnds) < 2:
            continue

        anchored = 0
        for ck in gnds:
            toks = []
            for pk in (pins.get(ck) or {}):
                sfx = pk[4:] if pk.startswith("pin_") else pk
                nk = token_net.get(f"{ck}.{sfx}")
                if nk:
                    toks.extend(net_tokens.get(nk, []))
            ok = False
            for oc in {t[0] for t in toks if t[0] != ck}:
                for pk, pin in (pins.get(oc) or {}).items():
                    if isinstance(pin, dict) and GROUND.search(str(pin.get("pinname", ""))):
                        ok = True; break
                if ok:
                    break
            if ok:
                anchored += 1

        g = None
        crops = []
        for comp in gnds.values():
            b = comp.get("bbox")
            if not b or len(b) != 4:
                continue
            if g is None:
                try:
                    g = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
                except Exception:
                    break
            cr = crop(g, b)
            if cr is not None and len(crops) < 14:
                crops.append(cr)
        if len(crops) >= 2:
            sheets.append((source_of(c.json_path), anchored, len(gnds), crops))

    nsheet = len(sheets)
    any_anchor = sum(1 for _, a, _, _ in sheets if a > 0)
    tot_g = sum(n for _, _, n, _ in sheets)
    tot_a = sum(a for _, a, _, _ in sheets)
    print(f"sheets with >=2 gnd: {nsheet}   gnd symbols: {tot_g}   anchored: {tot_a} "
          f"({tot_a/max(tot_g,1)*100:.1f}%)")
    print(f"** sheets with AT LEAST ONE anchored gnd: {any_anchor}/{nsheet} "
          f"= {any_anchor/max(nsheet,1)*100:.1f}%  <- the number that matters for seeding **")
    print(f"sheets with ZERO anchored gnd: {nsheet-any_anchor}")
    print(f"mean gnd per sheet: {tot_g/max(nsheet,1):.2f}")

    # ---- within-sheet vs cross-sheet similarity ----
    within, cross = [], []
    rng = np.random.default_rng(0)
    pools: dict[str, list[np.ndarray]] = defaultdict(list)
    for src, _, _, crops in sheets:
        for i in range(len(crops)):
            for j in range(i + 1, len(crops)):
                within.append(ncc(crops[i], crops[j]))
        if len(pools[src]) < 60:
            pools[src].extend(crops[:4])

    for src, pool in pools.items():
        if len(pool) < 4:
            continue
        idx = rng.permutation(len(pool))
        for i in range(0, len(idx) - 1, 2):
            cross.append(ncc(pool[idx[i]], pool[idx[i + 1]]))

    print(f"\n=== gnd crop self-similarity (NCC of 32x32 crops) ===")
    if within:
        w = np.asarray(within, dtype=float)
        print(f"  WITHIN sheet : n={len(w):6d}  mean {w.mean():.3f}  "
              f"p10 {np.percentile(w,10):.3f}  p50 {np.percentile(w,50):.3f}  "
              f"p90 {np.percentile(w,90):.3f}")
    if cross:
        x = np.asarray(cross, dtype=float)
        print(f"  ACROSS sheets: n={len(x):6d}  mean {x.mean():.3f}  "
              f"p10 {np.percentile(x,10):.3f}  p50 {np.percentile(x,50):.3f}  "
              f"p90 {np.percentile(x,90):.3f}")
    print("\n  (if WITHIN >> ACROSS, per-sheet template propagation is viable;")
    print("   note crops are resized to a fixed square, so this is shape-only)")


if __name__ == "__main__":
    main()
