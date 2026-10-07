"""GND matched-filter feasibility probe.

Question: can a fixed n*n kernel (a "matched filter") find GND symbols on real
sheets, across the four source families?

We answer it with three measurements:
  1. per-source mean template + cross-source normalised correlation
     (is the glyph the same shape in KiCad / Altium / Datasheet / jlc?)
  2. NCC of a source's mean template against its OWN gnd crops vs. against
     random non-gnd ink of the same size  -> ROC-AUC
  3. at the threshold that keeps 90% recall, how many false positives per
     megapixel of non-blank ink do we get?
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser import data as S

TILE = 32
MIN_BOX = 10.0
MAX_BOX = 60.0
SEED = 0


def source_of(path) -> str:
    n = Path(path).name
    for tag in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        if tag in n:
            return tag
    return "unknown"


def load_gray(png) -> np.ndarray:
    im = Image.open(str(png)).convert("L")
    return np.asarray(im, dtype=np.uint8)


def crop_norm(g: np.ndarray, box, h: int, tile: int = TILE) -> np.ndarray | None:
    """box is [x1, y1, x2, y2] in bottom-left-origin JSON coords."""
    x1, y1, x2, y2 = box
    iy1 = h - y2
    iy2 = h - y1
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    iy1, iy2 = sorted((int(round(iy1)), int(round(iy2))))
    if x2 - x1 < 3 or iy2 - iy1 < 3:
        return None
    sub = g[max(0, iy1):min(g.shape[0], iy2), max(0, x1):min(g.shape[1], x2)]
    if sub.size == 0:
        return None
    return np.asarray(Image.fromarray(sub).resize((tile, tile), Image.BILINEAR), dtype=np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


def main() -> None:
    rng = np.random.default_rng(SEED)
    root = S.DEFAULT_DATA_ROOT
    cases = S.discover(root)

    per_src_gnd: dict[str, list[np.ndarray]] = defaultdict(list)
    per_src_other: dict[str, list[np.ndarray]] = defaultdict(list)
    counts: Counter = Counter()

    for c in cases:
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        comps = doc.get("components") or {}
        if not comps:
            continue
        src = source_of(c.json_path)
        counts[src] += 1
        g = None
        for key, comp in comps.items():
            t = str(comp.get("type", "")).strip()
            box = comp.get("bbox")
            if not box or len(box) != 4:
                continue
            w = abs(box[2] - box[0])
            hh = abs(box[3] - box[1])
            if not (MIN_BOX <= max(w, hh) <= MAX_BOX):
                continue
            if g is None:
                try:
                    g = load_gray(c.png_path)
                except Exception:
                    break
            patch = crop_norm(g, [float(v) for v in box], g.shape[0])
            if patch is None:
                continue
            if t == "gnd":
                if len(per_src_gnd[src]) < 400:
                    per_src_gnd[src].append(patch)
            elif len(per_src_other[src]) < 800:
                per_src_other[src].append(patch)

    srcs = [s for s in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other")
            if len(per_src_gnd.get(s, [])) >= 8]
    print("=== cases per source ===")
    for s, n in counts.most_common():
        print(f"  {s:18s} cases={n:4d}  gnd_patches={len(per_src_gnd.get(s, [])):4d}"
              f"  other_patches={len(per_src_other.get(s, [])):4d}")
    print(f"\nsources with enough gnd: {srcs}\n")

    if len(srcs) < 2:
        print("not enough sources to compare")
        return

    means = {s: np.mean(np.stack(per_src_gnd[s]), axis=0) for s in srcs}

    print("=== cross-source NCC of the mean template (same shape?) ===")
    print("            " + "".join(f"{s[:10]:>12s}" for s in srcs))
    for a in srcs:
        row = "".join(f"{ncc(means[a], means[b]):12.3f}" for b in srcs)
        print(f"{a[:11]:11s} {row}")

    print("\n=== within-source spread: NCC of each gnd patch vs its own mean ===")
    for s in srcs:
        vals = [ncc(p, means[s]) for p in per_src_gnd[s]]
        print(f"  {s:18s} n={len(vals):4d}  mean {np.mean(vals):.3f}  "
              f"p10 {np.percentile(vals,10):.3f}  p50 {np.percentile(vals,50):.3f}")

    print("\n=== matched filter: gnd-vs-other separation, per source ===")
    print(f"{'source':18s} {'AUC':>7s} {'thr@R90':>9s} {'R@thr':>7s} {'P@thr':>7s} {'medgnd':>8s} {'medoth':>8s}")
    for s in srcs:
        pos = np.array([ncc(p, means[s]) for p in per_src_gnd[s]])
        neg = np.array([ncc(p, means[s]) for p in per_src_other.get(s, [])])
        if len(neg) < 10:
            continue
        allv = np.concatenate([pos, neg])
        labs = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(allv)
        ranks = np.empty(len(allv), dtype=np.float64)
        ranks[order] = np.arange(1, len(allv) + 1)
        npos, nneg = len(pos), len(neg)
        auc = (ranks[labs == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        thr = float(np.percentile(pos, 10))
        tp = int((pos >= thr).sum()); fp = int((neg >= thr).sum())
        fn = npos - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"{s:18s} {auc:7.3f} {thr:9.3f} {rec:7.3f} {prec:7.3f} "
              f"{np.median(pos):8.3f} {np.median(neg):8.3f}")

    print("\n=== how many OTHER-class symbols look like gnd? (NCC vs own mean) ===")
    for s in srcs:
        if not per_src_other.get(s):
            continue
        vals = np.array([ncc(p, means[s]) for p in per_src_other[s]])
        for q in (0.5, 0.9, 0.99, 1.0):
            print(f"  {s:18s} other NCC p{int(q*100):<3d} = {np.percentile(vals, q*100):.3f}"
                  + ("  <- max" if q == 1.0 else ""))


if __name__ == "__main__":
    main()
