"""Follow-up to work/_gndbox.py -- are the k-means 'gnd modes' real, or are they
just degrees of GT-box misalignment?

Every GT gnd box is the SAME size (20x19 or 19x20) -- a fixed template.  And the
ink centroid sits a median 11.4 px away from the box centre.  So the natural
question: if we first RE-CENTRE each crop on its ink, does the multi-modality
survive?

Two representations of the same 2229 boxes:
  raw   : the box as annotated, resized to 32x32, zero-mean unit-var  (A's way)
  rec   : a 20x19-ish window centred on the ink centroid of the grown box,
          same normalisation

For each we sweep k=2..10 with k-means (fixed seed) and report inertia +
silhouette, plus the per-source share of empty boxes.

Read-only; writes nothing.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
TILE = 32
GROW = 30
import os as _os

# ⚠ 符号线是浅灰（151/152），不是深色；阈值必须 >=200，否则整片符号会被当成空。
INK_THR = int(_os.environ.get("GNDBOX_INK", "200"))


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def box_px(box, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    iy1, iy2 = sorted((h - y2, h - y1))
    x1, x2 = sorted((x1, x2))
    return x1, iy1, x2, iy2


def norm32(sub: np.ndarray) -> np.ndarray:
    a = np.asarray(Image.fromarray(sub).resize((TILE, TILE), Image.BILINEAR), dtype=np.float32)
    return ((a - a.mean()) / (a.std() + 1e-6)).ravel()


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(X) * k * X.shape[1] < 4e8 else None
        if d is None:
            d = np.empty((len(X), k), dtype=np.float32)
            for j in range(k):
                d[:, j] = ((X - C[j]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab


def silhouette(X: np.ndarray, lab: np.ndarray, sample: int = 600, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs, ls = X[idx], lab[idx]
    D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
    out = []
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, ls == c].mean() for c in np.unique(ls) if c != ls[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    cases = S.discover()
    raw, rec, srcs, empties = [], [], [], []
    for c in cases:
        if not c.json_path:
            continue
        gray = np.asarray(Image.open(c.png_path).convert("L"))
        h, w = gray.shape
        doc = S.load_json(c.json_path)
        for key, v in ((doc or {}).get("components") or {}).items():
            if str(v.get("type") or "").lower() != "gnd":
                continue
            bb = v.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = box_px(bb, h)
            a, b = max(0, int(y1)), min(h, int(round(y2)))
            cc, dd = max(0, int(x1)), min(w, int(round(x2)))
            sub = gray[a:b, cc:dd]
            if sub.size == 0 or min(sub.shape) < 5:
                continue
            ga, gb = max(0, int(y1) - GROW), min(h, int(round(y2)) + GROW)
            gc, gd = max(0, int(x1) - GROW), min(w, int(round(x2)) + GROW)
            gsub = gray[ga:gb, gc:gd]
            mask = gsub < INK_THR
            s = source_of(c.png_path)
            ink = float((sub < INK_THR).mean())
            if not mask.any():
                continue  # nothing to re-centre on
            ys, xs = np.nonzero(mask)
            cy = int(round(ys.mean())) + ga
            cx = int(round(xs.mean())) + gc
            bw, bh = max(6, int(round(x2 - x1))), max(6, int(round(y2 - y1)))
            ra, rb = max(0, cy - bh // 2), min(h, cy - bh // 2 + bh)
            rc, rd = max(0, cx - bw // 2), min(w, cx - bw // 2 + bw)
            r2 = gray[ra:rb, rc:rd]
            if r2.size == 0 or min(r2.shape) < 5:
                continue
            raw.append(norm32(sub))
            rec.append(norm32(r2))
            srcs.append(s)
            empties.append(ink <= 0.02)
    Xr = np.stack(raw)
    Xc = np.stack(rec)
    print(f"usable gnd boxes: {len(Xr)}   (skipped: no ink anywhere in +-{GROW}px)")
    cnt = Counter(srcs)
    emp = Counter(s for s, e in zip(srcs, empties) if e)
    print("\n-- share of boxes with <=2% ink, by source --")
    for s in sorted(cnt, key=lambda k: -cnt[k]):
        print(f"  {s:<20} {emp[s]:4d}/{cnt[s]:5d}  {100*emp[s]/cnt[s]:5.1f}%")
    tot_e = sum(empties)
    print(f"  {'ALL':<20} {tot_e:4d}/{len(empties):5d}  {100*tot_e/len(empties):5.1f}%")

    for name, X in (("raw (box as annotated)", Xr), ("recentred on ink", Xc)):
        print(f"\n=== {name} ===")
        print("   k   inertia      silhouette")
        best = (None, -2.0)
        for k in range(2, 11):
            lab = kmeans(X, k, seed=k)
            inertia = 0.0
            for j in range(k):
                m = lab == j
                if m.any():
                    inertia += float(((X[m] - X[m].mean(0)) ** 2).sum())
            sil = silhouette(X, lab)
            print(f"  {k:2d}  {inertia/1e5:9.2f}   {sil:8.4f}")
            if sil > best[1]:
                best = (k, sil)
        print(f"  -> best k={best[0]} silhouette={best[1]:.4f}")


if __name__ == "__main__":
    main()
