"""Control for work/_gndmodes3.py -- the same k-means/silhouette sweep on other
component types, so that "gnd silhouette is only 0.14" can be read against a
yardstick instead of in a vacuum.

usage:  python work/_modesctl.py gnd v c r net_input d
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

TILE = 32


def box_px(box, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    iy1, iy2 = sorted((h - y2, h - y1))
    return x1, iy1, sorted((x1, x2))[1], iy2


def norm32(sub: np.ndarray) -> np.ndarray:
    a = np.asarray(Image.fromarray(sub).resize((TILE, TILE), Image.BILINEAR), dtype=np.float32)
    return ((a - a.mean()) / (a.std() + 1e-6)).ravel()


def kmeans(X, k, seed=0, iters=50):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
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


def silhouette(X, lab, sample=500, seed=0):
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


def collect(want: str, limit: int = 2600):
    X, srcs = [], []
    for c in S.discover():
        if not c.json_path:
            continue
        doc = S.load_json(c.json_path)
        comps = (doc or {}).get("components") or {}
        if not any(str(v.get("type") or "").lower() == want for v in comps.values()):
            continue
        gray = np.asarray(Image.open(c.png_path).convert("L"))
        h, w = gray.shape
        for key, v in comps.items():
            if str(v.get("type") or "").lower() != want:
                continue
            bb = v.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [float(t) for t in bb]
            iy1, iy2 = sorted((h - y2, h - y1))
            x1, x2 = sorted((x1, x2))
            a, b = max(0, int(iy1)), min(h, int(round(iy2)))
            cc, dd = max(0, int(x1)), min(w, int(round(x2)))
            if b - a < 5 or dd - cc < 5:
                continue
            X.append(norm32(gray[a:b, cc:dd]))
            srcs.append(Path(c.png_path).name)
            if len(X) >= limit:
                return np.stack(X), srcs
    return (np.stack(X) if X else np.zeros((0, TILE * TILE), np.float32)), srcs


def main(types):
    for want in types:
        X, srcs = collect(want)
        n = len(X)
        if n < 60:
            print(f"\n=== {want}: too few usable boxes ({n}) ===")
            continue
        print(f"\n=== {want}: {n} boxes ===")
        best = (None, -2.0)
        line = []
        for k in (2, 3, 4, 6, 9, 12):
            lab = kmeans(X, k, seed=k)
            sil = silhouette(X, lab)
            line.append(f"k{k}={sil:.3f}")
            if sil > best[1]:
                best = (k, sil)
        print("  silhouette: " + "  ".join(line))
        print(f"  -> best k={best[0]} silhouette={best[1]:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["gnd", "v", "c", "r"]
    main(args)
