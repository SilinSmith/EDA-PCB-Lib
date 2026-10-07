"""Is `gnd` one shape or several?  And does splitting it help a detector?

Part 1  cluster every GT gnd crop by shape -> how many modes are there really,
        and do the modes line up with vendor / with the census forms?
Part 2  the decisive comparison, identical evaluation for both arms:
          1-kernel  = the naive "design one n*n matrix for GND" proposal
          k-kernel  = one kernel PER MODE, outputs unioned and collapsed to gnd
        Both are matched filters; only the number of kernels differs.
        Bar to beat: current YOLO on gnd = P 0.851 / R 0.762.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
TILE = 32
MIN_SIDE, MAX_SIDE = 9.0, 46.0
SCALES = [0.65, 0.8, 1.0, 1.25, 1.55]
MATCH_IOU = 0.25
PEAK_THR = 0.45
NMS_MIN_THR = 0.35
NMS_R = 8
NMS_MAX = 3000
MAX_SHEETS = 120


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def grab(g, box):
    """(normalised 32x32 float for clustering, original uint8 patch for matching)."""
    h = g.shape[0]
    x1, y1, x2, y2 = [float(v) for v in box]
    iy1, iy2 = sorted((h - y2, h - y1))
    x1, x2 = sorted((x1, x2))
    if not (MIN_SIDE <= max(x2 - x1, iy2 - iy1) <= MAX_SIDE):
        return None, None
    a, b = max(0, int(iy1)), int(iy2)
    c, d = max(0, int(x1)), int(x2)
    sub = g[a:b, c:d]
    if sub.size == 0 or min(sub.shape) < 5:
        return None, None
    norm = np.asarray(Image.fromarray(sub).resize((TILE, TILE), Image.BILINEAR), dtype=np.float32)
    norm = (norm - norm.mean()) / (norm.std() + 1e-6)
    pad = cv2.copyMakeBorder(sub, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=int(np.median(sub)))
    return norm, pad


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * (X @ C.T)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    d = (X ** 2).sum(1) + (C[lab] ** 2).sum(1) - 2.0 * (X * C[lab]).sum(1)
    return lab, C, float(np.maximum(d, 0).sum())


def silhouette(X, lab, sample=500, seed=0):
    """O(n*m) via the ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b identity."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    sub, sl = X[idx], lab[idx]
    uniq = np.unique(sl)
    if len(uniq) < 2:
        return 0.0
    Xn = (X ** 2).sum(1)
    D2 = Xn[None, :] + (sub ** 2).sum(1)[:, None] - 2.0 * (sub @ X.T)
    np.maximum(D2, 0, out=D2)
    D = np.sqrt(D2)
    out = []
    for i in range(len(sub)):
        own = lab == sl[i]
        n_own = int(own.sum()) - 1
        if n_own <= 0:
            continue
        a = D[i][own].sum() / n_own
        b = min(D[i][lab == u].mean() for u in uniq if u != sl[i])
        out.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(out)) if out else 0.0


def contact(clusters, path, per=10, cell=48):
    rows = []
    for ci, members in enumerate(clusters):
        row = np.full((cell + 16, cell * per, 3), 255, np.uint8)
        cv2.putText(row, f"mode {ci} (n={len(members)})", (4, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
        for j, (patch, _) in enumerate(members[:per]):
            p = cv2.resize(patch, (cell - 8, cell - 8), interpolation=cv2.INTER_NEAREST)
            row[16:16 + p.shape[0], j * cell + 4:j * cell + 4 + p.shape[1]] = \
                cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def collect(sheets, kernels):
    """Per sheet: all local maxima of the max-over-kernels/scales response.

    A peak counts as a hit when it falls inside a GT gnd box (+4 px slack):
    the question here is whether the filter FIRES on the right symbol, not
    whether it can also regress the box, so no predicted box size is used.
    """
    out = []
    for gray, gts in sheets:
        H, W = gray.shape
        best = None
        for k in kernels:
            for sc in SCALES:
                kk = cv2.resize(k, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                if kk.shape[0] >= H or kk.shape[1] >= W:
                    continue
                r = cv2.matchTemplate(gray, kk, cv2.TM_CCOEFF_NORMED)
                r = cv2.resize(r, (W, H), interpolation=cv2.INTER_LINEAR)
                best = r if best is None else np.maximum(best, r)
        peaks = []
        if best is not None:
            supp = np.zeros((H, W), bool)
            for idx in np.argsort(-best.ravel()):
                v = float(best.flat[idx])
                if v < NMS_MIN_THR or len(peaks) >= NMS_MAX:
                    break
                y, x = divmod(int(idx), W)
                if supp[y, x]:
                    continue
                peaks.append((v, x, y))
                supp[max(0, y - NMS_R):y + NMS_R + 1, max(0, x - NMS_R):x + NMS_R + 1] = True
        out.append((peaks, gts))
    return out


def score(collected, thr):
    tp = fp = fn = 0
    for peaks, gts in collected:
        used = [False] * len(gts)
        for v, x, y in peaks:
            if v < thr:
                continue
            hit = -1
            for gi, (gx1, gy1, gx2, gy2) in enumerate(gts):
                if used[gi]:
                    continue
                if gx1 - 4 <= x <= gx2 + 4 and gy1 - 4 <= y <= gy2 + 4:
                    hit = gi
                    break
            if hit >= 0:
                used[hit] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), fp, tp + fn


def main() -> None:
    normals, patches, srcs = [], [], []
    by_case = {}
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        src = source_of(c.json_path)
        gnds = [v for v in (doc.get("components") or {}).values()
                if str(v.get("type", "")).strip() == "gnd"]
        if not gnds:
            continue
        try:
            g = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        h = g.shape[0]
        boxes = []
        for comp in gnds:
            b = comp.get("bbox")
            if not b or len(b) != 4:
                continue
            n, p = grab(g, b)
            if n is None:
                continue
            normals.append(n)
            patches.append(p)
            srcs.append(src)
            x1, y1, x2, y2 = [float(v) for v in b]
            boxes.append((x1, h - y2, x2, h - y1))
        if boxes:
            by_case[c.case_id] = (g, boxes, src)

    X = np.stack(normals).reshape(len(normals), -1).astype(np.float64)
    print(f"gnd instances: {len(X)}   sheets: {len(by_case)}")
    print(f"source mix: {dict(Counter(srcs))}")

    print("\n=== part 1: clustering the shapes (32x32, zero-mean unit-var) ===")
    print(f"{'k':>3} {'inertia':>10} {'silhouette':>11}")
    chosen = None
    for k in (2, 3, 4, 5, 6, 7, 8, 10):
        lab, C, inr = kmeans(X, k)
        sil = silhouette(X, lab, sample=500)
        print(f"{k:>3} {inr:>10.0f} {sil:>11.3f}")
        if chosen is None or sil > chosen[0]:
            chosen = (sil, k, lab, C)
    sil, K, lab, C = chosen
    print(f"\nchosen k = {K} (best silhouette {sil:.3f})")

    clusters = []
    for j in range(K):
        idx = np.where(lab == j)[0]
        if len(idx) == 0:
            clusters.append([])
            continue
        d = ((X[idx] - C[j]) ** 2).sum(1)
        order = idx[np.argsort(d)]
        clusters.append([(patches[i], srcs[i]) for i in order])
        print(f"  mode {j}: n={len(idx):4d} ({len(idx)/len(X)*100:4.1f}%)  "
              f"medoid src={srcs[order[0]]:17s}  mix={dict(Counter(srcs[i] for i in idx))}")

    out = contact(clusters, Path(r"Y:\workspace\EDA\work\preview\gnd_modes.png"))
    print(f"\ncontact sheet -> {out}")

    print("\n=== part 2: matched filter, one kernel vs one kernel per mode ===")
    print("  (all kernels are built the SAME way -- shape-normalised 32x32 centroid")
    print("   rescaled to 0-255 -- so the only difference is how many of them)")

    def to_u8(vec):
        v = vec.reshape(TILE, TILE).astype(np.float64)
        return ((v - v.min()) / (np.ptp(v) + 1e-6) * 255.0).astype(np.uint8)

    def trim(g, pad=2):
        med = float(np.median(g))
        m = g < med - 8
        ys, xs = np.where(m)
        if len(xs) == 0:
            return g
        y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + 1 + pad)
        return g[y0:y1, x0:x1]

    glob_patch = cv2.resize(to_u8(C.mean(0)), (22, 22), interpolation=cv2.INTER_AREA)
    mode_flat = [cv2.resize(to_u8(C[j]), (22, 22), interpolation=cv2.INTER_AREA)
                 for j in range(K) if clusters[j]]
    mode_trim = [trim(to_u8(C[j])) for j in range(K) if clusters[j]]

    items = sorted(by_case.items())[:MAX_SHEETS]
    sheets = [(g, b) for _, (g, b, _) in items]
    mix = Counter(s for _, (_, _, s) in items)
    print(f"\n  evaluating on {len(sheets)} sheets  {dict(mix)}")
    print(f"  GT gnd instances on those sheets: {sum(len(b) for _, b in sheets)}")
    print(f"  {'arm':<26} {'thr':>5} {'P':>8} {'R':>7} {'FP/sheet':>9}")

    arms = [("1 kernel (global mean)", [glob_patch]),
            (f"{len(mode_flat)} kernels, per mode", mode_flat),
            (f"{len(mode_trim)} kernels, trimmed", mode_trim)]
    for name, kernels in arms:
        col = collect(sheets, kernels)
        for thr in (0.45, 0.55, 0.65, 0.75):
            p, r, fp, n_gt = score(col, thr)
            print(f"  {name:<26} {thr:>5.2f} {p:>8.3f} {r:>7.3f} {fp/max(len(sheets),1):>9.1f}")
        print()

    print(f"  {'reference  YOLO gnd':<26} {'-':>5} {0.851:>8.3f} {0.762:>7.3f} {'-':>9}")


if __name__ == "__main__":
    main()
