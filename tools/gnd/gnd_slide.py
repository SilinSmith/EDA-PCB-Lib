"""Does a SLIDING GND template (matched filter) find gnd symbols?

The first probe compared pre-cropped, resized patches -- that conflates shape
with crop-window alignment, and a real matched filter slides, so alignment is
free.  This is the correct experiment: convolve a gnd template over the sheet
at several scales, then ask
  recall   = fraction of GT gnd boxes whose neighbourhood contains a peak
  #false   = local maxima above threshold that are far from any GT gnd
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
SCALES = [0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75]
OUT = Path(r"Y:\workspace\EDA\work\preview")
KERNEL_DIR = OUT / "gnd_kernels"
MIN_SIDE, MAX_SIDE = 11.0, 44.0


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def read_gray(p) -> np.ndarray:
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def gt_boxes(doc, h):
    """gnd boxes in image (top-left) coords."""
    out = []
    for comp in (doc.get("components") or {}).values():
        if str(comp.get("type", "")).strip() != "gnd":
            continue
        b = comp.get("bbox")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        iy1, iy2 = h - y2, h - y1
        x1, x2 = sorted((x1, x2))
        iy1, iy2 = sorted((iy1, iy2))
        if not (MIN_SIDE <= max(x2 - x1, iy2 - iy1) <= MAX_SIDE):
            continue
        out.append((x1, iy1, x2, iy2))
    return out


def clean_kernel(gray, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    sub = gray[max(0, y1):y2, max(0, x1):x2]
    if sub.size == 0 or min(sub.shape) < 8:
        return None
    return sub


def response(gray, kernel):
    """max-over-scales TM_CCOEFF_NORMED, aligned back to the base grid."""
    best = np.full(gray.shape, -1.0, dtype=np.float32)
    kh, kw = kernel.shape
    for s in SCALES:
        nw, nh = max(8, int(round(kw * s))), max(8, int(round(kh * s)))
        if nh >= gray.shape[0] or nw >= gray.shape[1]:
            continue
        k = cv2.resize(kernel, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if k.std() < 1e-6:
            continue
        r = cv2.matchTemplate(gray, k, cv2.TM_CCOEFF_NORMED)
        # place response at the kernel centre
        oy, ox = nh // 2, nw // 2
        h, w = r.shape
        ys = np.arange(h) + oy
        xs = np.arange(w) + ox
        ok = (ys >= 0) & (ys < gray.shape[0])
        ox_ok = (xs >= 0) & (xs < gray.shape[1])
        if not ok.any() or not ox_ok.any():
            continue
        canvas = np.full(gray.shape, -1.0, dtype=np.float32)
        canvas[np.ix_(ys[ok], xs[ox_ok])] = r[np.ix_(np.arange(h)[ok], np.arange(w)[ox_ok])]
        np.maximum(best, canvas, out=best)
    return best


def peaks(score, thr, radius):
    """local maxima above thr; radius = suppression window."""
    k = max(3, int(radius) | 1)
    mx = cv2.dilate(score, np.ones((k, k), np.uint8))
    is_pk = (score >= thr) & (score >= mx - 1e-6)
    ys, xs = np.nonzero(is_pk)
    return ys, xs


def main() -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    cases = S.discover(S.DEFAULT_DATA_ROOT)

    by_src: dict[str, list] = defaultdict(list)
    for c in cases:
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        if doc.get("components"):
            by_src[source_of(c.json_path)].append((c, doc))

    for src in SRC_ORDER:
        lst = by_src.get(src) or []
        if not lst:
            continue
        # --- pick the cleanest kernel: the gnd crop with the most ink -------
        best_k, best_ink, best_name = None, -1.0, ""
        for c, doc in lst[:6]:
            g = read_gray(c.png_path)
            if g is None:
                continue
            for b in gt_boxes(doc, g.shape[0]):
                k = clean_kernel(g, b)
                if k is None:
                    continue
                ink = float((k < 160).mean())
                if 0.25 <= ink <= 0.85 and ink > best_ink:
                    best_k, best_ink, best_name = k, ink, f"{Path(c.png_path).stem}"
        if best_k is None:
            print(f"{src}: no usable kernel")
            continue
        cv2.imwrite(str(KERNEL_DIR / f"{src.replace(' ','_')}.png"), best_k)

        # --- evaluate on held-out sheets of the same source -----------------
        rec, fpc, ns = [], [], 0
        for c, doc in lst:
            if Path(c.png_path).stem in best_name:
                continue
            g = read_gray(c.png_path)
            if g is None:
                continue
            gtb = gt_boxes(doc, g.shape[0])
            if len(gtb) < 4:
                continue
            ns += 1
            sc = response(g, best_k)
            kh = max(best_k.shape)
            rad = max(6, kh // 2)

            def near_gt(y, x) -> bool:
                for (x1, y1, x2, y2) in gtb:
                    if x1 - rad <= x <= x2 + rad and y1 - rad <= y <= y2 + rad:
                        return True
                return False

            # detection = peak above thr with at least one GT box nearby
            for thr in (0.35, 0.45, 0.55, 0.65):
                ys, xs = peaks(sc, thr, rad)
                hit = np.zeros(len(gtb), dtype=bool)
                fp = 0
                for y, x in zip(ys.tolist(), xs.tolist()):
                    matched = False
                    for i, (x1, y1, x2, y2) in enumerate(gtb):
                        if x1 - rad <= x <= x2 + rad and y1 - rad <= y <= y2 + rad:
                            hit[i] = True
                            matched = True
                    if not matched:
                        fp += 1
                rec.append((thr, float(hit.mean()), fp, len(gtb)))
            if ns >= 8:
                break
        if not rec:
            continue
        print(f"\n=== {src}  (kernel {best_name[:44]}, {ns} sheets) ===")
        print(f"{'thr':>6s} {'recall':>8s} {'FP/sheet':>9s} {'FP per GT':>10s}")
        for thr in (0.35, 0.45, 0.55, 0.65):
            rows = [r for r in rec if r[0] == thr]
            if not rows:
                continue
            R = float(np.mean([r[1] for r in rows]))
            F = float(np.sum([r[2] for r in rows]))
            G = float(np.sum([r[3] for r in rows]))
            print(f"{thr:6.2f} {R:8.3f} {F/len(rows):9.2f} {F/max(G,1):10.3f}")


if __name__ == "__main__":
    main()
