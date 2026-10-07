"""Add geometric context to the per-mode GND matched filter and see if the
precision problem (P ~ 0.01 at every threshold) is actually fixable.

Two contextual gates, both cheap and both motivated by how a real ground symbol
sits on a sheet:

  WIRE-END  a ground symbol terminates a wire.  A point in the MIDDLE of a wire
            has thin-stroke ink in two OPPOSITE directions; a wire end has ink
            only in one contiguous arc.  So: fire only where the thin-stroke
            mask looks like an end (no opposite pair, few directions set).
  ISOLATED  a ground symbol is a small compact glyph.  A point on a long wire
            has ink stretching far in the window.  So: fire only where the ink
            fraction in a large window is low.

Arms:  none / wire-end / isolated / both.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S
from gnd_modes import SCALES, TILE, grab, kmeans, source_of

K_CLUSTERS = 10
NMS_MIN_THR = 0.35
NMS_R = 8
NMS_MAX = 3000
MAX_SHEETS = 120
R_DIR = 6
ISO_WIN = 81


def to_u8(vec):
    v = vec.reshape(TILE, TILE).astype(np.float64)
    return ((v - v.min()) / (np.ptp(v) + 1e-6) * 255.0).astype(np.uint8)


def context_maps(gray):
    """(end_like bool map, ink fraction in an ISO_WIN box)."""
    inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    ink = (inv > 0)
    thick = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thin = (ink & (thick == 0)).astype(np.uint8)

    offs = []
    for d in range(8):
        a = np.deg2rad(d * 45.0)
        offs.append((int(round(R_DIR * np.cos(a))), int(round(R_DIR * np.sin(a)))))
    hits = []
    for dx, dy in offs:
        hits.append(np.roll(np.roll(thin, -dy, axis=0), -dx, axis=1).astype(bool))
    n_dir = np.zeros_like(thin, dtype=np.uint8)
    opposite = np.zeros_like(thin, dtype=bool)
    for d in range(8):
        n_dir += hits[d].astype(np.uint8)
        opposite |= (hits[d] & hits[(d + 4) % 8])
    end_like = (thin > 0) & (n_dir <= 3) & (~opposite)

    k = np.ones((ISO_WIN, ISO_WIN), np.float32) / float(ISO_WIN * ISO_WIN)
    iso = cv2.filter2D(ink.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
    return end_like, iso, thin


def main() -> None:
    normals, by_case = [], {}
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
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
            n, _ = grab(g, b)
            if n is None:
                continue
            normals.append(n)
            x1, y1, x2, y2 = [float(v) for v in b]
            boxes.append((x1, h - y2, x2, h - y1))
        if boxes:
            by_case[c.case_id] = (g, boxes, source_of(c.json_path))

    X = np.stack(normals).reshape(len(normals), -1).astype(np.float64)
    lab, C, _ = kmeans(X, K_CLUSTERS)
    kernels = [cv2.resize(to_u8(C[j]), (22, 22), interpolation=cv2.INTER_AREA)
               for j in range(K_CLUSTERS)]
    print(f"gnd instances {len(X)}  sheets {len(by_case)}  kernels {len(kernels)}")

    items = sorted(by_case.items())[:MAX_SHEETS]
    records = []
    for cid, (gray, gts, src) in items:
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
                supp[max(0, y - NMS_R):y + NMS_R + 1,
                     max(0, x - NMS_R):x + NMS_R + 1] = True
        end_like, iso, _ = context_maps(gray)
        # dilate the end map so a peak just off the wire tip still counts
        end_d = cv2.dilate(end_like.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        records.append((peaks, gts, end_d, iso))
        print(f"  {cid:<28} peaks {len(peaks):5d}  gnd {len(gts):3d}  "
              f"endpx {int(end_like.sum()):7d}  src {src}")

    print(f"\nsheets {len(records)}   GT gnd {sum(len(r[1]) for r in records)}")

    def score(thr, need_end, iso_thr):
        tp = fp = fn = 0
        for peaks, gts, end_d, iso in records:
            used = [False] * len(gts)
            for v, x, y in peaks:
                if v < thr:
                    continue
                if need_end and not end_d[y, x]:
                    continue
                if iso_thr is not None and iso[y, x] > iso_thr:
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
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1), fp

    n = len(records)
    print(f"\n{'gate':<34} {'thr':>5} {'P':>8} {'R':>7} {'FP/sheet':>9}")
    for name, ne, it in [("none (baseline)", False, None),
                         ("wire-end only", True, None),
                         ("isolated only (iso<=0.12)", False, 0.12),
                         ("wire-end + isolated", True, 0.12)]:
        for thr in (0.55, 0.65, 0.75):
            p, r, fp = score(thr, ne, it)
            print(f"  {name:<32} {thr:>5.2f} {p:>8.3f} {r:>7.3f} {fp/n:>9.1f}")
        print()
    print(f"  {'reference  YOLO gnd':<32} {'-':>5} {0.851:>8.3f} {0.762:>7.3f} {'-':>9}")


if __name__ == "__main__":
    main()
