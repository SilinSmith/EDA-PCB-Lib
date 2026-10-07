"""Structural GND grammar detector: "a bar, a shorter bar below, a shorter one
below that" (the user's proposal), tested against GT gnd on whole sheets.

Discriminators that a raw NCC mask cannot express but a grammar can:
  * each bar is an isolated short horizontal component (a long wire is not a bar)
  * the stack's bars are centred on a common vertical axis
  * widths strictly decrease
  * vertical gaps are small and roughly equal
  * the bottom bar has nothing below it (the symbol terminates)
Rotations: 0 / 90 / 180 / 270 are all tested.

Output: recall of GT gnd, and false positives per sheet, so it is directly
comparable to the matched-filter numbers (best case R 0.607 @ 4.5 FP/sheet).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
OUT = Path(r"Y:\workspace\EDA\work\preview")
INK = 170
MIN_SIDE, MAX_SIDE = 9.0, 46.0
BAR_MIN_W, BAR_MAX_W, BAR_MAX_H = 4, 72, 7
MAX_GAP, MIN_GAP = 14, 1
CENTER_TOL = 0.30          # fraction of the wider bar


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def bars_of(mask: np.ndarray):
    """isolated short horizontal components -> [(x0,y0,x1,y1,w,h,cx,cy)]"""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if h > BAR_MAX_H or not (BAR_MIN_W <= w <= BAR_MAX_W):
            continue
        if area < 0.55 * w * h:          # must be a solid-ish bar, not a thin ring
            continue
        out.append((x, y, x + w, y + h, float(w), float(h), cen[i][0], cen[i][1]))
    return out


def stacks(bars, ascending: bool):
    """chain bars vertically with decreasing (or increasing) width"""
    if len(bars) < 2:
        return []
    bars = sorted(bars, key=lambda b: b[1])
    res = []
    for i, b in enumerate(bars):
        chain = [b]
        cur = b
        for j in range(i + 1, len(bars)):
            c = bars[j]
            gap = c[1] - (cur[1] + cur[5])
            if gap > MAX_GAP:
                break
            if gap < MIN_GAP:
                continue
            bigger = cur[4] if not ascending else c[4]
            smaller = c[4] if not ascending else cur[4]
            if smaller >= bigger * 0.96:            # must shrink by >=4%
                continue
            wider = max(cur[4], c[4])
            if abs(c[6] - cur[6]) > CENTER_TOL * wider:
                continue
            chain.append(c)
            cur = c
        if len(chain) >= 2:
            xs0 = min(c[0] for c in chain); ys0 = min(c[1] for c in chain)
            xs1 = max(c[2] for c in chain); ys1 = max(c[3] for c in chain)
            res.append((xs0, ys0, xs1, ys1, len(chain)))
    return res


def detect(gray: np.ndarray):
    mask = gray < INK
    cands = []
    for asc in (False, True):
        b = bars_of(mask)
        cands += stacks(b, asc)
        # 90-degree rotation
        bt = [(y, x, y + h, x + w, h, w, cy, cx) for (x, y, x1, y1, w, h, cx, cy) in b]
        cands += [tuple(v) for v in stacks(bt, asc)]
    return cands


def dedup(boxes, iou_thr=0.35):
    boxes = sorted(boxes, key=lambda b: -(b[4] if len(b) > 4 else 1))
    keep = []
    for b in boxes:
        x0, y0, x1, y1 = b[:4]
        ok = True
        for k in keep:
            ix = max(0, min(x1, k[2]) - max(x0, k[0]))
            iy = max(0, min(y1, k[3]) - max(y0, k[1]))
            inter = ix * iy
            u = (x1 - x0) * (y1 - y0) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if u > 0 and inter / u > iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def gt_gnd(doc, h):
    out = []
    for comp in (doc.get("components") or {}).values():
        if str(comp.get("type", "")).strip() != "gnd":
            continue
        b = comp.get("bbox")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        iy1, iy2 = h - y2, h - y1
        x1, x2 = sorted((x1, x2)); iy1, iy2 = sorted((iy1, iy2))
        out.append((x1, iy1, x2, iy2))
    return out


def hit(a, b, pad=6.0):
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def main() -> None:
    per_src = defaultdict(lambda: [0, 0, 0, 0])   # tp, gt, fp, sheets
    samples = []
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
            if not doc.get("components"):
                continue
        except Exception:
            continue
        g = cv2.imread(str(c.png_path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        gtb = [b for b in gt_gnd(doc, g.shape[0]) if MIN_SIDE <= max(b[2]-b[0], b[3]-b[1]) <= MAX_SIDE]
        if len(gtb) < 3:
            continue
        det = dedup([b for b in detect(g) if MIN_SIDE <= max(b[2]-b[0], b[3]-b[1]) <= MAX_SIDE])
        src = source_of(c.json_path)
        used = set()
        tp = 0
        for d in det:
            for i, gt in enumerate(gtb):
                if i in used:
                    continue
                if hit(d, gt):
                    used.add(i); tp += 1; break
        fp = len(det) - tp
        r = per_src[src]
        r[0] += tp; r[1] += len(gtb); r[2] += fp; r[3] += 1
        if len(samples) < 90:
            samples.append((src, tp, len(gtb), fp, c, gtb, det))

    print("=== structural bar-stack detector vs GT gnd (whole sheets) ===")
    print(f"{'source':18s} {'sheets':>7s} {'TP':>6s} {'GT':>6s} {'FP':>7s} "
          f"{'recall':>8s} {'prec':>7s} {'FP/sheet':>9s}")
    tp = gt = fp = sh = 0
    for s in SRC_ORDER + ["unknown"]:
        if s not in per_src:
            continue
        t, gg, f, n = per_src[s]
        tp += t; gt += gg; fp += f; sh += n
        print(f"{s:18s} {n:7d} {t:6d} {gg:6d} {f:7d} "
              f"{t/max(gg,1):8.3f} {t/max(t+f,1):7.3f} {f/max(n,1):9.2f}")
    print(f"{'ALL':18s} {sh:7d} {tp:6d} {gt:6d} {fp:7d} "
          f"{tp/max(gt,1):8.3f} {tp/max(tp+fp,1):7.3f} {fp/max(sh,1):9.2f}")

    # overlay a few sheets so we can see what it is actually firing on
    show = [s for s in samples if s[3] > 0][:6]
    if show:
        tiles = []
        for src, t, gg, f, c, gtb, det in show:
            g = cv2.imread(str(c.png_path), cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            vis = cv2.resize(vis, (760, int(760 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
            sx = 760 / g.shape[1]; sy = vis.shape[0] / g.shape[0]
            for (x0, y0, x1, y1) in gtb:
                cv2.rectangle(vis, (int(x0*sx), int(y0*sy)), (int(x1*sx), int(y1*sy)), (0, 200, 0), 1)
            for d in det:
                cv2.rectangle(vis, (int(d[0]*sx), int(d[1]*sy)), (int(d[2]*sx), int(d[3]*sy)), (0, 0, 255), 1)
            cv2.putText(vis, f"{src} TP{t}/{gg} FP{f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            tiles.append(vis)
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) for t in tiles]
        rows = [np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]
        w = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for r in rows]
        p = OUT / "gnd_grammar.png"
        cv2.imwrite(str(p), np.vstack(rows))
        print(f"\nwrote {p}   green = GT gnd, red = detector")


if __name__ == "__main__":
    main()
