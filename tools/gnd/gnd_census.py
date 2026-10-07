"""Census of GND glyph FORMS in the real corpus.

Question: is there a small set of structural grammars -- "a horizontal bar, then
a shorter one, then a shorter one" (the user's proposal), or a triangle -- that
covers the gnd instances actually present, across KiCad / Altium / Datasheet / jlc?

For every GT gnd box we cut the crop, take the per-row ink width profile, split
it into horizontal runs separated by blank rows, and classify:
  bars_desc   : 2..4 runs, widths strictly decreasing top->bottom   (ground stack)
  bars_bottom : same but reversed (upside-down / rotated 180)
  tri_down    : one run, width rises to a max then falls (filled triangle)
  bar_single  : 1..2 runs of roughly equal width
  other       : anything else
Also reports whether the run centres are horizontally aligned, and the
vertical gaps, since those are the extra constraints that kill false positives.
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

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
OUT = Path(r"Y:\workspace\EDA\work\preview")
MIN_SIDE, MAX_SIDE = 9.0, 46.0
PAD = 2
INK = 170


def source_of(p) -> str:
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def crop(g, box):
    x1, y1, x2, y2 = [float(v) for v in box]
    h = g.shape[0]
    iy1, iy2 = h - y2, h - y1
    x1, x2 = sorted((x1, x2))
    iy1, iy2 = sorted((iy1, iy2))
    if not (MIN_SIDE <= max(x2 - x1, iy2 - iy1) <= MAX_SIDE):
        return None
    x1, iy1 = int(round(x1)) - PAD, int(round(iy1)) - PAD
    x2, iy2 = int(round(x2)) + PAD, int(round(iy2)) + PAD
    sub = g[max(0, iy1):min(h, iy2), max(0, x1):min(g.shape[1], x2)]
    if sub.size == 0 or min(sub.shape) < 6:
        return None
    return sub


def runs_of(prof: np.ndarray, min_w: float = 1.5):
    """split a row-width profile into runs separated by blank rows"""
    out, cur = [], None
    for i, w in enumerate(prof.tolist()):
        if w >= min_w:
            if cur is None:
                cur = [i, i, [w], []]
            else:
                cur[1] = i
                cur[2].append(w)
        else:
            if cur is not None:
                out.append(cur)
                cur = None
    if cur is not None:
        out.append(cur)
    res = []
    for a, b, ws, _ in out:
        ws = np.asarray(ws, dtype=float)
        res.append({"r0": a, "r1": b, "peak": float(ws.max()),
                    "mean": float(ws.mean()), "rows": b - a + 1})
    return res


def classify(prof: np.ndarray):
    rs = [r for r in runs_of(prof) if r["peak"] >= 2.5]
    n = len(rs)
    if n == 0:
        return "empty", rs, False
    peaks = [r["peak"] for r in rs]
    # triangle: single run whose width rises then falls
    if n == 1:
        seg = prof[rs[0]["r0"]:rs[0]["r1"] + 1]
        if len(seg) >= 5:
            k = int(np.argmax(seg))
            left = seg[:k + 1]
            right = seg[k:]
            if (len(left) < 2 or left[-1] >= left[0]) and (len(right) < 2 or right[0] >= right[-1]):
                return "tri", rs, False
        return "bar_single", rs, False
    desc = all(peaks[i] > peaks[i + 1] * 1.02 for i in range(n - 1))
    asc = all(peaks[i] < peaks[i + 1] * 0.98 for i in range(n - 1))
    if 2 <= n <= 5 and desc:
        return "bars_desc", rs, False
    if 2 <= n <= 5 and asc:
        return "bars_asc", rs, False
    if n <= 2:
        return "bar_single", rs, False
    return "other", rs, False


def main() -> None:
    tally: Counter = Counter()
    per_src: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    maxhw: list[tuple[float, str]] = []

    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
            if not doc.get("components"):
                continue
        except Exception:
            continue
        g = None
        src = source_of(c.json_path)
        for comp in (doc.get("components") or {}).values():
            if str(comp.get("type", "")).strip() != "gnd":
                continue
            b = comp.get("bbox")
            if not b or len(b) != 4:
                continue
            if g is None:
                try:
                    g = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
                except Exception:
                    break
            sub = crop(g, b)
            if sub is None:
                continue
            ink = (sub < INK)
            prof = ink.sum(axis=1).astype(float)
            if prof.max() < 2.5:
                continue
            cls, rs, _ = classify(prof)
            tally[cls] += 1
            per_src[src][cls] += 1
            if len(samples[cls]) < 16:
                samples[cls].append(sub)
            if rs:
                w = max(r["peak"] for r in rs)
                hw = len(prof) / max(w, 1.0)
                if len(maxhw) < 4000:
                    maxhw.append((float(hw), cls))

    tot = sum(tally.values())
    print(f"=== gnd form census: {tot} instances ===")
    for k, v in tally.most_common():
        print(f"  {k:12s} {v:6d}  {v/tot*100:5.1f}%")

    print("\n=== by source (row %) ===")
    srcs = [s for s in SRC_ORDER if per_src.get(s)]
    print(f"{'class':12s}" + "".join(f"{s[:10]:>12s}" for s in srcs))
    for k, _ in tally.most_common():
        row = "".join(
            f"{per_src[s].get(k,0)/max(sum(per_src[s].values()),1)*100:11.1f}%" for s in srcs
        )
        print(f"{k:12s}{row}")

    print("\n=== shape stats by class (aspect h/w) ===")
    for k in tally:
        vals = [v for v, c in maxhw if c == k]
        if vals:
            print(f"  {k:12s} n={len(vals):5d} p10 {np.percentile(vals,10):.2f} "
                  f"p50 {np.percentile(vals,50):.2f} p90 {np.percentile(vals,90):.2f}")

    # contact sheet per class so we can verify the classifier is not lying
    rows = [(k, samples[k]) for k in tally if samples.get(k)]
    if rows:
        TILE = 40
        cols = max(len(v) for _, v in rows)
        W = cols * (TILE + 4) + 4
        H = len(rows) * (TILE + 22) + 4
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        for r, (k, ims) in enumerate(rows):
            y = 4 + r * (TILE + 22) + 14
            d.text((4, y - 13), f"{k} (n={tally[k]})", fill=(0, 0, 0))
            for i, im in enumerate(ims):
                t = Image.fromarray(im).convert("L").resize((TILE, TILE), Image.BILINEAR)
                sheet.paste(t.convert("RGB"), (4 + i * (TILE + 4), y))
        p = OUT / "gnd_forms.png"
        sheet.save(p)
        print(f"\nwrote {p}  {sheet.size[0]}x{sheet.size[1]}  rows={[k for k,_ in rows]}")


if __name__ == "__main__":
    main()
