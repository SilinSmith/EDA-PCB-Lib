# -*- coding: utf-8 -*-
"""某个真值文字框在 textseg 里的命运：属于哪个墨迹分量、为什么被丢。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_probe3.py 0113 299,89,312,102
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textseg  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _textseg_eval import cases  # noqa: E402


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cid = argv[0]
    box = tuple(float(v) for v in argv[1].split(","))
    cfg = {"ink_mode": "max", "min_dim": 2, "min_area": 4, "line_len": 15}
    png = None
    for p, _j in cases(0):
        if p.parent.name == cid:
            png = p
    bgr = imread_unicode(png)
    im = textseg.ink_mask(bgr, textseg.INK, cfg["ink_mode"])
    lines = textseg.line_mask(im, cfg["line_len"])
    n, lab, st, _ = cv2.connectedComponentsWithStats(im, connectivity=8)
    has_line = np.zeros(max(1, n), bool)
    if n > 1:
        hit = np.unique(lab[lines > 0])
        has_line[hit[hit > 0]] = True

    x1, y1, x2, y2 = (int(v) for v in box)
    sub = lab[y1:y2, x1:x2]
    cnt = Counter(int(v) for v in sub.ravel() if v > 0)
    print(f"{cid} img {bgr.shape[1]}x{bgr.shape[0]}  框 {box}")
    print(f"  框内墨迹像素 {sum(cnt.values())}  涉及分量 {len(cnt)} 个")
    for k, v in cnt.most_common(6):
        x, y, w, h, a = (int(t) for t in st[k])
        struct = has_line[k] and max(w, h) >= cfg.get("struct_dim", textseg.STRUCT_DIM)
        ok = textseg._glyph_ok(w, h, a, cfg["min_dim"], textseg.MIN_H, textseg.MAX_H,
                               textseg.MIN_W, textseg.MAX_W, cfg["min_area"])
        print(f"    分量{k}: box=({x},{y},{x+w},{y+h}) {w}x{h} area={a} "
              f"has_line={bool(has_line[k])} 结构={struct} glyph_ok={ok} 框内像素={v}")
        if struct:
            m = (lab == k)
            resid = np.uint8(np.bitwise_and(m, np.bitwise_not(lines > 0))) * 255
            n2, _l2, st2, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
            rows = []
            for i in range(1, n2):
                rx, ry, rw, rh, ra = (int(t) for t in st2[i])
                if rx + rw < x1 - 30 or rx > x2 + 30 or ry + rh < y1 - 30 or ry > y2 + 30:
                    continue
                rows.append((rx, ry, rw, rh, ra,
                             textseg._glyph_ok(rw, rh, ra, textseg.RESID_MIN_DIM,
                                               textseg.RESID_MIN_H, textseg.MAX_H,
                                               textseg.MIN_W, textseg.MAX_W,
                                               cfg["min_area"])))
            print(f"      残留碎片 {n2-1} 个，附近 {len(rows)} 个：")
            for r in rows[:8]:
                print(f"        ({r[0]},{r[1]}) {r[2]}x{r[3]} area={r[4]} ok={r[5]}")

    g = [b for b in textseg.glyphs(bgr, **cfg)]
    det = [b for b in textseg.detect_boxes(bgr, **cfg)]
    def near(bs, k=4):
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        bs = sorted(bs, key=lambda b: (0.5 * (b[0] + b[2]) - c[0]) ** 2
                    + (0.5 * (b[1] + b[3]) - c[1]) ** 2)
        return bs[:k]
    print(f"  glyphs {len(g)}  detect {len(det)}")
    print("  最近 glyphs:", [[round(v, 1) for v in b] for b in near(g)])
    print("  最近 detect:", [[round(v, 1) for v in b] for b in near(det)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
