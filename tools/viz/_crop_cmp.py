# -*- coding: utf-8 -*-
"""裁片对照表：同一个真值框，三种裁法放大后并排 + 各自读出来的串。

每行 = 一个真值文字框：
    左：真值紧框原样      中：旧 textbox + 外扩      右：新 textseg + 外扩
右边列写出 GT 串与三次读数，读对画绿、读错画红。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_crop_cmp.py 0113 [--n 40] [--h 34]
    产物 -> Y:\\workspace\\EDA\\work\\preview\\cropcmp_<case>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import TextEngine, imread_unicode, pad_box  # noqa: E402
from _textseg_eval import ROLE_OF, cases, iou_matrix  # noqa: E402

H_CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
             resid_min_h=4, resid_min_dim=1)
GREEN = (40, 160, 40)
RED = (40, 40, 200)
GREY = (120, 120, 120)


def asc(s):
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)


def crop_to(img, box, h):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = img.shape[:2]
    pad = 4
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.full((h, h), 255, np.uint8)
    c = img[y1:y2, x1:x2]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    w = max(4, int(round(h * g.shape[1] / max(1, g.shape[0]))))
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="0113")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--h", type=int, default=34)
    args = ap.parse_args(argv)

    png = js = None
    for p, j in cases(0):
        if p.parent.name == args.case:
            png, js = p, j
    if png is None:
        print("找不到 case"); return 1
    bgr = imread_unicode(png)
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    gts = []
    for t in (d.get("texts") or {}).values():
        loc = t.get("location") or {}
        c = (t.get("content") or "").strip()
        role = ROLE_OF.get(str(t.get("details") or ""))
        if not role or not c or len(c) > 24 or not c.isascii():
            continue
        b = tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2"))
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            continue
        gts.append((b, c, role))
    old = [tuple(float(v) for v in b) for b in textbox.detect_boxes(bgr)]
    new = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **H_CFG)]
    rr = TextEngine().recognizer()
    # 只挑「old 和 H 都命中」的框，这样三种裁法可以直接对比
    keep = []
    for b, c, role in gts:
        Mo = iou_matrix([b], old)[0] if old else np.zeros(0)
        Mn = iou_matrix([b], new)[0] if new else np.zeros(0)
        if (Mo.size and Mo.max() >= 0.05) and (Mn.size and Mn.max() >= 0.05):
            keep.append((b, c, role))
    gts = keep[:args.n]
    print(f"候选 {len(keep)} 行，取 {len(gts)}")

    rows = []
    for b, c, role in gts:
        Mo = iou_matrix([b], old)[0] if old else np.zeros(0)
        Mn = iou_matrix([b], new)[0] if new else np.zeros(0)
        bo = old[int(np.argmax(Mo))] if Mo.size else None
        bn = new[int(np.argmax(Mn))] if Mn.size else None
        pair = [(("gt", b), "真值紧框"), ]
        arms = []
        for name, bx in (("old", bo), ("new", bn)):
            if bx is None:
                arms.append((name, None, ""))
                continue
            arms.append((name, pad_box(bx), ""))
        # 读
        outs = rr.read_many(bgr, [(b, role)] + [(a[1], role) for a in arms if a[1]])
        texts = [outs[0][0] or ""]
        k = 1
        for i, a in enumerate(arms):
            if a[1] is None:
                continue
            texts.append(outs[k][0] or "")
            k += 1
        i = 1
        fixed = []
        for a in arms:
            if a[1] is None:
                fixed.append((a[0], None, ""))
            else:
                fixed.append((a[0], a[1], texts[i])); i += 1
        rows.append((b, c, role, texts[0], fixed))

    H = args.h
    hdr = 24
    lineh = H + 8
    W = 900
    img = np.full((hdr + lineh * len(rows), W, 3), 255, np.uint8)
    cv2.putText(img, f"{args.case}  每行左起: [真值紧框] [旧+pad] [新+pad]   "
                     f"右列: 真值 / 三次读数(绿=对 红=错)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for r, (b, c, role, gtread, fixed) in enumerate(rows):
        y = hdr + r * lineh
        x = 6
        for lbl, bx, _t in [("gt", b, gtread)] + [(f[0], f[1], f[2]) for f in fixed]:
            if bx is None:
                x += 60
                continue
            g = crop_to(bgr, bx, H)
            if y + H <= img.shape[0] and x + g.shape[1] <= W:
                img[y:y + H, x:x + g.shape[1]] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img, (x - 1, y - 1), (x + g.shape[1], y + H),
                              GREY, 1)
            x += g.shape[1] + 10
        tx = 330
        cv2.putText(img, f"{role[:10]:10s}", (tx - 120, y + H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, asc(c)[:16], (tx, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"gt={asc(gtread)[:14]}", (tx, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    GREEN if gtread == c else RED, 1, cv2.LINE_AA)
        for j, (nm, _bx, t) in enumerate(fixed):
            cv2.putText(img, f"{nm}={asc(t)[:12]}", (tx + 150 + j * 175, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        GREEN if t == c else RED, 1, cv2.LINE_AA)
    out = ROOT / "work" / "preview" / f"cropcmp_{args.case}.png"
    cv2.imwrite(str(out), img)
    print(f"{len(rows)} 行 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
