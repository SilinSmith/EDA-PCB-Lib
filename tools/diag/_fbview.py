# -*- coding: utf-8 -*-
"""把「GT 有真名、我们没配上」的元件连同它的 GT 键一起画出来，人眼核对：
图上印的位号到底是不是 GT 那个键？

    python work\\tools\\diag\\_fbview.py --pred work\\_submit_out220y\\final_results --cases 0001,0020,0050 --n 6 --pad 70
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import cv2                                                # noqa: E402
import numpy as np                                        # noqa: E402
import schematic_parser.metrics as M                      # noqa: E402
import stage2_eval as SE                                  # noqa: E402
from stage2.text import imread_unicode                    # noqa: E402

IC = ("box", "boxic", "ic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pad", type=float, default=70.0)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--out", default=str(ROOT / "work" / "preview" / "_fbview.png"))
    a = ap.parse_args()

    root = Path(a.pred)
    s1dir = root.parent / "runs" / "stage1"
    ex, di = SE.build_index(SE.DEFAULT_DATA_ROOT)
    want = [c.strip() for c in a.cases.split(",") if c.strip()]

    files = sorted(root.glob("*_Pred.json"))
    tiles, labels = [], []
    for f in files:
        cid = f.stem.replace("_Pred", "")
        if want and cid not in want:
            continue
        s1f = s1dir / (cid + ".stage1.json")
        if not s1f.is_file():
            continue
        try:
            c, _ = SE.resolve(cid, ex, di)
        except Exception:
            continue
        g = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        p = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = g.get("components") or {}, p.get("components") or {}
        if not gc or not pc:
            continue
        _f1, km = M.assign_components(g, p, M.EvalConfig(), {})
        used = set(km.values())
        bgr = imread_unicode(c.png_path)
        H = float(bgr.shape[0])
        for gk in gc:
            if len(tiles) >= a.n:
                break
            if gk in used or str(gk).startswith("\u2205"):
                continue
            gb = gc[gk].get("bbox")
            if not gb:
                continue
            gib = (gb[0], H - gb[3], gb[2], H - gb[1])
            pd = a.pad
            x0 = int(max(0, gib[0] - pd)); y0 = int(max(0, gib[1] - pd))
            x1 = int(min(bgr.shape[1], gib[2] + pd)); y1 = int(min(bgr.shape[0], gib[3] + pd))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            im = bgr[y0:y1, x0:x1].copy()
            cv2.rectangle(im, (int(gib[0] - x0), int(gib[1] - y0)),
                          (int(gib[2] - x0), int(gib[3] - y0)), (0, 0, 255), 2)
            sc = a.scale
            im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            hdr = np.full((30, im.shape[1], 3), 255, np.uint8)
            cv2.putText(hdr, "%s  GT:%s  type=%s" % (cid, gk, M.canon_type(gc[gk].get("type"))),
                        (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA)
            tiles.append(np.vstack([hdr, im]))
        if len(tiles) >= a.n:
            break

    if not tiles:
        print("没有可画的元件")
        return
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    cw = max(t.shape[1] for t in tiles) + 10
    ch = max(t.shape[0] for t in tiles) + 10
    out = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        out[r * ch:r * ch + t.shape[0], cc * cw:cc * cw + t.shape[1]] = t
    cv2.imwrite(a.out, out)
    print("saved", a.out, out.shape)


if __name__ == "__main__":
    main()
