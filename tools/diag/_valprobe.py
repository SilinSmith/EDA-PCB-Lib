# -*- coding: utf-8 -*-
"""把某张图某个 GT 元件附近的 stage1 文字框全部列出来，并当场 OCR，
看「值」到底有没有被检出、被读成什么。

用法：python work\tools\diag\_valprobe.py --case 0001 --key C12 --pad 120
      python work\tools\diag\_valprobe.py --case 0001 --all-missing --limit 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))
sys.path.insert(0, str(REPO / "work" / "tools" / "viz"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from stage2.text import TextEngine, pad_box  # noqa: E402
from _crop_one import find_png  # noqa: E402


def s1_path(case):
    d = REPO / "work" / "stage1_v6"
    for p in d.glob("%s*.stage1.json" % case):
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--all-missing", action="store_true")
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v13"))
    ap.add_argument("--pad", type=float, default=120)
    ap.add_argument("--limit", type=int, default=6)
    a = ap.parse_args()

    import json
    import numpy as np
    import cv2
    s1p = s1_path(a.case)
    s1 = json.loads(s1p.read_text(encoding="utf-8"))
    H = s1["image"]["h"]
    gt = load_json(next(Path(DEFAULT_DATA_ROOT).rglob("%s*_target_named.json" % a.case)))
    gc = gt.get("components") or {}
    pred = json.loads((Path(a.pred) / ("%s.json" % a.case)).read_text(encoding="utf-8"))
    pc = pred.get("components") or {}

    keys = [a.key] if a.key else []
    if a.all_missing:
        for k in gc:
            if k in pc and (gc[k].get("type") in ("c", "r", "l", "d")) \
                    and gc[k].get("value") and not pc[k].get("value"):
                keys.append(k)
    keys = keys[:a.limit]

    png = find_png(a.case)
    im = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_COLOR)
    eng = TextEngine()
    texts = s1.get("texts") or []

    for key in keys:
        g = gc[key]
        gb = g["bbox"]
        # GT 左下角 → 图像坐标
        ix1, ix2 = gb[0] - a.pad, gb[2] + a.pad
        iy1, iy2 = H - gb[3] - a.pad, H - gb[1] + a.pad
        sel = []
        for t in texts:
            b = t["bbox_img"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if ix1 <= cx <= ix2 and iy1 <= cy <= iy2:
                sel.append(b)
        print("\n=== %s  GT value=%r  type=%s  bbox=%s  附近文字框 %d ==="
              % (key, g.get("value"), g.get("type"), gb, len(sel)))
        if not sel:
            print("   （附近一个文字框都没有）")
            continue
        reads = eng.read(im, sel)
        for r, b in zip(reads, sel):
            d = min(max(abs((b[0] + b[2]) / 2 - (gb[0] + gb[2]) / 2),
                        abs((b[1] + b[3]) / 2 - (H - (gb[1] + gb[3]) / 2))), 9999)
            txt = " | ".join("%s=%r(%.2f)" % (k, v[0], v[1])
                             for k, v in (r["reads"] or {}).items() if v[0])
            print("   bbox=%-30s d=%6.1f  %s"
                  % ([round(x, 1) for x in b], d, txt or "(空)"))


if __name__ == "__main__":
    main()
