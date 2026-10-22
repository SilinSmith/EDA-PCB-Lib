# -*- coding: utf-8 -*-
"""抽检 `work/_boxes_ocr.jsonl` 里 near 层的标签纯度：按角色分层取样本，拼一张放大对照图。

用法：python -X utf8 eda-pcb/tools/dev/_sheet_ocr.py --n 12 --tier near
输出：work/_ocr_sheet.png（每格 = 原图裁剪 ×4，格内有编号）+ 控制台表格。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[3]
CELL = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tier", default="near")
    ap.add_argument("--src", default="work/_boxes_ocr.jsonl")
    ap.add_argument("--out", default="work/_ocr_sheet.png")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (REPO / a.src).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["tier"] == a.tier]
    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    pick = []
    per = max(1, a.n // max(1, len(by_role)))
    for role, rs in sorted(by_role.items()):
        rs.sort(key=lambda r: (len(r["text"]) < 2, r["edit"], r["dist"]))
        pick += rs[:per]
    cols = min(4, len(pick))
    rows_n = (len(pick) + cols - 1) // cols
    sheet = np.full((rows_n * CELL, cols * CELL, 3), 240, np.uint8)
    from stage2.text import imread_unicode
    for i, r in enumerate(pick):
        img = imread_unicode(r["image"])
        x0, y0, x1, y1 = [int(v) for v in r["bbox"]]
        m = 3
        crop = img[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        s = min(4.0, (CELL - 12) / max(1, h), (CELL - 12) / max(1, w))
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_NEAREST)
        rr, cc = divmod(i, cols)
        oy, ox = rr * CELL + 6, cc * CELL + 6
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.putText(sheet, str(i), (ox, oy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
        print("  %2d %-10s %-11s GT=%-10r edit=%d dist=%s" % (
            i, r["case"], r["role"], r["text"], r["edit"], r["dist"]))
    cv2.imencode(".png", sheet)[1].tofile(str(REPO / a.out))
    print("->", REPO / a.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
