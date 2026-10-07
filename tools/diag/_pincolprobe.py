# -*- coding: utf-8 -*-
"""探针：把「框外号码窗」里到底有什么量化出来。

给一个 case + 元件，打印：
  * 元件 bbox（图像空间）；
  * GT 的引脚点（图像空间），按边分组；
  * stage1 的 `texts` 里落在每条边外侧带内的文字框（中心 + 尺寸）；
  * `pincol._side_positions` 的**原始**（未吸附）桩位置；
让「哪条边的信号来自什么」一目了然。

    python work\\tools\\diag\\_pincolprobe.py --case 0143 --key U1
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

from schematic_parser.data import discover, load_json          # noqa: E402
from stage1 import pincol                                      # noqa: E402
from stage2.text import imread_unicode                         # noqa: E402

DIRS = [ROOT / "work" / "_submit_out220r36" / "runs" / "stage1",
        ROOT / "work" / "_fusion_out220no4" / "runs" / "stage1"]


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="0143")
    ap.add_argument("--key", default="U1")
    ap.add_argument("--reach", type=float, default=40.0, help="框外搜索深度")
    ap.add_argument("--dir", default="")
    a = ap.parse_args(argv)

    hit = next((c for c in discover() if c.case_id == a.case and c.json_path), None)
    if hit is None:
        print("没找到 case"); return 1
    gt = load_json(hit.json_path)
    bgr = imread_unicode(hit.png_path)
    H = bgr.shape[0]
    sp = Path(a.dir) if a.dir else next((d / ("%s.stage1.json" % a.case) for d in DIRS
                                         if (d / ("%s.stage1.json" % a.case)).exists()), None)
    doc = json.loads(sp.read_text(encoding="utf-8"))
    texts = doc.get("texts") or []
    gv = (gt.get("components") or {}).get(a.key) or {}
    gj = gv.get("bbox") or [0, 0, 0, 0]
    gbb = (gj[0], H - gj[3], gj[2], H - gj[1])
    best, bi = 0.0, None
    for c in doc["components"]:
        v = iou(gbb, [float(x) for x in c["bbox_img"]])
        if v > best:
            best, bi = v, c
    bb = [float(x) for x in bi["bbox_img"]]
    x0, y0, x1, y1 = bb
    print("case %s %s  图 %dx%d" % (a.case, a.key, bgr.shape[1], H))
    print("stage1 框 %s type=%s IoU=%.3f  bbox_img=(%.1f,%.1f,%.1f,%.1f)"
          % (bi["id"], bi["type"], best, x0, y0, x1, y1))
    print("尺寸 %.0f x %.0f" % (x1 - x0, y1 - y0))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink_all = (gray < 170).astype(np.uint8)
    clean = pincol.text_free_ink(ink_all, texts, pad=0.0)

    gtpts = [((p["point"]["x"]), H - p["point"]["y"])
             for p in ((gt.get("pins") or {}).get(a.key) or {}).values()]
    print("GT 引脚 %d 个" % len(gtpts))

    # --- 原始桩（未吸附）
    print("\n【原始桩（未吸附、未补格点）】")
    for side in ("L", "R", "T", "B"):
        raw = pincol._side_positions(clean, bb, side, pincol.BAND,
                                     pincol.MIN_STUB, pincol.MIN_RUN)
        print("  %s: %d 个  %s" % (side, len(raw), [round(v, 1) for v in raw[:40]]))
        if not raw:
            continue
        d = np.diff(sorted(raw))
        if d.size:
            print("     相邻间距 %s" % [round(float(v), 1) for v in d[:40]])

    # --- 文字框：按「中心落在框外哪个带里」分派
    def cents():
        for i, t in enumerate(texts):
            b = t.get("bbox_img") if isinstance(t, dict) else t
            if not b:
                continue
            yield i, (float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0, \
                (float(b[2]) - float(b[0]), float(b[3]) - float(b[1]))

    for side, band in (("L", (x0 - a.reach, x0 - 1)),
                       ("R", (x1 + 1, x1 + a.reach)),
                       ("T", None), ("B", None)):
        if band is None:
            continue
        got = []
        for i, cx, cy, (w, h) in cents():
            if band[0] <= cx <= band[1] and y0 - 6 <= cy <= y1 + 6:
                got.append((round(cy, 1), round(cx, 1), int(w), int(h), i))
        got.sort()
        print("\n【%s 边外侧 %.0fpx 内的文字框】%d 个" % (side, a.reach, len(got)))
        for g in got:
            print("      cy=%7.1f cx=%7.1f  %2dx%-2d  text#%d" % g)

    # --- GT 引脚按边分组
    print("\n【GT 引脚（图像坐标）按边】")
    for side, test in (("L", lambda p: p[0] < (x0 + x1) / 2 and abs(p[0] - x0) < 60),
                       ("R", lambda p: abs(p[0] - x1) < 60 and p[0] >= (x0 + x1) / 2),
                       ("T", lambda p: p[1] < (y0 + y1) / 2 and abs(p[1] - y0) < 60),
                       ("B", lambda p: abs(p[1] - y1) < 60 and p[1] >= (y0 + y1) / 2)):
        sel = sorted((p for p in gtpts if test(p)), key=lambda p: p[1])
        print("  %s: %d 个 %s" % (side, len(sel),
                                 [(round(p[0], 1), round(p[1], 1)) for p in sel[:34]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
