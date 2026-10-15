# -*- coding: utf-8 -*-
"""§5.114 取证：队友黑白框实测校准（`recover_frame_pins`）为什么在 case N 上一条不取。

对某个 case 的 stage1 产物 + 原图：
  * 复刻 `pin_evidence._black_ink(bgr, ink)` 的掩膜（ink = gray < ray_mod.INK）；
  * 对每个 box/bigic 元件的四条边调用 `_frame_edge(mask, box, edge)`，
    打印 `confirmed` / `coverage` / `coord`（`_frame_edge` 的返回值）；
  * 再沿「框边所在的列/行」打印暗像素游程，看**画出来的框线**到底在哪一行/列，
    与 bbox 给的坐标差多少 ⇒ 判定「框确认失败」是不是因为 bbox 没落在框线上。

用法（cwd 任意，脚本自己找根）：
    python -X utf8 work\\_monodbg.py --case 8 [--run work/_fast_out_mono8] [--comp c0000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
import numpy as np  # noqa: E402
import cv2  # noqa: E402

from schematic_parser import data as data_mod  # noqa: E402
from stage1 import pin_evidence as ev  # noqa: E402
from stage1 import pintrace as ray_mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="8")
ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_mono8")
ap.add_argument("--comp", default=None)
ap.add_argument("--rows", type=int, default=6)
a = ap.parse_args()

run = Path(a.run)
s1 = json.loads((run / "runs" / "stage1" / (a.case + ".stage1.json")).read_text("utf-8"))
img_path = s1["image"]["path"]
bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ink = (gray < ray_mod.INK).astype(np.uint8)
mask, colored = ev._black_ink(bgr, ink)
print("图像 %s  h=%d w=%d  ray_mod.INK=%s" % (Path(img_path).name, gray.shape[0], gray.shape[1],
                                              getattr(ray_mod, "INK", "?")))
print("mask 像素 %d（暗/低彩）| colored %d" % (int(mask.sum()), int(colored.sum())))
print("pin_mono:", json.dumps(s1.get("pin_mono") or {}, ensure_ascii=False)[:220])

boxes = [c for c in s1["components"] if str(c.get("type")).lower() in ("box", "bigic")]
print("box/bigic 元件 %d 个" % len(boxes))
for c in boxes:
    cid = c["id"]
    if a.comp and cid != a.comp:
        continue
    box = tuple(float(v) for v in c["bbox_img"])
    print("\n== %s  type=%s source=%s bbox=%s  旧脚数=%d" % (
        cid, c.get("type"), c.get("source"), [round(v, 1) for v in box],
        len(s1["pins"].get(cid) or [])))
    frames = [ev._frame_edge(mask, box, e) for e in range(4)]
    for e, (coord, confirmed, coverage) in enumerate(frames):
        print("   edge %-6s coord=%-8s confirmed=%-5s coverage=%.3f"
              % (ev._EDGES[e], coord, confirmed, coverage))
    # 实测：沿框边所在的行/列找暗像素（含灰，不要求低彩），看看画出来的框线在哪
    for e, name in enumerate(ev._EDGES):
        vertical = e in (1, 3)
        if vertical:
            x = int(round(box[0] if name == "left" else box[2]))
            lo, hi = int(box[1]), int(box[3])
            seg = gray[lo:hi + 1, max(0, x - 4):x + 5]
            dark = (seg < 200).sum(axis=1)
            rows = [(lo + i, int(v)) for i, v in enumerate(dark) if v >= 6]
            print("   [%s] 列 x=%d 上暗行（横跨 ≥6px）%d 行%s"
                  % (name, x, len(rows), "" if len(rows) <= a.rows else " …"))
            for r, v in rows[:a.rows]:
                print("        y=%d  暗宽=%d" % (r, v))
        else:
            y = int(round(box[1] if name == "top" else box[3]))
            lo, hi = int(box[0]), int(box[2])
            seg = gray[max(0, y - 4):y + 5, lo:hi + 1]
            dark = (seg < 200).sum(axis=0)
            cols = [(lo + i, int(v)) for i, v in enumerate(dark) if v >= 6]
            print("   [%s] 行 y=%d 上暗列 %d 列" % (name, y, len(cols)))
            for cc, v in cols[:a.rows]:
                print("        x=%d  暗高=%d" % (cc, v))
