# -*- coding: utf-8 -*-
"""「红点引脚标记 vs 真名字框」能不能用**墨迹统计**分开？（零 VLM 调用）

背景（`eda-pcb/docs/RESULTS.md` §5.91/§5.92）：10GT case 5 的 `U3` pin1 名字框那一行有两个
候选 —— 真名 `ADJ(GND)`（67x13）与红点引脚标记（8x8，CRNN 读 `Q`）。名轮把红点绑给
pin1、把真名挤到 pin2。几何（宽高比）分不开：`L`（合法名，7x10）与红点（8x8）形状相同。
**像素**分得开吗？红点是**有色实心**墨迹，文字是深色细笔画。

实测（10GT case 5）：
  * 真名字框 `ADJ(GND)`/`VOUT(TAB)`/`VIN` → `mxmn` = **0.000**、`sat` = 0.000
  * 红点引脚标记（8x8 / 8x10）        → `mxmn` = **0.641 / 0.484 / 0.500**
  * 左侧短名 `L`（7x10）              → `mxmn` = 0.000（几何分不开、颜色分得开）
⇒ 这条统计就是 `stage2/text.py::colored_frac` / `STAGE2_PINNAME_CFRAC` 的依据。
（落地后实测：加上「器件名框放行」修复后，该颜色闸门在 10 张上**中性**——
两臂逐位相同，故 默认仍为 0，见 §5.92 的四档闸门对比表。）

用法：python -X utf8 eda-pcb\\tools\\diag\\_inkstat.py [--case 5] [--run work\\_fast_out_cd10on]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover    # noqa: E402


def stat(img, box, flip_h=0):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if flip_h:
        y0, y1 = flip_h - y1, flip_h - y0
    x0, x1 = max(0, x0), min(img.shape[1], x1)
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark = float((g < 128).mean())
    verydark = float((g < 96).mean())
    sat = float((hsv[:, :, 1] > 60).mean())
    sub = crop.astype(np.int16)
    mxmn = sub.max(2) - sub.min(2)
    gmean = sub.mean(2)
    mxmn_frac = float(((mxmn >= 60) & (gmean < 160)).mean())
    return dict(w=x1 - x0, h=y1 - y0, dark=dark, verydark=verydark, sat=sat,
                mxmn=mxmn_frac,
                gmean=float(g.mean()), bgr=[int(v) for v in crop.reshape(-1, 3).mean(axis=0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="5")
    ap.add_argument("--run", default=r"work\_fast_out_cd10on")
    a = ap.parse_args()
    cid = a.case
    img_p = None
    for c in discover(DEFAULT_DATA_ROOT):
        if c.case_id == cid:
            img_p = Path(c.json_path).parent / ("%s.png" % cid)
            break
    img = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
    H = img.shape[0]
    asg = json.load(open(ROOT / a.run / "runs" / "fusion" / "_assigned" / ("%s.json" % cid),
                        encoding="utf-8"))
    print("图 %s  %dx%d   H=%d" % (img_p.name, img.shape[1], img.shape[0], H))
    print("%-28s %-22s %5s %5s %6s %8s %6s %6s %-16s" %
          ("comp/slot", "read", "w", "h", "dark", "verydark", "sat", "mxmn", "meanBGR"))
    for key, cv in (asg.get("comps") or {}).items():
        for si, sv in (cv.get("pins") or {}).items():
            nm = (sv or {}).get("name")
            if not nm or len(str(nm.get("text") or "")) > 2:
                continue
            box = nm.get("box") or []
            if len(box) != 4:
                continue
            tag = "%s/%s" % (((cv.get("designator") or {}).get("canon") or key), si)
            for label, flip in (("as-is", 0), ("flipY", H)):
                st = stat(img, box, flip)
                if st is None:
                    continue
                print("%-28s %-22s %5d %5d %6.3f %8.3f %6.3f %6.3f %-16s  [%s]" %
                      (tag, repr(str(nm.get("text")))[:20], st["w"], st["h"], st["dark"],
                       st["verydark"], st["sat"], st["mxmn"], st["bgr"], label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
