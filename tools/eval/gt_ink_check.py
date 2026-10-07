# -*- coding: utf-8 -*-
"""GT edge 压墨率卡在 ~48%：是坐标偏了，还是线宽不对？

对 (dx, dy) 与线宽做小范围扫描。若存在一个平移把压墨率抬到 85%+，
说明坐标约定差了一个常量；若怎么平移都停在 50% 上下，
说明 GT 的 2 点直边本来就不落在画出来的走线上（那是它的建模方式）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))

import cv2
import numpy as np
from schematic_parser.data import discover

ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
INK = 170
SAMPLE = None          # None = 全部；正数 = 只取前 N 张
OFFS = [(-3, -3), (-2, 0), (0, -2), (0, 0), (0, 2), (2, 0), (2, 2), (3, 3), (0, 4), (4, 0)]


def main():
    cases = discover(str(ROOT))
    if SAMPLE:
        cases = cases[:SAMPLE]
    # 只加载一次图像，反复用
    for thick in (1, 3, 5):
        res = {}
        for dx, dy in OFFS:
            tot = ink = 0
            for c in cases:
                img = cv2.imread(str(c.png_path))
                if img is None:
                    continue
                H, W = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                doc = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
                mask = np.zeros((H, W), np.uint8)
                for _n, net in (doc.get("nets") or {}).items():
                    for _e, pts in (net.get("edges") or {}).items():
                        if len(pts) < 2:
                            continue
                        a = np.asarray([[float(p["x"]), float(p["y"])] for p in pts], float)
                        a[:, 1] = H - a[:, 1]
                        a[:, 0] += dx
                        a[:, 1] += dy
                        cv2.polylines(mask, [np.round(a).astype(np.int32)], False, 255, thick)
                m = mask > 0
                tot += int(m.sum())
                ink += int((gray[m] < INK).sum())
            res[(dx, dy)] = ink / max(tot, 1) * 100
        best = max(res.items(), key=lambda kv: kv[1])
        print(f"线宽 {thick}: " + "  ".join(f"({dx:+d},{dy:+d}) {v:5.1f}%" for (dx, dy), v in
                                            sorted(res.items(), key=lambda kv: -kv[1])[:5]))
        print(f"          最佳 {best[0]} -> {best[1]:.1f}%")


main()
