# -*- coding: utf-8 -*-
"""任意「图像坐标」小框的放大对照图（诊断用，零模型）。

与已有探针的分工：
  * `tools/diag/_boxcrop.py`  —— 按 **GT 元件键**裁框（读数据集 JSON、y 向上换算）；
  * `tools/diag/_numcrop.py`  —— 把 **stage1 检出的号码框**堆成接触表；
  * **本文件** —— 给**任意 4 个图像坐标**，逐个渲染「框 + 邻域」的最近邻放大，
    并用红框标出裁片边界；用于回答「这块框里到底是什么」。

用法（在 `eda-pcb` 下运行）：

    python -X utf8 tools\\diag\\_boxzoom.py --case 8 --box "967,701,994,711" ^
        --box "967,470,996,485" --out ..\\work\\_c8box.png

坐标系：`--box` 用 **图像左上原点**，即 `stage1` 的 `texts[*].bbox_img` /
`stage2` 的 `_assigned` 那套；`final_results/*_Pred.json` 与 GT 文档是**左下原点**，
换算 `y_img = H - y_json`（H = 图片高）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = ROOT / "work" / "_fast_in"


def parse_boxes(vals):
    out = []
    for v in vals:
        parts = [p for p in v.replace(" ", "").split(",") if p]
        if len(parts) != 4:
            raise SystemExit("--box 需要 4 个数字：x0,y0,x1,y1（收到 %r）" % v)
        x0, y0, x1, y1 = (float(p) for p in parts)
        if x1 <= x0 or y1 <= y0:
            raise SystemExit("--box 的 x1/y1 必须大于 x0/y0：%r" % v)
        out.append((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
    return out


def render(img, boxes, zoom=6, pad=8):
    tiles = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        h, w = img.shape[:2]
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
        sub = img[cy0:cy1, cx0:cx1]
        big = cv2.resize(sub, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        rx0, ry0 = (x0 - cx0) * zoom, (y0 - cy0) * zoom
        rx1, ry1 = (x1 - cx0) * zoom, (y1 - cy0) * zoom
        cv2.rectangle(big, (rx0, ry0), (max(rx0 + 1, rx1 - 1), max(ry0 + 1, ry1 - 1)),
                      (0, 0, 255), 1)
        bar = np.full((26, big.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, "#%d %d,%d,%d,%d (%dx%d)"
                    % (i + 1, x0, y0, x1, y1, x1 - x0, y1 - y0),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, big]))
    hh = max(t.shape[0] for t in tiles)
    out = np.full((hh, sum(t.shape[1] for t in tiles) + 20 * (len(tiles) - 1), 3),
                  255, np.uint8)
    x = 0
    for t in tiles:
        out[0:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 20
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="任意图像坐标小框的放大对照图")
    ap.add_argument("--case", required=True, help="case id（图片名，如 8 / 0092）")
    ap.add_argument("--box", action="append", required=True,
                    help="x0,y0,x1,y1（图像左上原点；可重复）")
    ap.add_argument("--zoom", type=int, default=6, help="放大倍数（默认 6）")
    ap.add_argument("--pad", type=int, default=8, help="四周留白像素（默认 8）")
    ap.add_argument("--img", default=None, help="图片目录（默认 work/_fast_in）")
    ap.add_argument("--out", default=None, help="输出 png（默认 work/_boxzoom_<case>.png）")
    a = ap.parse_args(argv)

    boxes = parse_boxes(a.box)
    img_dir = Path(a.img) if a.img else IMG_DIR
    if not img_dir.is_absolute():
        img_dir = ROOT / img_dir
    png = img_dir / ("%s.png" % a.case)
    if not png.exists():
        raise SystemExit("找不到图片：%s" % png)
    img = cv2.imread(str(png))
    if img is None:
        raise SystemExit("读不出图片：%s" % png)
    out = render(img, boxes, zoom=a.zoom, pad=a.pad)
    dst = Path(a.out) if a.out else (ROOT / "work" / ("_boxzoom_%s.png" % a.case))
    if not dst.is_absolute():
        dst = ROOT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    print("图片 %s %dx%d ⇒ %s （%d 个框，zoom=%d pad=%d）"
          % (png.name, img.shape[1], img.shape[0], dst, len(boxes), a.zoom, a.pad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
