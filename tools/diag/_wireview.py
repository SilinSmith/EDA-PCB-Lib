# -*- coding: utf-8 -*-
"""把 stage1 的导线段按「连通分量」上色画在原图上，看分组到底碎在哪。

用法:
    python work\\tools\\diag\\_wireview.py 0001 [--stage1 work\\_submit_out220w\\runs\\stage1]
                                               [--out work\\preview\\_wireview.png]
                                               [--min-group 2] [--only-single]

颜色: 单元素分量 = 红；其它 = 按分量循环的一组颜色。
组件框 = 青，引脚点 = 黄。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from stage2.text import imread_unicode  # noqa: E402

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 100, 100), (100, 255, 255),
    (255, 0, 200), (150, 150, 255), (255, 150, 0), (0, 255, 200),
]
SINGLE = (255, 40, 40)


def _seg_xy(s):
    """段可能是 [[x1,y1],[x2,y2]] / [x1,y1,x2,y2] / {'p0':..}。统一成 x1,y1,x2,y2。"""
    if isinstance(s, dict):
        for a, b in (("p0", "p1"), ("a", "b"), ("start", "end")):
            if a in s and b in s:
                return _seg_xy([s[a], s[b]])
        if "x1" in s:
            return float(s["x1"]), float(s["y1"]), float(s["x2"]), float(s["y2"])
        raise SystemExit("不认识的段格式: %s" % list(s))
    if len(s) == 4 and not isinstance(s[0], (list, tuple)):
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])
    (ax, ay), (bx, by) = s[0], s[1]
    return float(ax), float(ay), float(bx), float(by)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--stage1", default=str(ROOT / "work" / "_submit_out220w" / "runs" / "stage1"))
    ap.add_argument("--out", default=str(ROOT / "work" / "preview" / "_wireview.png"))
    ap.add_argument("--only-single", action="store_true", help="只画单元素分量")
    ap.add_argument("--no-comps", action="store_true")
    args = ap.parse_args(argv)

    import json
    p = Path(args.stage1) / ("%s.stage1.json" % args.case)
    doc = json.loads(p.read_text(encoding="utf-8"))
    wd = doc["wire_detail"]
    segs = [_seg_xy(s) for s in wd["segments"]]
    groups = list(wd["groups"])
    assert len(segs) == len(groups), (len(segs), len(groups))

    # 每个分量有多少段
    cnt = Counter(groups)
    singles = {g for g, n in cnt.items() if n == 1}
    print("case %s：段 %d，分量 %d，其中单元素 %d（%.1f%%）"
          % (args.case, len(segs), len(cnt), len(singles),
             100.0 * len(singles) / max(1, len(cnt))))
    hist = Counter(cnt.values())
    print("分量规模直方(段数:分量数): " + "  ".join(
        "%d:%d" % (k, hist[k]) for k in sorted(hist)))

    img_path = doc.get("image")
    if isinstance(img_path, dict):
        img_path = img_path.get("path")
    if img_path and not Path(img_path).is_absolute():
        img_path = str(ROOT / img_path)
    bgr = imread_unicode(img_path)
    H, W = bgr.shape[:2]
    im = Image.fromarray(bgr[:, :, ::-1]).convert("RGB")
    dr = ImageDraw.Draw(im)

    if not args.no_comps:
        for c in doc.get("components", []) or []:
            bb = c.get("bbox_img") or c.get("bbox")
            if not bb:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bb]
            except Exception:
                continue
            dr.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=1)

    order = sorted(cnt, key=lambda g: -cnt[g])
    cmap = {}
    for i, g in enumerate(order):
        cmap[g] = SINGLE if g in singles else PALETTE[i % len(PALETTE)]

    for i, (x1, y1, x2, y2) in enumerate(segs):
        g = groups[i]
        if args.only_single and g not in singles:
            continue
        dr.line([x1, y1, x2, y2], fill=cmap[g], width=2)

    for pt in (wd.get("endpoints") or []):
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            continue
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 255, 0))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(outp))
    print("写出 %s  (%dx%d)" % (outp, W, H))
    return 0


if __name__ == "__main__":
    sys.exit(main())
