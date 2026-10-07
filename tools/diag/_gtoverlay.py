"""把某个 case 的 GT 框全部画回原图（可缩放、可圈定区域），用来判断
「JSON 与 PNG 是否对齐」以及「框锚点落在符号的哪个位置」。

用法:
    python work\tools\diag\_gtoverlay.py --case 0050 --cx 1616 --cy 2540 --w 700 --h 500
    python work\tools\diag\_gtoverlay.py --case 0050 --full --maxw 1600
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.join(r"Y:\workspace\EDA\eda-pcb", "schematic_parser"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from schematic_parser.data import discover, DEFAULT_DATA_ROOT        # noqa: E402
import stage2_eval                                                   # noqa: E402

COLORS = {
    "r": (0, 0, 255), "c": (0, 160, 0), "l": (255, 0, 0), "gnd": (0, 140, 255),
    "v": (200, 0, 200), "box": (255, 0, 0), "net_input": (0, 200, 200),
    "testpoint": (128, 0, 255), "d": (0, 100, 0), "led": (0, 60, 200),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="0050")
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)          # 图像坐标
    ap.add_argument("--w", type=int, default=700)
    ap.add_argument("--h", type=int, default=500)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--maxw", type=int, default=1600)
    ap.add_argument("--types", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    exact, digits = stage2_eval.build_index(DEFAULT_DATA_ROOT)
    c, how = stage2_eval.resolve(a.case, exact, digits)
    gt = json.load(open(c.json_path, encoding="utf-8"))
    bgr = cv2.imdecode(np.fromfile(c.png_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    keep = set(x.strip() for x in a.types.split(",") if x.strip())

    if a.full:
        s = min(1.0, a.maxw / float(W))
        x0, y0 = 0.0, 0.0
        win = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    else:
        cx = a.cx if a.cx is not None else W / 2.0
        cy = a.cy if a.cy is not None else H / 2.0
        x0 = max(0.0, cx - a.w / 2.0); y0 = max(0.0, cy - a.h / 2.0)
        x1 = min(float(W), x0 + a.w); y1 = min(float(H), y0 + a.h)
        win = bgr[int(y0):int(y1), int(x0):int(x1)].copy()
        s = 1.0

    n = 0
    for k, v in (gt.get("components") or {}).items():
        t = (v.get("type") or "?").lower()
        if keep and t not in keep:
            continue
        b = v.get("bbox") or v.get("bbox_json")
        if not b:
            continue
        b = [float(x) for x in b]
        ix0, iy0 = (b[0] - x0) * s, (H - b[3] - y0) * s
        ix1, iy1 = (b[2] - x0) * s, (H - b[1] - y0) * s
        if ix1 < 0 or iy1 < 0 or ix0 > win.shape[1] or iy0 > win.shape[0]:
            continue
        col = COLORS.get(t, (80, 80, 80))
        cv2.rectangle(win, (int(ix0), int(iy0)), (int(ix1), int(iy1)), col, 1)
        if s >= 0.8:
            cv2.putText(win, k, (int(ix0), max(10, int(iy0) - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        n += 1
    print("%s  %dx%d  缩放 %.3f  窗口 (%d,%d)-(%d,%d)  画了 %d 个框"
          % (c.case_id, W, H, s, int(x0), int(y0), int(x0) + win.shape[1], int(y0) + win.shape[0], n))
    for t in ("r", "c", "l", "gnd", "v", "box"):
        print("   %-10s %s" % (t, COLORS.get(t)))
    o = a.out or (r"Y:\workspace\EDA\work\preview\_gtover_%s.png" % c.case_id)
    cv2.imwrite(o, win)
    print("saved", o, win.shape)


if __name__ == "__main__":
    main()
