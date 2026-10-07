"""宽视野看某个 GT 元件的上下文（含值/位号文字），用于判断符号到底是什么。

用法:
    python work\tools\diag\_rzoom.py --case 0050 --keys R1,R3,R11 --pad 90
    python work\tools\diag\_rzoom.py --case 0050 --type r --n 3 --pad 90
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="0050")
    ap.add_argument("--keys", default="")
    ap.add_argument("--type", default="r")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pad", type=int, default=90)
    ap.add_argument("--scale", type=float, default=0.0, help="0=自动缩到每格宽 420px")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    exact, digits = stage2_eval.build_index(DEFAULT_DATA_ROOT)
    c, how = stage2_eval.resolve(a.case, exact, digits)
    gt = json.load(open(c.json_path, encoding="utf-8"))
    bgr = cv2.imdecode(np.fromfile(c.png_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    print("%s  %s  %dx%d" % (c.case_id, c.png_path, W, H))

    comps = gt.get("components") or {}
    if a.keys:
        want = [k.strip() for k in a.keys.split(",")]
    else:
        want = [k for k, v in comps.items()
                if (v.get("type") or "").lower() == a.type][:a.n]

    tiles = []
    for k in want:
        v = comps.get(k)
        if not v:
            print("  没有元件 %s" % k); continue
        b = [float(x) for x in (v.get("bbox") or v.get("bbox_json"))]
        ib = (b[0], H - b[3], b[2], H - b[1])
        cx = (ib[0] + ib[2]) / 2.0; cy = (ib[1] + ib[3]) / 2.0
        x0 = max(0, int(cx - a.pad)); x1 = min(W, int(cx + a.pad))
        y0 = max(0, int(cy - a.pad)); y1 = min(H, int(cy + a.pad))
        im = bgr[y0:y1, x0:x1].copy()
        # 画 GT 框（图像坐标）
        cv2.rectangle(im, (int(ib[0] - x0), int(ib[1] - y0)),
                      (int(ib[2] - x0), int(ib[3] - y0)), (0, 0, 255), 2)
        s = a.scale or (420.0 / max(1, im.shape[1]))
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        head = np.full((30, im.shape[1], 3), 255, np.uint8)
        cv2.putText(head, "%s  type=%s  value=%s" % (k, v.get("type"), v.get("value")),
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, im]))
        print("  %-8s type=%-4s value=%-12s bbox=%.0fx%.0f center=(%.0f,%.0f)"
              % (k, v.get("type"), v.get("value"), b[2] - b[0], b[3] - b[1], cx, cy))

    if not tiles:
        raise SystemExit("没有可渲染的元件")
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw = max(t.shape[1] for t in tiles); th = max(t.shape[0] for t in tiles)
    out = np.full((rows * th, cols * tw, 3), 245, np.uint8)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        out[r * th:r * th + t.shape[0], cc * tw:cc * tw + t.shape[1]] = t
    o = a.out or (r"Y:\workspace\EDA\work\preview\_rzoom_%s.png" % a.case)
    cv2.imwrite(o, out)
    print("saved", o, out.shape)


if __name__ == "__main__":
    main()
