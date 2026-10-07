"""决定性检查：CVJsonStd 的 texts[].location 到底对不对得上图像像素。

跑法（CWD = Y:\\workspace\\EDA\\eda-pcb）：
    $env:PYTHONIOENCODING='utf-8'
    python Y:\\workspace\\EDA\\work\\_cvjson_align.py 0059 0003 0016
产物：Y:\\workspace\\EDA\\work\\preview\\cvjson_align_<id>.png（原图 + 红框 = CVJson location）
"""
import glob
import json
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def imread_u(p):
    """cv2.imread 读不了非 ASCII 路径（Dataset300 目录名带中文），必须走 fromfile。"""
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset300我不知道干嘛用的先放置"
OUT = r"Y:\workspace\EDA\work\preview"


def show(cid):
    d = os.path.join(ROOT, cid)
    js = glob.glob(os.path.join(d, "*_CVJsonStd.json"))
    pngs = [p for p in glob.glob(os.path.join(d, "*.png"))]
    if not js or not pngs:
        print(f"{cid}: missing")
        return
    j = json.load(open(js[0], encoding="utf-8"))
    img = imread_u(pngs[0])
    H, W = img.shape[:2]
    cv = j.get("canvas") or {}
    print(f"{cid}: png {W}x{H}  canvas {cv.get('width')}x{cv.get('height')}  "
          f"texts {len(j.get('texts') or {})}")
    vis = img.copy()
    n = 0
    for tid, t in (j.get("texts") or {}).items():
        loc = t.get("location") or {}
        try:
            x1, y1, x2, y2 = (float(loc["x1"]), float(loc["y1"]),
                              float(loc["x2"]), float(loc["y2"]))
        except Exception:
            continue
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        n += 1
    p = os.path.join(OUT, f"cvjson_align_{cid}.png")
    cv2.imwrite(p, vis)
    print(f"   画了 {n} 个框 -> {p}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        show(a)
