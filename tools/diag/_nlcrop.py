# -*- coding: utf-8 -*-
"""_nlcrop.py —— 把「单端错」样本裁成小图，画出 GT 段（绿）/我方段（红）/对端（黄），
用于**肉眼**判断：错端到底是该切在结点/引脚，还是端点沿轴长短不对。

坐标：doc 坐标（y 向上）-> 图像 `im[H - y, x]`。放大 n 倍。
用法：python -X utf8 tools/diag/_nlcrop.py --cases 10 --idx 0,1,2 --out Y:/workspace/EDA/work/_crop
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, DEFAULT_DATA_ROOT   # noqa: E402

TOL = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="Y:/workspace/EDA/work/_nlpair2.json")
    ap.add_argument("--cases", default="10")
    ap.add_argument("--idx", default="0,1,2")
    ap.add_argument("--out", default="Y:/workspace/EDA/work/_crop")
    ap.add_argument("--half", type=int, default=90)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="该 case 全部 one_end 样本（否则按 --idx）")
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    os.makedirs(a.out, exist_ok=True)
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    rows = json.load(io.open(a.json, encoding="utf-8"))["rows"]
    want = set(x.strip() for x in a.cases.split(",") if x.strip())
    idxx = set(int(x) for x in a.idx.split(",") if x.strip())
    for r in rows:
        cid = r["case"]
        if cid not in want:
            continue
        case = index.get(cid)
        im = Image.open(case.png_path).convert("RGB")
        W, H = im.size
        n = 0
        for si, s in enumerate(r["samples"]):
            if s["kind"] != "one_end":
                continue
            if not a.all and si not in idxx:
                continue
            bad = "a" if s["a"]["d"] > TOL else "b"
            GP = s["pred"][0] if bad == "a" else s["pred"][1]
            GB = s["gt"][0] if bad == "a" else s["gt"][1]
            GO = s["pred"][1] if bad == "a" else s["pred"][0]
            other_gt = s["gt"][1] if bad == "a" else s["gt"][0]
            cx, cy = GB
            x0, y0 = int(cx - a.half), int(cy - a.half)
            x1, y1 = int(cx + a.half), int(cy + a.half)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            crop = im.crop((x0, y0, x1, y1))
            z = a.zoom
            crop = crop.resize((crop.width * z, crop.height * z), Image.NEAREST)
            dr = ImageDraw.Draw(crop)

            def P(p):
                return ((p[0] - x0) * z, (H - p[1] - y0) * z)

            dr.line([P(GP), P(GO)], fill=(255, 0, 0), width=1)        # 我方段 红
            dr.line([P(GB), P(other_gt)], fill=(0, 255, 0), width=1)  # GT 段 绿
            dr.ellipse([P(GB)[0] - 3, P(GB)[1] - 3, P(GB)[0] + 3, P(GB)[1] + 3],
                       outline=(0, 255, 255), width=2)                # GT 错端 青圈
            dr.ellipse([P(GP)[0] - 3, P(GP)[1] - 3, P(GP)[0] + 3, P(GP)[1] + 3],
                       outline=(255, 255, 0), width=2)                # 我方错端 黄圈
            fn = os.path.join(a.out, "c%s_s%02d_%s_d%s.png" % (cid, si, s[bad]["cls"], s[bad]["d"]))
            crop.save(fn)
            print("%s  cls=%s d=%s t_gt=%s Lp=%s gt=%s->%s pred=%s->%s"
                  % (fn, s[bad]["cls"], s[bad]["d"], s[bad]["axial"], s[bad]["Lp"],
                     GB, other_gt, GP, GO))
            n += 1
        print("case %s: %d 张" % (cid, n))


if __name__ == "__main__":
    main()
