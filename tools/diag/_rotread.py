# -*- coding: utf-8 -*-
"""**旋转文字体检**：竖直（上下边）引脚的引脚名/号在图上常是**旋转 90°** 的，
而 CRNN（`textrec/infer.py`）只吃水平裁片 —— 本脚本量「同一批检出框在 4 个朝向下的读数」，
回答「旋转这条路能不能救回这些脚」。

用法：
    python -X utf8 tools\\diag\\_rotread.py --case 6 --comp U1
    python -X utf8 tools\\diag\\_rotread.py --case 6 --comp U3 --roles pin_name,pin_number
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                             # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402
from stage2.text import DEFAULT_CKPT, detect_texts                     # noqa: E402
from textrec.infer import Recognizer                                   # noqa: E402

ROT = {"0": None, "90": cv2.ROTATE_90_CLOCKWISE,
       "270": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="6")
    ap.add_argument("--comp", default="")
    ap.add_argument("--roles", default="pin_name,pin_number")
    ap.add_argument("--pad", type=float, default=34.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="")
    ap.add_argument("--region", action="store_true",
                    help="整块区域按 4 个朝向各跑一遍检测+识别（看哪个朝向读得出）")
    a = ap.parse_args()

    case = None
    for c in discover(DEFAULT_DATA_ROOT):
        if c.case_id == a.case and c.split in ("10GT", "10_GTcase", "200_train_cases"):
            case = c
            break
    if case is None:
        print("!! 找不到 case", a.case)
        return 1
    bgr = imread_unicode(case.png_path)
    H, W = bgr.shape[:2]
    gt = load_json(Path(case.json_path))
    boxes = detect_texts(bgr)
    print("case %s  %dx%d  检出文字框 %d" % (a.case, W, H, len(boxes)))
    rec = Recognizer.load(DEFAULT_CKPT, device=a.device)
    print("识别器:", rec.debug())

    comps = [a.comp] if a.comp else sorted(gt.get("pins") or {})
    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    for ck in comps:
        bb = (gt.get("components") or {}).get(ck, {}).get("bbox")
        if not bb:
            continue
        x0, y0, x1, y1 = [float(t) for t in bb]
        py0, py1 = H - y1, H - y0                     # JSON 帧 -> PNG 帧
        gtp = gt.get("pins", {}).get(ck) or {}
        print("\n== %s bbox(png)=[%.0f,%.0f,%.0f,%.0f] GT 引脚 %d 个: %s"
              % (ck, x0, py0, x1, py1, len(gtp),
                 ", ".join("%s=%s" % (k, (v or {}).get("pinname")) for k, v in gtp.items())))
        sel = []
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            if x0 - a.pad <= cx <= x1 + a.pad and py0 - a.pad <= cy <= py1 + a.pad:
                sel.append(b)
        print("   区域内检出框 %d 个" % len(sel))
        if a.region:
            rx0, ry0 = max(0, int(x0 - a.pad)), max(0, int(py0 - a.pad))
            rx1, ry1 = min(W, int(x1 + a.pad)), min(H, int(py1 + a.pad))
            reg = bgr[ry0:ry1, rx0:rx1]
            for name, flag in ROT.items():
                g = reg if flag is None else cv2.rotate(reg, flag)
                rb = detect_texts(g)
                outs = []
                for b in rb:
                    x_a, y_a, x_b, y_b = [max(0, int(round(v))) for v in b]
                    crop = g[y_a:y_b, x_a:x_b]
                    if crop.size == 0:
                        continue
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    hh, ww = gray.shape[:2]
                    t, cf = rec.read_many(gray, [((0.0, 0.0, float(ww), float(hh)),
                                                  "pin_name")])[0]
                    if t:
                        outs.append("%r(%.2f)" % (t, cf))
                print("   [region %s°] 检测 %d 框 -> %s"
                      % (name, len(rb), " ".join(outs[:40])))
        for b in sel:
            x_a, y_a, x_b, y_b = [int(round(v)) for v in b]
            x_a, y_a = max(0, x_a), max(0, y_a)
            x_b, y_b = min(W, x_b), min(H, y_b)
            crop = bgr[y_a:y_b, x_a:x_b]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            row = []
            for name, flag in ROT.items():
                g = gray if flag is None else cv2.rotate(gray, flag)
                hh, ww = g.shape[:2]
                got = []
                for r in roles:
                    t, cf = rec.read_many(g, [((0.0, 0.0, float(ww), float(hh)), r)])[0]
                    got.append("%s:%r(%.2f)" % (r.split("_")[-1], t, cf))
                row.append("%s°[%s]" % (name, " ".join(got)))
            print("   框[%.0f,%.0f,%.0f,%.0f] w=%d h=%d %s"
                  % (b[0], b[1], b[2], b[3], x_b - x_a, y_b - y_a, " | ".join(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
