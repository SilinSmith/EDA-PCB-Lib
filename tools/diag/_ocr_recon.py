"""侦察：文字定位器 API + 官方 GT 的文字真值锚点。

跑法（CWD 必须是 Y:\\workspace\\EDA\\eda-pcb）：
    $env:PYTHONIOENCODING='utf-8'; python Y:\\workspace\\EDA\\work\\_ocr_recon.py
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))

import cv2
import numpy as np

from schematic_parser import data as gtdata

sys.stdout.reconfigure(encoding="utf-8")

# ---- 1) textbox API ----
import inspect  # noqa: E402

from cvparts import textbox as tb

print("=== cvparts.textbox ===")
print("  detect_boxes", inspect.signature(tb.detect_boxes))
print("  text_mask   ", inspect.signature(tb.text_mask))
print("  constants   ", {k: getattr(tb, k) for k in ("INK", "LINE_LEN", "MIN_H", "MAX_H",
                                                    "MIN_W", "MAX_W", "MIN_AREA",
                                                    "GAP_FRAC", "V_OVERLAP", "H_RATIO")})

# ---- 2) 官方 GT 的文字真值 ----
cases = gtdata.discover(gtdata.DEFAULT_DATA_ROOT)
print(f"\n=== 官方 case {len(cases)} 张 ===")

roles = Counter()
for c in cases:
    doc = gtdata.load_json(c.json_path)
    for cid, comp in (doc.get("components") or {}).items():
        roles["designator"] += 1
        if comp.get("value"):
            roles["value"] += 1
    for cid, pins in (doc.get("pins") or {}).items():
        for pk, pv in pins.items():
            if pv.get("pinname"):
                roles["pin_name"] += 1
            roles["pin_number"] += 1
    for nid, net in (doc.get("nets") or {}).items():
        pass
print("GT 文字真值计数：", dict(roles))

# ---- 3) 一张具体 case 看锚点几何 ----
for want in ("0059", "0001", "0025"):
    c = gtdata.find_case(want, gtdata.DEFAULT_DATA_ROOT)
    if c is None:
        continue
    doc = gtdata.load_json(c.json_path)
    img = cv2.imread(c.png_path)
    H, W = img.shape[:2]
    boxes = tb.detect_boxes(img)
    print(f"\n--- {c.case_id} ({c.split}) {W}x{H} 检测文字框 {len(boxes)} ---")
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}
    show = 0
    for cid, comp in list(comps.items()):
        bx = comp["bbox"]
        cb = (bx[0], H - bx[3], bx[2], H - bx[1])  # JSON y-up -> image
        # 找最近的文字框
        best = None
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            dx = max(cb[0] - cx, 0, cx - cb[2])
            dy = max(cb[1] - cy, 0, cy - cb[3])
            d = (dx * dx + dy * dy) ** 0.5
            if best is None or d < best[0]:
                best = (d, b)
        if show < 6:
            print(f"  {cid:>10} type={comp.get('type'):<14} Name={comp.get('Name')!r:<8} "
                  f"value={comp.get('value')!r:<8} 最近框距离={best[0]:.1f}px "
                  f"框=({best[1][0]:.0f},{best[1][1]:.0f},{best[1][2]:.0f},{best[1][3]:.0f})")
            show += 1
    # 引脚名锚点：pin point 在图像坐标
    shown = 0
    for cid, pp in list(pins.items()):
        for pk, pv in list(pp.items()):
            pt = pv["point"]
            px, py = pt["x"], H - pt["y"]
            best = None
            for b in boxes:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, b)
            if shown < 5:
                print(f"    pin {cid}.{pk} name={pv.get('pinname')!r:<14} "
                      f"pt=({px:.0f},{py:.0f}) 最近框中心距={best[0]:.1f}px")
                shown += 1
