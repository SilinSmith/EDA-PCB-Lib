# -*- coding: utf-8 -*-
"""把 stage1 的「引脚号码框」按 y 位置对到 GT 的 `pin_<n>` 上，得到**权威真值**，
不再靠肉眼读接触表。

GT 的 `pins[comp]['pin_7'] = {'pinname':'VDD','point':{'x':947,'y':1088}}`，
`point` 是 **y-UP** ⇒ 图像坐标 y_png = H - y_gt。左列引脚 x≈947、右列 x≈1334，
号码框都在同一 y 带上，所以只要按 y 配对、按 x 分左右。

用法：
    python work\\tools\\diag\\_numtruth.py 1 [--texts <stage1目录>] [--tol 12]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT))

from schematic_parser import data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--texts", default=str(ROOT / "work" / "_submit_out220r32" / "runs" / "stage1"))
    ap.add_argument("--tol", type=float, default=12.0)
    ap.add_argument("--min-w", type=float, default=8.0)
    ap.add_argument("--min-h", type=float, default=8.0)
    a = ap.parse_args()

    cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)}
    c = cases[a.case]
    img = cv2.imread(str(c.png_path))
    H = img.shape[0]
    gt = D.load_json(c.json_path)

    # 收集所有 GT 引脚：(comp, number, name, x_img, y_img)
    pts = []
    for comp, group in (gt.get("pins") or {}).items():
        if not isinstance(group, dict):
            continue
        for pk, info in group.items():
            p = (info or {}).get("point") or {}
            if "x" not in p or "y" not in p:
                continue
            pts.append((comp, D.pin_key_to_number(pk), (info or {}).get("pinname"),
                        float(p["x"]), H - float(p["y"])))

    src = Path(a.texts)
    if src.is_dir():
        src = src / ("%s.stage1.json" % a.case)
    j = json.loads(src.read_text(encoding="utf-8"))

    boxes = []
    for t in j.get("texts") or []:
        x0, y0, x1, y1 = t["bbox_img"]
        if x1 - x0 < a.min_w or y1 - y0 < a.min_h:
            continue
        boxes.append((t["id"], (x0 + x1) / 2.0, (y0 + y1) / 2.0, tuple(t["bbox_img"])))
    boxes.sort(key=lambda r: r[2])

    print("case=%s H=%d  检出框 %d   GT 引脚 %d" % (a.case, H, len(boxes), len(pts)))
    used = set()
    for tid, cx, cy, bb in boxes:
        side = 0 if cx < 1150 else 1
        cand = [p for p in pts
                if (0 if p[3] < 1150 else 1) == side and abs(p[4] - cy) <= a.tol]
        cand.sort(key=lambda p: abs(p[4] - cy))
        if not cand:
            print("  %-6s y=%6.1f x=%6.1f  -> (无 GT 引脚)" % (tid, cy, cx))
            continue
        comp, num, name, px, py = cand[0]
        dup = "*" if (comp, num) in used else " "
        used.add((comp, num))
        print("  %-6s y=%6.1f x=%6.1f  -> %s%s.pin_%s (%s)  dy=%.1f"
              % (tid, cy, cx, dup, comp, num, name, abs(py - cy)))
    miss = [p for p in pts if (p[0], p[1]) not in used]
    print("未被任何检出框覆盖的 GT 引脚（%d）：" % len(miss))
    for comp, num, name, px, py in sorted(miss, key=lambda p: (p[0], p[4])):
        if comp == "U1":
            print("   %s.pin_%s (%s) x=%.0f y_png=%.0f" % (comp, num, name, px, py))


if __name__ == "__main__":
    main()
