# -*- coding: utf-8 -*-
"""引脚号码识别：真值取自 GT 的 `pin_<n>`（按 y 配对，权威），扫「裁剪几何」看 exact。

背景：case 1 的号码框接触表（`work\preview\_numcrop_1.png`）显示**框都很干净**，
但 CRNN 读错一批（17->1、39->38、114->14、69->89、90->80、9->8、68->80 …）。
`crop_gray` 把裁剪缩到高 32、宽按纵横比，CTC 时步 T = new_w // 8；号码框很扁，
当前超参下 new_w 只有 ~31-39 ⇒ **T 只有 3-5 个时步**。本脚本直接扫这个。

用法：
    python work\\tools\\diag\\_numsweep.py [--case 1]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT))

from schematic_parser import data as D  # noqa: E402
from stage2.text import TextEngine, pad_box  # noqa: E402
from textrec import config as C  # noqa: E402

LEFT_BAND = (860.0, 960.0)
RIGHT_BAND = (1310.0, 1420.0)


def build_truth(case_id, stage1_dir, comp_filter=None, tol=16.0, min_w=8.0, min_h=8.0):
    cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)}
    c = cases[case_id]
    img = cv2.imread(str(c.png_path))
    H = img.shape[0]
    gt = D.load_json(c.json_path)
    pts = []
    for comp, group in (gt.get("pins") or {}).items():
        if not isinstance(group, dict):
            continue
        if comp_filter and comp != comp_filter:
            continue
        for pk, info in group.items():
            p = (info or {}).get("point") or {}
            if "x" not in p or "y" not in p:
                continue
            pts.append((comp, D.pin_key_to_number(pk), float(p["x"]), H - float(p["y"])))

    src = Path(stage1_dir)
    if src.is_dir():
        src = src / ("%s.stage1.json" % case_id)
    j = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for t in j.get("texts") or []:
        x0, y0, x1, y1 = t["bbox_img"]
        if x1 - x0 < min_w or y1 - y0 < min_h:
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        band = None
        for name, (a, b) in (("L", LEFT_BAND), ("R", RIGHT_BAND)):
            if a <= cx <= b:
                band = name
        if band is None:
            continue
        cand = [p for p in pts
                if (p[2] < 1150.0) == (band == "L") and abs(p[3] - cy) <= tol]
        if not cand:
            continue
        cand.sort(key=lambda p: abs(p[3] - cy))
        comp, num = cand[0][0], cand[0][1]
        rows.append((t["id"], num, tuple(float(v) for v in t["bbox_img"]), comp))
    rows.sort(key=lambda r: (r[3], r[0]))
    return img, c.png_path, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--comp", default="U1", help="只把这个元件的引脚当真值（'' = 全部）")
    ap.add_argument("--stage1", default=str(ROOT / "work" / "_submit_out220r32" / "runs" / "stage1"))
    a = ap.parse_args()

    img, png, rows = build_truth(a.case, a.stage1, comp_filter=(a.comp or None))
    print("case=%s  号码框 %d 个（真值取自 GT pin_<n>）" % (a.case, len(rows)))
    gray = Image.open(png).convert("L")
    eng = TextEngine()
    rec = eng.recognizer()
    if rec is None:
        print("识别器没加载：%s" % eng.note)
        return

    def run(pad_min, pad_frac, pad_frac_y, extra_x, pad_x_frac, pad_y_frac):
        old = (C.PAD_X_FRAC, C.PAD_Y_FRAC)
        C.PAD_X_FRAC, C.PAD_Y_FRAC = pad_x_frac, pad_y_frac
        try:
            boxes = []
            for tid, gt, bb, comp in rows:
                cb = pad_box(bb, pad_min, pad_frac, pad_min_y=pad_min, pad_frac_y=pad_frac_y)
                if extra_x:
                    cb = (cb[0] - extra_x, cb[1], cb[2] + extra_x, cb[3])
                boxes.append(cb)
            got = rec.read_many(gray, [(b, "pin_number") for b in boxes])
        finally:
            C.PAD_X_FRAC, C.PAD_Y_FRAC = old
        ok, bad, ts = 0, [], []
        for (tid, gt, bb, comp), (txt, conf) in zip(rows, got):
            h = bb[3] - bb[1]
            cbw = (bb[2] - bb[0]) + 2 * max(pad_min, pad_frac * h)
            cbh = h + 2 * max(pad_min, pad_frac_y * h)
            cw = cbw + 2 * C.PAD_X_FRAC * cbh
            ch = cbh * (1 + 2 * C.PAD_Y_FRAC)
            nw = max(C.MIN_W, min(C.MAX_W, int(round(C.IMG_H * cw / max(1.0, ch)))))
            ts.append(nw // C.WIDTH_REDUCTION)
            if txt == gt:
                ok += 1
            else:
                bad.append("%s %s->%s" % (comp, gt, txt))
        return ok, bad, ts

    print("\n== 基线（TextEngine 默认 pad 0.30 + crop_gray 默认）==")
    ok, bad, ts = run(2.0, 0.30, 0.30, 0, 0.12, 0.25)
    print("exact %d/%d   T 分布 %s" % (ok, len(rows), sorted(set(ts))))
    print("  错：", " | ".join(bad))

    cfgs = []
    for pf in (0.60, 1.00):
        for pfy in (0.60, 1.00):
            for pxf in (0.12,):
                cfgs.append((2.0, pf, pfy, 0, pxf, 0.25))
    out = []
    for g in cfgs:
        ok, bad, ts = run(*g)
        out.append((ok, g, ts, bad))
    out.sort(key=lambda r: -r[0])
    print("\n== 扫描（按 exact 降序）==")
    for ok, g, ts, bad in out:
        print("exact %2d/%d  pad_min=%.1f pad_frac=%.2f pad_frac_y=%.2f extra_x=%2d PAD_X=%.2f PAD_Y=%.2f  T%s"
              % (ok, len(rows), g[0], g[1], g[2], g[3], g[4], g[5], sorted(set(ts))))
        if bad:
            print("      错：", " | ".join(bad))


if __name__ == "__main__":
    main()
